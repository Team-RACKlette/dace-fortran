# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scratch arrays each kernel iteration only uses for itself, held per iteration instead of in device memory.

A Riemann solver may write a few velocities of the face it works on into a module array and read one of them back
for the same face (``v(j, k, l, d(i)) = ...`` for each ``i``, then ``... = v(j, k, l, d(1))``), with nothing else
reading the array. As the substitute's own transient it still takes a buffer over the whole grid, and every face
stores its values to device memory and loads one back.

``localise_iteration_scratch`` takes every transient array of the SDFG that only nested SDFGs inside top-level maps
(kernel bodies) access, through the maps around them, where in every such body

* every element read was written earlier in the same execution of the body, that is in the same iteration
  (``incall_scratch.uncovered_reads`` on the body alone), and
* along each dimension, either every access is the same point whose symbols the body takes from outside and never
  reassigns (the iteration's own element: the dimension goes), or the dimension keeps its whole literal extent;
  the dimensions kept hold at most ``limit`` elements.

Each such body then holds the array as a transient of its own over the dimensions kept, the memlets that led to it
through the maps go, and the array goes from the SDFG once nothing accesses it. No value crosses iterations, so the
results are the same.
"""

from __future__ import annotations

import math
import re

import dace
from dace import nodes, subsets, symbolic
from dace.sdfg.state import ConditionalBlock, LoopRegion

from dace_fortran.substitute.incall_scratch import uncovered_reads


def _control_flow_code(sdfg: dace.SDFG) -> str:
    """The text of every interstate edge, loop header and branch condition of ``sdfg`` (not of its nested SDFGs)."""
    parts = []
    for e in sdfg.all_interstate_edges():
        parts.append(e.data.condition.as_string)
        parts.extend(e.data.assignments.values())
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion):
            parts += [
                c.as_string
                for c in (region.loop_condition, region.init_statement, region.update_statement)
                if c is not None
            ]
        elif isinstance(region, ConditionalBlock):
            parts += [cond.as_string for cond, _ in region.branches if cond is not None]
    return "\n".join(parts)


def _mentions(code: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", code) is not None


def _assigned_symbols(sdfg: dace.SDFG) -> set[str]:
    """The symbols ``sdfg``'s own control flow assigns: interstate assignments and loop variables."""
    out = {k for e in sdfg.all_interstate_edges() for k in e.data.assignments}
    out |= {r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable}
    return out


def _inner_subsets(body: dace.SDFG, inner: str) -> list[tuple] | None:
    """Every (edge, attribute) of ``body`` that subsets ``inner``, with the subset; None if ``inner`` is accessed in a
    way the rewrite does not follow (a nested SDFG, a view, a memlet that leaves its subset of it implicit)."""
    out = []
    for state in body.all_states():
        for n in state.data_nodes():
            if n.data != inner:
                continue
            for e in state.all_edges(n):
                if e.data.is_empty():
                    continue
                other = e.src if e.dst is n else e.dst
                if isinstance(other, nodes.NestedSDFG) or (
                    isinstance(other, nodes.AccessNode)
                    and isinstance(body.arrays[other.data], (dace.data.View, dace.data.Reference))
                ):
                    return None
                attr = "subset" if e.data.data == inner else "other_subset"
                sub = getattr(e.data, attr)
                if sub is None:
                    return None
                out.append((e, attr, sub))
    return out


def _kept_dimensions(body: dace.SDFG, inner: str, accesses: list[tuple], outside: set[str]) -> list[int] | None:
    """The dimensions of ``inner`` the iteration's own copy keeps: those not accessed at one and the same point
    everywhere (symbols from ``outside`` only); None if one of them has no literal extent."""
    desc = body.arrays[inner]
    kept = []
    for d in range(len(desc.shape)):
        points = set()
        for _e, _attr, sub in accesses:
            b, end, _step = sub.ndrange()[d]
            if symbolic.simplify(b - end) != 0:
                points = None
                break
            points.add(symbolic.simplify(b))
        if points is not None and len(points) == 1:
            (point,) = points
            if {str(s) for s in symbolic.symlist(point)} <= outside:
                continue
        if symbolic.issymbolic(desc.shape[d]):
            return None
        kept.append(d)
    return kept


def _body_plan(body: nodes.NestedSDFG, inner: str, limit: int) -> list[int] | None:
    """The dimensions ``body`` keeps when it holds ``inner`` as its own, or None when it cannot."""
    sdfg = body.sdfg
    desc = sdfg.arrays.get(inner)
    if not isinstance(desc, dace.data.Array) or isinstance(desc, (dace.data.View, dace.data.Reference)):
        return None
    if _mentions(_control_flow_code(sdfg), inner):
        return None
    accesses = _inner_subsets(sdfg, inner)
    if accesses is None:
        return None
    outside = set(body.symbol_mapping) - _assigned_symbols(sdfg)
    kept = _kept_dimensions(sdfg, inner, accesses, outside)
    if kept is None or math.prod(int(desc.shape[d]) for d in kept) > limit:
        return None
    if uncovered_reads(sdfg, [inner])[inner]:
        return None
    return kept


def _leaves(state: dace.SDFGState, node: nodes.AccessNode) -> list | None:
    """The innermost edges of every memlet path from or to ``node``; None if one leaves through no map or ends
    elsewhere than at a nested SDFG."""
    out = []
    for e in state.all_edges(node):
        if e.data.is_empty():
            continue
        if not isinstance(e.dst if e.src is node else e.src, (nodes.MapEntry, nodes.MapExit)):
            return None
        for leaf in state.memlet_tree(e).leaves():
            end = leaf.dst if e.src is node else leaf.src
            if not isinstance(end, nodes.NestedSDFG):
                return None
            out.append(leaf)
    return out


def _plans(sdfg: dace.SDFG, name: str, limit: int) -> dict | None:
    """Per (state, body) that takes the array ``name``: the body's name for it, the dimensions it keeps and the
    innermost edges that lead it in and out; None unless every access of ``name`` is such a body that can hold it."""
    plans = {}
    for state in sdfg.all_states():
        for n in state.data_nodes():
            if n.data != name:
                continue
            leaves = _leaves(state, n) if state.entry_node(n) is None else None
            if leaves is None:
                return None
            for leaf in leaves:
                body = leaf.dst if isinstance(leaf.dst, nodes.NestedSDFG) else leaf.src
                inner = leaf.dst_conn if leaf.dst is body else leaf.src_conn
                if (state, body) not in plans:
                    kept = _body_plan(body, inner, limit)
                    if kept is None:
                        return None
                    plans[(state, body)] = (inner, kept, [])
                if plans[(state, body)][0] != inner:
                    return None
                plans[(state, body)][2].append(leaf)
    return plans


def _localise(state: dace.SDFGState, body: nodes.NestedSDFG, inner: str, kept: list[int], leaves: list) -> None:
    """Make ``inner`` a register array of ``body`` over the dimensions ``kept`` and remove the memlet paths that led
    it in and out of the body."""
    sdfg = body.sdfg
    desc = sdfg.arrays[inner]
    for e, attr, sub in _inner_subsets(sdfg, inner):
        setattr(e.data, attr, subsets.Range([sub.ndrange()[d] for d in kept] if kept else [(0, 0, 1)]))
    sdfg.remove_data(inner, validate=False)
    sdfg.add_array(
        inner,
        [desc.shape[d] for d in kept] or [1],
        desc.dtype,
        storage=dace.StorageType.Register,
        transient=True,
        lifetime=dace.AllocationLifetime.Scope,
    )
    for leaf in leaves:
        state.remove_memlet_path(leaf)


def _drop_ordering_nodes(sdfg: dace.SDFG, name: str) -> bool:
    """Remove the access nodes of ``name`` that only order others (empty edges), passing that order on; False if one
    still carries data."""
    for state in sdfg.all_states():
        for n in [n for n in state.data_nodes() if n.data == name]:
            if any(not e.data.is_empty() for e in state.all_edges(n)):
                return False
            for p in state.in_edges(n):
                for s in state.out_edges(n):
                    state.add_nedge(p.src, s.dst, dace.Memlet())
            state.remove_node(n)
    return True


def localise_iteration_scratch(sdfg: dace.SDFG, limit: int = 64) -> dict[str, list[str]]:
    """Hold every transient array of ``sdfg`` that kernel bodies only use within one iteration in each body instead
    (see the module docstring). Returns, per array, the bodies that hold it now."""
    code = _control_flow_code(sdfg)
    done = {}
    for name, desc in list(sdfg.arrays.items()):
        if (
            not desc.transient
            or not isinstance(desc, dace.data.Array)
            or isinstance(desc, (dace.data.View, dace.data.Reference))
            or _mentions(code, name)
        ):
            continue
        plans = _plans(sdfg, name, limit)
        if not plans:
            continue
        for (state, body), (inner, kept, leaves) in plans.items():
            _localise(state, body, inner, kept, leaves)
        if _drop_ordering_nodes(sdfg, name):
            sdfg.remove_data(name, validate=False)
        done[name] = sorted(body.label for _state, body in plans)
    return done

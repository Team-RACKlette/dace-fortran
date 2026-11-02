# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scratch images of arrays of records cut down to the elements the SDFG accesses.

A scratch image holds every element of a module array of records (``flux(d)%vf(l)%sf`` for every ``d`` and ``l``),
as the binding packed it before the image became the substitute's own transient. Once the element loops are
unrolled, every access picks its elements by literals, and the elements never accessed need no storage: an image
keeps, along each record dimension, the range from the first element accessed to the last.
"""

from __future__ import annotations

import dace
from dace import nodes, subsets, symbolic


def shrink_record_ranges(sdfg: dace.SDFG, record_dims: dict[str, int]) -> dict[str, tuple]:
    """Cut each transient image in ``record_dims`` (image -> its record dimensions, the leading ones before its field
    dimensions) down, along those dimensions (those with a literal extent), to the range of elements the SDFG
    accesses, in the nested SDFGs too. An image with an access whose element is not a literal keeps the dimension.
    Returns, per image changed, the new shape."""
    done = {}
    for image, dims in sorted(record_dims.items()):
        desc = sdfg.arrays.get(image)
        if desc is None:
            continue
        changed = False
        for dim in range(min(dims, len(desc.shape) - 1)):
            if symbolic.issymbolic(desc.shape[dim]):
                continue
            accessed = _accessed(sdfg, image, dim)
            if accessed is None or not accessed:
                continue
            lo, hi = min(accessed), max(accessed)
            if lo == 0 and hi == int(desc.shape[dim]) - 1:
                continue
            for edge in sdfg.all_interstate_edges():
                if any(m.data == image for m in edge.data.get_read_memlets(sdfg.arrays)):
                    raise ValueError(f"{image}: an interstate edge reads it, so its extents cannot shrink")
            _shift(sdfg, image, dim, lo, hi - lo + 1)
            changed = True
        if changed:
            done[image] = tuple(desc.shape)
    return done


def _accessed(container: dace.SDFG, array: str, dim: int) -> set[int] | None:
    """The literal indices along ``dim`` of every access of ``array`` in ``container`` and the nested SDFGs that take
    it; None when an access is not a literal."""
    found: set[int] = set()
    for state in container.all_states():
        for edge in state.edges():
            subset = _subset(edge, array)
            if subset is None:
                continue
            lo, hi, _step = subset.ranges[dim]
            if symbolic.issymbolic(lo) or symbolic.issymbolic(hi):
                return None
            found.update(range(int(lo), int(hi) + 1))
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                for edge in [*state.in_edges(node), *state.out_edges(node)]:
                    conn = edge.dst_conn if edge.dst is node else edge.src_conn
                    if edge.data.data == array and conn and conn in node.sdfg.arrays \
                            and len(node.sdfg.arrays[conn].shape) == len(container.arrays[array].shape):
                        inner = _accessed(node.sdfg, conn, dim)
                        if inner is None:
                            return None
                        found.update(inner)
    return found


def _shift(container: dace.SDFG, array: str, dim: int, lo: int, extent: int) -> None:
    """Every access of ``array`` along ``dim`` moved down by ``lo``, in the nested SDFGs that take it too, and the
    dimension's extent set to ``extent``."""
    desc = container.arrays[array]
    for state in container.all_states():
        for edge in state.edges():
            subset = _subset(edge, array)
            if subset is not None:
                a, b, step = subset.ranges[dim]
                subset.ranges[dim] = (a - lo, b - lo, step)
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                for edge in [*state.in_edges(node), *state.out_edges(node)]:
                    conn = edge.dst_conn if edge.dst is node else edge.src_conn
                    if edge.data.data == array and conn and conn in node.sdfg.arrays \
                            and len(node.sdfg.arrays[conn].shape) == len(desc.shape):
                        _shift(node.sdfg, conn, dim, lo, extent)
    shape = list(desc.shape)
    shape[dim] = extent
    desc.set_shape(tuple(shape))


def _subset(edge, array: str) -> subsets.Range | None:
    """The subset of ``array`` an edge's memlet names, on either side; None when the memlet does not touch it."""
    memlet = edge.data
    if memlet.is_empty():
        return None
    if memlet.data == array:
        return memlet.subset
    if memlet.other_subset is not None and any(
            isinstance(n, nodes.AccessNode) and n.data == array for n in (edge.src, edge.dst)):
        return memlet.other_subset
    return None

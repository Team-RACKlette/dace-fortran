# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Arrays of records handed over member by member instead of packed.

The binding of a kernel reading ``q(i)%sf`` packs the members of all elements into one buffer, on every call, and
unpacks what the kernel wrote.  Once every access of the SDFG picks its element by a constant -- as after
specialising the element loops away -- the buffer splits into one array per element
(:func:`dace_fortran.array_split.split_array`), and each of them is a member as the caller holds it: the binding
aliases ``q(k)%sf`` in place (:func:`dace_fortran.bindings.loop_copy.render_aos_alloc_members`).

The buffer spans the largest extent of each member dimension over all elements. Members of different shapes make it
much larger than they are: boundary condition types kept as one plane per face of a domain, each thin along its
own axis, take a buffer as large as a field per face. With ``own_shapes``, such an array (one with several
record dimensions too, or with its elements read in control-flow code) splits as well, each element keeping its
member's own extents and lower bounds.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import dace
from dace import SDFG

from dace_fortran.array_split import constant_indices, rebase, split_array
from dace_fortran.bindings.flatten_plan import FlattenPlan, FlattenRecipe
from dace_fortran.bindings.frozen_signature import attach_to_sdfg, get_frozen_signature
from dace_fortran.bindings.loop_copy import aos_outer_rank, own_shape_symbols


def _split_own_shapes(sdfg: SDFG, recipe: FlattenRecipe, flat: str) -> dict[tuple[int, ...], str] | None:
    """Split the companion ``flat`` along its record dimensions (the leading ones) into one array per element its
    accesses pick, each with its member's own extents and lower bounds: the symbols :func:`own_shape_symbols` names,
    its subscripts moved from the lower bounds the elements share (the recipe's) to its own. Returns the arrays by
    element indices (0-based); None, with ``sdfg`` unchanged, unless every access, in control-flow code too, picks its
    element by literals."""
    rank = aos_outer_rank(recipe)
    indices = [constant_indices(sdfg, flat, d, code=True) for d in range(rank)]
    if any(not found for found in indices):
        return None
    names = [f"{flat}_{'_'.join(map(str, index))}" for index in itertools.product(*map(sorted, indices))]
    symbols = [s for name in names for group in own_shape_symbols(name, len(recipe.cap_symbols)) for s in group]
    if any(n in sdfg.arrays or n in sdfg.symbols for n in names + symbols):
        return None
    parts = {(): flat}
    for _ in range(rank):
        parts = {
            (*index, k): name
            for index, whole in parts.items()
            for k, name in split_array(sdfg, whole, 0, code=True).items()
        }
    for name in parts.values():
        caps, lbs = own_shape_symbols(name, len(recipe.cap_symbols))
        for symbol in caps + lbs:
            sdfg.add_symbol(symbol, dace.int64)
        rebase(sdfg, name, caps, [f"{shared} - {own}" for shared, own in zip(recipe.lb_symbols, lbs, strict=True)])
    return parts


def split_record_companions(sdfg: SDFG, plan: FlattenPlan, own_shapes: bool = False) -> dict[str, dict]:
    """Split the packed companion of every array of records whose accesses all pick their element by a constant, and
    return the arrays each split into, by element index.

    The frozen signature attached to ``sdfg`` takes one argument per element in place of the companion
    (``FrozenArg.element_of``), so the binding passes the members themselves.

    ``own_shapes``: a companion this leaves packed splits by :func:`_split_own_shapes` where it can; its arrays are
    returned by tuples of element indices, and their arguments are marked ``own_shape``, with the new extents and
    lower bounds among the signature's symbols.
    """
    frozen = get_frozen_signature(sdfg)
    if frozen is None:
        return {}
    args = list(frozen.args)
    symbols = list(frozen.free_symbols)
    dtypes = dict(frozen.symbol_dtypes)
    split: dict[str, dict] = {}
    for entry in plan.entries:
        recipe = entry.recipe
        if not recipe.aos_alloc:
            continue
        flat = recipe.flat_names[0]
        position = next((i for i, a in enumerate(args) if a.sdfg_name == flat), None)
        if position is None:
            continue
        companion = args[position]
        # One record dimension only: an element of a rank-N array of records is no single split index.
        names = split_array(sdfg, flat) if aos_outer_rank(recipe) == 1 else None
        if names is not None:
            args[position : position + 1] = [
                replace(
                    companion,
                    sdfg_name=name,
                    rank=companion.rank - 1,
                    shape=tuple(companion.shape[1:]),
                    element_of=flat,
                    element_index=index,
                )
                for index, name in names.items()
            ]
            split[flat] = names
            continue
        parts = _split_own_shapes(sdfg, recipe, flat) if own_shapes else None
        if parts is None:
            continue
        rank = len(recipe.cap_symbols)
        args[position : position + 1] = [
            replace(
                companion,
                sdfg_name=name,
                rank=rank,
                shape=own_shape_symbols(name, rank)[0],
                element_of=flat,
                element_indices=index,
                own_shape=True,
            )
            for index, name in parts.items()
        ]
        for name in parts.values():
            for symbol in (s for group in own_shape_symbols(name, rank) for s in group):
                symbols.append(symbol)
                dtypes[symbol] = "int64"
        split[flat] = parts
    if split:
        attach_to_sdfg(sdfg, replace(frozen, args=tuple(args), free_symbols=tuple(symbols), symbol_dtypes=dtypes))
    return split

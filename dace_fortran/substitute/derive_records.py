"""Records of a grid no recording run has seen, derived from the records of two smaller grids.

A specialised substitute is built from the values its recording runs saw: module variables, array extents, value
sets per rank. Recording at the grid the substitute is for needs the caller's whole memory at that grid, so a grid
the caller alone cannot hold could never be recorded. Most recorded values do not depend on the grid at all, and
those that do follow a few per-rank quantities (the cells a rank holds along each axis) plus a constant: an array
extent is its rank's cells plus its halo. :func:`derive` takes the records of the same recording stage at two grids
A and B -- the same ranks, the same decomposition shape -- and the per-rank values of those quantities (the
*drivers*) at A, B and the target, and writes the records the target would have given:

- a value equal at A and B is the target's value;
- a value that differs is a driver plus a constant at both grids: the target's driver plus that constant (when two
  drivers fit, as two axes with the same cells do, they must give the same target value);
- a value given in ``overrides`` (a case parameter that follows the grid, such as a time step) is that value;
- a value that changes from call to call at both grids (the simulation time) is no value a build relies on (it is
  neither baked nor one of a rank's set): A's values stand;
- so is a value the build reads on every call (``runtime``, as make_substitute's ``--runtime``), such as a
  coefficient computed once from the cell size: A's values stand;
- anything else is refused, by name: the records would not be the target's.

A record file holds ``name value`` lines, one call per block, blocks separated by an empty line; a run on several
ranks has one file per rank (``<record>.<rank>``) next to the combined file (``<record>``: the first calls of every
rank, rank by rank). Alias records (``<record>.aliases``, member storage the calls shared) and the recorder's types
(``recorded_types.json`` next to the record) do not depend on the grid: they are copied, after checking that A and B
agree (alias records on every call both recorded: a run stopped after some calls may have recorded one more).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path


class DerivationError(ValueError):
    """The records of A and B do not determine the target's."""


def runtime_names(runtime: Iterable[str]) -> frozenset[str]:
    """The values a build reads on every call (``--runtime``, ``module::name``) as a record may name them: lower-case,
    and the bare ``name`` with each (a record of the binding's symbols and scalar inputs names a module variable so)."""
    names = {n.lower() for n in runtime}
    return frozenset(names | {n.rsplit("::", 1)[-1] for n in names})


def _calls(text: str) -> list[list[tuple[str, str]]]:
    """The calls of a record file: per call its ``(name, value)`` lines in order."""
    calls, current = [], []
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            current.append((parts[0], parts[1].strip()))
        elif not line.strip() and current:
            calls.append(current)
            current = []
    if current:
        calls.append(current)
    return calls


def _render(calls: list[list[tuple[str, str]]]) -> str:
    return "".join("".join(f"{n} {v}\n" for n, v in call) + "\n" for call in calls)


def _values(calls) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {}
    for call in calls:
        for n, v in call:
            seen.setdefault(n, set()).add(v)
    return seen


def _as_int(v: str) -> int | None:
    return int(v) if re.fullmatch(r"[-+]?\d+", v) else None


def _target_values(a: dict[str, set[str]], b: dict[str, set[str]], da: dict[str, int], db: dict[str, int],
                   dt: dict[str, int], overrides: dict[str, str], runtime: frozenset[str],
                   where: str) -> dict[str, str]:
    """name -> the target's value for every name whose value differs between A and B (one rank)."""
    if a.keys() != b.keys():
        raise DerivationError(f"{where}: grids A and B recorded different names (only A: "
                              f"{sorted(a.keys() - b.keys())[:6]}, only B: {sorted(b.keys() - a.keys())[:6]}); "
                              "pick grids that split across the ranks alike")
    out = {}
    for name in sorted(a):
        if a[name] == b[name]:
            continue
        if name in overrides:
            out[name] = overrides[name]
            continue
        if name in runtime or (len(a[name]) > 1 and len(b[name]) > 1):
            continue
        va, vb = (_as_int(next(iter(s))) if len(s) == 1 else None for s in (a[name], b[name]))
        if va is None or vb is None:
            raise DerivationError(f"{where}: {name} differs between A and B ({sorted(a[name])[:3]} / "
                                  f"{sorted(b[name])[:3]}) and is neither one integer per run, overridden nor "
                                  "read on every call")
        targets = {dt[d] + (va - da[d]) for d in da if va - da[d] == vb - db[d]}
        if len(targets) != 1:
            raise DerivationError(f"{where}: {name} ({va} at A, {vb} at B) is "
                                  + ("no driver plus a constant" if not targets else
                                     f"several drivers plus constants giving {sorted(targets)} at the target"))
        out[name] = str(targets.pop())
    return out


def _companions(record: Path, ranks: int) -> tuple[list[Path], list[Path]]:
    """A record file's rank files and alias files (those that exist)."""
    rank_files = [record.with_name(f"{record.name}.{r}") for r in range(ranks)] if ranks > 1 else []
    aliases = [record.with_name(record.name + ".aliases")]
    aliases += [record.with_name(f"{record.name}.aliases.{r}") for r in range(ranks)]
    return rank_files, [f for f in aliases if f.is_file()]


def derive(a: Path, b: Path, out: Path, drivers: dict, overrides: dict[str, str] | None = None,
           runtime: list[str] = ()) -> list[str]:
    """Write ``out`` (and its rank and alias files) as the target's record of the recording whose records at grids A
    and B are ``a`` and ``b``. ``drivers``: ``{"a": [...], "b": [...], "target": [...]}``, per rank a mapping of driver
    name -> value. ``runtime``: the names the build reads on every call instead of baking them in. Both name module
    variables as ``module::name``, which covers the bare ``name`` too: a record of the binding's symbols and scalar
    inputs names a scalar module variable so. Returns the names given new values (rank 0's, for a report).

    :raises DerivationError: when the records do not determine the target's (see the module documentation).
    """
    overrides = overrides or {}
    overrides = {**{n.rsplit("::", 1)[-1]: v for n, v in overrides.items()}, **overrides}
    runtime = runtime_names(runtime)
    ranks = len(drivers["target"])
    if not (len(drivers["a"]) == len(drivers["b"]) == ranks):
        raise DerivationError("the drivers name a different number of ranks at A, B and the target")
    out.parent.mkdir(parents=True, exist_ok=True)
    rank_a, alias_a = _companions(a, ranks)
    rank_b, alias_b = _companions(b, ranks)
    if [f.name for f in alias_a] != [f.name for f in alias_b] or not all(f.is_file() for f in [a, b, *rank_a, *rank_b]):
        raise DerivationError(f"{a.name}: the files of A and B differ (a rank's record or an alias record is missing)")
    for fa, fb in zip(alias_a, alias_b, strict=True):
        calls_a, calls_b = _calls(fa.read_text()), _calls(fb.read_text())
        if calls_a[:len(calls_b)] != calls_b[:len(calls_a)]:
            raise DerivationError(f"{fa.name}: the alias records of A and B differ")
        shutil.copyfile(fa, out.with_name(out.name + fa.name[len(a.name):]))
    types_a, types_b = a.with_name("recorded_types.json"), b.with_name("recorded_types.json")
    if types_a.is_file() or types_b.is_file():
        if not (types_a.is_file() and types_b.is_file()) or types_a.read_text() != types_b.read_text():
            raise DerivationError(f"{types_a.name}: the recorders' types at A and B differ")
        shutil.copyfile(types_a, out.with_name(types_a.name))
    per_rank, changed = [], []
    for r in range(ranks):
        fa, fb = (rank_a[r], rank_b[r]) if ranks > 1 else (a, b)
        ca, cb = _calls(fa.read_text()), _calls(fb.read_text())
        for grid, calls in (("a", ca), ("b", cb)):  # the drivers' own records check the values the caller gave
            for n, v in drivers[grid][r].items():
                if n in _values(calls) and _values(calls)[n] != {str(v)}:
                    raise DerivationError(f"{fa.name}: {n} recorded {sorted(_values(calls)[n])} at "
                                          f"{grid.upper()}, the drivers say {v}")
        target = _target_values(_values(ca), _values(cb), drivers["a"][r], drivers["b"][r], drivers["target"][r],
                                overrides, runtime, fa.name)
        per_rank.append(target)
        if ranks > 1:
            out.with_name(f"{out.name}.{r}").write_text(_render([[(n, target.get(n, v)) for n, v in call] for call in ca]))
        if r == 0:
            changed = sorted(target)
    # the combined file: the first calls of every rank, rank by rank
    combined = _calls(a.read_text())
    per = len(combined) // ranks if ranks > 1 else len(combined)
    if ranks > 1 and per * ranks != len(combined):
        raise DerivationError(f"{a.name}: {len(combined)} calls do not split into {ranks} ranks alike")
    out.write_text(_render([[(n, per_rank[min(i // per, ranks - 1)].get(n, v)) for n, v in call]
                            for i, call in enumerate(combined)]))
    return changed


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", type=Path, help="a record at grid A (its rank and alias files next to it)")
    ap.add_argument("b", type=Path, help="the same recording's record at grid B")
    ap.add_argument("out", type=Path, help="the target's record to write")
    ap.add_argument("--drivers", type=Path, required=True,
                    help='JSON {"a": [{name: value} per rank], "b": [...], "target": [...]}')
    ap.add_argument("--override", action="append", default=[], metavar="NAME=VALUE",
                    help="a value that follows the grid by the case's choice (repeatable)")
    ap.add_argument("--runtime", action="append", default=[], metavar="NAME",
                    help="a value the build reads on every call: A's values stand (repeatable)")
    args = ap.parse_args(argv)
    overrides = dict(o.split("=", 1) for o in args.override)
    try:
        changed = derive(args.a, args.b, args.out, json.loads(args.drivers.read_text()), overrides, args.runtime)
    except DerivationError as e:
        print(f"derive_records: {e}", file=sys.stderr)
        return 1
    print(f"{args.out}: {len(changed)} values follow the grid (rank 0): {changed[:12]}{' ...' if len(changed) > 12 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

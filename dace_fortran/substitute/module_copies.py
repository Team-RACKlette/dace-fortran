"""Copies of a build's modules with accessors to private variables, for a binding or a stand-in to reach them.

link_substitute.sh compiles each copy with the build's own command for the module and links it in place of the
original object (a substitute.json "access" entry: module, source, copy). One copy per module serves everything linked
with it, so the variables of every user -- a binding and the substitute entry checking baked values, say -- go into
the same copy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

_MODULE = re.compile(r"^\s*module\s+(?!procedure\b)([a-z_]\w*)\s*$", re.IGNORECASE | re.MULTILINE)


def module_files(dirs: Iterable[Path]) -> dict[str, Path]:
    """Module name -> the source file defining it, among the ``*.f90`` files of ``dirs`` (the first directory wins)."""
    files: dict[str, Path] = {}
    for path in (p for d in dirs for p in sorted(Path(d).glob("*.f90"))):
        for name in _MODULE.findall(path.read_text(errors="replace")):
            files.setdefault(name.lower(), path)
    return files


def write_access_copies(
    wanted: Mapping[str, Sequence[str]], files: Mapping[str, Path], out: Path,
    releases: Mapping[str, Sequence] | None = None
) -> list[dict]:
    """For each module of ``wanted`` (and of ``releases``): a copy of its source (from ``files``) in ``out`` with
    accessors to the variables ``wanted`` names and release procedures for ``releases`` (module ->
    ``module_access.ReleaseSpec``), the variables (``ExposedVariable`` by name) and the ``use`` lines their
    declarations need.

    :raises dace_fortran.bindings.module_access.ModuleAccessError: a variable cannot be made accessible.
    """
    from dace_fortran.bindings.module_access import declaration_imports, expose_module_variables  # noqa: PLC0415

    sources = {mod: path.read_text(errors="replace") for mod, path in files.items()}
    out.mkdir(parents=True, exist_ok=True)
    access = []
    wanted = {**{mod: [] for mod in (releases or {})}, **wanted}
    for mod, names in sorted(wanted.items()):
        names = list(dict.fromkeys(n.lower() for n in names))
        copy_text, variables, shrunk = expose_module_variables(sources[mod], mod, names, (releases or {}).get(mod, ()))
        copy = out / files[mod].name
        copy.write_text(copy_text)
        access.append({
            "module": mod,
            "source": str(files[mod]),
            "copy": str(copy),
            "variables": variables,
            "imports": declaration_imports(list(variables.values()), mod, sources),
            "shrunk": shrunk,
        })
        print(f"accessors for {mod}: {' '.join(names)}"
              + (f"; release procedures: {' '.join(r.name for r in releases[mod])}" if releases and mod in releases else ""),
              flush=True)
    return access


def access_entries(access: Sequence[dict]) -> list[dict]:
    """The substitute.json "access" entries of ``access`` (write_access_copies): what link_substitute.sh compiles and
    swaps, and the variables each copy exposes."""
    return [
        {"module": a["module"], "source": a["source"], "copy": a["copy"], "variables": sorted(a["variables"])}
        for a in access
    ]

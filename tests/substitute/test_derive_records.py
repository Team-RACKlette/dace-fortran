"""The target's records derived from two smaller grids' (dace_fortran.substitute.derive_records)."""
from pathlib import Path

import pytest

from dace_fortran.substitute.derive_records import DerivationError, derive


def _write(folder: Path, name: str, calls_per_rank: list[list[dict[str, str]]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    def render(calls):
        return "".join("".join(f"{k} {v}\n" for k, v in call.items()) + "\n" for call in calls)
    for r, calls in enumerate(calls_per_rank):
        (folder / f"{name}.{r}").write_text(render(calls))
    (folder / name).write_text("".join(render(calls[:2]) for calls in calls_per_rank))


def _grid(cells: tuple[int, int], dt: str, step0: int = 0) -> list[list[dict[str, str]]]:
    """Two ranks, x split: each rank holds cells[r] cells; three calls with a step counter."""
    return [[{"mod::m": str(c - 1), "mod::bc_beg": str(-1 if r == 0 else -2), "q_d0": str(c + 8), "flux_d0": str(c + 1),
              "coef_d0": "5", "mod::dt": dt, "t_step": str(step0 + k)} for k in range(3)] for r, c in enumerate(cells)]


DRIVERS = {"a": [{"mod::m": 49}, {"mod::m": 49}], "b": [{"mod::m": 29}, {"mod::m": 29}],
           "target": [{"mod::m": 499}, {"mod::m": 499}]}


def test_values_following_the_grid_take_the_targets(tmp_path: Path):
    _write(tmp_path / "a", "rec", _grid((50, 50), "1.0E-006"))
    _write(tmp_path / "b", "rec", _grid((30, 30), "2.0E-006"))
    (tmp_path / "a" / "rec.aliases").write_text("same q 0 p 0\n\n")
    (tmp_path / "b" / "rec.aliases").write_text("same q 0 p 0\n\n")
    for g in "ab":
        (tmp_path / g / "recorded_types.json").write_text('{"mod::m": "integer"}')
    (tmp_path / "a" / "unrelated.json").write_text("{}")  # other files of a stage's folder are not records
    changed = derive(tmp_path / "a" / "rec", tmp_path / "b" / "rec", tmp_path / "t" / "rec", DRIVERS, {"mod::dt": "1.0E-007"})
    assert changed == ["flux_d0", "mod::dt", "mod::m", "q_d0"]
    rank1 = (tmp_path / "t" / "rec.1").read_text()
    assert "q_d0 508\n" in rank1 and "flux_d0 501\n" in rank1 and "mod::m 499\n" in rank1
    assert "coef_d0 5\n" in rank1 and "mod::bc_beg -2\n" in rank1 and "mod::dt 1.0E-007\n" in rank1
    assert rank1.count("\n\n") == 3 and "t_step 2\n" in rank1  # every call kept, the step counter as A had it
    combined = (tmp_path / "t" / "rec").read_text()
    assert combined.count("\n\n") == 4 and combined.count("q_d0 508\n") == 4
    assert (tmp_path / "t" / "rec.aliases").read_text() == "same q 0 p 0\n\n"
    assert (tmp_path / "t" / "recorded_types.json").read_text() == '{"mod::m": "integer"}'
    assert not (tmp_path / "t" / "unrelated.json").exists()


def test_values_changing_from_call_to_call_keep_a_values(tmp_path: Path):
    a, b = _grid((50, 50), "1.0E-006"), _grid((30, 30), "2.0E-006")
    for grid, dt in ((a, 1.0e-6), (b, 2.0e-6)):
        for calls in grid:
            for k, call in enumerate(calls):
                call["mod::mytime"] = f"{k * dt:.6E}"
    _write(tmp_path / "a", "rec", a)
    _write(tmp_path / "b", "rec", b)
    derive(tmp_path / "a" / "rec", tmp_path / "b" / "rec", tmp_path / "t" / "rec", DRIVERS, {"mod::dt": "1.0E-007"})
    rank0 = (tmp_path / "t" / "rec.0").read_text()
    assert "mod::mytime 0.000000E+00\n" in rank0 and "mod::mytime 2.000000E-06\n" in rank0
    for calls in b:
        calls[0]["mod::mytime"] = calls[1]["mod::mytime"] = calls[2]["mod::mytime"]  # constant at B: refused
    _write(tmp_path / "c", "rec", b)
    with pytest.raises(DerivationError, match="mod::mytime"):
        derive(tmp_path / "a" / "rec", tmp_path / "c" / "rec", tmp_path / "t2" / "rec", DRIVERS, {"mod::dt": "1.0E-007"})


def test_alias_records_agree_on_the_calls_both_recorded(tmp_path: Path):
    _write(tmp_path / "a", "rec", _grid((50, 50), "1.0E-006"))
    _write(tmp_path / "b", "rec", _grid((30, 30), "2.0E-006"))
    block = "member s_sf 0\nalias s_sf 0 s\n\n"
    (tmp_path / "a" / "rec.aliases").write_text(block * 3)
    (tmp_path / "b" / "rec.aliases").write_text(block * 4)  # stopped one call later
    derive(tmp_path / "a" / "rec", tmp_path / "b" / "rec", tmp_path / "t" / "rec", DRIVERS, {"mod::dt": "1.0E-007"})
    assert (tmp_path / "t" / "rec.aliases").read_text() == block * 3
    (tmp_path / "b" / "rec.aliases").write_text(block + "member s_sf 0\nalias s_sf 0 other\n\n")
    with pytest.raises(DerivationError, match="alias records"):
        derive(tmp_path / "a" / "rec", tmp_path / "b" / "rec", tmp_path / "t2" / "rec", DRIVERS, {"mod::dt": "1.0E-007"})


def test_values_read_on_every_call_keep_a_values(tmp_path: Path):
    a, b = _grid((50, 50), "1.0E-006"), _grid((30, 30), "2.0E-006")
    for grid, alf in ((a, "4.0E-004"), (b, "1.1E-003")):  # set once from the cell size, the same on every call
        for calls in grid:
            for call in calls:
                call["mod::alf"] = alf
    _write(tmp_path / "a", "rec", a)
    _write(tmp_path / "b", "rec", b)
    with pytest.raises(DerivationError, match="mod::alf"):
        derive(tmp_path / "a" / "rec", tmp_path / "b" / "rec", tmp_path / "t" / "rec", DRIVERS, {"mod::dt": "1.0E-007"})
    changed = derive(tmp_path / "a" / "rec", tmp_path / "b" / "rec", tmp_path / "t" / "rec", DRIVERS,
                     {"mod::dt": "1.0E-007"}, ["MOD::alf"])
    assert "mod::alf" not in changed and "mod::alf 4.0E-004\n" in (tmp_path / "t" / "rec.1").read_text()


def test_module_names_cover_the_bare_names_of_a_symbols_record(tmp_path: Path):
    def grid(cells, dt, alf):  # as a binding records its symbols and scalar inputs: no module prefix
        return [[{"mod::m": str(c - 1), "q_d0": str(c + 8), "dt": dt, "alf": alf}] for c in cells]
    _write(tmp_path / "a", "sym", grid((50, 50), "1.0E-006", "4.0E-004"))
    _write(tmp_path / "b", "sym", grid((30, 30), "2.0E-006", "1.1E-003"))
    changed = derive(tmp_path / "a" / "sym", tmp_path / "b" / "sym", tmp_path / "t" / "sym", DRIVERS,
                     {"mod::dt": "1.0E-007"}, ["mod::alf"])
    rank0 = (tmp_path / "t" / "sym.0").read_text()
    assert changed == ["dt", "mod::m", "q_d0"] and "dt 1.0E-007\n" in rank0 and "alf 4.0E-004\n" in rank0


def test_unexplained_values_and_differing_layouts_are_refused(tmp_path: Path):
    _write(tmp_path / "a", "rec", _grid((50, 50), "1.0E-006"))
    _write(tmp_path / "b", "rec", _grid((30, 30), "2.0E-006"))
    with pytest.raises(DerivationError, match="mod::dt"):  # a float following the grid needs an override
        derive(tmp_path / "a" / "rec", tmp_path / "b" / "rec", tmp_path / "t" / "rec", DRIVERS)
    uneven = _grid((30, 29), "2.0E-006")
    for call in uneven[1]:
        call["extra"] = "1"  # a name only an uneven split records
    _write(tmp_path / "c", "rec", uneven)
    uneven_drivers = {**DRIVERS, "b": [{"mod::m": 29}, {"mod::m": 28}]}
    with pytest.raises(DerivationError, match="different names"):
        derive(tmp_path / "a" / "rec", tmp_path / "c" / "rec", tmp_path / "t2" / "rec", uneven_drivers, {"mod::dt": "1.0E-007"})
    bad = {**DRIVERS, "a": [{"mod::m": 48}, {"mod::m": 49}]}
    with pytest.raises(DerivationError, match="the drivers say"):
        derive(tmp_path / "a" / "rec", tmp_path / "b" / "rec", tmp_path / "t3" / "rec", bad, {"mod::dt": "1.0E-007"})


def test_two_drivers_fitting_must_agree_at_the_target(tmp_path: Path):
    def grid(nx, ny):
        return [[{"mod::m": str(nx - 1), "mod::n": str(ny - 1), "q_d1": str(ny + 8)}]]
    _write(tmp_path / "a", "r", grid(40, 40))
    _write(tmp_path / "b", "r", grid(20, 20))
    drv = lambda m, n: [{"mod::m": m, "mod::n": n}]
    derive(tmp_path / "a" / "r", tmp_path / "b" / "r", tmp_path / "t" / "r", {"a": drv(39, 39), "b": drv(19, 19), "target": drv(99, 99)})
    assert "q_d1 108\n" in (tmp_path / "t" / "r").read_text()
    with pytest.raises(DerivationError, match="several drivers"):
        derive(tmp_path / "a" / "r", tmp_path / "b" / "r", tmp_path / "t2" / "r", {"a": drv(39, 39), "b": drv(19, 19), "target": drv(99, 59)})


def test_runtime_names_cover_the_bare_name_too():
    from dace_fortran.substitute.derive_records import runtime_names

    assert runtime_names(["M_IGR::Alf_igr", "t_stop"]) == frozenset({"m_igr::alf_igr", "alf_igr", "t_stop"})
    assert runtime_names([]) == frozenset()

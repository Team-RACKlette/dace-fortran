"""Tests of iteration_scratch (CPU).  Run: python -m pytest test_iteration_scratch.py"""

import dace
import numpy as np
from dace.sdfg import nodes

from dace_fortran.substitute.iteration_scratch import localise_iteration_scratch

N = 7


def _body(written=(0, 1, 2), read=1) -> dace.SDFG:
    """body(i): A[i, c] = f_c(X[i]) for c in ``written``; then OUT[i] = 3 * A[i, read] + A[i, 0]: the iteration
    writes its own row of the scratch array A and reads it back, as a Riemann solver does its face's velocities."""
    body = dace.SDFG("body")
    body.add_symbol("i", dace.int64)
    body.add_array("X", [N], dace.float64)
    body.add_array("A", [N, 3], dace.float64)
    body.add_array("OUT", [N], dace.float64)
    fill = body.add_state("fill")
    t = fill.add_tasklet(
        "fill", {"_x"}, {f"_a{c}" for c in written}, "\n".join(f"_a{c} = _x * {c + 2}" for c in written)
    )
    fill.add_edge(fill.add_read("X"), None, t, "_x", dace.Memlet("X[i]"))
    a = fill.add_write("A")
    for c in written:
        fill.add_edge(t, f"_a{c}", a, None, dace.Memlet(f"A[i, {c}]"))
    use = body.add_state_after(fill, "use")
    u = use.add_tasklet("use", {"_r", "_z"}, {"_o"}, "_o = 3 * _r + _z")
    a = use.add_read("A")
    use.add_edge(a, None, u, "_r", dace.Memlet(f"A[i, {read}]"))
    use.add_edge(a, None, u, "_z", dace.Memlet("A[i, 0]"))
    use.add_edge(u, "_o", use.add_write("OUT"), None, dace.Memlet("OUT[i]"))
    return body


def _kernel(state: dace.SDFGState, out: str, **kwargs) -> nodes.NestedSDFG:
    """A map over i around a body that uses the transient A and writes ``out``."""
    me, mx = state.add_map(f"kernel_{out}", {"i": f"0:{N}"})
    body = state.add_nested_sdfg(_body(**kwargs), {"X", "A"}, {"A", "OUT"}, symbol_mapping={"i": "i"})
    state.add_memlet_path(state.add_read("X"), me, body, dst_conn="X", memlet=dace.Memlet(f"X[0:{N}]"))
    state.add_memlet_path(state.add_read("A"), me, body, dst_conn="A", memlet=dace.Memlet(f"A[0:{N}, 0:3]"))
    state.add_memlet_path(body, mx, state.add_write("A"), src_conn="A", memlet=dace.Memlet(f"A[0:{N}, 0:3]"))
    state.add_memlet_path(body, mx, state.add_write(out), src_conn="OUT", memlet=dace.Memlet(f"{out}[0:{N}]"))
    return body


def _program(name: str, kernels=("B",), **kwargs) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("X", [N], dace.float64)
    sdfg.add_transient("A", [N, 3], dace.float64)
    state = sdfg.add_state()
    for out in kernels:
        sdfg.add_array(out, [N], dace.float64)
        _kernel(state, out, **kwargs)
        state = sdfg.add_state_after(state)
    return sdfg


def _run(sdfg: dace.SDFG, kernels=("B",)) -> dict:
    x = np.arange(1, N + 1, dtype=np.float64)
    outs = {out: np.zeros(N) for out in kernels}
    sdfg(X=x, **outs)
    return outs


def test_a_row_each_iteration_writes_and_reads_back_is_held_per_iteration():
    sdfg = _program("scratch_held")
    assert localise_iteration_scratch(sdfg) == {"A": ["body"]}
    sdfg.validate()
    assert "A" not in sdfg.arrays
    body = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    held = body.sdfg.arrays["A"]
    assert held.transient and tuple(held.shape) == (3,) and held.storage == dace.StorageType.Register
    assert "A" not in body.in_connectors and "A" not in body.out_connectors
    x = np.arange(1, N + 1, dtype=np.float64)
    np.testing.assert_array_equal(_run(sdfg)["B"], 3 * (x * 3) + x * 2)


def test_kernels_sharing_the_array_each_hold_their_own():
    kernels = ("B", "C")
    sdfg = _program("scratch_shared", kernels, read=2)
    assert localise_iteration_scratch(sdfg) == {"A": ["body", "body"]}
    sdfg.validate()
    assert "A" not in sdfg.arrays
    x = np.arange(1, N + 1, dtype=np.float64)
    out = _run(sdfg, kernels)
    for name in kernels:
        np.testing.assert_array_equal(out[name], 3 * (x * 4) + x * 2)


def test_a_read_the_iteration_did_not_write_keeps_the_array():
    sdfg = _program("scratch_uncovered", written=(0, 1), read=2)
    assert localise_iteration_scratch(sdfg) == {}
    assert "A" in sdfg.arrays


def test_a_reader_outside_the_kernels_keeps_the_array():
    sdfg = _program("scratch_read_after")
    sdfg.add_array("S", [1], dace.float64)
    after = sdfg.add_state_after(list(sdfg.states())[-1])
    t = after.add_tasklet("first", {"_a"}, {"_s"}, "_s = _a")
    after.add_edge(after.add_read("A"), None, t, "_a", dace.Memlet("A[0, 1]"))
    after.add_edge(t, "_s", after.add_write("S"), None, dace.Memlet("S[0]"))
    assert localise_iteration_scratch(sdfg) == {}
    assert "A" in sdfg.arrays


def test_more_elements_per_iteration_than_the_limit_keep_the_array():
    sdfg = _program("scratch_limit")
    assert localise_iteration_scratch(sdfg, limit=2) == {}
    assert "A" in sdfg.arrays

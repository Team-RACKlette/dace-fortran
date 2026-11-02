"""join_stencil_states: a stencil's producer and consumer in two states that only symbol assignments keep apart (loop
bounds of later code a source sets between them) fuse into one state, the assignments moving past the consumer, so
the stencil fusion sees the pair."""
import dace
import numpy as np
import pytest

from dace_fortran.substitute.fuse_stencils import fuse_stencil_pairs, join_stencil_states
from tests.substitute.test_fuse_limit import _body


def _split_pair(consumer_range: str = "1:14") -> dace.SDFG:
    """tmp[i] = 2 * a[i] + 1 over 1..14, then iend = p, then b[j] = tmp[j + 1] - 1 over ``consumer_range``, then
    c[0] = iend."""
    sdfg = dace.SDFG("split_pair")
    sdfg.add_symbol("p", dace.int64)
    sdfg.add_symbol("iend", dace.int64)
    for name, size in (("a", 16), ("b", 16), ("c", 1)):
        sdfg.add_array(name, [size], dace.float64)
    sdfg.add_transient("tmp", [16], dace.float64)
    first = sdfg.add_state("produce_state", is_start_block=True)
    pentry, pexit = first.add_map("produce", {"i": "1:15"})
    produce = first.add_nested_sdfg(_body("produce", {"x": "a[i]"}, ("tmp", "tmp[i]"), ["o = 2 * x", "o = x + 1"]),
                                    {"a"}, {"tmp"}, {"i": "i"})
    first.add_memlet_path(first.add_read("a"), pentry, produce, dst_conn="a", memlet=dace.Memlet("a[0:16]"))
    first.add_memlet_path(produce, pexit, first.add_write("tmp"), src_conn="tmp", memlet=dace.Memlet("tmp[i]"))
    second = sdfg.add_state("consume_state")
    centry, cexit = second.add_map("consume", {"j": consumer_range})
    consume = second.add_nested_sdfg(_body("consume", {"r": "tmp[i + 1]"}, ("b", "b[i]"), ["o = r - 1"]),
                                     {"tmp"}, {"b"}, {"i": "j"})
    second.add_memlet_path(second.add_read("tmp"), centry, consume, dst_conn="tmp", memlet=dace.Memlet("tmp[j + 1]"))
    second.add_memlet_path(consume, cexit, second.add_write("b"), src_conn="b", memlet=dace.Memlet("b[j]"))
    third = sdfg.add_state("use_state")
    tasklet = third.add_tasklet("use", {}, {"o"}, "o = iend")
    third.add_edge(tasklet, "o", third.add_write("c"), None, dace.Memlet("c[0]"))
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"iend": "p"}))
    sdfg.add_edge(second, third, dace.InterstateEdge())
    return sdfg


def test_an_assignment_the_consumer_does_not_read_moves_past_it_and_the_pair_fuses(tmp_path):
    sdfg = _split_pair()
    assert join_stencil_states(sdfg) == 1
    assert len(list(sdfg.states())) == 2
    (edge,) = sdfg.edges()
    assert edge.data.assignments == {"iend": "p"}  # now after the consumer, before the state that reads it
    # The joined SDFG computes what the split one does (the fused one is not built: the recomputing fusion of these
    # hand-made bodies leaves their parameter symbol free, joined or not).
    sdfg.build_folder = str(tmp_path / "build")
    a = np.arange(16, dtype=np.float64)
    b = np.zeros(16)
    c = np.zeros(1)
    sdfg(a=a, b=b, c=c, p=7)
    expected = np.zeros(16)
    expected[1:14] = 2 * a[2:15] + 1 - 1
    np.testing.assert_array_equal(b, expected)
    assert c[0] == 7
    assert fuse_stencil_pairs(_split_pair()) == ["tmp (recompute)"]  # joined first, then fused


def test_an_assignment_the_consumer_reads_takes_its_value_there(tmp_path):
    """The consumer's map runs to iend: in its state iend is p, which replaces it, and the states join."""
    sdfg = _split_pair(consumer_range="1:iend")
    assert join_stencil_states(sdfg) == 1
    consume = next(n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.MapEntry)
                   and n.map.label == "consume")
    assert str(consume.map.range[0][1]) == "p - 1"
    sdfg.build_folder = str(tmp_path / "build")
    a, b, c = np.arange(16, dtype=np.float64), np.zeros(16), np.zeros(1)
    sdfg(a=a, b=b, c=c, p=14)
    expected = np.zeros(16)
    expected[1:14] = 2 * a[2:15]
    np.testing.assert_array_equal(b, expected)
    assert c[0] == 14


def test_an_assignment_of_a_value_read_from_data_keeps_the_states_apart():
    """The value reads a container the consumer's state might write: the assignment stays before it."""
    sdfg = _split_pair()
    sdfg.add_scalar("n", dace.int64)
    first, second = list(sdfg.states())[:2]
    sdfg.edges_between(first, second)[0].data.assignments = {"iend": "n"}
    assert join_stencil_states(sdfg) == 0
    assert sdfg.edges_between(first, second)[0].data.assignments == {"iend": "n"}


def test_a_state_before_the_producer_with_nothing_assigned_between_stays_apart():
    """A state filling what the producer's state reads, nothing assigned between them, is not joined with it: that
    join would take the producer's state out of the list the pass walks, and the pair would stay apart (MHD's y and z
    reconstructions on several ranks)."""
    sdfg = _split_pair()
    produce_state = next(s for s in sdfg.states() if s.label == "produce_state")
    sdfg.add_transient("pre", [16], dace.float64)
    sdfg.add_array("d", [16], dace.float64)
    fill = sdfg.add_state_before(produce_state, "fill_state", is_start_block=True)
    fill.add_mapped_tasklet("fill", {"i": "0:16"}, {"x": dace.Memlet("a[i]")}, "y = x", {"y": dace.Memlet("pre[i]")},
                            external_edges=True)
    produce_state.add_mapped_tasklet("use_pre", {"i": "0:16"}, {"x": dace.Memlet("pre[i]")}, "y = x",
                                     {"y": dace.Memlet("d[i]")}, external_edges=True)
    assert join_stencil_states(sdfg) == 1
    maps = {n.map.label: s for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.MapEntry)}
    assert maps["produce"] is maps["consume"]
    assert maps["fill_map"] is not maps["produce"]


def _racing_pair(assigned: bool = False) -> dace.SDFG:
    """_split_pair, with ``assigned`` iend = p between the producer's and the consumer's state (else nothing, and the
    assignment after the consumer's), and in each of the two states a component of its own writing the argument x:
    x[1] = a[0] in the producer's, x[0:2] = a[2:4] in the consumer's (the element copies of a record assignment the
    unrolling split over the states). State fusion refuses their write-write race, though the maps' dataflow allows
    it."""
    sdfg = _split_pair()
    sdfg.add_array("x", [2], dace.float64)
    first, second, third = (next(s for s in sdfg.states() if s.label == label)
                            for label in ("produce_state", "consume_state", "use_state"))
    if not assigned:
        sdfg.edges_between(first, second)[0].data.assignments = {}
        sdfg.edges_between(second, third)[0].data.assignments = {"iend": "p"}
    element = first.add_tasklet("element", {"v"}, {"o"}, "o = v")
    first.add_edge(first.add_read("a"), None, element, "v", dace.Memlet("a[0]"))
    first.add_edge(element, "o", first.add_write("x"), None, dace.Memlet("x[1]"))
    second.add_nedge(second.add_read("a"), second.add_write("x"), dace.Memlet(data="a", subset="2:4", other_subset="0:2"))
    return sdfg


@pytest.mark.parametrize("assigned", [False, True])
def test_states_racing_only_in_components_without_a_map_join_once_those_move_out(tmp_path, assigned):
    sdfg = _racing_pair(assigned)
    assert join_stencil_states(sdfg) == 1
    maps = {n.map.label: s for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.MapEntry)}
    assert maps["produce"] is maps["consume"]
    joined = maps["produce"]
    assert not any(n.data == "x" for n in joined.data_nodes())  # the element copies are in states of their own
    sdfg.build_folder = str(tmp_path / "build")
    a, b, c, x = np.arange(16, dtype=np.float64), np.zeros(16), np.zeros(1), np.zeros(2)
    sdfg(a=a, b=b, c=c, x=x, p=7)
    expected = np.zeros(16)
    expected[1:14] = 2 * a[2:15] + 1 - 1
    np.testing.assert_array_equal(b, expected)
    np.testing.assert_array_equal(x, a[2:4])  # the consumer's state's copy still comes last
    assert c[0] == 7
    assert fuse_stencil_pairs(_racing_pair(assigned)) == ["tmp (recompute)"]


def test_states_the_moves_do_not_join_get_their_components_back():
    """The consumer's state also overwrites the producer's input in a map of its own: no move helps, and every
    component is back in its state."""
    sdfg = _racing_pair()
    second = next(s for s in sdfg.states() if s.label == "consume_state")
    second.add_mapped_tasklet("overwrite", {"k": "0:16"}, {}, "y = 0", {"y": dace.Memlet("a[k]")},
                              external_edges=True)
    labels = sorted(s.label for s in sdfg.states())
    assert join_stencil_states(sdfg) == 0
    assert sorted(s.label for s in sdfg.states()) == labels
    for label, count in (("produce_state", 1), ("consume_state", 1)):
        state = next(s for s in sdfg.states() if s.label == label)
        assert sum(1 for n in state.data_nodes() if n.data == "x") == count
    assert sdfg.start_block.label == "produce_state"
    sdfg.validate()

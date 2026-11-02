"""An array of records with several record dimensions whose members differ in shape (one plane per face of a
domain, each thin along its own axis) and that the kernel reads in conditions is handed over member by member too,
each member with its own extents and lower bounds, instead of packed into a buffer spanning the largest of them."""

import ctypes
from pathlib import Path

import numpy as np

from dace_fortran.bindings.build_fortran_library import build_fortran_library
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.frozen_signature import get_frozen_signature, refreeze
from dace_fortran.bindings.record_split import split_record_companions
from dace_fortran.bindings.specialization import Specialization
from tests._util import build_sdfg

_SRC = """
module face_types
  implicit none
  type ifield
    integer, pointer, dimension(:, :) :: sf => null()
  end type ifield
end module face_types

module kern
  implicit none
contains
  subroutine edges(bc, out, m, n)
    use face_types, only: ifield
    type(ifield), dimension(2, 2), intent(in) :: bc
    integer, intent(in) :: m, n
    real(8), dimension(0:m, 0:n), intent(inout) :: out
    integer :: j, k
    do k = 0, n
      if (bc(1, 1)%sf(0, k) == -2) out(0, k) = out(0, k) + 1
      if (bc(1, 2)%sf(0, k) == -2) out(m, k) = out(m, k) + 2
    end do
    do j = 0, m
      if (bc(2, 1)%sf(j, 0) == -2) out(j, 0) = out(j, 0) + 4
      if (bc(2, 2)%sf(j, 0) == -2) out(j, n) = out(j, n) + 8
    end do
  end subroutine edges
end module kern
"""

# The faces along x are (0:0, 0:n), those along y (-1:m+1, 0:0): other shapes and other lower bounds. ``strided``:
# the first face is a section of a larger array instead (not contiguous).
_SHIM = """
subroutine run_edges(m, n, x1, x2, y1, y2, out, strided, declined_out) bind(c, name="run_edges")
  use iso_c_binding
  use face_types, only: ifield
  use edges_dace_bindings, only: edges_dace
  integer(c_int), value :: m, n, strided
  integer(c_int), intent(in) :: x1(0:n), x2(0:n), y1(-1:m + 1), y2(-1:m + 1)
  real(c_double), intent(inout) :: out(0:m, 0:n)
  integer(c_int), intent(out) :: declined_out
  type(ifield) :: bc(2, 2)
  integer, target, allocatable :: fx1(:, :), fx2(:, :), fy1(:, :), fy2(:, :), wide(:, :)
  logical :: declined
  allocate(fx1(0:0, 0:n), fx2(0:0, 0:n), fy1(-1:m + 1, 0:0), fy2(-1:m + 1, 0:0), wide(0:1, 0:n))
  fx1(0, :) = x1
  fx2(0, :) = x2
  fy1(:, 0) = y1
  fy2(:, 0) = y2
  wide(0, :) = x1
  bc(1, 1)%sf => fx1
  if (strided /= 0) bc(1, 1)%sf => wide(0:0, :)
  bc(1, 2)%sf => fx2
  bc(2, 1)%sf => fy1
  bc(2, 2)%sf => fy2
  call edges_dace(bc, out, m, n, dace_declined=declined)
  declined_out = merge(1, 0, declined)
end subroutine run_edges
"""


def _library(tmp_path: Path):
    src = tmp_path / "edges.f90"
    src.write_text(_SRC)
    builder = build_sdfg(_SRC, tmp_path / "sdfg", name="edges", entry="kern::edges")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "edges"
    sdfg.build_folder = str(tmp_path / "dacecache")
    general = get_frozen_signature(sdfg)
    split = split_record_companions(sdfg, plan, own_shapes=True)
    refreeze(sdfg)
    iface = build_auto_interface(sdfg._fortran_interface_raw, "edges")
    shim = tmp_path / "shim.f90"
    shim.write_text(_SHIM)
    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(tmp_path / "lib"),
        prelude_sources=[src],
        extra_sources=[shim],
        specialization=Specialization({}, general),
    )
    fn = lib.load().run_edges
    fn.restype = None
    fn.argtypes = [ctypes.c_int] * 2 + [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p]
    return split, fn, next((tmp_path / "lib").glob("*_bindings.f90")).read_text()


def _faces(m: int, n: int):
    rng = np.random.default_rng(7)
    x1, x2 = (rng.choice([-1, -2], n + 1).astype(np.int32) for _ in range(2))
    y1, y2 = (rng.choice([-1, -2], m + 3).astype(np.int32) for _ in range(2))
    return x1, x2, y1, y2


def _expected(m, n, x1, x2, y1, y2):
    out = np.zeros((m + 1, n + 1))
    out[0, :] += 1 * (x1 == -2)
    out[m, :] += 2 * (x2 == -2)
    out[:, 0] += 4 * (y1[1 : m + 2] == -2)  # y1 starts at -1
    out[:, n] += 8 * (y2[1 : m + 2] == -2)
    return out


def _run(fn, m, n, faces, strided=0):
    out = np.zeros((m + 1, n + 1), order="F")
    declined = np.zeros(1, dtype=np.int32)
    fn(m, n, *(f.ctypes.data for f in faces), out.ctypes.data, strided, declined.ctypes.data)
    return out, bool(declined[0])


def test_faces_of_their_own_shapes_are_aliased_in_place(tmp_path):
    split, fn, text = _library(tmp_path)
    assert split == {"bc_sf": {(0, 0): "bc_sf_0_0", (1, 0): "bc_sf_1_0", (0, 1): "bc_sf_0_1", (1, 1): "bc_sf_1_1"}}
    assert "allocate(bc_sf(" not in text  # no buffer spanning the largest face
    assert "bc_sf_1_0 => bc(i1, i2)%sf" in text
    assert "cap_bc_sf_1_0_d0 = size(bc(i1, i2)%sf, 1)" in text
    m, n = 5, 3
    faces = _faces(m, n)
    out, declined = _run(fn, m, n, faces)
    assert not declined
    np.testing.assert_array_equal(out, _expected(m, n, *faces))


def test_a_face_not_contiguous_declines_the_call(tmp_path):
    _split, fn, _text = _library(tmp_path)
    m, n = 4, 4
    out, declined = _run(fn, m, n, _faces(m, n), strided=1)
    assert declined
    np.testing.assert_array_equal(out, 0.0)

"""Round-trip tests for the Kokkos-View <-> nanobind-ndarray zero-copy bridge.

Exercises tpx::python::{ndarray_to_view, view_to_ndarray, vector_to_ndarray} through the
interop_test_module built next to this file. Validates the suite's array contract: x-fastest
(LayoutLeft) Kokkos fields export as Fortran-order (nx,ny,nz) NumPy arrays, values survive a
device round-trip, and exports reference (not copy) the View's memory.
"""
import numpy as np
import pytest

interop = pytest.importorskip("interop_test_module")


def test_flat_roundtrip():
    a = np.arange(8, dtype=np.float64)
    b = interop.double_it(a)
    assert b.dtype == np.float64
    assert b.shape == (8,)
    np.testing.assert_allclose(b, a * 2.0)


def test_field_is_fortran_xfastest():
    nx, ny, nz = 4, 3, 2
    f = interop.make_field(nx, ny, nz)
    assert f.shape == (nx, ny, nz)
    # x-fastest LayoutLeft -> Fortran-contiguous with element strides {1, nx, nx*ny}.
    assert f.flags["F_CONTIGUOUS"]
    assert [s // f.itemsize for s in f.strides] == [1, nx, nx * ny]
    expect = np.fromfunction(lambda x, y, z: x + y * nx + z * nx * ny, (nx, ny, nz))
    np.testing.assert_array_equal(f, expect)


def test_export_is_zero_copy_view():
    # A copy-free export must reference an owner object (the capsule holding the Kokkos View),
    # i.e. numpy reports a non-None base rather than owning the buffer itself.
    f = interop.make_field(2, 2, 2)
    assert f.base is not None


def test_vector_to_ndarray_strides():
    nx, ny, nz = 4, 3, 2
    v = interop.vec_field(nx, ny, nz)
    assert v.shape == (nx, ny, nz)
    assert [s // v.itemsize for s in v.strides] == [1, nx, nx * ny]
    np.testing.assert_array_equal(v.flatten(order="F"), np.arange(nx * ny * nz))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

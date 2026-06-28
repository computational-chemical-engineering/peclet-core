// Test module for the Kokkos-View <-> nanobind-ndarray zero-copy bridge
// (include/tpx/python/ndarray_interop.hpp). Built as a self-contained nanobind module exercised by
// test_ndarray_interop.py. Mirrors how every real binding (sdflow, dem, vorflow, tpx_amr) uses the
// bridge: import a Python array into a Kokkos View, run a device kernel, export the View back.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <Kokkos_Core.hpp>

#include "tpx/python/ndarray_interop.hpp"

namespace nb = nanobind;
using namespace tpx::python;

// Kernels live in their own (non-auto-returning) functions: nvcc forbids a KOKKOS_LAMBDA inside a
// function with a deduced return type, and the wrappers below return `auto` (the bridge's return type
// differs host vs device). Real bindings already keep kernels in solver headers, so this only matters
// for these inline test kernels.
static void scale2(tpx::View<double> v) {
  Kokkos::parallel_for("x2", v.extent(0), KOKKOS_LAMBDA(int i) { v(i) *= 2.0; });
  Kokkos::fence();
}
static void fillLinear(tpx::Field3D<double> f, int nx, int ny, int nz) {
  Kokkos::parallel_for(
      "fill", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx, ny, nz}),
      KOKKOS_LAMBDA(int x, int y, int z) { f(x, y, z) = double(x + y * nx + z * nx * ny); });
  Kokkos::fence();
}

// Import a contiguous array into a flat device View, scale on-device, export the View (zero-copy on
// the host backend; DLPack/CuPy on a device backend).
static auto double_it(nb::ndarray<double, nb::c_contig> a) {
  auto v = ndarray_to_view<double>(nb::ndarray<>(a), "double_it.tmp");
  scale2(v);
  return view_to_ndarray(v);
}

// Build an x-fastest LayoutLeft field of logical shape (nx,ny,nz), fill with the linear index
// I = x + y*nx + z*nx*ny, and export zero-copy. Verifies shape/stride/value round-trip.
static auto make_field(int nx, int ny, int nz) {
  tpx::Field3D<double> f("f", nx, ny, nz);
  fillLinear(f, nx, ny, nz);
  return view_to_ndarray(f);
}

// Move a ghost-stripped x-fastest host vector out as a Fortran-order (nx,ny,nz) NumPy array with no
// extra copy (the sdflow getter path).
static auto vec_field(int nx, int ny, int nz) {
  std::vector<double> v(static_cast<std::size_t>(nx) * ny * nz);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = double(i);
  return vector_to_ndarray(std::move(v), {(std::size_t)nx, (std::size_t)ny, (std::size_t)nz},
                           {1, (std::int64_t)nx, (std::int64_t)nx * ny});
}

NB_MODULE(interop_test_module, m) {
  if (!Kokkos::is_initialized()) Kokkos::initialize();
  nb::module_::import_("atexit").attr("register")(nb::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  }));
  m.attr("execution_space") = nb::str(Kokkos::DefaultExecutionSpace::name());
  m.def("double_it", &double_it);
  m.def("make_field", &make_field);
  m.def("vec_field", &vec_field);
}

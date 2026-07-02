// core — shared zero-copy bridge between Kokkos Views and Python arrays (nanobind).
//
// One place defines how every Kokkos-backed Python binding in the suite (flow, dem, voro,
// core's own tpx_amr) moves arrays across the C++/Python boundary, so the array
// contract — shapes, strides, dtype, host-vs-device — is identical everywhere instead of being
// re-hand-rolled (the old per-module `to_xyz`/`to_vec`/`upload` helpers).
//
// The win over the retired pybind11 path is that nanobind's `nb::ndarray` carries a DLPack device
// tag and arbitrary strides, so:
//   * a host View exports as a NumPy array that *references* the View's memory (no copy), and
//   * a device (CUDA/HIP) View exports as a DLPack-capable array that CuPy/PyTorch consume
//     zero-copy via `cupy.from_dlpack(...)` / `torch.from_dlpack(...)`.
// Lifetime is correct because the exported array owns a capsule holding a *copy* of the View, and
// Kokkos Views are reference-counted — the allocation lives exactly as long as Python references
// it.
//
// This header is only included by binding translation units (which link nanobind + Kokkos). It is
// NOT pulled into the device kernels. Layout note: the suite is x-fastest (LayoutLeft), so a
// `peclet::core::Field3D<T>` of logical shape (nx,ny,nz) exports with element strides {1, nx,
// nx*ny} — i.e. a Fortran-order NumPy array indexed [x,y,z], matching docs/CONVENTIONS.md §6.
#ifndef PECLET_CORE_PYTHON_NDARRAY_INTEROP_HPP
#define PECLET_CORE_PYTHON_NDARRAY_INTEROP_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "peclet/core/common/view.hpp"

namespace peclet::core::python {

namespace nb = nanobind;

/// True when `MemorySpace` is reachable from the host (NumPy export path); false for a
/// device-resident space (CUDA/HIP — DLPack/CuPy export path).
template <class MemorySpace>
inline constexpr bool is_host_space_v =
    Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible;

/// Map a Kokkos memory space to a DLPack {device_type, device_id} pair. Host → cpu; otherwise the
/// active GPU backend (CUDA, or ROCm under a HIP build), with the default device's id.
template <class MemorySpace>
inline std::pair<int, int> dlpack_device() {
  if constexpr (is_host_space_v<MemorySpace>) {
    return {nb::device::cpu::value, 0};
  } else {
#if defined(KOKKOS_ENABLE_HIP)
    return {nb::device::rocm::value, Kokkos::device_id()};
#else
    return {nb::device::cuda::value, Kokkos::device_id()};
#endif
  }
}

/// Throw if `a`'s element type is not `T`.
template <class T>
inline void require_dtype(const nb::ndarray<>& a, const char* who) {
  if (a.dtype() != nb::dtype<T>())
    throw std::runtime_error(std::string(who) + ": array dtype does not match the expected scalar");
}

/// Export a Kokkos View to a Python array **without copying**. Works for any rank and either a host
/// or device View. The returned array references `view`'s memory and keeps it alive via a capsule
/// owning a copy of the (ref-counted) View. Host views come back as `numpy.ndarray`; device views
/// come back as a DLPack array (consume with `cupy.from_dlpack`). Shape and element-unit strides
/// are taken from the View, so non-contiguous / padded views (e.g. a ghosted inner region) export
/// correctly.
template <class V>
auto view_to_ndarray(const V& view) {
  using T = std::remove_const_t<typename V::value_type>;
  using Mem = typename V::memory_space;
  constexpr std::size_t N = V::rank;

  std::array<std::size_t, N> shape;
  std::array<std::int64_t, N> strides;
  for (std::size_t i = 0; i < N; ++i) {
    shape[i] = view.extent(i);
    strides[i] = static_cast<std::int64_t>(view.stride(i));
  }

  // Capsule owns a heap copy of the View; Kokkos ref-counts the allocation, so the buffer lives as
  // long as the Python array (or any array derived from it) does.
  auto* held = new V(view);
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<V*>(p); });

  auto* data = const_cast<T*>(view.data());
  if constexpr (is_host_space_v<Mem>) {
    return nb::ndarray<nb::numpy, T>(data, N, shape.data(), owner, strides.data(), nb::dtype<T>(),
                                     nb::device::cpu::value, 0);
  } else {
    auto [dev, id] = dlpack_device<Mem>();
    return nb::ndarray<T>(data, N, shape.data(), owner, strides.data(), nb::dtype<T>(), dev, id);
  }
}

/// Export a host `std::vector<T>` as a NumPy array of the given logical `shape` and element-unit
/// `strides`, **without the extra copy** the old `to_xyz` made: the vector is moved into a capsule
/// that backs the array. Use for solver getters that already return a host vector (e.g. a
/// ghost-stripped x-fastest field → shape {nx,ny,nz}, strides {1,nx,nx*ny}).
template <class T>
nb::ndarray<nb::numpy, T> vector_to_ndarray(std::vector<T>&& v,
                                            std::initializer_list<std::size_t> shape,
                                            std::initializer_list<std::int64_t> strides) {
  auto* held = new std::vector<T>(std::move(v));
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T>(held->data(), shape, owner, strides, nb::dtype<T>(),
                                   nb::device::cpu::value, 0);
}

/// Import a (contiguous) Python array into a freshly-allocated flat device `peclet::core::View<T>`.
/// A host array is staged up (deep_copy from an unmanaged host wrap == the old
/// `peclet::core::toDevice`); a device array on this build's backend is wrapped unmanaged and
/// copied **on-device** (no host bounce — the CuPy zero-copy-onto-GPU path). An array on an
/// incompatible device throws. The caller must pass a contiguous array (bindings enforce this via a
/// typed `nb::ndarray<T, nb::c_contig>` / `nb::f_contig` parameter); the data is read as a flat
/// buffer of `a.size()` elements.
template <class T>
peclet::core::View<T> ndarray_to_view(const nb::ndarray<>& a, const std::string& label) {
  require_dtype<T>(a, "ndarray_to_view");
  const std::size_t n = a.size();
  peclet::core::View<T> d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), n);
  const int dev = a.device_type();
  if (dev == nb::device::cpu::value) {
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(
        static_cast<const T*>(a.data()), n);
    Kokkos::deep_copy(d, hv);
  } else if constexpr (!is_host_space_v<peclet::core::MemSpace>) {
    if (dev != dlpack_device<peclet::core::MemSpace>().first)
      throw std::runtime_error(
          "ndarray_to_view: array is on an incompatible device for this build");
    Kokkos::View<const T*, peclet::core::MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv(
        static_cast<const T*>(a.data()), n);
    Kokkos::deep_copy(d, dv);
  } else {
    throw std::runtime_error("ndarray_to_view: array is on an incompatible device for this build");
  }
  return d;
}

/// Import a (contiguous) Python array into a host `std::vector<T>`. A host array is memcpy'd; a
/// device array is copied down to the host (for solver APIs whose setters take host vectors). The
/// caller enforces contiguity / element order via a typed binding parameter; the array is read as a
/// flat buffer of `a.size()` elements.
template <class T>
std::vector<T> ndarray_to_vector(const nb::ndarray<>& a) {
  require_dtype<T>(a, "ndarray_to_vector");
  const std::size_t n = a.size();
  std::vector<T> out(n);
  const int dev = a.device_type();
  if (dev == nb::device::cpu::value) {
    if (n)
      std::memcpy(out.data(), a.data(), n * sizeof(T));
  } else if constexpr (!is_host_space_v<peclet::core::MemSpace>) {
    if (dev != dlpack_device<peclet::core::MemSpace>().first)
      throw std::runtime_error(
          "ndarray_to_vector: array is on an incompatible device for this build");
    Kokkos::View<const T*, peclet::core::MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv(
        static_cast<const T*>(a.data()), n);
    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(out.data(), n);
    Kokkos::deep_copy(hv, dv);
  } else {
    throw std::runtime_error(
        "ndarray_to_vector: array is on an incompatible device for this build");
  }
  return out;
}

}  // namespace peclet::core::python

#endif  // PECLET_CORE_PYTHON_NDARRAY_INTEROP_HPP

// core — shared Kokkos device-memory abstraction for the suite's portable path.
//
// One place defines the execution/memory spaces, the 1D device array the halo exchange operates on
// (it gathers/scatters by flat x-fastest local index, so a contiguous per-cell field is a View<T>),
// and the 3D structured field (x-fastest => Kokkos::LayoutLeft). Including this header requires
// Kokkos (build with -DPECLET_CORE_ENABLE_KOKKOS=ON); the CPU and legacy-CUDA paths do NOT pull it
// in.
#ifndef PECLET_CORE_COMMON_VIEW_HPP
#define PECLET_CORE_COMMON_VIEW_HPP

#include <Kokkos_Core.hpp>
#include <string>
#include <utility>
#include <vector>

#include "peclet/core/common/types.hpp"

namespace peclet::core {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;

/// 1D device array. The halo addresses cells by precomputed flat local index, so a contiguous
/// per-cell field is a View<T>; the x-fastest ordering lives in the indices, not the View layout.
template <class T>
using View = Kokkos::View<T*, MemSpace>;

/// Host-accessible mirror of View<T> (identical to View<T> on host backends). Defined via the
/// actual create_mirror_view return type to stay correct across Kokkos backends/versions.
template <class T>
using HostView = decltype(Kokkos::create_mirror_view(std::declval<View<T>>()));

/// Device array of grid/particle indices (the matched send/recv/self-copy lists).
using IndexView = Kokkos::View<Index*, MemSpace>;

/// 3D structured field in the suite's x-fastest convention: LayoutLeft makes the leftmost (x) index
/// contiguous, i.e. I = x + y*nx + z*nx*ny — identical to peclet::core::Index linearization and
/// cfd-gpu grids.
template <class T>
using Field3D = Kokkos::View<T***, Kokkos::LayoutLeft, MemSpace>;

/// Upload a host std::vector into a freshly-sized device View (empty vector => empty view).
template <class T>
inline View<T> toDevice(const std::vector<T>& h, const std::string& label) {
  View<T> d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), h.size());
  if (!h.empty()) {
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(h.data(),
                                                                                          h.size());
    Kokkos::deep_copy(d, hv);
  }
  return d;
}

/// Copy a (host- or device-resident) Kokkos View of any rank into a contiguous host std::vector,
/// flattened in C/row order (last index fastest) — an Nx3 View comes back as `out[i*3 + c] ==
/// view(i,c)`, the suite's particle-SoA Python export order. This replaces the redundant
/// "create_mirror_view → deep_copy → element-by-element loop → std::vector" idiom in the binding
/// getters (S2a) with a single device→host transfer and no hand-written loop. The row-order flatten
/// is layout-correct on every backend: a LayoutLeft (e.g. CUDA-default) device View is transposed
/// on the host hop rather than blindly memcpy'd, so the exported ordering matches across CPU/GPU
/// builds.
template <class V>
std::vector<std::remove_const_t<typename V::value_type>> toVector(const V& view) {
  using T = std::remove_const_t<typename V::value_type>;
  std::vector<T> out(view.size());
  if (view.size()) {
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    Kokkos::LayoutRight layout;
    for (std::size_t i = 0; i < V::rank; ++i)
      layout.dimension[i] = view.extent(i);
    Kokkos::View<typename V::non_const_data_type, Kokkos::LayoutRight, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        dst(out.data(), layout);
    Kokkos::deep_copy(dst, host);
  }
  return out;
}

}  // namespace peclet::core

#endif  // PECLET_CORE_COMMON_VIEW_HPP

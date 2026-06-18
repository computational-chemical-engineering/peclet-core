// transport-core — shared Kokkos device-memory abstraction for the suite's portable path.
//
// One place defines the execution/memory spaces, the 1D device array the halo exchange operates on
// (it gathers/scatters by flat x-fastest local index, so a contiguous per-cell field is a View<T>),
// and the 3D structured field (x-fastest => Kokkos::LayoutLeft). Including this header requires
// Kokkos (build with -DTPX_ENABLE_KOKKOS=ON); the CPU and legacy-CUDA paths do NOT pull it in.
#ifndef TPX_COMMON_VIEW_HPP
#define TPX_COMMON_VIEW_HPP

#include <Kokkos_Core.hpp>

#include <string>
#include <utility>
#include <vector>

#include "tpx/common/types.hpp"

namespace tpx {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;

/// 1D device array. The halo addresses cells by precomputed flat local index, so a contiguous
/// per-cell field is a View<T>; the x-fastest ordering lives in the indices, not the View layout.
template <class T>
using View = Kokkos::View<T*, MemSpace>;

/// Host-accessible mirror of View<T> (identical to View<T> on host backends). Defined via the actual
/// create_mirror_view return type to stay correct across Kokkos backends/versions.
template <class T>
using HostView = decltype(Kokkos::create_mirror_view(std::declval<View<T>>()));

/// Device array of grid/particle indices (the matched send/recv/self-copy lists).
using IndexView = Kokkos::View<Index*, MemSpace>;

/// 3D structured field in the suite's x-fastest convention: LayoutLeft makes the leftmost (x) index
/// contiguous, i.e. I = x + y*nx + z*nx*ny — identical to tpx::Index linearization and cfd-gpu grids.
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

}  // namespace tpx

#endif  // TPX_COMMON_VIEW_HPP

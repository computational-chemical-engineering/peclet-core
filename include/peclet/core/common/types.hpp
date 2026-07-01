// core — shared types and conventions (see suite/docs/CONVENTIONS.md).
//
// Pillars: axis order is x-fastest (linear index I = x + y*nx + z*nx*ny), and SDF sign is
// negative-inside-solid (enforced in peclet::core::geom, not here). This header is C++17-clean so it can be
// pulled into CUDA translation units; C++20-only constructs live in host-only headers.
#ifndef PECLET_CORE_COMMON_TYPES_HPP
#define PECLET_CORE_COMMON_TYPES_HPP

#include <array>
#include <cstdint>

namespace peclet::core {

/// Signed index type for grids and particles (supersedes block_decomposer's `long int IndxT`).
using Index = std::int64_t;

/// Default host floating type. Device kernels may use float; conversions happen at the boundary.
using Real = double;

/// Multi-dimensional integer index / coordinate (x-fastest order by convention).
template <int Dim>
using IVec = std::array<Index, Dim>;

/// Multi-dimensional real vector.
template <int Dim>
using Vec = std::array<Real, Dim>;

/// Half-open range [begin, end) carrying a label (e.g. a block id). Ported from pbs::IndxRange.
template <typename Label = Index>
struct Range {
  std::size_t begin = 0;
  std::size_t end = 0;
  Label label{};
};

/// Periodic wrap of a coordinate into [0, n): wrap(x, n) = (x % n + n) % n.
inline Index wrap(Index x, Index n) { return ((x % n) + n) % n; }

/// Compile-time-unrolled nested loop over [bgn, end) in row-major (axis 0 fastest) order.
/// func receives the current IVec<Dim> index by const-ref. Ported from pbs::NestedLoop.
template <int Dim, int Axis>
struct NestedLoop {
  template <typename Func>
  static void run(IVec<Dim>& idx, const IVec<Dim>& bgn, const IVec<Dim>& end, Func&& func) {
    for (idx[Axis] = bgn[Axis]; idx[Axis] < end[Axis]; ++idx[Axis]) {
      NestedLoop<Dim, Axis - 1>::run(idx, bgn, end, func);
    }
  }
};

template <int Dim>
struct NestedLoop<Dim, 0> {
  template <typename Func>
  static void run(IVec<Dim>& idx, const IVec<Dim>& bgn, const IVec<Dim>& end, Func&& func) {
    for (idx[0] = bgn[0]; idx[0] < end[0]; ++idx[0]) {
      func(const_cast<const IVec<Dim>&>(idx));
    }
  }
};

/// Iterate the box [bgn, end) calling func(const IVec<Dim>&).
template <int Dim, typename Func>
inline void forEachInBox(const IVec<Dim>& bgn, const IVec<Dim>& end, Func&& func) {
  IVec<Dim> idx{};
  NestedLoop<Dim, Dim - 1>::run(idx, bgn, end, func);
}

}  // namespace peclet::core

#endif  // PECLET_CORE_COMMON_TYPES_HPP

// core — host/device portability shim for the shared geometry layer.
//
// Two things live here, both required by peclet/core/geom/primitives.hpp:
//
//   PECLET_HD   — the host+device function annotation. Three branches, copied from morton's
//                 MORTON_HD (morton/include/morton/hd.hpp): Kokkos when Kokkos is in the
//                 translation unit, raw __host__ __device__ under nvcc/hipcc, nothing otherwise.
//                 The nvcc/hipcc branch is what makes the include order forgiving: a Kokkos+CUDA
//                 build compiles through nvcc, so even a TU that reaches this header before
//                 Kokkos_Core.hpp still gets a device-callable annotation. Wrapped in
//                 `#if !defined(PECLET_HD)` so a build can override it.
//
//   Vec3<Real>  — a Real-templated POD 3-vector. Deliberately NOT peclet::core::Vec<3>: that is
//                 std::array<Real, 3> with `Real` a fixed double typedef (common/types.hpp), so it
//                 is neither Real-templated nor a natural device type. Vec3 is what dem's F3
//                 becomes and what voro's loose `Real x, y, z` parameters collapse into.
//
// C++17-clean (no concepts, no C++20 constructs) so it can be pulled into CUDA translation units,
// matching the pledge in common/types.hpp. Design: suite/docs/ANALYTIC_SDF_GEOMETRY.md.
#ifndef PECLET_CORE_COMMON_PORTABLE_HPP
#define PECLET_CORE_COMMON_PORTABLE_HPP

#if !defined(PECLET_HD)
#if defined(KOKKOS_VERSION)
// KOKKOS_INLINE_FUNCTION (not KOKKOS_FUNCTION as in morton): these are header-only templates that
// must inline.
#define PECLET_HD KOKKOS_INLINE_FUNCTION
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define PECLET_HD __host__ __device__ inline
#else
#define PECLET_HD inline
#endif
#endif  // !defined(PECLET_HD)

#include <cmath>

namespace peclet::core {

/// Real-templated POD 3-vector — the geometry layer's point type on host and device.
template <class Real>
struct Vec3 {
  Real x, y, z;
};

namespace detail {

// Math shims. When Kokkos is present the geometry MUST route through Kokkos::* so the formulas are
// bit-identical to dem's originals (dem_portable.hpp uses Kokkos::sqrt/fabs/fmax/fmin throughout);
// on a host-only core build there is no Kokkos, so std::* stands in. Both map to the same libm
// call on host, which is what keeps the bit-exactness gate meaningful in either build.
#if defined(KOKKOS_VERSION)
template <class Real>
PECLET_HD Real hdSqrt(Real v) {
  return Kokkos::sqrt(v);
}
template <class Real>
PECLET_HD Real hdAbs(Real v) {
  return Kokkos::fabs(v);
}
template <class Real>
PECLET_HD Real hdMax(Real a, Real b) {
  return Kokkos::fmax(a, b);
}
template <class Real>
PECLET_HD Real hdMin(Real a, Real b) {
  return Kokkos::fmin(a, b);
}
#else
template <class Real>
PECLET_HD Real hdSqrt(Real v) {
  return std::sqrt(v);
}
template <class Real>
PECLET_HD Real hdAbs(Real v) {
  return std::fabs(v);
}
template <class Real>
PECLET_HD Real hdMax(Real a, Real b) {
  return std::fmax(a, b);
}
template <class Real>
PECLET_HD Real hdMin(Real a, Real b) {
  return std::fmin(a, b);
}
#endif

/// Comparison-based max, `<`-ordered (ties keep the earlier operand) — std::max semantics, NOT
/// fmax's NaN-swallowing ones. HollowCylinderShell needs exactly this to stay bit-faithful to
/// core's `std::max({...})` and voro's chained ternaries; everything ported from dem uses hdMax.
template <class Real>
PECLET_HD Real cmpMax(Real a, Real b) {
  return (a < b) ? b : a;
}

}  // namespace detail
}  // namespace peclet::core

#endif  // PECLET_CORE_COMMON_PORTABLE_HPP

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
//   Vec3<Real>  — a Real-templated POD 3-vector, plus the small vector/quaternion algebra the
//                 scene transform stack needs. Deliberately NOT peclet::core::Vec<3>, which is
//                 std::array<Real, 3> with `Real` a fixed double typedef (common/types.hpp):
//                 neither Real-templated nor a natural device type. Vec3 is what dem's F3 becomes
//                 and what voro's loose `Real x, y, z` parameters collapse into. Bundling the
//                 algebra here mirrors how dem organises the same material (F3 ships alongside
//                 add3/sub3/scale3/dot3/len3/cross3v/rotateVector in dem_portable.hpp), so the
//                 rung-3 port is a rename rather than a rewrite.
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

/// Quaternion, {x, y, z, w} — dem's F4 ordering (dem_portable.hpp), NOT {w, x, y, z}.
template <class Real>
struct Quat {
  Real x, y, z, w;
};

// --- vector algebra (association order transcribed from dem's math_utils heritage) -------------

template <class Real>
PECLET_HD Vec3<Real> add(Vec3<Real> a, Vec3<Real> b) {
  return Vec3<Real>{a.x + b.x, a.y + b.y, a.z + b.z};
}
template <class Real>
PECLET_HD Vec3<Real> sub(Vec3<Real> a, Vec3<Real> b) {
  return Vec3<Real>{a.x - b.x, a.y - b.y, a.z - b.z};
}
template <class Real>
PECLET_HD Vec3<Real> scale(Vec3<Real> a, Real s) {
  return Vec3<Real>{a.x * s, a.y * s, a.z * s};
}
template <class Real>
PECLET_HD Real dot(Vec3<Real> a, Vec3<Real> b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <class Real>
PECLET_HD Vec3<Real> cross(Vec3<Real> a, Vec3<Real> b) {
  return Vec3<Real>{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// Rotate v by q. Verbatim port of dem `rotateVector` (dem_portable.hpp) — keep the operation
/// order: a rewritten-but-algebraically-equal form would break the rung-3 bit-exactness gate.
template <class Real>
PECLET_HD Vec3<Real> rotate(Quat<Real> q, Vec3<Real> v) {
  const Vec3<Real> qv{q.x, q.y, q.z};
  const Vec3<Real> t = scale(cross(qv, v), Real(2));
  return add(add(v, scale(t, q.w)), cross(qv, t));
}

/// Rotate v by the conjugate of q — dem `invRotateVector`.
template <class Real>
PECLET_HD Vec3<Real> invRotate(Quat<Real> q, Vec3<Real> v) {
  return rotate(Quat<Real>{-q.x, -q.y, -q.z, q.w}, v);
}

/// Hamilton product a ⊗ b in the (x, y, z, w) storage convention `rotate` uses, so
/// rotate(mulQuat(a, b), v) == rotate(a, rotate(b, v)) — i.e. b applies first.
template <class Real>
PECLET_HD Quat<Real> mulQuat(Quat<Real> a, Quat<Real> b) {
  return Quat<Real>{a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

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
template <class Real>
PECLET_HD Real hdPow(Real a, Real b) {
  return Kokkos::pow(a, b);
}
template <class Real>
PECLET_HD Real hdNearbyint(Real v) {
  return Kokkos::nearbyint(v);
}
template <class Real>
PECLET_HD Real hdFloor(Real v) {
  return Kokkos::floor(v);
}
template <class Real>
PECLET_HD Real hdFma(Real a, Real b, Real c) {
  return Kokkos::fma(a, b, c);
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
template <class Real>
PECLET_HD Real hdPow(Real a, Real b) {
  return std::pow(a, b);
}
template <class Real>
PECLET_HD Real hdNearbyint(Real v) {
  return std::nearbyint(v);
}
template <class Real>
PECLET_HD Real hdFloor(Real v) {
  return std::floor(v);
}
template <class Real>
PECLET_HD Real hdFma(Real a, Real b, Real c) {
  return std::fma(a, b, c);
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

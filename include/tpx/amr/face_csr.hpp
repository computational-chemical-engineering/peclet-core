// transport-core — shared, backend-agnostic kernels for an assembled face-CSR operator.
//
// The "host" serial reference solver (tpx::amr::AmrCutCell / AmrFlow, pure C++20) and the "device"
// Kokkos solver (tpx::amr::DeviceMomentumOp + device_momentum.hpp) used to carry two independent
// encodings of the SAME per-cell arithmetic — a real drift risk. This header is the single source of
// that arithmetic: the assembled operator
//
//     (A u)_i = diag_i · u_i + Σ_{k ∈ [start_i, start_{i+1})} coef_k · u[nbr_k]
//                              ( + optional implicit-FOU advection over a second CSR )
//
// expressed once as MORTON_HD row kernels over a *templated accessor*, so the SAME body serves
//   - the host: a plain serial loop, arrays = raw pointers (no Kokkos dependency — MORTON_HD is empty
//     in a non-Kokkos, non-CUDA build, so these compile as ordinary inline functions); and
//   - the device: inside Kokkos::parallel_for, arrays = Kokkos::View.
//
// Mesh-agnostic: it names no octree types and assumes nothing beyond the CSR, so it is equally the
// kernel layer for a future Voronoi-cell (unstructured polyhedral) operator.
#ifndef TPX_AMR_FACE_CSR_HPP
#define TPX_AMR_FACE_CSR_HPP

#include "tpx/common/types.hpp"

// MORTON_HD: KOKKOS_FUNCTION under a Kokkos build, __host__ __device__ under nvcc, empty otherwise —
// so the row kernels are device-callable when compiled for the device and ordinary host functions in
// the pure-C++ build. morton.hpp defines it for all three cases; when the (optional) morton checkout
// is absent the operator is host-only, so an empty fallback is exactly right.
#if defined(TPX_HAVE_MORTON)
#include <morton/morton.hpp>
#endif
#ifndef MORTON_HD
#define MORTON_HD
#endif

namespace tpx::amr {

/// A uniform accessor over a raw host array, giving it the `operator()(i)` that Kokkos::View has, so
/// the row kernels can be written once for both. (Device passes a `View<const T>` directly.)
template <class T>
struct HostArr {
  const T* p = nullptr;
  HostArr() = default;
  explicit HostArr(const T* ptr) : p(ptr) {}
  MORTON_HD T operator()(Index i) const { return p[i]; }
};

/// A backend-agnostic view of an assembled face-CSR operator. `D` is the floating array accessor
/// (HostArr<double> or Kokkos::View<const double>), `I` the index accessor. `hasAdv` toggles an
/// optional second (advection) CSR with a per-cell outflow diagonal `advDiag` + inflow `advCoef`.
template <class D, class I>
struct FaceCsrOpT {
  Index n = 0;
  D diag, coef;
  I start, nbr;
  bool hasAdv = false;
  D advDiag, advCoef;
  I advStart, advNbr;
};

/// (A u)_i — one assembled-operator row.
template <class Op, class U>
MORTON_HD inline double faceCsrApplyRow(const Op& op, Index i, const U& u) {
  double acc = op.diag(i) * u(i);
  for (Index k = op.start(i); k < op.start(i + 1); ++k) acc += op.coef(k) * u(op.nbr(k));
  if (op.hasAdv) {
    acc += op.advDiag(i) * u(i);
    for (Index k = op.advStart(i); k < op.advStart(i + 1); ++k) acc += op.advCoef(k) * u(op.advNbr(k));
  }
  return acc;
}

/// Off-diagonal sum and the (advection-inclusive) diagonal for the point smoothers: out `off` =
/// Σ coef·u[nbr] (+ advection inflow), `d` = diag (+ advection outflow). The point update of
/// A u = b is then u_i ← (1−ω)u_i + ω·(b_i − off)/d.
template <class Op, class U>
MORTON_HD inline void faceCsrOffDiag(const Op& op, Index i, const U& u, double& off, double& d) {
  off = 0.0;
  d = op.diag(i);
  for (Index k = op.start(i); k < op.start(i + 1); ++k) off += op.coef(k) * u(op.nbr(k));
  if (op.hasAdv) {
    for (Index k = op.advStart(i); k < op.advStart(i + 1); ++k) off += op.advCoef(k) * u(op.advNbr(k));
    d += op.advDiag(i);
  }
}

/// The damped point update used by both Jacobi and (multicolour) Gauss–Seidel: returns the new u_i
/// given the right-hand side `b_i`, the off-diagonal sum `off`, the diagonal `d`, the old value
/// `uOld`, and the relaxation `omega`. A zero diagonal (an inactive/identity row) leaves u_i fixed.
MORTON_HD inline double faceCsrPointUpdate(double b_i, double off, double d, double uOld,
                                           double omega) {
  const double nu = (d != 0.0) ? (b_i - off) / d : uOld;
  return (1.0 - omega) * uOld + omega * nu;
}

}  // namespace tpx::amr

#endif  // TPX_AMR_FACE_CSR_HPP

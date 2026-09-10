// core — the small device (Kokkos) vector primitives the CSR solvers are written in.
//
// Lifted verbatim from the AMR tree (peclet/core/amr/momentum.hpp + pcg.hpp, 2026-09-10;
// QUALITY_PLAN G.2). Plain (unweighted) dot, BiCGStab / CG direction updates, axpy, negate: each is
// one parallel_for / parallel_reduce over a peclet::core::View<double>. The dot is a device
// reduction, so its summation order — and therefore the last bits of a Krylov iterate — depends on
// the backend and thread count; the elementwise updates are bit-exact.
#ifndef PECLET_CORE_SOLVER_VECTOR_OPS_HPP
#define PECLET_CORE_SOLVER_VECTOR_OPS_HPP

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::core::solver {

/// Plain (unweighted) dot product.
inline double dotPlain(View<const double> a, View<const double> b, Index n) {
  double s = 0.0;
  Kokkos::parallel_reduce(
      "amr::mom_dot", n, KOKKOS_LAMBDA(const Index i, double& acc) { acc += a(i) * b(i); }, s);
  return s;
}

/// BiCGStab direction update: p = r + β(p − ω v). (Free function — an extended
/// __host__ __device__ lambda may not live in a private/protected member function.)
inline void bicgPUpdate(View<double> p, View<const double> r, View<const double> v, double beta,
                        double omega, Index n) {
  Kokkos::parallel_for(
      "amr::mom_pupdate", n,
      KOKKOS_LAMBDA(const Index i) { p(i) = r(i) + beta * (p(i) - omega * v(i)); });
}

/// y += a·x
inline void axpy(View<double> y, double a, View<const double> x, Index n) {
  Kokkos::parallel_for("amr::pcg_axpy", n, KOKKOS_LAMBDA(const Index i) { y(i) += a * x(i); });
}

/// p = z + b·p  (CG direction update)
inline void zpby(View<double> p, View<const double> z, double b, Index n) {
  Kokkos::parallel_for(
      "amr::pcg_zpby", n, KOKKOS_LAMBDA(const Index i) { p(i) = z(i) + b * p(i); });
}

/// y = −x  (negate in place)
inline void negate(View<double> x, Index n) {
  Kokkos::parallel_for("amr::pcg_negate", n, KOKKOS_LAMBDA(const Index i) { x(i) = -x(i); });
}

}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_VECTOR_OPS_HPP

// transport-core — device (Kokkos) momentum operator + smoother/solver for the AMR
// collocated flow step.
//
// The cut-cell momentum operator A = (ρ/dt)I − μ∇² (+ implicit-FOU advection + the
// ξ-polynomial Dirichlet overlay on cut cells) assembled by AmrCutCell::assembleOperator
// as a per-cell diagonal + general face CSR, uploaded once and applied / solved entirely
// in device kernels. This replaces the host-serial AmrCutCell::gaussSeidel that the drag
// study found to be the bottleneck (the pressure Poisson was already on the device path).
//
// A is generally NON-symmetric (the cut-cell D_rescale row scaling and the upwind
// advection break symmetry), so the smoother is weighted Jacobi (parallel, deterministic:
// reads only the previous iterate) and the Krylov accelerator is BiCGStab (handles
// non-symmetric A) rather than CG. For moderate dt the reaction diagonal (ρ/dt) makes A
// strongly diagonally dominant ⇒ Jacobi alone converges fast; BiCGStab covers the
// large-dt / steady regime where the operator approaches the (ill-conditioned) viscous
// Laplacian.
//
// Validation is by convergence + agreement with the host operator to tolerance (the GPU
// matvec differs from host in the last bit by FMA contraction). Requires a Kokkos build +
// the morton checkout (TPX_HAVE_MORTON).
#ifndef TPX_AMR_DEVICE_MOMENTUM_HPP
#define TPX_AMR_DEVICE_MOMENTUM_HPP

#ifdef TPX_HAVE_MORTON

#include <cmath>

#include "tpx/amr/device_multigrid.hpp"  // optional Helmholtz V-cycle preconditioner
#include "tpx/amr/device_pcg.hpp"        // dotPlain-style primitives: axpy, zpby, negate
#include "tpx/common/view.hpp"

namespace tpx::amr {

/// Assembled momentum operator on the device: (A u)_i = diag_i u_i + Σ coef·u[nbr].
struct DeviceMomentumOp {
  View<double> diag;      ///< size n
  View<Index> faceStart;  ///< CSR row offsets, size n+1
  View<Index> faceNbr;    ///< neighbour leaf per off-diagonal, size nnz
  View<double> faceCoef;  ///< off-diagonal coefficient, size nnz
  Index n = 0;
};

/// Au = A u.
inline void deviceApplyMom(const DeviceMomentumOp& op, View<const double> u, View<double> Au) {
  auto diag = op.diag;
  auto fs = op.faceStart;
  auto fn = op.faceNbr;
  auto fc = op.faceCoef;
  Kokkos::parallel_for(
      "amr::mom_apply", op.n, KOKKOS_LAMBDA(const Index i) {
        double acc = diag(i) * u(i);
        for (Index k = fs(i); k < fs(i + 1); ++k) acc += fc(k) * u(fn(k));
        Au(i) = acc;
      });
}

/// res = b − A u.
inline void deviceResidualMom(const DeviceMomentumOp& op, View<const double> u,
                              View<const double> b, View<double> res) {
  auto diag = op.diag;
  auto fs = op.faceStart;
  auto fn = op.faceNbr;
  auto fc = op.faceCoef;
  Kokkos::parallel_for(
      "amr::mom_residual", op.n, KOKKOS_LAMBDA(const Index i) {
        double acc = diag(i) * u(i);
        for (Index k = fs(i); k < fs(i + 1); ++k) acc += fc(k) * u(fn(k));
        res(i) = b(i) - acc;
      });
}

/// One weighted-Jacobi sweep of A u = b (in place). `tmp` is scratch (size n). Pass 1
/// reads only the previous iterate, pass 2 updates ⇒ order-independent / deterministic.
inline void deviceJacobiMom(const DeviceMomentumOp& op, View<double> u, View<const double> b,
                            View<double> tmp, double omega) {
  auto diag = op.diag;
  auto fs = op.faceStart;
  auto fn = op.faceNbr;
  auto fc = op.faceCoef;
  Kokkos::parallel_for(
      "amr::mom_jacobi_compute", op.n, KOKKOS_LAMBDA(const Index i) {
        double off = 0.0;
        for (Index k = fs(i); k < fs(i + 1); ++k) off += fc(k) * u(fn(k));
        const double d = diag(i);
        tmp(i) = (d != 0.0) ? (b(i) - off) / d : u(i);
      });
  Kokkos::parallel_for(
      "amr::mom_jacobi_update", op.n,
      KOKKOS_LAMBDA(const Index i) { u(i) = (1.0 - omega) * u(i) + omega * tmp(i); });
}

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

// ---------------------------------------------------------------------------
// Jacobi-preconditioned BiCGStab for the (non-symmetric) momentum operator. Reuses the
// device matvec + Kokkos reductions; the preconditioner is `jacPre` damped-Jacobi sweeps
// of A (diagonal-dominant ⇒ a cheap, effective smoother-preconditioner). Robust where
// plain Jacobi stalls (large dt / weak reaction term).
// ---------------------------------------------------------------------------
template <unsigned Bits = 21u>
class DeviceMomentumSolver {
 public:
  void setJacobi(int preSweeps, double omega) {
    jacPre_ = preSweeps;
    omega_ = omega;
  }

  /// Use a Helmholtz DeviceMultigrid (built with setHelmholtz(idiag, −μ) on the same mesh)
  /// as the BiCGStab preconditioner instead of damped-Jacobi sweeps. The V-cycle gives the
  /// multigrid smooth-mode coverage Jacobi lacks, so the momentum iteration count stops
  /// growing with N. `vcycles` V-cycles per preconditioner application. Pass nullptr to
  /// revert to the Jacobi preconditioner. The preconditioner never changes the converged
  /// solution (BiCGStab matvec is the exact operator) — only the iteration count.
  void setMgPreconditioner(DeviceMultigrid<3, Bits>* mg, int vcycles = 1) {
    mgPre_ = mg;
    mgVcycles_ = vcycles;
  }

  /// Plain weighted-Jacobi solve (the simple parallel mirror of the host GS smoother):
  /// `sweeps` damped-Jacobi sweeps of A u = b in place. Returns the final residual L2.
  double solveJacobi(const DeviceMomentumOp& op, View<double> u, View<const double> b,
                     int sweeps) {
    ensure(op.n);
    for (int s = 0; s < sweeps; ++s) deviceJacobiMom(op, u, b, tmp_, omega_);
    deviceResidualMom(op, View<const double>(u), b, r_);
    return std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), op.n));
  }

  /// Jacobi-preconditioned BiCGStab solve of A u = b in place. `maxIters` caps the outer
  /// iterations; `tol` is relative to ||b−Au0||. Returns {iters, final residual L2}.
  struct Result {
    int iters = 0;
    double res0 = 0.0;
    double res = 0.0;
  };
  Result solveBiCGStab(const DeviceMomentumOp& op, View<double> u, View<const double> b,
                       int maxIters = 500, double tol = 1e-10) {
    const Index n = op.n;
    ensure(n);
    Result R;
    // r = b − A u
    deviceResidualMom(op, View<const double>(u), b, r_);
    Kokkos::deep_copy(rhat_, r_);  // shadow residual
    R.res0 = std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), n));
    if (R.res0 == 0.0) return R;
    double rho = 1, alpha = 1, omega = 1;
    Kokkos::deep_copy(v_, 0.0);
    Kokkos::deep_copy(p_, 0.0);
    double rnorm = R.res0;
    int it = 0;
    for (; it < maxIters; ++it) {
      double rhoNew = dotPlain(View<const double>(rhat_), View<const double>(r_), n);
      if (rhoNew == 0.0) break;
      double beta = (rhoNew / rho) * (alpha / omega);
      // p = r + beta (p − omega v)
      bicgPUpdate(p_, View<const double>(r_), View<const double>(v_), beta, omega, n);
      applyPrec(op, p_, phat_);  // phat = M^{-1} p
      deviceApplyMom(op, View<const double>(phat_), v_);
      double rhatV = dotPlain(View<const double>(rhat_), View<const double>(v_), n);
      alpha = rhoNew / rhatV;
      // s = r − alpha v
      Kokkos::deep_copy(s_, r_);
      axpy(s_, -alpha, View<const double>(v_), n);
      double snorm = std::sqrt(dotPlain(View<const double>(s_), View<const double>(s_), n));
      if (snorm <= tol * R.res0) {
        axpy(u, alpha, View<const double>(phat_), n);  // u += alpha phat
        rnorm = snorm;
        ++it;
        break;
      }
      applyPrec(op, s_, shat_);  // shat = M^{-1} s
      deviceApplyMom(op, View<const double>(shat_), t_);
      double tt = dotPlain(View<const double>(t_), View<const double>(t_), n);
      omega = (tt != 0.0) ? dotPlain(View<const double>(t_), View<const double>(s_), n) / tt : 0.0;
      // u += alpha phat + omega shat
      axpy(u, alpha, View<const double>(phat_), n);
      axpy(u, omega, View<const double>(shat_), n);
      // r = s − omega t
      Kokkos::deep_copy(r_, s_);
      axpy(r_, -omega, View<const double>(t_), n);
      rnorm = std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), n));
      if (rnorm <= tol * R.res0) {
        ++it;
        break;
      }
      rho = rhoNew;
      if (omega == 0.0) break;
    }
    R.iters = it;
    R.res = rnorm;
    return R;
  }

 private:
  // z = M^{-1} v : a Helmholtz MG V-cycle if set, else `jacPre_` damped-Jacobi sweeps of
  // A z = v starting from z = 0.
  void applyPrec(const DeviceMomentumOp& op, View<double> v, View<double> z) {
    if (mgPre_) {  // V-cycle of the Helmholtz operator (≈ the momentum operator)
      Kokkos::deep_copy(mgPre_->b(0), v);
      Kokkos::deep_copy(mgPre_->x(0), 0.0);
      for (int k = 0; k < mgVcycles_; ++k) mgPre_->vcycle(2, 2, 60, 0.8);
      Kokkos::deep_copy(z, mgPre_->x(0));
      return;
    }
    Kokkos::deep_copy(z, 0.0);
    if (jacPre_ <= 0) {  // no preconditioner ⇒ identity
      Kokkos::deep_copy(z, v);
      return;
    }
    for (int s = 0; s < jacPre_; ++s) deviceJacobiMom(op, z, View<const double>(v), tmp_, omega_);
  }
  void ensure(Index n) {
    if (r_.extent(0) == static_cast<std::size_t>(n)) return;
    auto mk = [&](const char* l) { return View<double>(l, static_cast<std::size_t>(n)); };
    r_ = mk("mom_r");
    rhat_ = mk("mom_rhat");
    p_ = mk("mom_p");
    phat_ = mk("mom_phat");
    v_ = mk("mom_v");
    s_ = mk("mom_s");
    shat_ = mk("mom_shat");
    t_ = mk("mom_t");
    tmp_ = mk("mom_tmp");
  }

  View<double> r_, rhat_, p_, phat_, v_, s_, shat_, t_, tmp_;
  int jacPre_ = 2;
  double omega_ = 0.7;
  DeviceMultigrid<3, Bits>* mgPre_ = nullptr;
  int mgVcycles_ = 1;
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_MOMENTUM_HPP

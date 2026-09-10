// core — preconditioned BiCGStab (and defect correction / plain Jacobi) for an assembled face-CSR
// operator on the device (Kokkos).
//
// Lifted verbatim from the AMR tree (peclet/core/amr/momentum.hpp, 2026-09-10; QUALITY_PLAN G.2).
// The class carried a vestigial `Bits` template parameter there (it never touched the octree) —
// dropped here; the AMR package keeps `MomentumSolver<Bits>` as an alias template. Reuses the
// device matvec + Kokkos reductions of csr_operator.hpp; the default preconditioner is `jacPre`
// damped-Jacobi sweeps of A (diagonal-dominant ⇒ a cheap, effective smoother-preconditioner), and
// setPreconditioner() takes any z = M⁻¹ r callable (the AMR Galerkin momentum multigrid, an AMG).
// setDistributed() makes the same iteration run multi-rank through two callables (ghost refresh +
// global dot), so this header stays MPI-free. Consumers: peclet-amr's AmrFlow, voro's OT optimiser.
#ifndef PECLET_CORE_SOLVER_CSR_BICGSTAB_HPP
#define PECLET_CORE_SOLVER_CSR_BICGSTAB_HPP

#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/csr_operator.hpp"
#include "peclet/core/solver/vector_ops.hpp"

namespace peclet::core::solver {

// ---------------------------------------------------------------------------
// Jacobi-preconditioned BiCGStab for the (non-symmetric) momentum operator. Reuses the
// device matvec + Kokkos reductions; the preconditioner is `jacPre` damped-Jacobi sweeps
// of A (diagonal-dominant ⇒ a cheap, effective smoother-preconditioner). Robust where
// plain Jacobi stalls (large dt / weak reaction term).
// ---------------------------------------------------------------------------
class MomentumSolver {
 public:
  void setJacobi(int preSweeps, double omega) {
    jacPre_ = preSweeps;
    omega_ = omega;
  }

  /// Distributed solve (docs/amr_distributed_flow.md, rung 2): `refresh` re-fills the ghost tail
  /// [op.n, nExt) of a vector from its owners (LeafHaloExchange::exchange) and is called before
  /// EVERY read of a vector's neighbour entries — the initial residual, each preconditioner
  /// Jacobi sweep, each matvec of a preconditioned direction. `dotReduce` folds a local dot
  /// into the global one (an MPI_Allreduce lambda — kept as a callable so this header stays
  /// MPI-free; local rows only, ghosts are never summed). Scratch vectors are allocated at
  /// nExt so they can carry ghost tails. Jacobi preconditioning reads only the previous
  /// iterate, so the distributed iterate sequence matches the single-rank one bit-for-bit up
  /// to the dots' reduction order. Unset (default): the single-rank behaviour, bit-identical.
  void setDistributed(std::function<void(View<double>)> refresh,
                      std::function<double(double)> dotReduce, Index nExt) {
    haloFn_ = std::move(refresh);
    dotReduce_ = std::move(dotReduce);
    nExt_ = nExt;
  }

  /// Set a generic preconditioner `z = M⁻¹ r` (a host callable that launches device kernels) — the
  /// multigrid V-cycle gives the smooth-mode coverage Jacobi lacks, so the momentum iteration count
  /// stops growing with N. Decoupled from the MG type (Galerkin MomentumMG or rediscretized
  /// VelocityMG) via std::function, so the two coarse-operator strategies are interchangeable
  /// (and the solver carries no MG type). Pass an empty function to revert to damped-Jacobi. The
  /// preconditioner never changes the converged solution (the matvec is the exact operator).
  void setPreconditioner(std::function<void(View<const double>, View<double>)> fn) {
    precFn_ = std::move(fn);
  }

  /// Plain weighted-Jacobi solve (the simple parallel mirror of the host GS smoother):
  /// `sweeps` damped-Jacobi sweeps of A u = b in place. Returns the final residual L2.
  double solveJacobi(const MomentumOp& op, View<double> u, View<const double> b, int sweeps) {
    ensure(op.n);
    for (int s = 0; s < sweeps; ++s) {
      sync(u);
      jacobiMom(op, u, b, tmp_, omega_);
    }
    sync(u);
    residualMom(op, View<const double>(u), b, r_);
    return std::sqrt(dot(View<const double>(r_), View<const double>(r_), op.n));
  }

  struct Result {
    int iters = 0;
    double res0 = 0.0;
    double res = 0.0;
  };

  /// MG-preconditioned defect-correction (Richardson) solve of A u = b in place:
  /// u ← u + M⁻¹(b − A u), M = the preconditioner (velocity-MG if set, else Jacobi sweeps).
  /// Unlike BiCGStab it cannot break down — robust for the strongly non-symmetric momentum
  /// operator with implicit-FOU advection, where the velocity-MG (built from the viscous base)
  /// is only an approximate inverse. Converges when the advection is a perturbation of the
  /// viscous+reaction operator (low–moderate cell Reynolds number). `maxIters` caps the
  /// iterations; `tol` is relative to ||b−Au₀||.
  Result solveDefectCorrection(const MomentumOp& op, View<double> u, View<const double> b,
                               int maxIters = 200, double tol = 1e-8) {
    const Index n = op.n;
    ensure(n);
    Result R;
    sync(u);
    residualMom(op, View<const double>(u), b, r_);
    R.res0 = std::sqrt(dot(View<const double>(r_), View<const double>(r_), n));
    if (R.res0 == 0.0)
      return R;
    double rnorm = R.res0;
    int it = 0;
    for (; it < maxIters; ++it) {
      applyPrec(op, r_, phat_);                    // phat = M⁻¹ r
      axpy(u, 1.0, View<const double>(phat_), n);  // u += phat
      sync(u);
      residualMom(op, View<const double>(u), b, r_);
      rnorm = std::sqrt(dot(View<const double>(r_), View<const double>(r_), n));
      if (rnorm <= tol * R.res0) {
        ++it;
        break;
      }
    }
    R.iters = it;
    R.res = rnorm;
    return R;
  }

  /// Jacobi-preconditioned BiCGStab solve of A u = b in place. `maxIters` caps the outer
  /// iterations; `tol` is relative to ||b−Au0||. Returns {iters, final residual L2}.
  Result solveBiCGStab(const MomentumOp& op, View<double> u, View<const double> b,
                       int maxIters = 500, double tol = 1e-10) {
    const Index n = op.n;
    ensure(n);
    Result R;
    // r = b − A u
    sync(u);
    residualMom(op, View<const double>(u), b, r_);
    Kokkos::deep_copy(rhat_, r_);  // shadow residual
    R.res0 = std::sqrt(dot(View<const double>(r_), View<const double>(r_), n));
    if (R.res0 == 0.0)
      return R;
    double rho = 1, alpha = 1, omega = 1;
    Kokkos::deep_copy(v_, 0.0);
    Kokkos::deep_copy(p_, 0.0);
    double rnorm = R.res0;
    int it = 0;
    for (; it < maxIters; ++it) {
      double rhoNew = dot(View<const double>(rhat_), View<const double>(r_), n);
      if (rhoNew == 0.0)
        break;
      double beta = (rhoNew / rho) * (alpha / omega);
      // p = r + beta (p − omega v)
      bicgPUpdate(p_, View<const double>(r_), View<const double>(v_), beta, omega, n);
      applyPrec(op, p_, phat_);  // phat = M^{-1} p
      sync(phat_);
      applyMom(op, View<const double>(phat_), v_);
      double rhatV = dot(View<const double>(rhat_), View<const double>(v_), n);
      alpha = rhoNew / rhatV;
      // s = r − alpha v
      Kokkos::deep_copy(s_, r_);
      axpy(s_, -alpha, View<const double>(v_), n);
      double snorm = std::sqrt(dot(View<const double>(s_), View<const double>(s_), n));
      if (snorm <= tol * R.res0) {
        axpy(u, alpha, View<const double>(phat_), n);  // u += alpha phat
        rnorm = snorm;
        ++it;
        break;
      }
      applyPrec(op, s_, shat_);  // shat = M^{-1} s
      sync(shat_);
      applyMom(op, View<const double>(shat_), t_);
      double tt = dot(View<const double>(t_), View<const double>(t_), n);
      omega = (tt != 0.0) ? dot(View<const double>(t_), View<const double>(s_), n) / tt : 0.0;
      // u += alpha phat + omega shat
      axpy(u, alpha, View<const double>(phat_), n);
      axpy(u, omega, View<const double>(shat_), n);
      // r = s − omega t
      Kokkos::deep_copy(r_, s_);
      axpy(r_, -omega, View<const double>(t_), n);
      rnorm = std::sqrt(dot(View<const double>(r_), View<const double>(r_), n));
      if (rnorm <= tol * R.res0) {
        ++it;
        break;
      }
      rho = rhoNew;
      if (omega == 0.0)
        break;
    }
    R.iters = it;
    R.res = rnorm;
    return R;
  }

 private:
  // z = M^{-1} v : the generic MG preconditioner if set, else `jacPre_` damped-Jacobi sweeps of
  // A z = v starting from z = 0.
  void applyPrec(const MomentumOp& op, View<double> v, View<double> z) {
    if (precFn_) {
      precFn_(View<const double>(v), z);
      return;
    }
    Kokkos::deep_copy(z, 0.0);
    if (jacPre_ <= 0) {  // no preconditioner ⇒ identity
      Kokkos::deep_copy(z, v);
      return;
    }
    for (int s = 0; s < jacPre_; ++s) {
      if (s)
        sync(z);  // ghosts of the previous iterate (first sweep: z = 0 everywhere already)
      jacobiMom(op, z, View<const double>(v), tmp_, omega_);
    }
  }
  /// Refresh the ghost tail of a vector before its neighbour entries are read (no-op
  /// single-rank).
  void sync(View<double> v) const {
    if (haloFn_)
      haloFn_(v);
  }
  /// Local dot over the owned rows, globally reduced when distributed.
  double dot(View<const double> a, View<const double> b, Index n) const {
    const double s = dotPlain(a, b, n);
    return dotReduce_ ? dotReduce_(s) : s;
  }
  void ensure(Index n) {
    if (nExt_ > n)
      n = nExt_;  // scratch carries the ghost tail in distributed solves
    if (r_.extent(0) == static_cast<std::size_t>(n))
      return;
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
  std::function<void(View<const double>, View<double>)> precFn_;  // generic z = M^{-1} r
  std::function<void(View<double>)> haloFn_;  // ghost-tail refresh (unset ⇒ no-op)
  std::function<double(double)> dotReduce_;   // global dot reduction (unset ⇒ local)
  Index nExt_ = 0;                            // extended (local+ghost) scratch size
};

}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_CSR_BICGSTAB_HPP

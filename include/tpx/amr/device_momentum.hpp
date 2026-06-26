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
#include <map>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/device_multigrid.hpp"  // deviceRestrict / deviceProlongAdd transfer kernels
#include "tpx/amr/device_pcg.hpp"        // dotPlain-style primitives: axpy, zpby, negate
#include "tpx/common/view.hpp"

namespace tpx::amr {

/// Assembled momentum operator on the device: (A u)_i = diag_i u_i + Σ coef·u[nbr], with an
/// optional implicit-FOU advection part (rebuilt each step from the lagged velocity): a
/// per-cell outflow diagonal `advDiag` + per-face inflow coefficients over a second
/// (face-geometry) CSR. hasAdv=false ⇒ the pure cut-cell operator, bit-exact unchanged.
struct DeviceMomentumOp {
  View<double> diag;      ///< size n
  View<Index> faceStart;  ///< CSR row offsets, size n+1
  View<Index> faceNbr;    ///< neighbour leaf per off-diagonal, size nnz
  View<double> faceCoef;  ///< off-diagonal coefficient, size nnz
  Index n = 0;
  // Optional implicit-FOU advection (over the face-geometry CSR):
  bool hasAdv = false;
  View<double> advDiag;   ///< per-cell outflow (diagonal) advection weight, size n
  View<Index> advStart;   ///< face-geom CSR row offsets, size n+1
  View<Index> advNbr;     ///< face-geom neighbour per face, size nFaces
  View<double> advCoef;   ///< per-face inflow advection coefficient (0 on outflow/solid faces)
};

/// Au = A u (cut-cell operator + optional implicit-FOU advection).
inline void deviceApplyMom(const DeviceMomentumOp& op, View<const double> u, View<double> Au) {
  auto diag = op.diag;
  auto fs = op.faceStart;
  auto fn = op.faceNbr;
  auto fc = op.faceCoef;
  const bool hasAdv = op.hasAdv;
  auto ad = op.advDiag;
  auto as = op.advStart;
  auto an = op.advNbr;
  auto ac = op.advCoef;
  Kokkos::parallel_for(
      "amr::mom_apply", op.n, KOKKOS_LAMBDA(const Index i) {
        double acc = diag(i) * u(i);
        for (Index k = fs(i); k < fs(i + 1); ++k) acc += fc(k) * u(fn(k));
        if (hasAdv) {
          acc += ad(i) * u(i);
          for (Index k = as(i); k < as(i + 1); ++k) acc += ac(k) * u(an(k));
        }
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
  const bool hasAdv = op.hasAdv;
  auto ad = op.advDiag;
  auto as = op.advStart;
  auto an = op.advNbr;
  auto ac = op.advCoef;
  Kokkos::parallel_for(
      "amr::mom_residual", op.n, KOKKOS_LAMBDA(const Index i) {
        double acc = diag(i) * u(i);
        for (Index k = fs(i); k < fs(i + 1); ++k) acc += fc(k) * u(fn(k));
        if (hasAdv) {
          acc += ad(i) * u(i);
          for (Index k = as(i); k < as(i + 1); ++k) acc += ac(k) * u(an(k));
        }
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
  const bool hasAdv = op.hasAdv;
  auto ad = op.advDiag;
  auto as = op.advStart;
  auto an = op.advNbr;
  auto ac = op.advCoef;
  Kokkos::parallel_for(
      "amr::mom_jacobi_compute", op.n, KOKKOS_LAMBDA(const Index i) {
        double off = 0.0;
        for (Index k = fs(i); k < fs(i + 1); ++k) off += fc(k) * u(fn(k));
        double d = diag(i);
        if (hasAdv) {
          for (Index k = as(i); k < as(i + 1); ++k) off += ac(k) * u(an(k));
          d += ad(i);
        }
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

// ===========================================================================
// DeviceMomentumMG — Galerkin geometric multigrid for the momentum operator.
//
// The cut-cell momentum operator carries the ξ-polynomial Dirichlet overlay and its
// D_rescale row scaling, so a *rediscretised* coarse operator (the openness-Helmholtz
// attempt) mismatches it and makes a poor preconditioner. Instead the coarse operators are
// built by **Galerkin coarsening** A_c = R·A·P of the exact assembled fine CSR: R = volume
// average over a coarse cell's children, P = piecewise-constant injection (the same transfer
// pair the pressure MG uses). This is consistent with the fine operator by construction — it
// inherits the cut-cell stencil and row scaling, and a coarse cell whose children are all
// solid (identity rows) stays an identity row (ε-solid-on-coarse emerges for free). The
// hierarchy is the uniformly-coarsened octree; the smoother is deviceJacobiMom, the residual
// restriction / correction prolongation are the shared deviceRestrict / deviceProlongAdd.
// Used as the momentum BiCGStab preconditioner ⇒ the iteration count stays ~flat with N.
// ===========================================================================
template <unsigned Bits = 21u>
class DeviceMomentumMG {
 public:
  using Octree = BlockOctree<3, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;

  /// Build the Galerkin hierarchy from the finest octree + the assembled fine operator CSR
  /// (diag + face CSR, as produced by AmrCutCell::assembleOperator).
  void build(const Octree& finest, const std::vector<double>& diag0,
             const std::vector<Index>& start0, const std::vector<Index>& nbr0,
             const std::vector<double>& coef0) {
    octs_.clear();
    octs_.push_back(finest);
    for (;;) {
      Octree c = octs_.back();
      Index merged = c.coarsenIf([](Code, unsigned) { return true; });
      if (merged == 0 || c.numLeaves() == octs_.back().numLeaves()) break;
      octs_.push_back(c);
      if (c.numLeaves() == 1) break;
    }
    const std::size_t nl = octs_.size();
    levels_.clear();
    levels_.resize(nl);

    std::vector<double> hdiag = diag0, hcoef = coef0;
    std::vector<Index> hstart = start0, hnbr = nbr0;
    uploadLevel(0, hdiag, hstart, hnbr, hcoef);

    for (std::size_t L = 0; L + 1 < nl; ++L) {
      const Octree& f = octs_[L];
      const Octree& c = octs_[L + 1];
      const Index nf = f.numLeaves(), nc = c.numLeaves();
      std::vector<Index> c2p(static_cast<std::size_t>(nf));
      std::vector<Index> cnt(static_cast<std::size_t>(nc), 0);
      for (Index i = 0; i < nf; ++i) {
        Code par = M::from_code(f.code(i)).ancestor(f.level(i) + 1).code();
        Index p = c.find(par);
        c2p[static_cast<std::size_t>(i)] = p;
        if (p >= 0) ++cnt[static_cast<std::size_t>(p)];
      }
      std::vector<Index> cstart(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p)
        cstart[static_cast<std::size_t>(p) + 1] = cstart[static_cast<std::size_t>(p)] + cnt[static_cast<std::size_t>(p)];
      std::vector<Index> cidx(static_cast<std::size_t>(nf));
      std::vector<Index> cur(cstart.begin(), cstart.end() - 1);
      for (Index i = 0; i < nf; ++i) {
        Index p = c2p[static_cast<std::size_t>(i)];
        if (p >= 0) cidx[static_cast<std::size_t>(cur[static_cast<std::size_t>(p)]++)] = i;
      }
      levels_[L].c2p = toDevice(c2p, "mmg_c2p");
      levels_[L].childStart = toDevice(cstart, "mmg_cstart");
      levels_[L].childIdx = toDevice(cidx, "mmg_cidx");

      // Galerkin A_c[p][q] = (1/n_ch[p]) Σ_{i child of p} ( A[i] entries mapped to parents ).
      std::vector<std::map<Index, double>> acc(static_cast<std::size_t>(nc));
      for (Index i = 0; i < nf; ++i) {
        Index p = c2p[static_cast<std::size_t>(i)];
        if (p < 0) continue;
        const double w = 1.0 / static_cast<double>(cnt[static_cast<std::size_t>(p)]);
        acc[static_cast<std::size_t>(p)][p] += w * hdiag[static_cast<std::size_t>(i)];
        for (Index k = hstart[static_cast<std::size_t>(i)]; k < hstart[static_cast<std::size_t>(i) + 1]; ++k) {
          Index q = c2p[static_cast<std::size_t>(hnbr[static_cast<std::size_t>(k)])];
          if (q < 0) continue;
          acc[static_cast<std::size_t>(p)][q] += w * hcoef[static_cast<std::size_t>(k)];
        }
      }
      std::vector<double> cdiag(static_cast<std::size_t>(nc), 0.0);
      std::vector<Index> cs(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p) {
        int off = 0;
        for (auto& e : acc[static_cast<std::size_t>(p)]) {
          if (e.first == p)
            cdiag[static_cast<std::size_t>(p)] = e.second;
          else
            ++off;
        }
        cs[static_cast<std::size_t>(p) + 1] = cs[static_cast<std::size_t>(p)] + off;
      }
      std::vector<Index> cn(static_cast<std::size_t>(cs[static_cast<std::size_t>(nc)]));
      std::vector<double> ccoef(static_cast<std::size_t>(cs[static_cast<std::size_t>(nc)]));
      for (Index p = 0; p < nc; ++p) {
        Index k = cs[static_cast<std::size_t>(p)];
        for (auto& e : acc[static_cast<std::size_t>(p)])
          if (e.first != p) {
            cn[static_cast<std::size_t>(k)] = e.first;
            ccoef[static_cast<std::size_t>(k)] = e.second;
            ++k;
          }
      }
      uploadLevel(L + 1, cdiag, cs, cn, ccoef);
      hdiag = cdiag;
      hstart = cs;
      hnbr = cn;
      hcoef = ccoef;
    }
    for (auto& lv : levels_) {
      lv.x = View<double>("mmg_x", static_cast<std::size_t>(lv.op.n));
      lv.b = View<double>("mmg_b", static_cast<std::size_t>(lv.op.n));
      lv.res = View<double>("mmg_res", static_cast<std::size_t>(lv.op.n));
      lv.tmp = View<double>("mmg_tmp", static_cast<std::size_t>(lv.op.n));
    }
  }

  /// One V-cycle on level L solving A u = b (correction scheme), Jacobi smoother.
  void vcycle(int pre = 2, int post = 2, int bottom = 30, double omega = 0.7, std::size_t L = 0) {
    Level& lv = levels_[L];
    View<const double> bc(lv.b);
    if (L + 1 == levels_.size()) {
      for (int s = 0; s < bottom; ++s) deviceJacobiMom(lv.op, lv.x, bc, lv.tmp, omega);
      return;
    }
    for (int s = 0; s < pre; ++s) deviceJacobiMom(lv.op, lv.x, bc, lv.tmp, omega);
    deviceResidualMom(lv.op, View<const double>(lv.x), bc, lv.res);
    Level& cl = levels_[L + 1];
    deviceRestrict(lv.childStart, lv.childIdx, View<const double>(lv.res), cl.b, cl.op.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(pre, post, bottom, omega, L + 1);
    deviceProlongAdd(lv.c2p, View<const double>(cl.x), lv.x, lv.op.n);
    for (int s = 0; s < post; ++s) deviceJacobiMom(lv.op, lv.x, bc, lv.tmp, omega);
  }

  std::size_t numLevels() const { return levels_.size(); }
  Index numLeaves(std::size_t L = 0) const { return levels_[L].op.n; }
  View<double> x(std::size_t L = 0) { return levels_[L].x; }
  View<double> b(std::size_t L = 0) { return levels_[L].b; }
  const DeviceMomentumOp& op(std::size_t L = 0) const { return levels_[L].op; }

 private:
  struct Level {
    DeviceMomentumOp op;
    View<double> x, b, res, tmp;
    View<Index> c2p, childStart, childIdx;
  };
  void uploadLevel(std::size_t L, const std::vector<double>& diag, const std::vector<Index>& start,
                   const std::vector<Index>& nbr, const std::vector<double>& coef) {
    Level& lv = levels_[L];
    lv.op.n = static_cast<Index>(diag.size());
    lv.op.diag = toDevice(diag, "mmg_diag");
    lv.op.faceStart = toDevice(start, "mmg_start");
    lv.op.faceNbr = toDevice(nbr, "mmg_nbr");
    lv.op.faceCoef = toDevice(coef, "mmg_coef");
  }
  std::vector<Octree> octs_;
  std::vector<Level> levels_;
};

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

  /// Use a Galerkin DeviceMomentumMG (built from the same assembled operator) as the BiCGStab
  /// preconditioner instead of damped-Jacobi sweeps. The V-cycle gives the multigrid
  /// smooth-mode coverage Jacobi lacks, so the momentum iteration count stops growing with N.
  /// `vcycles` V-cycles per preconditioner application. Pass nullptr to revert to Jacobi. The
  /// preconditioner never changes the converged solution (the BiCGStab matvec is the exact
  /// operator) — only the iteration count.
  void setMgPreconditioner(DeviceMomentumMG<Bits>* mg, int vcycles = 1) {
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
  Result solveDefectCorrection(const DeviceMomentumOp& op, View<double> u, View<const double> b,
                               int maxIters = 200, double tol = 1e-8) {
    const Index n = op.n;
    ensure(n);
    Result R;
    deviceResidualMom(op, View<const double>(u), b, r_);
    R.res0 = std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), n));
    if (R.res0 == 0.0) return R;
    double rnorm = R.res0;
    int it = 0;
    for (; it < maxIters; ++it) {
      applyPrec(op, r_, phat_);                       // phat = M⁻¹ r
      axpy(u, 1.0, View<const double>(phat_), n);     // u += phat
      deviceResidualMom(op, View<const double>(u), b, r_);
      rnorm = std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), n));
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
  DeviceMomentumMG<Bits>* mgPre_ = nullptr;
  int mgVcycles_ = 1;
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_MOMENTUM_HPP

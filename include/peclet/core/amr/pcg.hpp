// core — device (Kokkos) multigrid-preconditioned CG for the AMR FV Poisson.
//
// A Krylov accelerator on top of the existing device machinery: the matvec is the
// consistent conservative FV Laplacian `applyFv` (poisson.hpp), the
// preconditioner is one (or a few) Multigrid V-cycle(s) (multigrid.hpp),
// and the inner products / vector updates are Kokkos reductions / parallel_fors. This
// is exactly flow's structured MG-PCG, ported onto the AMR octree CSR: CG accelerates
// the geometric MG so a given residual is reached in far fewer fine-grid matvecs than
// stationary V-cycling, on whatever backend Kokkos targets (CUDA / HIP / OpenMP).
//
// SPD subtlety: the FV operator L = D^{-1} S (D = diag(cell volume), S the symmetric
// stencil) is *not* symmetric in the Euclidean inner product, but it is symmetric and
// negative-definite in the volume-weighted inner product <u,v>_D = Σ V_i u_i v_i. So CG
// runs on A := −L (SPD in <·,·>_D) and every dot product is volume-weighted. The V-cycle
// preconditioner solves L z = −r (correction scheme) ⇒ z ≈ A^{-1} r.
//
// Singular (periodic, pure-Neumann) case: A has the constant nullspace. The RHS, the
// residual, and the preconditioned residual are projected volume-weighted-mean-zero each
// iteration (deflated CG) so the iteration stays in the range space. The homogeneous-
// Dirichlet build is non-singular (bcDiag > 0) ⇒ no projection.
//
// The MG preconditioner is bit-exact deterministic (Jacobi smoother); CG itself depends
// on global-reduction summation order, so this is the *performance* path — validated by
// convergence + matching the V-cycle's converged solution, not by host bit-exactness.
//
// Requires a Kokkos build + the morton checkout (PECLET_CORE_HAVE_MORTON).
#ifndef PECLET_CORE_AMR_PCG_HPP
#define PECLET_CORE_AMR_PCG_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <cmath>
#include <functional>

#include "peclet/core/amr/fv_op.hpp"
#include "peclet/core/amr/multigrid.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::core::amr {

// ---- small device vector primitives (volume-weighted where the FV operator needs it) ----

/// Volume-weighted dot <u,v>_D = Σ_i V_i u_i v_i, V_i = 1/invVol_i.
inline double dotVol(View<const double> u, View<const double> v, View<const double> invVol,
                     Index n) {
  double s = 0.0;
  Kokkos::parallel_reduce(
      "amr::pcg_dotvol", n,
      KOKKOS_LAMBDA(const Index i, double& acc) { acc += (u(i) * v(i)) / invVol(i); }, s);
  return s;
}

/// Build the fluid mask: mask(i)=1 where the operator diagonal Σ_f w_f (+ bcDiag) is non-trivial,
/// 0 for a solid cell (every face closed by the cut-cell openness ⇒ Σ w_f = 0). These solid cells
/// carry their own (per connected solid region) constant null modes; left in the Krylov space CG
/// amplifies them (the cut-cell-openness near-nullspace blow-up). We project them out — same role
/// as flow's mg_mask_solid_k. Geometry-fixed, so it is rebuilt once per solve.
inline void buildFluidMask(const FvOp& op, View<double> mask, Index n) {
  auto start = op.faceStart;
  auto w = op.faceW;
  auto bc = op.bcDiag;
  Kokkos::parallel_for(
      "amr::pcg_fluidmask", n, KOKKOS_LAMBDA(const Index i) {
        double d = bc(i);
        for (Index k = start(i); k < start(i + 1); ++k)
          d += w(k);
        mask(i) = (d > 1e-30) ? 1.0 : 0.0;
      });
}

/// Zero the solid cells (project out the solid null modes).
inline void maskSolid(View<double> u, View<const double> mask, Index n) {
  Kokkos::parallel_for("amr::pcg_masksolid", n, KOKKOS_LAMBDA(const Index i) { u(i) *= mask(i); });
}

/// Project u onto the FLUID range: zero solid cells, then subtract the volume-weighted mean over
/// the fluid cells only (the constant null mode of the connected fluid region). The mean must
/// exclude the pinned solid cells — including them dilutes it and lets the solid drift.
/// `reduce` folds each local mean sum into the global one (identity single-rank; an
/// MPI_Allreduce lambda distributed — the mean is over the GLOBAL fluid region).
inline void removeMeanVolReduced(View<double> u, View<const double> invVol,
                                 View<const double> mask, Index n,
                                 const std::function<double(double)>& reduce) {
  Kokkos::parallel_for("amr::pcg_masksolid", n, KOKKOS_LAMBDA(const Index i) { u(i) *= mask(i); });
  double su = 0.0, sv = 0.0;
  Kokkos::parallel_reduce(
      "amr::pcg_meannum", n,
      KOKKOS_LAMBDA(const Index i, double& a) { a += mask(i) * u(i) / invVol(i); }, su);
  Kokkos::parallel_reduce(
      "amr::pcg_meanden", n, KOKKOS_LAMBDA(const Index i, double& a) { a += mask(i) / invVol(i); },
      sv);
  if (reduce) {
    su = reduce(su);
    sv = reduce(sv);
  }
  const double m = (sv > 0.0) ? su / sv : 0.0;
  Kokkos::parallel_for(
      "amr::pcg_meansub", n, KOKKOS_LAMBDA(const Index i) { u(i) -= mask(i) * m; });
}

inline void removeMeanVol(View<double> u, View<const double> invVol, View<const double> mask,
                          Index n) {
  removeMeanVolReduced(u, invVol, mask, n, {});
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

// ---------------------------------------------------------------------------
// MG-preconditioned CG over a Multigrid, driving the system L x = rhs on its
// finest level. Owns the Krylov scratch; reuses the multigrid's own finest x/b as
// transient preconditioner storage. Solves A x = b_A with A := −L (SPD in <·,·>_D).
// ---------------------------------------------------------------------------
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class PCG {
 public:
  using MG = Multigrid<Dim, Bits>;

  struct Result {
    int iters = 0;
    double res0 = 0.0;  ///< initial volume-weighted residual norm
    double res = 0.0;   ///< final volume-weighted residual norm
  };

  /// V-cycle parameters used for the preconditioner application.
  void setVcycle(int pre, int post, int bottom, double omega) {
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    omega_ = omega;
  }
  /// Number of V-cycles per preconditioner application (default 1).
  void setCyclesPerPrec(int k) { cyclesPerPrec_ = k; }
  /// Whether to project out the constant nullspace (default true; set false for the
  /// non-singular homogeneous-Dirichlet operator).
  void setSingular(bool s) { singular_ = s; }

  /// Distributed solve (docs/amr_distributed_flow.md, rung 3): `refresh` re-fills the ghost
  /// tail [n, nExt) of a vector before its neighbour entries are read (the CG direction ahead
  /// of every matvec); `dotReduce` folds a local sum into the global one (an MPI_Allreduce
  /// lambda — a callable so this header stays MPI-free), applied to every volume-weighted dot
  /// AND to the nullspace-projection mean sums. Scratch (and the search direction) allocate at
  /// nExt so they carry ghost tails; the multigrid passed to solve() must size its finest
  /// x(0)/b(0) at the same nExt (DistributedFlowMultigrid does). Unset (default): the
  /// single-rank behaviour, bit-identical.
  void setDistributed(std::function<void(View<double>)> refresh,
                      std::function<double(double)> dotReduce, Index nExt) {
    haloFn_ = std::move(refresh);
    dotReduce_ = std::move(dotReduce);
    nExt_ = nExt;
  }

  /// Solve L x = rhs on mg's finest level into `x` (size n; nExt distributed). Returns
  /// iteration count and residual history. `tol` is relative to the initial residual;
  /// `maxIters` caps the CG iterations. The multigrid `mg` must already be built (operator +
  /// hierarchy); any type exposing op(0)/x(0)/b(0)/numLeaves(0)/vcycle works
  /// (Multigrid, DistributedFlowMultigrid).
  template <class MGT = MG>
  Result solve(MGT& mg, View<double> x, View<const double> rhs, int maxIters = 200,
               double tol = 1e-10) {
    const Index n = mg.numLeaves(0);
    const FvOp& op = mg.op(0);
    View<const double> invVol(op.invVol);
    ensure(n);
    buildFluidMask(op, mask_, n);
    View<const double> mask(mask_);
    Result R;
    // Project onto the fluid range: always zero the solid cells (their per-region null modes); for
    // the singular (periodic/all-Neumann) operator also remove the fluid constant. Applied to every
    // Krylov quantity so the iteration stays in the well-posed fluid range (mirrors flow
    // removeMean∘maskSolid).
    auto project = [&](View<double> u) {
      if (singular_)
        removeMeanVolReduced(u, invVol, mask, n, dotReduce_);
      else
        maskSolid(u, mask, n);
    };
    auto vdot = [&](View<const double> a, View<const double> b) {
      const double s = dotVol(a, b, invVol, n);
      return dotReduce_ ? dotReduce_(s) : s;
    };

    // x = 0 ; r = b_A − A·0 = b_A = −rhs (A = −L, b_A = −rhs). Element copy over the local
    // rows (not deep_copy): rhs may be local-sized while the scratch carries a ghost tail.
    Kokkos::deep_copy(x, 0.0);
    {
      auto r = r_;
      Kokkos::parallel_for(
          "amr::pcg_r0", n, KOKKOS_LAMBDA(const Index i) { r(i) = -rhs(i); });
    }
    project(r_);
    R.res0 = std::sqrt(vdot(View<const double>(r_), View<const double>(r_)));
    if (R.res0 == 0.0)
      return R;

    applyPrec(mg, r_, z_, n);  // z = M^{-1} r ≈ A^{-1} r
    project(z_);
    Kokkos::deep_copy(p_, z_);
    double rz = vdot(View<const double>(r_), View<const double>(z_));

    int it = 0;
    double rnorm = R.res0;
    for (; it < maxIters; ++it) {
      // Ap = A p = −L p, projected back onto the fluid range (keeps the search directions there).
      sync(p_);  // ghost tail of the direction before the matvec (no-op single-rank)
      applyFv(op, View<const double>(p_), Ap_);
      negate(Ap_, n);
      project(Ap_);
      double pAp = vdot(View<const double>(p_), View<const double>(Ap_));
      if (pAp == 0.0)
        break;
      double alpha = rz / pAp;
      axpy(x, alpha, View<const double>(p_), n);     // x += α p
      axpy(r_, -alpha, View<const double>(Ap_), n);  // r −= α Ap
      project(r_);
      rnorm = std::sqrt(vdot(View<const double>(r_), View<const double>(r_)));
      if (rnorm <= tol * R.res0) {
        ++it;
        break;
      }
      applyPrec(mg, r_, z_, n);
      project(z_);
      double rzNew = vdot(View<const double>(r_), View<const double>(z_));
      double beta = rzNew / rz;
      zpby(p_, View<const double>(z_), beta, n);  // p = z + β p
      rz = rzNew;
    }
    project(x);  // solid cells exactly 0; fluid mean removed (singular)
    R.iters = it;
    R.res = rnorm;
    return R;
  }

 private:
  // z = M^{-1} r : solve L z = −r with `cyclesPerPrec_` V-cycles (correction scheme),
  // using the multigrid's own finest x/b as scratch. (A = −L ⇒ A z = r ⟺ L z = −r.)
  // The distributed multigrid sizes its finest x/b at the same nExt as this solver's scratch,
  // so the deep_copies stay extent-matched.
  template <class MGT>
  void applyPrec(MGT& mg, View<double> r, View<double> z, Index n) {
    Kokkos::deep_copy(mg.b(0), r);
    negate(mg.b(0), n);
    Kokkos::deep_copy(mg.x(0), 0.0);
    for (int k = 0; k < cyclesPerPrec_; ++k)
      mg.vcycle(pre_, post_, bottom_, omega_);
    Kokkos::deep_copy(z, mg.x(0));
  }

  /// Refresh the ghost tail of a vector (no-op single-rank).
  void sync(View<double> v) const {
    if (haloFn_)
      haloFn_(v);
  }

  void ensure(Index n) {
    if (nExt_ > n)
      n = nExt_;  // scratch carries the ghost tail in distributed solves
    if (r_.extent(0) == static_cast<std::size_t>(n))
      return;
    r_ = View<double>("pcg_r", static_cast<std::size_t>(n));
    z_ = View<double>("pcg_z", static_cast<std::size_t>(n));
    p_ = View<double>("pcg_p", static_cast<std::size_t>(n));
    Ap_ = View<double>("pcg_Ap", static_cast<std::size_t>(n));
    mask_ = View<double>("pcg_fluidmask", static_cast<std::size_t>(n));
  }

  View<double> r_, z_, p_, Ap_, mask_;
  int pre_ = 2, post_ = 2, bottom_ = 40;
  double omega_ = 0.8;
  int cyclesPerPrec_ = 1;
  bool singular_ = true;
  std::function<void(View<double>)> haloFn_;  // ghost-tail refresh (unset ⇒ no-op)
  std::function<double(double)> dotReduce_;   // global reduction (unset ⇒ local)
  Index nExt_ = 0;                            // extended (local+ghost) scratch size
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_PCG_HPP

// core — device (Kokkos) geometric-multigrid V-cycle on a BlockOctree,
// with the *consistent* graded (cut-cell-ready) operator.
//
// The device compute path's capstone: a full MG V-cycle running entirely in device
// kernels over the leaf Views, on a genuinely graded octree. The hierarchy is the
// octree coarsened uniformly one level at a time (same as host AmrMultigrid); each
// level's operator is the conservative two-point FV Laplacian
//     (L u)_i = invVol_i · Σ_f w_f (u[nbr_f] − u_i),  w_f = openness·A_f/d_f,
// captured as a precomputed face CSR (built on the host from AmrPoisson's exact
// forEachFaceNeighbor enumeration, so the 2:1 sub-faces and coefficients match the
// host operator bit-for-bit). This replaces the earlier *plain* (u_j−u_i)/h0²
// operator, which ignored 2:1-interface geometry and so stalled the graded V-cycle.
//
// Every stage is deterministic / order-independent ⇒ bit-identical to the same
// Jacobi V-cycle on the host (validated in test_amr_device_multigrid_kokkos):
//   * smoother    — weighted Jacobi over the face CSR (reads only the prev iterate);
//   * restriction — each coarse cell sums its children in fixed CSR order (no atomics);
//   * prolongation— piecewise-constant, per fine cell.
//
// 2nd-order at 2:1 interfaces: solveQuad() wraps the V-cycle in deferred correction
// with the quadratic coarse-fine flux (AmrPoisson::coarseStar), evaluated on device
// as a second precomputed CSR (quadDelta). Openness (cut-cell) flows in for
// free: build with an AmrPoisson that has openness set and w_f carries it.
//
// Requires a Kokkos build + the morton checkout (PECLET_CORE_HAVE_MORTON).
#ifndef PECLET_CORE_AMR_MULTIGRID_HPP
#define PECLET_CORE_AMR_MULTIGRID_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/assembly.hpp"  // assembleFv (device per-level operator rebuild, D5)
#include "peclet/core/amr/fv_op.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::core::amr {

/// Restrict: coarse(p) = mean over p's children (CSR fixed order ⇒ deterministic).
inline void restrictField(View<const Index> childStart, View<const Index> childIdx,
                           View<const double> fine, View<double> coarse, Index nCoarse) {
  Kokkos::parallel_for(
      "amr::device_restrict", nCoarse, KOKKOS_LAMBDA(const Index p) {
        const Index a = childStart(p), z = childStart(p + 1);
        double s = 0.0;
        for (Index k = a; k < z; ++k) s += fine(childIdx(k));
        coarse(p) = (z > a) ? s / static_cast<double>(z - a) : 0.0;
      });
}

/// Restrict, κ-weighted: coarse(p) = Σ_child κ_c·fine_c / Σ_child κ_c, with κ the fine
/// cell's fluid-fraction weight (mean face aperture). Downweights nearly-solid children
/// at thin cut features. Reduces to restrictField when all κ are equal (openness-free).
/// (Experimental — opt-in via Multigrid::setKappaRestrict; see the comparison test.
/// NOTE: unlike the plain volume-average, this is *not* exactly conservative, so the
/// restricted residual of a mean-zero RHS need not stay mean-zero.)
inline void restrictKappa(View<const Index> childStart, View<const Index> childIdx,
                                View<const double> fine, View<const double> kappa,
                                View<double> coarse, Index nCoarse) {
  Kokkos::parallel_for(
      "amr::device_restrict_kappa", nCoarse, KOKKOS_LAMBDA(const Index p) {
        const Index a = childStart(p), z = childStart(p + 1);
        double sw = 0.0, swv = 0.0;
        for (Index k = a; k < z; ++k) {
          const Index ch = childIdx(k);
          const double w = kappa(ch);
          sw += w;
          swv += w * fine(ch);
        }
        coarse(p) = (sw > 0.0) ? swv / sw : 0.0;
      });
}

/// Prolong (piecewise-constant) + correct: fine(i) += coarse(c2p(i)).
inline void prolongAdd(View<const Index> c2p, View<const double> coarse, View<double> fine,
                             Index nFine) {
  Kokkos::parallel_for(
      "amr::device_prolong", nFine, KOKKOS_LAMBDA(const Index i) {
        const Index p = c2p(i);
        if (p >= 0) fine(i) += coarse(p);
      });
}

/// Masked piecewise-constant prolong + correct: fine(i) += coarse(c2p(i)) only on non-excluded fine
/// cells (mirrors flow VelocityMG::prolongMasked — no correction into a cut/solid cell). Generic.
inline void prolongAddMasked(View<const Index> c2p, View<const double> coarse,
                                   View<const char> excl, View<double> fine, Index nFine) {
  Kokkos::parallel_for(
      "amr::device_prolong_masked", nFine, KOKKOS_LAMBDA(const Index i) {
        if (excl(i)) return;
        const Index p = c2p(i);
        if (p >= 0) fine(i) += coarse(p);
      });
}

/// Zero `v` at excluded cells (excl != 0). Mirrors flow's mg_mul_mask: applied to the fine
/// residual before restriction so the inconsistent cut-cell + solid residuals never reach the
/// coarse grid (the clean-fluid exclude). Generic.
inline void zeroMasked(View<double> v, View<const char> excl, Index n) {
  Kokkos::parallel_for(
      "amr::device_zero_masked", n, KOKKOS_LAMBDA(const Index i) { if (excl(i)) v(i) = 0.0; });
}

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class Multigrid {
 public:
  using Octree = BlockOctree<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;
  using Poisson = AmrPoisson<Dim, Bits>;

  /// Build + upload the hierarchy from a finest octree (uniform coarsening), openness-
  /// free. `h0` is the finest spacing (every level shares it; a coarse leaf's higher
  /// `level` encodes its width).
  void build(const Octree& finest, double h0) {
    hmg_ = std::make_unique<AmrMultigrid<Dim, Bits>>();
    hmg_->build(finest, h0);
    buildFromHostMg();
  }

  /// Build with cut-cell openness `openFn(faceCentreWorld, axis) → [0,1]`, coarsened
  /// to **every** level by area-averaging (AmrMultigrid::setOpenness — the same
  /// coarsenOpenAvg the host uses): each level's face weight is α·A/d with α the
  /// coarsened aperture, so the coarse operators stay consistent cut-cell operators.
  template <class OpenFn>
  void build(const Octree& finest, double h0, OpenFn&& openFn, bool periodic = true,
             bool immersedWall = false) {
    hmg_ = std::make_unique<AmrMultigrid<Dim, Bits>>();
    hmg_->build(finest, h0);
    hmg_->setOpenness(std::forward<OpenFn>(openFn));
    hmg_->setPeriodic(periodic);          // non-periodic ⇒ homogeneous Dirichlet domain walls
    hmg_->setImmersedWall(immersedWall);  // true ⇒ velocity operator (solid faces = no-slip walls)
    buildFromHostMg();
  }

  /// Opt-in: use κ-weighted (fluid-fraction) restriction instead of the default plain
  /// volume-average. Experimental — validate with the comparison test before relying on it.
  void setKappaRestrict(bool on) { kappaRestrict_ = on; }

  /// Opt-in (default off): project the correction to mean-zero over fluid cells at every V-cycle
  /// level — the singular (periodic pure-Neumann) nullspace removal, mirroring flow
  /// CutcellMG::vcycle. Needed when the V-cycle is the MG-PCG preconditioner for a singular
  /// operator (otherwise the cycle drifts / amplifies a near-nullspace mode and the projection
  /// blows up under large transient divergence). The bit-exact-vs-host MG test keeps this OFF.
  void setRemoveMean(bool on) { removeMean_ = on; }

  /// Turn every level's operator into the Helmholtz form H = c0·I + cD·L (default c0=0,
  /// cD=1 ⇒ the pure Laplacian L). With c0=idiag, cD=−μ the hierarchy represents the
  /// momentum operator idiag·I − μ∇² and (being non-singular for c0≠0) is an effective
  /// V-cycle preconditioner for the momentum BiCGStab. The reaction c0 is held constant on
  /// every level (standard for a Helmholtz MG preconditioner); cD scales the coarsened L.
  void setHelmholtz(double c0, double cD) {
    for (auto& lv : levels_) {
      lv.op.c0 = c0;
      lv.op.cD = cD;
    }
  }

  std::size_t numLevels() const { return levels_.size(); }
  Index numLeaves(std::size_t L = 0) const { return levels_[L].n; }
  Code octreeCode(std::size_t L, Index i) const { return hmg_->op(L).octree().code(i); }
  View<double> x(std::size_t L = 0) { return levels_[L].x; }
  View<double> b(std::size_t L = 0) { return levels_[L].b; }
  const FvOp& op(std::size_t L = 0) const { return levels_[L].op; }
  View<Index> quadStart() const { return qStart_; }
  View<Index> quadSlot() const { return qSlot_; }
  View<double> quadCoef() const { return qCoef_; }

  /// One V-cycle on level `L` (default finest) of the *standard* (consistent
  /// conservative) operator, correction scheme.
  void vcycle(int pre = 2, int post = 2, int bottom = 40, double omega = 0.8, std::size_t L = 0) {
    Level& lv = levels_[L];
    View<const double> bc(lv.b);
    if (L + 1 == levels_.size()) {
      for (int s = 0; s < bottom; ++s) jacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
      if (removeMean_) removeMeanFv(lv.op, lv.x);
      return;
    }
    for (int s = 0; s < pre; ++s) jacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
    residualFv(lv.op, View<const double>(lv.x), bc, lv.res);
    Level& cl = levels_[L + 1];
    if (kappaRestrict_)
      restrictKappa(lv.childStart, lv.childIdx, View<const double>(lv.res),
                          View<const double>(lv.kappa), cl.b, cl.n);
    else
      restrictField(lv.childStart, lv.childIdx, View<const double>(lv.res), cl.b, cl.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(pre, post, bottom, omega, L + 1);
    prolongAdd(lv.c2p, View<const double>(cl.x), lv.x, lv.n);
    for (int s = 0; s < post; ++s) jacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
    if (removeMean_) removeMeanFv(lv.op, lv.x);
  }

  /// Solve L_quad u = rhs (the 2nd-order graded operator) by deferred correction:
  /// each outer step solves L_std u = rhs − (L_quad−L_std)u with the quadratic
  /// correction lagged, via `cyclesPerOuter` standard V-cycles. The finest `b`
  /// holds rhs on entry and is restored on return. Returns the final L_quad
  /// residual L2 norm.
  double solveQuad(int outer = 20, int cyclesPerOuter = 1, int pre = 2, int post = 2,
                   int bottom = 40, double omega = 0.8) {
    Level& lv = levels_[0];
    Kokkos::deep_copy(b0true_, lv.b);  // true rhs
    auto dq = dq_;
    auto b0 = b0true_;
    auto bb = lv.b;
    double r = 0.0;
    for (int o = 0; o < outer; ++o) {
      quadDelta(View<const Index>(qStart_), View<const Index>(qSlot_),
                      View<const double>(qCoef_), View<const double>(lv.x), dq_, lv.n);
      Kokkos::parallel_for(
          "amr::quad_rhsp", lv.n, KOKKOS_LAMBDA(const Index i) { bb(i) = b0(i) - dq(i); });
      for (int c = 0; c < cyclesPerOuter; ++c) vcycle(pre, post, bottom, omega, 0);
    }
    // r = || b0true − (L_std + quad) x ||
    residualFv(lv.op, View<const double>(lv.x), View<const double>(b0true_), lv.res);
    quadDelta(View<const Index>(qStart_), View<const Index>(qSlot_),
                    View<const double>(qCoef_), View<const double>(lv.x), dq_, lv.n);
    auto res = lv.res;
    double s = 0.0;
    Kokkos::parallel_reduce(
        "amr::quad_resnorm", lv.n,
        KOKKOS_LAMBDA(const Index i, double& acc) {
          double rr = res(i) - dq(i);
          acc += rr * rr;
        },
        s);
    r = std::sqrt(s);
    Kokkos::deep_copy(lv.b, b0true_);  // restore rhs
    return r;
  }

 private:
  struct Level {
    FvOp op;
    Index n = 0;
    View<double> x, b, res, tmp;
    View<double> kappa;                     // per-cell fluid-fraction weight (κ restriction)
    View<Index> c2p, childStart, childIdx;  // describe L → L+1 (unused on coarsest)
  };

  // Upload the device hierarchy from the host AmrMultigrid hmg_ (which already holds
  // the uniform-coarsened octrees + per-level operators with coarsened openness). Each
  // level's face CSR is built from hmg_->op(L), so the device operator == that host
  // operator bit-for-bit on every level.
  void buildFromHostMg() {
    const std::size_t nl = hmg_->numLevels();
    levels_.clear();
    levels_.resize(nl);
    for (std::size_t L = 0; L < nl; ++L) {
      const Poisson& ap = hmg_->op(L);
      const Octree& oct = ap.octree();
      Level& lv = levels_[L];
      lv.n = oct.numLeaves();
      buildFaceCsr(ap, oct, lv);
      // Per-cell κ weight for the optional κ-weighted restriction = mean face aperture
      // (1 fully open ⇒ reduces to plain average; <1 near cuts).
      std::vector<double> kap(static_cast<std::size_t>(lv.n));
      for (Index i = 0; i < lv.n; ++i) {
        double s = 0.0;
        for (int axis = 0; axis < Dim; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) s += ap.faceOpenness(i, axis, dir);
        kap[static_cast<std::size_t>(i)] = s / static_cast<double>(2 * Dim);
      }
      lv.kappa = toDevice(kap, "mg_kappa");
      lv.x = View<double>("mg_x", static_cast<std::size_t>(lv.n));
      lv.b = View<double>("mg_b", static_cast<std::size_t>(lv.n));
      lv.res = View<double>("mg_res", static_cast<std::size_t>(lv.n));
      lv.tmp = View<double>("mg_tmp", static_cast<std::size_t>(lv.n));
    }
    for (std::size_t L = 0; L + 1 < nl; ++L) {
      const Octree& f = hmg_->op(L).octree();
      const Octree& c = hmg_->op(L + 1).octree();
      const Index nf = f.numLeaves(), nc = c.numLeaves();
      std::vector<Index> c2p(static_cast<std::size_t>(nf));
      std::vector<Index> cnt(static_cast<std::size_t>(nc), 0);
      for (Index i = 0; i < nf; ++i) {
        Code parent = M::from_code(f.code(i)).ancestor(f.level(i) + 1).code();
        Index p = c.find(parent);
        c2p[static_cast<std::size_t>(i)] = p;
        if (p >= 0) ++cnt[static_cast<std::size_t>(p)];
      }
      std::vector<Index> start(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p)
        start[static_cast<std::size_t>(p) + 1] = start[static_cast<std::size_t>(p)] + cnt[static_cast<std::size_t>(p)];
      std::vector<Index> idx(static_cast<std::size_t>(nf));
      std::vector<Index> cur(start.begin(), start.end() - 1);
      for (Index i = 0; i < nf; ++i) {
        Index p = c2p[static_cast<std::size_t>(i)];
        if (p >= 0) idx[static_cast<std::size_t>(cur[static_cast<std::size_t>(p)]++)] = i;
      }
      levels_[L].c2p = toDevice(c2p, "mg_c2p");
      levels_[L].childStart = toDevice(start, "mg_cstart");
      levels_[L].childIdx = toDevice(idx, "mg_cidx");
    }
    // Quadratic coarse-fine correction CSR on the finest level (for solveQuad);
    // openness (if any) flows in through hmg_->op(0)'s aperture + coarseStar gating.
    buildQuadCsr(hmg_->op(0), hmg_->op(0).octree());
    const Index n0 = levels_[0].n;
    dq_ = View<double>("mg_dq", static_cast<std::size_t>(n0));
    b0true_ = View<double>("mg_b0", static_cast<std::size_t>(n0));
  }

  // Build the consistent face CSR (+ invVol/bcDiag) for one level by assembling it ON THE DEVICE
  // (D5/D2): upload this level's leaf set, then assembleFv reproduces AmrPoisson's exact
  // forEachFaceNeighbor enumeration + openness·A_f/d_f weights — bit-for-bit identical to the host
  // walk on OpenMP (validated in test_amr_assembly), but with no host CSR build / no
  // re-upload. The Helmholtz c0/cD of this level are preserved (assembleFv returns the pure-L
  // defaults; setHelmholtz / a prior value is re-applied), so a reassemble after the geometry moves
  // keeps the momentum-preconditioner form.
  void buildFaceCsr(const Poisson& ap, const Octree& t, Level& lv) {
    const double c0 = lv.op.c0, cD = lv.op.cD;
    BlockOctreeView<Dim, Bits> ov;
    ov.upload(t);
    lv.op = assembleFv<Dim, Bits>(ap, ov);
    lv.op.c0 = c0;
    lv.op.cD = cD;
  }

 public:
  /// Re-assemble every level's operator ON THE DEVICE from the (host) hierarchy's current geometry —
  /// the dynamic-AMR rebuild hook (D5/D6). After a moving boundary re-samples each level's openness on
  /// the host AmrPoisson (hmg_), this rebuilds all per-level FvOps on device with no host CSR walk and
  /// no round-trip, preserving each level's Helmholtz c0/cD. Topology (c2p/child maps) is unchanged.
  void reassembleOperators() {
    const std::size_t nl = hmg_->numLevels();
    for (std::size_t L = 0; L < nl && L < levels_.size(); ++L)
      buildFaceCsr(hmg_->op(L), hmg_->op(L).octree(), levels_[L]);
  }

 private:

  // Build the quadratic coarse-fine correction CSR: dq_i = Σ coef·u[slot] equals
  // (L_quad − L_std)u (AmrPoisson::coarseStar, expressed as a linear stencil).
  void buildQuadCsr(const Poisson& ap, const Octree& t) {
    const Index n = t.numLeaves();
    std::vector<std::vector<std::pair<Index, double>>> per(static_cast<std::size_t>(n));
    const double h0 = ap.h0();
    for (Index i = 0; i < n; ++i) {
      const unsigned Li = t.level(i);
      const double invV = 1.0 / ap.cellVolume(i);
      ap.forEachFaceNeighbor(i, [&](Index j, Real c, int axis, double a) {
        const unsigned Lj = t.level(j);
        if (Lj == Li) return;
        const Index coarse = (Lj > Li) ? j : i;
        const Index fine = (Lj > Li) ? i : j;
        const double scale = invV * (a * c) * ((Lj > Li) ? 1.0 : -1.0);
        addCoarseStarStencil(ap, t, per[static_cast<std::size_t>(i)], coarse, fine, axis, scale, h0);
      });
    }
    std::vector<Index> qs(static_cast<std::size_t>(n) + 1, 0);
    for (Index i = 0; i < n; ++i)
      qs[static_cast<std::size_t>(i) + 1] = qs[static_cast<std::size_t>(i)] + static_cast<Index>(per[static_cast<std::size_t>(i)].size());
    const Index nq = qs[static_cast<std::size_t>(n)];
    std::vector<Index> slot(static_cast<std::size_t>(nq));
    std::vector<double> coef(static_cast<std::size_t>(nq));
    Index k = 0;
    for (Index i = 0; i < n; ++i)
      for (auto& e : per[static_cast<std::size_t>(i)]) {
        slot[static_cast<std::size_t>(k)] = e.first;
        coef[static_cast<std::size_t>(k)] = e.second;
        ++k;
      }
    qStart_ = toDevice(qs, "mg_qstart");
    qSlot_ = toDevice(slot, "mg_qslot");
    qCoef_ = toDevice(coef, "mg_qcoef");
  }

  // coarseStar(coarse,fine,axis) − u[coarse] as a linear stencil over the coarse
  // cell + its tangential coarse neighbours, scaled, appended to `out`. Mirrors
  // AmrPoisson::coarseStar exactly (same gating, same coefficients).
  static void addCoarseStarStencil(const Poisson& ap, const Octree& t,
                                   std::vector<std::pair<Index, double>>& out, Index coarse,
                                   Index fine, int axis, double scale, double h0) {
    auto bc = t.bounds(coarse);
    auto bf = t.bounds(fine);
    const double H = ap.cellWidth(coarse);
    const double sc = static_cast<double>(Index(1) << t.level(coarse));
    const double sf = static_cast<double>(Index(1) << t.level(fine));
    for (int tt = 0; tt < Dim; ++tt) {
      if (tt == axis) continue;
      const double dt = ((static_cast<double>(bf[0][tt]) + 0.5 * sf) -
                         (static_cast<double>(bc[0][tt]) + 0.5 * sc)) * h0;
      Index cp = ap.periodicNeighbor(coarse, tt, +1);
      Index cm = ap.periodicNeighbor(coarse, tt, -1);
      if (cp < 0 || cm < 0) continue;
      if (t.level(cp) != t.level(coarse) || t.level(cm) != t.level(coarse)) continue;
      if (ap.faceOpenness(coarse, tt, +1) < 0.5 || ap.faceOpenness(coarse, tt, -1) < 0.5) continue;
      const double cUp = dt / (2.0 * H) + 0.5 * dt * dt / (H * H);
      const double cUm = -dt / (2.0 * H) + 0.5 * dt * dt / (H * H);
      const double cUc = -dt * dt / (H * H);
      out.emplace_back(coarse, scale * cUc);
      out.emplace_back(cp, scale * cUp);
      out.emplace_back(cm, scale * cUm);
    }
  }

  // Owns the host hierarchy + per-level operators (with coarsened openness). The
  // operators hold pointers into its octrees, so keep it stable (unique_ptr) for the
  // lifetime of this object.
  std::unique_ptr<AmrMultigrid<Dim, Bits>> hmg_;
  std::vector<Level> levels_;
  bool kappaRestrict_ = false;  // default: plain volume-average restriction
  bool removeMean_ = false;     // default off; per-level nullspace projection for singular MG-PCG
  View<Index> qStart_, qSlot_;  // finest-level quadratic correction CSR
  View<double> qCoef_;
  View<double> dq_, b0true_;  // finest-level deferred-correction scratch
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_MULTIGRID_HPP

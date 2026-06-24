// transport-core — device (Kokkos) geometric-multigrid V-cycle on a BlockOctree,
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
// as a second precomputed CSR (deviceQuadDelta). Openness (cut-cell) flows in for
// free: build with an AmrPoisson that has openness set and w_f carries it.
//
// Requires a Kokkos build + the morton checkout (TPX_HAVE_MORTON).
#ifndef TPX_AMR_DEVICE_MULTIGRID_HPP
#define TPX_AMR_DEVICE_MULTIGRID_HPP

#ifdef TPX_HAVE_MORTON

#include <cmath>
#include <utility>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/device_poisson.hpp"
#include "tpx/amr/poisson.hpp"
#include "tpx/common/view.hpp"

namespace tpx::amr {

/// Restrict: coarse(p) = mean over p's children (CSR fixed order ⇒ deterministic).
inline void deviceRestrict(View<const Index> childStart, View<const Index> childIdx,
                           View<const double> fine, View<double> coarse, Index nCoarse) {
  Kokkos::parallel_for(
      "amr::device_restrict", nCoarse, KOKKOS_LAMBDA(const Index p) {
        const Index a = childStart(p), z = childStart(p + 1);
        double s = 0.0;
        for (Index k = a; k < z; ++k) s += fine(childIdx(k));
        coarse(p) = (z > a) ? s / static_cast<double>(z - a) : 0.0;
      });
}

/// Prolong (piecewise-constant) + correct: fine(i) += coarse(c2p(i)).
inline void deviceProlongAdd(View<const Index> c2p, View<const double> coarse, View<double> fine,
                             Index nFine) {
  Kokkos::parallel_for(
      "amr::device_prolong", nFine, KOKKOS_LAMBDA(const Index i) {
        const Index p = c2p(i);
        if (p >= 0) fine(i) += coarse(p);
      });
}

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DeviceMultigrid {
 public:
  using Octree = BlockOctree<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;
  using Poisson = AmrPoisson<Dim, Bits>;

  /// Build + upload the hierarchy from a finest octree (uniform coarsening). `h0` is
  /// the finest spacing (every level shares it; a coarse leaf's higher `level`
  /// encodes its width). If `proto` is given its finest-level openness (cut-cell)
  /// is applied on level 0 (coarse levels stay openness-free for now).
  void build(const Octree& finest, double h0, const Poisson* proto = nullptr) {
    levels_.clear();
    std::vector<Octree> host;
    host.push_back(finest);
    for (;;) {
      Octree c = host.back();
      Index merged = c.coarsenIf([](Code, unsigned) { return true; });
      if (merged == 0 || c.numLeaves() == host.back().numLeaves()) break;
      host.push_back(c);
      if (c.numLeaves() == 1) break;
    }
    levels_.resize(host.size());
    for (std::size_t L = 0; L < host.size(); ++L) {
      Level& lv = levels_[L];
      lv.n = host[L].numLeaves();
      Poisson ap;
      ap.init(host[L], h0);
      if (proto && L == 0 && proto->hasOpenness()) ap.setOpennessRaw(proto->opennessRaw());
      buildFaceCsr(ap, host[L], lv);
      lv.x = View<double>("mg_x", static_cast<std::size_t>(lv.n));
      lv.b = View<double>("mg_b", static_cast<std::size_t>(lv.n));
      lv.res = View<double>("mg_res", static_cast<std::size_t>(lv.n));
      lv.tmp = View<double>("mg_tmp", static_cast<std::size_t>(lv.n));
    }
    // fine→coarse map + CSR coarse→children.
    for (std::size_t L = 0; L + 1 < host.size(); ++L) {
      const Octree& f = host[L];
      const Octree& c = host[L + 1];
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
    // Quadratic coarse-fine correction CSR on the finest level (for solveQuad).
    {
      Poisson ap;
      ap.init(host[0], h0);
      if (proto && proto->hasOpenness()) ap.setOpennessRaw(proto->opennessRaw());
      buildQuadCsr(ap, host[0]);
      const Index n0 = levels_[0].n;
      dq_ = View<double>("mg_dq", static_cast<std::size_t>(n0));
      b0true_ = View<double>("mg_b0", static_cast<std::size_t>(n0));
    }
  }

  std::size_t numLevels() const { return levels_.size(); }
  Index numLeaves(std::size_t L = 0) const { return levels_[L].n; }
  View<double> x(std::size_t L = 0) { return levels_[L].x; }
  View<double> b(std::size_t L = 0) { return levels_[L].b; }
  const DeviceFvOp& op(std::size_t L = 0) const { return levels_[L].op; }
  View<Index> quadStart() const { return qStart_; }
  View<Index> quadSlot() const { return qSlot_; }
  View<double> quadCoef() const { return qCoef_; }

  /// One V-cycle on level `L` (default finest) of the *standard* (consistent
  /// conservative) operator, correction scheme.
  void vcycle(int pre = 2, int post = 2, int bottom = 40, double omega = 0.8, std::size_t L = 0) {
    Level& lv = levels_[L];
    View<const double> bc(lv.b);
    if (L + 1 == levels_.size()) {
      for (int s = 0; s < bottom; ++s) deviceJacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
      return;
    }
    for (int s = 0; s < pre; ++s) deviceJacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
    deviceResidualFv(lv.op, View<const double>(lv.x), bc, lv.res);
    Level& cl = levels_[L + 1];
    deviceRestrict(lv.childStart, lv.childIdx, View<const double>(lv.res), cl.b, cl.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(pre, post, bottom, omega, L + 1);
    deviceProlongAdd(lv.c2p, View<const double>(cl.x), lv.x, lv.n);
    for (int s = 0; s < post; ++s) deviceJacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
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
      deviceQuadDelta(View<const Index>(qStart_), View<const Index>(qSlot_),
                      View<const double>(qCoef_), View<const double>(lv.x), dq_, lv.n);
      Kokkos::parallel_for(
          "amr::quad_rhsp", lv.n, KOKKOS_LAMBDA(const Index i) { bb(i) = b0(i) - dq(i); });
      for (int c = 0; c < cyclesPerOuter; ++c) vcycle(pre, post, bottom, omega, 0);
    }
    // r = || b0true − (L_std + quad) x ||
    deviceResidualFv(lv.op, View<const double>(lv.x), View<const double>(b0true_), lv.res);
    deviceQuadDelta(View<const Index>(qStart_), View<const Index>(qSlot_),
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
    DeviceFvOp op;
    Index n = 0;
    View<double> x, b, res, tmp;
    View<Index> c2p, childStart, childIdx;  // describe L → L+1 (unused on coarsest)
  };

  // Build the consistent face CSR (+ invVol) for one level from its AmrPoisson, in
  // the exact face order AmrPoisson::forEachFaceNeighbor emits ⇒ bit-exact to host.
  void buildFaceCsr(const Poisson& ap, const Octree& t, Level& lv) {
    const Index n = t.numLeaves();
    std::vector<Index> start(static_cast<std::size_t>(n) + 1, 0);
    for (Index i = 0; i < n; ++i) {
      Index cnt = 0;
      ap.forEachFaceNeighbor(i, [&](Index, Real, int, double) { ++cnt; });
      start[static_cast<std::size_t>(i) + 1] = start[static_cast<std::size_t>(i)] + cnt;
    }
    const Index nf = start[static_cast<std::size_t>(n)];
    std::vector<Index> nbr(static_cast<std::size_t>(nf));
    std::vector<double> w(static_cast<std::size_t>(nf));
    std::vector<double> invVol(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
      invVol[static_cast<std::size_t>(i)] = 1.0 / ap.cellVolume(i);
      Index k = start[static_cast<std::size_t>(i)];
      ap.forEachFaceNeighbor(i, [&](Index j, Real c, int, double a) {
        nbr[static_cast<std::size_t>(k)] = j;
        w[static_cast<std::size_t>(k)] = a * c;
        ++k;
      });
    }
    lv.op.n = n;
    lv.op.invVol = toDevice(invVol, "mg_invVol");
    lv.op.faceStart = toDevice(start, "mg_fstart");
    lv.op.faceNbr = toDevice(nbr, "mg_fnbr");
    lv.op.faceW = toDevice(w, "mg_fw");
  }

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

  std::vector<Level> levels_;
  View<Index> qStart_, qSlot_;  // finest-level quadratic correction CSR
  View<double> qCoef_;
  View<double> dq_, b0true_;  // finest-level deferred-correction scratch
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_MULTIGRID_HPP

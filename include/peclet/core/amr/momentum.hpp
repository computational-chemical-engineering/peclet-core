// core — the AMR momentum solve: the Galerkin momentum multigrid (MomentumMG) over the shared
// face-CSR solver layer.
//
// The assembled operator (MomentumOp), its device kernels (apply / residual / weighted Jacobi /
// multicolour Gauss–Seidel), the greedy graph colouring and the preconditioned BiCGStab
// (MomentumSolver) were LIFTED verbatim to peclet/core/solver/ on 2026-09-10 (suite/docs/
// QUALITY_PLAN.md G.2) — voro's mesh optimiser consumes them, so they are core infrastructure, not
// AMR. This header keeps every AMR spelling resolving (the using-declarations and the
// `MomentumSolver<Bits>` alias template below) and holds what IS octree-specific: MomentumMG,
// whose Galerkin hierarchy is the uniformly-coarsened octree.
//
// Requires a Kokkos build + the morton checkout (PECLET_CORE_HAVE_MORTON).
#ifndef PECLET_CORE_AMR_MOMENTUM_HPP
#define PECLET_CORE_AMR_MOMENTUM_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <cstddef>
#include <map>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/face_csr.hpp"   // shared host+device assembled-operator row kernels
#include "peclet/core/amr/multigrid.hpp"  // restrictField / prolongAdd transfer kernels
#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/coloring.hpp"
#include "peclet/core/solver/csr_bicgstab.hpp"
#include "peclet/core/solver/csr_operator.hpp"
#include "peclet/core/solver/vector_ops.hpp"

namespace peclet::core::amr {

using solver::applyMom;
using solver::bicgPUpdate;
using solver::Coloring;
using solver::dotPlain;
using solver::greedyColoring;
using solver::jacobiMom;
using solver::MomentumOp;
using solver::momView;
using solver::multicolorGSMom;
using solver::residualMom;

/// The BiCGStab solver never touched the octree: its `Bits` parameter was vestigial and is gone in
/// core. Kept here as an alias template so `MomentumSolver<Bits>` keeps compiling.
template <unsigned Bits = 21u>
using MomentumSolver = solver::MomentumSolver;

// ===========================================================================
// MomentumMG — Galerkin geometric multigrid for the momentum operator.
//
// The cut-cell momentum operator carries the ξ-polynomial Dirichlet overlay and its
// D_rescale row scaling, so a *rediscretised* coarse operator (the openness-Helmholtz
// attempt) mismatches it and makes a poor preconditioner. Instead the coarse operators are
// built by **Galerkin coarsening** A_c = R·A·P of the exact assembled fine CSR: R = volume
// average over a coarse cell's children, P = piecewise-constant injection (the same transfer
// pair the pressure MG uses). This is consistent with the fine operator by construction — it
// inherits the cut-cell stencil and row scaling, and a coarse cell whose children are all
// solid (identity rows) stays an identity row (ε-solid-on-coarse emerges for free). The
// hierarchy is the uniformly-coarsened octree; the smoother is jacobiMom, the residual
// restriction / correction prolongation are the shared restrictField / prolongAdd.
// Used as the momentum BiCGStab preconditioner ⇒ the iteration count stays ~flat with N.
// ===========================================================================
template <unsigned Bits = 21u>
class MomentumMG {
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
      if (merged == 0 || c.numLeaves() == octs_.back().numLeaves())
        break;
      octs_.push_back(c);
      if (c.numLeaves() == 1)
        break;
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
        // Covering-leaf c2p (see multigrid.hpp): == ancestor+find for merged children, correct
        // (identity) for root-level rows in mixed-depth ladders, block-alignment-independent.
        Index p = c.find(f.code(i));
        c2p[static_cast<std::size_t>(i)] = p;
        if (p >= 0)
          ++cnt[static_cast<std::size_t>(p)];
      }
      std::vector<Index> cstart(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p)
        cstart[static_cast<std::size_t>(p) + 1] =
            cstart[static_cast<std::size_t>(p)] + cnt[static_cast<std::size_t>(p)];
      std::vector<Index> cidx(static_cast<std::size_t>(nf));
      std::vector<Index> cur(cstart.begin(), cstart.end() - 1);
      for (Index i = 0; i < nf; ++i) {
        Index p = c2p[static_cast<std::size_t>(i)];
        if (p >= 0)
          cidx[static_cast<std::size_t>(cur[static_cast<std::size_t>(p)]++)] = i;
      }
      levels_[L].c2p = toDevice(c2p, "mmg_c2p");
      levels_[L].childStart = toDevice(cstart, "mmg_cstart");
      levels_[L].childIdx = toDevice(cidx, "mmg_cidx");

      // Galerkin A_c[p][q] = (1/n_ch[p]) Σ_{i child of p} ( A[i] entries mapped to parents ).
      std::vector<std::map<Index, double>> acc(static_cast<std::size_t>(nc));
      for (Index i = 0; i < nf; ++i) {
        Index p = c2p[static_cast<std::size_t>(i)];
        if (p < 0)
          continue;
        const double w = 1.0 / static_cast<double>(cnt[static_cast<std::size_t>(p)]);
        acc[static_cast<std::size_t>(p)][p] += w * hdiag[static_cast<std::size_t>(i)];
        for (Index k = hstart[static_cast<std::size_t>(i)];
             k < hstart[static_cast<std::size_t>(i) + 1]; ++k) {
          Index q = c2p[static_cast<std::size_t>(hnbr[static_cast<std::size_t>(k)])];
          if (q < 0)
            continue;
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

  /// Opt-in: use multicolour Gauss–Seidel as the smoother (per-level colouring built at build)
  /// instead of weighted Jacobi — ~2× better smoothing, fewer V-cycles. Default off (Jacobi).
  void setGaussSeidel(bool on) { useGS_ = on; }

  /// One V-cycle on level L solving A u = b (correction scheme).
  void vcycle(int pre = 2, int post = 2, int bottom = 30, double omega = 0.7, std::size_t L = 0) {
    Level& lv = levels_[L];
    View<const double> bc(lv.b);
    if (L + 1 == levels_.size()) {
      smooth(lv, bottom, omega);
      return;
    }
    smooth(lv, pre, omega);
    residualMom(lv.op, View<const double>(lv.x), bc, lv.res);
    Level& cl = levels_[L + 1];
    restrictField(lv.childStart, lv.childIdx, View<const double>(lv.res), cl.b, cl.op.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(pre, post, bottom, omega, L + 1);
    prolongAdd(lv.c2p, View<const double>(cl.x), lv.x, lv.op.n);
    smooth(lv, post, omega);
  }

  std::size_t numLevels() const { return levels_.size(); }
  Index numLeaves(std::size_t L = 0) const { return levels_[L].op.n; }
  View<double> x(std::size_t L = 0) { return levels_[L].x; }
  View<double> b(std::size_t L = 0) { return levels_[L].b; }
  const MomentumOp& op(std::size_t L = 0) const { return levels_[L].op; }

 private:
  struct Level {
    MomentumOp op;
    View<double> x, b, res, tmp;
    View<Index> c2p, childStart, childIdx;
    Coloring col;
  };
  void smooth(Level& lv, int sweeps, double omega) {
    View<const double> bc(lv.b);
    if (useGS_)
      // Multicolour GS is stable undamped on this diagonally-dominant operator (ρ/dt + 6μ + FOU >
      // 0); the passed omega is Jacobi's damping limit (~0.7) and would needlessly weaken GS, so
      // use 1.0.
      for (int s = 0; s < sweeps; ++s)
        multicolorGSMom(lv.op, lv.x, bc, lv.col, 1.0);
    else
      for (int s = 0; s < sweeps; ++s)
        jacobiMom(lv.op, lv.x, bc, lv.tmp, omega);
  }
  void uploadLevel(std::size_t L, const std::vector<double>& diag, const std::vector<Index>& start,
                   const std::vector<Index>& nbr, const std::vector<double>& coef) {
    Level& lv = levels_[L];
    lv.op.n = static_cast<Index>(diag.size());
    lv.op.diag = toDevice(diag, "mmg_diag");
    lv.op.faceStart = toDevice(start, "mmg_start");
    lv.op.faceNbr = toDevice(nbr, "mmg_nbr");
    lv.op.faceCoef = toDevice(coef, "mmg_coef");
    lv.col = greedyColoring(start, nbr, lv.op.n);
  }
  std::vector<Octree> octs_;
  std::vector<Level> levels_;
  bool useGS_ = false;  // multicolour Gauss–Seidel smoother (opt-in; default weighted Jacobi)
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_MOMENTUM_HPP

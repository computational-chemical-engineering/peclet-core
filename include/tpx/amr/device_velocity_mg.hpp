// transport-core — device (Kokkos) REDISCRETIZED velocity multigrid for the AMR momentum solve.
//
// The rediscretized counterpart of the Galerkin MomentumMG, mirroring sdflow's VelocityMG
// (mac_velocity_mg.hpp) on the octree. The fine level is the *sharp* cut-cell operator (the
// assembled FaceCsrOp = MomentumOp the BiCGStab matvec uses); the coarse levels are a
// **staircase rediscretization** of the geometry rather than a Galerkin R·A·P of the fine
// operator:
//   * coarsen the cell fluid-fraction κ to each level (average of children);
//   * classify κ < ½ → solid (identity row, pinned to 0 by the smoother), else fluid;
//   * a fluid cell is the per-axis const-coeff Helmholtz  diag = ρ/dt + 6μ/H²,  off = −μ/H²  to
//     each of its 6 (periodic) face neighbours (H = level cell width) — a face into a classified
//     solid cell is a no-slip wall implicitly (the neighbour is pinned to 0). This is
//     ε-solid-on-coarse for free, exactly as sdflow's buildVelocityStaircase.
// Transfers: average restriction + masked piecewise-constant prolongation (no correction into a
// solid fine cell). Smoother: weighted Jacobi (deviceJacobiMom) — P5 swaps in multicolor-GS.
//
// Used as the momentum BiCGStab preconditioner exactly like MomentumMG; selectable so the
// two coarse-operator strategies can be benchmarked head-to-head on the AMR. The implicit-FOU
// advection on coarse levels (mirror buildUpwindCoarse) is a follow-up — the viscous staircase is
// the diffusion preconditioner. Requires a Kokkos build + the morton checkout (TPX_HAVE_MORTON).
#ifndef TPX_AMR_DEVICE_VELOCITY_MG_HPP
#define TPX_AMR_DEVICE_VELOCITY_MG_HPP

#ifdef TPX_HAVE_MORTON

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "tpx/amr/device_momentum.hpp"
#include "tpx/amr/device_multigrid.hpp"
#include "tpx/amr/poisson.hpp"
#include "tpx/common/view.hpp"

namespace tpx::amr {

template <unsigned Bits = 21u>
class VelocityMG {
 public:
  using Octree = BlockOctree<3, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;
  using Poisson = AmrPoisson<3, Bits>;

  /// Build the rediscretized hierarchy. `fineOp` is the sharp cut-cell operator (level 0); `kappa`
  /// / `fluid` are the per-fine-cell fluid fraction + flag (AmrCutCell::kappa / isFluid). `idiag` =
  /// ρ/dt, `mu` = μ. The fine octree is coarsened uniformly (one local octree per level), as in
  /// AmrMultigrid. The fine operator is referenced (not copied); rebuild it (e.g. with the FOU)
  /// before each solve and the V-cycle picks it up.
  /// `minCoarse` clamps the coarsening depth (sdflow's pore-scale cap): levels are dropped once a
  /// level would have fewer than `minCoarse` cells, so the coarsest grid still resolves the
  /// immersed feature. Coarsening below the feature scale makes a small object vanish from the
  /// staircase classification, leaving an inconsistent coarse operator that diverges (deep
  /// coarsening only sets the rate, not the answer — the fine smoother + exclude carry it).
  void build(const Octree& finest, double h0, double idiag, double mu, const MomentumOp& fineOp,
             const std::vector<double>& kappa, const std::vector<char>& fluid,
             const std::vector<char>& cut, Index minCoarse = 256) {
    hmg_ = std::make_unique<AmrMultigrid<3, Bits>>();
    hmg_->build(finest, h0);  // octree hierarchy + per-level AmrPoisson (periodicNeighbor etc.)
    std::size_t nl = hmg_->numLevels();
    while (nl > 1 && hmg_->op(nl - 1).octree().numLeaves() < minCoarse) --nl;  // pore-scale cap
    levels_.clear();
    levels_.resize(nl);

    // Per-level κ (level 0 = fine; coarser = average of children) for the staircase classification.
    std::vector<std::vector<double>> kap(nl);
    kap[0] = kappa;
    for (std::size_t L = 0; L + 1 < nl; ++L) {
      const Octree& f = hmg_->op(L).octree();
      const Octree& c = hmg_->op(L + 1).octree();
      const Index nf = f.numLeaves(), nc = c.numLeaves();
      std::vector<double> ks(static_cast<std::size_t>(nc), 0.0), kn(static_cast<std::size_t>(nc), 0.0);
      for (Index i = 0; i < nf; ++i) {
        Code par = M::from_code(f.code(i)).ancestor(f.level(i) + 1).code();
        Index p = c.find(par);
        if (p >= 0) {
          ks[static_cast<std::size_t>(p)] += kap[L][static_cast<std::size_t>(i)];
          kn[static_cast<std::size_t>(p)] += 1.0;
        }
      }
      for (Index p = 0; p < nc; ++p)
        ks[static_cast<std::size_t>(p)] /= (kn[static_cast<std::size_t>(p)] > 0 ? kn[static_cast<std::size_t>(p)] : 1.0);
      kap[L + 1] = std::move(ks);
    }

    // Level 0: the sharp fine operator + the clean-fluid EXCLUDE mask (= cut OR solid). This is
    // sdflow VelocityMG's Phase-3 fix (doc/velocity_mg_plan.md): the cut cells' D_rescale-scaled
    // residuals and the solid cells are excluded from the coarse defect (zeroed before restriction
    // + skipped in prolongation), so the inconsistent sharp-IBM rows never reach the coarse grid —
    // the fine smoother owns the cut band, the coarse grid solves the clean interior. Without it the
    // staircase V-cycle diverges at large dt.
    levels_[0].op = fineOp;
    {
      std::vector<char> excl(fluid.size());
      for (std::size_t i = 0; i < fluid.size(); ++i) excl[i] = (!fluid[i] || cut[i]) ? 1 : 0;
      levels_[0].solid = toDevice(excl, "vmg_excl0");
      allocScratch(levels_[0], static_cast<Index>(fluid.size()));
      // Colour the sharp fine operator (its face graph) for the optional GS smoother — copy the
      // face CSR off the device once.
      std::vector<Index> hs(static_cast<std::size_t>(fineOp.n) + 1), hn(fineOp.faceNbr.extent(0));
      auto ms = Kokkos::create_mirror_view(fineOp.faceStart);
      auto mn = Kokkos::create_mirror_view(fineOp.faceNbr);
      Kokkos::deep_copy(ms, fineOp.faceStart);
      Kokkos::deep_copy(mn, fineOp.faceNbr);
      for (std::size_t i = 0; i < hs.size(); ++i) hs[i] = ms(static_cast<Index>(i));
      for (std::size_t i = 0; i < hn.size(); ++i) hn[i] = mn(static_cast<Index>(i));
      levels_[0].col = greedyColoring(hs, hn, fineOp.n);
    }
    // Coarse levels: rediscretized staircase operator + classified solid mask.
    for (std::size_t L = 1; L < nl; ++L) buildStaircase(L, idiag, mu, kap[L]);

    // Transfer maps L → L+1 (child→parent + child CSR), as in MomentumMG.
    for (std::size_t L = 0; L + 1 < nl; ++L) buildTransfer(L);
  }

  std::size_t numLevels() const { return levels_.size(); }
  Index numLeaves(std::size_t L = 0) const { return levels_[L].op.n; }
  View<double> x(std::size_t L = 0) { return levels_[L].x; }
  View<double> b(std::size_t L = 0) { return levels_[L].b; }
  /// Re-point level 0 at the (possibly FOU-updated) fine operator before a solve.
  void setFineOp(const MomentumOp& fineOp) { levels_[0].op = fineOp; }
  /// Opt-in: use multicolour Gauss–Seidel as the smoother (per-level colouring) instead of Jacobi —
  /// the strong fine smoother that "owns the cut band" (sdflow uses RB-GS), markedly improving the
  /// staircase at high resolution. Default off (Jacobi).
  void setGaussSeidel(bool on) { useGS_ = on; }

  /// One V-cycle (correction scheme) solving the fine operator. Average restriction, masked
  /// piecewise-constant prolongation, clean-fluid residual exclude; Jacobi or multicolour-GS smoother.
  void vcycle(int pre = 2, int post = 2, int bottom = 30, double omega = 0.7, std::size_t L = 0) {
    Level& lv = levels_[L];
    View<const double> bc(lv.b);
    if (L + 1 == levels_.size()) {
      smooth(lv, bottom, omega);
      return;
    }
    smooth(lv, pre, omega);
    deviceResidualMom(lv.op, View<const double>(lv.x), bc, lv.res);
    // Clean-fluid exclude: zero the residual at cut/solid cells before restriction so the
    // inconsistent sharp-IBM cut-cell residuals never pollute the coarse defect (the fix for the
    // large-dt staircase divergence — sdflow VelocityMG mg_mul_mask).
    deviceZeroMasked(lv.res, View<const char>(lv.solid), lv.op.n);
    Level& cl = levels_[L + 1];
    deviceRestrict(lv.childStart, lv.childIdx, View<const double>(lv.res), cl.b, cl.op.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(pre, post, bottom, omega, L + 1);
    deviceProlongAddMasked(lv.c2p, View<const double>(cl.x), View<const char>(lv.solid), lv.x, lv.op.n);
    smooth(lv, post, omega);
  }

 private:
  struct Level {
    MomentumOp op;
    View<char> solid;
    View<double> x, b, res, tmp;
    View<Index> c2p, childStart, childIdx;
    Coloring col;
  };
  void smooth(Level& lv, int sweeps, double omega) {
    View<const double> bc(lv.b);
    if (useGS_)
      // Undamped GS (omega=1.0): stable on the staircase Helmholtz diagonal, and the passed omega is
      // Jacobi's damping limit (~0.7) which would needlessly weaken the GS smoothing.
      for (int s = 0; s < sweeps; ++s) deviceMulticolorGSMom(lv.op, lv.x, bc, lv.col, 1.0);
    else
      for (int s = 0; s < sweeps; ++s) deviceJacobiMom(lv.op, lv.x, bc, lv.tmp, omega);
  }

  void allocScratch(Level& lv, Index n) {
    lv.x = View<double>("vmg_x", static_cast<std::size_t>(n));
    lv.b = View<double>("vmg_b", static_cast<std::size_t>(n));
    lv.res = View<double>("vmg_res", static_cast<std::size_t>(n));
    lv.tmp = View<double>("vmg_tmp", static_cast<std::size_t>(n));
  }

  // Rediscretized staircase operator on coarse level L from the coarsened κ.
  void buildStaircase(std::size_t L, double idiag, double mu, const std::vector<double>& kap) {
    const Poisson& ap = hmg_->op(L);
    const Octree& oct = ap.octree();
    const Index n = oct.numLeaves();
    // Shift floor on the coarse reaction: at large dt (idiag→0) a small immersed object can vanish
    // from the binary κ classification on the coarsest levels, leaving a singular periodic
    // Laplacian whose bottom solve diverges. Flooring idiag to μ/L² (the slowest diffusion mode)
    // keeps every coarse level non-singular (Galerkin avoids this by inheriting the sharp operator).
    const double Ldom =
        ap.h0() * static_cast<double>(oct.brick()[0] * (Index(1) << oct.lmax()));
    const double id = std::max(idiag, mu / (Ldom * Ldom));
    std::vector<double> diag(static_cast<std::size_t>(n), 1.0);
    std::vector<char> solid(static_cast<std::size_t>(n), 1);
    std::vector<Index> start(static_cast<std::size_t>(n) + 1, 0);
    std::vector<std::vector<std::pair<Index, double>>> rows(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
      if (kap[static_cast<std::size_t>(i)] < 0.5) continue;  // classified solid: identity row
      solid[static_cast<std::size_t>(i)] = 0;
      const double H = ap.cellWidth(i);   // h0·2^L (uniform per level)
      const double coef = mu / (H * H);   // μ·area/dist/V = μ/H² (isotropic coarse cell)
      double dsum = id;
      for (int axis = 0; axis < 3; ++axis)
        for (int dir = -1; dir <= 1; dir += 2) {
          Index j = ap.periodicNeighbor(i, axis, dir);  // periodic wrap; same-level neighbour
          if (j < 0) continue;                            // domain boundary (non-periodic): wall
          rows[static_cast<std::size_t>(i)].emplace_back(j, -coef);
          dsum += coef;  // every face counts toward the diagonal (a wall to a solid nbr too)
        }
      diag[static_cast<std::size_t>(i)] = dsum;
    }
    for (Index i = 0; i < n; ++i)
      start[static_cast<std::size_t>(i) + 1] =
          start[static_cast<std::size_t>(i)] + static_cast<Index>(rows[static_cast<std::size_t>(i)].size());
    const Index nnz = start[static_cast<std::size_t>(n)];
    std::vector<Index> nbr(static_cast<std::size_t>(nnz));
    std::vector<double> coef(static_cast<std::size_t>(nnz));
    for (Index i = 0; i < n; ++i) {
      Index k = start[static_cast<std::size_t>(i)];
      for (auto& e : rows[static_cast<std::size_t>(i)]) {
        nbr[static_cast<std::size_t>(k)] = e.first;
        coef[static_cast<std::size_t>(k)] = e.second;
        ++k;
      }
    }
    Level& lv = levels_[L];
    lv.op.n = n;
    lv.op.diag = toDevice(diag, "vmg_diag");
    lv.op.faceStart = toDevice(start, "vmg_fstart");
    lv.op.faceNbr = toDevice(nbr, "vmg_fnbr");
    lv.op.faceCoef = toDevice(coef, "vmg_fcoef");
    lv.solid = toDevice(solid, "vmg_solid");
    lv.col = greedyColoring(start, nbr, n);  // for the optional GS smoother
    allocScratch(lv, n);
  }

  void buildTransfer(std::size_t L) {
    const Octree& f = hmg_->op(L).octree();
    const Octree& c = hmg_->op(L + 1).octree();
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
    levels_[L].c2p = toDevice(c2p, "vmg_c2p");
    levels_[L].childStart = toDevice(cstart, "vmg_cstart");
    levels_[L].childIdx = toDevice(cidx, "vmg_cidx");
  }

  std::unique_ptr<AmrMultigrid<3, Bits>> hmg_;
  std::vector<Level> levels_;
  bool useGS_ = false;  // multicolour Gauss–Seidel smoother (opt-in; default weighted Jacobi)
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_VELOCITY_MG_HPP

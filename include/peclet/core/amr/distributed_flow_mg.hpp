// core — distributed device openness multigrid for the AMR flow pressure (rung 3 of
// docs/amr_distributed_flow.md): the distributed counterpart of Multigrid (multigrid.hpp),
// i.e. the aperture-path pressure hierarchy AmrFlow::presMG_ runs — graded octree, cut-cell
// openness area-averaged down the ladder, Jacobi smoothing, per-level nullspace projection —
// with every level a ghost-slot face CSR + its own LeafHalo.
//
// Structure (all proven pieces composed):
//   * Level ladder: per-rank coarsenIf of the LOCAL octree (same as AmrMultigrid::build),
//     each level a COPY of the flow's DistributedOctree with the coarsened local() — the copy
//     carries the decomposition, so a rebalanced (weighted-ORB) flow octree keeps correct
//     owner lookups. Ranks reaching their root brick early PAD to the global max level count
//     (Allreduce MAX) by repeating the root brick: the padded levels' transfers are identity
//     (the covering c2p maps a leaf to itself) and their smoothing exactly mirrors what the
//     whole-domain COMM_SELF ladder does in its already-coarse regions (the
//     GradedDistributedMultigrid argument) — and the level counts must match anyway for the
//     per-level halo point-to-points to pair up.
//   * Per level: an AmrPoisson with the LeafHalo resolver seam (setResolver/setGhosts/
//     setFrameShift — probes that exit the block resolve to ghost slots; world-coordinate
//     evaluations in the GLOBAL frame), built to the miss-collect fixpoint, then host
//     assembleFv → device FvOp whose faceNbr may reference the ghost tail.
//   * Openness: level 0 α from the world-coord openFn (ghost rows sampled locally — the same
//     world points as the owner, bit-identical); level L>0 LOCAL α by the exact
//     AmrMultigrid::coarsenOpenness child-face averaging (children of a local coarse cell are
//     always local, visited in the same relative Z-order as COMM_SELF ⇒ bit-identical), and
//     GHOST α rows exchanged from the owner once per build (exact by construction — never
//     re-derived).
//   * V-cycle: jacobiFv with a ghost refresh before every sweep, local restrict/prolong
//     (parents never cross blocks), Allreduce'd volume-weighted mean removal (removeMean).
//
// Bit-exactness: at np=1 every probe resolves locally (zero ghosts) and the whole cycle is
// the single-rank Multigrid arithmetic verbatim. Across ranks the smoother/transfers are
// order-independent ⇒ WORLD==SELF bit-exact with removeMean OFF; the mean removal (a global
// reduction) and PCG dots are np-invariant only to reduction order ⇒ tolerance, the suite's
// Krylov contract.
//
// Kokkos + MPI header (include in device TUs; the AmrFlow oracle stays single-rank).
#ifndef PECLET_CORE_AMR_DISTRIBUTED_FLOW_MG_HPP
#define PECLET_CORE_AMR_DISTRIBUTED_FLOW_MG_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>
#include <memory>
#include <vector>

#include "peclet/core/common/view.hpp"

#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/fv_op.hpp"
#include "peclet/core/amr/leaf_halo.hpp"
#include "peclet/core/amr/multigrid.hpp"  // restrictField / prolongAdd (shared transfer kernels)
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/mpi.hpp"

namespace peclet::core::amr {

/// removeMeanFv with the two mean sums folded across ranks (identical kernels; the Allreduce
/// sits between the reduce and the subtract). np=1 == removeMeanFv bit-for-bit.
inline void removeMeanFvDist(const FvOp& op, View<double> u, MPI_Comm comm) {
  auto invVol = op.invVol;
  auto fs = op.faceStart;
  auto fw = op.faceW;
  auto bc = op.bcDiag;
  double sx = 0.0, sv = 0.0;
  Kokkos::parallel_reduce(
      "amr::fv_rmean", op.n,
      KOKKOS_LAMBDA(const Index i, double& an, double& ad) {
        double d = bc(i);
        for (Index k = fs(i); k < fs(i + 1); ++k)
          d += fw(k);
        if (d > 1e-30) {
          an += u(i) / invVol(i);
          ad += 1.0 / invVol(i);
        }
      },
      sx, sv);
  double loc[2] = {sx, sv}, glob[2] = {0.0, 0.0};
  MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, comm);
  if (glob[1] <= 0.0)
    return;
  const double m = glob[0] / glob[1];
  Kokkos::parallel_for(
      "amr::fv_rmean_sub", op.n, KOKKOS_LAMBDA(const Index i) {
        double d = bc(i);
        for (Index k = fs(i); k < fs(i + 1); ++k)
          d += fw(k);
        if (d > 1e-30)
          u(i) -= m;
      });
}

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DistributedFlowMultigrid {
 public:
  using DO = DistributedOctree<Dim, Bits>;
  using Octree = typename DO::Octree;
  using Poisson = AmrPoisson<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;

  /// Build with cut-cell openness `openFn(faceCentreWorld, axis) → [0,1]` on the flow's
  /// distributed octree (graded + cross-block 2:1 balanced). Collective.
  ///
  /// `shared0` (optional): a FINALIZED LeafHalo whose registry is a superset of level 0's
  /// face probes — the distributed AmrFlow passes its ±2 flow registry so the level-0
  /// operator's ghost columns index the SAME extended layout as the flow's cell vectors
  /// (φ, the overlay chains, the PCG/BiCGStab scratch): one layout, no cross-indexing.
  /// Level-0 probes (±1 face reach incl. finer sub-neighbours) are a subset of the flow's
  /// forEachFaceFull discovery, so every probe resolves from the frozen registry.
  template <class OpenFn>
  void build(const DO& finest, double h0, OpenFn&& openFn,
             const LeafHalo<Dim, Bits>* shared0 = nullptr) {
    buildImpl(finest, h0, shared0);
    // Openness ladder: finest level directly from the world-coord openFn (local + ghost rows,
    // both exact); coarser levels by the exact single-rank child-face averaging for local rows
    // + a one-time owner exchange for ghost rows.
    levels_[0]->ap.buildOpenness(openFn);
    for (std::size_t L = 0; L + 1 < levels_.size(); ++L)
      coarsenOpennessTo(L);
    finishOps();
  }

  void setRemoveMean(bool on) { removeMean_ = on; }

  std::size_t numLevels() const { return levels_.size(); }
  Index numLeaves(std::size_t L = 0) const { return levels_[L]->n; }
  Index extendedSize(std::size_t L = 0) const { return levels_[L]->nExt; }
  View<double> x(std::size_t L = 0) { return levels_[L]->x; }
  View<double> b(std::size_t L = 0) { return levels_[L]->b; }
  const FvOp& op(std::size_t L = 0) const { return levels_[L]->op; }
  MPI_Comm comm() const { return comm_; }
  const Poisson& poisson(std::size_t L = 0) const { return levels_[L]->ap; }
  const LeafHalo<Dim, Bits>& halo(std::size_t L = 0) const { return *levels_[L]->hp; }

  /// Refresh the ghost tail of a level-L vector (the PCG matvec hook uses level 0).
  void sync(std::size_t L, View<double> v) const { levels_[L]->ex.exchange(v); }

  /// One V-cycle on level L (correction scheme), the distributed mirror of
  /// Multigrid::vcycle: ghost refresh before every Jacobi sweep / residual, local transfers,
  /// Allreduce'd mean removal.
  void vcycle(int pre = 2, int post = 2, int bottom = 40, double omega = 0.8, std::size_t L = 0) {
    Level& lv = *levels_[L];
    View<const double> bc(lv.b);
    if (L + 1 == levels_.size()) {
      for (int s = 0; s < bottom; ++s) {
        lv.ex.exchange(lv.x);
        jacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
      }
      if (removeMean_)
        removeMeanFvDist(lv.op, lv.x, comm_);
      return;
    }
    for (int s = 0; s < pre; ++s) {
      lv.ex.exchange(lv.x);
      jacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
    }
    lv.ex.exchange(lv.x);
    residualFv(lv.op, View<const double>(lv.x), bc, lv.res);
    Level& cl = *levels_[L + 1];
    restrictField(lv.childStart, lv.childIdx, View<const double>(lv.res), cl.b, cl.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(pre, post, bottom, omega, L + 1);
    prolongAdd(lv.c2p, View<const double>(cl.x), lv.x, lv.n);
    for (int s = 0; s < post; ++s) {
      lv.ex.exchange(lv.x);
      jacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
    }
    if (removeMean_)
      removeMeanFvDist(lv.op, lv.x, comm_);
  }

 private:
  struct Level {
    DO d;  // coarsened copy of the flow octree (carries the decomposition)
    Poisson ap;
    LeafHalo<Dim, Bits> halo;                  // own registry (unused when hp aliases shared0)
    const LeafHalo<Dim, Bits>* hp = nullptr;   // the registry in force (own or the shared one)
    LeafHaloExchange ex;
    FvOp op;
    Index n = 0, nExt = 0;
    View<double> x, b, res, tmp;  // x/b sized nExt (PCG deep_copies match); res/tmp local
    View<Index> c2p, childStart, childIdx;
    std::vector<Index> c2pHost;  // kept for the openness coarsening
  };

  void buildImpl(const DO& finest, double h0, const LeafHalo<Dim, Bits>* shared0) {
    comm_ = finest.comm();
    h0_ = h0;
    levels_.clear();
    // Ladder of coarsened copies of the SAME distributed octree (decomposition preserved).
    {
      auto l0 = std::make_unique<Level>();
      l0->d = finest;
      levels_.push_back(std::move(l0));
      for (;;) {
        Octree c = levels_.back()->d.local();
        const Index before = c.numLeaves();
        const Index merged = c.coarsenIf([](Code, unsigned) { return true; });
        if (merged == 0 || c.numLeaves() == before)
          break;
        auto lv = std::make_unique<Level>();
        lv->d = levels_.back()->d;
        lv->d.local() = std::move(c);
        levels_.push_back(std::move(lv));
        if (levels_.back()->d.local().numLeaves() == 1)
          break;
      }
      // Pad to the global max level count (identity root-brick levels): the per-level halo
      // point-to-points and build collectives must pair up across ranks, and the extra
      // smoothing exactly mirrors COMM_SELF's already-coarse regions.
      int nl = static_cast<int>(levels_.size()), gnl = nl;
      MPI_Allreduce(&nl, &gnl, 1, MPI_INT, MPI_MAX, comm_);
      while (static_cast<int>(levels_.size()) < gnl) {
        auto lv = std::make_unique<Level>();
        lv->d = levels_.back()->d;
        levels_.push_back(std::move(lv));
      }
    }
    // Per level: seam install + discovery fixpoint + topology freeze. Collective per level —
    // every rank walks its levels in the same order (counts padded), so the resolveMisses /
    // coverLevels rounds stay matched.
    const Vec<Dim> gorigin = finest.globalGeometry().origin;
    std::array<long, Dim> shift{};
    for (int a = 0; a < Dim; ++a)
      shift[a] = finest.blockFineOrigin()[a];
    shift_ = shift;
    bool first = true;
    for (auto& lvp : levels_) {
      Level& lv = *lvp;
      lv.n = lv.d.local().numLeaves();
      lv.ap.init(lv.d.local(), h0_);
      lv.ap.setOrigin(gorigin);
      lv.ap.setFrameShift(shift);
      // Install the ghost metadata of a registry into ap (level/lo lookups during the walks).
      auto installGhosts = [&](const LeafHalo<Dim, Bits>& h) {
        std::vector<std::array<long, Dim>> glo(static_cast<std::size_t>(h.numGhosts()));
        std::vector<unsigned> glv(static_cast<std::size_t>(h.numGhosts()));
        for (Index g = 0; g < h.numGhosts(); ++g) {
          for (int a = 0; a < Dim; ++a)
            glo[static_cast<std::size_t>(g)][a] =
                static_cast<long>(h.ghostCoord(g)[a]) - shift[a];
          glv[static_cast<std::size_t>(g)] = static_cast<unsigned>(h.level(h.numLocal() + g));
        }
        lv.ap.setGhosts(std::move(glo), std::move(glv));
      };
      if (first && shared0) {
        // Level 0 on the flow's frozen ±2 registry: same extended layout as the flow's cell
        // vectors; every ±1 face probe is already cached ⇒ const lookups, no discovery.
        lv.hp = shared0;
        lv.ap.setResolver([shared0, shift](const std::array<long, Dim>& p) -> Index {
          std::array<long, Dim> g = p;
          for (int a = 0; a < Dim; ++a)
            g[a] += shift[a];
          return shared0->lookupGlobal(g);
        });
        installGhosts(*shared0);
      } else {
        lv.hp = &lv.halo;
        lv.halo.init(lv.d);
        LeafHalo<Dim, Bits>* hp = &lv.halo;
        lv.ap.setResolver([hp, shift](const std::array<long, Dim>& p) -> Index {
          std::array<long, Dim> g = p;
          for (int a = 0; a < Dim; ++a)
            g[a] += shift[a];
          return hp->resolveGlobal(g);
        });
        // Ghosts must be (re-)installed into ap at the TOP of every fixpoint round: a probe
        // that resolved in an earlier round returns its ghost slot immediately, and probeSlot
        // then reads ap.levelOf(slot) — which must already cover it (newly-PENDING coords are
        // fine: they return kPending and are skipped until the next round).
        for (;;) {
          installGhosts(lv.halo);
          for (Index i = 0; i < lv.n; ++i)
            lv.ap.forEachFaceNeighbor(i, [](Index, Real, int, double) {});
          if (lv.halo.resolveMisses() == 0)
            break;
        }
        lv.halo.finalize();
      }
      lv.nExt = lv.hp->extendedSize();
      lv.ex.init(*lv.hp);
      first = false;
    }
    // Local covering-leaf transfers (parents never cross blocks); identity on padded levels.
    for (std::size_t L = 0; L + 1 < levels_.size(); ++L) {
      const Octree& f = levels_[L]->d.local();
      const Octree& c = levels_[L + 1]->d.local();
      const Index nf = f.numLeaves(), nc = c.numLeaves();
      std::vector<Index>& c2p = levels_[L]->c2pHost;
      c2p.assign(static_cast<std::size_t>(nf), -1);
      std::vector<Index> cnt(static_cast<std::size_t>(nc), 0);
      for (Index i = 0; i < nf; ++i) {
        const Index p = c.find(f.code(i));  // covering construction (== single-rank Multigrid)
        c2p[static_cast<std::size_t>(i)] = p;
        if (p >= 0)
          ++cnt[static_cast<std::size_t>(p)];
      }
      std::vector<Index> start(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p)
        start[static_cast<std::size_t>(p) + 1] =
            start[static_cast<std::size_t>(p)] + cnt[static_cast<std::size_t>(p)];
      std::vector<Index> idx(static_cast<std::size_t>(start[static_cast<std::size_t>(nc)]));
      std::vector<Index> cur(start.begin(), start.end() - 1);
      for (Index i = 0; i < nf; ++i) {  // fine order ⇒ deterministic restrict accumulation
        const Index p = c2p[static_cast<std::size_t>(i)];
        if (p >= 0)
          idx[static_cast<std::size_t>(cur[static_cast<std::size_t>(p)]++)] = i;
      }
      levels_[L]->c2p = toDevice(c2p, "dfmg_c2p");
      levels_[L]->childStart = toDevice(start, "dfmg_cstart");
      levels_[L]->childIdx = toDevice(idx, "dfmg_cidx");
    }
  }

  // Area-average level-L face openness onto level L+1 — the EXACT AmrMultigrid::coarsenOpenness
  // arithmetic for the local rows (children of a local coarse cell are local, summed in local
  // Z-order == COMM_SELF's relative order ⇒ bit-identical), then ghost α rows exchanged from
  // the owners through the level-(L+1) halo (kFaces one-time host exchanges).
  void coarsenOpennessTo(std::size_t L) {
    const Octree& f = levels_[L]->d.local();
    const Octree& c = levels_[L + 1]->d.local();
    Poisson& capL = levels_[L]->ap;
    const int F = 2 * Dim;
    const Index nc = c.numLeaves();
    const Index ngc = levels_[L + 1]->hp->numGhosts();
    std::vector<double> ca(static_cast<std::size_t>(nc + ngc) * F, 0.0);
    std::vector<int> cnt(static_cast<std::size_t>(nc) * F, 0);
    const std::vector<Index>& c2p = levels_[L]->c2pHost;
    for (Index i = 0; i < f.numLeaves(); ++i) {
      const Index p = c2p[static_cast<std::size_t>(i)];
      if (p < 0)
        continue;
      const std::size_t base = static_cast<std::size_t>(p) * F;
      if (c.level(p) == f.level(i)) {
        for (int axis = 0; axis < Dim; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) {
            const int fi = Poisson::faceIndex(axis, dir);
            ca[base + static_cast<std::size_t>(fi)] += capL.faceOpenness(i, axis, dir);
            cnt[base + static_cast<std::size_t>(fi)] += 1;
          }
      } else {
        const unsigned oct = M::from_code(f.code(i)).child_index(f.level(i));
        for (int axis = 0; axis < Dim; ++axis) {
          const int dir = ((oct >> axis) & 1) ? +1 : -1;
          const int fi = Poisson::faceIndex(axis, dir);
          ca[base + static_cast<std::size_t>(fi)] += capL.faceOpenness(i, axis, dir);
          cnt[base + static_cast<std::size_t>(fi)] += 1;
        }
      }
    }
    for (std::size_t k = 0; k < static_cast<std::size_t>(nc) * F; ++k)
      ca[k] = cnt[k] ? ca[k] / cnt[k] : 1.0;
    // Ghost rows from the owners (their local rows were computed by the identical arithmetic).
    const LeafHalo<Dim, Bits>& h = *levels_[L + 1]->hp;
    std::vector<double> col(static_cast<std::size_t>(levels_[L + 1]->nExt), 0.0);
    for (int fi = 0; fi < F; ++fi) {
      for (Index i = 0; i < nc; ++i)
        col[static_cast<std::size_t>(i)] = ca[static_cast<std::size_t>(i) * F + fi];
      h.exchangeHost(col);
      for (Index g = 0; g < ngc; ++g)
        ca[static_cast<std::size_t>(nc + g) * F + fi] =
            col[static_cast<std::size_t>(nc + g)];
    }
    levels_[L + 1]->ap.setOpennessRaw(std::move(ca));
  }

  // Host-assemble each level's operator through the resolver seam and upload (the distributed
  // mirror of the D5 device assembly — same CSR, ghost columns included), then the scratch.
  void finishOps() {
    for (auto& lvp : levels_) {
      Level& lv = *lvp;
      auto A = lv.ap.assembleFv();
      lv.op.n = lv.n;
      lv.op.invVol = toDevice(A.invVol, "dfmg_invvol");
      lv.op.faceStart = toDevice(A.start, "dfmg_start");
      lv.op.faceNbr = toDevice(A.nbr, "dfmg_nbr");
      lv.op.faceW = toDevice(A.coef, "dfmg_w");
      lv.op.bcDiag = toDevice(A.bcDiag, "dfmg_bc");
      lv.x = View<double>("dfmg_x", static_cast<std::size_t>(lv.nExt));
      lv.b = View<double>("dfmg_b", static_cast<std::size_t>(lv.nExt));
      lv.res = View<double>("dfmg_res", static_cast<std::size_t>(lv.n));
      lv.tmp = View<double>("dfmg_tmp", static_cast<std::size_t>(lv.n));
      Kokkos::deep_copy(lv.x, 0.0);
      Kokkos::deep_copy(lv.b, 0.0);
    }
  }

  MPI_Comm comm_ = MPI_COMM_NULL;
  double h0_ = 1.0;
  std::array<long, Dim> shift_{};
  std::vector<std::unique_ptr<Level>> levels_;
  bool removeMean_ = false;
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_DISTRIBUTED_FLOW_MG_HPP

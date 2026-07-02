// core — consistent graded FV Laplacian on a *distributed* octree.
//
// The distributed analog of AmrPoisson / Multigrid's consistent operator:
// the conservative two-point FV Laplacian
//     (L u)_i = invVol_i · Σ_f w_f (u[nbr_f] − u_i),   w_f = A_f / d_f,
// on a graded DistributedOctree (refined + cross-block 2:1 balanced), with the
// coarse side of every 2:1 interface carrying one face per fine sub-neighbour —
// including sub-neighbours that live on another rank.
//
// Topology built once: each leaf's face stencil is enumerated exactly like
// AmrPoisson::forEachFaceNeighbor (same axis/dir/sub-k order), with in-block
// neighbours resolved locally and cross-block neighbours turned into *ghost*
// entries. Ghost cells are discovered by one owner gather of the across-face
// covering level (DistributedOctree::coverLevels) — enough to know whether a
// remote face is same/coarser (one neighbour) or finer (2^(Dim-1) sub-neighbours,
// each possibly on a different owner). Each matvec then fills the ghost values with
// one owner gather (coverValues). Because the per-entry term w(val−u_i) is summed
// in the same face order whether a neighbour is local or ghost, the operator is
// bit-identical across rank counts (COMM_WORLD == COMM_SELF) and, on a single
// block, bit-identical to host AmrPoisson::applyLaplacian.
//
// Scope: openness-free (w_f = A_f/d_f); the smoother is weighted Jacobi. Openness
// (cut-cell) and a graded distributed *multigrid* hierarchy build on this.
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON.
#ifndef PECLET_CORE_AMR_DISTRIBUTED_FV_HPP
#define PECLET_CORE_AMR_DISTRIBUTED_FV_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/distributed_poisson.hpp"
#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::amr {

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DistributedFvOperator {
 public:
  using DO = DistributedOctree<Dim, Bits>;
  using M = typename DO::M;
  using Code = typename DO::Code;
  using Coord = typename DO::Coord;

  /// Build the consistent face stencil, openness-free (w_f = A_f/d_f).
  void init(DO& d) {
    init(d, [](const Vec<Dim>&, int) { return 1.0; });
  }

  /// Build with cut-cell openness: w_f = α_f · A_f/d_f, where α_f = openFn(face
  /// centroid, axis) ∈ [0,1] (1 fluid, 0 solid) evaluated at the *finer* side's
  /// (sub-)face centroid in world coords. Both sides of a face — even across a block
  /// boundary — evaluate openFn at the same point, so α is symmetric (no exchange
  /// needed) and the operator stays conservative. Matches AmrPoisson::buildOpenness.
  /// (One owner round to learn cross-block covering levels; octree must be 2:1.)
  template <class OpenFn>
  void init(DO& d, OpenFn&& openFn) {
    d_ = &d;
    h0_ = d.h0();
    const auto& t = d.local();
    const Index n = t.numLeaves();
    const IVec<Dim> bfo = d.blockFineOrigin();
    const Vec<Dim> org = d.globalGeometry().origin;
    invVol_.assign(static_cast<std::size_t>(n), 0.0);
    diag_.assign(static_cast<std::size_t>(n), 0.0);

    // α at the finer side's (sub-)face: flo = finer cell global fine lower corner,
    // s its size (fine units), faceDir its outward normal toward the neighbour.
    auto alphaOf = [&](const std::array<long, Dim>& flo, Coord s, int axis, int faceDir) -> double {
      Vec<Dim> fc{};
      long plane = flo[axis] + (faceDir > 0 ? static_cast<long>(s) : 0L);
      for (int d2 = 0; d2 < Dim; ++d2)
        fc[d2] =
            (d2 == axis)
                ? org[d2] + static_cast<double>(plane) * h0_
                : org[d2] + (static_cast<double>(flo[d2]) + 0.5 * static_cast<double>(s)) * h0_;
      double a = openFn(fc, axis);
      return a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);
    };
    auto gLo = [&](const std::array<Coord, Dim>& lc) {
      std::array<long, Dim> g{};
      for (int d2 = 0; d2 < Dim; ++d2)
        g[d2] = static_cast<long>(lc[d2]) + static_cast<long>(bfo[d2]);
      return g;
    };
    // Global fine lower corner of the k-th fine sub-neighbour across a coarse cell's
    // (axis,dir) face: clo is the coarse cell's global lo, si its size, sj=si/2. Used
    // for the sub-face α centroid — the *finer* cell's lo, not the probe point.
    auto fineLoGlobal = [&](const std::array<long, Dim>& clo, Coord si, Coord sj, int axis, int dir,
                            int k) {
      std::array<long, Dim> flo{};
      flo[axis] = (dir > 0) ? clo[axis] + static_cast<long>(si) : clo[axis] - static_cast<long>(sj);
      int bit = 0;
      for (int t = 0; t < Dim; ++t) {
        if (t == axis)
          continue;
        flo[t] = clo[t] + (((k >> bit) & 1) ? static_cast<long>(sj) : 0L);
        ++bit;
      }
      for (int d2 = 0; d2 < Dim; ++d2) {
        long gf = static_cast<long>(d.globalFineSize()[d2]);
        if (flo[d2] < 0)
          flo[d2] += gf;
        else if (flo[d2] >= gf)
          flo[d2] -= gf;
      }
      return flo;
    };

    // Pass 1: per-leaf entries in face order; remote faces become markers.
    struct E {
      double w = 0.0;
      Index localRef = -1;  // >=0 local neighbour slot
      int rfId = -1;        // >=0 unresolved remote face
    };
    std::vector<std::vector<E>> ent(static_cast<std::size_t>(n));
    struct RFace {
      std::array<Coord, Dim> gc{};
      int Li = 0;
      Coord si = 1;
      int axis = 0;
      int dir = 1;
      std::array<long, Dim> iLo{};  // requesting leaf's global fine lower corner
    };
    std::vector<RFace> rfaces;

    for (Index i = 0; i < n; ++i) {
      const int Li = static_cast<int>(t.level(i));
      const Coord si = Coord(Coord(1) << Li);
      invVol_[static_cast<std::size_t>(i)] = 1.0 / cellVol(Li);
      auto b = t.bounds(i);
      const auto& lo = b[0];
      for (int axis = 0; axis < Dim; ++axis)
        for (int dir = -1; dir <= 1; dir += 2) {
          typename DO::FaceInfo fi = d.faceAcross(i, axis, dir);
          if (fi.state == 2)
            continue;           // DomainNone
          if (fi.state == 0) {  // InBlock: enumerate locally
            Index j = fi.localNb;
            if (j < 0)
              continue;
            const int Lj = static_cast<int>(t.level(j));
            if (Lj >= Li) {  // finer side is i: α at i's face
              const double a = alphaOf(gLo(lo), si, axis, dir);
              ent[static_cast<std::size_t>(i)].push_back(
                  E{a * coeff(si, Coord(Coord(1) << Lj)), j, -1});
            } else {  // finer side is the sub-neighbour jj: α at jj's face (−dir)
              const Coord sj = Coord(si >> 1);
              const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                        : static_cast<long>(lo[axis]) - 1;
              const int nsub = 1 << (Dim - 1);
              for (int k = 0; k < nsub; ++k) {
                std::array<Coord, Dim> q = lo;
                q[axis] = static_cast<Coord>(pc);
                int bit = 0;
                for (int tt = 0; tt < Dim; ++tt) {
                  if (tt == axis)
                    continue;
                  const Coord off = ((k >> bit) & 1) ? sj : Coord(0);
                  q[tt] = static_cast<Coord>(lo[tt] + off);
                  ++bit;
                }
                Index jj = t.find(M::encode(q).code());
                const double a =
                    alphaOf(fineLoGlobal(gLo(lo), si, sj, axis, dir, k), sj, axis, -dir);
                ent[static_cast<std::size_t>(i)].push_back(E{a * coeff(si, sj), jj, -1});
              }
            }
          } else {  // Remote: defer (need covering level)
            E e;
            e.rfId = static_cast<int>(rfaces.size());
            ent[static_cast<std::size_t>(i)].push_back(e);
            RFace rf{fi.gc, Li, si, axis, dir, {}};
            rf.iLo = gLo(lo);
            rfaces.push_back(rf);
          }
        }
    }

    // One owner gather of the across-face covering level for every remote face.
    std::vector<std::array<Coord, Dim>> coverCoords(rfaces.size());
    for (std::size_t r = 0; r < rfaces.size(); ++r)
      coverCoords[r] = rfaces[r].gc;
    std::vector<int> coverLv = d.coverLevels(coverCoords);

    // Pass 2: expand into the final CSR; ghost coords deduped to ghost slots.
    std::map<std::array<Coord, Dim>, Index> ghostSlot;
    std::vector<std::array<Coord, Dim>> ghostCoords;
    auto ghostRef = [&](const std::array<Coord, Dim>& gc) -> Index {
      auto it = ghostSlot.find(gc);
      if (it != ghostSlot.end())
        return it->second;
      Index s = static_cast<Index>(ghostCoords.size());
      ghostSlot.emplace(gc, s);
      ghostCoords.push_back(gc);
      return s;
    };

    start_.assign(static_cast<std::size_t>(n) + 1, 0);
    for (Index i = 0; i < n; ++i) {
      for (const E& e : ent[static_cast<std::size_t>(i)]) {
        if (e.rfId < 0) {
          ref_.push_back(e.localRef);
          w_.push_back(e.w);
          diag_[static_cast<std::size_t>(i)] += e.w;
        } else {
          const RFace& rf = rfaces[static_cast<std::size_t>(e.rfId)];
          const int Lj = coverLv[static_cast<std::size_t>(e.rfId)];
          if (Lj < 0)
            continue;         // no neighbour
          if (Lj >= rf.Li) {  // finer side is i: α at i's face
            const double a = alphaOf(rf.iLo, rf.si, rf.axis, rf.dir);
            const double ww = a * coeff(rf.si, Coord(Coord(1) << Lj));
            ref_.push_back(d_->local().numLeaves() + ghostRef(rf.gc));
            w_.push_back(ww);
            diag_[static_cast<std::size_t>(i)] += ww;
          } else {  // finer side is each remote sub-neighbour: α at its face (−dir)
            const Coord sj = Coord(rf.si >> 1);
            const int nsub = 1 << (Dim - 1);
            for (int k = 0; k < nsub; ++k) {
              std::array<Coord, Dim> sc = subCoord(rf.gc, rf.axis, k, sj);
              const double a = alphaOf(fineLoGlobal(rf.iLo, rf.si, sj, rf.axis, rf.dir, k), sj,
                                       rf.axis, -rf.dir);
              const double ww = a * coeff(rf.si, sj);
              ref_.push_back(d_->local().numLeaves() + ghostRef(sc));
              w_.push_back(ww);
              diag_[static_cast<std::size_t>(i)] += ww;
            }
          }
        }
      }
      start_[static_cast<std::size_t>(i) + 1] = static_cast<Index>(ref_.size());
    }
    ghostCoords_ = std::move(ghostCoords);
    n_ = n;
  }

  Index numLeaves() const { return n_; }

  /// Lu = L u (consistent conservative FV Laplacian, negative-definite).
  void apply(const std::vector<double>& u, std::vector<double>& Lu) const {
    std::vector<double> g = d_->coverValues(ghostCoords_, u);
    Lu.assign(static_cast<std::size_t>(n_), 0.0);
    for (Index i = 0; i < n_; ++i) {
      const double ui = u[static_cast<std::size_t>(i)];
      double acc = 0.0;
      for (Index k = start_[static_cast<std::size_t>(i)];
           k < start_[static_cast<std::size_t>(i) + 1]; ++k) {
        const Index r = ref_[static_cast<std::size_t>(k)];
        const double val =
            (r < n_) ? u[static_cast<std::size_t>(r)] : g[static_cast<std::size_t>(r - n_)];
        acc += w_[static_cast<std::size_t>(k)] * (val - ui);
      }
      Lu[static_cast<std::size_t>(i)] = invVol_[static_cast<std::size_t>(i)] * acc;
    }
  }

  /// res = rhs − L u; returns the global L2 norm (volume-weighted).
  double residual(const std::vector<double>& u, const std::vector<double>& rhs,
                  std::vector<double>& res) const {
    std::vector<double> lu;
    apply(u, lu);
    res.assign(static_cast<std::size_t>(n_), 0.0);
    double s = 0.0;
    for (Index i = 0; i < n_; ++i) {
      double r = rhs[static_cast<std::size_t>(i)] - lu[static_cast<std::size_t>(i)];
      res[static_cast<std::size_t>(i)] = r;
      s += r * r / invVol_[static_cast<std::size_t>(i)];  // V_i r²
    }
    double g = 0.0;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, d_->comm());
    return std::sqrt(g);
  }

  double residualNorm(const std::vector<double>& u, const std::vector<double>& rhs) const {
    std::vector<double> res;
    return residual(u, rhs, res);
  }

  /// `sweeps` weighted-Jacobi relaxations of L u = rhs (point solve
  /// u_i ← (Σ w u_nb − V_i rhs_i)/Σ w). Reads only the previous iterate (one ghost
  /// gather per sweep) ⇒ bit-identical across rank counts.
  void jacobi(std::vector<double>& u, const std::vector<double>& rhs, int sweeps,
              double omega = 0.8) const {
    std::vector<double> tmp(static_cast<std::size_t>(n_));
    for (int s = 0; s < sweeps; ++s) {
      std::vector<double> g = d_->coverValues(ghostCoords_, u);
      for (Index i = 0; i < n_; ++i) {
        double sumOff = 0.0;
        for (Index k = start_[static_cast<std::size_t>(i)];
             k < start_[static_cast<std::size_t>(i) + 1]; ++k) {
          const Index r = ref_[static_cast<std::size_t>(k)];
          const double val =
              (r < n_) ? u[static_cast<std::size_t>(r)] : g[static_cast<std::size_t>(r - n_)];
          sumOff += w_[static_cast<std::size_t>(k)] * val;
        }
        const double Vrhs = rhs[static_cast<std::size_t>(i)] / invVol_[static_cast<std::size_t>(i)];
        tmp[static_cast<std::size_t>(i)] =
            (diag_[static_cast<std::size_t>(i)] != 0.0)
                ? (sumOff - Vrhs) / diag_[static_cast<std::size_t>(i)]
                : u[static_cast<std::size_t>(i)];
      }
      for (Index i = 0; i < n_; ++i)
        u[static_cast<std::size_t>(i)] = (1.0 - omega) * u[static_cast<std::size_t>(i)] +
                                         omega * tmp[static_cast<std::size_t>(i)];
    }
  }

 private:
  double cellWidth(int level) const { return h0_ * static_cast<double>(Index(1) << level); }
  double cellVol(int level) const {
    double w = cellWidth(level), v = 1.0;
    for (int d = 0; d < Dim; ++d)
      v *= w;
    return v;
  }
  double areaOf(Coord s) const {
    double a = 1.0;
    for (int d = 0; d < Dim - 1; ++d)
      a *= static_cast<double>(s) * h0_;
    return a;
  }
  double coeff(Coord si, Coord sj) const {
    double dist = 0.5 * (static_cast<double>(si) + static_cast<double>(sj)) * h0_;
    return areaOf(si < sj ? si : sj) / dist;
  }

  // tangential sub-coord for a finer 2:1 face (global, wrapped to the domain).
  std::array<Coord, Dim> subCoord(const std::array<Coord, Dim>& gc, int axis, int k,
                                  Coord sj) const {
    std::array<Coord, Dim> q = gc;
    int bit = 0;
    for (int tt = 0; tt < Dim; ++tt) {
      if (tt == axis)
        continue;
      const Coord off = ((k >> bit) & 1) ? sj : Coord(0);
      long v = static_cast<long>(gc[tt]) + static_cast<long>(off);
      const long gf = static_cast<long>(d_->globalFineSize()[tt]);
      if (v >= gf)
        v -= gf;  // periodic tangential wrap (in-domain otherwise)
      q[tt] = static_cast<Coord>(v);
      ++bit;
    }
    return q;
  }

  DO* d_ = nullptr;
  double h0_ = 1.0;
  Index n_ = 0;
  std::vector<double> invVol_, diag_, w_;
  std::vector<Index> start_, ref_;                   // face CSR (ref<n local, else ghost)
  std::vector<std::array<Coord, Dim>> ghostCoords_;  // dedup ghost cells (slot = ref−n)
};

/// Geometric-multigrid V-cycle on a *graded* distributed octree, built on
/// DistributedFvOperator. The hierarchy keeps the same ORB blocks and coarsens
/// each rank's local octree (coarsenIf, which never merges the root brick — it is
/// guarded by `level < lmax` — so every rank stops at the uniform root brick and no
/// cross-block re-decomposition is needed). 2:1 grading is preserved by uniform
/// coarsening; the consistent per-level operator handles whatever grading remains.
/// Ranks reach the root brick after DIFFERENT numbers of coarsening steps (deep vs
/// shallow blocks), so the level count is padded to the global max with identity
/// root-brick levels — otherwise the per-level collective gather deadlocks at np>1
/// (see buildImpl).
///
/// Transfers are local: a fine leaf's covering coarse leaf is in the same block
/// (parents never cross root cells), so restriction (average children) and
/// prolongation (piecewise-constant) need no communication — only the per-level
/// Jacobi smoother uses the operator's ghost halo. Jacobi + local transfers are all
/// order-independent / per-cell ⇒ the V-cycle is bit-identical COMM_WORLD vs
/// COMM_SELF. The coarsest level (uniform root brick) is solved with extra Jacobi.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class GradedDistributedMultigrid {
 public:
  using DO = DistributedOctree<Dim, Bits>;
  using BO = typename DO::Octree;
  using M = typename DO::M;

  /// Build the hierarchy from an already-graded, 2:1-balanced finest octree
  /// (openness-free; the coarsest is chained to the uniform DistributedMultigrid).
  void build(const DO& finest) {
    buildImpl(finest, [](const Vec<Dim>&, int) { return 1.0; }, false);
  }

  /// Build with cut-cell openness `openFn` on every level (each level re-samples the
  /// geometry at its own face centroids — a rediscretized coarse operator). The
  /// chained uniform bottom solve is openness-free, so for openness the coarsest is
  /// bottom-solved with Jacobi on the (correct, openness-carrying) coarsest operator.
  template <class OpenFn>
  void build(const DO& finest, OpenFn&& openFn) {
    buildImpl(finest, std::forward<OpenFn>(openFn), true);
  }

 private:
  template <class OpenFn>
  void buildImpl(const DO& finest, OpenFn&& openFn, bool hasOpen) {
    hasOpen_ = hasOpen;
    levels_.clear();
    const IVec<Dim> g = finest.globalRootSize();
    const unsigned lmax = finest.lmax();
    const AmrGeometry<Dim> geo = finest.globalGeometry();
    const auto per = finest.periodic();
    MPI_Comm comm = finest.comm();

    auto l0 = std::make_unique<Level>();
    l0->d = finest;  // copy (level 0 owned by the hierarchy)
    levels_.push_back(std::move(l0));
    for (;;) {
      BO coarse = levels_.back()->d.local();  // copy
      const Index before = coarse.numLeaves();
      coarse.coarsenIf([](typename DO::Code, unsigned) { return true; });
      if (coarse.numLeaves() == before)
        break;  // uniform root brick reached
      auto lv = std::make_unique<Level>();
      lv->d.init(g, lmax, geo, per, comm);
      lv->d.local() = coarse;  // same ORB block, coarsened local octree
      levels_.push_back(std::move(lv));
    }
    // The coarsening above stops per-rank when this block reaches its root brick, so a
    // rank owning shallow blocks builds FEWER levels than one owning deep refinement.
    // Each level's op.init / vcycle drives a COLLECTIVE owner request/reply (coverLevels,
    // coverValues over `comm`), so an unequal level count would mismatch the collectives
    // and deadlock at np>1. Pad every rank up to the global max with repeats of its root
    // brick: the extra levels are identity transfers (c2p maps a root cell to itself), a
    // no-op in the V-cycle — and they exactly mirror what the whole-domain COMM_SELF
    // reference does in its already-coarse regions, so the bit-exact WORLD==SELF check is
    // preserved. (Allreduce on COMM_SELF is a no-op, leaving the serial path unchanged.)
    {
      int localLevels = static_cast<int>(levels_.size());
      int globalLevels = localLevels;
      MPI_Allreduce(&localLevels, &globalLevels, 1, MPI_INT, MPI_MAX, comm);
      while (static_cast<int>(levels_.size()) < globalLevels) {
        auto lv = std::make_unique<Level>();
        lv->d.init(g, lmax, geo, per, comm);
        lv->d.local() = levels_.back()->d.local();  // repeat the root brick (identity level)
        levels_.push_back(std::move(lv));
      }
    }
    for (auto& lv : levels_) {
      lv->op.init(lv->d, openFn);
      const Index n = lv->d.local().numLeaves();
      lv->x.assign(static_cast<std::size_t>(n), 0.0);
      lv->b.assign(static_cast<std::size_t>(n), 0.0);
      lv->res.assign(static_cast<std::size_t>(n), 0.0);
    }
    // local fine→coarse maps (covering coarse leaf, same block).
    for (std::size_t L = 0; L + 1 < levels_.size(); ++L) {
      const BO& f = levels_[L]->d.local();
      const BO& c = levels_[L + 1]->d.local();
      const Index nf = f.numLeaves();
      auto& c2p = levels_[L]->c2p;
      c2p.assign(static_cast<std::size_t>(nf), -1);
      for (Index i = 0; i < nf; ++i)
        c2p[static_cast<std::size_t>(i)] = c.find(f.code(i));
    }

    // Bottom solver (openness-free only): a uniform DistributedMultigrid on the root
    // grid, below the root brick. The coarsest graded level IS the uniform root brick
    // (root cells at level lmax); its operator is L=∇² at spacing h0·2^lmax, the same
    // operator the uniform MG's finest applies (DistributedPoisson is also L=∇²) on
    // the same root grid — so the bottom solve of L e = res is inner.vcycle(e, res).
    // With openness the uniform inner is inconsistent, so the coarsest is bottom-solved
    // with Jacobi on its own (openness-carrying) operator instead.
    if (!hasOpen_) {
      AmrGeometry<Dim> ig = geo;
      ig.h0 = geo.h0 * static_cast<double>(Index(1) << lmax);  // root-cell width
      inner_ = std::make_unique<DistributedMultigrid<Dim, Bits>>();
      inner_->build(g, ig, per, comm);
      DO& coarsest = levels_.back()->d;
      nCoarse_ = coarsest.local().numLeaves();
      innerMap_.assign(static_cast<std::size_t>(nCoarse_), -1);
      for (Index i = 0; i < nCoarse_; ++i)
        innerMap_[static_cast<std::size_t>(i)] =
            inner_->octree(0).findGlobalRoot(coarsest.globalRootOf(i));
    }
  }

 public:
  std::size_t numLevels() const { return levels_.size(); }
  DistributedFvOperator<Dim, Bits>& op(std::size_t L = 0) { return levels_[L]->op; }
  Index numLeaves(std::size_t L = 0) const { return levels_[L]->d.local().numLeaves(); }

  /// One V-cycle of L u = rhs on level `L` (default finest), correction scheme. The
  /// coarsest (uniform root brick) is solved by `innerCycles` V-cycles of the uniform
  /// DistributedMultigrid on the root grid.
  void vcycle(std::vector<double>& x, const std::vector<double>& b, int pre = 2, int post = 2,
              int innerCycles = 6, double omega = 0.8, std::size_t L = 0) {
    auto& lv = *levels_[L];
    if (L + 1 == levels_.size()) {
      if (inner_)
        bottomSolve(lv.op, x, b, innerCycles);  // openness-free: chained uniform MG
      else
        lv.op.jacobi(x, b, bottomSweeps_, omega);  // openness: Jacobi on the correct op
      return;
    }
    lv.op.jacobi(x, b, pre, omega);
    std::vector<double> res;
    lv.op.residual(x, b, res);
    // restrict residual → coarse rhs (local average over children).
    auto& cl = *levels_[L + 1];
    const Index nc = cl.d.local().numLeaves();
    std::vector<double> cb(static_cast<std::size_t>(nc), 0.0),
        cn(static_cast<std::size_t>(nc), 0.0);
    const auto& c2p = lv.c2p;
    const Index nf = lv.d.local().numLeaves();
    for (Index i = 0; i < nf; ++i) {
      Index p = c2p[static_cast<std::size_t>(i)];
      if (p < 0)
        continue;
      cb[static_cast<std::size_t>(p)] += res[static_cast<std::size_t>(i)];
      cn[static_cast<std::size_t>(p)] += 1.0;
    }
    for (Index p = 0; p < nc; ++p)
      if (cn[static_cast<std::size_t>(p)] > 0.0)
        cb[static_cast<std::size_t>(p)] /= cn[static_cast<std::size_t>(p)];
    std::vector<double> cx(static_cast<std::size_t>(nc), 0.0);
    vcycle(cx, cb, pre, post, innerCycles, omega, L + 1);
    for (Index i = 0; i < nf; ++i) {
      Index p = c2p[static_cast<std::size_t>(i)];
      if (p >= 0)
        x[static_cast<std::size_t>(i)] += cx[static_cast<std::size_t>(p)];
    }
    lv.op.jacobi(x, b, post, omega);
  }

 private:
  // Solve L x = b on the uniform root brick via the inner uniform MG (correction
  // scheme): res = b − Lx; solve L e = res with `cycles` inner V-cycles; x += e.
  // Both operators are L = ∇² (same suite-wide sign), so no sign flip is needed.
  // Index map coarsest→inner is by global root cell (identity in practice).
  void bottomSolve(DistributedFvOperator<Dim, Bits>& op, std::vector<double>& x,
                   const std::vector<double>& b, int cycles) {
    std::vector<double> res;
    op.residual(x, b, res);
    const Index ni = inner_->numLeaves(0);
    std::vector<double> bi(static_cast<std::size_t>(ni), 0.0),
        ei(static_cast<std::size_t>(ni), 0.0);
    for (Index i = 0; i < nCoarse_; ++i) {
      Index m = innerMap_[static_cast<std::size_t>(i)];
      if (m >= 0)
        bi[static_cast<std::size_t>(m)] = res[static_cast<std::size_t>(i)];
    }
    for (int c = 0; c < cycles; ++c)
      inner_->vcycle(ei, bi);
    for (Index i = 0; i < nCoarse_; ++i) {
      Index m = innerMap_[static_cast<std::size_t>(i)];
      if (m >= 0)
        x[static_cast<std::size_t>(i)] += ei[static_cast<std::size_t>(m)];
    }
  }

  struct Level {
    DO d;
    DistributedFvOperator<Dim, Bits> op;
    std::vector<Index> c2p;         // fine leaf → covering coarse leaf (local)
    std::vector<double> x, b, res;  // scratch
  };
  std::vector<std::unique_ptr<Level>> levels_;
  std::unique_ptr<DistributedMultigrid<Dim, Bits>> inner_;  // uniform bottom solver (openness-free)
  std::vector<Index> innerMap_;                             // coarsest leaf → inner finest leaf
  Index nCoarse_ = 0;
  bool hasOpen_ = false;
  int bottomSweeps_ = 400;  // Jacobi sweeps on the coarsest op when openness present
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_DISTRIBUTED_FV_HPP

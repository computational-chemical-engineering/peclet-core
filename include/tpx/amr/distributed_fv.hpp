// transport-core — consistent graded FV Laplacian on a *distributed* octree.
//
// The distributed analog of AmrPoisson / DeviceMultigrid's consistent operator:
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
// Header-only, guarded by TPX_HAVE_MORTON.
#ifndef TPX_AMR_DISTRIBUTED_FV_HPP
#define TPX_AMR_DISTRIBUTED_FV_HPP

#ifdef TPX_HAVE_MORTON

#include <array>
#include <cmath>
#include <map>
#include <vector>

#include "tpx/amr/distributed_octree.hpp"
#include "tpx/common/mpi.hpp"
#include "tpx/common/types.hpp"

namespace tpx::amr {

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DistributedFvOperator {
 public:
  using DO = DistributedOctree<Dim, Bits>;
  using M = typename DO::M;
  using Code = typename DO::Code;
  using Coord = typename DO::Coord;

  /// Build the consistent face stencil (one owner round to learn cross-block
  /// covering levels). The octree must already be graded/2:1-balanced.
  void init(DO& d) {
    d_ = &d;
    h0_ = d.h0();
    const auto& t = d.local();
    const Index n = t.numLeaves();
    invVol_.assign(static_cast<std::size_t>(n), 0.0);
    diag_.assign(static_cast<std::size_t>(n), 0.0);

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
          if (fi.state == 2) continue;  // DomainNone
          if (fi.state == 0) {          // InBlock: enumerate locally
            Index j = fi.localNb;
            if (j < 0) continue;
            const int Lj = static_cast<int>(t.level(j));
            if (Lj >= Li) {
              ent[static_cast<std::size_t>(i)].push_back(E{coeff(si, Coord(Coord(1) << Lj)), j, -1});
            } else {
              const Coord sj = Coord(si >> 1);
              const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                        : static_cast<long>(lo[axis]) - 1;
              const int nsub = 1 << (Dim - 1);
              for (int k = 0; k < nsub; ++k) {
                std::array<Coord, Dim> q = lo;
                q[axis] = static_cast<Coord>(pc);
                int bit = 0;
                for (int tt = 0; tt < Dim; ++tt) {
                  if (tt == axis) continue;
                  const Coord off = ((k >> bit) & 1) ? sj : Coord(0);
                  q[tt] = static_cast<Coord>(lo[tt] + off);
                  ++bit;
                }
                Index jj = t.find(M::encode(q).code());
                ent[static_cast<std::size_t>(i)].push_back(E{coeff(si, sj), jj, -1});
              }
            }
          } else {  // Remote: defer (need covering level)
            E e;
            e.rfId = static_cast<int>(rfaces.size());
            ent[static_cast<std::size_t>(i)].push_back(e);
            rfaces.push_back(RFace{fi.gc, Li, si, axis});
          }
        }
    }

    // One owner gather of the across-face covering level for every remote face.
    std::vector<std::array<Coord, Dim>> coverCoords(rfaces.size());
    for (std::size_t r = 0; r < rfaces.size(); ++r) coverCoords[r] = rfaces[r].gc;
    std::vector<int> coverLv = d.coverLevels(coverCoords);

    // Pass 2: expand into the final CSR; ghost coords deduped to ghost slots.
    std::map<std::array<Coord, Dim>, Index> ghostSlot;
    std::vector<std::array<Coord, Dim>> ghostCoords;
    auto ghostRef = [&](const std::array<Coord, Dim>& gc) -> Index {
      auto it = ghostSlot.find(gc);
      if (it != ghostSlot.end()) return it->second;
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
          if (Lj < 0) continue;  // no neighbour
          if (Lj >= rf.Li) {
            const double ww = coeff(rf.si, Coord(Coord(1) << Lj));
            ref_.push_back(d_->local().numLeaves() + ghostRef(rf.gc));
            w_.push_back(ww);
            diag_[static_cast<std::size_t>(i)] += ww;
          } else {
            const Coord sj = Coord(rf.si >> 1);
            const double ww = coeff(rf.si, sj);
            const int nsub = 1 << (Dim - 1);
            for (int k = 0; k < nsub; ++k) {
              std::array<Coord, Dim> sc = subCoord(rf.gc, rf.axis, k, sj);
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
      for (Index k = start_[static_cast<std::size_t>(i)]; k < start_[static_cast<std::size_t>(i) + 1]; ++k) {
        const Index r = ref_[static_cast<std::size_t>(k)];
        const double val = (r < n_) ? u[static_cast<std::size_t>(r)] : g[static_cast<std::size_t>(r - n_)];
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
        for (Index k = start_[static_cast<std::size_t>(i)]; k < start_[static_cast<std::size_t>(i) + 1]; ++k) {
          const Index r = ref_[static_cast<std::size_t>(k)];
          const double val = (r < n_) ? u[static_cast<std::size_t>(r)] : g[static_cast<std::size_t>(r - n_)];
          sumOff += w_[static_cast<std::size_t>(k)] * val;
        }
        const double Vrhs = rhs[static_cast<std::size_t>(i)] / invVol_[static_cast<std::size_t>(i)];
        tmp[static_cast<std::size_t>(i)] =
            (diag_[static_cast<std::size_t>(i)] != 0.0) ? (sumOff - Vrhs) / diag_[static_cast<std::size_t>(i)]
                                                        : u[static_cast<std::size_t>(i)];
      }
      for (Index i = 0; i < n_; ++i)
        u[static_cast<std::size_t>(i)] =
            (1.0 - omega) * u[static_cast<std::size_t>(i)] + omega * tmp[static_cast<std::size_t>(i)];
    }
  }

 private:
  double cellWidth(int level) const { return h0_ * static_cast<double>(Index(1) << level); }
  double cellVol(int level) const {
    double w = cellWidth(level), v = 1.0;
    for (int d = 0; d < Dim; ++d) v *= w;
    return v;
  }
  double areaOf(Coord s) const {
    double a = 1.0;
    for (int d = 0; d < Dim - 1; ++d) a *= static_cast<double>(s) * h0_;
    return a;
  }
  double coeff(Coord si, Coord sj) const {
    double dist = 0.5 * (static_cast<double>(si) + static_cast<double>(sj)) * h0_;
    return areaOf(si < sj ? si : sj) / dist;
  }

  // tangential sub-coord for a finer 2:1 face (global, wrapped to the domain).
  std::array<Coord, Dim> subCoord(const std::array<Coord, Dim>& gc, int axis, int k, Coord sj) const {
    std::array<Coord, Dim> q = gc;
    int bit = 0;
    for (int tt = 0; tt < Dim; ++tt) {
      if (tt == axis) continue;
      const Coord off = ((k >> bit) & 1) ? sj : Coord(0);
      long v = static_cast<long>(gc[tt]) + static_cast<long>(off);
      const long gf = static_cast<long>(d_->globalFineSize()[tt]);
      if (v >= gf) v -= gf;  // periodic tangential wrap (in-domain otherwise)
      q[tt] = static_cast<Coord>(v);
      ++bit;
    }
    return q;
  }

  DO* d_ = nullptr;
  double h0_ = 1.0;
  Index n_ = 0;
  std::vector<double> invVol_, diag_, w_;
  std::vector<Index> start_, ref_;                 // face CSR (ref<n local, else ghost)
  std::vector<std::array<Coord, Dim>> ghostCoords_;  // dedup ghost cells (slot = ref−n)
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DISTRIBUTED_FV_HPP

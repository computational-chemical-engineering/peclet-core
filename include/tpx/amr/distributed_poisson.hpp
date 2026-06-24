// transport-core — distributed Poisson operator + smoother on a DistributedOctree.
//
// The distributed compute path for the AMR octree: the operator (Laplacian matvec
// and a weighted-Jacobi smoother) runs per rank on the local block, with the
// neighbour values across block faces supplied by DistributedOctree's owner-based
// ghost exchange (faceNeighborGather) — the same NBX halo used for cross-block
// balance. Jacobi uses only the previous iterate, so a distributed sweep is
// bit-identical to the serial one (the halo carries exactly the cells a serial
// single-block solve would read). This is the analog of sdflow's MPI-folded
// CutcellMG, at the operator/smoother level.
//
// Scope: the plain (uniform, openness-free) Laplacian L = ∇² in grid spacing h0,
// on a periodic global domain — the clean first distributed milestone. Graded /
// openness / a full distributed V-cycle build on this + the per-level halos.
//
// Sign convention (suite-wide): the operator IS the Laplacian L = ∇² (negative-
// definite; diagonal −2*Dim/h², off +1/h²), matching AmrPoisson::applyLaplacian,
// DistributedFvOperator and the device operators. Solvers solve L u = rhs.
//
// Header-only, guarded by TPX_HAVE_MORTON; uses the MPI shim.
#ifndef TPX_AMR_DISTRIBUTED_POISSON_HPP
#define TPX_AMR_DISTRIBUTED_POISSON_HPP

#ifdef TPX_HAVE_MORTON

#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

#include "tpx/amr/distributed_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/common/mpi.hpp"
#include "tpx/common/types.hpp"

namespace tpx::amr {

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DistributedPoisson {
 public:
  void init(DistributedOctree<Dim, Bits>& d, double h0) {
    d_ = &d;
    h0_ = h0;
  }

  Index numLeaves() const { return d_->local().numLeaves(); }

  /// y = L x with L = ∇² (negative-definite: diagonal −2*Dim/h², off +1/h²).
  /// Cross-block neighbours come from the owner-based halo; periodic global domain ⇒
  /// every face has a neighbour.
  void apply(const std::vector<double>& x, std::vector<double>& y) const {
    auto g = d_->faceNeighborGather(x);
    const Index n = numLeaves();
    const int F = 2 * Dim;
    const double inv = 1.0 / (h0_ * h0_);
    y.assign(static_cast<std::size_t>(n), 0.0);
    for (Index i = 0; i < n; ++i) {
      double s = 0.0;
      for (int f = 0; f < F; ++f) s += g[static_cast<std::size_t>(i) * F + f] - x[static_cast<std::size_t>(i)];
      y[static_cast<std::size_t>(i)] = inv * s;  // L = ∇²
    }
  }

  /// `sweeps` damped-Jacobi relaxations of L u = b (in place), L = ∇². The point
  /// update is u_i += ω (L u_i − b_i)/diag with diag = 2*Dim/h² (= −L_ii, so the
  /// damping has the right sign for the negative-definite L). Reads only the previous
  /// iterate (one halo gather per sweep) ⇒ bit-identical to a serial single-block sweep.
  void jacobi(std::vector<double>& x, const std::vector<double>& b, int sweeps, double omega = 0.8) const {
    const Index n = numLeaves();
    const int F = 2 * Dim;
    const double inv = 1.0 / (h0_ * h0_), diag = F * inv;
    std::vector<double> lx;
    for (int s = 0; s < sweeps; ++s) {
      apply(x, lx);  // lx = L x; gathers ghosts
      for (Index i = 0; i < n; ++i)
        x[static_cast<std::size_t>(i)] +=
            omega * (lx[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]) / diag;
    }
  }

  /// res = b − L x (local vector); also returns its global L2 norm.
  double residual(const std::vector<double>& x, const std::vector<double>& b,
                  std::vector<double>& res) const {
    std::vector<double> ax;
    apply(x, ax);
    const Index n = numLeaves();
    res.assign(static_cast<std::size_t>(n), 0.0);
    double s = 0.0;
    for (Index i = 0; i < n; ++i) {
      double r = b[static_cast<std::size_t>(i)] - ax[static_cast<std::size_t>(i)];
      res[static_cast<std::size_t>(i)] = r;
      s += r * r;
    }
    double g = 0.0;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, d_->comm());
    return std::sqrt(g);
  }

  /// Global L2 norm of b − L x (across ranks).
  double residualNorm(const std::vector<double>& x, const std::vector<double>& b) const {
    std::vector<double> ax;
    apply(x, ax);
    double s = 0.0;
    for (Index i = 0; i < numLeaves(); ++i) {
      double r = b[static_cast<std::size_t>(i)] - ax[static_cast<std::size_t>(i)];
      s += r * r;
    }
    double g = 0.0;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, d_->comm());
    return std::sqrt(g);
  }

 private:
  DistributedOctree<Dim, Bits>* d_ = nullptr;
  double h0_ = 1.0;
};

/// Distributed geometric-multigrid V-cycle for the plain Laplacian on a uniform
/// (lmax==0) octree, the distributed analog of the host AmrMultigrid + sdflow's
/// MPI-folded CutcellMG. The hierarchy is a stack of DistributedOctrees on the
/// successively halved global root grid, each ORB-decomposed over the *same* comm.
///
/// The decompositions **nest**: for a power-of-two grid and rank count, ORB bisects
/// at proportional positions, so rank r's coarse block is exactly its fine block
/// halved. Hence every fine cell's parent is owned by the same rank — restriction
/// and prolongation are purely local (no comm); only the Jacobi smoother needs the
/// per-level halo. build() asserts this nesting (each c2p entry resolves locally).
///
/// Jacobi smoother + local averaging restriction + piecewise-constant prolongation
/// are all order-independent / per-cell, so the whole V-cycle is **bit-identical**
/// across rank counts (COMM_WORLD == COMM_SELF), the suite's distributed-validation
/// contract.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DistributedMultigrid {
 public:
  /// Build the hierarchy from the finest global root grid `g0` (lmax==0) on `comm`.
  /// Coarsens by halving until an axis would drop below 2 or the cell count below the
  /// rank count. `geo.h0` is the finest spacing; coarser levels use 2^k·h0.
  void build(const IVec<Dim>& g0, const AmrGeometry<Dim>& geo,
             const std::array<bool, Dim>& periodic, MPI_Comm comm) {
    levels_.clear();
    int size = 1;
    MPI_Comm_size(comm, &size);
    IVec<Dim> g = g0;
    double h = geo.h0;
    for (;;) {
      auto lvl = std::make_unique<Level>();
      AmrGeometry<Dim> lg = geo;
      lg.h0 = h;
      lvl->d.init(g, /*lmax=*/0, lg, periodic, comm);
      lvl->op.init(lvl->d, h);
      levels_.push_back(std::move(lvl));
      // Can we coarsen one more level?
      bool ok = true;
      long prod = 1;
      IVec<Dim> ng{};
      for (int d = 0; d < Dim; ++d) {
        if (g[d] % 2 != 0 || g[d] / 2 < 2) ok = false;
        ng[d] = g[d] / 2;
        prod *= ng[d];
      }
      if (!ok || prod < size) break;
      g = ng;
      h *= 2.0;
    }
    // Nested fine→coarse maps (local; asserts nesting holds).
    for (std::size_t L = 0; L + 1 < levels_.size(); ++L) {
      auto& fine = levels_[L]->d;
      auto& coarse = levels_[L + 1]->d;
      const Index nf = fine.local().numLeaves();
      auto& c2p = levels_[L]->c2p;
      c2p.assign(static_cast<std::size_t>(nf), -1);
      for (Index i = 0; i < nf; ++i) {
        IVec<Dim> gf = fine.globalRootOf(i), gc{};
        for (int d = 0; d < Dim; ++d) gc[d] = gf[d] / 2;
        Index p = coarse.findGlobalRoot(gc);
        assert(p >= 0 && "ORB decompositions must nest (power-of-two grid+ranks)");
        c2p[static_cast<std::size_t>(i)] = p;
      }
    }
  }

  std::size_t numLevels() const { return levels_.size(); }
  DistributedOctree<Dim, Bits>& octree(std::size_t L = 0) { return levels_[L]->d; }
  DistributedPoisson<Dim, Bits>& op(std::size_t L = 0) { return levels_[L]->op; }
  Index numLeaves(std::size_t L = 0) const { return levels_[L]->d.local().numLeaves(); }

  /// One V-cycle of A x = b on level `L` (default the finest), correction scheme.
  void vcycle(std::vector<double>& x, const std::vector<double>& b, int pre = 2,
              int post = 2, int bottom = 30, std::size_t L = 0) {
    auto& op = levels_[L]->op;
    if (L + 1 == levels_.size()) {  // coarsest: a few smooths
      op.jacobi(x, b, bottom);
      return;
    }
    op.jacobi(x, b, pre);
    std::vector<double> res;
    op.residual(x, b, res);
    // Restrict residual → coarse rhs (local volume-average over children).
    const Index nc = levels_[L + 1]->d.local().numLeaves();
    std::vector<double> cb(static_cast<std::size_t>(nc), 0.0);
    std::vector<double> cn(static_cast<std::size_t>(nc), 0.0);
    const auto& c2p = levels_[L]->c2p;
    const Index nf = levels_[L]->d.local().numLeaves();
    for (Index i = 0; i < nf; ++i) {
      Index p = c2p[static_cast<std::size_t>(i)];
      cb[static_cast<std::size_t>(p)] += res[static_cast<std::size_t>(i)];
      cn[static_cast<std::size_t>(p)] += 1.0;
    }
    for (Index p = 0; p < nc; ++p)
      if (cn[static_cast<std::size_t>(p)] > 0.0) cb[static_cast<std::size_t>(p)] /= cn[static_cast<std::size_t>(p)];
    // Coarse solve.
    std::vector<double> cx(static_cast<std::size_t>(nc), 0.0);
    vcycle(cx, cb, pre, post, bottom, L + 1);
    // Prolong (piecewise-constant) + correct.
    for (Index i = 0; i < nf; ++i)
      x[static_cast<std::size_t>(i)] += cx[static_cast<std::size_t>(c2p[static_cast<std::size_t>(i)])];
    op.jacobi(x, b, post);
  }

 private:
  struct Level {
    DistributedOctree<Dim, Bits> d;
    DistributedPoisson<Dim, Bits> op;
    std::vector<Index> c2p;  // fine leaf → coarse leaf (local)
  };
  std::vector<std::unique_ptr<Level>> levels_;
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DISTRIBUTED_POISSON_HPP

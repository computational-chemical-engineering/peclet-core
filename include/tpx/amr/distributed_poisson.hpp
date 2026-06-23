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
// Scope: the plain (uniform, openness-free) Laplacian -A = ∇² in grid spacing h0,
// on a periodic global domain — the clean first distributed milestone. Graded /
// openness / a full distributed V-cycle build on this + the per-level halos.
//
// Header-only, guarded by TPX_HAVE_MORTON; uses the MPI shim.
#ifndef TPX_AMR_DISTRIBUTED_POISSON_HPP
#define TPX_AMR_DISTRIBUTED_POISSON_HPP

#ifdef TPX_HAVE_MORTON

#include <cmath>
#include <vector>

#include "tpx/amr/distributed_octree.hpp"
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

  /// y = A x with A = −∇² (positive: diagonal 2*Dim/h², off −1/h²). Cross-block
  /// neighbours come from the owner-based halo; periodic global domain ⇒ every face
  /// has a neighbour.
  void apply(const std::vector<double>& x, std::vector<double>& y) const {
    auto g = d_->faceNeighborGather(x);
    const Index n = numLeaves();
    const int F = 2 * Dim;
    const double inv = 1.0 / (h0_ * h0_);
    y.assign(static_cast<std::size_t>(n), 0.0);
    for (Index i = 0; i < n; ++i) {
      double s = 0.0;
      for (int f = 0; f < F; ++f) s += g[static_cast<std::size_t>(i) * F + f] - x[static_cast<std::size_t>(i)];
      y[static_cast<std::size_t>(i)] = -inv * s;  // A = -∇²
    }
  }

  /// `sweeps` damped-Jacobi relaxations of A x = b (in place). Jacobi reads only the
  /// previous iterate (one halo gather per sweep), so the result is bit-identical to
  /// a serial single-block Jacobi.
  void jacobi(std::vector<double>& x, const std::vector<double>& b, int sweeps, double omega = 0.8) const {
    const Index n = numLeaves();
    const int F = 2 * Dim;
    const double inv = 1.0 / (h0_ * h0_), diag = F * inv;
    std::vector<double> ax;
    for (int s = 0; s < sweeps; ++s) {
      apply(x, ax);  // gathers ghosts
      for (Index i = 0; i < n; ++i)
        x[static_cast<std::size_t>(i)] +=
            omega * (b[static_cast<std::size_t>(i)] - ax[static_cast<std::size_t>(i)]) / diag;
    }
  }

  /// Global L2 norm of b − A x (across ranks).
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

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DISTRIBUTED_POISSON_HPP

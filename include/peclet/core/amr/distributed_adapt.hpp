// transport-core — distributed dynamic (solution-adaptive) AMR on a DistributedOctree.
//
// The distributed counterpart of adapt.hpp, keeping the existing ORB block ownership.
// (Adaptation can leave the per-rank leaf counts uneven; DistributedOctree::rebalance()
// re-decomposes on leaf-count weights and migrates leaves+fields to restore balance.)
// It composes pieces that are already validated:
//   * the Löhner indicator is evaluated from the owner-based face-neighbour halo
//     (DistributedOctree::faceNeighborGather), so each rank sees the same neighbour
//     values a whole-domain solve would — the flags are identical to the serial ones;
//   * refine / coarsen run on each rank's *local* octree (sibling groups never cross
//     root cells, so coarsening is purely local), then DistributedOctree::balance()
//     restores global 2:1 across blocks;
//   * the field is remapped locally with transferField (refine/coarsen/balance only
//     change a block's internal structure), so it is conservative per block and needs
//     no field communication.
// Because flags come from the (deterministic) halo gather, balance() is deterministic,
// and the remap is per-cell, the adapted mesh + field are bit-identical across rank
// counts (COMM_WORLD == COMM_SELF).
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON.
#ifndef PECLET_CORE_AMR_DISTRIBUTED_ADAPT_HPP
#define PECLET_CORE_AMR_DISTRIBUTED_ADAPT_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <cmath>
#include <vector>

#include "peclet/core/amr/adapt.hpp"
#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::amr {

/// Löhner indicator per local leaf, using the owner-based face-neighbour halo so
/// cross-block neighbours contribute exactly as in a whole-domain solve.
template <int Dim, unsigned Bits>
std::vector<double> lohnerIndicatorDistributed(const DistributedOctree<Dim, Bits>& d,
                                               const std::vector<double>& u, double eps = 0.01) {
  const auto g = d.faceNeighborGather(u);  // out[i*F + 2*axis + (dir>0?0:1)]
  const double sentinel = DistributedOctree<Dim, Bits>::kNoNeighbor;
  const Index n = d.local().numLeaves();
  const int F = 2 * Dim;
  std::vector<double> e(static_cast<std::size_t>(n), 0.0);
  for (Index i = 0; i < n; ++i) {
    const double ui = u[static_cast<std::size_t>(i)];
    double num2 = 0.0, den2 = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const double up = g[static_cast<std::size_t>(i) * F + 2 * axis + 0];
      const double um = g[static_cast<std::size_t>(i) * F + 2 * axis + 1];
      if (up == sentinel || um == sentinel) continue;
      const double d2 = up - 2.0 * ui + um;
      const double nrm = std::fabs(up - ui) + std::fabs(ui - um) +
                         eps * (std::fabs(up) + 2.0 * std::fabs(ui) + std::fabs(um));
      num2 += d2 * d2;
      den2 += nrm * nrm;
    }
    e[static_cast<std::size_t>(i)] = (den2 > 0.0) ? std::sqrt(num2 / den2) : 0.0;
  }
  return e;
}

/// One distributed solution-adaptive step. Mutates `d`'s local octree (refine /
/// coarsen one level + cross-block 2:1 balance) and returns the field `f` remapped
/// onto the new local mesh (conservative). Same ORB ownership as before.
template <int Dim, unsigned Bits>
std::vector<double> distributedAdapt(DistributedOctree<Dim, Bits>& d, const std::vector<double>& f,
                                     double refineThresh, double coarsenThresh,
                                     unsigned finestLevel = 0, double eps = 0.01, bool linear = true) {
  using BO = typename DistributedOctree<Dim, Bits>::Octree;
  using Code = typename BO::Code;
  using M = typename BO::M;

  auto ind = lohnerIndicatorDistributed(d, f, eps);
  auto flags = flagByIndicator(d.local(), ind, refineThresh, coarsenThresh, finestLevel);

  const BO oldLocal = d.local();  // snapshot for the remap + flag lookup
  // coarsen sibling groups whose every child is flagged kCoarsen
  d.local().coarsenIf([&](Code parent, unsigned pl) {
    for (unsigned oct = 0; oct < (1u << Dim); ++oct) {
      Code cc = M::from_code(parent).child(pl, oct).code();
      Index ci = oldLocal.find(cc);
      if (ci < 0 || flags[static_cast<std::size_t>(ci)] != kCoarsen) return false;
    }
    return true;
  });
  // refine leaves flagged kRefine
  d.local().refineIf([&](Code c, unsigned) {
    Index ci = oldLocal.find(c);
    return ci >= 0 && flags[static_cast<std::size_t>(ci)] == kRefine;
  });
  d.balance();  // cross-block 2:1 to a global fixpoint
  return transferField(oldLocal, f, d.local(), linear);
}

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_DISTRIBUTED_ADAPT_HPP

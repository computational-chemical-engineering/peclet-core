// core — distributed dynamic (solution-adaptive) AMR on a DistributedOctree.
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
      if (up == sentinel || um == sentinel)
        continue;
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

/// transferField's minmod prolongation gradients on a DistributedOctree: bit-identical to the
/// block-local transferFieldGradients wherever the stencil is in-block, block-crossing faces
/// completed from the owners (coverValues + coverLevels — the covering leaf across a 2:1
/// interface is probed at the same corner coordinate faceNeighbor uses), and DOMAIN-crossing
/// probes counted as missing — the single-rank faceNeighbor convention — so np=1 reproduces
/// the local gradients bit-for-bit and the transfer is bit-exact WORLD==SELF. (The centroid
/// distance along the probe axis is 0.5·(s_i+s_j) fine units on 2:1-aligned faces; small
/// integers and halves are exact in double, so the quotient is bit-identical to the local
/// centroid-difference expression.)
template <int Dim, unsigned Bits>
std::vector<std::array<double, Dim>> transferGradients(const DistributedOctree<Dim, Bits>& d,
                                                       const std::vector<double>& f) {
  using DO = DistributedOctree<Dim, Bits>;
  using BO = typename DO::Octree;
  using Coord = typename BO::Coord;
  using M = typename BO::M;
  const BO& t = d.local();
  const Index n = t.numLeaves();
  const int S = 2 * Dim;

  struct Side {
    double val = 0.0, sj = 0.0;
    bool ok = false;
  };
  std::vector<Side> sides(static_cast<std::size_t>(n) * S);
  std::vector<std::array<Coord, Dim>> rq;
  std::vector<std::size_t> rslot;
  for (Index i = 0; i < n; ++i) {
    auto b = t.bounds(i);
    const auto& lo = b[0];
    const long si = 1L << t.level(i);
    for (int axis = 0; axis < Dim; ++axis)
      for (int dd = 0; dd < 2; ++dd) {  // dd 0 = +1 side, 1 = −1 side
        const std::size_t slot = static_cast<std::size_t>(i) * S + axis * 2 + dd;
        const long pc = (dd == 0) ? static_cast<long>(lo[axis]) + si
                                  : static_cast<long>(lo[axis]) - 1;
        if (pc >= 0 &&
            pc < static_cast<long>(d.blockBrick()[axis]) * static_cast<long>(d.rootSpan())) {
          std::array<Coord, Dim> p = lo;
          p[axis] = static_cast<Coord>(pc);
          const Index j = t.find(M::encode(p).code());
          if (j >= 0) {
            sides[slot].val = f[static_cast<std::size_t>(j)];
            sides[slot].sj = static_cast<double>(1L << t.level(j));
            sides[slot].ok = true;
          }
        } else {
          // Block-crossing: complete from the owner IF the probe stays inside the domain —
          // a DOMAIN-crossing probe is missing (the single-rank faceNeighbor convention;
          // transferField never wraps periodically).
          std::array<long, Dim> g{};
          bool inDomain = true;
          for (int a = 0; a < Dim; ++a)
            g[a] = static_cast<long>(lo[a]) + d.blockFineOrigin()[a];
          g[axis] = pc + d.blockFineOrigin()[axis];
          for (int a = 0; a < Dim; ++a)
            if (g[a] < 0 || g[a] >= static_cast<long>(d.globalFineSize()[a])) {
              inDomain = false;
              break;
            }
          if (inDomain) {
            std::array<Coord, Dim> gc{};
            for (int a = 0; a < Dim; ++a)
              gc[a] = static_cast<Coord>(g[a]);
            rslot.push_back(slot);
            rq.push_back(gc);
          }
        }
      }
  }
  const std::vector<double> vals = d.coverValues(rq, f);
  const std::vector<int> lvls = d.coverLevels(rq);
  for (std::size_t k = 0; k < rq.size(); ++k)
    if (lvls[k] >= 0) {
      sides[rslot[k]].val = vals[k];
      sides[rslot[k]].sj = static_cast<double>(1L << lvls[k]);
      sides[rslot[k]].ok = true;
    }

  std::vector<std::array<double, Dim>> grad(static_cast<std::size_t>(n),
                                            std::array<double, Dim>{});
  for (Index i = 0; i < n; ++i) {
    const double si = static_cast<double>(1L << t.level(i));
    const double ui = f[static_cast<std::size_t>(i)];
    for (int axis = 0; axis < Dim; ++axis) {
      const Side& P = sides[static_cast<std::size_t>(i) * S + axis * 2 + 0];
      const Side& Mi = sides[static_cast<std::size_t>(i) * S + axis * 2 + 1];
      if (!P.ok || !Mi.ok)
        continue;  // == the jp<0 || jm<0 skip
      const double sp = (P.val - ui) / (0.5 * (si + P.sj));
      const double sm = (ui - Mi.val) / (0.5 * (si + Mi.sj));
      grad[static_cast<std::size_t>(i)][axis] = detail::minmod(sp, sm);
    }
  }
  return grad;
}

/// One distributed solution-adaptive step. Mutates `d`'s local octree (refine /
/// coarsen one level + cross-block 2:1 balance) and returns the field `f` remapped
/// onto the new local mesh (conservative). Same ORB ownership as before.
template <int Dim, unsigned Bits>
std::vector<double> distributedAdapt(DistributedOctree<Dim, Bits>& d, const std::vector<double>& f,
                                     double refineThresh, double coarsenThresh,
                                     unsigned finestLevel = 0, double eps = 0.01,
                                     bool linear = true) {
  using BO = typename DistributedOctree<Dim, Bits>::Octree;
  using Code = typename BO::Code;
  using M = typename BO::M;

  auto ind = lohnerIndicatorDistributed(d, f, eps);
  auto flags = flagByIndicator(d.local(), ind, refineThresh, coarsenThresh, finestLevel);

  // Halo-completed prolongation gradients (BEFORE the mutation, while d still holds the old
  // mesh): the block-local stencil zeroes the gradient at interior block boundaries, which
  // would make the remap np-dependent there.
  std::vector<std::array<double, Dim>> grads;
  if (linear)
    grads = transferGradients(d, f);

  const BO oldLocal = d.local();  // snapshot for the remap + flag lookup
  // coarsen sibling groups whose every child is flagged kCoarsen
  d.local().coarsenIf([&](Code parent, unsigned pl) {
    for (unsigned oct = 0; oct < (1u << Dim); ++oct) {
      Code cc = M::from_code(parent).child(pl, oct).code();
      Index ci = oldLocal.find(cc);
      if (ci < 0 || flags[static_cast<std::size_t>(ci)] != kCoarsen)
        return false;
    }
    return true;
  });
  // refine leaves flagged kRefine
  d.local().refineIf([&](Code c, unsigned) {
    Index ci = oldLocal.find(c);
    return ci >= 0 && flags[static_cast<std::size_t>(ci)] == kRefine;
  });
  d.balance();  // cross-block 2:1 to a global fixpoint
  return transferField(oldLocal, f, d.local(), linear, linear ? &grads : nullptr);
}

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_DISTRIBUTED_ADAPT_HPP

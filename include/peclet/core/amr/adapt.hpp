// transport-core — dynamic (solution-adaptive) AMR for a BlockOctree: flag from an
// error indicator, refine/coarsen + 2:1 balance, and remap the field onto the new
// mesh.
//
// The core reusable piece is `transferField` — a conservative field remap between
// any two octrees on the same domain:
//   * a new leaf at the same level as the covering old leaf  → copy;
//   * a new leaf finer than the old (refinement / balance)   → prolong (piecewise-
//     constant, or minmod-limited linear — both conservative because the children of
//     a cell are symmetric about its centroid, so the linear term integrates to 0);
//   * a new leaf coarser than the old (coarsening)           → volume-weighted average
//     of the old leaves it covers.
// This one operation powers refinement transfer, coarsening transfer, and the extra
// cells inserted by 2:1 balancing, uniformly.
//
// `adaptField` applies a per-leaf flag set (refine / keep / coarsen) to produce the
// new octree and the remapped field; `adapt` is the all-in-one driver using the
// Löhner indicator (indicators.hpp). The level convention is the octree's: refining
// *decreases* level toward 0 (finest), coarsening increases it toward `lmax` (root).
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON.
#ifndef PECLET_CORE_AMR_ADAPT_HPP
#define PECLET_CORE_AMR_ADAPT_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/indicators.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::amr {

namespace detail {
inline double minmod(double a, double b) {
  if (a * b <= 0.0) return 0.0;
  return (std::fabs(a) < std::fabs(b)) ? a : b;
}
}  // namespace detail

/// Conservative remap of a leaf field from `oldT` to `newT` (same domain). `linear`
/// selects minmod-limited linear prolongation (more accurate, still conservative and
/// non-overshooting) vs piecewise-constant injection for cells that get finer.
template <int Dim, unsigned Bits>
std::vector<double> transferField(const BlockOctree<Dim, Bits>& oldT,
                                  const std::vector<double>& oldF,
                                  const BlockOctree<Dim, Bits>& newT, bool linear = true) {
  using BO = BlockOctree<Dim, Bits>;
  using Code = typename BO::Code;
  using Coord = typename BO::Coord;
  using M = typename BO::M;

  auto centroid = [](const BO& t, Index i) {
    auto o = M::from_code(t.code(i)).decode();
    const double s = static_cast<double>(Coord(1) << t.level(i));
    std::array<double, Dim> c{};
    for (int d = 0; d < Dim; ++d) c[d] = static_cast<double>(o[d]) + 0.5 * s;
    return c;
  };

  const Index no = oldT.numLeaves();
  // Per-old-leaf minmod gradient (per fine-coordinate unit) for linear prolongation.
  std::vector<std::array<double, Dim>> grad;
  if (linear) {
    grad.assign(static_cast<std::size_t>(no), std::array<double, Dim>{});
    for (Index i = 0; i < no; ++i) {
      auto ci = centroid(oldT, i);
      const double ui = oldF[static_cast<std::size_t>(i)];
      for (int axis = 0; axis < Dim; ++axis) {
        const Index jp = oldT.faceNeighbor(i, axis, +1);
        const Index jm = oldT.faceNeighbor(i, axis, -1);
        if (jp < 0 || jm < 0) continue;
        auto cp = centroid(oldT, jp);
        auto cm = centroid(oldT, jm);
        const double sp = (oldF[static_cast<std::size_t>(jp)] - ui) / (cp[axis] - ci[axis]);
        const double sm = (ui - oldF[static_cast<std::size_t>(jm)]) / (ci[axis] - cm[axis]);
        grad[static_cast<std::size_t>(i)][axis] = detail::minmod(sp, sm);
      }
    }
  }

  const Index nn = newT.numLeaves();
  std::vector<double> nf(static_cast<std::size_t>(nn), 0.0);
  std::vector<Index> n2o(static_cast<std::size_t>(nn), -1);  // prolong: new cell → source old leaf
  for (Index j = 0; j < nn; ++j) {
    const Code cj = newT.code(j);
    const unsigned Lj = newT.level(j);
    const Index o = oldT.find(cj);
    if (o < 0) continue;
    const unsigned Lo = oldT.level(o);
    if (Lo == Lj) {
      nf[static_cast<std::size_t>(j)] = oldF[static_cast<std::size_t>(o)];  // copy
    } else if (Lo > Lj) {
      // new cell is finer → prolong from old leaf o
      n2o[static_cast<std::size_t>(j)] = o;
      if (!linear) {
        nf[static_cast<std::size_t>(j)] = oldF[static_cast<std::size_t>(o)];
      } else {
        auto co = centroid(oldT, o);
        auto cn = centroid(newT, j);
        double v = oldF[static_cast<std::size_t>(o)];
        for (int d = 0; d < Dim; ++d) v += grad[static_cast<std::size_t>(o)][d] * (cn[d] - co[d]);
        nf[static_cast<std::size_t>(j)] = v;
      }
    } else {
      // new cell is coarser → volume-weighted average of the old leaves it covers
      // (descendants of cj are a contiguous Morton run starting at o).
      auto oj = M::from_code(cj).decode();
      const Coord sj = Coord(Coord(1) << Lj);
      double vol = 0.0, acc = 0.0;
      for (Index k = o; k < no; ++k) {
        auto ok = M::from_code(oldT.code(k)).decode();
        bool inside = true;
        for (int d = 0; d < Dim; ++d)
          if (ok[d] < oj[d] || ok[d] >= oj[d] + sj) {
            inside = false;
            break;
          }
        if (!inside) break;
        const double w = std::pow(2.0, static_cast<double>(Dim * static_cast<int>(oldT.level(k))));
        acc += w * oldF[static_cast<std::size_t>(k)];
        vol += w;
      }
      nf[static_cast<std::size_t>(j)] = (vol > 0.0) ? acc / vol : 0.0;
    }
  }

  // Conservation fix for prolongation: a cell's new children must volume-average back
  // to the parent value. Exact for PC (already true) and for linear under uniform
  // refinement; this correction also restores it when 2:1 balance refines a cell
  // non-uniformly (so the symmetric-offset cancellation no longer holds). One scalar
  // shift per source old leaf — preserves the reconstructed shape, fixes the mean.
  if (linear) {
    std::vector<double> sv(static_cast<std::size_t>(no), 0.0), vv(static_cast<std::size_t>(no), 0.0);
    for (Index j = 0; j < nn; ++j) {
      const Index o = n2o[static_cast<std::size_t>(j)];
      if (o < 0) continue;
      const double w = std::pow(2.0, static_cast<double>(Dim * static_cast<int>(newT.level(j))));
      sv[static_cast<std::size_t>(o)] += w * nf[static_cast<std::size_t>(j)];
      vv[static_cast<std::size_t>(o)] += w;
    }
    for (Index j = 0; j < nn; ++j) {
      const Index o = n2o[static_cast<std::size_t>(j)];
      if (o < 0 || vv[static_cast<std::size_t>(o)] <= 0.0) continue;
      nf[static_cast<std::size_t>(j)] +=
          oldF[static_cast<std::size_t>(o)] - sv[static_cast<std::size_t>(o)] / vv[static_cast<std::size_t>(o)];
    }
  }
  return nf;
}

/// Per-leaf adaptation flags from an indicator: refine where `ind > refineThresh`
/// (and the leaf can go finer, level > finestLevel), coarsen where `ind <
/// coarsenThresh` (and it can go coarser, level < lmax), else keep.
enum AdaptFlag : int { kCoarsen = -1, kKeep = 0, kRefine = 1 };

template <int Dim, unsigned Bits>
std::vector<int> flagByIndicator(const BlockOctree<Dim, Bits>& t, const std::vector<double>& ind,
                                 double refineThresh, double coarsenThresh,
                                 unsigned finestLevel = 0) {
  const Index n = t.numLeaves();
  std::vector<int> f(static_cast<std::size_t>(n), kKeep);
  for (Index i = 0; i < n; ++i) {
    const unsigned L = t.level(i);
    const double e = ind[static_cast<std::size_t>(i)];
    if (e > refineThresh && L > finestLevel)
      f[static_cast<std::size_t>(i)] = kRefine;
    else if (e < coarsenThresh && L < t.lmax())
      f[static_cast<std::size_t>(i)] = kCoarsen;
  }
  return f;
}

template <int Dim, unsigned Bits>
struct AdaptResult {
  BlockOctree<Dim, Bits> octree;
  std::vector<double> field;
};

/// Apply adaptation `flags` (one level of refine/coarsen) to `t` carrying field `f`,
/// then 2:1-balance and remap. A sibling group coarsens only if *all* its children
/// are flagged kCoarsen; refine flags are honoured per leaf. Flags are looked up by
/// the original leaf's code, so they survive the intermediate index changes.
template <int Dim, unsigned Bits>
AdaptResult<Dim, Bits> adaptField(const BlockOctree<Dim, Bits>& t, const std::vector<double>& f,
                                  const std::vector<int>& flags, bool linear = true) {
  using BO = BlockOctree<Dim, Bits>;
  using Code = typename BO::Code;
  using M = typename BO::M;
  BO nt = t;
  // coarsen sibling groups whose every child is flagged kCoarsen
  nt.coarsenIf([&](Code parent, unsigned pl) {
    for (unsigned oct = 0; oct < (1u << Dim); ++oct) {
      Code cc = M::from_code(parent).child(pl, oct).code();
      Index ci = t.find(cc);
      if (ci < 0 || flags[static_cast<std::size_t>(ci)] != kCoarsen) return false;
    }
    return true;
  });
  // refine leaves flagged kRefine
  nt.refineIf([&](Code c, unsigned) {
    Index ci = t.find(c);
    return ci >= 0 && flags[static_cast<std::size_t>(ci)] == kRefine;
  });
  nt.balance2to1();
  std::vector<double> nf = transferField(t, f, nt, linear);
  return AdaptResult<Dim, Bits>{std::move(nt), std::move(nf)};
}

/// All-in-one solution-adaptive step: Löhner indicator → flags → adaptField.
template <int Dim, unsigned Bits>
AdaptResult<Dim, Bits> adapt(const BlockOctree<Dim, Bits>& t, const std::vector<double>& f,
                             double refineThresh, double coarsenThresh, unsigned finestLevel = 0,
                             double eps = 0.01, bool linear = true) {
  auto ind = lohnerIndicator(t, f, eps);
  auto flags = flagByIndicator(t, ind, refineThresh, coarsenThresh, finestLevel);
  return adaptField(t, f, flags, linear);
}

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_ADAPT_HPP

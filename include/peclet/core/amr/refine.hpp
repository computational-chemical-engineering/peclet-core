// core — refinement criteria for a BlockOctree.
//
// Topology-mutation helpers that drive BlockOctree::refineIf from a geometric
// criterion. The headline one is SDF-driven: refine leaves the solid surface
// passes through (plus a band) down to a target level, leaving the interior /
// far field coarse — the usual AMR pattern for the suite's cut-cell IBM. Reuses
// the shared peclet::core::geom SDF (any callable returning a signed distance at a world
// point) and the AmrGeometry world mapping.
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON.
#ifndef PECLET_CORE_AMR_REFINE_HPP
#define PECLET_CORE_AMR_REFINE_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::amr {

/// Refine leaves near an SDF interface down to `targetLevel`.
///
/// A leaf is refined while its level exceeds `targetLevel` and the surface lies
/// within the cell or a band around it: |phi(center)| <= halfDiagonal + band*h0,
/// where halfDiagonal = 0.5*sqrt(Dim)*cellWidth. Iterated until no eligible leaf
/// remains; returns the number of refinements performed. `sdf` is any callable
/// `Real(const Vec<Dim>&)` (e.g. a lambda over a peclet::core::geom shape). Optionally
/// restore 2:1 balance afterwards.
template <int Dim, unsigned Bits, class SdfFn>
Index refineToSdf(BlockOctree<Dim, Bits>& t, const AmrGeometry<Dim>& geo, SdfFn&& sdf,
                  unsigned targetLevel, Real band = 1.0, bool balance = true) {
  using Code = typename BlockOctree<Dim, Bits>::Code;
  const Real halfDiagFactor = 0.5 * std::sqrt(static_cast<Real>(Dim));
  Index total = 0;
  for (;;) {
    std::vector<Code> toRefine;
    for (Index i = 0; i < t.numLeaves(); ++i) {
      const unsigned L = t.level(i);
      if (L <= targetLevel)
        continue;
      auto b = t.bounds(i);
      Vec<Dim> c = geo.center(b);
      const Real width = geo.leafSize(L);
      if (std::fabs(static_cast<Real>(sdf(c))) <= halfDiagFactor * width + band * geo.h0)
        toRefine.push_back(t.code(i));
    }
    if (toRefine.empty())
      break;
    std::sort(toRefine.begin(), toRefine.end());
    toRefine.erase(std::unique(toRefine.begin(), toRefine.end()), toRefine.end());
    total += t.refineIf(
        [&](Code c, unsigned) { return std::binary_search(toRefine.begin(), toRefine.end(), c); });
  }
  if (balance)
    total += t.balance2to1();
  return total;
}

/// Graded variant of `refineToSdf`: a per-point TARGET LEVEL function instead of one global
/// `targetLevel` — the mesh-generator side of the mixed-level cut band
/// (docs/amr_mixed_level_cut_band_plan.md §7). Cut cells then live at several levels: fine in
/// narrow throats / near contacts, coarse on smooth wide caps.
///
/// `targetFn` is any callable `unsigned(const Vec<Dim>&)` giving the coarsest acceptable level
/// at a world point (0 = finest). A leaf at level L is refined while `L > targetFn(center)` and
/// the surface lies within `halfDiagonal + band*h(L-1)` of the center — the band margin is
/// measured in cells of the level being CREATED (not in h0 as in `refineToSdf`), which is what
/// keeps the margin meaningful where the target level is coarse. Iterated to a fixpoint;
/// optionally 2:1-balanced afterwards. Returns the number of refinements performed.
///
/// Requires the sampled ghost overlay (`AmrFlow::setGhostSampled(true)`) on the solver side: the
/// classic overlay contracts a uniform finest band and throws on the level jumps this produces.
template <int Dim, unsigned Bits, class SdfFn, class TargetFn>
Index refineToSdfGraded(BlockOctree<Dim, Bits>& t, const AmrGeometry<Dim>& geo, SdfFn&& sdf,
                        TargetFn&& targetFn, Real band = 2.0, bool balance = true) {
  using Code = typename BlockOctree<Dim, Bits>::Code;
  const Real halfDiagFactor = 0.5 * std::sqrt(static_cast<Real>(Dim));
  Index total = 0;
  for (;;) {
    std::vector<Code> toRefine;
    for (Index i = 0; i < t.numLeaves(); ++i) {
      const unsigned L = t.level(i);
      if (L == 0)
        continue;
      auto b = t.bounds(i);
      Vec<Dim> c = geo.center(b);
      const Real width = geo.leafSize(L);
      // Band predicate first: targetFn is the expensive one (a gap/medial-axis field, often a
      // Python callable) and only band cells can ever be refined.
      if (std::fabs(static_cast<Real>(sdf(c))) > halfDiagFactor * width + band * 0.5 * width)
        continue;
      if (L > targetFn(c))
        toRefine.push_back(t.code(i));
    }
    if (toRefine.empty())
      break;
    std::sort(toRefine.begin(), toRefine.end());
    toRefine.erase(std::unique(toRefine.begin(), toRefine.end()), toRefine.end());
    total += t.refineIf(
        [&](Code c, unsigned) { return std::binary_search(toRefine.begin(), toRefine.end(), c); });
  }
  if (balance)
    total += t.balance2to1();
  return total;
}

/// The plan's §7 criterion 1 as a ready-made `targetFn` for `refineToSdfGraded`: the GAP-WIDTH
/// FLOOR. A cell at level L may be cut only where the local fluid gap satisfies
/// `gap >= n * h_L` (n = 4 per the M1/M2 measurements) — so the target level at a point is the
/// COARSEST level still clearing the floor, clamped to [0, coarsestLevel]. This is the AMReX
/// multi-valued-cell rule inverted: coarsening may never merge or disconnect fluid (plan D3).
///
/// `gapFn(x)` is the local gap proxy — for a sphere packing the two-closest-surfaces sum
/// `d1 + d2` (distance to the nearest surface plus the second-nearest), which is the pore/throat
/// width along the line joining them; a medial-axis or `peclet.pnm` throat-radius field can be
/// substituted verbatim.
template <int Dim, class GapFn>
auto gapFloorTarget(GapFn&& gapFn, Real h0, unsigned coarsestLevel, Real n = 4.0) {
  return [gapFn = std::forward<GapFn>(gapFn), h0, coarsestLevel, n](const Vec<Dim>& p) -> unsigned {
    const Real g = static_cast<Real>(gapFn(p));
    unsigned L = 0;
    while (L < coarsestLevel && h0 * static_cast<Real>(Index(1) << (L + 1)) * n <= g)
      ++L;
    return L;
  };
}

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_REFINE_HPP

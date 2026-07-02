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

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_REFINE_HPP

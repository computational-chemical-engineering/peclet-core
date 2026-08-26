// SDF-driven refinement (peclet::core::amr::refineToSdf): refining a uniform coarse block
// around a sphere must (a) drive every surface-crossing leaf to the target level,
// (b) leave interior/far-field leaves coarse (genuine adaptivity, far fewer cells
// than a uniform fine grid), and (c) stay 2:1 balanced.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "peclet/core/common/types.hpp"
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <algorithm>
#include <cmath>
#include <utility>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/amr/refine.hpp"
#include "peclet/core/geom/sdf.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

void run() {
  using BO = BlockOctree<3, 21>;
  // 1 root cell, lmax=5 -> a 32^3 fine domain available, starting fully coarse.
  BO t(IVec<3>{1, 1, 1}, 5);
  PECLET_CORE_CHECK_EQ((long long)t.numLeaves(), 1LL);

  AmrGeometry<3> geo;
  geo.origin = {0.0, 0.0, 0.0};
  geo.h0 = 1.0;  // fine cell = 1 world unit; domain is [0,32]^3

  peclet::core::geom::Sphere sph;
  sph.center = {16.0, 16.0, 16.0};
  sph.radius = 8.0;
  auto sdf = [&](const Vec<3>& p) { return sph.eval(p); };

  const unsigned target = 1;  // refine the surface band down to level 1 (2-wide cells)
  Index nref = refineToSdf(t, geo, sdf, target, /*band=*/1.0, /*balance=*/true);
  PECLET_CORE_CHECK(nref > 0);
  PECLET_CORE_CHECK(t.isBalanced());

  // (a) every leaf the surface actually passes through is at the target level.
  // (b) adaptivity: far fewer leaves than a uniform grid at the target level.
  const Real halfDiagFactor = 0.5 * std::sqrt(3.0);
  int crossing = 0;
  bool allCrossingFine = true;
  for (Index i = 0; i < t.numLeaves(); ++i) {
    auto b = t.bounds(i);
    Vec<3> c = geo.center(b);
    Real width = geo.leafSize(t.level(i));
    if (std::fabs(sph.eval(c)) <= halfDiagFactor * width) {  // surface within the cell
      ++crossing;
      if (t.level(i) != target)
        allCrossingFine = false;
    }
  }
  PECLET_CORE_CHECK(crossing > 0);
  PECLET_CORE_CHECK(allCrossingFine);

  // Uniform grid at level `target` would be (32 / 2)^3 = 4096 cells; adaptivity
  // must beat that comfortably.
  PECLET_CORE_CHECK(t.numLeaves() < 4096);

  // Volume is conserved (refinement only splits).
  long vol = 0;
  for (Index i = 0; i < t.numLeaves(); ++i) {
    long s = 1L << t.level(i);
    vol += s * s * s;
  }
  PECLET_CORE_CHECK_EQ((long long)vol, (long long)(32L * 32L * 32L));
}

/// Graded variant (refineToSdfGraded + gapFloorTarget): a per-point target level must put cut
/// cells at SEVERAL levels — the mesh-generator side of the mixed-level cut band
/// (docs/amr_mixed_level_cut_band_plan.md §7).
void runGraded() {
  using BO = BlockOctree<3, 21>;
  AmrGeometry<3> geo;
  geo.origin = {0.0, 0.0, 0.0};
  geo.h0 = 1.0;  // domain [0,64]^3 at lmax=6

  peclet::core::geom::Sphere sph;
  sph.center = {32.0, 32.0, 32.0};
  sph.radius = 16.0;
  auto sdf = [&](const Vec<3>& p) { return sph.eval(p); };

  // (1) Constant target reduces to a uniform finest band: no cut cell above level 0.
  BO tu(IVec<3>{1, 1, 1}, 6);
  refineToSdfGraded(
      tu, geo, sdf, [](const Vec<3>&) { return 0u; }, /*band=*/2.0, /*balance=*/true);
  PECLET_CORE_CHECK(tu.isBalanced());

  // (2) Latitude two-level map: fine below z = 32, one level coarser above. Cut cells must then
  // exist at BOTH levels (the seam the sampled overlay exists for).
  BO tg(IVec<3>{1, 1, 1}, 6);
  refineToSdfGraded(
      tg, geo, sdf, [](const Vec<3>& p) { return p[2] < 32.0 ? 0u : 1u; }, 2.0, true);
  PECLET_CORE_CHECK(tg.isBalanced());

  const Real hdf = 0.5 * std::sqrt(3.0);
  auto cutLevels = [&](const BO& t) {
    int lo = 99, hi = -1;
    for (Index i = 0; i < t.numLeaves(); ++i) {
      Vec<3> c = geo.center(t.bounds(i));
      if (std::fabs(sph.eval(c)) <= hdf * geo.leafSize(t.level(i))) {
        lo = std::min(lo, static_cast<int>(t.level(i)));
        hi = std::max(hi, static_cast<int>(t.level(i)));
      }
    }
    return std::pair<int, int>{lo, hi};
  };
  auto [ulo, uhi] = cutLevels(tu);
  PECLET_CORE_CHECK_EQ(ulo, 0);
  PECLET_CORE_CHECK_EQ(uhi, 0);  // uniform band: every cut cell finest
  auto [glo, ghi] = cutLevels(tg);
  PECLET_CORE_CHECK_EQ(glo, 0);
  PECLET_CORE_CHECK_EQ(ghi, 1);  // graded: cut cells at two levels
  PECLET_CORE_CHECK(tg.numLeaves() < tu.numLeaves());  // and it is cheaper

  // (3) gapFloorTarget: the coarsest level clearing gap >= n*h_L, clamped.
  auto tgt = gapFloorTarget<3>([](const Vec<3>& p) { return p[0]; }, /*h0=*/1.0,
                               /*coarsestLevel=*/3, /*n=*/4.0);
  PECLET_CORE_CHECK_EQ((int)tgt(Vec<3>{3.0, 0, 0}), 0);    // gap 3 < 4*h_1=8 -> finest
  PECLET_CORE_CHECK_EQ((int)tgt(Vec<3>{8.0, 0, 0}), 1);    // 8 >= 4*h_1, < 4*h_2=16
  PECLET_CORE_CHECK_EQ((int)tgt(Vec<3>{20.0, 0, 0}), 2);   // 16 <= 20 < 32
  PECLET_CORE_CHECK_EQ((int)tgt(Vec<3>{1e6, 0, 0}), 3);    // clamped at coarsestLevel
}

}  // namespace

int main() {
  run();
  runGraded();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping AMR SDF refinement test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

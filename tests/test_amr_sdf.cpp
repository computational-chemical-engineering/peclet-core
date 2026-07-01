// SDF-driven refinement (peclet::core::amr::refineToSdf): refining a uniform coarse block
// around a sphere must (a) drive every surface-crossing leaf to the target level,
// (b) leave interior/far-field leaves coarse (genuine adaptivity, far fewer cells
// than a uniform fine grid), and (c) stay 2:1 balanced.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"
#include "peclet/core/common/types.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>

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
      if (t.level(i) != target) allCrossingFine = false;
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

}  // namespace

int main() {
  run();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping AMR SDF refinement test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

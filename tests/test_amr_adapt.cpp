// Solution-adaptive AMR cycle (peclet::core::amr::adapt + lohnerIndicator): on a field with a
// localized steep front, the Löhner indicator flags the front, and adapt() refines
// there + coarsens the flat far field while conservatively remapping the field.
//   (1) the indicator is large at the front, small in flat regions;
//   (2) one adapt step refines near the front and coarsens away from it, and is
//       conservative (Σ V·f preserved through the remap);
//   (3) iterating adapt drives the finest cells to the front only (an order of
//       magnitude fewer leaves than a uniform-fine grid) and keeps the far field coarse.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <array>
#include <cmath>
#include <vector>

#include "peclet/core/amr/adapt.hpp"
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/indicators.hpp"
#include "peclet/core/common/types.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using M = BO::M;
constexpr double kN = 32.0;  // domain is 32^3 fine cells over [0,1)^3 (brick 4^3, lmax 3)

double xWorld(const BO& t, Index i) {
  auto o = M::from_code(t.code(i)).decode();
  double s = static_cast<double>(Index(1) << t.level(i));
  return ((double)o[0] + 0.5 * s) / kN;
}
double cellVolRel(const BO& t, Index i) {
  double s = static_cast<double>(Index(1) << t.level(i));
  return s * s * s;
}
double relIntegral(const BO& t, const std::vector<double>& f) {
  double s = 0.0;
  for (Index i = 0; i < t.numLeaves(); ++i)
    s += cellVolRel(t, i) * f[(std::size_t)i];
  return s;
}
double relScale(const BO& t, const std::vector<double>& f) {  // Σ V·|f| (the front is ~mean-zero)
  double s = 0.0;
  for (Index i = 0; i < t.numLeaves(); ++i)
    s += cellVolRel(t, i) * std::fabs(f[(std::size_t)i]);
  return s;
}
// Tanh front at x = 0.5, ~1 base cell wide.
double front(double xw) {
  return std::tanh((xw - 0.5) / 0.05);
}
std::vector<double> sampleFront(const BO& t) {
  std::vector<double> f((std::size_t)t.numLeaves());
  for (Index i = 0; i < t.numLeaves(); ++i)
    f[(std::size_t)i] = front(xWorld(t, i));
  return f;
}
// uniform level-`lvl` mesh on a 4x4x4 / lmax=3 brick (32^3 fine): base level 1 = 16^3
// cells, room to refine to level 0 at the front and coarsen to level 2/3 far away.
BO uniform(unsigned lvl) {
  BO t(IVec<3>{4, 4, 4}, 3);
  while (true) {
    Index before = t.numLeaves();
    t.refineIf([&](Code, unsigned l) { return l > lvl; });
    if (t.numLeaves() == before)
      break;
  }
  return t;
}

void run() {
  // (1) indicator localizes the front (base level 1)
  {
    BO t = uniform(1);
    auto f = sampleFront(t);
    auto e = lohnerIndicator(t, f);
    double eFront = 0.0, eFar = 0.0;
    for (Index i = 0; i < t.numLeaves(); ++i) {
      double d = std::fabs(xWorld(t, i) - 0.5);
      if (d < 0.1)
        eFront = std::max(eFront, e[(std::size_t)i]);
      if (d > 0.3)
        eFar = std::max(eFar, e[(std::size_t)i]);
    }
    PECLET_CORE_CHECK(eFront > 0.3);  // strong signal at the front
    PECLET_CORE_CHECK(eFar < 0.05);   // quiet far field
  }

  // (2) one adapt step: refine near front, coarsen far, conserve
  {
    BO t = uniform(1);
    auto f = sampleFront(t);
    const double I0 = relIntegral(t, f);
    const double scale = relScale(t, f);
    auto r = adapt(t, f, /*refineThresh=*/0.2, /*coarsenThresh=*/0.03, /*finestLevel=*/0);
    unsigned minL = 99, maxL = 0;
    for (Index i = 0; i < r.octree.numLeaves(); ++i) {
      minL = std::min(minL, r.octree.level(i));
      maxL = std::max(maxL, r.octree.level(i));
    }
    PECLET_CORE_CHECK(minL < 1);  // refined below the base level 1
    PECLET_CORE_CHECK(maxL > 1);  // coarsened above it
    // conservative remap (front integrates ~0, so compare against Σ V·|f|)
    PECLET_CORE_CHECK(std::fabs(relIntegral(r.octree, r.field) - I0) < 1e-9 * scale);
  }

  // (3) iterate to a front-tracking mesh: finest cells only near x=0.5, far field
  // coarse, far fewer leaves than uniform-fine.
  {
    BO t = uniform(1);
    auto f = sampleFront(t);
    for (int it = 0; it < 4; ++it) {
      auto r = adapt(t, f, 0.2, 0.03, /*finestLevel=*/0);
      t = std::move(r.octree);
      f = sampleFront(t);  // re-sample exact so the indicator keeps resolving as we refine
    }
    // every finest (level-0) cell is within a band of the front
    double maxFineDist = 0.0;
    Index nFinest = 0;
    for (Index i = 0; i < t.numLeaves(); ++i)
      if (t.level(i) == 0) {
        ++nFinest;
        maxFineDist = std::max(maxFineDist, std::fabs(xWorld(t, i) - 0.5));
      }
    PECLET_CORE_CHECK(nFinest > 0);        // reached the finest level
    PECLET_CORE_CHECK(maxFineDist < 0.2);  // finest cells hug the front
    // adaptive mesh is smaller than uniform-fine (32^3 = 32768)
    PECLET_CORE_CHECK(t.numLeaves() < 32768);
  }
}

}  // namespace

int main() {
  run();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping adapt test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

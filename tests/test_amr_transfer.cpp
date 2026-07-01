// Field remap between octrees (peclet::core::amr::transferField), the core of dynamic AMR:
//   (1) conservation — the volume-weighted integral Σ V·f is preserved when the mesh
//       is refined (piecewise-constant prolong) and when it is coarsened (volume
//       average);
//   (2) round-trip identity — refine-all then coarsen-all returns the original mesh
//       and the original field exactly (PC);
//   (3) restrict exactness — coarsening a linear field is exact (volume average of a
//       linear function = its value at the cell centroid);
//   (4) linear > PC accuracy — minmod-limited linear prolongation beats PC injection
//       on a smooth (quadratic) field, and is itself conservative.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <array>
#include <cmath>
#include <vector>

#include "peclet/core/amr/adapt.hpp"
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/common/types.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using M = BO::M;

std::array<double, 3> centroid(const BO& t, Index i) {
  auto o = M::from_code(t.code(i)).decode();
  double s = static_cast<double>(Index(1) << t.level(i));
  return {(double)o[0] + 0.5 * s, (double)o[1] + 0.5 * s, (double)o[2] + 0.5 * s};
}
double cellVolRel(const BO& t, Index i) {
  double s = static_cast<double>(Index(1) << t.level(i));
  return s * s * s;
}
double relIntegral(const BO& t, const std::vector<double>& f) {
  double s = 0.0;
  for (Index i = 0; i < t.numLeaves(); ++i) s += cellVolRel(t, i) * f[(std::size_t)i];
  return s;
}
template <class Fn>
std::vector<double> sample(const BO& t, Fn&& fn) {
  std::vector<double> f((std::size_t)t.numLeaves());
  for (Index i = 0; i < t.numLeaves(); ++i) {
    auto c = centroid(t, i);
    f[(std::size_t)i] = fn(c[0], c[1], c[2]);
  }
  return f;
}

BO graded() {
  BO t(IVec<3>{2, 2, 2}, 3);  // 16^3 fine
  for (int k = 0; k < 2; ++k) t.refineIf([](Code, unsigned l) { return l > 0; });
  t.refineIf([&](Code c, unsigned) {
    auto o = M::from_code(c).decode();
    return o[0] < 8 && o[1] < 8 && o[2] < 8;
  });
  t.balance2to1();
  return t;
}

void run() {
  auto linear = [](double x, double y, double z) { return 1.0 + 2.0 * x - 1.5 * y + 0.7 * z; };

  // (1) conservation under refine and coarsen on a graded mesh
  {
    BO t = graded();
    auto f = sample(t, linear);
    const double I0 = relIntegral(t, f);

    BO tr = t;
    tr.refineIf([](Code, unsigned l) { return l > 0; });  // refine all eligible one level
    auto fr = transferField(t, f, tr, /*linear=*/false);  // PC prolong
    PECLET_CORE_CHECK(std::fabs(relIntegral(tr, fr) - I0) < 1e-9 * (std::fabs(I0) + 1e-30));

    BO tc = t;
    tc.coarsenIf([](Code, unsigned) { return true; });  // coarsen all full groups
    auto fc = transferField(t, f, tc, false);
    PECLET_CORE_CHECK(std::fabs(relIntegral(tc, fc) - I0) < 1e-9 * (std::fabs(I0) + 1e-30));
  }

  // (2) round-trip identity on a uniform mesh: refine-all then coarsen-all
  {
    BO t(IVec<3>{2, 2, 2}, 3);
    t.refineIf([](Code, unsigned l) { return l > 0; });  // uniform level 2
    auto f = sample(t, linear);

    BO tr = t;
    tr.refineIf([](Code, unsigned l) { return l > 0; });
    auto fr = transferField(t, f, tr, false);
    BO tc = tr;
    tc.coarsenIf([](Code, unsigned) { return true; });
    auto fc = transferField(tr, fr, tc, false);

    PECLET_CORE_CHECK(tc.numLeaves() == t.numLeaves());
    double maxd = 0.0;
    for (Index i = 0; i < t.numLeaves(); ++i) {
      Index k = tc.find(t.code(i));
      maxd = std::max(maxd, std::fabs(fc[(std::size_t)k] - f[(std::size_t)i]));
    }
    PECLET_CORE_CHECK(maxd < 1e-12);
  }

  // (3) restrict of a linear field is exact (volume avg = centroid value)
  {
    BO fine(IVec<3>{2, 2, 2}, 3);
    fine.refineIf([](Code, unsigned l) { return l > 0; });  // level 2
    fine.refineIf([](Code, unsigned l) { return l > 0; });  // level 1
    auto f = sample(fine, linear);
    BO coarse = fine;
    coarse.coarsenIf([](Code, unsigned) { return true; });  // level 2
    auto fc = transferField(fine, f, coarse, false);
    auto exact = sample(coarse, linear);
    double maxd = 0.0;
    for (Index i = 0; i < coarse.numLeaves(); ++i)
      maxd = std::max(maxd, std::fabs(fc[(std::size_t)i] - exact[(std::size_t)i]));
    PECLET_CORE_CHECK(maxd < 1e-9);
  }

  // (4) minmod-linear prolong beats PC on a smooth quadratic, and conserves
  {
    // Non-symmetric (monotonic over the domain) so the minmod slopes don't vanish by
    // symmetry — a symmetric bump would make minmod pick 0 and revert linear→PC.
    auto quad = [](double x, double y, double z) {
      return x * x + 0.7 * y * y + 0.4 * z * z + 3.0 * x + y + z;
    };
    BO coarse(IVec<3>{2, 2, 2}, 3);
    coarse.refineIf([](Code, unsigned l) { return l > 0; });  // level 2
    auto fc = sample(coarse, quad);
    const double I0 = relIntegral(coarse, fc);

    BO fine = coarse;
    fine.refineIf([](Code, unsigned l) { return l > 0; });  // level 1
    auto fpc = transferField(coarse, fc, fine, false);
    auto flin = transferField(coarse, fc, fine, true);
    auto exact = sample(fine, quad);

    double ePC = 0.0, eLIN = 0.0;
    for (Index i = 0; i < fine.numLeaves(); ++i) {
      ePC += std::fabs(fpc[(std::size_t)i] - exact[(std::size_t)i]);
      eLIN += std::fabs(flin[(std::size_t)i] - exact[(std::size_t)i]);
    }
    PECLET_CORE_CHECK(eLIN < ePC);                                                    // more accurate
    PECLET_CORE_CHECK(std::fabs(relIntegral(fine, flin) - I0) < 1e-9 * std::fabs(I0)); // still conservative
  }
}

}  // namespace

int main() {
  run();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping transfer test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

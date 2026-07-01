// Barnes–Hut over the octree (peclet::core::amr::BarnesHut, particle-tree mode):
//   (1) theta = 0 recurses fully, so the approximate acceleration equals the
//       direct O(N^2) sum to round-off;
//   (2) theta = 0.3 keeps the per-particle acceleration within a few percent of
//       direct (controlled multipole error).
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <cstdint>
#include <vector>

#include "peclet/core/amr/barnes_hut.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/common/types.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

double pseudo(std::uint64_t& s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>((s >> 11) & 0xFFFFF) / static_cast<double>(0x100000);
}

double maxRelErr(const std::vector<Vec<3>>& a, const std::vector<Vec<3>>& b) {
  double me = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double da = 0, na = 0;
    for (int d = 0; d < 3; ++d) {
      double diff = a[i][d] - b[i][d];
      da += diff * diff;
      na += b[i][d] * b[i][d];
    }
    double rel = std::sqrt(da) / (std::sqrt(na) + 1e-30);
    if (rel > me) me = rel;
  }
  return me;
}

void run() {
  const int N = 300;
  std::vector<Vec<3>> pos(N);
  std::vector<double> mass(N, 1.0);
  std::uint64_t s = 987654321;
  for (int i = 0; i < N; ++i) pos[i] = {pseudo(s), pseudo(s), pseudo(s)};  // in [0,1)^3

  AmrGeometry<3> geo;  // origin 0, h0 1 ... but box must span the particles
  geo.origin = {0.0, 0.0, 0.0};
  const unsigned lmax = 6;       // 64^3 fine cells over the unit box
  geo.h0 = 1.0 / static_cast<double>(1u << lmax);

  // Direct reference.
  BarnesHut<3> bh;
  bh.build(pos, mass, geo, lmax, /*theta=*/0.0, /*soft=*/0.02);
  std::vector<Vec<3>> direct(N);
  for (int i = 0; i < N; ++i) direct[i] = bh.accelerationDirect(i);

  // (1) theta = 0 == direct sum.
  std::vector<Vec<3>> exact0 = bh.accelerations();
  PECLET_CORE_CHECK(maxRelErr(exact0, direct) < 1e-9);

  // (2) theta = 0.3 within a few percent.
  BarnesHut<3> bh2;
  bh2.build(pos, mass, geo, lmax, /*theta=*/0.3, /*soft=*/0.02);
  std::vector<Vec<3>> approx = bh2.accelerations();
  double err = maxRelErr(approx, direct);
  PECLET_CORE_CHECK(err < 0.05);
}

}  // namespace

int main() {
  run();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping Barnes-Hut test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

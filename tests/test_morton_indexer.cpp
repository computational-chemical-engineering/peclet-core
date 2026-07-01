// Serial correctness of the Morton/Z-order cell indexer (peclet::core::decomp::MortonIndexer):
//  - codeOf / multiIndex round-trip over a block (origin-relative),
//  - codeOf agrees with a direct morton::Morton encode,
//  - codes are a bijection over the block's cells (no collisions),
//  - neighborCode steps exactly one cell along an axis in Morton space.
//
// Guarded by PECLET_CORE_HAVE_MORTON: without the morton sibling checkout this is a no-op
// pass (the indexer header defines nothing).
#include "test_util.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/morton_indexer.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <set>

#include "morton/morton.hpp"

using namespace peclet::core;
using peclet::core::decomp::MortonIndexer;

static void test3d() {
  const IVec<3> origin{4, 8, 16};
  const IVec<3> size{5, 6, 7};
  MortonIndexer<3> idx(origin, size);
  using Code = MortonIndexer<3>::Code;

  std::set<Code> seen;
  IVec<3> bgn{origin[0], origin[1], origin[2]};
  IVec<3> end{origin[0] + size[0], origin[1] + size[1], origin[2] + size[2]};
  forEachInBox<3>(bgn, end, [&](const IVec<3>& g) {
    Code c = idx.codeOf(g);

    // Agrees with a direct origin-relative morton encode.
    auto ref = morton::Morton<3, 21>::encode(
        (std::uint32_t)(g[0] - origin[0]), (std::uint32_t)(g[1] - origin[1]),
        (std::uint32_t)(g[2] - origin[2]));
    PECLET_CORE_CHECK_EQ((long long)c, (long long)ref.code());

    // Round-trips back to the same global multi-index.
    IVec<3> g2 = idx.multiIndex(c);
    PECLET_CORE_CHECK(g2[0] == g[0] && g2[1] == g[1] && g2[2] == g[2]);

    // Bijection: no two cells share a code.
    PECLET_CORE_CHECK(seen.insert(c).second);
  });
  PECLET_CORE_CHECK_EQ((long long)seen.size(), (long long)(size[0] * size[1] * size[2]));

  // neighborCode steps exactly one cell along each axis (interior cell).
  IVec<3> g{origin[0] + 2, origin[1] + 2, origin[2] + 2};
  Code c = idx.codeOf(g);
  for (int axis = 0; axis < 3; ++axis) {
    IVec<3> up = idx.multiIndex(idx.neighborCode(c, axis, +1));
    IVec<3> dn = idx.multiIndex(idx.neighborCode(c, axis, -1));
    for (int k = 0; k < 3; ++k) {
      PECLET_CORE_CHECK_EQ((long long)up[k], (long long)(g[k] + (k == axis ? 1 : 0)));
      PECLET_CORE_CHECK_EQ((long long)dn[k], (long long)(g[k] - (k == axis ? 1 : 0)));
    }
  }
}

static void test2d() {
  const IVec<2> origin{0, 0};
  const IVec<2> size{8, 8};
  MortonIndexer<2> idx(origin, size);
  using Code = MortonIndexer<2>::Code;
  std::set<Code> seen;
  forEachInBox<2>(IVec<2>{0, 0}, IVec<2>{8, 8}, [&](const IVec<2>& g) {
    Code c = idx.codeOf(g);
    IVec<2> g2 = idx.multiIndex(c);
    PECLET_CORE_CHECK(g2[0] == g[0] && g2[1] == g[1]);
    PECLET_CORE_CHECK(seen.insert(c).second);
  });
  PECLET_CORE_CHECK_EQ((long long)seen.size(), 64LL);
}

int main() {
  test3d();
  test2d();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping morton indexer test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

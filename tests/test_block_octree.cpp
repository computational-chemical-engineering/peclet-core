// Serial correctness of the per-block adaptive octree (tpx::amr::BlockOctree):
//  - uniform construction tiles the block; bounds / point-location are correct,
//  - refineIf keeps leaves sorted (Z-order) and conserves total fine volume,
//  - the leaf set matches morton_octree::Octree (the std::map reference oracle)
//    after the same sequence of refinements,
//  - balance2to1 produces a 2:1-graded mesh and is idempotent,
//  - faceNeighbor agrees with point-location across faces.
//
// Guarded by TPX_HAVE_MORTON: without the morton sibling checkout this is a no-op
// pass (the octree header defines nothing).
#include "test_util.hpp"
#include "tpx/common/types.hpp"

#ifdef TPX_HAVE_MORTON
#include <array>
#include <cstdint>

#include "morton/morton.hpp"
#include "morton_octree/octree.hpp"
#include "tpx/amr/block_octree.hpp"

using namespace tpx;
using tpx::amr::BlockOctree;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Oracle = morton_octree::Octree<3, kBits, int>;
using Code = BO::Code;
using Coord = BO::Coord;

// Total number of level-0 fine cells covered by all leaves.
long fineVolume(const BO& t) {
  long v = 0;
  for (Index i = 0; i < t.numLeaves(); ++i) {
    long s = 1L << t.level(i);
    v += s * s * s;
  }
  return v;
}

// A deterministic pseudo-random fine-coordinate point inside the block.
std::array<Coord, 3> randPoint(std::uint64_t& st, Coord span) {
  std::array<Coord, 3> p{};
  for (int d = 0; d < 3; ++d) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    p[d] = static_cast<Coord>((st >> 33) % span);
  }
  return p;
}

// Build the oracle as a uniform brick of root cells at level lmax.
Oracle uniformOracle(IVec<3> brick, unsigned lmax) {
  Oracle o;
  const Coord rootSpan = Coord(Coord(1) << lmax);
  for (Index z = 0; z < brick[2]; ++z)
    for (Index y = 0; y < brick[1]; ++y)
      for (Index x = 0; x < brick[0]; ++x) {
        std::array<Coord, 3> origin{Coord(x) * rootSpan, Coord(y) * rootSpan, Coord(z) * rootSpan};
        o.insert(origin, lmax, 0);
      }
  return o;
}

// Compare a BlockOctree's (sorted) leaf set to the oracle's (map is Z-ordered).
bool sameLeaves(const BO& t, const Oracle& o) {
  if (t.numLeaves() != static_cast<Index>(o.size())) return false;
  Index i = 0;
  for (auto it = o.begin(); it != o.end(); ++it, ++i) {
    if (t.code(i) != it->first.code()) return false;
    if (t.level(i) != it->second.level) return false;
  }
  return true;
}

void test_uniform_and_locate() {
  const IVec<3> brick{2, 3, 2};
  const unsigned lmax = 2;
  BO t(brick, lmax);
  TPX_CHECK_EQ((long long)t.numLeaves(), (long long)(brick[0] * brick[1] * brick[2]));
  TPX_CHECK(t.isBalanced());

  const long total = (long)(brick[0] * brick[1] * brick[2]) * (1L << lmax) * (1L << lmax) * (1L << lmax);
  TPX_CHECK_EQ((long long)fineVolume(t), (long long)total);

  // Sorted ascending by code.
  for (Index i = 1; i < t.numLeaves(); ++i) TPX_CHECK(t.code(i - 1) < t.code(i));

  // Point location lands in a leaf whose bounds contain the point.
  std::uint64_t st = 12345;
  const Coord span = Coord(Coord(1) << lmax) * 2;  // within brick extent (>=2 every axis)
  for (int n = 0; n < 200; ++n) {
    auto p = randPoint(st, span);
    Index leaf = t.find(p);
    TPX_CHECK(leaf >= 0);
    if (leaf < 0) continue;
    auto b = t.bounds(leaf);
    for (int d = 0; d < 3; ++d) TPX_CHECK(p[d] >= b[0][d] && p[d] <= b[1][d]);
  }
}

void test_refine_matches_oracle() {
  const IVec<3> brick{2, 2, 2};
  const unsigned lmax = 3;
  BO t(brick, lmax);
  Oracle o = uniformOracle(brick, lmax);
  TPX_CHECK(sameLeaves(t, o));

  std::uint64_t st = 99;
  const Coord span = Coord(Coord(1) << lmax) * 2;
  for (int step = 0; step < 60; ++step) {
    auto p = randPoint(st, span);
    Code pc = BO::M::encode(p).code();

    Index leaf = t.find(pc);
    if (leaf >= 0 && t.level(leaf) > 0) {
      Code target = t.code(leaf);
      t.refineIf([&](Code c, unsigned) { return c == target; });
    }
    auto it = o.find(BO::M::from_code(pc));
    if (it != o.end() && it->second.level > 0) o.refine(it, [](auto, unsigned) { return 0; });

    TPX_CHECK(sameLeaves(t, o));
    // Volume is conserved by refinement.
    const long totalV =
        (long)(brick[0] * brick[1] * brick[2]) * (1L << lmax) * (1L << lmax) * (1L << lmax);
    TPX_CHECK_EQ((long long)fineVolume(t), (long long)totalV);
  }
}

void refineLeafAt(BO& t, std::array<Coord, 3> p) {
  Index leaf = t.find(p);
  if (leaf >= 0 && t.level(leaf) > 0) {
    Code target = t.code(leaf);
    t.refineIf([&](Code c, unsigned) { return c == target; });
  }
}

void test_balance() {
  const IVec<3> brick{4, 4, 4};
  const unsigned lmax = 2;  // root cells are size 4 in fine units
  BO t(brick, lmax);

  // Create a genuine 2-level jump: refine the corner root cell [0,4)^3 to level 1
  // (size-2 children), then refine the child on its +x face down to level 0. A
  // size-1 cell at x=3 then sits next to the unrefined size-4 neighbour at x=4.
  refineLeafAt(t, {0, 0, 0});  // level 2 -> 1
  refineLeafAt(t, {3, 0, 0});  // level 1 -> 0
  TPX_CHECK(!t.isBalanced());  // unbalanced before (level 0 adjacent to level 2)

  const long volBefore = fineVolume(t);
  Index nref = t.balance2to1();
  TPX_CHECK(nref > 0);
  TPX_CHECK(t.isBalanced());
  TPX_CHECK_EQ((long long)fineVolume(t), (long long)volBefore);  // balance conserves volume

  // Idempotent: a balanced mesh needs no further refinement.
  TPX_CHECK_EQ((long long)t.balance2to1(), 0LL);
  TPX_CHECK(t.isBalanced());
}

void test_face_neighbor() {
  const IVec<3> brick{2, 2, 2};
  const unsigned lmax = 2;
  BO t(brick, lmax);
  // For a uniform grid, the +x neighbour of leaf i (when interior) shares the
  // y/z origin and sits one root-span over.
  for (Index i = 0; i < t.numLeaves(); ++i) {
    auto b = t.bounds(i);
    Index up = t.faceNeighbor(i, 0, +1);
    bool atEdge = b[1][0] + 1 >= (Coord)(brick[0] * (1 << lmax));
    if (atEdge) {
      TPX_CHECK(up < 0);
    } else {
      TPX_CHECK(up >= 0);
      auto bn = t.bounds(up);
      TPX_CHECK_EQ((long long)bn[0][0], (long long)(b[1][0] + 1));
      TPX_CHECK_EQ((long long)bn[0][1], (long long)b[0][1]);
      TPX_CHECK_EQ((long long)bn[0][2], (long long)b[0][2]);
    }
  }
}

}  // namespace

int main() {
  test_uniform_and_locate();
  test_refine_matches_oracle();
  test_balance();
  test_face_neighbor();
  TPX_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping block octree test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

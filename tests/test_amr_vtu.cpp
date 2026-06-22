// Leaf fields + AMR visualization (tpx::amr::LeafField / writeVtu): a refined,
// balanced octree written as a VTK UnstructuredGrid must round-trip its structure
// — one cell per leaf, 2^Dim points per cell, and a per-leaf CellData scalar that
// reads back exactly.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"
#include "tpx/common/types.hpp"

#ifdef TPX_HAVE_MORTON
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/amr/vtu_io.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

std::string attr(const std::string& s, const std::string& key) {
  std::string needle = key + "=\"";
  auto p = s.find(needle);
  if (p == std::string::npos) return "";
  p += needle.size();
  return s.substr(p, s.find('"', p) - p);
}

void run() {
  using BO = BlockOctree<3, 21>;
  using Code = BO::Code;
  using Coord = BO::Coord;
  BO t(IVec<3>{2, 2, 2}, 3);
  auto refineAt = [&](std::array<Coord, 3> p) {
    Index leaf = t.find(p);
    if (leaf >= 0 && t.level(leaf) > 0) {
      Code target = t.code(leaf);
      t.refineIf([&](Code c, unsigned) { return c == target; });
    }
  };
  refineAt({0, 0, 0});
  refineAt({0, 0, 0});
  refineAt({15, 15, 15});
  t.balance2to1();
  TPX_CHECK(t.isBalanced());

  // Per-leaf field = refinement level, plus geometry placing the block at a
  // non-trivial world origin / spacing.
  LeafField<double> lvl(t);
  for (Index i = 0; i < t.numLeaves(); ++i) lvl[i] = static_cast<double>(t.level(i));
  AmrGeometry<3> geo;
  geo.origin = {1.0, -2.0, 0.5};
  geo.h0 = 0.25;

  const std::string path = "amr_test.vtu";
  writeVtu(path, t, geo, "level", lvl);

  // Re-read and validate structure.
  std::ifstream f(path);
  TPX_CHECK(static_cast<bool>(f));
  std::stringstream buf;
  buf << f.rdbuf();
  const std::string s = buf.str();

  long npts = std::stol(attr(s, "NumberOfPoints"));
  long ncells = std::stol(attr(s, "NumberOfCells"));
  TPX_CHECK_EQ((long long)ncells, (long long)t.numLeaves());
  TPX_CHECK_EQ((long long)npts, (long long)(8 * t.numLeaves()));

  // CellData scalar reads back exactly equal to the leaf levels.
  auto cd = s.find("<CellData");
  auto da = s.find("<DataArray", cd);
  auto open = s.find('>', da);
  auto close = s.find("</DataArray>", open);
  std::istringstream data(s.substr(open + 1, close - open - 1));
  long count = 0;
  double v = 0;
  bool ok = true;
  while (data >> v) {
    if (count < t.numLeaves() && v != static_cast<double>(t.level(count))) ok = false;
    ++count;
  }
  TPX_CHECK(ok);
  TPX_CHECK_EQ((long long)count, (long long)t.numLeaves());

  std::remove(path.c_str());
}

}  // namespace

int main() {
  run();
  TPX_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping AMR VTU test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

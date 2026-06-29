// Device collocated FACE-GEOMETRY assembly (tpx::amr::deviceAssembleFaceGeom) must reproduce host
// buildFaceGeom bit-for-bit on OpenMP: same forEachFaceFull enumeration + per-face geometry
// (nbr/axis/dir/α·area/raw area/dist/α/upstream probes) + per-cell invVol/fluid. The D4 anti-drift lock.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/block_octree_view.hpp"
#include "tpx/amr/device_facegeom_assembly.hpp"
#include "tpx/amr/flow.hpp"
#include "tpx/amr/poisson.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;

template <class T>
std::vector<T> down(const View<T>& d) {
  std::vector<T> h(d.extent(0));
  auto m = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(m, d);
  for (std::size_t i = 0; i < h.size(); ++i) h[i] = m(i);
  return h;
}

template <class T>
int mismatch(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return -1;
  int m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) ++m;
  return m;
}

void run() {
  BO t(IVec<3>{2, 2, 2}, 3);
  auto refineAt = [&](std::array<BO::Coord, 3> p) {
    Index leaf = t.find(p);
    if (leaf >= 0 && t.level(leaf) > 0) {
      Code target = t.code(leaf);
      t.refineIf([&](Code c, unsigned) { return c == target; });
    }
  };
  refineAt({0, 0, 0});
  refineAt({15, 15, 15});
  refineAt({8, 8, 8});
  t.balance2to1();

  const double h0 = 0.1;
  const Vec<3> org{-0.8, -0.8, -0.8};
  AmrPoisson<3, kBits> ap;
  ap.init(t, h0);
  ap.setOrigin(org);
  ap.buildOpenness([](const Vec<3>& fc, int axis) {
    return 0.5 + 0.45 * std::sin(0.7 * fc[0] + 1.1 * fc[1] - 0.9 * fc[2] + 0.3 * axis);
  });

  const Index n = t.numLeaves();
  std::vector<char> fluid(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    auto b = t.bounds(i);
    double s = static_cast<double>(Index(1) << t.level(i)), r2 = 0;
    for (int d = 0; d < 3; ++d) {
      double c = org[d] + (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
      r2 += c * c;
    }
    fluid[static_cast<std::size_t>(i)] = (std::sqrt(r2) - 0.45 > 0.0) ? 1 : 0;  // outside sphere = fluid
  }

  // Host reference (walk + upload) vs device assembly.
  FaceGeom hg = buildFaceGeom<3, kBits>(ap, [&](Index i) { return fluid[static_cast<std::size_t>(i)] != 0; });
  BlockOctreeView<3, kBits> dev;
  dev.upload(t);
  FaceGeom dg = deviceAssembleFaceGeom<kBits>(ap, fluid, dev);

  std::printf("  n=%lld nFaces host=%lld dev=%lld\n", static_cast<long long>(n),
              static_cast<long long>(hg.nbr.extent(0)), static_cast<long long>(dg.nbr.extent(0)));

  TPX_CHECK_EQ(mismatch(down(hg.start), down(dg.start)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.nbr), down(dg.nbr)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.axis), down(dg.axis)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.dir), down(dg.dir)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.alphaArea), down(dg.alphaArea)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.rawArea), down(dg.rawArea)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.dist), down(dg.dist)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.alpha), down(dg.alpha)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.upupI), down(dg.upupI)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.upupJ), down(dg.upupJ)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.invVol), down(dg.invVol)), 0);
  TPX_CHECK_EQ(mismatch(down(hg.fluid), down(dg.fluid)), 0);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  TPX_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping device face-geometry assembly test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

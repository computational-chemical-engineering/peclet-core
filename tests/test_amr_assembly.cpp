// Device FV (pressure) operator ASSEMBLY (peclet::core::amr::assembleFv, built on the S1 device CSR-fill
// primitive) must reproduce the host AmrPoisson::assembleFv weight-CSR bit-for-bit on the OpenMP
// backend: same face enumeration (forEachFaceNeighbor order, 2:1 sub-faces), same openness·A_f/d_f
// weights, same invVol/bcDiag. This is the D1+D2 anti-drift lock — the device assembler replaces the
// host walk + upload in the dynamic-geometry path.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/block_octree_view.hpp"
#include "peclet/core/amr/assembly.hpp"
#include "peclet/core/amr/fv_op.hpp"
#include "peclet/core/amr/poisson.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;

// Pull a device View<T> back to a host std::vector for exact comparison.
template <class T>
std::vector<T> down(const View<T>& d) {
  std::vector<T> h(d.extent(0));
  auto m = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(m, d);
  for (std::size_t i = 0; i < h.size(); ++i) h[i] = m(i);
  return h;
}

template <class T>
int countMismatch(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return -1;
  int m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) ++m;
  return m;
}

// Assemble FvOp host-side (AmrPoisson::assembleFv) and device-side (assembleFv) and assert every
// CSR array is bit-identical.
void checkCase(const char* name, const BO& t, double h0, bool periodic, bool wall, bool withOpen) {
  AmrPoisson<3, kBits> ap;
  ap.init(t, h0);
  ap.setPeriodic(periodic);
  ap.setImmersedWall(wall);
  if (withOpen)
    ap.buildOpenness([](const Vec<3>& fc, int axis) {
      // A smooth, deterministic aperture in [0,1] that varies per face centroid + axis, so the
      // weights and bcDiag exercise the openness path (not all-ones).
      return 0.5 + 0.45 * std::sin(0.7 * fc[0] + 1.1 * fc[1] - 0.9 * fc[2] + 0.3 * axis);
    });

  const auto H = ap.assembleFv();  // host reference

  BlockOctreeView<3, kBits> dev;
  dev.upload(t);
  FvOp op = assembleFv(ap, dev);

  // Sizes
  PECLET_CORE_CHECK_EQ(static_cast<Index>(op.n), t.numLeaves());
  std::vector<Index> dstart = down(op.faceStart);
  std::vector<Index> dnbr = down(op.faceNbr);
  std::vector<double> dcoef = down(op.faceW);
  std::vector<double> dinv = down(op.invVol);
  std::vector<double> dbc = down(op.bcDiag);

  std::printf("  [%s] n=%lld nFaces host=%lld dev=%lld\n", name,
              static_cast<long long>(op.n), static_cast<long long>(H.nbr.size()),
              static_cast<long long>(dnbr.size()));

  PECLET_CORE_CHECK_EQ(countMismatch(H.start, dstart), 0);
  PECLET_CORE_CHECK_EQ(countMismatch(H.nbr, dnbr), 0);
  PECLET_CORE_CHECK_EQ(countMismatch(H.coef, dcoef), 0);
  PECLET_CORE_CHECK_EQ(countMismatch(H.invVol, dinv), 0);
  PECLET_CORE_CHECK_EQ(countMismatch(H.bcDiag, dbc), 0);

  // And the assembled operator APPLIES identically: applyFv == host shared-FV apply.
  const Index n = t.numLeaves();
  std::vector<double> x(static_cast<std::size_t>(n));
  std::uint64_t s = 0x9e3779b97f4a7c15ULL;
  for (auto& v : x) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    v = static_cast<double>((s >> 11) & 0xFFFFF) / static_cast<double>(0x100000) - 0.5;
  }
  std::vector<double> hout;
  ap.applyFvShared(x, hout);
  View<const double> dx = toDevice(x, "x");
  View<double> dLu("Lu", static_cast<std::size_t>(n));
  applyFv(op, dx, dLu);
  std::vector<double> dout = down(dLu);
  PECLET_CORE_CHECK_EQ(countMismatch(hout, dout), 0);
}

void run() {
  // A refined, balanced octree (mixed levels exercise the 2:1 sub-face enumeration).
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

  const double h0 = 0.5;
  checkCase("periodic", t, h0, /*periodic=*/true, /*wall=*/false, /*open=*/false);
  checkCase("periodic+open", t, h0, true, false, true);
  checkCase("immersed-wall+open", t, h0, true, /*wall=*/true, true);
  checkCase("dirichlet+open", t, h0, /*periodic=*/false, false, true);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping device AMR assembly test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

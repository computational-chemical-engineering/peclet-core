// Cut-cell openness coarsened across the device (Kokkos) MG levels
// (tpx::amr::DeviceMultigrid::build(finest, h0, openFn)). The openness is set on the
// finest level and area-averaged to every coarser level (via AmrMultigrid::setOpenness),
// so each level is a consistent cut-cell operator. Validates:
//   (1) the device operator on EVERY level == host AmrMultigrid::op(L).applyLaplacian
//       (with the same coarsened openness) bit-for-bit — i.e. the openness coarsening
//       matches the host on all levels, not just the finest;
//   (2) the openness V-cycle converges on the graded mesh (manufactured RHS).
// Runs on whatever backend Kokkos was built for (CUDA / HIP / OpenMP).
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/device_multigrid.hpp"
#include "tpx/amr/poisson.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using M = BO::M;

// Smooth openness in [~0.25, ~0.95] (never fully solid ⇒ no zero-diagonal cells).
double openFn(const Vec<3>& p, int /*axis*/) {
  const double k = 2.0 * M_PI;
  return 0.6 + 0.35 * std::sin(k * p[0]) * std::cos(k * p[1]) * std::cos(k * p[2]);
}

void setDev(View<double> v, const std::vector<double>& h) {
  auto m = Kokkos::create_mirror_view(v);
  for (std::size_t i = 0; i < h.size(); ++i) m((Index)i) = h[i];
  Kokkos::deep_copy(v, m);
}
std::vector<double> getDev(View<double> v, Index n) {
  std::vector<double> h((std::size_t)n);
  auto m = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(m, v);
  for (Index i = 0; i < n; ++i) h[(std::size_t)i] = m(i);
  return h;
}

void run() {
  // Graded multilevel octree: 2×2×2 brick (level 3), lower octant refined to level 0.
  BO t(IVec<3>{2, 2, 2}, 3);
  for (int kk = 0; kk < 2; ++kk) t.refineIf([](Code, unsigned l) { return l > 0; });
  t.refineIf([&](Code c, unsigned) {
    auto o = M::from_code(c).decode();
    return o[0] < 8 && o[1] < 8 && o[2] < 8;
  });
  t.balance2to1();
  const double h0 = 1.0 / 16.0;

  // Host AmrMultigrid with openness coarsened to every level (the reference).
  AmrMultigrid<3, kBits> hmg;
  hmg.build(t, h0);
  hmg.setOpenness(openFn);

  // Device MG with the same openFn (coarsened across levels internally).
  DeviceMultigrid<3, kBits> mg;
  mg.build(t, h0, openFn);
  TPX_CHECK(mg.numLevels() == hmg.numLevels());
  TPX_CHECK(mg.numLevels() >= 3);

  // ===== (1) device operator == host op(L).applyLaplacian on EVERY level (bit-exact) =====
  std::uint64_t s = 99991;
  int totalMism = 0;
  for (std::size_t L = 0; L < hmg.numLevels(); ++L) {
    const Index n = hmg.op(L).octree().numLeaves();
    TPX_CHECK(mg.numLeaves(L) == n);
    std::vector<double> x((std::size_t)n);
    for (auto& v : x) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      v = (double)((s >> 11) & 0xFFFFF) / (double)0x100000 - 0.5;
    }
    View<double> dx("x", (std::size_t)n), dLu("Lu", (std::size_t)n);
    setDev(dx, x);
    deviceApplyFv(mg.op(L), View<const double>(dx), dLu);
    auto dLh = getDev(dLu, n);
    std::vector<double> hLu;
    hmg.op(L).applyLaplacian(x, hLu);
    for (Index i = 0; i < n; ++i)
      if (dLh[(std::size_t)i] != hLu[(std::size_t)i]) ++totalMism;
  }
  TPX_CHECK_EQ(totalMism, 0);

  // ===== (2) openness V-cycle converges on the graded mesh (manufactured RHS) =====
  const Index n0 = mg.numLeaves(0);
  std::vector<double> uex((std::size_t)n0);
  for (Index i = 0; i < n0; ++i) {
    auto o = M::from_code(hmg.op(0).octree().code(i)).decode();
    double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0, cz = ((double)o[2] + 0.5) * h0;
    const double k = 2.0 * M_PI;
    uex[(std::size_t)i] = std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz);
  }
  // b = L·u_exact (exactly mean-zero by conservation; α symmetric across each face)
  View<double> duex("uex", (std::size_t)n0), db("b", (std::size_t)n0);
  setDev(duex, uex);
  deviceApplyFv(mg.op(0), View<const double>(duex), db);
  std::vector<double> b = getDev(db, n0);
  setDev(mg.b(0), b);
  Kokkos::deep_copy(mg.x(0), 0.0);

  auto resNorm = [&]() {
    std::vector<double> x = getDev(mg.x(0), n0);
    std::vector<double> lu;
    hmg.op(0).applyLaplacian(x, lu);
    double s2 = 0.0;
    for (Index i = 0; i < n0; ++i) {
      double r = b[(std::size_t)i] - lu[(std::size_t)i];
      s2 += r * r;
    }
    return std::sqrt(s2);
  };
  const double r0 = resNorm();
  for (int c = 0; c < 40; ++c) mg.vcycle(2, 2, 60, 0.8);
  const double r1 = resNorm();
  TPX_CHECK(r1 < r0 * 1e-3);
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
  std::printf("TPX_HAVE_MORTON not set — skipping device openness test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

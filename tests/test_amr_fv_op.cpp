// Device (Kokkos) Poisson operator/smoother (peclet::core::amr::laplacian /
// jacobiSweep) must match the host BlockOctree operator bit-for-bit, on
// whatever backend Kokkos was built for (CUDA / HIP / OpenMP). Same face-neighbour
// walk + arithmetic, just run as parallel_for over the leaf Views.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/block_octree_view.hpp"
#include "peclet/core/amr/fv_op.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;

// Host Laplacian using the same face-neighbour semantics as the device op.
double hostLap(const BO& t, const std::vector<double>& x, Index i, double inv) {
  double s = 0.0;
  for (int axis = 0; axis < 3; ++axis)
    for (int dir = -1; dir <= 1; dir += 2) {
      Index j = t.faceNeighbor(i, axis, dir);
      if (j >= 0)
        s += x[static_cast<std::size_t>(j)] - x[static_cast<std::size_t>(i)];
    }
  return inv * s;
}

void run() {
  // A refined, balanced octree (mix of levels exercises the neighbour walk).
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
  t.balance2to1();
  const Index n = t.numLeaves();
  const double inv = 4.0;  // arbitrary 1/h0²

  std::vector<double> x(static_cast<std::size_t>(n));
  std::uint64_t s = 12345;
  for (auto& v : x) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    v = static_cast<double>((s >> 11) & 0xFFFFF) / static_cast<double>(0x100000) - 0.5;
  }

  BlockOctreeView<3, kBits> dev;
  dev.upload(t);
  View<double> dx = toDevice(x, "x");
  View<double> dy("y", static_cast<std::size_t>(n));

  // ---- matvec: device == host ----
  laplacian<3, kBits>(dev, dx, dy, inv);
  auto hy = Kokkos::create_mirror_view(dy);
  Kokkos::deep_copy(hy, dy);
  int mism = 0;
  for (Index i = 0; i < n; ++i)
    if (hy(i) != hostLap(t, x, i, inv))
      ++mism;
  PECLET_CORE_CHECK_EQ(mism, 0);

  // ---- a few Jacobi sweeps: device == host ----
  std::vector<double> b(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i)
    b[static_cast<std::size_t>(i)] = std::sin(0.1 * i);
  const double omega = 0.7, diag = 2.0 * 3 * inv;

  // device
  View<double> du("u", static_cast<std::size_t>(n));  // starts 0
  View<const double> db = toDevice(b, "b");
  View<double> dax("ax", static_cast<std::size_t>(n));
  for (int it = 0; it < 5; ++it)
    jacobiSweep<3, kBits>(dev, du, db, dax, inv, omega);
  auto hu = Kokkos::create_mirror_view(du);
  Kokkos::deep_copy(hu, du);

  // host reference (same sweep): L = ∇², update u += ω(Lu − b)/diag
  std::vector<double> u(static_cast<std::size_t>(n), 0.0);
  for (int it = 0; it < 5; ++it) {
    std::vector<double> lx(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i)
      lx[static_cast<std::size_t>(i)] = hostLap(t, u, i, inv);
    for (Index i = 0; i < n; ++i)
      u[static_cast<std::size_t>(i)] +=
          omega * (lx[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]) / diag;
  }
  int jmis = 0;
  for (Index i = 0; i < n; ++i)
    if (hu(i) != u[static_cast<std::size_t>(i)])
      ++jmis;
  PECLET_CORE_CHECK_EQ(jmis, 0);
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping device Poisson test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

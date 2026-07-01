// Device-resident distributed AMR multigrid (peclet::core::amr::DistributedMultigridDevice, C2): the V-cycle
// runs entirely in Kokkos kernels over the device field, mirroring only the compact gather buffer
// across MPI. It must (1) reproduce the HOST DistributedMultigrid on the same decomposition bit-for-bit
// (the device port is exact), (2) match the single-block MPI_COMM_SELF reference bit-for-bit (consistent
// across rank counts), and (3) actually solve. np = 1,2,4,8.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include "peclet/core/amr/distributed_device.hpp"
#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/distributed_poisson.hpp"
#include "peclet/core/common/mpi.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

double bAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = (static_cast<double>(o[0]) + 0.5) * h0;
  double cy = (static_cast<double>(o[1]) + 0.5) * h0;
  double cz = (static_cast<double>(o[2]) + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cy);
}

std::vector<double> down(const View<double>& d, Index n) {
  std::vector<double> h(static_cast<std::size_t>(n));
  auto m = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(m, d);
  for (Index i = 0; i < n; ++i) h[static_cast<std::size_t>(i)] = m(i);
  return h;
}

void run() {
  const long N = 16;  // 16^3 periodic [0,1)^3 ⇒ 16→8→4→2 (4 levels)
  const double h0 = 1.0 / N;
  AmrGeometry<3> geo;
  geo.h0 = h0;
  const std::array<bool, 3> per{true, true, true};
  const int cycles = 8;

  // --- device, distributed over MPI_COMM_WORLD ---
  DistributedMultigridDevice<3, kBits> dmg;
  dmg.build(IVec<3>{N, N, N}, geo, per, MPI_COMM_WORLD);
  const Index nw = dmg.numLeaves();
  std::vector<double> bwh(static_cast<std::size_t>(nw));
  for (Index i = 0; i < nw; ++i)
    bwh[static_cast<std::size_t>(i)] = bAt(dmg.octree().globalCode(i), h0);
  View<double> bw = toDevice(bwh, "bw");
  View<double> xw = dmg.x(0);
  Kokkos::deep_copy(xw, 0.0);
  const double r0 = dmg.op().residualNorm(View<const double>(xw), View<const double>(bw));
  for (int c = 0; c < cycles; ++c) dmg.vcycle(xw, View<const double>(bw));
  const double r1 = dmg.op().residualNorm(View<const double>(xw), View<const double>(bw));
  std::vector<double> xwh = down(xw, nw);

  // --- host reference on the SAME decomposition (device must reproduce it bit-for-bit) ---
  DistributedMultigrid<3, kBits> hmg;
  hmg.build(IVec<3>{N, N, N}, geo, per, MPI_COMM_WORLD);
  std::vector<double> bh(static_cast<std::size_t>(nw)), xh(static_cast<std::size_t>(nw), 0.0);
  for (Index i = 0; i < nw; ++i) bh[static_cast<std::size_t>(i)] = bAt(hmg.octree().globalCode(i), h0);
  for (int c = 0; c < cycles; ++c) hmg.vcycle(xh, bh);

  // --- device, single block on MPI_COMM_SELF (consistency across rank counts) ---
  DistributedMultigridDevice<3, kBits> dms;
  dms.build(IVec<3>{N, N, N}, geo, per, MPI_COMM_SELF);
  const Index ns = dms.numLeaves();
  std::vector<double> bsh(static_cast<std::size_t>(ns));
  for (Index i = 0; i < ns; ++i) bsh[static_cast<std::size_t>(i)] = bAt(dms.octree().globalCode(i), h0);
  View<double> bs = toDevice(bsh, "bs");
  View<double> xs = dms.x(0);
  Kokkos::deep_copy(xs, 0.0);
  for (int c = 0; c < cycles; ++c) dms.vcycle(xs, View<const double>(bs));
  std::vector<double> xsh = down(xs, ns);

  int mismH = 0, mismS = 0;
  double maxH = 0.0, maxS = 0.0;
  for (Index i = 0; i < nw; ++i) {
    // device-vs-host on the same WORLD decomposition (index-aligned).
    maxH = std::max(maxH, std::fabs(xwh[static_cast<std::size_t>(i)] - xh[static_cast<std::size_t>(i)]));
    // device-WORLD vs device-SELF: SELF block covers the whole domain ⇒ local code == global code.
    Index si = dms.octree().local().find(dmg.octree().globalCode(i));
    if (si < 0) {
      ++mismS;
      continue;
    }
    maxS = std::max(maxS, std::fabs(xwh[static_cast<std::size_t>(i)] - xsh[static_cast<std::size_t>(si)]));
  }
  (void)mismH;
  PECLET_CORE_CHECK_EQ(mismS, 0);
  PECLET_CORE_CHECK(maxH == 0.0);  // device == host on the same decomposition, bit-for-bit
  PECLET_CORE_CHECK(maxS == 0.0);  // device WORLD == device SELF, bit-for-bit across rank counts
  PECLET_CORE_CHECK(dmg.numLevels() == 4);
  PECLET_CORE_CHECK(r1 < r0 * 1e-3);  // MG actually solves
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int fails = peclet::core::test::g_failures, total = 0;
  MPI_Reduce(&fails, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (rank == 0) {
    if (total == 0) {
      std::printf("OK\n");
      return 0;
    }
    std::fprintf(stderr, "%d failure(s)\n", total);
    return 1;
  }
  return 0;
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping distributed device test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

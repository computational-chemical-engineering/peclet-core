// Device LeafHaloExchange (peclet::core::amr::LeafHaloExchange): the device-resident value
// refresh over a finalized LeafHalo — device pack/scatter, compact host-staged MPI buffers
// (docs/amr_distributed_flow.md, rung 1). Validates on a graded cross-block octree:
//   (1) exchange() == exchangeHost() bit-for-bit (ghost values are unmodified double copies);
//   (2) exchange3() (batched 3-component, one message round) == three exchange() calls
//       bit-for-bit;
//   (3) the local part of the extended field is untouched by the exchange.
// np = 1,2,4,8.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/common/view.hpp"

#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/leaf_halo.hpp"
#include "peclet/core/common/mpi.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

double fAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0,
         cz = ((double)o[2] + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cx);
}

void makeGraded(DO& d) {
  for (int pass = 0; pass < 2; ++pass) {
    d.local().refineIf([&](Code c, unsigned lvl) -> bool {
      if (lvl == 0)
        return false;
      auto o = M::from_code(c).decode();
      for (int dd = 0; dd < 3; ++dd)
        if ((long)o[dd] + d.blockFineOrigin()[dd] >= 8)
          return false;
      return true;
    });
  }
  d.balance();
}

std::vector<double> down(const View<double>& d) {
  std::vector<double> h(d.extent(0));
  auto m = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(m, d);
  for (std::size_t i = 0; i < h.size(); ++i)
    h[i] = m(i);
  return h;
}

void run() {
  const long Nr = 4;
  const unsigned lmax = 2;
  const double h0 = 1.0 / (Nr * (1 << lmax));
  AmrGeometry<3> geo;
  geo.h0 = h0;
  const std::array<bool, 3> per{true, true, true};

  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeGraded(world);
  const Index n = world.local().numLeaves();

  // Same builder-style probe set as the host test (±1 and ±2 reach), fixpoint, finalize.
  LeafHalo<3, kBits> halo;
  halo.init(world);
  std::vector<std::array<long, 3>> probes;
  for (Index i = 0; i < n; ++i) {
    auto b = world.local().bounds(i);
    const long s = 1L << world.local().level(i);
    std::array<long, 3> gLo{};
    for (int a = 0; a < 3; ++a)
      gLo[a] = (long)b[0][a] + world.blockFineOrigin()[a];
    for (int a = 0; a < 3; ++a)
      for (long off : {-1L, -2L, s, s + 1}) {
        std::array<long, 3> p = gLo;
        p[a] += off;
        probes.push_back(p);
      }
  }
  for (;;) {
    for (auto& p : probes)
      (void)halo.resolveGlobal(p);
    if (halo.resolveMisses() == 0)
      break;
  }
  halo.finalize();
  const Index ext = halo.extendedSize();

  // Three host fields (locals filled, ghost tail poisoned) + the host-exchange reference.
  std::array<std::vector<double>, 3> hx;
  for (int c = 0; c < 3; ++c) {
    hx[c].assign(static_cast<std::size_t>(ext), -7e77);
    for (Index i = 0; i < n; ++i) {
      const double f = fAt(world.globalCode(i), h0);
      hx[c][static_cast<std::size_t>(i)] = (c == 0) ? f : (c == 1) ? 2.0 * f + 1.0 : -f;
    }
  }
  std::array<std::vector<double>, 3> ref = hx;
  for (int c = 0; c < 3; ++c)
    halo.exchangeHost(ref[c]);

  LeafHaloExchange ex;
  ex.init(halo);
  PECLET_CORE_CHECK_EQ(ex.numGhosts(), halo.numGhosts());

  // (1) single-field device exchange == host exchange bit-for-bit; locals untouched.
  {
    View<double> x = toDevice(hx[0], "x");
    ex.exchange(x);
    std::vector<double> got = down(x);
    for (Index i = 0; i < ext; ++i)
      PECLET_CORE_CHECK(got[static_cast<std::size_t>(i)] == ref[0][static_cast<std::size_t>(i)]);
  }

  // (2) batched 3-component exchange == three single exchanges bit-for-bit.
  {
    View<double> x0 = toDevice(hx[0], "x0"), x1 = toDevice(hx[1], "x1"),
                 x2 = toDevice(hx[2], "x2");
    ex.exchange3(x0, x1, x2);
    std::vector<double> g0 = down(x0), g1 = down(x1), g2 = down(x2);
    for (Index i = 0; i < ext; ++i) {
      PECLET_CORE_CHECK(g0[static_cast<std::size_t>(i)] == ref[0][static_cast<std::size_t>(i)]);
      PECLET_CORE_CHECK(g1[static_cast<std::size_t>(i)] == ref[1][static_cast<std::size_t>(i)]);
      PECLET_CORE_CHECK(g2[static_cast<std::size_t>(i)] == ref[2][static_cast<std::size_t>(i)]);
    }
  }
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping LeafHaloExchange test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

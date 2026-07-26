// LeafHalo (peclet::core::amr::LeafHalo): the ghost registry + value halo the distributed
// AmrFlow builders thread their neighbour probes through (docs/amr_distributed_flow.md, rung 1).
// On a genuinely graded, cross-block 2:1-balanced octree it validates:
//   (1) the builder fixpoint (resolve → resolveMisses rounds) terminates and resolves every
//       probe — ±1 face probes AND ±2 multi-hop probes, the closure/SOU reach;
//   (2) canonical dedup: probes into the same remote covering leaf share ONE ghost slot, and
//       the recorded (anchor, level) metadata matches the owner (coverLevels oracle);
//   (3) exchangeHost fills every ghost slot with the owner's value bit-for-bit — checked
//       against both the analytic per-leaf field and the coverValues owner-gather oracle;
//   (4) the frozen topology is reusable: a second exchange after the field changed refreshes;
//   (5) np=1: every wrapped probe resolves locally ⇒ ZERO ghosts (the bit-exact single-rank
//       path never touches a ghost slot).
// np = 1,2,4,8.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <array>
#include <cmath>
#include <vector>

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
using Coord = DO::Coord;

// Field keyed on the global Morton code of the leaf's lo corner, so every rank can compute the
// expected value of any (local or ghost) leaf without communication.
double fAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0,
         cz = ((double)o[2] + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cx);
}

// Refine the lower octant two levels, then 2:1-balance — graded, with cross-block interfaces.
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

void run() {
  const long Nr = 4;  // 4^3 root cells, lmax 2 ⇒ 16^3 fine, periodic [0,1)^3
  const unsigned lmax = 2;
  const double h0 = 1.0 / (Nr * (1 << lmax));
  AmrGeometry<3> geo;
  geo.h0 = h0;
  const std::array<bool, 3> per{true, true, true};
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeGraded(world);
  const Index n = world.local().numLeaves();

  // ---- builder-style probe enumeration: ±1 face probes + ±2 multi-hop probes ----
  LeafHalo<3, kBits> halo;
  halo.init(world);
  std::vector<std::array<long, 3>> probes;
  for (Index i = 0; i < n; ++i) {
    auto b = world.local().bounds(i);
    const long s = 1L << world.local().level(i);
    std::array<long, 3> gLo{};
    for (int a = 0; a < 3; ++a)
      gLo[a] = (long)b[0][a] + world.blockFineOrigin()[a];
    for (int a = 0; a < 3; ++a) {
      for (long off : {-1L, -2L, s, s + 1}) {  // ±1 face reach and the ±2 closure/SOU reach
        std::array<long, 3> p = gLo;
        p[a] += off;
        probes.push_back(p);
      }
    }
  }

  // Fixpoint: attempt the pass, resolve the misses collectively, repeat until globally clean.
  std::vector<Index> slot(probes.size(), LeafHalo<3, kBits>::kPending);
  int rounds = 0;
  for (;;) {
    for (std::size_t k = 0; k < probes.size(); ++k)
      slot[k] = halo.resolveGlobal(probes[k]);
    ++rounds;
    if (halo.resolveMisses() == 0)
      break;
    PECLET_CORE_CHECK(rounds < 6);  // bounded reach ⇒ bounded rounds
  }
  for (std::size_t k = 0; k < probes.size(); ++k)
    PECLET_CORE_CHECK(slot[k] >= 0);  // periodic domain: every probe resolves
  halo.finalize();

  if (size == 1)
    PECLET_CORE_CHECK_EQ(halo.numGhosts(), 0);  // np=1: zero ghosts by construction

  // Ghost metadata vs the owner (coverLevels oracle): anchor and level agree.
  {
    std::vector<std::array<Coord, 3>> anchors(
        static_cast<std::size_t>(halo.numGhosts()));
    for (Index g = 0; g < halo.numGhosts(); ++g)
      anchors[static_cast<std::size_t>(g)] = halo.ghostCoord(g);
    std::vector<int> lv = world.coverLevels(anchors);
    for (Index g = 0; g < halo.numGhosts(); ++g) {
      PECLET_CORE_CHECK_EQ(lv[static_cast<std::size_t>(g)], halo.level(halo.numLocal() + g));
      for (int a = 0; a < 3; ++a)  // anchor is the covering leaf's lo: aligned to its level
        PECLET_CORE_CHECK_EQ(
            (long)(halo.ghostCoord(g)[a] >> lv[static_cast<std::size_t>(g)])
                << lv[static_cast<std::size_t>(g)],
            (long)halo.ghostCoord(g)[a]);
    }
  }

  // ---- host exchange: every ghost slot holds the owner's value, bit-for-bit ----
  std::vector<double> x(static_cast<std::size_t>(halo.extendedSize()), 0.0);
  for (Index i = 0; i < n; ++i)
    x[static_cast<std::size_t>(i)] = fAt(world.globalCode(i), h0);
  halo.exchangeHost(x);
  for (Index g = 0; g < halo.numGhosts(); ++g) {
    const Code gc = M::encode(halo.ghostCoord(g)).code();
    PECLET_CORE_CHECK(x[static_cast<std::size_t>(halo.numLocal() + g)] == fAt(gc, h0));
  }

  // Cross-check against the per-probe coverValues owner gather (the pre-LeafHalo oracle).
  {
    std::vector<std::array<Coord, 3>> wrapped(probes.size());
    for (std::size_t k = 0; k < probes.size(); ++k) {
      std::array<long, 3> p = probes[k];
      PECLET_CORE_CHECK(halo.wrap(p));
      for (int a = 0; a < 3; ++a)
        wrapped[k][a] = static_cast<Coord>(p[a]);
    }
    std::vector<double> local(x.begin(), x.begin() + static_cast<std::size_t>(n));
    std::vector<double> oracle = world.coverValues(wrapped, local);
    for (std::size_t k = 0; k < probes.size(); ++k)
      PECLET_CORE_CHECK(x[static_cast<std::size_t>(slot[k])] == oracle[k]);
  }

  // ---- topology reuse: change the field, exchange again, ghosts refresh ----
  for (Index i = 0; i < n; ++i)
    x[static_cast<std::size_t>(i)] = 3.0 * x[static_cast<std::size_t>(i)] + 1.0;
  halo.exchangeHost(x);
  for (Index g = 0; g < halo.numGhosts(); ++g) {
    const Code gc = M::encode(halo.ghostCoord(g)).code();
    PECLET_CORE_CHECK(x[static_cast<std::size_t>(halo.numLocal() + g)] ==
                      3.0 * fAt(gc, h0) + 1.0);
  }

  // Dedup effectiveness: at np>1 the probe count into ghosts far exceeds the slot count.
  long nProbeGhost = 0;
  for (std::size_t k = 0; k < probes.size(); ++k)
    if (slot[k] >= halo.numLocal())
      ++nProbeGhost;
  if (size > 1 && nProbeGhost > 0)
    PECLET_CORE_CHECK(halo.numGhosts() < nProbeGhost);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  run();
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping LeafHalo test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

// The distributed MIXED-LEVEL cut band (docs/amr_march_perf_and_distributed_plan.md, Phase D):
// AmrFlow::setGhostSampled through the LeafHalo seam. The mesh is deliberately SEAMED — a
// two-level latitude map, the geometry family that actually produces least-squares sample rows
// (a hemisphere jump perpendicular to the wall produces none, M2's lesson) — so the overlay's
// sample slots, not just its identity slots, are exercised.
//
// The acceptance ladder is DD4's: WORLD (initMpi) vs SELF (single-rank on the same octree),
// compared leaf by leaf through the global Morton code.
//   np = 1 : BITWISE (the classic distributed contract — every wrapped probe lands back in the
//            block, so zero ghosts exist and the two paths must execute identical arithmetic).
//   np > 1 : the established decomposition-independence class. The clouds themselves are
//            order-canonical by construction (candidates are sorted by bin then by the GLOBAL
//            Morton code of the leaf's lo corner, so a cloud accumulates in the same order on
//            every rank that can see it); what remains is the solvers' own global reductions.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/flow.hpp"
#include "peclet/core/common/types.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

// A sphere well inside the periodic box; the cut band is what the sampled overlay lives on.
double sphereSdf(const Vec<3>& p) {
  const double dx = p[0] - 0.5, dy = p[1] - 0.5, dz = p[2] - 0.5;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - 0.22;
}

/// Two-level latitude map: refine the band to level 0 below z = c − R/2 and to level 1 above, so
/// the 2:1 jump plane crosses the wall OBLIQUELY and the overlay must build virtual samples.
void makeSeamedMesh(DO& d, double h0) {
  const double c = 0.5, R = 0.22;
  for (int pass = 0; pass < 3; ++pass) {
    d.local().refineIf([&](Code cd, unsigned lvl) -> bool {
      if (lvl == 0)
        return false;
      auto o = M::from_code(cd).decode();
      const double s = static_cast<double>(1u << lvl);
      Vec<3> ctr{};
      for (int a = 0; a < 3; ++a)
        ctr[a] = (static_cast<double>((long)o[a] + d.blockFineOrigin()[a]) + 0.5 * s) * h0;
      const unsigned tgt =
          std::getenv("SEAM_UNIFORM") ? 0u : (ctr[2] < c - 0.5 * R ? 0u : 1u);
      if (lvl <= tgt)
        return false;
      const double w = s * h0;
      return std::fabs(sphereSdf(ctr)) < w + 3.0 * h0;  // generous band (ghost margin)
    });
  }
  d.balance();
}

struct Fields {
  std::array<std::vector<double>, 3> u;
  std::vector<double> p;
};

Fields runSteps(AmrFlow<kBits>& f, int steps) {
  for (int s = 0; s < steps; ++s)
    f.step(200, 60);
  Fields r;
  for (int c = 0; c < 3; ++c)
    r.u[(std::size_t)c] = f.velocity(c);
  r.p = f.pressure();
  return r;
}

void configure(AmrFlow<kBits>& f) {
  f.setDensity(1.0);
  f.setViscosity(1.0);
  f.setBodyForce(1.0, 0.0, 0.0);
  f.setDt(1e6);
  f.setGhostProjection(true, 2, 2);
  f.setGhostSampled(true);
  f.setSolid(sphereSdf);
}

void run() {
  const long Nr = 4;  // 4^3 roots, lmax 2 ⇒ 16^3 fine, periodic [0,1)^3
  // SEAM_LMAX raises the resolution (lmax 4 ⇒ 64^3 fine): the gated ctest runs the small mesh,
  // while the D2 setup-scaling measurement wants enough leaves for a meaningful us/leaf.
  const unsigned lmax = std::getenv("SEAM_LMAX") ? (unsigned)std::atoi(std::getenv("SEAM_LMAX")) : 2;
  const double h0 = 1.0 / (Nr * (1 << lmax));
  AmrGeometry<3> geo;
  geo.h0 = h0;
  const std::array<bool, 3> per{true, true, true};
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeSeamedMesh(world, h0);
  const Index n = world.local().numLeaves();
  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_SELF);
  makeSeamedMesh(self, h0);

  AmrFlow<kBits> fw;
  fw.initMpi(world);
  const double tSolid0 = MPI_Wtime();
  configure(fw);
  const double tSolid = MPI_Wtime() - tSolid0;
  {  // D2: what a rank pays for the seamed distributed setup.
    double tmax = 0.0;
    long nl = (long)n, ntot = 0;
    MPI_Allreduce(&tSolid, &tmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&nl, &ntot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("[seam-mpi] setup np=%d: %ld leaves total, slowest rank %.3f s = %.1f us/leaf "
                  "(per-rank mean %ld leaves)\n",
                  size, ntot, tmax, 1e6 * tmax / ((double)ntot / size), ntot / size);
  }
  if (size > 1)
    PECLET_CORE_CHECK(fw.numGhostCells() > 0);
  else
    PECLET_CORE_CHECK_EQ(fw.numGhostCells(), 0);  // np=1: every wrapped probe lands back in-block
  // Two diagnostic knobs, both defaulting to the gated configuration: SEAM_STEPS shortens or
  // lengthens the march (the WORLD/SELF difference peaks mid-transient and DECAYS, so a single
  // step count hides the shape), and SEAM_UNIFORM rebuilds the band uniform, which turns every
  // chain slot into an identity slot and so isolates the sample machinery from the rest of the
  // distributed path. Both earned their keep finding the D2 frame bug.
  const int kSteps = std::getenv("SEAM_STEPS") ? std::atoi(std::getenv("SEAM_STEPS")) : 3;
  const Fields w = runSteps(fw, kSteps);

  AmrFlow<kBits> fs;
  fs.init(self.local(), h0, Vec<3>{0.0, 0.0, 0.0});
  configure(fs);
  const Fields s = runSteps(fs, kSteps);

  double dmax = 0.0, scale = 0.0;
  for (Index i = 0; i < n; ++i) {
    const Index si = self.local().find(world.globalCode(i));
    PECLET_CORE_CHECK(si >= 0);
    for (int c = 0; c < 3; ++c) {
      scale = std::max(scale, std::fabs(s.u[(std::size_t)c][(std::size_t)si]));
      dmax = std::max(dmax, std::fabs(w.u[(std::size_t)c][(std::size_t)i] -
                                      s.u[(std::size_t)c][(std::size_t)si]));
    }
    dmax = std::max(dmax, std::fabs(w.p[(std::size_t)i] - s.p[(std::size_t)si]));
  }
  double gdmax = 0.0, gscale = 0.0;
  MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&scale, &gscale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("[seam-mpi] np=%d sampled overlay WORLD vs SELF: |d|max %.3e (scale %.3e, rel "
                "%.3e)\n",
                size, gdmax, gscale, gdmax / (gscale + 1e-300));
  if (size == 1)
    PECLET_CORE_CHECK(gdmax == 0.0);  // BITWISE, the np=1 contract
  else
    PECLET_CORE_CHECK(gdmax <= 5e-6 * gscale);  // decomposition independence (DD4)
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping distributed seam test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

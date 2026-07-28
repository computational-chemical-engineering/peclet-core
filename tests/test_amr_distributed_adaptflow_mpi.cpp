// Distributed mid-run adaptivity + load rebalance under the full AmrFlow NS step
// (docs/amr_distributed_flow.md, rung 6): on the graded sphere mesh, run NS (auto-ghost),
// then mid-run
//   (a) beginAdapt → distributedAdapt (Löhner on |u|, ownership-preserving) + the driver's
//       geometry-band re-refinement + cross-block balance → finishAdapt (block-local
//       conservative transfer + full distributed setSolid rebuild) → continue;
//   (b) rebalanceMpi: weighted-ORB re-decomposition migrating the state with the leaves,
//       full rebuild on the new block → continue.
// The COMM_SELF reference runs the IDENTICAL adapt policy (distributedAdapt's flags are
// bit-identical across rank counts) without the rebalance. Gates, at np = 1,2,4,8:
//   np=1: u/p BIT-EXACT vs single-rank through the whole sequence (adapt AND rebalance);
//   np>1: Krylov tolerance — which also proves the rebalance migrated the state exactly
//   (any migration error would break the continued trajectory).
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/common/view.hpp"

#include "peclet/core/amr/distributed_adapt.hpp"
#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/flow.hpp"
#include "peclet/core/common/mpi.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

double sphereSdf(const Vec<3>& p) {
  const double dx = p[0] - 0.5, dy = p[1] - 0.5, dz = p[2] - 0.5;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - 0.2;
}

// The finest geometry band around the surface (the ghost-overlay margin contract).
void refineBand(DO& d, double h0) {
  for (int pass = 0; pass < 2; ++pass) {
    d.local().refineIf([&](Code c, unsigned lvl) -> bool {
      if (lvl == 0)
        return false;
      auto o = M::from_code(c).decode();
      const double s = static_cast<double>(1u << lvl);
      Vec<3> ctr{};
      for (int a = 0; a < 3; ++a)
        ctr[a] = (static_cast<double>((long)o[a] + d.blockFineOrigin()[a]) + 0.5 * s) * h0;
      const double w = s * h0;
      return std::fabs(sphereSdf(ctr)) < w + 3.0 * h0;
    });
  }
  d.balance();
}

// Deterministic mid-run adapt policy, identical on every rank count: Löhner on an ANALYTIC
// field of the cell centre (so the flags cannot flip on the O(1e-7) Krylov noise between the
// WORLD and SELF trajectories — a SOLUTION-driven policy is inherently threshold-sensitive to
// that noise, which is a property of the policy, not of the machinery under test; the
// solution-driven distributedAdapt path itself is np-invariance-locked on exact fields in
// test_amr_distributed_adapt_mpi). Then the geometry band is restored to finest + balance.
void adaptPolicy(DO& d, double h0) {
  const Index n = d.local().numLeaves();
  std::vector<double> g((std::size_t)n);
  for (Index i = 0; i < n; ++i) {
    auto b = d.local().bounds(i);
    const double s = static_cast<double>(Index(1) << d.local().level(i));
    Vec<3> ctr{};
    for (int a = 0; a < 3; ++a)
      ctr[a] = (static_cast<double>((long)b[0][a] + d.blockFineOrigin()[a]) + 0.5 * s) * h0;
    const double dx = ctr[0] - 0.30, dy = ctr[1] - 0.55, dz = ctr[2] - 0.45;
    g[(std::size_t)i] = std::exp(-40.0 * (dx * dx + dy * dy + dz * dz));
  }
  (void)distributedAdapt(d, g, /*refine=*/0.2, /*coarsen=*/0.02, /*finestLevel=*/0);
  refineBand(d, h0);
}

void run() {
  const long Nr = 4;  // 4^3 roots, lmax 2 ⇒ 16^3 fine, periodic [0,1)^3
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
  refineBand(world, h0);
  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_SELF);
  refineBand(self, h0);

  auto configure = [&](AmrFlow<kBits>& f) {
    f.setDensity(1.0);
    f.setViscosity(1.0);
    f.setAdvection(true);  // NS ⇒ AUTO ghost projection
    f.setDt(60.0);
    f.setBodyForce(1.0, 0.0, 0.0);
    f.setSolid(sphereSdf);
  };

  AmrFlow<kBits> fw;
  fw.initMpi(world);
  configure(fw);
  AmrFlow<kBits> fs;
  fs.init(self.local(), h0, Vec<3>{0.0, 0.0, 0.0});
  fs.initMpi(self);  // COMM_SELF distributed context: same adapt/rebalance API, 1 "rank"
  configure(fs);

  auto stepBoth = [&](int k) {
    for (int s = 0; s < k; ++s) {
      fw.step(200, 60);
      fs.step(200, 60);
    }
  };
  auto compare = [&](const char* what) {
    std::array<std::vector<double>, 3> uw, us;
    for (int c = 0; c < 3; ++c) {
      uw[(std::size_t)c] = fw.velocity(c);
      us[(std::size_t)c] = fs.velocity(c);
    }
    std::vector<double> pw = fw.pressure(), ps = fs.pressure();
    double scale = 0.0;
    for (std::size_t i = 0; i < us[0].size(); ++i)
      scale = std::max(scale, std::fabs(us[0][i]));
    double dmax = 0.0;
    const Index n = world.local().numLeaves();
    for (Index i = 0; i < n; ++i) {
      const Index si = self.local().find(world.globalCode(i));
      PECLET_CORE_CHECK(si >= 0);
      for (int c = 0; c < 3; ++c)
        dmax = std::max(dmax,
                        std::fabs(uw[(std::size_t)c][(std::size_t)i] -
                                  us[(std::size_t)c][(std::size_t)si]));
      dmax = std::max(dmax, std::fabs(pw[(std::size_t)i] - ps[(std::size_t)si]));
    }
    double gdmax = 0.0;
    MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("[%s] gdmax=%.3e scale=%.3e rel=%.3e\n", what, gdmax, scale,
                  scale > 0 ? gdmax / scale : 0.0);
    if (size == 1)
      PECLET_CORE_CHECK(gdmax == 0.0);
    else
      PECLET_CORE_CHECK(gdmax <= 1e-5 * scale);
  };

  // Phase 1: NS steps on the initial mesh.
  stepBoth(2);
  compare("pre-adapt");

  // Phase 2: mid-run distributed adapt (ownership-preserving) — identical policy both sides.
  fw.beginAdapt();
  adaptPolicy(world, h0);
  fw.finishAdapt(sphereSdf);
  fs.beginAdapt();
  adaptPolicy(self, h0);
  fs.finishAdapt(sphereSdf);
  {  // the adapted meshes agree globally (bit-identical flags + deterministic balance)
    long lw = (long)world.local().numLeaves(), gw = 0;
    MPI_Allreduce(&lw, &gw, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    PECLET_CORE_CHECK(gw == (long)self.local().numLeaves());
  }
  stepBoth(2);
  compare("post-adapt");

  // Phase 3: load rebalance, then continue. WORLD genuinely migrates state with the leaves;
  // the SELF reference "rebalances" over COMM_SELF (no migration, same rebuild + uf restart —
  // keeps the two advection paths symmetric so np=1 stays bit-exact).
  fw.rebalanceMpi(sphereSdf);
  fs.rebalanceMpi(sphereSdf);
  stepBoth(1);
  compare("post-rebalance");
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping distributed adapt-flow test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

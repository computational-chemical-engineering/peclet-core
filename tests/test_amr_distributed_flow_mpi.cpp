// The FULL distributed AmrFlow step (docs/amr_distributed_flow.md, rung 4): AmrFlow::initMpi
// on an ORB block of a graded sphere mesh, the whole Stokes step — momentum BiCGStab with the
// halo per matvec, the distributed pressure (aperture MG-PCG AND the ghost projection's
// binary-MG-preconditioned BiCGStab with the collective band guard), overlays, projection
// tail — multi-rank. Validated against the single-rank AmrFlow on the whole domain:
//   (1) np=1: u and p BIT-EXACT after several steps (the distributed path degenerates to the
//       single-rank arithmetic by construction — zero ghosts, identity reductions);
//   (2) np>1: u and p to Krylov tolerance (iterate sequences differ only through the dots'
//       reduction order), for BOTH the aperture and the ghost projection;
//   (3) the ±2 registry is genuinely exercised at np>1 (ghost count > 0), and the ghost-mode
//       run keeps the ghost-closed divergence at the single-rank scale.
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

void makeMesh(DO& d, double h0) {
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
      return std::fabs(sphereSdf(ctr)) < w + 3.0 * h0;  // generous finest band (ghost margin)
    });
  }
  d.balance();
}

struct Fields {
  std::array<std::vector<double>, 3> u;
  std::vector<double> p;
};

template <class FlowT>
Fields runSteps(FlowT& f, int steps) {
  for (int s = 0; s < steps; ++s)
    f.step(200, 60);
  Fields r;
  for (int c = 0; c < 3; ++c)
    r.u[(std::size_t)c] = f.velocity(c);
  r.p = f.pressure();
  return r;
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
  const int steps = 3;

  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeMesh(world, h0);
  const Index n = world.local().numLeaves();
  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_SELF);
  makeMesh(self, h0);
  const Index ns = self.local().numLeaves();

  // mode: 0 = Stokes aperture, 1 = Stokes ghost, 2 = NS (advection on ⇒ aperture default + the
  // uⁿ advection halo + the uf face field across steps).
  auto configure = [&](AmrFlow<kBits>& f, int mode) {
    f.setDensity(1.0);
    f.setViscosity(1.0);
    f.setBodyForce(1.0, 0.0, 0.0);
    if (mode == 2) {
      f.setAdvection(true);
      f.setDt(60.0);  // transient NS regime
    } else {
      f.setDt(1e6);  // steady-drag regime (the hard, large-dt momentum system)
      f.setGhostProjection(mode == 1);
    }
    f.setSolid(sphereSdf);
  };

  // One mode = one WORLD/SELF pair; compare u,p mapped by global code.
  auto compareMode = [&](int mode, double& gdmax, double& scale) {
    AmrFlow<kBits> fw;
    fw.initMpi(world);
    configure(fw, mode);
    if (size > 1)
      PECLET_CORE_CHECK(fw.numGhostCells() > 0);
    else
      PECLET_CORE_CHECK_EQ(fw.numGhostCells(), 0);
    Fields w = runSteps(fw, steps);

    AmrFlow<kBits> fs;
    fs.init(self.local(), h0, Vec<3>{0.0, 0.0, 0.0});
    configure(fs, mode);
    Fields s = runSteps(fs, steps);

    scale = 0.0;
    for (Index i = 0; i < ns; ++i)
      scale = std::max(scale, std::fabs(s.u[0][(std::size_t)i]));
    double dmax = 0.0;
    for (Index i = 0; i < n; ++i) {
      const Index si = self.local().find(world.globalCode(i));
      for (int c = 0; c < 3; ++c)
        dmax = std::max(dmax, std::fabs(w.u[(std::size_t)c][(std::size_t)i] -
                                        s.u[(std::size_t)c][(std::size_t)si]));
      dmax = std::max(dmax, std::fabs(w.p[(std::size_t)i] - s.p[(std::size_t)si]));
    }
    gdmax = 0.0;
    MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // The distributed divergence diagnostic stays at the single-rank scale.
    const double dw = fw.divNormL2(), dsv = fs.divNormL2();
    PECLET_CORE_CHECK(std::isfinite(dw));
    PECLET_CORE_CHECK(dw <= 10.0 * dsv + 1e-12);
  };

  // ---- aperture (Stokes) / ghost (Stokes) / NS (auto-ghost + advection halo) ----
  for (int mode = 0; mode < 3; ++mode) {
    double gdmax = 0.0, scale = 0.0;
    compareMode(mode, gdmax, scale);
    if (size == 1)
      PECLET_CORE_CHECK(gdmax == 0.0);
    else
      PECLET_CORE_CHECK(gdmax <= 5e-6 * scale);
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping distributed AmrFlow test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

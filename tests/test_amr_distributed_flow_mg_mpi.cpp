// Distributed device openness multigrid + MG-PCG for the AMR flow pressure
// (peclet::core::amr::DistributedFlowMultigrid, docs/amr_distributed_flow.md rung 3).
// On a graded, cross-block 2:1-balanced octree with a genuine cut-cell aperture openness
// (sphere) it validates, at np = 1,2,4,8:
//   (1) hierarchy parity: the distributed ladder has exactly the single-rank level count
//       (per-rank coarsenIf + global-max padding == the whole-domain ladder);
//   (2) V-cycle WORLD==SELF BIT-EXACT with mean removal off: a fixed number of V-cycles on
//       the distributed hierarchy reproduces the single-rank device Multigrid bit-for-bit on
//       every backend config (Jacobi smoothing, local transfers and the per-level halo are
//       all order-independent — the only reductions live in the mean removal, which is off);
//   (3) the distributed MG-PCG (removeMean + fluid-mask projection + Allreduce'd dots) solves
//       the singular periodic openness Poisson and matches the single-rank PCG solution:
//       np=1 bit-exact, np>1 to Krylov tolerance (dot reduction order).
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/common/view.hpp"

#include "peclet/core/amr/distributed_flow_mg.hpp"
#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/multigrid.hpp"
#include "peclet/core/amr/pcg.hpp"
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

double fAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0,
         cz = ((double)o[2] + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cx);
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
      return std::fabs(sphereSdf(ctr)) < w + 2.0 * h0;
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
  const long Nr = 4;  // 4^3 roots, lmax 2 ⇒ 16^3 fine, periodic [0,1)^3
  const unsigned lmax = 2;
  const double h0 = 1.0 / (Nr * (1 << lmax));
  AmrGeometry<3> geo;
  geo.h0 = h0;
  const std::array<bool, 3> per{true, true, true};
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // A symmetric world-coordinate aperture openness from the sphere SDF.
  auto openFn = [&](const Vec<3>& fc, int) -> double {
    const double a = 0.5 + sphereSdf(fc) / h0;
    return a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);
  };

  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeMesh(world, h0);
  const Index n = world.local().numLeaves();

  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_SELF);
  makeMesh(self, h0);
  const Index ns = self.local().numLeaves();

  DistributedFlowMultigrid<3, kBits> dmg;
  dmg.build(world, h0, openFn);
  Multigrid<3, kBits> smg;
  smg.build(self.local(), h0, openFn, /*periodic=*/true);

  // (1) hierarchy parity.
  PECLET_CORE_CHECK_EQ((long)dmg.numLevels(), (long)smg.numLevels());
  if (size > 1)
    PECLET_CORE_CHECK(dmg.halo(0).numGhosts() > 0);
  if (size == 1)
    PECLET_CORE_CHECK_EQ(dmg.halo(0).numGhosts(), 0);

  // RHS keyed on the global code (WORLD and SELF agree per cell).
  std::vector<double> bw((std::size_t)dmg.extendedSize(0), 0.0), bs((std::size_t)ns);
  for (Index i = 0; i < n; ++i)
    bw[(std::size_t)i] = fAt(world.globalCode(i), h0);
  for (Index i = 0; i < ns; ++i)
    bs[(std::size_t)i] = fAt(self.globalCode(i), h0);

  // (2) V-cycles bit-exact WORLD==SELF (mean removal off; order-independent pieces only).
  {
    dmg.setRemoveMean(false);
    View<double> bwv = toDevice(bw, "bw");
    Kokkos::deep_copy(dmg.b(0), bwv);
    Kokkos::deep_copy(dmg.x(0), 0.0);
    View<double> bsv = toDevice(bs, "bs");
    Kokkos::deep_copy(smg.b(0), bsv);
    Kokkos::deep_copy(smg.x(0), 0.0);
    for (int c = 0; c < 3; ++c) {
      dmg.vcycle(2, 2, 60, 0.8);
      smg.vcycle(2, 2, 60, 0.8);
    }
    std::vector<double> xw = down(dmg.x(0)), xs = down(smg.x(0));
    double dmax = 0.0;
    for (Index i = 0; i < n; ++i) {
      const Index si = self.local().find(world.globalCode(i));
      dmax = std::max(dmax, std::fabs(xw[(std::size_t)i] - xs[(std::size_t)si]));
    }
    double gdmax = 0.0;
    MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    PECLET_CORE_CHECK(gdmax == 0.0);  // bit-for-bit across rank counts
  }

  // (3) distributed MG-PCG vs single-rank MG-PCG on the singular periodic openness Poisson.
  {
    dmg.setRemoveMean(true);
    PCG<3, kBits> dpcg, spcg;
    dpcg.setVcycle(2, 2, 60, 0.8);
    spcg.setVcycle(2, 2, 60, 0.8);
    dpcg.setSingular(true);
    spcg.setSingular(true);
    dpcg.setDistributed([&dmg](View<double> v) { dmg.sync(0, v); },
                        [](double s) {
                          double g = 0.0;
                          MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                          return g;
                        },
                        dmg.extendedSize(0));
    View<double> bwv = toDevice(bw, "bw2");
    View<double> bsv = toDevice(bs, "bs2");
    View<double> xw("xw", (std::size_t)dmg.extendedSize(0)), xs("xs", (std::size_t)ns);
    auto rw = dpcg.solve(dmg, xw, View<const double>(bwv), 200, 1e-10);
    // The single-rank reference must run the SAME mean-removal policy in the preconditioner.
    smg.setRemoveMean(true);
    auto rs = spcg.solve(smg, xs, View<const double>(bsv), 200, 1e-10);
    PECLET_CORE_CHECK(rw.res <= 1e-9 * rw.res0);
    PECLET_CORE_CHECK(rs.res <= 1e-9 * rs.res0);
    std::vector<double> hw = down(xw), hs = down(xs);
    double umax = 0.0;
    for (Index i = 0; i < ns; ++i)
      umax = std::max(umax, std::fabs(hs[(std::size_t)i]));
    double dmax = 0.0;
    for (Index i = 0; i < n; ++i) {
      const Index si = self.local().find(world.globalCode(i));
      dmax = std::max(dmax, std::fabs(hw[(std::size_t)i] - hs[(std::size_t)si]));
    }
    double gdmax = 0.0;
    MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (size == 1) {
      PECLET_CORE_CHECK(gdmax == 0.0);  // np=1: bit-exact vs single-rank
    } else {
      PECLET_CORE_CHECK(gdmax <= 1e-7 * umax);  // Krylov tolerance (dot order differs)
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping distributed flow-MG test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

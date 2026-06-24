// Geometric-multigrid V-cycle on a *graded* distributed octree
// (tpx::amr::GradedDistributedMultigrid). On a genuinely graded, cross-block
// 2:1-balanced octree it validates:
//   (1) the full V-cycle distributed over ORB blocks (MPI_COMM_WORLD) == the same on
//       the whole domain as one block (MPI_COMM_SELF) bit-for-bit (consistent per-level
//       operator + local average restriction + piecewise-constant prolongation, all
//       order-independent / per-cell);
//   (2) the V-cycle actually solves — the residual drops well past plain Jacobi.
// np = 1,2,4,8.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include "tpx/amr/distributed_fv.hpp"
#include "tpx/amr/distributed_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/common/mpi.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

double fAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0, cz = ((double)o[2] + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cx);
}

void makeGraded(DO& d) {
  for (int pass = 0; pass < 2; ++pass) {
    d.local().refineIf([&](Code c, unsigned lvl) -> bool {
      if (lvl == 0) return false;
      auto o = M::from_code(c).decode();
      for (int dd = 0; dd < 3; ++dd)
        if ((long)o[dd] + d.blockFineOrigin()[dd] >= 8) return false;
      return true;
    });
  }
  d.balance();
}

void run() {
  const long Nr = 4;
  const unsigned lmax = 2;  // 16^3 fine, periodic [0,1)^3
  const double h0 = 1.0 / (Nr * (1 << lmax));
  AmrGeometry<3> geo;
  geo.h0 = h0;
  const std::array<bool, 3> per{true, true, true};
  const int cycles = 25;
  const int bottom = 200;  // bottom = uniform 4^3 root brick; Jacobi-solve it well

  // Manufactured RHS b = L·u_exact: exactly volume-weighted-mean-zero (conservation
  // cancels the face contributions bit-wise), so the singular operator has no
  // nullspace floor and the residual converges to ~0. b is computed by the bit-exact
  // apply(), so it matches at every cell WORLD vs SELF.

  // ---- distributed (COMM_WORLD) ----
  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeGraded(world);
  GradedDistributedMultigrid<3, kBits> mgw;
  mgw.build(world);
  const Index nw = mgw.numLeaves();
  std::vector<double> uw((std::size_t)nw), bw, xw((std::size_t)nw, 0.0);
  for (Index i = 0; i < nw; ++i) uw[(std::size_t)i] = fAt(world.globalCode(i), h0);
  mgw.op().apply(uw, bw);
  const double r0 = mgw.op().residualNorm(xw, bw);
  for (int c = 0; c < cycles; ++c) mgw.vcycle(xw, bw, 2, 2, bottom);
  const double r1 = mgw.op().residualNorm(xw, bw);

  // ---- serial reference (whole domain, COMM_SELF) ----
  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_SELF);
  makeGraded(self);
  GradedDistributedMultigrid<3, kBits> mgs;
  mgs.build(self);
  const Index ns = mgs.numLeaves();
  std::vector<double> us((std::size_t)ns), bs, xs((std::size_t)ns, 0.0);
  for (Index i = 0; i < ns; ++i) us[(std::size_t)i] = fAt(self.globalCode(i), h0);
  mgs.op().apply(us, bs);
  for (int c = 0; c < cycles; ++c) mgs.vcycle(xs, bs, 2, 2, bottom);

  // (1) bit-for-bit WORLD == SELF
  int mism = 0;
  for (Index i = 0; i < nw; ++i) {
    Index si = self.local().find(world.globalCode(i));
    if (si < 0) {
      ++mism;
      continue;
    }
    if (xw[(std::size_t)i] != xs[(std::size_t)si]) ++mism;
  }
  TPX_CHECK_EQ(mism, 0);
  TPX_CHECK(mgw.numLevels() >= 3);

  // (2) the V-cycle solves: residual converges to ~round-off (≥8 orders) on the
  // manufactured RHS — the graded distributed MG is a genuine solver.
  TPX_CHECK(r1 < r0 * 1e-8);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  run();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int fails = tpx::test::g_failures, total = 0;
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
  std::printf("TPX_HAVE_MORTON not set — skipping graded distributed MG test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

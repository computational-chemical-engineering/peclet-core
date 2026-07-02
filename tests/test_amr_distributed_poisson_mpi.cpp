// Distributed Poisson operator/smoother (peclet::core::amr::DistributedPoisson) on the
// owner-based octree halo: a weighted-Jacobi solve distributed over ORB blocks
// (MPI_COMM_WORLD) must match the same solve on the whole domain as one block
// (MPI_COMM_SELF) bit-for-bit — Jacobi reads only the previous iterate, so the
// halo supplies exactly the cells a single-block solve would. np = 1,2,4.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <vector>

#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/distributed_poisson.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/common/mpi.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

// RHS as a function of a leaf's global Morton code (so WORLD and SELF agree at the
// same physical cell). Uniform mesh: level 0, cell size 1 fine unit, centre+0.5.
double bAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = (static_cast<double>(o[0]) + 0.5) * h0;
  double cy = (static_cast<double>(o[1]) + 0.5) * h0;
  double cz = (static_cast<double>(o[2]) + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cy);
}

void run() {
  const long N = 8;  // uniform N^3 periodic domain [0,1)^3 (root cells = finest)
  const double h0 = 1.0 / N;
  AmrGeometry<3> geo;
  geo.h0 = h0;
  const std::array<bool, 3> per{true, true, true};
  const int sweeps = 40;

  // distributed over MPI_COMM_WORLD
  DO world;
  world.init(IVec<3>{N, N, N}, /*lmax=*/0, geo, per, MPI_COMM_WORLD);
  DistributedPoisson<3, kBits> dpw;
  dpw.init(world, h0);
  const Index nw = world.local().numLeaves();
  std::vector<double> bw(static_cast<std::size_t>(nw)), xw(static_cast<std::size_t>(nw), 0.0);
  for (Index i = 0; i < nw; ++i)
    bw[static_cast<std::size_t>(i)] = bAt(world.globalCode(i), h0);
  dpw.jacobi(xw, bw, sweeps);

  // serial reference: whole domain as ONE block on MPI_COMM_SELF, same solve.
  DO self;
  self.init(IVec<3>{N, N, N}, 0, geo, per, MPI_COMM_SELF);
  DistributedPoisson<3, kBits> dps;
  dps.init(self, h0);
  const Index ns = self.local().numLeaves();
  std::vector<double> bs(static_cast<std::size_t>(ns)), xs(static_cast<std::size_t>(ns), 0.0);
  for (Index i = 0; i < ns; ++i)
    bs[static_cast<std::size_t>(i)] = bAt(self.globalCode(i), h0);
  dps.jacobi(xs, bs, sweeps);

  // SELF block covers the whole domain with origin 0, so its local code == global code.
  int mism = 0;
  double maxdiff = 0.0;
  for (Index i = 0; i < nw; ++i) {
    Index si = self.local().find(world.globalCode(i));
    if (si < 0) {
      ++mism;
      continue;
    }
    double d = std::fabs(xw[static_cast<std::size_t>(i)] - xs[static_cast<std::size_t>(si)]);
    maxdiff = std::max(maxdiff, d);
  }
  PECLET_CORE_CHECK_EQ(mism, 0);
  PECLET_CORE_CHECK(maxdiff == 0.0);  // bit-for-bit (Jacobi is order-independent)
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping distributed Poisson test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

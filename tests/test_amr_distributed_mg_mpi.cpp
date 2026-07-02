// Distributed geometric-multigrid V-cycle (peclet::core::amr::DistributedMultigrid): a full
// V-cycle hierarchy on the ORB-decomposed octree (MPI_COMM_WORLD) must (1) match the
// same V-cycle on the whole domain as one block (MPI_COMM_SELF) bit-for-bit, and
// (2) actually solve — the Poisson residual drops by orders of magnitude in a handful
// of cycles (the point of MG over plain Jacobi). np = 1,2,4,8.
//
// Bit-exactness across rank counts holds because every ingredient is order-independent
// or per-cell: Jacobi smoother, local volume-average restriction (children of a coarse
// cell are all on one rank — the nested ORB decompositions guarantee it), and
// piecewise-constant prolongation.
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

// Compatible (mean-zero over the periodic domain) RHS, keyed on the global Morton code
// so WORLD and SELF agree at the same physical cell.
double bAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = (static_cast<double>(o[0]) + 0.5) * h0;
  double cy = (static_cast<double>(o[1]) + 0.5) * h0;
  double cz = (static_cast<double>(o[2]) + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cy);
}

void run() {
  const long N = 16;  // 16^3 periodic [0,1)^3 ⇒ hierarchy 16→8→4→2 (4 levels)
  const double h0 = 1.0 / N;
  AmrGeometry<3> geo;
  geo.h0 = h0;
  const std::array<bool, 3> per{true, true, true};
  const int cycles = 8;

  // distributed over MPI_COMM_WORLD
  DistributedMultigrid<3, kBits> mgw;
  mgw.build(IVec<3>{N, N, N}, geo, per, MPI_COMM_WORLD);
  const Index nw = mgw.numLeaves();
  std::vector<double> bw(static_cast<std::size_t>(nw)), xw(static_cast<std::size_t>(nw), 0.0);
  for (Index i = 0; i < nw; ++i)
    bw[static_cast<std::size_t>(i)] = bAt(mgw.octree().globalCode(i), h0);
  const double r0 = mgw.op().residualNorm(xw, bw);
  for (int c = 0; c < cycles; ++c)
    mgw.vcycle(xw, bw);
  const double r1 = mgw.op().residualNorm(xw, bw);

  // serial reference: whole domain as ONE block on MPI_COMM_SELF, same V-cycles.
  DistributedMultigrid<3, kBits> mgs;
  mgs.build(IVec<3>{N, N, N}, geo, per, MPI_COMM_SELF);
  const Index ns = mgs.numLeaves();
  std::vector<double> bs(static_cast<std::size_t>(ns)), xs(static_cast<std::size_t>(ns), 0.0);
  for (Index i = 0; i < ns; ++i)
    bs[static_cast<std::size_t>(i)] = bAt(mgs.octree().globalCode(i), h0);
  for (int c = 0; c < cycles; ++c)
    mgs.vcycle(xs, bs);

  // The SELF block covers the whole domain (origin 0) ⇒ local code == global code.
  int mism = 0;
  double maxdiff = 0.0;
  for (Index i = 0; i < nw; ++i) {
    Index si = mgs.octree().local().find(mgw.octree().globalCode(i));
    if (si < 0) {
      ++mism;
      continue;
    }
    maxdiff = std::max(
        maxdiff, std::fabs(xw[static_cast<std::size_t>(i)] - xs[static_cast<std::size_t>(si)]));
  }
  PECLET_CORE_CHECK_EQ(mism, 0);
  PECLET_CORE_CHECK(maxdiff == 0.0);  // bit-for-bit across rank counts
  PECLET_CORE_CHECK(mgw.numLevels() == 4);
  PECLET_CORE_CHECK(r1 < r0 * 1e-3);  // MG actually solves (≥ 3 orders in 8 cycles)
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping distributed MG test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

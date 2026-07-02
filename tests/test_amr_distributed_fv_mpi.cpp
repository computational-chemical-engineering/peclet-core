// Consistent graded FV operator on a *distributed* octree
// (peclet::core::amr::DistributedFvOperator). On a genuinely graded, cross-block 2:1-balanced
// octree it validates:
//   (1) the operator distributed over ORB blocks (MPI_COMM_WORLD) == the same on the
//       whole domain as one block (MPI_COMM_SELF) bit-for-bit — apply AND a Jacobi
//       solve (the per-entry w(val−u_i) is summed in the same face order whether the
//       neighbour is local or a ghost, incl. the 2^(Dim-1) fine sub-neighbours);
//   (2) the single-block operator == host AmrPoisson::applyLaplacian bit-for-bit
//       (same conservative coeffs + 2:1 sub-face enumeration);
//   (3) the distributed Jacobi smoother reduces the residual.
// np = 1,2,4,8.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <vector>

#include "peclet/core/amr/distributed_fv.hpp"
#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/mpi.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

// Field keyed on the global Morton code so WORLD and SELF agree at the same cell.
double fAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0,
         cz = ((double)o[2] + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cx);
}

// Refine the lower octant (global fine < 8 on every axis) two levels, then 2:1-balance
// — produces a graded octree with cross-block interfaces.
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

  // ----- distributed (COMM_WORLD) -----
  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeGraded(world);
  DistributedFvOperator<3, kBits> opw;
  opw.init(world);
  const Index nw = world.local().numLeaves();

  // ----- serial reference (whole domain, COMM_SELF) -----
  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_SELF);
  makeGraded(self);
  DistributedFvOperator<3, kBits> ops;
  ops.init(self);
  const Index ns = self.local().numLeaves();

  // global leaf count matches (sanity that the graded octree is the same one)
  long lw = (long)nw, gw = 0;
  MPI_Allreduce(&lw, &gw, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  PECLET_CORE_CHECK(gw == (long)ns);

  // fields
  std::vector<double> fw((std::size_t)nw), fs((std::size_t)ns);
  for (Index i = 0; i < nw; ++i)
    fw[(std::size_t)i] = fAt(world.globalCode(i), h0);
  for (Index i = 0; i < ns; ++i)
    fs[(std::size_t)i] = fAt(self.globalCode(i), h0);

  // ===== (2) SELF operator == host AmrPoisson::applyLaplacian (bit-for-bit) =====
  {
    AmrPoisson<3, kBits> ap;
    ap.init(self.local(), h0);
    std::vector<double> hostLu, opLu;
    ap.applyLaplacian(fs, hostLu);
    ops.apply(fs, opLu);
    int mism = 0;
    for (Index i = 0; i < ns; ++i)
      if (opLu[(std::size_t)i] != hostLu[(std::size_t)i])
        ++mism;
    PECLET_CORE_CHECK_EQ(mism, 0);
  }

  // ===== (1a) apply: WORLD == SELF bit-for-bit =====
  std::vector<double> luw, lus;
  opw.apply(fw, luw);
  ops.apply(fs, lus);
  int amis = 0;
  for (Index i = 0; i < nw; ++i) {
    Index si = self.local().find(world.globalCode(i));
    if (si < 0) {
      ++amis;
      continue;
    }
    if (luw[(std::size_t)i] != lus[(std::size_t)si])
      ++amis;
  }
  PECLET_CORE_CHECK_EQ(amis, 0);

  // ===== (1b) Jacobi solve: WORLD == SELF bit-for-bit, and (3) residual drops =====
  std::vector<double> bw((std::size_t)nw), bs((std::size_t)ns), xw((std::size_t)nw, 0.0),
      xs((std::size_t)ns, 0.0);
  // mean-zero-ish smooth RHS (same physical field on both)
  for (Index i = 0; i < nw; ++i)
    bw[(std::size_t)i] = fAt(world.globalCode(i), h0);
  for (Index i = 0; i < ns; ++i)
    bs[(std::size_t)i] = fAt(self.globalCode(i), h0);
  const double r0 = opw.residualNorm(xw, bw);
  const int sweeps = 200;
  opw.jacobi(xw, bw, sweeps, 0.8);
  ops.jacobi(xs, bs, sweeps, 0.8);
  const double r1 = opw.residualNorm(xw, bw);
  int jmis = 0;
  for (Index i = 0; i < nw; ++i) {
    Index si = self.local().find(world.globalCode(i));
    if (si < 0) {
      ++jmis;
      continue;
    }
    if (xw[(std::size_t)i] != xs[(std::size_t)si])
      ++jmis;
  }
  PECLET_CORE_CHECK_EQ(jmis, 0);
  PECLET_CORE_CHECK(r1 < 0.5 * r0);  // the smoother reduces the residual
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping distributed FV test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

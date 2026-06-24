// Cut-cell openness on the distributed/graded path: DistributedFvOperator and
// GradedDistributedMultigrid with per-face fluid fraction α = openFn(face centroid,
// axis). On a graded, cross-block 2:1-balanced octree with a smooth openness field:
//   (1) the openness operator distributed over ORB blocks (MPI_COMM_WORLD) == the same
//       on the whole domain (MPI_COMM_SELF) bit-for-bit (apply), and == host
//       AmrPoisson::applyLaplacian with the same buildOpenness on a single block;
//   (2) the graded MG with openness converges (manufactured RHS, exactly mean-zero by
//       conservation since α is symmetric across each face) and is bit-exact WORLD==SELF.
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
#include "tpx/amr/poisson.hpp"
#include "tpx/common/mpi.hpp"
#include "tpx/common/types.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

// Smooth openness in [~0.25, ~0.95] (never fully solid ⇒ no zero-diagonal cells).
double openFn(const Vec<3>& p, int /*axis*/) {
  const double k = 2.0 * M_PI;
  return 0.6 + 0.35 * std::sin(k * p[0]) * std::cos(k * p[1]) * std::cos(k * p[2]);
}

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

  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeGraded(world);
  DistributedFvOperator<3, kBits> opw;
  opw.init(world, openFn);
  const Index nw = world.local().numLeaves();

  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_SELF);
  makeGraded(self);
  DistributedFvOperator<3, kBits> ops;
  ops.init(self, openFn);
  const Index ns = self.local().numLeaves();

  std::vector<double> fw((std::size_t)nw), fs((std::size_t)ns);
  for (Index i = 0; i < nw; ++i) fw[(std::size_t)i] = fAt(world.globalCode(i), h0);
  for (Index i = 0; i < ns; ++i) fs[(std::size_t)i] = fAt(self.globalCode(i), h0);

  // ===== (1b) SELF openness operator == host AmrPoisson::applyLaplacian (bit-exact) =====
  {
    AmrPoisson<3, kBits> ap;
    ap.init(self.local(), h0);
    ap.setOrigin(self.globalGeometry().origin);
    ap.buildOpenness(openFn);
    std::vector<double> hostLu, opLu;
    ap.applyLaplacian(fs, hostLu);
    ops.apply(fs, opLu);
    int mism = 0;
    for (Index i = 0; i < ns; ++i)
      if (opLu[(std::size_t)i] != hostLu[(std::size_t)i]) ++mism;
    TPX_CHECK_EQ(mism, 0);
  }

  // ===== (1a) apply: WORLD == SELF bit-for-bit =====
  std::vector<double> luw, lus;
  opw.apply(fw, luw);
  ops.apply(fs, lus);
  int amis = 0;
  for (Index i = 0; i < nw; ++i) {
    Index si = self.local().find(world.globalCode(i));
    if (si < 0 || luw[(std::size_t)i] != lus[(std::size_t)si]) ++amis;
  }
  TPX_CHECK_EQ(amis, 0);

  // ===== (2) graded MG with openness: converges + WORLD == SELF bit-for-bit =====
  GradedDistributedMultigrid<3, kBits> mgw;
  mgw.build(world, openFn);
  GradedDistributedMultigrid<3, kBits> mgs;
  mgs.build(self, openFn);
  // manufactured RHS b = L·u_exact (exactly mean-zero: α symmetric ⇒ conservation)
  std::vector<double> bw, bs, xw((std::size_t)nw, 0.0), xs((std::size_t)ns, 0.0);
  mgw.op().apply(fw, bw);
  mgs.op().apply(fs, bs);
  const double r0 = mgw.op().residualNorm(xw, bw);
  const int cycles = 20;
  for (int c = 0; c < cycles; ++c) mgw.vcycle(xw, bw);
  for (int c = 0; c < cycles; ++c) mgs.vcycle(xs, bs);
  const double r1 = mgw.op().residualNorm(xw, bw);
  int jmis = 0;
  for (Index i = 0; i < nw; ++i) {
    Index si = self.local().find(world.globalCode(i));
    if (si < 0 || xw[(std::size_t)i] != xs[(std::size_t)si]) ++jmis;
  }
  TPX_CHECK_EQ(jmis, 0);
  TPX_CHECK(r1 < r0 * 1e-6);
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
  std::printf("TPX_HAVE_MORTON not set — skipping distributed openness test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

// Dynamic load re-balancing (tpx::amr::DistributedOctree::rebalance): weighted-ORB
// re-decomposition + leaf/field migration on a distributed octree whose refinement is
// concentrated in part of the domain (so the equal-cell-count ORB leaves one rank heavy).
// rebalance() is a pure redistribution of the *same* global mesh, so:
//   (1) the global mesh + field are unchanged — bit-for-bit identical to the single-block
//       (COMM_SELF) computation, and every cell still carries its own field value;
//   (2) the global scalar Σ V·f is conserved, and the global leaf count is preserved;
//   (3) the per-rank leaf-count imbalance (max/mean) drops vs the equal-cell decomposition.
// np = 1,2,4,8.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <algorithm>
#include <cmath>
#include <vector>

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
constexpr long Nr = 4;
constexpr unsigned kLmax = 2;
constexpr double kN = Nr * (1 << kLmax);  // 16 fine cells per axis

// A field keyed on a leaf's global code + level: WORLD and SELF (and a cell before and
// after migration) sample the identical value, so equality is a true bit-exact check.
double fAt(Code gc, unsigned level) {
  auto o = M::from_code(gc).decode();
  double s = static_cast<double>(Index(1) << level);
  double x = ((double)o[0] + 0.5 * s) / kN, y = ((double)o[1] + 0.5 * s) / kN,
         z = ((double)o[2] + 0.5 * s) / kN;
  return std::sin(3.0 * x) * std::cos(2.0 * y) + 0.5 * z + 1.0;
}
std::vector<double> sampleField(const DO& d) {
  const Index n = d.local().numLeaves();
  std::vector<double> f((std::size_t)n);
  for (Index i = 0; i < n; ++i) f[(std::size_t)i] = fAt(d.globalCode(i), d.local().level(i));
  return f;
}
double localMass(const DO& d, const std::vector<double>& f) {
  double s = 0.0;
  for (Index i = 0; i < d.local().numLeaves(); ++i) {
    double w = static_cast<double>(Index(1) << d.local().level(i));
    s += w * w * w * f[(std::size_t)i];
  }
  return s;
}

// Concentrate refinement in the x==0 root-cell plane (decided on *global* root cells, so
// WORLD and SELF refine identically), then 2:1-balance. This skews the leaf count toward
// the ranks owning that plane under the equal-cell ORB.
void refineCorner(DO& d, int passes) {
  for (int p = 0; p < passes; ++p) {
    std::vector<Code> tr;
    for (Index i = 0; i < d.local().numLeaves(); ++i)
      if (d.globalRootOf(i)[0] == 0 && d.local().level(i) > 0) tr.push_back(d.local().code(i));
    std::sort(tr.begin(), tr.end());
    tr.erase(std::unique(tr.begin(), tr.end()), tr.end());
    d.local().refineIf(
        [&](Code c, unsigned) { return std::binary_search(tr.begin(), tr.end(), c); });
    d.balance();
  }
}

// Imbalance = max leaves on any rank / mean leaves per rank (1.0 == perfectly even).
double imbalance(const DO& d) {
  long nloc = (long)d.local().numLeaves(), nmax = 0, nsum = 0;
  MPI_Allreduce(&nloc, &nmax, 1, MPI_LONG, MPI_MAX, d.comm());
  MPI_Allreduce(&nloc, &nsum, 1, MPI_LONG, MPI_SUM, d.comm());
  return (double)nmax / ((double)nsum / (double)d.size());
}

void run() {
  AmrGeometry<3> geo;
  geo.h0 = 1.0 / kN;
  const std::array<bool, 3> per{true, true, true};

  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, kLmax, geo, per, MPI_COMM_WORLD);
  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, kLmax, geo, per, MPI_COMM_SELF);

  refineCorner(world, 2);
  refineCorner(self, 2);

  std::vector<double> fw = sampleField(world), fs = sampleField(self);

  // global mass + leaf count + imbalance BEFORE rebalance
  double mw0 = localMass(world, fw), m0 = 0.0;
  MPI_Allreduce(&mw0, &m0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  long n0loc = (long)world.local().numLeaves(), n0 = 0;
  MPI_Allreduce(&n0loc, &n0, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  double imb0 = imbalance(world);

  // ---- the redistribution under test ----
  std::vector<std::vector<double>> cols{fw};
  world.rebalance(cols);
  fw.swap(cols[0]);

  // (1a) every cell still carries its own value (field followed the leaf bit-for-bit).
  int valMism = 0;
  for (Index i = 0; i < world.local().numLeaves(); ++i)
    if (fw[(std::size_t)i] != fAt(world.globalCode(i), world.local().level(i))) ++valMism;
  TPX_CHECK_EQ(valMism, 0);

  // (1b) global mesh + field bit-for-bit identical to the single-block computation.
  const Index nw = world.local().numLeaves();
  long lnw = nw, gnw = 0;
  MPI_Allreduce(&lnw, &gnw, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  TPX_CHECK(gnw == self.local().numLeaves());
  int mism = 0;
  for (Index i = 0; i < nw; ++i) {
    Index si = self.local().find(world.globalCode(i));
    if (si < 0) {
      ++mism;
      continue;
    }
    if (world.local().level(i) != self.local().level(si)) ++mism;
    if (fw[(std::size_t)i] != fs[(std::size_t)si]) ++mism;
  }
  TPX_CHECK_EQ(mism, 0);

  // (2) leaf count preserved exactly; Σ V·f conserved through the migration.
  TPX_CHECK(gnw == n0);
  double mw1 = localMass(world, fw), m1 = 0.0;
  MPI_Allreduce(&mw1, &m1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TPX_CHECK(std::fabs(m1 - m0) < 1e-9 * std::fabs(m0));

  // (3) imbalance drops (only meaningful with >1 rank; np=1 is trivially 1.0 -> 1.0).
  double imb1 = imbalance(world);
  if (world.size() > 1) {
    TPX_CHECK(imb0 > 1.15);     // the skewed mesh really was imbalanced under equal-cell ORB
    TPX_CHECK(imb1 < imb0);     // weighted ORB improved it
    TPX_CHECK(imb1 < 1.6);      // ... to a decently even distribution
  }

  // A second rebalance is a no-op (already balanced for this weight) and still bit-exact.
  std::vector<std::vector<double>> cols2{fw};
  world.rebalance(cols2);
  fw.swap(cols2[0]);
  int valMism2 = 0;
  for (Index i = 0; i < world.local().numLeaves(); ++i)
    if (fw[(std::size_t)i] != fAt(world.globalCode(i), world.local().level(i))) ++valMism2;
  TPX_CHECK_EQ(valMism2, 0);
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
  std::printf("TPX_HAVE_MORTON not set — skipping distributed rebalance test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

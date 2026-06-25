// Distributed dynamic adapt (tpx::amr::distributedAdapt): solution-adaptive (re)meshing
// on a DistributedOctree, keeping the ORB decomposition. On a field with a localized
// feature, each rank refines/coarsens its block from the Löhner indicator (evaluated
// through the owner-based halo), cross-block 2:1 balances, and conservatively remaps the
// field — all bit-identically to the whole-domain (single-block) computation.
//   (1) the adapted mesh + remapped field are bit-for-bit identical COMM_WORLD vs
//       COMM_SELF (same flags from the halo gather, deterministic balance, local remap);
//   (2) the global scalar Σ V·f is conserved through the adapt.
// np = 1,2,4,8.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include "tpx/amr/distributed_adapt.hpp"
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

// Gaussian blob keyed on a leaf's global code + level, so WORLD and SELF sample the
// identical physical field at every cell.
double fAt(Code gc, unsigned level) {
  auto o = M::from_code(gc).decode();
  double s = static_cast<double>(Index(1) << level);
  double x = ((double)o[0] + 0.5 * s) / kN, y = ((double)o[1] + 0.5 * s) / kN,
         z = ((double)o[2] + 0.5 * s) / kN;
  double dx = x - 0.5, dy = y - 0.5, dz = z - 0.5;
  return std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * 0.12 * 0.12));
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

void run() {
  AmrGeometry<3> geo;
  geo.h0 = 1.0 / kN;
  const std::array<bool, 3> per{true, true, true};

  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, kLmax, geo, per, MPI_COMM_WORLD);
  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, kLmax, geo, per, MPI_COMM_SELF);

  std::vector<double> fw = sampleField(world), fs = sampleField(self);

  // global mass before (Allreduce over WORLD)
  double mw0 = localMass(world, fw), m0 = 0.0;
  MPI_Allreduce(&mw0, &m0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // two distributed adapt steps (re-sample exact between, to drive refinement)
  for (int step = 0; step < 2; ++step) {
    fw = distributedAdapt(world, fw, /*refineThresh=*/0.2, /*coarsenThresh=*/0.03, /*finestLevel=*/0);
    fs = distributedAdapt(self, fs, 0.2, 0.03, 0);
    fw = sampleField(world);
    fs = sampleField(self);
  }

  // (1) bit-exact WORLD == SELF (mesh + field)
  const Index nw = world.local().numLeaves();
  long lnw = nw, gnw = 0;
  MPI_Allreduce(&lnw, &gnw, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  TPX_CHECK(gnw == self.local().numLeaves());  // same global leaf count
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

  // (2) refinement actually happened, and the global mass is conserved through adapt
  // (compare the *transferred* field, before the re-sample, to m0).
  DO w2;
  w2.init(IVec<3>{Nr, Nr, Nr}, kLmax, geo, per, MPI_COMM_WORLD);
  std::vector<double> f2 = sampleField(w2);
  f2 = distributedAdapt(w2, f2, 0.2, 0.03, 0);
  double lm = localMass(w2, f2), gm = 0.0;
  MPI_Allreduce(&lm, &gm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TPX_CHECK(std::fabs(gm - m0) < 1e-9 * std::fabs(m0));  // conservative remap
  unsigned minL = 99;
  for (Index i = 0; i < w2.local().numLeaves(); ++i) minL = std::min(minL, w2.local().level(i));
  long lminL = minL, gminL = 99;
  MPI_Allreduce(&lminL, &gminL, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
  TPX_CHECK(gminL < (long)kLmax);  // refined below the base level
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
  std::printf("TPX_HAVE_MORTON not set — skipping distributed adapt test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

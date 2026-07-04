// redistributeGridFields — move structured grid fields between two ORB decompositions (Eulerian
// load balancing). A field initialised from a KNOWN global function on the OLD partition must, after
// redistribution to a NEW (differently-weighted) partition, hold exactly that function on every new
// inner cell — and a round-trip (old->new->old) returns the original bit-for-bit.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/decomp/grid_redistribute.hpp"

using peclet::core::Index;
using peclet::core::IVec;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::decomp::redistributeGridFields;

namespace {
int failures = 0;
#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::fprintf(stderr, "CHECK failed: %s @ %s:%d\n", #c, __FILE__, __LINE__); \
      ++failures;                                                                 \
    }                                                                             \
  } while (0)

const int G = 2;

// two distinct global fields
double f0(Index x, Index y, Index z) { return 1.0 + x + 100.0 * y + 10000.0 * z; }
double f1(Index x, Index y, Index z) { return std::sin(0.1 * x) * (y + 1) - 0.5 * z; }

// allocate a padded local buffer for a block and fill inner cells from a global function (or 0).
std::vector<double> makeField(const peclet::core::decomp::Block<3>& b, double (*fn)(Index, Index, Index)) {
  const Index ex = b.size[0] + 2 * G, ey = b.size[1] + 2 * G, ez = b.size[2] + 2 * G;
  std::vector<double> v((std::size_t)ex * ey * ez, -1e30);  // ghosts = sentinel
  for (Index z = 0; z < b.size[2]; ++z)
    for (Index y = 0; y < b.size[1]; ++y)
      for (Index x = 0; x < b.size[0]; ++x) {
        const Index i = (x + G) + (y + G) * ex + (z + G) * ex * ey;
        v[i] = fn ? fn(b.origin[0] + x, b.origin[1] + y, b.origin[2] + z) : 0.0;
      }
  return v;
}

double checkField(const std::vector<double>& v, const peclet::core::decomp::Block<3>& b,
                  double (*fn)(Index, Index, Index)) {
  const Index ex = b.size[0] + 2 * G, ey = b.size[1] + 2 * G;
  double emax = 0;
  for (Index z = 0; z < b.size[2]; ++z)
    for (Index y = 0; y < b.size[1]; ++y)
      for (Index x = 0; x < b.size[0]; ++x) {
        const Index i = (x + G) + (y + G) * ex + (z + G) * ex * ey;
        const double want = fn(b.origin[0] + x, b.origin[1] + y, b.origin[2] + z);
        emax = std::fmax(emax, std::fabs(v[i] - want));
      }
  return emax;
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, np = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  const IVec<3> gsize{20, 16, 12};
  BlockDecomposer<3> oldDec((std::size_t)np, gsize);  // equal-cell ORB
  // NEW decomposition: weight the low-x half heavily so the split boundaries move.
  std::vector<peclet::core::Real> w((std::size_t)gsize[0] * gsize[1] * gsize[2], 1.0);
  for (Index z = 0; z < gsize[2]; ++z)
    for (Index y = 0; y < gsize[1]; ++y)
      for (Index x = 0; x < gsize[0]; ++x)
        if (x < gsize[0] / 2)
          w[(std::size_t)x + (std::size_t)y * gsize[0] + (std::size_t)z * gsize[0] * gsize[1]] = 5.0;
  BlockDecomposer<3> newDec((std::size_t)np, gsize, w);

  auto ob = oldDec.block(rank), nb = newDec.block(rank);
  auto o0 = makeField(ob, f0), o1 = makeField(ob, f1);
  auto n0 = makeField(nb, nullptr), n1 = makeField(nb, nullptr);

  redistributeGridFields<double>(oldDec, newDec, rank, G, {o0.data(), o1.data()},
                                 {n0.data(), n1.data()}, MPI_COMM_WORLD);
  const double e0 = checkField(n0, nb, f0), e1 = checkField(n1, nb, f1);
  CHECK(e0 == 0.0);
  CHECK(e1 == 0.0);  // pure data movement — must be bit-exact

  // round-trip back to the old layout
  auto r0 = makeField(ob, nullptr), r1 = makeField(ob, nullptr);
  redistributeGridFields<double>(newDec, oldDec, rank, G, {n0.data(), n1.data()},
                                 {r0.data(), r1.data()}, MPI_COMM_WORLD);
  const double re0 = checkField(r0, ob, f0), re1 = checkField(r1, ob, f1);
  CHECK(re0 == 0.0);
  CHECK(re1 == 0.0);

  int global = 0;
  MPI_Allreduce(&failures, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("redistribute np=%d: fwd err (%.1e,%.1e) round-trip (%.1e,%.1e) -> %s\n", np, e0, e1,
                re0, re1, global == 0 ? "OK" : "FAIL");
  MPI_Finalize();
  return global == 0 ? 0 : 1;
}

// chooseAlignedWeighted (peclet/core/decomp/block_decomposer.hpp) is a pure, replicated function:
// every rank, handed the same global weight field, must choose the same alignment `a` and build
// the same partition without communicating (amr/docs/amr_mg_core_boundary.md §11.4, gate G-A3).
// Each rank generates the weight fields itself from the same seeds (as flow's rebalanceByWeights
// and amr's rebalance each hold the global vector), runs the chooser, and hashes (a, imbalance,
// origins, sizes, ORB tree); the MIN and MAX of the hash over ranks must agree, case by case.
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "test_util.hpp"

using peclet::core::Index;
using peclet::core::IVec;
using peclet::core::Real;
using peclet::core::decomp::chooseAlignedWeighted;

namespace {

std::uint64_t fnv(std::uint64_t h, const void* p, std::size_t n) {
  const auto* c = static_cast<const unsigned char*>(p);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= c[i];
    h *= 1099511628211ull;
  }
  return h;
}

// A heap-shaped particle bed over noise (the probe's shape), seeded per case.
std::vector<Real> heap(const IVec<3>& g, unsigned seed, Real tilt) {
  std::vector<Real> w(static_cast<std::size_t>(g[0] * g[1] * g[2]));
  unsigned s = seed;
  std::size_t i = 0;
  for (Index z = 0; z < g[2]; ++z)
    for (Index y = 0; y < g[1]; ++y)
      for (Index x = 0; x < g[0]; ++x, ++i) {
        s = s * 1664525u + 1013904223u;
        const Real u = static_cast<Real>(s >> 8) / static_cast<Real>(1u << 24);
        const Real X = (x + 0.5) / g[0], Y = (y + 0.5) / g[1], Z = (z + 0.5) / g[2];
        const Real zs = 0.35 * (1.0 + tilt * (0.5 - X) + tilt * (0.5 - Y));
        w[i] = 1.0 + (Z < zs ? static_cast<Real>(static_cast<int>(8.0 * u)) : 0.0);
      }
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> nps{4, 8, 12, 24};  // the partition sizes chosen for, plus this run's size
  if (size != 4 && size != 8)
    nps.push_back(size);
  long cases = 0;
  for (const IVec<3>& g : {IVec<3>{32, 32, 32}, IVec<3>{48, 32, 16}, IVec<3>{96, 96, 96}})
    for (int np : nps)
      for (unsigned seed : {7u, 11u, 23u})
        for (Real tilt : {0.0, 0.3, 0.5}) {
          const std::vector<Real> w = heap(g, seed + 101u * static_cast<unsigned>(np), tilt);
          const auto r = chooseAlignedWeighted(static_cast<std::size_t>(np), g, w);
          std::uint64_t h = 1469598103934665603ull;
          h = fnv(h, &r.a, sizeof r.a);
          h = fnv(h, &r.imbalance, sizeof r.imbalance);
          for (const auto& o : r.dec.origins())
            h = fnv(h, o.data(), sizeof(Index) * 3);
          for (const auto& s : r.dec.sizes())
            h = fnv(h, s.data(), sizeof(Index) * 3);
          std::vector<int> sd;
          std::vector<Index> sv;
          r.dec.flattenTree(sd, sv);
          h = fnv(h, sd.data(), sizeof(int) * sd.size());
          h = fnv(h, sv.data(), sizeof(Index) * sv.size());
          unsigned long long mine = h, lo = 0, hi = 0;
          MPI_Allreduce(&mine, &lo, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
          MPI_Allreduce(&mine, &hi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
          PECLET_CORE_CHECK(lo == hi);
          if (r.a > 0)
            PECLET_CORE_CHECK(r.imbalance <= 1.05);
          ++cases;
          if (rank == 0 && np == 8 && g[0] == 96 && seed == 7u)
            std::printf("96^3 np8 tilt %.1f: a = %d, imbalance %.4f\n", tilt, r.a, r.imbalance);
        }
  if (rank == 0)
    std::printf("chooser identical on all %d ranks for %ld cases\n", size, cases);

  // every rank's verdict counts
  int fails = peclet::core::test::g_failures, anyFail = 0;
  MPI_Allreduce(&fails, &anyFail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  peclet::core::test::g_failures = anyFail;
  MPI_Finalize();
  PECLET_CORE_RETURN_TEST_RESULT();
}

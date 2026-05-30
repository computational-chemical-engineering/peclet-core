// Halo-exchange microbenchmark: compares the NBX engine vs the persistent neighborhood-collective
// engine on a block-decomposed grid, and exercises compute/comm overlap.
//
// Usage: mpirun -np N bench_halo [cellsPerRankPerAxis] [ghost] [iters]
// Weak scaling: each rank owns ~cellsPerRankPerAxis^3 cells, so the global grid grows with N.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/grid_halo.hpp"

using namespace tpx;
using tpx::decomp::BlockDecomposer;
using tpx::halo::GridFieldView;
using tpx::halo::GridHalo;

static constexpr int kDim = 3;

static double now() { return MPI_Wtime(); }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Index per = (argc > 1) ? std::atoll(argv[1]) : 48;
  int ghost = (argc > 2) ? std::atoi(argv[2]) : 1;
  int iters = (argc > 3) ? std::atoi(argv[3]) : 200;

  // Weak scaling: total cells ~ size * per^3, laid out as a near-cubic global grid.
  double scale = std::cbrt(static_cast<double>(size));
  Index nx = static_cast<Index>(std::llround(per * scale));
  IVec<kDim> gsize{nx, nx, nx};
  std::array<bool, kDim> periodic{true, true, true};

  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);
  GridHalo<kDim> halo;
  double tBuild0 = now();
  halo.buildTopology(dec, rank, ghost, periodic, MPI_COMM_WORLD);
  double tBuild = now() - tBuild0;

  const auto& idx = halo.indexer();
  std::vector<double> a(idx.numCellsInclGhost(), 0.0);
  idx.forEachInner([&](const IVec<kDim>& lmd) {
    a[idx.localMdToLocal(lmd)] = static_cast<double>(rank);
  });
  GridFieldView<double> field{a.data()};

  auto bench = [&](const char* label, auto&& doExchange) {
    for (int w = 0; w < 5; ++w) doExchange();  // warmup
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = now();
    for (int it = 0; it < iters; ++it) doExchange();
    MPI_Barrier(MPI_COMM_WORLD);
    double t = (now() - t0) / iters;
    double tmax = 0.0;
    MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      std::printf("  %-26s %9.2f us/exchange\n", label, tmax * 1e6);
    }
  };

  if (rank == 0) {
    std::printf("# transport-core halo benchmark\n");
    std::printf("# ranks=%d  global=%lldx%lldx%lld  ghost=%d  iters=%d\n", size, (long long)gsize[0],
                (long long)gsize[1], (long long)gsize[2], ghost, iters);
    std::printf("# neighbors(rank0)=%zu  ghostRecv(rank0)=%zu  buildTopology=%.1f us\n",
                halo.numNeighbors(), halo.numGhostRecv(), tBuild * 1e6);
  }

  bench("NBX (blocking)", [&]() { halo.exchangeNbx(field); });
  bench("persistent neighbor", [&]() { halo.exchangePersistent(field); });
  bench("NBX start+wait (overlap)", [&]() {
    halo.start(field);
    // pretend-interior compute would go here, overlapping the in-flight messages
    halo.wait(field);
  });

  MPI_Finalize();
  return 0;
}

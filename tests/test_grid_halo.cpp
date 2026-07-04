// MPI correctness of the grid ghost-layer exchange.
//
// Rank r owns block r of an ORB decomposition. Each inner cell is initialised to a value unique to
// its GLOBAL coordinate (its global linear index). After a halo exchange, every ghost cell must
// hold the value of the (periodically wrapped) global cell it shadows; ghost cells outside a
// non-periodic boundary must remain untouched. We check this for both engines (NBX and persistent)
// and several periodicity settings, across whatever rank count ctest launches.
#include <mpi.h>

#include <array>
#include <cstdio>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::GridFieldView;
using peclet::core::halo::GridHaloTopology;

static constexpr int kDim = 3;
static constexpr double kSentinel = -1.0;

// Returns number of local mismatches (0 == good).
static int runCase(const BlockDecomposer<kDim>& dec, int rank, int ghost,
                   std::array<bool, kDim> periodic, bool persistent) {
  GridHaloTopology<kDim> halo;
  halo.buildTopology(dec, rank, ghost, periodic, MPI_COMM_WORLD);
  const auto& idx = halo.indexer();

  std::vector<double> a(idx.numCellsInclGhost(), kSentinel);
  idx.forEachInner([&](const IVec<kDim>& lmd) {
    IVec<kDim> g{};
    for (int i = 0; i < kDim; ++i)
      g[i] = lmd[i] + idx.originInclGhost()[i];
    a[idx.localMdToLocal(lmd)] = static_cast<double>(dec.linearGlobal(g));
  });

  GridFieldView<double> field{a.data()};
  if (persistent) {
    halo.exchangePersistent(field);
  } else {
    halo.exchangeNbx(field);
  }

  const IVec<kDim>& gsize = dec.globalSize();
  int fail = 0;
  idx.forEachAll([&](const IVec<kDim>& lmd) {
    if (idx.isInner(lmd))
      return;
    IVec<kDim> gw{};
    bool skip = false;
    for (int i = 0; i < kDim; ++i) {
      Index c = lmd[i] + idx.originInclGhost()[i];
      if (c < 0 || c >= gsize[i]) {
        if (periodic[i]) {
          c = wrap(c, gsize[i]);
        } else {
          skip = true;
          break;
        }
      }
      gw[i] = c;
    }
    double got = a[idx.localMdToLocal(lmd)];
    if (skip) {
      if (got != kSentinel)
        ++fail;  // boundary ghost must stay untouched
      return;
    }
    double expect = static_cast<double>(dec.linearGlobal(gw));
    if (got != expect)
      ++fail;
  });
  return fail;
}

// reverseAdd (transpose+accumulate): put 1.0 in every ghost cell that maps to a valid (wrapped)
// global cell and 0 in the inner cells; fold. Each ghost's 1.0 lands on exactly one owner, so the
// GLOBAL sum over inner cells must equal the GLOBAL count of valid ghost cells. Returns (localInner
// sum, localGhost count) via out-params; the caller allreduces + compares.
static void runReverseAdd(const BlockDecomposer<kDim>& dec, int rank, int ghost,
                          std::array<bool, kDim> periodic, double& innerSum, double& ghostCount) {
  GridHaloTopology<kDim> halo;
  halo.buildTopology(dec, rank, ghost, periodic, MPI_COMM_WORLD);
  const auto& idx = halo.indexer();
  const IVec<kDim>& gsize = dec.globalSize();
  std::vector<double> a(idx.numCellsInclGhost(), 0.0);
  ghostCount = 0.0;
  idx.forEachAll([&](const IVec<kDim>& lmd) {
    if (idx.isInner(lmd))
      return;
    for (int i = 0; i < kDim; ++i) {  // only ghosts shadowing a valid (wrapped) global cell
      Index c = lmd[i] + idx.originInclGhost()[i];
      if ((c < 0 || c >= gsize[i]) && !periodic[i])
        return;
    }
    a[idx.localMdToLocal(lmd)] = 1.0;
    ghostCount += 1.0;
  });
  GridFieldView<double> field{a.data()};
  halo.reverseAdd(field);
  innerSum = 0.0;
  idx.forEachInner([&](const IVec<kDim>& lmd) { innerSum += a[idx.localMdToLocal(lmd)]; });
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  IVec<kDim> gsize{16, 12, 10};
  int ghost = 1;
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);

  struct Case {
    std::array<bool, kDim> periodic;
    bool persistent;
    const char* name;
  };
  const Case cases[] = {
      {{true, true, true}, false, "periodic/NBX"},
      {{true, true, true}, true, "periodic/persistent"},
      {{false, false, false}, false, "open/NBX"},
      {{false, false, false}, true, "open/persistent"},
      {{true, false, true}, false, "mixed/NBX"},
      {{true, false, true}, true, "mixed/persistent"},
  };

  int totalFail = 0;
  for (const auto& c : cases) {
    int localFail = runCase(dec, rank, ghost, c.periodic, c.persistent);
    int globalFail = 0;
    MPI_Allreduce(&localFail, &globalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0 && globalFail != 0) {
      std::fprintf(stderr, "  case %-22s FAILED with %d mismatches\n", c.name, globalFail);
    }
    totalFail += globalFail;
  }

  // reverseAdd conservation: global inner sum == global ghost count, for each periodicity.
  for (auto per : {std::array<bool, kDim>{true, true, true},
                   std::array<bool, kDim>{true, false, true}}) {
    double li = 0, lg = 0, gi = 0, gg = 0;
    runReverseAdd(dec, rank, ghost, per, li, lg);
    MPI_Allreduce(&li, &gi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&lg, &gg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (gi != gg) {
      ++totalFail;
      if (rank == 0)
        std::fprintf(stderr, "  reverseAdd conservation FAILED: inner sum %.0f != ghost count %.0f\n",
                     gi, gg);
    }
  }

  if (rank == 0) {
    if (totalFail == 0) {
      std::printf("OK (np=%d): all %zu cases + reverseAdd passed\n", size,
                  sizeof(cases) / sizeof(cases[0]));
    } else {
      std::fprintf(stderr, "FAILED (np=%d): %d total mismatches\n", size, totalFail);
    }
  }
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

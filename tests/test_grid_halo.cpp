// MPI correctness of the grid ghost-layer exchange.
//
// Rank r owns block r of an ORB decomposition. Each inner cell is initialised to a value unique to
// its GLOBAL coordinate (its global linear index). After a halo exchange, every ghost cell must hold
// the value of the (periodically wrapped) global cell it shadows; ghost cells outside a non-periodic
// boundary must remain untouched. We check this for both engines (NBX and persistent) and several
// periodicity settings, across whatever rank count ctest launches.
#include <mpi.h>

#include <array>
#include <cstdio>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/grid_halo_topology.hpp"

using namespace tpx;
using tpx::decomp::BlockDecomposer;
using tpx::halo::GridFieldView;
using tpx::halo::GridHaloTopology;

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
    for (int i = 0; i < kDim; ++i) g[i] = lmd[i] + idx.originInclGhost()[i];
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
    if (idx.isInner(lmd)) return;
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
      if (got != kSentinel) ++fail;  // boundary ghost must stay untouched
      return;
    }
    double expect = static_cast<double>(dec.linearGlobal(gw));
    if (got != expect) ++fail;
  });
  return fail;
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

  if (rank == 0) {
    if (totalFail == 0) {
      std::printf("OK (np=%d): all %zu cases passed\n", size, sizeof(cases) / sizeof(cases[0]));
    } else {
      std::fprintf(stderr, "FAILED (np=%d): %d total mismatches\n", size, totalFail);
    }
  }
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

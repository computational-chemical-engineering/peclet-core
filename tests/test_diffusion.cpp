// End-to-end validation: a distributed explicit heat-diffusion solver built on GridHaloTopology, checked
// cell-for-cell against an independent serial reference.
//
// This is the real point of the halo layer: each step exchanges the ghost layer, then updates inner
// cells with a 2*Dim+1 point Laplacian stencil. If the decomposition, ownership, periodic wrap and
// async exchange are all correct, the distributed field must match the serial integration of the
// same periodic problem exactly (to round-off). We also check mass conservation. Runs with both halo
// engines.
#include <mpi.h>

#include <cmath>
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
static constexpr double kCoeff = 0.1;  // alpha*dt/dx^2; stability: kCoeff*2*Dim = 0.6 < 1
static constexpr int kSteps = 40;

static double initField(const BlockDecomposer<kDim>& dec, const IVec<kDim>& g) {
  std::uint64_t h = static_cast<std::uint64_t>(dec.linearGlobal(g)) * 2654435761u + 12345u;
  h ^= h >> 33;
  return static_cast<double>(h & 0xFFFFFFu) / static_cast<double>(0x1000000u);
}

// Serial periodic diffusion reference over the full global grid (x-fastest linear index).
static std::vector<double> serialReference(const BlockDecomposer<kDim>& dec) {
  const IVec<kDim>& n = dec.globalSize();
  Index total = 1;
  for (int i = 0; i < kDim; ++i) total *= n[i];
  std::vector<double> u(total), un(total);
  IVec<kDim> bgn{}, end{};
  for (int i = 0; i < kDim; ++i) end[i] = n[i];
  forEachInBox<kDim>(bgn, end, [&](const IVec<kDim>& g) { u[dec.linearGlobal(g)] = initField(dec, g); });

  for (int s = 0; s < kSteps; ++s) {
    forEachInBox<kDim>(bgn, end, [&](const IVec<kDim>& g) {
      double lap = 0.0;
      for (int i = 0; i < kDim; ++i) {
        IVec<kDim> gp = g, gm = g;
        gp[i] = wrap(g[i] + 1, n[i]);
        gm[i] = wrap(g[i] - 1, n[i]);
        lap += u[dec.linearGlobal(gp)] + u[dec.linearGlobal(gm)];
      }
      lap -= 2.0 * kDim * u[dec.linearGlobal(g)];
      un[dec.linearGlobal(g)] = u[dec.linearGlobal(g)] + kCoeff * lap;
    });
    std::swap(u, un);
  }
  return u;
}

static int runDistributed(const BlockDecomposer<kDim>& dec, int rank,
                          const std::vector<double>& ref, bool persistent) {
  const std::array<bool, kDim> periodic{true, true, true};
  GridHaloTopology<kDim> halo;
  halo.buildTopology(dec, rank, /*ghost=*/1, periodic, MPI_COMM_WORLD);
  const auto& idx = halo.indexer();

  std::vector<double> u(idx.numCellsInclGhost(), 0.0), un(idx.numCellsInclGhost(), 0.0);
  idx.forEachInner([&](const IVec<kDim>& lmd) {
    IVec<kDim> g{};
    for (int i = 0; i < kDim; ++i) g[i] = lmd[i] + idx.originInclGhost()[i];
    u[idx.localMdToLocal(lmd)] = initField(dec, g);
  });

  for (int s = 0; s < kSteps; ++s) {
    GridFieldView<double> fv{u.data()};
    if (persistent) {
      halo.exchangePersistent(fv);
    } else {
      halo.start(fv);
      halo.wait(fv);
    }
    idx.forEachInner([&](const IVec<kDim>& lmd) {
      Index c = idx.localMdToLocal(lmd);
      double lap = 0.0;
      IVec<kDim> e{};
      for (int i = 0; i < kDim; ++i) {
        IVec<kDim> lp = lmd, lm = lmd;
        lp[i] += 1;
        lm[i] -= 1;
        lap += u[idx.localMdToLocal(lp)] + u[idx.localMdToLocal(lm)];
      }
      lap -= 2.0 * kDim * u[c];
      un[c] = u[c] + kCoeff * lap;
    });
    std::swap(u, un);
  }

  // Compare inner cells to the serial reference; check local mass too.
  int fail = 0;
  double localSum = 0.0;
  idx.forEachInner([&](const IVec<kDim>& lmd) {
    IVec<kDim> g{};
    for (int i = 0; i < kDim; ++i) g[i] = lmd[i] + idx.originInclGhost()[i];
    double got = u[idx.localMdToLocal(lmd)];
    double exp = ref[dec.linearGlobal(g)];
    if (std::fabs(got - exp) > 1e-9) ++fail;
    localSum += got;
  });

  double globalSum = 0.0;
  MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // Mass conservation: distributed sum must equal the serial field's sum.
  double refSum = 0.0;
  for (double v : ref) refSum += v;
  if (rank == 0 && std::fabs(globalSum - refSum) > 1e-6) {
    std::fprintf(stderr, "  mass mismatch: dist=%.9f ref=%.9f\n", globalSum, refSum);
    ++fail;
  }
  return fail;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  IVec<kDim> gsize{32, 24, 20};
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);
  std::vector<double> ref = serialReference(dec);

  int fail = 0;
  for (bool persistent : {false, true}) {
    int localFail = runDistributed(dec, rank, ref, persistent);
    int globalFail = 0;
    MPI_Allreduce(&localFail, &globalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0 && globalFail) {
      std::fprintf(stderr, "  engine=%s FAILED (%d)\n", persistent ? "persistent" : "nbx",
                   globalFail);
    }
    fail += globalFail;
  }

  if (rank == 0) {
    if (fail == 0)
      std::printf("OK (np=%d): distributed diffusion matches serial over %d steps (both engines)\n",
                  size, kSteps);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d\n", size, fail);
  }
  MPI_Finalize();
  return fail == 0 ? 0 : 1;
}

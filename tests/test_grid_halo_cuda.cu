// MPI + CUDA correctness of the GPU-resident ghost-layer exchange.
//
// Build the halo topology on the host, then run the SAME logical exchange two ways: on the CPU
// (GridHalo::exchangeNbx) and on the GPU (DeviceGridExchange, host-staged). The device result must
// match the CPU result bit-for-bit, and both must equal the analytic global field at every ghost
// cell. Validates pack/unpack/self-copy kernels + the staging path across ranks (all sharing one GPU).
#include <mpi.h>

#include <array>
#include <cstdio>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/grid_halo.hpp"
#include "tpx/halo/grid_halo_cuda.cuh"

using namespace tpx;
using tpx::decomp::BlockDecomposer;
using tpx::halo::DeviceGridExchange;
using tpx::halo::GridFieldView;
using tpx::halo::GridHalo;

static constexpr int kDim = 3;
static constexpr double kSentinel = -1.0;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  cudaSetDevice(0);  // all ranks share the single GPU

  IVec<kDim> gsize{24, 20, 16};
  std::array<bool, kDim> periodic{true, true, true};
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);

  GridHalo<kDim> halo;
  halo.buildTopology(dec, rank, /*ghost=*/1, periodic, MPI_COMM_WORLD);
  const auto& idx = halo.indexer();
  const Index n = idx.numCellsInclGhost();

  // Initial field: inner = analytic global value, ghost = sentinel.
  std::vector<double> a0(n, kSentinel);
  idx.forEachInner([&](const IVec<kDim>& lmd) {
    IVec<kDim> g{};
    for (int i = 0; i < kDim; ++i) g[i] = lmd[i] + idx.originInclGhost()[i];
    a0[idx.localMdToLocal(lmd)] = static_cast<double>(dec.linearGlobal(g));
  });

  // CPU reference exchange.
  std::vector<double> aCpu = a0;
  GridFieldView<double> cpuField{aCpu.data()};
  halo.exchangeNbx(cpuField);

  // GPU exchange.
  double* dField = nullptr;
  cudaMalloc(&dField, n * sizeof(double));
  cudaMemcpy(dField, a0.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  DeviceGridExchange<double> dev;
  dev.init(halo);
  dev.exchange(dField);
  std::vector<double> aGpu(n);
  cudaMemcpy(aGpu.data(), dField, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dField);

  // Compare: GPU == CPU everywhere, and ghosts == analytic.
  int fail = 0;
  for (Index c = 0; c < n; ++c) {
    if (aGpu[c] != aCpu[c]) ++fail;
  }
  idx.forEachAll([&](const IVec<kDim>& lmd) {
    if (idx.isInner(lmd)) return;
    IVec<kDim> gw{};
    bool skip = false;
    for (int i = 0; i < kDim; ++i) {
      Index ci = lmd[i] + idx.originInclGhost()[i];
      if (ci < 0 || ci >= gsize[i]) {
        if (periodic[i]) {
          ci = wrap(ci, gsize[i]);
        } else {
          skip = true;
          break;
        }
      }
      gw[i] = ci;
    }
    if (skip) return;
    double expect = static_cast<double>(dec.linearGlobal(gw));
    if (aGpu[idx.localMdToLocal(lmd)] != expect) ++fail;
  });

  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): GPU halo exchange matches CPU and analytic field\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d mismatches\n", size, totalFail);
  }
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

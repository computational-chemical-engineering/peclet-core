// MPI + Kokkos correctness of the portable GPU-resident ghost-layer exchange.
//
// Build the halo topology on the host, then run the SAME logical exchange two ways: on the CPU
// (GridHaloTopology::exchangeNbx) and on the device via Kokkos (GridHalo). The device result
// must match the CPU result bit-for-bit, and both must equal the analytic global field at every
// ghost cell. Validates the pack/unpack/self-copy parallel_for kernels + the host-staging path across
// ranks (all sharing one device). Mirrors test_grid_halo_cuda.cu but on the Kokkos backend.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <array>
#include <cstdio>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"
#include "peclet/core/halo/grid_halo.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::GridHalo;
using peclet::core::halo::GridFieldView;
using peclet::core::halo::GridHaloTopology;

static constexpr int kDim = 3;
static constexpr double kSentinel = -1.0;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  int size = 1;
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    IVec<kDim> gsize{24, 20, 16};
    std::array<bool, kDim> periodic{true, true, true};
    BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);

    GridHaloTopology<kDim> halo;
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

    // Kokkos device exchange.
    View<double> dField(Kokkos::view_alloc("field", Kokkos::WithoutInitializing),
                        static_cast<std::size_t>(n));
    auto hField = Kokkos::create_mirror_view(dField);
    for (Index c = 0; c < n; ++c) hField(c) = a0[c];
    Kokkos::deep_copy(dField, hField);

    GridHalo<double> dev;
    dev.init(halo);
    dev.exchange(dField);

    Kokkos::deep_copy(hField, dField);

    // Compare: device == CPU everywhere, and ghosts == analytic.
    for (Index c = 0; c < n; ++c)
      if (hField(c) != aCpu[c]) ++fail;
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
      if (hField(idx.localMdToLocal(lmd)) != expect) ++fail;
    });
  }

  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): Kokkos halo exchange matches CPU and analytic field\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d mismatches\n", size, totalFail);
  }

  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

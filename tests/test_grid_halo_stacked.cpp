// GridHalo stacked exchange and fill depth (flow doc/vof_sharp_viscous_stress.md §7: the per-step
// exchange of the sharp interface's four fit fields, tau_hat x 3 and D_nn, as ONE exchange).
//
// For every case below, on a FULL topology and on one built with depth 1 (ghost width 2): K = 4
// fields stacked at offsets j * stride of one view (stride = cells incl. ghosts + 3 padding
// cells), inner cells of copy j = a hash of (global cell, j, exchange number), every other cell a
// sentinel. After GridHalo::init(topo, K, stride) and one exchange (tag 0: persistent requests or
// the on-node shared-memory transport; tag 12: Isend/Irecv), each ghost cell of each copy within
// the fill depth on every axis and with an owner must hold exactly (bitwise) its owner's value of
// THAT copy; every ghost cell beyond the depth, without an owner, and every padding cell must still
// hold the sentinel. The single-field (K = 1) exchange on the same topology is checked the same
// way.
#include <mpi.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::GridHalo;
using peclet::core::halo::GridHaloTopology;
using peclet::core::halo::Neighborhood;

namespace {
constexpr int kDim = 3;
constexpr int kIters = 20;
constexpr double kSentinel = -7.0;

KOKKOS_INLINE_FUNCTION double hashValue(Index g, int j, int it) {
  std::uint64_t x = static_cast<std::uint64_t>(g) * 0x9E3779B97F4A7C15ull ^
                    (static_cast<std::uint64_t>(it * 8 + j) + 1) * 0xBF58476D1CE4E5B9ull;
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
}

struct Case {
  const char* name;
  IVec<kDim> gsize;
  std::array<bool, kDim> periodic;
};

int runCase(const Case& c, int depth, int K, int rank, int size) {
  constexpr int ghost = 2;
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), c.gsize);
  GridHaloTopology<kDim> topo;
  topo.buildTopology(dec, rank, ghost, c.periodic, MPI_COMM_WORLD, Neighborhood::Full, depth);
  const auto& idx = topo.indexer();
  const Index n = idx.numCellsInclGhost();
  const Index stride = n + 3;
  const int fill = depth < 0 ? ghost : depth;

  // Per local cell: 1 inner, 0 ghost to be filled, -1 left alone; its global id.
  std::vector<Index> gidH(static_cast<std::size_t>(n), -1);
  std::vector<int> kindH(static_cast<std::size_t>(n), -1);
  idx.forEachAll([&](const IVec<kDim>& lmd) {
    const Index l = idx.localMdToLocal(lmd);
    IVec<kDim> g{};
    for (int a = 0; a < kDim; ++a) {
      const Index hi = ghost + idx.sizeInner()[a];
      const Index d = lmd[a] < ghost ? ghost - lmd[a] : (lmd[a] >= hi ? lmd[a] - hi + 1 : 0);
      if (d > fill)
        return;
      Index ci = lmd[a] + idx.originInclGhost()[a];
      if (ci < 0 || ci >= c.gsize[a]) {
        if (!c.periodic[a])
          return;
        ci = wrap(ci, c.gsize[a]);
      }
      g[a] = ci;
    }
    gidH[l] = dec.linearGlobal(g);
    kindH[l] = idx.isInner(lmd) ? 1 : 0;
  });
  Kokkos::View<Index*, MemSpace> gid("gid", n);
  Kokkos::View<int*, MemSpace> kind("kind", n);
  {
    auto hg = Kokkos::create_mirror_view(gid);
    auto hk = Kokkos::create_mirror_view(kind);
    for (Index l = 0; l < n; ++l) {
      hg(l) = gidH[l];
      hk(l) = kindH[l];
    }
    Kokkos::deep_copy(gid, hg);
    Kokkos::deep_copy(kind, hk);
  }

  GridHalo<double> halo;
  halo.setLabel(c.name);
  halo.init(topo, K, stride);
  View<double> f(Kokkos::view_alloc("stacked", Kokkos::WithoutInitializing),
                 static_cast<std::size_t>(K * stride));
  const Index total = K * stride;
  int bad = 0;
  for (int variant = 0; variant < 2; ++variant) {
    const int tag = variant == 0 ? 0 : 12;
    for (int it = 0; it < kIters; ++it) {
      const int seq = variant * kIters + it;
      Kokkos::parallel_for(
          "fill", Kokkos::RangePolicy<ExecSpace>(0, total), KOKKOS_LAMBDA(const Index m) {
            const Index j = m / stride, l = m % stride;
            f(m) = (l < n && kind(l) == 1) ? hashValue(gid(l), (int)j, seq) : kSentinel;
          });
      Kokkos::fence();
      halo.exchange(f, tag);
      int nb = 0;
      Kokkos::parallel_reduce(
          "check", Kokkos::RangePolicy<ExecSpace>(0, total),
          KOKKOS_LAMBDA(const Index m, int& b) {
            const Index j = m / stride, l = m % stride;
            const double want =
                (l >= n || kind(l) == -1) ? kSentinel : hashValue(gid(l), (int)j, seq);
            if (Kokkos::bit_cast<std::uint64_t>(f(m)) != Kokkos::bit_cast<std::uint64_t>(want))
              b += 1;
          },
          nb);
      bad += nb;
    }
  }
  return bad;
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int totalFail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const Case cases[] = {
        {"periodic", {24, 20, 16}, {true, true, true}},
        {"irregular_wall", {37, 29, 23}, {true, false, true}},
    };
    for (const auto& c : cases)
      for (int depth : {-1, 1})
        for (int K : {1, 4}) {
          const int bad = runCase(c, depth, K, rank, size);
          int badAll = 0;
          MPI_Allreduce(&bad, &badAll, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
          if (rank == 0)
            std::printf("%s %-15s depth %-2d K %d: %d mismatches\n", badAll ? "FAIL" : "ok  ",
                        c.name, depth, K, badAll);
          totalFail += badAll;
        }
    if (rank == 0)
      std::printf("%s (np=%d): GridHalo stacked exchange + fill depth\n",
                  totalFail ? "FAILED" : "OK", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

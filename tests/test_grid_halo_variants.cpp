// G-HALO (flow doc/vof_distributed_host_comm_design.md §10): the GridHalo transport variants under
// stress. For every decomposition below and every variant, 10^4 exchanges of a fresh random field
// (inner cells = a hash of (global cell, exchange number), ghosts reset to a sentinel), with a
// random per-rank delay of 0–50 µs before exchangeBegin and between begin and end (skew between
// neighbours, and the interior sweep of the overlap split). After every exchange each ghost cell
// with an owner must hold exactly (bitwise) its owner's value, and each ghost cell without one (a
// non-periodic face) must still hold the sentinel.
//
// Variants reached through the public API, so no test hook exists in the class:
//   tag 0      the persistent requests built at init (MPI_Startall / MPI_Waitall)
//   tag 12     the Isend/Irecv path (any tag != 0)
// and, on a host memory space, both copy paths: blocks whose gather/unpack hold at most
// kHaloSerialCopy elements run on the calling thread, the larger ones as one parallel_for.
// A hang is caught by the ctest TIMEOUT; the instance run with PECLET_CORE_HALO_TIMEOUT also drives
// the persistent requests through the test-and-poll diagnostic wait.
#include <mpi.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Core.hpp>
#include <random>
#include <type_traits>
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

namespace {
constexpr int kDim = 3;
constexpr int kExchanges = 10000;

KOKKOS_INLINE_FUNCTION double hashValue(Index g, int it) {
  std::uint64_t x = static_cast<std::uint64_t>(g) * 0x9E3779B97F4A7C15ull ^
                    (static_cast<std::uint64_t>(it) + 1) * 0xBF58476D1CE4E5B9ull;
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);  // [0, 1), 53 bits
}

struct Case {
  const char* name;
  IVec<kDim> gsize;
  std::array<bool, kDim> periodic;
  int ghost;
};

template <class T>
int runCase(const Case& c, int rank, int size, std::mt19937& rng, Index& nCopyOut) {
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), c.gsize);
  GridHaloTopology<kDim> topo;
  topo.buildTopology(dec, rank, c.ghost, c.periodic, MPI_COMM_WORLD);
  const auto& idx = topo.indexer();
  const Index n = idx.numCellsInclGhost();

  // Per local cell: 1 = inner, 0 = ghost with an owner, -1 = ghost without one; its global id.
  std::vector<Index> gidH(static_cast<std::size_t>(n), -1);
  std::vector<int> kindH(static_cast<std::size_t>(n), -1);
  idx.forEachAll([&](const IVec<kDim>& lmd) {
    const Index l = idx.localMdToLocal(lmd);
    IVec<kDim> g{};
    for (int a = 0; a < kDim; ++a) {
      Index ci = lmd[a] + idx.originInclGhost()[a];
      if (ci < 0 || ci >= c.gsize[a]) {
        if (!c.periodic[a])
          return;  // kind -1
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

  GridHalo<T> halo;
  halo.setLabel(c.name);
  halo.init(topo);
  {
    const auto t = topo.flatten();
    nCopyOut = static_cast<Index>(t.selfSrc.size() + t.sendIdx.size());
  }

  View<T> f(Kokkos::view_alloc("field", Kokkos::WithoutInitializing), static_cast<std::size_t>(n));
  std::uniform_int_distribution<int> delay(0, 50);
  int bad = 0;
  for (int variant = 0; variant < 2; ++variant) {
    const int tag = variant == 0 ? 0 : 12;
    for (int it = 0; it < kExchanges; ++it) {
      const int seq = variant * kExchanges + it;
      Kokkos::parallel_for(
          "fill", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const Index l) {
            f(l) = kind(l) == 1 ? static_cast<T>(hashValue(gid(l), seq)) : static_cast<T>(-1);
          });
      Kokkos::fence();
      usleep(static_cast<useconds_t>(delay(rng)));
      halo.exchangeBegin(f, tag);
      usleep(static_cast<useconds_t>(delay(rng)));
      halo.exchangeEnd(f);
      int nb = 0;
      Kokkos::parallel_reduce(
          "check", Kokkos::RangePolicy<ExecSpace>(0, n),
          KOKKOS_LAMBDA(const Index l, int& b) {
            const T want =
                kind(l) == -1 ? static_cast<T>(-1) : static_cast<T>(hashValue(gid(l), seq));
            // bitwise, not ==: a copy must move the exact bytes
            using U = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
            if (Kokkos::bit_cast<U>(f(l)) != Kokkos::bit_cast<U>(want))
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
    std::mt19937 rng(1234u + 7919u * static_cast<unsigned>(rank));
    // regular: 4x2x1 at np 8 (24^3 blocks, gather above kHaloSerialCopy); irregular: an odd grid
    // the ORB splits unevenly, one wall axis, ghost width 2; small: gathers below the cutoff.
    const Case cases[] = {
        {"regular", {96, 48, 24}, {true, true, true}, 1},
        {"irregular", {37, 29, 23}, {true, false, true}, 2},
        {"small", {12, 10, 8}, {true, true, true}, 1},
        {"small_g2_wall", {12, 10, 8}, {false, true, true}, 2},
    };
    for (const auto& c : cases) {
      for (int prec = 0; prec < 2; ++prec) {
        Index nCopy = 0;
        const int bad = prec == 0 ? runCase<double>(c, rank, size, rng, nCopy)
                                  : runCase<float>(c, rank, size, rng, nCopy);
        int badAll = 0;
        Index lo = 0, hi = 0;
        MPI_Allreduce(&bad, &badAll, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&nCopy, &lo, 1, MPI_INT64_T, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&nCopy, &hi, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0)
          std::printf(
              "%s %-14s %-6s gather %lld..%lld elements (serial cutoff %lld): %d mismatches\n",
              badAll ? "FAIL" : "ok  ", c.name, prec == 0 ? "double" : "float",
              static_cast<long long>(lo), static_cast<long long>(hi),
              static_cast<long long>(peclet::core::halo::kHaloSerialCopy), badAll);
        totalFail += badAll;
      }
    }
    if (rank == 0)
      std::printf("%s (np=%d): GridHalo variants, %d exchanges per variant and case\n",
                  totalFail ? "FAILED" : "OK", size, kExchanges);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

// MPI correctness of Lagrangian particle migration.
//
// N particles (ids 0..N-1) start scattered across ranks by id%size. Each carries its id as payload.
// Over several random-walk + migrate steps we require, globally and every step:
//   - particle count is conserved (== N),
//   - every particle resides on the rank that owns its position (ownerOf),
//   - the id multiset is exactly {0..N-1} (checked via SUM and XOR reductions — together these catch
//     any loss, duplication, or corruption).
#include <mpi.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_migrator.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::DomainMap;
using peclet::core::halo::ParticleMigrator;

static constexpr int kDim = 3;

// Cheap deterministic hash -> [0,1).
static double frac(std::uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<double>(x & 0xFFFFFFULL) / static_cast<double>(0x1000000ULL);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  IVec<kDim> gsize{20, 16, 12};
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);

  DomainMap<kDim> map;
  for (int i = 0; i < kDim; ++i) {
    map.origin[i] = 0.0;
    map.cellSize[i] = 1.0;
    map.periodic[i] = true;
  }
  ParticleMigrator<kDim> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);

  const std::int64_t N = 5000;
  std::vector<Vec<kDim>> pos;
  std::vector<char> payload;  // stride 8: int64 id
  const std::size_t stride = sizeof(std::int64_t);
  for (std::int64_t id = rank; id < N; id += size) {
    Vec<kDim> x{};
    for (int i = 0; i < kDim; ++i)
      x[i] = frac(static_cast<std::uint64_t>(id) * 3 + i) * gsize[i];
    pos.push_back(x);
    std::size_t off = payload.size();
    payload.resize(off + stride);
    std::memcpy(&payload[off], &id, stride);
  }

  // Expected invariants of the full id set.
  std::int64_t expectSum = N * (N - 1) / 2;
  std::int64_t expectXor = 0;
  for (std::int64_t i = 0; i < N; ++i) expectXor ^= i;

  int fail = 0;
  for (int step = 0; step < 6; ++step) {
    mig.migrate(pos, payload, stride);

    // (a) every local particle is owned by this rank; gather id sum/xor.
    std::int64_t localSum = 0, localXor = 0;
    std::int64_t localCount = static_cast<std::int64_t>(pos.size());
    for (std::size_t k = 0; k < pos.size(); ++k) {
      if (mig.ownerOf(pos[k]) != rank) ++fail;
      std::int64_t id;
      std::memcpy(&id, &payload[k * stride], stride);
      localSum += id;
      localXor ^= id;
    }

    std::int64_t gCount = 0, gSum = 0, gXor = 0;
    MPI_Allreduce(&localCount, &gCount, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localSum, &gSum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localXor, &gXor, 1, MPI_INT64_T, MPI_BXOR, MPI_COMM_WORLD);
    if (gCount != N || gSum != expectSum || gXor != expectXor) {
      if (rank == 0)
        std::fprintf(stderr, "  step %d FAILED: count=%lld sum=%lld xor=%lld\n", step,
                     (long long)gCount, (long long)gSum, (long long)gXor);
      ++fail;
    }

    // (b) random-walk move for the next step.
    for (std::size_t k = 0; k < pos.size(); ++k) {
      std::int64_t id;
      std::memcpy(&id, &payload[k * stride], stride);
      for (int i = 0; i < kDim; ++i) {
        double d = (frac(static_cast<std::uint64_t>(id) * 7 + i + step * 131) - 0.5) * 6.0;
        pos[k][i] += d;
      }
    }
  }

  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): %lld particles conserved & correctly placed over 6 steps\n", size,
                  (long long)N);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d\n", size, totalFail);
  }
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

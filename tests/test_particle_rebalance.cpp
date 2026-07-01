// MPI correctness of particle-count load re-balancing (peclet::core::halo::rebalanceByParticleCount).
//
// N particles are scattered with positions concentrated in one corner of the domain, so the
// equal-cell-count ORB leaves the corner-owning ranks heavily overloaded. After one rebalance —
// bin particle counts, weighted re-decompose, migrate — we require, globally:
//   - the particle count is conserved (== N) and the id multiset is exactly {0..N-1} (SUM + XOR),
//   - every particle resides on the rank that owns its position under the NEW decomposition,
//   - positions are unchanged (each id still carries its generated coordinates, bit-for-bit),
//   - the per-rank count imbalance (max/mean) drops vs the equal-cell decomposition.
// A second rebalance is a near no-op (already balanced). np = 1,2,4,8.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_migrator.hpp"
#include "peclet/core/halo/particle_rebalance.hpp"

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

// Position generated from a particle id, concentrated toward the origin corner (square biases the
// coordinate toward 0), so the particle density is strongly non-uniform.
static Vec<kDim> posOf(std::int64_t id, const IVec<kDim>& gsize) {
  Vec<kDim> x{};
  for (int i = 0; i < kDim; ++i) {
    double u = frac(static_cast<std::uint64_t>(id) * 3 + i);
    x[i] = u * u * gsize[i];
  }
  return x;
}

// max local particle count / mean — 1.0 is perfectly even.
static double imbalance(std::size_t localCount, std::int64_t N, int size, MPI_Comm comm) {
  long lc = static_cast<long>(localCount), mx = 0;
  MPI_Allreduce(&lc, &mx, 1, MPI_LONG, MPI_MAX, comm);
  return static_cast<double>(mx) / (static_cast<double>(N) / static_cast<double>(size));
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  IVec<kDim> gsize{16, 16, 16};
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);

  DomainMap<kDim> map;
  for (int i = 0; i < kDim; ++i) {
    map.origin[i] = 0.0;
    map.cellSize[i] = 1.0;
    map.periodic[i] = true;
  }
  ParticleMigrator<kDim> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);

  const std::int64_t N = 8000;
  const std::size_t stride = sizeof(std::int64_t);  // payload: int64 id
  std::vector<Vec<kDim>> pos;
  std::vector<char> payload;
  for (std::int64_t id = rank; id < N; id += size) {  // round-robin start, placed by migrate
    pos.push_back(posOf(id, gsize));
    std::size_t off = payload.size();
    payload.resize(off + stride);
    std::memcpy(&payload[off], &id, stride);
  }

  const std::int64_t expectSum = N * (N - 1) / 2;
  std::int64_t expectXor = 0;
  for (std::int64_t i = 0; i < N; ++i) expectXor ^= i;

  int fail = 0;
  auto checkInvariants = [&](const char* tag) {
    std::int64_t localSum = 0, localXor = 0, localCount = static_cast<std::int64_t>(pos.size());
    for (std::size_t k = 0; k < pos.size(); ++k) {
      if (mig.ownerOf(pos[k]) != rank) ++fail;  // placed on its new owner
      std::int64_t id;
      std::memcpy(&id, &payload[k * stride], stride);
      Vec<kDim> want = posOf(id, gsize);  // position unchanged, bit-for-bit
      for (int i = 0; i < kDim; ++i)
        if (pos[k][i] != want[i]) ++fail;
      localSum += id;
      localXor ^= id;
    }
    std::int64_t gCount = 0, gSum = 0, gXor = 0;
    MPI_Allreduce(&localCount, &gCount, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localSum, &gSum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localXor, &gXor, 1, MPI_INT64_T, MPI_BXOR, MPI_COMM_WORLD);
    if (gCount != N || gSum != expectSum || gXor != expectXor) {
      if (rank == 0)
        std::fprintf(stderr, "  %s FAILED: count=%lld sum=%lld xor=%lld\n", tag, (long long)gCount,
                     (long long)gSum, (long long)gXor);
      ++fail;
    }
  };

  // Place particles under the equal-cell ORB, then measure its (skewed) imbalance.
  mig.migrate(pos, payload, stride);
  checkInvariants("pre-rebalance");
  double imb0 = imbalance(pos.size(), N, size, MPI_COMM_WORLD);

  // Rebalance: weighted re-decompose by particle count + migrate.
  std::vector<double> weight;
  peclet::core::halo::rebalanceByParticleCount(dec, mig, pos, payload, stride, MPI_COMM_WORLD, &weight);
  checkInvariants("post-rebalance");
  double imb1 = imbalance(pos.size(), N, size, MPI_COMM_WORLD);

  // The agreed weight grid must account for every particle exactly once.
  double wsum = 0.0;
  for (double w : weight) wsum += w;
  if (wsum != static_cast<double>(N)) ++fail;

  if (size > 1) {
    if (!(imb0 > 1.25)) ++fail;       // equal-cell ORB really was imbalanced
    if (!(imb1 < imb0)) ++fail;       // weighted ORB improved it
    if (!(imb1 < 1.5)) ++fail;        // ... to a decently even distribution
  }

  // A second rebalance stays correct and does not worsen the balance.
  peclet::core::halo::rebalanceByParticleCount(dec, mig, pos, payload, stride, MPI_COMM_WORLD);
  checkInvariants("second-rebalance");
  double imb2 = imbalance(pos.size(), N, size, MPI_COMM_WORLD);
  if (size > 1 && !(imb2 < 1.5)) ++fail;

  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): %lld particles, imbalance %.2f -> %.2f -> %.2f\n", size, (long long)N,
                  imb0, imb1, imb2);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d\n", size, totalFail);
  }
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

// MPI correctness of Lagrangian ghost-particle gathering (ParticleMigrator::gatherGhosts).
//
// After migration each rank owns the particles in its block. gatherGhosts(rcut) must give each rank
// exactly the particles owned by OTHER ranks that lie within `rcut` of its block (periodic). We check
// this against a brute-force reference: every rank Allgathers all particles, then independently
// computes which ones should be its ghosts; the gathered id-set must equal the reference id-set, with
// no duplicates and no self-owned particles.
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_migrator.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::DomainMap;
using peclet::core::halo::ParticleMigrator;

static double frac(std::uint64_t x, int s) {
  x ^= (std::uint64_t)s * 2654435761u;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return (double)(x & 0xFFFFFF) / (double)0x1000000;
}

struct GP {
  double p[3];
  std::int64_t id;
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const double dmin[3] = {0, 0, 0}, dsize[3] = {10, 8, 6};
  IVec<3> gsize{40, 32, 24};
  double rcut = 0.9;
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), gsize);
  DomainMap<3> map;
  for (int i = 0; i < 3; ++i) {
    map.origin[i] = dmin[i];
    map.cellSize[i] = dsize[i] / gsize[i];
    map.periodic[i] = true;
  }
  ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);

  const std::int64_t N = 3000;
  const std::size_t stride = sizeof(std::int64_t);
  std::vector<Vec<3>> pos;
  std::vector<char> payload;
  for (std::int64_t id = rank; id < N; id += size) {
    Vec<3> x{dmin[0] + frac(id, 0) * dsize[0], dmin[1] + frac(id, 1) * dsize[1],
             dmin[2] + frac(id, 2) * dsize[2]};
    pos.push_back(x);
    std::size_t off = payload.size();
    payload.resize(off + stride);
    std::memcpy(&payload[off], &id, stride);
  }
  mig.migrate(pos, payload, stride);  // each rank now owns its particles

  // --- gather ghosts ---
  std::vector<Vec<3>> gpos;
  std::vector<char> gpay;
  mig.gatherGhosts(pos, payload, stride, rcut, gpos, gpay);
  std::vector<std::int64_t> gotIds(gpos.size());
  for (std::size_t i = 0; i < gpos.size(); ++i) std::memcpy(&gotIds[i], &gpay[i * stride], stride);

  // --- brute-force reference: Allgather all owned particles ---
  int local = (int)pos.size();
  std::vector<int> counts(size), displs(size);
  MPI_Allgather(&local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int r = 0; r < size; ++r) {
    displs[r] = tot;
    tot += counts[r];
  }
  std::vector<GP> mine(local), all(tot);
  for (int i = 0; i < local; ++i) {
    mine[i].p[0] = pos[i][0];
    mine[i].p[1] = pos[i][1];
    mine[i].p[2] = pos[i][2];
    std::memcpy(&mine[i].id, &payload[i * stride], stride);
  }
  std::vector<int> cb(size), db(size);
  for (int r = 0; r < size; ++r) {
    cb[r] = counts[r] * (int)sizeof(GP);
    db[r] = displs[r] * (int)sizeof(GP);
  }
  MPI_Allgatherv(mine.data(), local * (int)sizeof(GP), MPI_BYTE, all.data(), cb.data(), db.data(),
                 MPI_BYTE, MPI_COMM_WORLD);

  std::vector<std::int64_t> expectIds;
  for (const auto& g : all) {
    Vec<3> x{g.p[0], g.p[1], g.p[2]};
    if (mig.ownerOf(x) == rank) continue;  // owned here -> not a ghost
    Vec<3> img;
    if (mig.withinRcutOfBlock(x, rank, rcut, img)) expectIds.push_back(g.id);
  }

  std::sort(gotIds.begin(), gotIds.end());
  std::sort(expectIds.begin(), expectIds.end());
  int fail = 0;
  if (gotIds.size() != expectIds.size())
    ++fail;
  else
    for (std::size_t i = 0; i < gotIds.size(); ++i)
      if (gotIds[i] != expectIds[i]) {
        ++fail;
        break;
      }
  // no duplicates among gathered ghosts
  for (std::size_t i = 1; i < gotIds.size(); ++i)
    if (gotIds[i] == gotIds[i - 1]) ++fail;

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  long long gtot = 0, etot = 0, lg = (long long)gotIds.size(), le = (long long)expectIds.size();
  MPI_Reduce(&lg, &gtot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&le, &etot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("# ghosts: gathered=%lld expected=%lld (rcut=%.2f)\n", gtot, etot, rcut);
    if (total == 0)
      std::printf("OK (np=%d): ghost particles match brute-force reference exactly\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d rank(s) mismatched\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}

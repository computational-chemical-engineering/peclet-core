// MPI correctness of the persistent ParticleHaloTopology (forward / forwardPositions / reverse-accumulate).
//
// After migration each rank owns its block's particles and builds the halo. We check against a
// brute-force Allgather reference:
//   forward(id)        -> each ghost carries its owner's id (ghost id-set == expected),
//   forwardPositions   -> each ghost sits at the correct periodic image of its owner,
//   reverse(ones, sum) -> each owned particle accumulates a count == how many ranks hold it as a
//                         ghost (the key new primitive: ghost contributions summed onto owners).
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_halo_topology.hpp"
#include "peclet/core/halo/particle_migrator.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::DomainMap;
using peclet::core::halo::ParticleHaloTopology;
using peclet::core::halo::ParticleMigrator;

struct GP {
  double p[3];
  std::int64_t id;
};
static double frac(std::uint64_t x, int s) {
  x ^= (std::uint64_t)s * 2654435761u;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return (double)(x & 0xFFFFFF) / (double)0x1000000;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const double dmin[3] = {0, 0, 0}, dsize[3] = {10, 8, 6};
  IVec<3> gsize{40, 32, 24};
  const double rcut = 0.9;
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
    pos.push_back({dmin[0] + frac(id, 0) * dsize[0], dmin[1] + frac(id, 1) * dsize[1],
                   dmin[2] + frac(id, 2) * dsize[2]});
    std::size_t off = payload.size();
    payload.resize(off + stride);
    std::memcpy(&payload[off], &id, stride);
  }
  mig.migrate(pos, payload, stride);
  std::size_t Nown = pos.size();
  std::vector<std::int64_t> myid(Nown);
  for (std::size_t i = 0; i < Nown; ++i) std::memcpy(&myid[i], &payload[i * stride], stride);

  // --- build halo ---
  ParticleHaloTopology<3> halo;
  halo.init(mig);
  halo.build(pos, rcut);
  std::size_t G = halo.numGhost();

  // forward(id): each ghost gets its owner's id
  std::vector<double> ownIdD(Nown), ghIdD(G);
  for (std::size_t i = 0; i < Nown; ++i) ownIdD[i] = (double)myid[i];
  halo.forward(ownIdD.data(), ghIdD.data());

  // reverse(ones): each owned accumulates a count of its ghost copies
  std::vector<double> ones(G, 1.0), ownCount(Nown, 0.0);
  halo.reverse(ones.data(), ownCount.data());

  // --- brute-force reference: Allgather all owned particles ---
  int local = (int)Nown;
  std::vector<int> counts(size), displs(size);
  MPI_Allgather(&local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int r = 0; r < size; ++r) {
    displs[r] = tot;
    tot += counts[r];
  }
  std::vector<GP> mine(local), all(tot);
  for (int i = 0; i < local; ++i) {
    mine[i] = {{pos[i][0], pos[i][1], pos[i][2]}, myid[i]};
  }
  std::vector<int> cb(size), db(size);
  for (int r = 0; r < size; ++r) {
    cb[r] = counts[r] * (int)sizeof(GP);
    db[r] = displs[r] * (int)sizeof(GP);
  }
  MPI_Allgatherv(mine.data(), local * (int)sizeof(GP), MPI_BYTE, all.data(), cb.data(), db.data(),
                 MPI_BYTE, MPI_COMM_WORLD);

  int fail = 0;

  // (1) forward id-set + (2) forwardPositions image match
  std::vector<std::int64_t> expectIds;
  for (const auto& g : all) {
    Vec<3> x{g.p[0], g.p[1], g.p[2]}, img;
    if (mig.ownerOf(x) == rank) continue;
    if (mig.withinRcutOfBlock(x, rank, rcut, img)) expectIds.push_back(g.id);
  }
  std::vector<std::int64_t> gotIds(G);
  for (std::size_t s = 0; s < G; ++s) gotIds[s] = (std::int64_t)std::llround(ghIdD[s]);
  {
    auto a = gotIds, b = expectIds;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) ++fail;
  }
  // each ghost sits at the correct periodic image of its (id'd) owner
  const auto& gp = halo.ghostPositions();
  for (std::size_t s = 0; s < G; ++s) {
    std::int64_t gid = gotIds[s];
    const GP* src = nullptr;
    for (const auto& g : all)
      if (g.id == gid) {
        src = &g;
        break;
      }
    Vec<3> x{src->p[0], src->p[1], src->p[2]}, img;
    mig.withinRcutOfBlock(x, rank, rcut, img);
    for (int d = 0; d < 3; ++d)
      if (std::fabs(gp[s][d] - img[d]) > 1e-9) {
        ++fail;
        break;
      }
  }

  // (3) reverse count == number of ranks holding each owned particle as a ghost
  for (std::size_t i = 0; i < Nown; ++i) {
    Vec<3> img;
    int expect = 0;
    for (int r = 0; r < size; ++r)
      if (r != rank && mig.withinRcutOfBlock(pos[i], r, rcut, img)) ++expect;
    if ((int)std::llround(ownCount[i]) != expect) ++fail;
  }

  // (4) periodic self-ghosts (includePeriodicSelf): the cross-rank ghosts above + LOCAL copies of an
  // owned particle at its own periodic image(s) within rcut of this rank's block (needed for an
  // undecomposed periodic axis / np=1). Brute-force oracle: a self-ghost exists for each non-identity
  // periodic image (±L per axis) of an owned particle that comes within rcut of this rank's own block;
  // forwardPositions must place it at owner+shift, and forward(id) must carry the owner's id.
  ParticleHaloTopology<3> halo2;
  halo2.init(mig);
  halo2.build(pos, rcut, /*includePeriodicSelf=*/true);
  const std::size_t G2 = halo2.numGhost();
  std::vector<double> ghId2(G2);
  halo2.forward(ownIdD.data(), ghId2.data());
  const auto& gp2 = halo2.ghostPositions();
  const auto& bo = dec.origins()[rank];
  const auto& bs = dec.sizes()[rank];
  int selfExpect = 0, selfMatched = 0;
  for (std::size_t i = 0; i < Nown; ++i) {
    for (int sx = -1; sx <= 1; ++sx)
      for (int sy = -1; sy <= 1; ++sy)
        for (int sz = -1; sz <= 1; ++sz) {
          if (sx == 0 && sy == 0 && sz == 0) continue;
          const int sc[3] = {sx, sy, sz};
          double d2 = 0;
          Vec<3> imgpos;
          for (int d = 0; d < 3; ++d) {
            if (sc[d] != 0 && !map.periodic[d]) { d2 = 1e30; break; }
            const double L = map.cellSize[d] * gsize[d];
            const double lo = map.origin[d] + bo[d] * map.cellSize[d];
            const double hi = map.origin[d] + (bo[d] + bs[d]) * map.cellSize[d];
            const double p = pos[i][d] + sc[d] * L;
            imgpos[d] = p;
            const double gap = (p < lo) ? (lo - p) : (p > hi) ? (p - hi) : 0.0;
            d2 += gap * gap;
          }
          if (d2 >= rcut * rcut) continue;
          ++selfExpect;
          // find a self-ghost (id == owner id, position == imgpos) in the tail of halo2's ghost list.
          for (std::size_t s = 0; s < G2; ++s) {
            if ((std::int64_t)std::llround(ghId2[s]) != myid[i]) continue;
            if (std::fabs(gp2[s][0] - imgpos[0]) < 1e-9 && std::fabs(gp2[s][1] - imgpos[1]) < 1e-9 &&
                std::fabs(gp2[s][2] - imgpos[2]) < 1e-9) { ++selfMatched; break; }
          }
        }
  }
  if (selfMatched != selfExpect) ++fail;
  // total ghosts must be the cross-rank set (G) plus exactly the self set.
  if (G2 != G + (std::size_t)selfExpect) ++fail;

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  long long gG = 0, lG = (long long)G;
  MPI_Reduce(&lG, &gG, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("# ParticleHaloTopology: ghosts=%lld (forward+forwardPositions+reverse checked)\n", gG);
    if (total == 0)
      std::printf("OK (np=%d): forward / forwardPositions / reverse(sum) match brute force\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}

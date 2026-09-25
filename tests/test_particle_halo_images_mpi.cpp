// ParticleHaloTopology::build(..., allImages = true): every periodic image within rcut is sent.
//
// The scene is a slab, periodic on every axis, whose z extent (3) is so thin that the ORB never
// splits z up to np = 8: z is an UNDECOMPOSED periodic axis at every rank count this test runs.
// There the one-image rule of the default build (the image nearest each destination block) always
// picks the unshifted image on z, because its z gap is 0, and drops the wrapped one although it
// lies inside rcut. A pair that crosses a rank face while wrapping z is then invisible on both of
// its owners (suite dem-contacts/docs/contact_evidence/FOLLOWUPS.md §4b).
//
// Brute-force oracle over an Allgather of every particle, enumerating all 3^3 image shifts by hand
// (no core image routine is used to build the expectation):
//   (1) image-count oracle: the received cross-rank ghosts, keyed (owner id, integer image), are
//       EXACTLY the images of other ranks' particles within rcut of this block -- no miss, no
//       duplicate -- and each owned particle's send entries count its images within rcut of every
//       other block;
//   (2) every allImages shift is exactly 0 or +-L per axis, and forwardPositions places each ghost
//       at owner + shift, bitwise;
//   (3) reverse sums EVERY image's contribution onto the owner (image-coded integer values, exact);
//   (4) at np >= 2 some particle does reach one rank at two images (the case the default drops),
//       and the default build misses exactly the oracle's surplus;
//   (5) the default (allImages off) send lists and shifts equal an independent reimplementation of
//       the one-image rule, byte for byte; at np = 1 the flag changes nothing at all.
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
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

namespace {

struct GP {
  double p[3];
  std::int64_t id;
};

double frac(std::uint64_t x, int s) {
  x ^= (std::uint64_t)s * 2654435761u;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return (double)(x & 0xFFFFFF) / (double)0x1000000;
}

using Key = std::tuple<std::int64_t, int, int, int>;  // (owner id, image sx, sy, sz)

// Integer image code in [0, 27): the value a ghost at that image contributes in the reverse test.
int imageCode(int sx, int sy, int sz) {
  return (sx + 1) + 3 * (sy + 1) + 9 * (sz + 1);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const double dsize[3] = {12, 12, 3};
  IVec<3> gsize{48, 48, 12};
  const double rcut = 1.0;
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), gsize);
  DomainMap<3> map;
  double L[3];
  for (int d = 0; d < 3; ++d) {
    map.origin[d] = 0.0;
    map.cellSize[d] = dsize[d] / gsize[d];
    map.periodic[d] = true;
    L[d] = map.cellSize[d] * static_cast<double>(gsize[d]);
  }
  ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);

  int fail = 0;
  auto check = [&](bool ok, const char* what) {
    if (!ok) {
      ++fail;
      std::fprintf(stderr, "rank %d: FAILED %s\n", rank, what);
    }
  };

  // The premise: z is undecomposed on every block.
  for (int r = 0; r < size; ++r)
    check(dec.origins()[r][2] == 0 && dec.sizes()[r][2] == gsize[2], "z undecomposed");

  const std::int64_t N = 4000;
  const std::size_t stride = sizeof(std::int64_t);
  std::vector<Vec<3>> pos;
  std::vector<char> payload;
  for (std::int64_t id = rank; id < N; id += size) {
    pos.push_back({frac(id, 0) * dsize[0], frac(id, 1) * dsize[1], frac(id, 2) * dsize[2]});
    std::size_t off = payload.size();
    payload.resize(off + stride);
    std::memcpy(&payload[off], &id, stride);
  }
  mig.migrate(pos, payload, stride);
  const std::size_t Nown = pos.size();
  std::vector<std::int64_t> myid(Nown);
  std::vector<double> ownIdD(Nown);
  for (std::size_t i = 0; i < Nown; ++i) {
    std::memcpy(&myid[i], &payload[i * stride], stride);
    ownIdD[i] = (double)myid[i];
  }

  // Allgather every particle for the brute-force oracle.
  int local = (int)Nown;
  std::vector<int> counts(size), displs(size);
  MPI_Allgather(&local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int r = 0; r < size; ++r) {
    displs[r] = tot;
    tot += counts[r];
  }
  std::vector<GP> mine(local), all(tot);
  for (int i = 0; i < local; ++i)
    mine[i] = {{pos[i][0], pos[i][1], pos[i][2]}, myid[i]};
  std::vector<int> cb(size), db(size);
  for (int r = 0; r < size; ++r) {
    cb[r] = counts[r] * (int)sizeof(GP);
    db[r] = displs[r] * (int)sizeof(GP);
  }
  MPI_Allgatherv(mine.data(), local * (int)sizeof(GP), MPI_BYTE, all.data(), cb.data(), db.data(),
                 MPI_BYTE, MPI_COMM_WORLD);
  std::vector<int> ownerOfAll(tot);
  for (int r = 0; r < size; ++r)
    for (int j = 0; j < counts[r]; ++j)
      ownerOfAll[displs[r] + j] = r;

  // Hand-written image enumeration against block r's AABB: calls f(sx, sy, sz, imagePos) for every
  // image of x within rcut.
  auto forEachImage = [&](const double x[3], int r, auto&& f) {
    const auto& o = dec.origins()[r];
    const auto& s = dec.sizes()[r];
    for (int sz = -1; sz <= 1; ++sz)
      for (int sy = -1; sy <= 1; ++sy)
        for (int sx = -1; sx <= 1; ++sx) {
          const int sc[3] = {sx, sy, sz};
          double p[3], d2 = 0.0;
          for (int d = 0; d < 3; ++d) {
            const double lo = map.origin[d] + o[d] * map.cellSize[d];
            const double hi = map.origin[d] + (o[d] + s[d]) * map.cellSize[d];
            p[d] = x[d] + sc[d] * L[d];
            const double gap = (p[d] < lo) ? (lo - p[d]) : (p[d] > hi) ? (p[d] - hi) : 0.0;
            d2 += gap * gap;
          }
          if (d2 < rcut * rcut)
            f(sx, sy, sz, p);
        }
  };

  // ---- allImages ON (with the self-ghosts, as the dem consumer builds it) ----
  ParticleHaloTopology<3> halo;
  halo.init(mig);
  halo.build(pos, rcut, /*includePeriodicSelf=*/true, /*allImages=*/true);
  const auto topo = halo.flatten();
  const std::size_t G = halo.numGhost();
  const std::size_t NR = static_cast<std::size_t>(topo.numReceived);
  std::vector<double> ghId(G);
  halo.forward(ownIdD.data(), ghId.data());

  // Per-ghost integer image from the shift; every shift component must be exactly 0 or +-L.
  std::vector<std::array<int, 3>> ghImg(G);
  for (std::size_t g = 0; g < G; ++g)
    for (int d = 0; d < 3; ++d) {
      const double sh = topo.shift[g][d];
      int k = (sh == 0.0) ? 0 : (sh == L[d]) ? 1 : (sh == -L[d]) ? -1 : 99;
      check(k != 99, "allImages shift is exactly 0 or +-L");
      ghImg[g][d] = k;
    }

  // (1a) received set == oracle set, as multisets keyed (id, image); no duplicate key.
  std::vector<Key> expect, got;
  for (int j = 0; j < tot; ++j) {
    if (ownerOfAll[j] == rank)
      continue;
    forEachImage(all[j].p, rank, [&](int sx, int sy, int sz, const double*) {
      expect.emplace_back(all[j].id, sx, sy, sz);
    });
  }
  for (std::size_t g = 0; g < NR; ++g)
    got.emplace_back((std::int64_t)std::llround(ghId[g]), ghImg[g][0], ghImg[g][1], ghImg[g][2]);
  std::sort(expect.begin(), expect.end());
  std::sort(got.begin(), got.end());
  check(got == expect, "received (id, image) set == oracle");
  check(std::adjacent_find(got.begin(), got.end()) == got.end(), "no duplicate (id, image)");
  check(NR == expect.size(), "received ghost count == oracle image count");

  // (1b) sender side: each owned particle's send entries == its images within rcut of every other
  // block; and (3) reverse of image-coded integers sums every image (self-ghosts included).
  std::vector<int> sendEntries(Nown, 0);
  for (Index i : topo.sendIdx)
    ++sendEntries[static_cast<std::size_t>(i)];
  std::vector<double> gval(G), acc(Nown, 0.0);
  for (std::size_t g = 0; g < G; ++g)
    gval[g] = 1.0 + imageCode(ghImg[g][0], ghImg[g][1], ghImg[g][2]);
  halo.reverse(gval.data(), acc.data());
  long long localEntries = 0, multiImage = 0;
  for (std::size_t i = 0; i < Nown; ++i) {
    const double x[3] = {pos[i][0], pos[i][1], pos[i][2]};
    int n = 0;
    double sum = 0.0;
    for (int r = 0; r < size; ++r) {
      int nr = 0;
      forEachImage(x, r, [&](int sx, int sy, int sz, const double*) {
        const bool identity = (sx == 0 && sy == 0 && sz == 0);
        if (r == rank && identity)
          return;  // a particle is never its own ghost
        if (r != rank)
          ++nr;
        sum += 1.0 + imageCode(sx, sy, sz);
      });
      n += nr;
      if (nr >= 2)
        ++multiImage;
    }
    check(sendEntries[i] == n, "send entries per particle == oracle image count");
    check(acc[i] == sum, "reverse sums every image's contribution");
    localEntries += sendEntries[i];
  }

  // (2) forwardPositions: owner + shift, bitwise, for received and self ghosts.
  std::vector<Vec<3>> gp(G);
  halo.forwardPositions(pos.data(), gp.data());
  const auto& gp0 = halo.ghostPositions();
  for (std::size_t g = 0; g < G; ++g) {
    const std::int64_t id = (std::int64_t)std::llround(ghId[g]);
    const GP* src = nullptr;
    for (const auto& a : all)
      if (a.id == id) {
        src = &a;
        break;
      }
    bool ok = src != nullptr;
    for (int d = 0; ok && d < 3; ++d) {
      const double want = src->p[d] + ghImg[g][d] * L[d];
      ok = (gp[g][d] == want) && (gp0[g][d] == want);
    }
    check(ok, "ghost position == owner + image shift (bitwise)");
  }

  // ---- allImages OFF (default) ----
  ParticleHaloTopology<3> off;
  off.init(mig);
  off.build(pos, rcut, /*includePeriodicSelf=*/true);
  const auto topoOff = off.flatten();

  // (5) independent one-image rule: per axis the candidate {x, x-L, x+L} with the strictly smallest
  // gap (first wins a tie), shift = image - x; ranks ascending, particles ascending.
  std::vector<int> refRanks;
  std::vector<Index> refIdx;
  std::vector<Vec<3>> refShift;
  for (int r = 0; r < size; ++r) {
    if (r == rank)
      continue;
    const auto& o = dec.origins()[r];
    const auto& s = dec.sizes()[r];
    bool any = false;
    for (std::size_t i = 0; i < Nown; ++i) {
      double d2 = 0.0;
      Vec<3> sh;
      for (int d = 0; d < 3; ++d) {
        const double lo = map.origin[d] + o[d] * map.cellSize[d];
        const double hi = map.origin[d] + (o[d] + s[d]) * map.cellSize[d];
        const double c[3] = {pos[i][d], pos[i][d] - L[d], pos[i][d] + L[d]};
        double best = 1e300, img = pos[i][d];
        for (double p : c) {
          const double gap = (p < lo) ? (lo - p) : (p > hi) ? (p - hi) : 0.0;
          if (gap < best) {
            best = gap;
            img = p;
          }
        }
        sh[d] = img - pos[i][d];
        d2 += best * best;
      }
      if (d2 < rcut * rcut) {
        any = true;
        refIdx.push_back(static_cast<Index>(i));
        refShift.push_back(sh);
      }
    }
    if (any)
      refRanks.push_back(r);
  }
  check(topoOff.sendRanks == refRanks, "default: send ranks == one-image rule");
  check(topoOff.sendIdx == refIdx, "default: send indices == one-image rule");
  // The send-side shifts are not exposed; the receiver's copy is. Each received default ghost
  // must carry exactly (bytes) the one-image rule's shift, image - x, for its owner's position.
  {
    const std::size_t G0 = off.numGhost();
    std::vector<double> id0(G0);
    off.forward(ownIdD.data(), id0.data());
    for (std::size_t g = 0; g < static_cast<std::size_t>(topoOff.numReceived); ++g) {
      const std::int64_t id = (std::int64_t)std::llround(id0[g]);
      const GP* src = nullptr;
      for (const auto& a : all)
        if (a.id == id) {
          src = &a;
          break;
        }
      Vec<3> x{src->p[0], src->p[1], src->p[2]}, img;
      mig.withinRcutOfBlock(x, rank, rcut, img);
      bool ok = true;
      for (int d = 0; d < 3; ++d) {
        const double want = img[d] - x[d];
        ok = ok && std::memcmp(&topoOff.shift[g][d], &want, sizeof(double)) == 0;
      }
      check(ok, "default: received shift == image - x (bytes)");
    }
  }
  // Self-ghosts are the same with the flag on or off.
  check(topoOff.selfIdx == topo.selfIdx, "self-ghosts independent of allImages");
  // (4) what the default misses is exactly the oracle's surplus.
  long long missedByDefault =
      (long long)expect.size() - (long long)static_cast<std::size_t>(topoOff.numReceived);

  // np = 1: no cross-rank exchange, so the flag must change nothing.
  if (size == 1) {
    check(topo.sendIdx.empty() && topoOff.sendIdx.empty() && NR == 0, "np=1: no cross-rank ghost");
    check(topo.shift.size() == topoOff.shift.size() &&
              std::memcmp(topo.shift.data(), topoOff.shift.data(),
                          topo.shift.size() * sizeof(Vec<3>)) == 0,
          "np=1: allImages is inert");
  }

  long long vals[4] = {(long long)NR, localEntries, multiImage, missedByDefault}, gsum[4] = {};
  MPI_Allreduce(vals, gsum, 4, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (gsum[0] != gsum[1])
      ++fail;  // every sent entry is received once
    if (size >= 2 && (gsum[2] == 0 || gsum[3] <= 0))
      ++fail;  // the scene must exercise the multi-image case the default drops
  }
  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf(
        "# allImages: ghosts=%lld sent=%lld (particle,rank) pairs with >=2 images=%lld; default "
        "build misses %lld images\n",
        gsum[0], gsum[1], gsum[2], gsum[3]);
    if (total == 0)
      std::printf(
          "OK (np=%d): allImages sends every periodic image; default is the one-image rule\n",
          size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}

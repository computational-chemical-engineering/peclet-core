// Back-to-back NBX rounds on one communicator must not leak messages between rounds.
//
// The race: a rank that has already observed the Ibarrier of round k complete posts round k+1's
// Issends, while a neighbour still draining round k probes MPI_ANY_SOURCE on the same tag and
// swallows the new message as an old one. Found on Snellius (2026-09-02) as a hang in the first
// halo exchange of a freshly built multigrid level: topology builds run one NBX round per level,
// and on a 64-rank sub-communicator ranks lost up to all 26 of their send partners. The engine now
// rotates the tag per round through a per-communicator attribute; this test drives many rounds
// with per-round payloads and checks every round received exactly its own messages, on
// MPI_COMM_WORLD and on a split sub-communicator (the telescoped-multigrid shape).
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "peclet/core/halo/nbx.hpp"

using peclet::core::halo::NbxEngine;

namespace {

// One round: every rank sends (round, src) to all other ranks in `comm`; returns the number of
// messages that were missing, duplicated, or carried another round's stamp.
int oneRound(MPI_Comm comm, int round) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  NbxEngine nbx(comm);
  int next = 0;
  auto packNext = [&](std::vector<char>& out) -> int {
    if (next == rank)
      ++next;
    if (next >= size)
      return -1;
    int payload[2] = {round, rank};
    out.resize(sizeof(payload));
    std::memcpy(out.data(), payload, sizeof(payload));
    return next++;
  };
  std::vector<int> got(size, 0);
  int wrongRound = 0;
  auto onRecv = [&](int src, std::vector<char>& msg) {
    int payload[2];
    std::memcpy(payload, msg.data(), sizeof(payload));
    if (payload[0] != round || payload[1] != src)
      ++wrongRound;
    ++got[src];
  };
  nbx.exchange(packNext, onRecv, /*tag=*/7301);
  int bad = wrongRound;
  for (int r = 0; r < size; ++r)
    if (r != rank && got[r] != 1)
      ++bad;
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // The per-communicator counter advances by one per round and starts at 0 on a fresh comm; the
  // wire tag sits in the reserved block of the caller's family, never on a direct tag.
  using namespace peclet::core::halo::detail;
  int fail = 0;
  {
    MPI_Comm dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &dup);
    const int t0 = nbxRoundTag(dup, 7301);
    const int t1 = nbxRoundTag(dup, 7301);
    const int w0 = nbxRoundTag(MPI_COMM_WORLD, 100);
    const int base7301 = kNbxTagBase + (7301 % kNbxFamilies) * static_cast<int>(kNbxRoundTags);
    if (t0 != base7301 || t1 != base7301 + 1 ||
        w0 != kNbxTagBase + (100 % kNbxFamilies) * static_cast<int>(kNbxRoundTags))
      ++fail;
    MPI_Comm_free(&dup);
  }
  {
    // The suite's NBX families land in distinct blocks, every rotated tag stays inside the
    // reserved range, and none of them can ever equal a direct point-to-point tag (all < base).
    // 2026-09-05: with tag = baseTag + round, family 7501's second round hit the particle halo's
    // direct forwardPositions tag 7502 (particle_halo_np8 hung on an oversubscribed 2-core
    // runner) and family 0 walked over the AMR gather tags 11/41/45 (wrong ghosts at np=4,8).
    const int families[] = {0, 11, 7301, 7401, 7402, 7411, 7501};
    const int directTags[] = {11, 41, 45, 7502, 7503, 7601, 7603, 7604, 4096, 20479};
    MPI_Comm dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &dup);
    std::vector<int> seen;
    for (int f : families) {
      MPI_Comm fresh;
      MPI_Comm_dup(MPI_COMM_WORLD, &fresh);
      for (int r = 0; r < 200; ++r) {
        const int tag = nbxRoundTag(fresh, f);
        if (tag < kNbxTagBase || tag >= kNbxTagBase + kNbxFamilies * kNbxRoundTags)
          ++fail;
        for (int d : directTags)
          if (tag == d)
            ++fail;
        if (r < kNbxRoundTags)
          seen.push_back(tag);
      }
      MPI_Comm_free(&fresh);
    }
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end())
      ++fail;  // two families shared a wire tag within the first 64 rounds
    MPI_Comm_free(&dup);
  }

  // Many rounds, no barrier between them, on the world communicator.
  const int rounds = 300;
  for (int k = 0; k < rounds; ++k)
    fail += oneRound(MPI_COMM_WORLD, k);

  // The multigrid shape: a sub-communicator of "roots" running its own rounds while the world
  // communicator is also used, plus the members idling. Every second rank is a root.
  MPI_Comm sub;
  MPI_Comm_split(MPI_COMM_WORLD, rank % 2 == 0 ? 0 : MPI_UNDEFINED, rank, &sub);
  for (int k = 0; k < 100; ++k) {
    fail += oneRound(MPI_COMM_WORLD, 1000 + k);
    if (sub != MPI_COMM_NULL)
      fail += oneRound(sub, 2000 + k);
  }
  if (sub != MPI_COMM_NULL)
    MPI_Comm_free(&sub);

  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("nbx_rounds np=%d: %d failures\n", size, totalFail);
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

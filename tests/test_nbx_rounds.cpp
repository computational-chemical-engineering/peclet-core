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

  // The per-communicator counter advances by one per round and starts at 0 on a fresh comm.
  int fail = 0;
  {
    MPI_Comm dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &dup);
    const int t0 = peclet::core::halo::detail::nbxRoundTag(dup, 7301);
    const int t1 = peclet::core::halo::detail::nbxRoundTag(dup, 7301);
    const int w0 = peclet::core::halo::detail::nbxRoundTag(MPI_COMM_WORLD, 100);
    if (t0 != 7301 || t1 != 7302 || w0 != 100)
      ++fail;
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

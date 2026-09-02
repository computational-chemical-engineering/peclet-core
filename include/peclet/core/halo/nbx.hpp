// core — NBX nonblocking-consensus sparse data exchange.
//
// A clean reimplementation of the async engine in block_decomposer/src/MPISync.hpp
// (pbs::MPISync::communicateParticleData) using the canonical NBX protocol of Hoefler et al.,
// "Scalable Communication Protocols for Dynamic Sparse Data Exchange":
//   1. Post every outgoing message as a nonblocking synchronous send (MPI_Issend).
//   2. Loop probing/receiving incoming messages; once all local Issends are matched, enter
//      MPI_Ibarrier; keep draining until the barrier completes — then every message has been
//      received globally.
// MPI_Issend completes only when the matching receive has been posted, which is what makes the
// barrier a correct global-termination signal without any prior agreement on message counts.
//
// This is the right engine for DYNAMIC / SPARSE patterns where neither the neighbour set nor the
// message sizes are known ahead of time (particle migration, halo-topology setup). For STATIC grid
// halos use peclet::core::halo::GridHaloTopology's persistent-neighbour path instead.
#ifndef PECLET_CORE_HALO_NBX_HPP
#define PECLET_CORE_HALO_NBX_HPP

#include <memory>
#include <vector>

#include "peclet/core/common/mpi.hpp"

namespace peclet::core::halo {

namespace detail {
/// Per-communicator round counter, kept as an MPI attribute so it lives and dies with the
/// communicator (a handle-keyed map would confuse a freed handle with its reuse). Every rank of a
/// communicator runs the same sequence of NBX rounds on it, so the counter agrees globally.
constexpr long kNbxRoundTags = 64;
inline int nbxRoundTag(MPI_Comm comm, int baseTag) {
  static int keyval = MPI_KEYVAL_INVALID;
  if (keyval == MPI_KEYVAL_INVALID)
    MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &keyval, nullptr);
  void* attr = nullptr;
  int flag = 0;
  MPI_Comm_get_attr(comm, keyval, &attr, &flag);
  const long round = flag ? reinterpret_cast<long>(attr) : 0;
  MPI_Comm_set_attr(comm, keyval, reinterpret_cast<void*>(round + 1));
  return baseTag + static_cast<int>(round % kNbxRoundTags);
}
}  // namespace detail

class NbxEngine {
 public:
  explicit NbxEngine(MPI_Comm comm = MPI_COMM_WORLD) : comm_(comm) {}

  MPI_Comm comm() const { return comm_; }
  int rank() const {
    int r;
    MPI_Comm_rank(comm_, &r);
    return r;
  }
  int size() const {
    int s;
    MPI_Comm_size(comm_, &s);
    return s;
  }

  // Drive one full sparse exchange.
  //   packNext(std::vector<char>& out) -> int : fill `out` with the next outgoing message and
  //       return its destination rank; return < 0 when this rank has no more messages to send.
  //   onRecv(int src, std::vector<char>& msg)  : handle a received message.
  //
  // `baseTag` is the caller's tag FAMILY: the round actually uses baseTag + (round % 64), where
  // the round counter is per communicator. Consecutive rounds on one communicator MUST NOT share a
  // tag: a rank that has already observed the Ibarrier complete goes on to post the next round's
  // Issends, while a neighbour that has not yet observed completion is still probing — and would
  // receive the next round's message as this round's (Hoefler et al. §4). Measured on Snellius
  // 2026-09-02: back-to-back topology builds (one per multigrid level) on a 64-rank
  // sub-communicator lost 0-20 of 26 send partners per rank, hanging the first exchange. Keep
  // families >= 64 apart.
  template <typename PackNext, typename OnRecv>
  void exchange(PackNext&& packNext, OnRecv&& onRecv, int baseTag = 0) {
    const int tag = detail::nbxRoundTag(comm_, baseTag);
    // 1) Post all sends as synchronous nonblocking.
    std::vector<std::unique_ptr<std::vector<char>>> sendBufs;
    std::vector<MPI_Request> sendReqs;
    for (;;) {
      auto buf = std::make_unique<std::vector<char>>();
      int dest = packNext(*buf);
      if (dest < 0)
        break;
      MPI_Request req;
      MPI_Issend(buf->data(), static_cast<int>(buf->size()), MPI_BYTE, dest, tag, comm_, &req);
      sendBufs.push_back(std::move(buf));
      sendReqs.push_back(req);
    }

    // 2) NBX consensus loop.
    bool barrierStarted = false;
    bool done = false;
    MPI_Request barrierReq = MPI_REQUEST_NULL;
    std::vector<char> recvBuf;
    while (!done) {
      int flag = 0;
      MPI_Status st;
      MPI_Iprobe(MPI_ANY_SOURCE, tag, comm_, &flag, &st);
      if (flag) {
        int count = 0;
        MPI_Get_count(&st, MPI_BYTE, &count);
        recvBuf.resize(count);
        MPI_Recv(recvBuf.data(), count, MPI_BYTE, st.MPI_SOURCE, st.MPI_TAG, comm_,
                 MPI_STATUS_IGNORE);
        onRecv(st.MPI_SOURCE, recvBuf);
        continue;  // keep draining before testing for completion
      }
      if (!barrierStarted) {
        int allSent = 0;
        MPI_Testall(static_cast<int>(sendReqs.size()), sendReqs.data(), &allSent,
                    MPI_STATUSES_IGNORE);
        if (allSent) {
          MPI_Ibarrier(comm_, &barrierReq);
          barrierStarted = true;
        }
      } else {
        int bdone = 0;
        MPI_Test(&barrierReq, &bdone, MPI_STATUS_IGNORE);
        if (bdone)
          done = true;
      }
    }
  }

 private:
  MPI_Comm comm_;
};

}  // namespace peclet::core::halo

#endif  // PECLET_CORE_HALO_NBX_HPP

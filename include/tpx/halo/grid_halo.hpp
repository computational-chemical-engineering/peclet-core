// transport-core — asynchronous ghost-layer exchange for a block-decomposed structured grid.
//
// Design (see suite/docs/INTERFACES.md): TOPOLOGY is separated from EXCHANGE.
//   * buildTopology() (done once, or on re-decomposition) figures out, for every ghost cell of this
//     rank's block, which rank owns the corresponding global cell (with periodic wrap), then does a
//     single NBX round so each owner learns which of its inner cells to send. The result is matched
//     send/recv index lists plus a local self-copy list (for periodic wrap onto one's own block).
//   * exchange*(field) (done every step) just moves payload. It is FIELD-AGNOSTIC: any type with
//     bytesPerElem()/pack(localIdx,dst)/unpack(localIdx,src) works, so a scalar grid field, a vector
//     grid field, or (later) a particle attribute array all flow through one path.
//
// Two interchangeable engines, identical results:
//   exchangeNbx(field)        — nonblocking-consensus; robust for dynamic/sparse patterns.
//   exchangePersistent(field) — MPI_Neighbor_alltoallv on a cached distributed-graph communicator;
//                               fastest for the STATIC neighbour pattern of a fixed grid.
// Both support compute/comm overlap via start()/wait() (NBX) — see exchangeNbx.
#ifndef TPX_HALO_GRID_HALO_HPP
#define TPX_HALO_GRID_HALO_HPP

#include <mpi.h>

#include <cstring>
#include <map>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/decomp/block_indexer.hpp"
#include "tpx/halo/nbx.hpp"

namespace tpx::halo {

/// A contiguous local array of `T` (one per cell of the extended block) viewed as a packable field.
template <typename T>
struct GridFieldView {
  T* data;
  std::size_t bytesPerElem() const { return sizeof(T); }
  void pack(Index localIdx, char* dst) const { std::memcpy(dst, &data[localIdx], sizeof(T)); }
  void unpack(Index localIdx, const char* src) { std::memcpy(&data[localIdx], src, sizeof(T)); }
};

template <int Dim>
class GridHalo {
 public:
  GridHalo() = default;

  /// Build the halo topology for `rank`'s block of `dec`, with the given ghost width and per-axis
  /// periodicity. Communicates once (NBX) to establish matched send lists.
  void buildTopology(const decomp::BlockDecomposer<Dim>& dec, int rank, int ghostWidth,
                     const std::array<bool, Dim>& periodic, MPI_Comm comm = MPI_COMM_WORLD) {
    comm_ = comm;
    rank_ = rank;
    indexer_.init(dec.origins()[rank], dec.sizes()[rank], ghostWidth);
    clear();

    // For each ghost cell, find the owner of its (wrapped) global cell.
    std::map<int, std::vector<Index>> recvIdxByRank;   // local ghost cell indices to fill
    std::map<int, std::vector<Index>> recvGlobByRank;  // the global cells we need from that owner
    const IVec<Dim>& gsize = dec.globalSize();

    indexer_.forEachAll([&](const IVec<Dim>& lmd) {
      if (indexer_.isInner(lmd)) return;
      IVec<Dim> g{}, gw{};
      bool skip = false;
      for (int i = 0; i < Dim; ++i) {
        g[i] = lmd[i] + indexer_.originInclGhost()[i];
        Index c = g[i];
        if (c < 0 || c >= gsize[i]) {
          if (periodic[i]) {
            c = wrap(c, gsize[i]);
          } else {
            skip = true;
            break;
          }
        }
        gw[i] = c;
      }
      if (skip) return;  // non-periodic physical boundary: BC fills these, no comm
      int owner = dec.ownerOf(gw);
      Index ghostLocal = indexer_.localMdToLocal(lmd);
      if (owner == rank_) {
        // Periodic wrap onto our own block (e.g. single block, periodic): pure local copy.
        selfDst_.push_back(ghostLocal);
        selfSrc_.push_back(indexer_.globalToLocal(gw));
      } else {
        recvIdxByRank[owner].push_back(ghostLocal);
        recvGlobByRank[owner].push_back(dec.linearGlobal(gw));
      }
    });

    for (auto& [r, idx] : recvIdxByRank) {
      recvRanks_.push_back(r);
      recvIdx_.push_back(std::move(idx));
      recvGlob_.push_back(std::move(recvGlobByRank[r]));
    }

    // One NBX round: tell each owner which global cells we need; learn what we must send.
    NbxEngine nbx(comm_);
    std::size_t k = 0;
    auto packNext = [&](std::vector<char>& out) -> int {
      if (k >= recvRanks_.size()) return -1;
      int dest = recvRanks_[k];
      const auto& g = recvGlob_[k];
      ++k;
      out.resize(g.size() * sizeof(Index));
      std::memcpy(out.data(), g.data(), out.size());
      return dest;
    };
    auto onRecv = [&](int src, std::vector<char>& msg) {
      std::size_t n = msg.size() / sizeof(Index);
      const Index* g = reinterpret_cast<const Index*>(msg.data());
      std::vector<Index> sidx(n);
      for (std::size_t i = 0; i < n; ++i) {
        sidx[i] = indexer_.globalToLocal(dec.multiGlobal(g[i]));
      }
      sendRanks_.push_back(src);
      sendIdx_.push_back(std::move(sidx));
    };
    nbx.exchange(packNext, onRecv, /*tag=*/7301);

    buildRecvLookup();
    persistentReady_ = false;
  }

  // --- Field-agnostic exchange (NBX engine, with overlap support) ---------------------------------

  /// Post all sends/recvs; returns immediately so the caller can compute the block interior.
  template <typename Field>
  void start(Field& field, int tag = 0) {
    applySelfCopy(field);
    const std::size_t es = field.bytesPerElem();

    // Post receives first.
    recvReqs_.assign(recvRanks_.size(), MPI_REQUEST_NULL);
    recvBufs_.resize(recvRanks_.size());
    for (std::size_t k = 0; k < recvRanks_.size(); ++k) {
      recvBufs_[k].resize(recvIdx_[k].size() * es);
      MPI_Irecv(recvBufs_[k].data(), static_cast<int>(recvBufs_[k].size()), MPI_BYTE, recvRanks_[k],
                tag, comm_, &recvReqs_[k]);
    }
    // Pack and post sends.
    sendReqs_.assign(sendRanks_.size(), MPI_REQUEST_NULL);
    sendBufs_.resize(sendRanks_.size());
    for (std::size_t k = 0; k < sendRanks_.size(); ++k) {
      auto& buf = sendBufs_[k];
      buf.resize(sendIdx_[k].size() * es);
      for (std::size_t i = 0; i < sendIdx_[k].size(); ++i) {
        field.pack(sendIdx_[k][i], buf.data() + i * es);
      }
      MPI_Isend(buf.data(), static_cast<int>(buf.size()), MPI_BYTE, sendRanks_[k], tag, comm_,
                &sendReqs_[k]);
    }
    pendingTag_ = tag;
  }

  /// Complete a started exchange: wait for receives and scatter them into the ghost cells.
  template <typename Field>
  void wait(Field& field) {
    const std::size_t es = field.bytesPerElem();
    // Unpack receives as they arrive.
    for (std::size_t done = 0; done < recvReqs_.size(); ++done) {
      int k = 0;
      MPI_Waitany(static_cast<int>(recvReqs_.size()), recvReqs_.data(), &k, MPI_STATUS_IGNORE);
      if (k == MPI_UNDEFINED) break;
      for (std::size_t i = 0; i < recvIdx_[k].size(); ++i) {
        field.unpack(recvIdx_[k][i], recvBufs_[k].data() + i * es);
      }
    }
    if (!sendReqs_.empty()) {
      MPI_Waitall(static_cast<int>(sendReqs_.size()), sendReqs_.data(), MPI_STATUSES_IGNORE);
    }
  }

  /// Convenience: full blocking exchange via the NBX-style engine (start + wait).
  template <typename Field>
  void exchangeNbx(Field& field, int tag = 0) {
    start(field, tag);
    wait(field);
  }

  // --- Field-agnostic exchange (persistent neighborhood collective) -------------------------------

  /// Full exchange via MPI_Neighbor_alltoallv on a cached distributed-graph communicator. Fastest
  /// for the fixed neighbour pattern of a static grid. Identical result to exchangeNbx.
  template <typename Field>
  void exchangePersistent(Field& field) {
    applySelfCopy(field);
    const std::size_t es = field.bytesPerElem();
    ensureGraphComm(es);

    // Pack the contiguous send buffer in destination order.
    for (std::size_t k = 0; k < sendRanks_.size(); ++k) {
      char* dst = graphSendBuf_.data() + graphSendDispl_[k];
      for (std::size_t i = 0; i < sendIdx_[k].size(); ++i) {
        field.pack(sendIdx_[k][i], dst + i * es);
      }
    }
    MPI_Neighbor_alltoallv(graphSendBuf_.data(), graphSendCnt_.data(), graphSendDispl_.data(),
                           MPI_BYTE, graphRecvBuf_.data(), graphRecvCnt_.data(),
                           graphRecvDispl_.data(), MPI_BYTE, graphComm_);
    // Scatter the contiguous recv buffer into ghost cells (source order == recvRanks_ order).
    for (std::size_t k = 0; k < recvRanks_.size(); ++k) {
      const char* src = graphRecvBuf_.data() + graphRecvDispl_[k];
      for (std::size_t i = 0; i < recvIdx_[k].size(); ++i) {
        field.unpack(recvIdx_[k][i], src + i * es);
      }
    }
  }

  // --- Flattened topology (contiguous, device-friendly; consumed by the CUDA exchange) ---
  struct FlatTopology {
    MPI_Comm comm = MPI_COMM_NULL;
    std::vector<int> sendRanks, recvRanks;     // neighbour ranks
    std::vector<int> sendCounts, recvCounts;   // elements per neighbour (same order)
    std::vector<Index> sendIdx, recvIdx;       // flattened local indices, grouped by neighbour
    std::vector<Index> selfSrc, selfDst;       // local periodic self-copy (inner -> ghost)
  };

  FlatTopology flatten() const {
    FlatTopology t;
    t.comm = comm_;
    t.sendRanks = std::vector<int>(sendRanks_.begin(), sendRanks_.end());
    t.recvRanks = std::vector<int>(recvRanks_.begin(), recvRanks_.end());
    for (const auto& v : sendIdx_) {
      t.sendCounts.push_back(static_cast<int>(v.size()));
      t.sendIdx.insert(t.sendIdx.end(), v.begin(), v.end());
    }
    for (const auto& v : recvIdx_) {
      t.recvCounts.push_back(static_cast<int>(v.size()));
      t.recvIdx.insert(t.recvIdx.end(), v.begin(), v.end());
    }
    t.selfSrc = selfSrc_;
    t.selfDst = selfDst_;
    return t;
  }

  // --- Introspection (used by tests/benchmarks) ---
  const decomp::BlockIndexer<Dim>& indexer() const { return indexer_; }
  std::size_t numNeighbors() const { return recvRanks_.size(); }
  std::size_t numGhostRecv() const {
    std::size_t n = 0;
    for (auto& v : recvIdx_) n += v.size();
    return n;
  }

  ~GridHalo() { freeGraphComm(); }

  GridHalo(const GridHalo&) = delete;
  GridHalo& operator=(const GridHalo&) = delete;

 private:
  // Free the cached graph communicator, but never after MPI_Finalize (e.g. if this object outlives
  // finalize on the stack of main()). Calling MPI_Comm_free post-finalize aborts the job.
  void freeGraphComm() {
    if (graphComm_ == MPI_COMM_NULL) return;
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Comm_free(&graphComm_);
    graphComm_ = MPI_COMM_NULL;
  }

  void clear() {
    sendRanks_.clear();
    sendIdx_.clear();
    recvRanks_.clear();
    recvIdx_.clear();
    recvGlob_.clear();
    selfSrc_.clear();
    selfDst_.clear();
    freeGraphComm();
    persistentReady_ = false;
  }

  void buildRecvLookup() {
    recvRankPos_.clear();
    for (std::size_t k = 0; k < recvRanks_.size(); ++k) recvRankPos_[recvRanks_[k]] = k;
  }

  template <typename Field>
  void applySelfCopy(Field& field) {
    // Periodic copy within our own block: read inner cell, write ghost cell. We go through the
    // field's pack/unpack so it works for any payload type.
    if (selfSrc_.empty()) return;
    std::vector<char> tmp(field.bytesPerElem());
    for (std::size_t i = 0; i < selfSrc_.size(); ++i) {
      field.pack(selfSrc_[i], tmp.data());
      field.unpack(selfDst_[i], tmp.data());
    }
  }

  void ensureGraphComm(std::size_t es) {
    if (persistentReady_ && graphElemSize_ == es) return;
    if (graphComm_ != MPI_COMM_NULL) MPI_Comm_free(&graphComm_);

    // Distributed graph: sources = ranks we receive from, destinations = ranks we send to.
    std::vector<int> sources(recvRanks_.begin(), recvRanks_.end());
    std::vector<int> dests(sendRanks_.begin(), sendRanks_.end());
    MPI_Dist_graph_create_adjacent(comm_, static_cast<int>(sources.size()), sources.data(),
                                    MPI_UNWEIGHTED, static_cast<int>(dests.size()), dests.data(),
                                    MPI_UNWEIGHTED, MPI_INFO_NULL, /*reorder=*/0, &graphComm_);

    // Counts/displacements (in bytes) in the graph's neighbour ordering, which for
    // create_adjacent matches the order we passed sources/dests.
    graphSendCnt_.resize(dests.size());
    graphSendDispl_.resize(dests.size());
    int off = 0;
    for (std::size_t k = 0; k < dests.size(); ++k) {
      graphSendCnt_[k] = static_cast<int>(sendIdx_[k].size() * es);
      graphSendDispl_[k] = off;
      off += graphSendCnt_[k];
    }
    graphSendBuf_.resize(off);

    graphRecvCnt_.resize(sources.size());
    graphRecvDispl_.resize(sources.size());
    off = 0;
    for (std::size_t k = 0; k < sources.size(); ++k) {
      graphRecvCnt_[k] = static_cast<int>(recvIdx_[k].size() * es);
      graphRecvDispl_[k] = off;
      off += graphRecvCnt_[k];
    }
    graphRecvBuf_.resize(off);

    graphElemSize_ = es;
    persistentReady_ = true;
  }

  MPI_Comm comm_ = MPI_COMM_WORLD;
  int rank_ = 0;
  decomp::BlockIndexer<Dim> indexer_;

  // Matched topology: element i of send list k corresponds to element i of the peer's recv list.
  std::vector<int> sendRanks_;
  std::vector<std::vector<Index>> sendIdx_;  // local inner-cell indices to send
  std::vector<int> recvRanks_;
  std::vector<std::vector<Index>> recvIdx_;   // local ghost-cell indices to fill
  std::vector<std::vector<Index>> recvGlob_;  // global cells requested (build-time only)
  std::map<int, std::size_t> recvRankPos_;
  std::vector<Index> selfSrc_, selfDst_;  // local periodic self-copy (inner -> ghost)

  // NBX exchange scratch.
  std::vector<MPI_Request> sendReqs_, recvReqs_;
  std::vector<std::vector<char>> sendBufs_, recvBufs_;
  int pendingTag_ = 0;

  // Persistent neighborhood-collective state.
  MPI_Comm graphComm_ = MPI_COMM_NULL;
  bool persistentReady_ = false;
  std::size_t graphElemSize_ = 0;
  std::vector<int> graphSendCnt_, graphSendDispl_, graphRecvCnt_, graphRecvDispl_;
  std::vector<char> graphSendBuf_, graphRecvBuf_;
};

}  // namespace tpx::halo

#endif  // TPX_HALO_GRID_HALO_HPP

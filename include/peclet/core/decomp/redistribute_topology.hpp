// core — the planned, repeatable movement of a multigrid level's fields onto a stage target and
// back (amr/docs/amr_mg_core_boundary.md §4; the suite decision "Coarse-level redistribution lives
// in core").
//
// Same split as GridHaloTopology / ParticleHaloTopology: build() (once per hierarchy build)
// establishes who sends which cells where; forward() / backward() (every V-cycle) only move
// payload, with no handshake. Host-staged; the buffers are a coarse level's cells.
//
//   forward  : level fields (this rank's current block)  -> target fields (the target block)
//   backward : target fields                             -> level fields   (OVERWRITES)
//
// LAYOUT CONTRACT. The topology never assumes where a cell lives in the caller's arrays: it calls
// `srcIndex(gcell)` for every global cell of this rank's current block and `dstIndex(gcell)` for
// every global cell of the target block this rank owns, once, at build, and moves values between
// those slots. flow's padded x-fastest boxes and amr's Morton-ordered leaves use it alike.
//
// The kinds (stage_target.hpp):
//   SiblingMerge — MPI_Gatherv / MPI_Scatterv on the group communicator, owner = root. Each member
//                  packs its block's cells in x-fastest global order and the owner unpacks member
//                  by member in group order: the same permutation as flow's Telescope
//                  (teleGather / teleScatterAdd; flow ADDS the scattered values into its level,
//                  which is the caller's arithmetic, not the movement's).
//   Replicated   — MPI_Allgatherv on the parent communicator, rank order; backward is a local
//                  pick (every rank holds the identical target), as amr's ReplicatedTailStage.
//   Repartition  — planned point-to-point on the parent communicator (design §11.3). The segments
//                  are the box intersections of this rank's current block with every target block
//                  (send) and of this rank's target block with every current block (receive), each
//                  x-fastest; every rank computes all of them from the two decompositions alone,
//                  so there is no handshake. One MPI_Isend / MPI_Irecv per non-empty segment, the
//                  self-intersection a direct copy, MPI_Waitall before returning. backward is the
//                  mirror. Each cell has one source and one destination, so the values are
//                  independent of message order — bitwise those of a one-shot
//                  redistributeGridFields over the same boxes.
// Several fields move in ONE collective (one message per segment for Repartition), field-major
// within each segment. Pure copies: every value arrives bitwise.
#ifndef PECLET_CORE_DECOMP_REDISTRIBUTE_TOPOLOGY_HPP
#define PECLET_CORE_DECOMP_REDISTRIBUTE_TOPOLOGY_HPP

#include <algorithm>
#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/decomp/stage_comm.hpp"
#include "peclet/core/decomp/stage_target.hpp"

namespace peclet::core::decomp {

template <int Dim, class T>
class RedistributeTopology {
  static_assert(std::is_trivially_copyable_v<T>, "RedistributeTopology moves T as raw bytes");

 public:
  RedistributeTopology() = default;

  /// Repartition's point-to-point tags: kTagBase + id for the per-topology `id` passed to build(),
  /// id in [0, kTagSpan) — [1, 10], below the AMR direct tags (11 / 41 / 45) and never the halo
  /// default 0 (core CLAUDE.md: direct tags stay below NBX's reserved [24576, 32768)).
  static constexpr int kTagBase = 1;
  static constexpr int kTagSpan = 10;

  /// Establish the movement from `src` (the level's current decomposition; parent rank r owns
  /// block r) to `dst` for the communicators `c` (makeStageComm(parent, dst)). Collective-free:
  /// every rank derives the whole pattern from the replicated decompositions. `srcIndex` /
  /// `dstIndex` map a global cell (IVec<Dim>) of this rank's current / target block to its slot in
  /// the caller's arrays; each is called once per cell here and never again. `id` (Repartition
  /// only; ignored otherwise) is the per-topology tag offset — the level index — so two levels'
  /// stages on one communicator cannot pair messages across each other.
  template <class SrcIndex, class DstIndex>
  void build(const BlockDecomposer<Dim>& src, const StageTarget<Dim>& dst, const StageComm& c,
             SrcIndex&& srcIndex, DstIndex&& dstIndex, int id = 0) {
    if (dst.kind == StageKind::Repartition) {
      buildRepartition(src, dst, c, srcIndex, dstIndex, id);
      return;
    }
    if (dst.kind != StageKind::SiblingMerge && dst.kind != StageKind::Replicated)
      throw std::invalid_argument("RedistributeTopology: an InPlace target has no movement");
    if (c.kind != dst.kind)
      throw std::invalid_argument("RedistributeTopology: StageComm was made for another target");
    if (src.globalSize() != dst.dec.globalSize())
      throw std::invalid_argument(
          "RedistributeTopology: source and target decompose different grids");
    int size = 1;
    MPI_Comm_rank(c.parent, &rank_);
    MPI_Comm_size(c.parent, &size);
    if (src.numBlocks() != static_cast<std::size_t>(size))
      throw std::invalid_argument(
          "RedistributeTopology: the current decomposition needs one block per parent rank");

    kind_ = dst.kind;
    active_ = c.active;
    comm_ = (kind_ == StageKind::SiblingMerge) ? c.group : c.parent;
    srcSlots_.clear();
    dstSlots_.clear();
    counts_.clear();
    displs_.clear();

    const Block<Dim> mine = src.block(static_cast<std::size_t>(rank_));
    forEachCell(mine, [&](const IVec<Dim>& g) { srcSlots_.push_back(Index(srcIndex(g))); });
    requireDistinct(srcSlots_, "srcIndex");

    // The segment layout: one segment per sending rank, in the collective's rank order (group
    // order for SiblingMerge, parent order for Replicated), each the sender's block x-fastest.
    auto addSegment = [&](const Block<Dim>& b) {
      displs_.push_back(static_cast<Index>(dstSlots_.size()));
      counts_.push_back(cellCount(b));
      forEachCell(b, [&](const IVec<Dim>& g) { dstSlots_.push_back(Index(dstIndex(g))); });
    };
    if (kind_ == StageKind::SiblingMerge) {
      if (active_) {
        const Block<Dim> tb = dst.dec.block(static_cast<std::size_t>(c.myTargetBlock));
        for (int m : c.members) {
          const Block<Dim> mb = src.block(static_cast<std::size_t>(m));
          if (!contains(tb, mb))
            throw std::logic_error(
                "RedistributeTopology: a member's block lies outside its target block");
          addSegment(mb);
        }
        if (static_cast<Index>(dstSlots_.size()) != cellCount(tb))
          throw std::logic_error("RedistributeTopology: the members do not tile the target block");
      }
    } else {
      for (int r = 0; r < size; ++r)
        addSegment(src.block(static_cast<std::size_t>(r)));
      if (static_cast<Index>(dstSlots_.size()) != cellCount(dst.dec.block(0)))
        throw std::logic_error("RedistributeTopology: the blocks do not tile the level grid");
    }
    requireDistinct(dstSlots_, "dstIndex");
  }

  StageKind kind() const { return kind_; }
  /// Whether this rank holds target fields (dst in forward, the input of backward).
  bool active() const { return active_; }
  /// Cells of this rank's current block (the extent `srcIndex` was called on).
  std::size_t numSourceCells() const { return srcSlots_.size(); }
  /// Cells of this rank's target block (0 on an inactive rank).
  std::size_t numTargetCells() const { return dstSlots_.size(); }

  /// Level -> target. Collective on the movement's communicator: every rank takes part, active or
  /// not. `src[f]` is field f on this rank's current block; `dst[f]` (read on active ranks only;
  /// may be empty elsewhere) receives it on the target block. Target slots not named by `dstIndex`
  /// are untouched.
  void forward(const std::vector<const T*>& src, const std::vector<T*>& dst) {
    const std::size_t nF = src.size();
    if (active_ && dst.size() != nF)
      throw std::invalid_argument("RedistributeTopology::forward: src/dst field counts differ");
    if (kind_ == StageKind::Repartition) {
      // send segments carry srcSlots_ from `src`, receive segments land in dstSlots_ of `dst`
      pointToPoint(src, srcSlots_, sendSeg_, dst, dstSlots_, recvSeg_);
      return;
    }
    const std::size_t n = srcSlots_.size();
    sbuf_.resize(n * nF);
    for (std::size_t f = 0; f < nF; ++f)
      for (std::size_t i = 0; i < n; ++i)
        sbuf_[f * n + i] = src[f][srcSlots_[i]];
    const bool unpack = active_;
    if (unpack)
      rbuf_.resize(dstSlots_.size() * nF);
    byteLayout(nF);
    if (kind_ == StageKind::SiblingMerge) {
      MPI_Gatherv(sbuf_.data(), bytes(n * nF), MPI_BYTE, unpack ? rbuf_.data() : nullptr,
                  unpack ? bc_.data() : nullptr, unpack ? bd_.data() : nullptr, MPI_BYTE, 0, comm_);
    } else {
      MPI_Allgatherv(sbuf_.data(), bytes(n * nF), MPI_BYTE, rbuf_.data(), bc_.data(), bd_.data(),
                     MPI_BYTE, comm_);
    }
    if (!unpack)
      return;
    for (std::size_t m = 0; m < counts_.size(); ++m) {
      const std::size_t cnt = static_cast<std::size_t>(counts_[m]);
      const std::size_t d0 = static_cast<std::size_t>(displs_[m]);
      const T* q = rbuf_.data() + d0 * nF;
      for (std::size_t f = 0; f < nF; ++f)
        for (std::size_t i = 0; i < cnt; ++i)
          dst[f][dstSlots_[d0 + i]] = q[f * cnt + i];
    }
  }

  /// Target -> level, overwriting `src[f]` on this rank's current block. SiblingMerge: collective
  /// on the group communicator (`dst` read on the owner only). Replicated: no communication — every
  /// rank picks its own cells from its replicated target. Repartition: the mirror of forward on the
  /// parent communicator (`dst` read on active ranks only).
  void backward(const std::vector<const T*>& dst, const std::vector<T*>& src) {
    const std::size_t nF = src.size();
    if (active_ && dst.size() != nF)
      throw std::invalid_argument("RedistributeTopology::backward: src/dst field counts differ");
    if (kind_ == StageKind::Repartition) {
      // the receive segments go back out of dstSlots_, the send segments come home to srcSlots_
      pointToPoint(dst, dstSlots_, recvSeg_, src, srcSlots_, sendSeg_);
      return;
    }
    const std::size_t n = srcSlots_.size();
    if (kind_ == StageKind::Replicated) {
      const std::size_t d0 = static_cast<std::size_t>(displs_[static_cast<std::size_t>(rank_)]);
      for (std::size_t f = 0; f < nF; ++f)
        for (std::size_t i = 0; i < n; ++i)
          src[f][srcSlots_[i]] = dst[f][dstSlots_[d0 + i]];
      return;
    }
    byteLayout(nF);
    if (active_) {
      sbuf_.resize(dstSlots_.size() * nF);
      for (std::size_t m = 0; m < counts_.size(); ++m) {
        const std::size_t cnt = static_cast<std::size_t>(counts_[m]);
        const std::size_t d0 = static_cast<std::size_t>(displs_[m]);
        T* q = sbuf_.data() + d0 * nF;
        for (std::size_t f = 0; f < nF; ++f)
          for (std::size_t i = 0; i < cnt; ++i)
            q[f * cnt + i] = dst[f][dstSlots_[d0 + i]];
      }
    }
    rbuf_.resize(n * nF);
    MPI_Scatterv(active_ ? sbuf_.data() : nullptr, active_ ? bc_.data() : nullptr,
                 active_ ? bd_.data() : nullptr, MPI_BYTE, rbuf_.data(), bytes(n * nF), MPI_BYTE, 0,
                 comm_);
    for (std::size_t f = 0; f < nF; ++f)
      for (std::size_t i = 0; i < n; ++i)
        src[f][srcSlots_[i]] = rbuf_[f * n + i];
  }

 private:
  /// One segment of the Repartition pattern: `peer` (a parent rank) and a run of `count` slots
  /// starting at `displ` in srcSlots_ (send) or dstSlots_ (receive).
  struct Segment {
    int peer;
    Index displ, count;
  };

  template <class SrcIndex, class DstIndex>
  void buildRepartition(const BlockDecomposer<Dim>& src, const StageTarget<Dim>& dst,
                        const StageComm& c, SrcIndex& srcIndex, DstIndex& dstIndex, int id) {
    if (c.kind != StageKind::Repartition)
      throw std::invalid_argument("RedistributeTopology: StageComm was made for another target");
    if (src.globalSize() != dst.dec.globalSize())
      throw std::invalid_argument(
          "RedistributeTopology: source and target decompose different grids");
    if (id < 0 || id >= kTagSpan)
      throw std::invalid_argument("RedistributeTopology: the topology id must lie in [0, " +
                                  std::to_string(kTagSpan) + ")");
    int size = 1;
    MPI_Comm_rank(c.parent, &rank_);
    MPI_Comm_size(c.parent, &size);
    if (src.numBlocks() != static_cast<std::size_t>(size))
      throw std::invalid_argument(
          "RedistributeTopology: the current decomposition needs one block per parent rank");
    const std::size_t npL = dst.dec.numBlocks();
    if (npL < 1 || npL > static_cast<std::size_t>(size) || dst.ownerOf.size() != npL)
      throw std::invalid_argument(
          "RedistributeTopology: a Repartition target needs 1..size blocks");
    for (std::size_t b = 0; b < npL; ++b)
      if (dst.ownerOf[b] != static_cast<int>(b))
        throw std::invalid_argument(
            "RedistributeTopology: a Repartition target's owners must be the identity");
    if (c.active != (static_cast<std::size_t>(rank_) < npL))
      throw std::invalid_argument("RedistributeTopology: StageComm was made for another target");

    kind_ = StageKind::Repartition;
    active_ = c.active;
    comm_ = c.parent;
    tag_ = kTagBase + id;
    srcSlots_.clear();
    dstSlots_.clear();
    counts_.clear();
    displs_.clear();
    sendSeg_.clear();
    recvSeg_.clear();

    // srcIndex once per cell of my block (block-local x-fastest), then gathered per segment.
    const Block<Dim> mine = src.block(static_cast<std::size_t>(rank_));
    std::vector<Index> mySlot;
    mySlot.reserve(static_cast<std::size_t>(cellCount(mine)));
    forEachCell(mine, [&](const IVec<Dim>& g) { mySlot.push_back(Index(srcIndex(g))); });
    for (std::size_t t = 0; t < npL; ++t) {
      Block<Dim> I;
      if (!intersect(mine, dst.dec.block(t), I))
        continue;
      sendSeg_.push_back({static_cast<int>(t), static_cast<Index>(srcSlots_.size()), cellCount(I)});
      forEachCell(I, [&](const IVec<Dim>& g) {
        srcSlots_.push_back(mySlot[static_cast<std::size_t>(localLinear(g, mine))]);
      });
    }
    if (static_cast<Index>(srcSlots_.size()) != cellCount(mine))
      throw std::logic_error(
          "RedistributeTopology: the send segments do not tile the source block");
    requireDistinct(srcSlots_, "srcIndex");

    if (active_) {
      const Block<Dim> tb = dst.dec.block(static_cast<std::size_t>(rank_));
      for (int s = 0; s < size; ++s) {
        Block<Dim> J;
        if (!intersect(tb, src.block(static_cast<std::size_t>(s)), J))
          continue;
        recvSeg_.push_back({s, static_cast<Index>(dstSlots_.size()), cellCount(J)});
        forEachCell(J, [&](const IVec<Dim>& g) { dstSlots_.push_back(Index(dstIndex(g))); });
      }
      if (static_cast<Index>(dstSlots_.size()) != cellCount(tb))
        throw std::logic_error(
            "RedistributeTopology: the receive segments do not tile the target block");
      requireDistinct(dstSlots_, "dstIndex");
    }
  }

  // Move `out` segments (slots `outSlots` of the fields `from`) to their peers and land `in`
  // segments (slots `inSlots` of the fields `to`) from theirs; the self segment is a direct copy.
  // forward: out = send, in = receive; backward: the reverse. `to` is not touched when `in` is
  // empty (an inactive rank's forward), nor `from` read when `out` is (its backward).
  void pointToPoint(const std::vector<const T*>& from, const std::vector<Index>& outSlots,
                    const std::vector<Segment>& out, const std::vector<T*>& to,
                    const std::vector<Index>& inSlots, const std::vector<Segment>& in) {
    const std::size_t nF = std::max(from.size(), to.size());
    if (!out.empty() && from.size() != nF)
      throw std::invalid_argument("RedistributeTopology: a sending rank needs every field");
    if (!in.empty() && to.size() != nF)
      throw std::invalid_argument("RedistributeTopology: a receiving rank needs every field");
    sbuf_.resize(outSlots.size() * nF);
    rbuf_.resize(inSlots.size() * nF);
    // pack each outgoing segment, field-major within the segment
    for (const Segment& s : out) {
      if (s.peer == rank_)
        continue;
      const std::size_t cnt = static_cast<std::size_t>(s.count),
                        d0 = static_cast<std::size_t>(s.displ);
      T* q = sbuf_.data() + d0 * nF;
      for (std::size_t f = 0; f < nF; ++f)
        for (std::size_t i = 0; i < cnt; ++i)
          q[f * cnt + i] = from[f][outSlots[d0 + i]];
    }
    reqs_.clear();
    reqs_.reserve(out.size() + in.size());
    for (const Segment& s : in) {
      if (s.peer == rank_)
        continue;
      reqs_.emplace_back();
      MPI_Irecv(rbuf_.data() + static_cast<std::size_t>(s.displ) * nF,
                bytes(static_cast<std::size_t>(s.count) * nF), MPI_BYTE, s.peer, tag_, comm_,
                &reqs_.back());
    }
    for (const Segment& s : out) {
      if (s.peer == rank_)
        continue;
      reqs_.emplace_back();
      MPI_Isend(sbuf_.data() + static_cast<std::size_t>(s.displ) * nF,
                bytes(static_cast<std::size_t>(s.count) * nF), MPI_BYTE, s.peer, tag_, comm_,
                &reqs_.back());
    }
    // self: both segments enumerate (my block ∩ my target block) x-fastest, cell for cell
    for (const Segment& so : out) {
      if (so.peer != rank_)
        continue;
      for (const Segment& si : in) {
        if (si.peer != rank_)
          continue;
        const std::size_t cnt = static_cast<std::size_t>(so.count);
        for (std::size_t f = 0; f < nF; ++f)
          for (std::size_t i = 0; i < cnt; ++i)
            to[f][inSlots[static_cast<std::size_t>(si.displ) + i]] =
                from[f][outSlots[static_cast<std::size_t>(so.displ) + i]];
      }
    }
    if (!reqs_.empty())
      MPI_Waitall(static_cast<int>(reqs_.size()), reqs_.data(), MPI_STATUSES_IGNORE);
    for (const Segment& s : in) {
      if (s.peer == rank_)
        continue;
      const std::size_t cnt = static_cast<std::size_t>(s.count),
                        d0 = static_cast<std::size_t>(s.displ);
      const T* q = rbuf_.data() + d0 * nF;
      for (std::size_t f = 0; f < nF; ++f)
        for (std::size_t i = 0; i < cnt; ++i)
          to[f][inSlots[d0 + i]] = q[f * cnt + i];
    }
  }

  static bool intersect(const Block<Dim>& a, const Block<Dim>& b, Block<Dim>& out) {
    for (int k = 0; k < Dim; ++k) {
      const Index lo = std::max(a.origin[k], b.origin[k]);
      const Index hi = std::min(a.origin[k] + a.size[k], b.origin[k] + b.size[k]);
      if (hi <= lo)
        return false;
      out.origin[k] = lo;
      out.size[k] = hi - lo;
    }
    return true;
  }
  // Block-local x-fastest index of a global cell of `b`.
  static Index localLinear(const IVec<Dim>& g, const Block<Dim>& b) {
    Index i = 0;
    for (int k = Dim - 1; k >= 0; --k)
      i = i * b.size[k] + (g[k] - b.origin[k]);
    return i;
  }

  template <class F>
  static void forEachCell(const Block<Dim>& b, F&& f) {
    IVec<Dim> end{};
    for (int k = 0; k < Dim; ++k)
      end[k] = b.origin[k] + b.size[k];
    forEachInBox<Dim>(b.origin, end, f);
  }
  static Index cellCount(const Block<Dim>& b) {
    Index n = 1;
    for (int k = 0; k < Dim; ++k)
      n *= b.size[k];
    return n;
  }
  static bool contains(const Block<Dim>& outer, const Block<Dim>& in) {
    for (int k = 0; k < Dim; ++k)
      if (in.origin[k] < outer.origin[k] ||
          in.origin[k] + in.size[k] > outer.origin[k] + outer.size[k])
        return false;
    return true;
  }
  static void requireDistinct(std::vector<Index> v, const char* what) {
    std::sort(v.begin(), v.end());
    if (std::adjacent_find(v.begin(), v.end()) != v.end())
      throw std::invalid_argument(std::string("RedistributeTopology: ") + what +
                                  " maps two cells to one slot");
  }
  static int bytes(std::size_t elems) {
    const std::size_t b = elems * sizeof(T);
    if (b > static_cast<std::size_t>(INT_MAX))
      throw std::overflow_error("RedistributeTopology: a message exceeds INT_MAX bytes");
    return static_cast<int>(b);
  }
  // Per-segment byte counts / displacements for nF fields (segments are contiguous, field-major).
  void byteLayout(std::size_t nF) {
    bc_.resize(counts_.size());
    bd_.resize(counts_.size());
    for (std::size_t m = 0; m < counts_.size(); ++m) {
      bc_[m] = bytes(static_cast<std::size_t>(counts_[m]) * nF);
      bd_[m] = bytes(static_cast<std::size_t>(displs_[m]) * nF);
    }
  }

  StageKind kind_ = StageKind::InPlace;
  bool active_ = false;
  int rank_ = 0;
  int tag_ = kTagBase;  // Repartition only
  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<Index> srcSlots_;  // this rank's current-block cells, x-fastest -> caller slot
  std::vector<Index> dstSlots_;  // the target block's cells, segment by segment -> caller slot
                                 // (Repartition: srcSlots_ too is segment by segment)
  std::vector<Index> counts_, displs_;  // per segment (cells), active ranks only for SiblingMerge
  std::vector<int> bc_, bd_;            // the same in bytes for the current field count
  std::vector<Segment> sendSeg_, recvSeg_;  // Repartition: runs of srcSlots_ / dstSlots_ by peer
  std::vector<MPI_Request> reqs_;
  std::vector<T> sbuf_, rbuf_;
};

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_DECOMP_REDISTRIBUTE_TOPOLOGY_HPP

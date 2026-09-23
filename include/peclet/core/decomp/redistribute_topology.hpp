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
// Several fields move in ONE collective, field-major within each member's segment. Pure copies:
// every value arrives bitwise.
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

  /// Establish the movement from `src` (the level's current decomposition; parent rank r owns
  /// block r) to `dst` for the communicators `c` (makeStageComm(parent, dst)). Collective-free:
  /// every rank derives the whole pattern from the replicated decompositions. `srcIndex` /
  /// `dstIndex` map a global cell (IVec<Dim>) of this rank's current / target block to its slot in
  /// the caller's arrays; each is called once per cell here and never again.
  template <class SrcIndex, class DstIndex>
  void build(const BlockDecomposer<Dim>& src, const StageTarget<Dim>& dst, const StageComm& c,
             SrcIndex&& srcIndex, DstIndex&& dstIndex) {
    if (dst.kind != StageKind::SiblingMerge && dst.kind != StageKind::Replicated)
      throw std::invalid_argument(
          "RedistributeTopology: only SiblingMerge and Replicated targets move (Repartition is "
          "S2)");
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
  /// rank picks its own cells from its replicated target.
  void backward(const std::vector<const T*>& dst, const std::vector<T*>& src) {
    const std::size_t nF = src.size();
    if (active_ && dst.size() != nF)
      throw std::invalid_argument("RedistributeTopology::backward: src/dst field counts differ");
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
  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<Index> srcSlots_;  // this rank's current-block cells, x-fastest -> caller slot
  std::vector<Index> dstSlots_;  // the target block's cells, segment by segment -> caller slot
  std::vector<Index> counts_, displs_;  // per segment (cells), active ranks only for SiblingMerge
  std::vector<int> bc_, bd_;            // the same in bytes for the current field count
  std::vector<T> sbuf_, rbuf_;
};

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_DECOMP_REDISTRIBUTE_TOPOLOGY_HPP

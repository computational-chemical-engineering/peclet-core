// transport-core — local<->global indexing for a block with a ghost layer.
//
// Ported from block_decomposer/src/BlockIndexer.hpp. A block stores an extended array =
// inner cells + a ghost layer of width `ghostWidth` on every face. Local linear indexing is
// x-fastest over the extended size.
#ifndef PECLET_CORE_DECOMP_BLOCK_INDEXER_HPP
#define PECLET_CORE_DECOMP_BLOCK_INDEXER_HPP

#include "peclet/core/common/types.hpp"

namespace peclet::core::decomp {

template <int Dim>
class BlockIndexer {
 public:
  BlockIndexer() = default;
  BlockIndexer(IVec<Dim> origin, IVec<Dim> size, int ghostWidth) {
    init(origin, size, ghostWidth);
  }

  void init(IVec<Dim> origin, IVec<Dim> size, int ghostWidth) {
    ghostWidth_ = ghostWidth;
    for (int i = 0; i < Dim; ++i) {
      originGhost_[i] = origin[i] - ghostWidth;
      sizeGhost_[i] = size[i] + 2 * ghostWidth;
      sizeInner_[i] = size[i];
    }
  }

  int ghostWidth() const { return ghostWidth_; }
  const IVec<Dim>& originInclGhost() const { return originGhost_; }
  const IVec<Dim>& sizeInclGhost() const { return sizeGhost_; }
  const IVec<Dim>& sizeInner() const { return sizeInner_; }

  /// Total number of cells in the extended (inner + ghost) array.
  Index numCellsInclGhost() const {
    Index n = 1;
    for (int i = 0; i < Dim; ++i) n *= sizeGhost_[i];
    return n;
  }

  /// Global multi-index -> local linear index in the extended array (x-fastest).
  Index globalToLocal(const IVec<Dim>& g) const {
    Index idx = g[Dim - 1] - originGhost_[Dim - 1];
    for (int i = Dim - 2; i >= 0; --i) {
      idx *= sizeGhost_[i];
      idx += (g[i] - originGhost_[i]);
    }
    return idx;
  }

  /// Local multi-index (in extended array, ghost included) -> local linear index (x-fastest).
  Index localMdToLocal(const IVec<Dim>& l) const {
    Index idx = l[Dim - 1];
    for (int i = Dim - 2; i >= 0; --i) {
      idx *= sizeGhost_[i];
      idx += l[i];
    }
    return idx;
  }

  /// Local linear index -> local multi-index (extended array coordinates, 0 at ghost corner).
  IVec<Dim> localToLocalMd(Index idx) const {
    IVec<Dim> l{};
    for (int i = 0; i < Dim; ++i) {
      l[i] = idx % sizeGhost_[i];
      idx /= sizeGhost_[i];
    }
    return l;
  }

  /// Local linear index -> global multi-index.
  IVec<Dim> localToGlobalMd(Index idx) const {
    IVec<Dim> g{};
    for (int i = 0; i < Dim; ++i) {
      g[i] = idx % sizeGhost_[i] + originGhost_[i];
      idx /= sizeGhost_[i];
    }
    return g;
  }

  /// True if a local multi-index lies in the inner (non-ghost) region.
  bool isInner(const IVec<Dim>& l) const {
    for (int i = 0; i < Dim; ++i) {
      if (l[i] < ghostWidth_ || l[i] >= ghostWidth_ + sizeInner_[i]) return false;
    }
    return true;
  }

  /// Visit every inner cell (func receives const IVec<Dim>& local multi-index).
  template <typename Func>
  void forEachInner(Func&& func) const {
    IVec<Dim> bgn{}, end{};
    for (int i = 0; i < Dim; ++i) {
      bgn[i] = ghostWidth_;
      end[i] = sizeGhost_[i] - ghostWidth_;
    }
    forEachInBox<Dim>(bgn, end, func);
  }

  /// Visit every cell including ghosts.
  template <typename Func>
  void forEachAll(Func&& func) const {
    IVec<Dim> bgn{}, end{};
    for (int i = 0; i < Dim; ++i) {
      bgn[i] = 0;
      end[i] = sizeGhost_[i];
    }
    forEachInBox<Dim>(bgn, end, func);
  }

 private:
  IVec<Dim> originGhost_{};
  IVec<Dim> sizeGhost_{};
  IVec<Dim> sizeInner_{};
  int ghostWidth_ = 1;
};

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_DECOMP_BLOCK_INDEXER_HPP

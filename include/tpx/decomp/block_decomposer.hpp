// transport-core — orthogonal recursive bisection (ORB) domain decomposition.
//
// Ported from block_decomposer/src/BlockDecomposer.hpp (pbs::BlockDecomposer), modernized into the
// tpx namespace. The global cell grid is split recursively along its largest axis into `numBlocks`
// rank-owned blocks. Adds ownerOf() (a tree walk) for halo topology construction.
#ifndef TPX_DECOMP_BLOCK_DECOMPOSER_HPP
#define TPX_DECOMP_BLOCK_DECOMPOSER_HPP

#include <cmath>
#include <cstddef>
#include <stack>
#include <vector>

#include "tpx/common/types.hpp"

namespace tpx::decomp {

/// A rank-owned axis-aligned block of the global cell grid.
template <int Dim>
struct Block {
  IVec<Dim> origin{};  ///< inclusive lower corner in global cell coordinates
  IVec<Dim> size{};    ///< extent in cells along each axis
};

template <int Dim>
class BlockDecomposer {
 public:
  BlockDecomposer() = default;
  BlockDecomposer(std::size_t numBlocks, IVec<Dim> globalSize) { init(numBlocks, globalSize); }

  /// Build the decomposition of a `globalSize` cell grid into `numBlocks` blocks.
  void init(std::size_t numBlocks, IVec<Dim> globalSize);

  std::size_t numBlocks() const { return origins_.size(); }
  const IVec<Dim>& globalSize() const { return globalSize_; }
  const std::vector<IVec<Dim>>& origins() const { return origins_; }
  const std::vector<IVec<Dim>>& sizes() const { return sizes_; }

  Block<Dim> block(std::size_t b) const { return {origins_[b], sizes_[b]}; }

  /// Owning block index of a global cell coordinate. Caller must wrap into [0, globalSize) first.
  int ownerOf(const IVec<Dim>& g) const {
    Index node = 0;
    while (tree_[node].splitDim != -1) {
      node = (g[tree_[node].splitDim] < tree_[node].splitValue) ? 2 * node + 1 : 2 * node + 2;
    }
    return static_cast<int>(tree_[node].splitValue);
  }

  /// Global multi-index -> global linear index (x-fastest: I = x + y*nx + z*nx*ny).
  Index linearGlobal(const IVec<Dim>& g) const {
    Index idx = g[Dim - 1];
    for (int i = Dim - 2; i >= 0; --i) {
      idx *= globalSize_[i];
      idx += g[i];
    }
    return idx;
  }

  /// Global linear index -> global multi-index (inverse of linearGlobal).
  IVec<Dim> multiGlobal(Index lin) const {
    IVec<Dim> g{};
    for (int i = 0; i < Dim; ++i) {
      g[i] = lin % globalSize_[i];
      lin /= globalSize_[i];
    }
    return g;
  }

 private:
  struct TreeNode {
    int splitDim = -1;      ///< -1 for a leaf
    Index splitValue = 0;   ///< split coordinate (internal) or leaf/block index (leaf)
  };

  IVec<Dim> globalSize_{};
  std::vector<IVec<Dim>> origins_;
  std::vector<IVec<Dim>> sizes_;
  std::vector<TreeNode> tree_;
};

template <int Dim>
void BlockDecomposer<Dim>::init(std::size_t numBlocks, IVec<Dim> globalSize) {
  globalSize_ = globalSize;
  origins_.clear();
  sizes_.clear();
  tree_.clear();

  struct StackBlock {
    IVec<Dim> origin;
    IVec<Dim> size;
    std::size_t numSub;
    Index treeIndx;
  };
  std::stack<StackBlock> stack;
  stack.push(StackBlock{IVec<Dim>{}, globalSize, numBlocks, 0});

  Index leafIndex = 0;
  while (!stack.empty()) {
    StackBlock cur = stack.top();
    stack.pop();

    if (static_cast<std::size_t>(cur.treeIndx) >= tree_.size()) {
      tree_.resize(cur.treeIndx + 1);
    }

    if (cur.numSub > 1) {
      // Split along the largest axis, proportionally to the sub-block count.
      int kLargest = 0;
      for (int k = 1; k < Dim; ++k) {
        if (cur.size[k] > cur.size[kLargest]) kLargest = k;
      }
      std::size_t numSub = cur.numSub / 2;
      Index szSub = static_cast<Index>(std::round(static_cast<double>(cur.size[kLargest]) *
                                                  static_cast<double>(numSub) /
                                                  static_cast<double>(cur.numSub)));

      StackBlock left = cur;
      left.numSub = numSub;
      left.size[kLargest] = szSub;
      left.treeIndx = 2 * cur.treeIndx + 1;

      StackBlock right = cur;
      right.origin[kLargest] += szSub;
      right.size[kLargest] -= szSub;
      right.numSub = cur.numSub - numSub;
      right.treeIndx = 2 * cur.treeIndx + 2;

      tree_[cur.treeIndx] = TreeNode{kLargest, cur.origin[kLargest] + szSub};
      stack.push(right);
      stack.push(left);
    } else {
      tree_[cur.treeIndx] = TreeNode{-1, leafIndex++};
      origins_.push_back(cur.origin);
      sizes_.push_back(cur.size);
    }
  }
}

}  // namespace tpx::decomp

#endif  // TPX_DECOMP_BLOCK_DECOMPOSER_HPP

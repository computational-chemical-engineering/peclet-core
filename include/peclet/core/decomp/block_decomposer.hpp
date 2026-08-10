// core — orthogonal recursive bisection (ORB) domain decomposition.
//
// Ported from block_decomposer/src/BlockDecomposer.hpp (pbs::BlockDecomposer), modernized into the
// tpx namespace. The global cell grid is split recursively along its largest axis into `numBlocks`
// rank-owned blocks. Adds ownerOf() (a tree walk) for halo topology construction.
#ifndef PECLET_CORE_DECOMP_BLOCK_DECOMPOSER_HPP
#define PECLET_CORE_DECOMP_BLOCK_DECOMPOSER_HPP

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stack>
#include <vector>

#include "peclet/core/common/types.hpp"

namespace peclet::core::decomp {

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
  BlockDecomposer(std::size_t numBlocks, IVec<Dim> globalSize, const std::vector<Real>& weights) {
    init(numBlocks, globalSize, weights);
  }
  /// Aligned ORB (coarsenable): see the aligned init() below.
  BlockDecomposer(std::size_t numBlocks, IVec<Dim> globalSize, const IVec<Dim>& align) {
    init(numBlocks, globalSize, align);
  }

  /// Build the decomposition of a `globalSize` cell grid into `numBlocks` blocks (equal cell
  /// count).
  void init(std::size_t numBlocks, IVec<Dim> globalSize) {
    align_ = IVec<Dim>{};
    for (int i = 0; i < Dim; ++i)
      align_[i] = 1;
    initImpl(numBlocks, globalSize, nullptr);
  }

  /// Aligned ORB: force every split position (and hence every block origin/size) on axis k to be a
  /// multiple of `align[k]`. This is what makes a decomposition safely COARSENABLE — a geometric
  /// multigrid can then derive each coarse level by `coarsened()` (halving in place) and every level
  /// nests, so restrict/prolong stay purely local. Set `align[k] = 2^(levels axis k coarsens)`.
  /// `align[k] == 1` is the classic unaligned split. `globalSize[k]` must be a multiple of `align[k]`.
  void init(std::size_t numBlocks, IVec<Dim> globalSize, const IVec<Dim>& align) {
    align_ = align;
    for (int i = 0; i < Dim; ++i)
      if (align_[i] < 1)
        align_[i] = 1;
    initImpl(numBlocks, globalSize, nullptr);
  }

  /// Anisotropic-cell aligned ORB: `cellExtent[k]` is how many underlying FINE cells one cell of
  /// this grid spans along axis k. It affects ONLY the choice of split axis, which then compares
  /// `size[k]*cellExtent[k]` (the physical extent) instead of the raw cell count. Required when
  /// decomposing a grid whose axes were coarsened by DIFFERENT factors — there equal cell counts do
  /// not mean equal extents, and comparing counts makes the ORB bisect the wrong axis (which is how
  /// a coarse-first partition ends up cutting a direction the fine grid would never have cut).
  /// All ones (the default) reproduces the isotropic ORB exactly.
  void init(std::size_t numBlocks, IVec<Dim> globalSize, const IVec<Dim>& align,
            const IVec<Dim>& cellExtent) {
    cellExtent_ = cellExtent;
    init(numBlocks, globalSize, align);
  }

  /// Weighted ORB: balance the *total weight* per block instead of the cell count. `weights` is a
  /// per-cell weight array over the global grid (size == product(globalSize), x-fastest). Each
  /// split is placed on the integer cell boundary whose cumulative weight is closest to the
  /// sub-block's target fraction. Reduces bit-exactly to the unweighted init() when all weights are
  /// equal.
  void init(std::size_t numBlocks, IVec<Dim> globalSize, const std::vector<Real>& weights) {
    assert(weights.size() == static_cast<std::size_t>([&] {
             Index v = 1;
             for (int i = 0; i < Dim; ++i)
               v *= globalSize[i];
             return v;
           }()) &&
           "weights array must cover the global grid (x-fastest)");
    align_ = IVec<Dim>{};
    for (int i = 0; i < Dim; ++i)
      align_[i] = 1;
    initImpl(numBlocks, globalSize, &weights);
  }

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

  /// Flatten the implicit ORB tree into two parallel arrays for a device-callable ownerOf: for node
  /// `i`, `splitDim[i]` is the split axis (−1 ⇒ leaf), `splitVal[i]` the split coordinate
  /// (internal) or the block index (leaf); children sit at 2i+1 / 2i+2. Uploaded once by
  /// ParticleMigratorView so the per-particle owner lookup runs on the device (D1). Mirrors
  /// `ownerOf` exactly.
  void flattenTree(std::vector<int>& splitDim, std::vector<Index>& splitVal) const {
    splitDim.resize(tree_.size());
    splitVal.resize(tree_.size());
    for (std::size_t i = 0; i < tree_.size(); ++i) {
      splitDim[i] = tree_[i].splitDim;
      splitVal[i] = tree_[i].splitValue;
    }
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

  /// Derive the NESTED coarse decomposition: each block, split value and the global size divided by
  /// `ratio` per axis (ratio[k] is 1 or the integer coarsening factor). The tree shape and leaf order
  /// are preserved, so rank r's coarse block is exactly rank r's fine block coarsened in place — the
  /// invariant a geometric-multigrid restrict/prolong relies on (coarse-local i ↔ fine-local ratio*i).
  /// Exact iff this decomposition was built aligned to a multiple of `ratio` on each coarsened axis
  /// (see the aligned `init`); asserts divisibility in debug builds.
  BlockDecomposer<Dim> coarsened(const IVec<Dim>& ratio) const {
    BlockDecomposer<Dim> c;
    for (int k = 0; k < Dim; ++k) {
      assert(ratio[k] >= 1 && globalSize_[k] % ratio[k] == 0 &&
             "coarsened(): global size not divisible by ratio (build the decomposition aligned)");
      c.globalSize_[k] = globalSize_[k] / ratio[k];
      c.align_[k] = align_[k] / ratio[k] >= 1 ? align_[k] / ratio[k] : 1;
    }
    c.tree_ = tree_;
    for (auto& nd : c.tree_)
      if (nd.splitDim != -1) {
        assert(nd.splitValue % ratio[nd.splitDim] == 0 &&
               "coarsened(): split not divisible by ratio (build the decomposition aligned)");
        nd.splitValue /= ratio[nd.splitDim];
      }
    c.origins_.resize(origins_.size());
    c.sizes_.resize(sizes_.size());
    for (std::size_t b = 0; b < origins_.size(); ++b)
      for (int k = 0; k < Dim; ++k) {
        assert(origins_[b][k] % ratio[k] == 0 && sizes_[b][k] % ratio[k] == 0 &&
               "coarsened(): block origin/size not divisible by ratio (build aligned)");
        c.origins_[b][k] = origins_[b][k] / ratio[k];
        c.sizes_[b][k] = sizes_[b][k] / ratio[k];
      }
    return c;
  }

  /// Derive the NESTED fine decomposition: the exact inverse of `coarsened()` — global size, split
  /// values, block origins and block sizes each MULTIPLIED by `ratio` per axis.
  ///
  /// This is the "decompose coarse, refine upward" route to a multigrid-safe partition, and it is
  /// strictly stronger than building the fine ORB with an alignment. Aligning the fine ORB decides
  /// the split *and then snaps it*, so a balanced split can be rounded into an unbalanced one (the
  /// pathological case: 96|96 snapped to 128|64, a 2:1 cascade). Refining upward instead lets the
  /// ORB balance on the coarse grid, where one coarse cell IS the alignment quantum — so the fine
  /// blocks are multiples of `ratio` by construction, `coarsened()` nests exactly for log2(ratio)
  /// levels, and the load balance is the best achievable at that granularity.
  ///
  /// Caller's responsibility: `ratio` should be the coarsening the hierarchy will actually perform
  /// (2^levels per axis, bounded by that axis's factors of two), and the coarse grid must hold at
  /// least `numBlocks` cells.
  BlockDecomposer<Dim> refined(const IVec<Dim>& ratio) const {
    BlockDecomposer<Dim> f;
    for (int k = 0; k < Dim; ++k) {
      assert(ratio[k] >= 1 && "refined(): ratio must be >= 1");
      f.globalSize_[k] = globalSize_[k] * ratio[k];
      f.align_[k] = align_[k] * ratio[k];
      f.cellExtent_[k] = 1;  // the refined grid's cells are the fine cells
    }
    f.tree_ = tree_;
    for (auto& nd : f.tree_)
      if (nd.splitDim != -1)
        nd.splitValue *= ratio[nd.splitDim];
    f.origins_.resize(origins_.size());
    f.sizes_.resize(sizes_.size());
    for (std::size_t b = 0; b < origins_.size(); ++b)
      for (int k = 0; k < Dim; ++k) {
        f.origins_[b][k] = origins_[b][k] * ratio[k];
        f.sizes_[b][k] = sizes_[b][k] * ratio[k];
      }
    return f;
  }

 private:
  struct TreeNode {
    int splitDim = -1;     ///< -1 for a leaf
    Index splitValue = 0;  ///< split coordinate (internal) or leaf/block index (leaf)
  };

  /// Shared decomposition driver. `weights == nullptr` ⇒ equal-cell-count split (the classic ORB);
  /// otherwise the split position balances cumulative weight.
  void initImpl(std::size_t numBlocks, IVec<Dim> globalSize, const std::vector<Real>* weights);

  /// Number of cells along `kLargest` that fall in the left child when splitting `box` so the left
  /// child receives `numSub` of `numTotal` sub-blocks. Returns a value in [1, size-1] for a
  /// splittable axis. `weights == nullptr` reproduces the classic proportional-to-count formula
  /// exactly.
  Index splitPosition(const IVec<Dim>& origin, const IVec<Dim>& size, int kLargest,
                      std::size_t numSub, std::size_t numTotal,
                      const std::vector<Real>* weights) const;

  IVec<Dim> globalSize_{};
  IVec<Dim> align_{};  ///< per-axis split alignment (1 = unaligned); see the aligned init / coarsened
  IVec<Dim> cellExtent_{};  ///< fine cells per cell of THIS grid, per axis; split-axis choice only
  std::vector<IVec<Dim>> origins_;
  std::vector<IVec<Dim>> sizes_;
  std::vector<TreeNode> tree_;
};

template <int Dim>
void BlockDecomposer<Dim>::initImpl(std::size_t numBlocks, IVec<Dim> globalSize,
                                    const std::vector<Real>* weights) {
  globalSize_ = globalSize;
  for (int k = 0; k < Dim; ++k)
    if (cellExtent_[k] <= 0)
      cellExtent_[k] = 1;
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
      // Split along the largest axis. The split position balances either the sub-block cell count
      // (unweighted) or the cumulative weight (weighted) of the two children.
      // Compare PHYSICAL extents: size*cellExtent. cellExtent is all-ones for an ordinary grid, so
      // this is the plain largest-cell-count test unless an anisotropically coarsened grid says
      // otherwise (see the cellExtent init).
      int kLargest = 0;
      for (int k = 1; k < Dim; ++k) {
        if (cur.size[k] * cellExtent_[k] > cur.size[kLargest] * cellExtent_[kLargest])
          kLargest = k;
      }
      std::size_t numSub = cur.numSub / 2;
      Index szSub = splitPosition(cur.origin, cur.size, kLargest, numSub, cur.numSub, weights);

      // Snap the split to a multiple of align_[kLargest] so the global split position
      // (cur.origin[kLargest] + szSub, with cur.origin already aligned) is a multiple too — the
      // precondition for coarsened() to divide cleanly. Skip if the box is too small to split aligned.
      const Index a = align_[kLargest];
      if (a > 1 && cur.size[kLargest] >= 2 * a) {
        szSub = ((szSub + a / 2) / a) * a;
        if (szSub < a)
          szSub = a;
        if (szSub > cur.size[kLargest] - a)
          szSub = cur.size[kLargest] - a;
      }

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

template <int Dim>
Index BlockDecomposer<Dim>::splitPosition(const IVec<Dim>& origin, const IVec<Dim>& size,
                                          int kLargest, std::size_t numSub, std::size_t numTotal,
                                          const std::vector<Real>* weights) const {
  const Index n = size[kLargest];

  // Unweighted (and the degenerate non-splittable axis): the classic proportional-to-count split.
  // Kept as the exact same expression so the unweighted API is byte-for-byte unchanged.
  if (weights == nullptr || n <= 1) {
    return static_cast<Index>(std::round(static_cast<double>(n) * static_cast<double>(numSub) /
                                         static_cast<double>(numTotal)));
  }

  // Weighted: accumulate the weight of each slab (fixed kLargest-coordinate) within the box, then
  // pick the boundary whose cumulative weight is closest to the target fraction of the total.
  std::vector<double> slab(static_cast<std::size_t>(n), 0.0);
  IVec<Dim> bgn = origin, end{};
  for (int i = 0; i < Dim; ++i)
    end[i] = origin[i] + size[i];
  forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& g) {
    slab[static_cast<std::size_t>(g[kLargest] - origin[kLargest])] += (*weights)[linearGlobal(g)];
  });

  double total = 0.0;
  for (double w : slab)
    total += w;
  const double target = total * static_cast<double>(numSub) / static_cast<double>(numTotal);

  // Search boundaries in [1, n-1] (non-empty children). Ties resolve to the larger boundary, which
  // mirrors std::round's half-away-from-zero rule so equal weights reproduce the unweighted split.
  double cum = 0.0;
  double bestDist = std::numeric_limits<double>::max();
  Index best = 1;
  for (Index s = 1; s < n; ++s) {
    cum += slab[static_cast<std::size_t>(s - 1)];
    const double dist = std::abs(cum - target);
    if (dist <= bestDist) {
      bestDist = dist;
      best = s;
    }
  }
  return best;
}

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_DECOMP_BLOCK_DECOMPOSER_HPP

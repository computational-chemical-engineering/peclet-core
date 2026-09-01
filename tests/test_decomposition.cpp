// Serial correctness of the ORB block decomposition:
//  - exactly `numBlocks` leaves are produced,
//  - the blocks tile the global grid exactly (every cell owned once, counts == block volumes),
//  - ownerOf() agrees with the block geometry.
// Plus the weighted ORB variant:
//  - equal weights reproduce the unweighted decomposition bit-for-bit,
//  - skewed weights balance the per-block total weight far better than equal-cell-count ORB.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;

template <int Dim>
Index volume(const IVec<Dim>& s) {
  Index v = 1;
  for (int i = 0; i < Dim; ++i)
    v *= s[i];
  return v;
}

template <int Dim>
void checkCase(IVec<Dim> globalSize, std::size_t numBlocks) {
  BlockDecomposer<Dim> dec(numBlocks, globalSize);

  // Exactly numBlocks leaves.
  PECLET_CORE_CHECK_EQ(static_cast<Index>(dec.numBlocks()), static_cast<Index>(numBlocks));

  // Block volumes sum to the global volume (no gaps, allowing for rounding distribution).
  Index total = volume<Dim>(globalSize);
  Index summed = 0;
  for (std::size_t b = 0; b < dec.numBlocks(); ++b)
    summed += volume<Dim>(dec.sizes()[b]);
  PECLET_CORE_CHECK_EQ(summed, total);

  // Every global cell is owned by exactly one block, and ownerOf agrees with that block's region.
  std::vector<Index> count(dec.numBlocks(), 0);
  IVec<Dim> bgn{}, end{};
  for (int i = 0; i < Dim; ++i) {
    bgn[i] = 0;
    end[i] = globalSize[i];
  }
  forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& g) {
    int owner = dec.ownerOf(g);
    PECLET_CORE_CHECK(owner >= 0 && owner < static_cast<int>(dec.numBlocks()));
    const auto& o = dec.origins()[owner];
    const auto& s = dec.sizes()[owner];
    for (int i = 0; i < Dim; ++i) {
      PECLET_CORE_CHECK(g[i] >= o[i] && g[i] < o[i] + s[i]);
    }
    ++count[owner];
  });
  for (std::size_t b = 0; b < dec.numBlocks(); ++b) {
    PECLET_CORE_CHECK_EQ(count[b], volume<Dim>(dec.sizes()[b]));
  }
}

// Total weight per block for a given weight field (x-fastest over the global grid).
template <int Dim>
std::vector<double> blockWeights(const BlockDecomposer<Dim>& dec,
                                 const std::vector<Real>& weights) {
  std::vector<double> w(dec.numBlocks(), 0.0);
  for (std::size_t b = 0; b < dec.numBlocks(); ++b) {
    const auto& o = dec.origins()[b];
    const auto& s = dec.sizes()[b];
    IVec<Dim> bgn = o, end{};
    for (int i = 0; i < Dim; ++i)
      end[i] = o[i] + s[i];
    forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& g) { w[b] += weights[dec.linearGlobal(g)]; });
  }
  return w;
}

template <int Dim>
double maxOverMin(const std::vector<double>& v) {
  double lo = v[0], hi = v[0];
  for (double x : v) {
    lo = std::min(lo, x);
    hi = std::max(hi, x);
  }
  return hi / lo;
}

// Equal weights must reproduce the unweighted decomposition exactly (same blocks, same tree).
template <int Dim>
void checkEqualWeightsBitExact(IVec<Dim> globalSize, std::size_t numBlocks) {
  Index total = volume<Dim>(globalSize);
  std::vector<Real> uniform(static_cast<std::size_t>(total), 1.0);

  BlockDecomposer<Dim> dec(numBlocks, globalSize);
  BlockDecomposer<Dim> decW(numBlocks, globalSize, uniform);

  PECLET_CORE_CHECK_EQ(static_cast<Index>(decW.numBlocks()), static_cast<Index>(dec.numBlocks()));
  for (std::size_t b = 0; b < dec.numBlocks(); ++b) {
    for (int i = 0; i < Dim; ++i) {
      PECLET_CORE_CHECK_EQ(decW.origins()[b][i], dec.origins()[b][i]);
      PECLET_CORE_CHECK_EQ(decW.sizes()[b][i], dec.sizes()[b][i]);
    }
  }
  // ownerOf agrees at every cell too (covers the implicit tree, not just leaf order).
  IVec<Dim> bgn{}, end{};
  for (int i = 0; i < Dim; ++i)
    end[i] = globalSize[i];
  forEachInBox<Dim>(
      bgn, end, [&](const IVec<Dim>& g) { PECLET_CORE_CHECK_EQ(decW.ownerOf(g), dec.ownerOf(g)); });
}

// A smooth, monotone weight gradient that equal-cell-count ORB balances poorly: the weighted ORB
// must drive the per-block total weight far closer to equal (and stay a valid tiling).
template <int Dim>
void checkSkewedBalances(IVec<Dim> globalSize, std::size_t numBlocks) {
  Index total = volume<Dim>(globalSize);
  std::vector<Real> weights(static_cast<std::size_t>(total), 0.0);
  IVec<Dim> bgn{}, end{};
  for (int i = 0; i < Dim; ++i)
    end[i] = globalSize[i];

  // weight grows steeply across the diagonal so every axis carries a gradient to balance.
  BlockDecomposer<Dim> probe(1, globalSize);  // just for linearGlobal
  forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& g) {
    double frac = 0.0;
    for (int i = 0; i < Dim; ++i)
      frac += static_cast<double>(g[i]) / globalSize[i];
    weights[probe.linearGlobal(g)] = 1.0 + 50.0 * frac;
  });

  BlockDecomposer<Dim> decUW(numBlocks, globalSize);
  BlockDecomposer<Dim> decW(numBlocks, globalSize, weights);

  // Weighted ORB is still an exact tiling of the grid (cell-aligned, no gaps/overlap).
  Index summed = 0;
  std::vector<Index> count(decW.numBlocks(), 0);
  for (std::size_t b = 0; b < decW.numBlocks(); ++b)
    summed += volume<Dim>(decW.sizes()[b]);
  PECLET_CORE_CHECK_EQ(summed, total);
  forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& g) {
    int owner = decW.ownerOf(g);
    PECLET_CORE_CHECK(owner >= 0 && owner < static_cast<int>(decW.numBlocks()));
    const auto& o = decW.origins()[owner];
    const auto& s = decW.sizes()[owner];
    for (int i = 0; i < Dim; ++i)
      PECLET_CORE_CHECK(g[i] >= o[i] && g[i] < o[i] + s[i]);
    ++count[owner];
  });
  for (std::size_t b = 0; b < decW.numBlocks(); ++b)
    PECLET_CORE_CHECK_EQ(count[b], volume<Dim>(decW.sizes()[b]));

  double ratioUW = maxOverMin<Dim>(blockWeights<Dim>(decUW, weights));
  double ratioW = maxOverMin<Dim>(blockWeights<Dim>(decW, weights));
  // The weighted split should be much more even, and meaningfully better than equal-cell ORB.
  PECLET_CORE_CHECK(ratioW < 1.25);
  PECLET_CORE_CHECK(ratioW < ratioUW);
}

// Aligned ORB + coarsened(): every split/origin/size on axis k is a multiple of align[k], and the
// coarse decomposition NESTS in the fine one (coarse block == fine block coarsened in place, same
// leaf/rank order) — the invariant a geometric multigrid's local restrict/prolong relies on.
template <int Dim>
void checkAlignedCoarsen(IVec<Dim> globalSize, std::size_t numBlocks, IVec<Dim> align,
                         IVec<Dim> ratio) {
  BlockDecomposer<Dim> dec(numBlocks, globalSize, align);
  // Every block origin & size is aligned.
  for (std::size_t b = 0; b < dec.numBlocks(); ++b)
    for (int k = 0; k < Dim; ++k) {
      PECLET_CORE_CHECK_EQ(dec.origins()[b][k] % align[k], static_cast<Index>(0));
      PECLET_CORE_CHECK_EQ(dec.sizes()[b][k] % align[k], static_cast<Index>(0));
    }
  // Still a valid tiling (no gaps/overlaps).
  Index summed = 0, total = 1;
  for (int k = 0; k < Dim; ++k)
    total *= globalSize[k];
  for (std::size_t b = 0; b < dec.numBlocks(); ++b)
    summed += volume<Dim>(dec.sizes()[b]);
  PECLET_CORE_CHECK_EQ(summed, total);

  // Coarsen and check nesting: coarse block r == fine block r coarsened in place.
  BlockDecomposer<Dim> c = dec.coarsened(ratio);
  PECLET_CORE_CHECK_EQ(static_cast<Index>(c.numBlocks()), static_cast<Index>(dec.numBlocks()));
  for (std::size_t b = 0; b < dec.numBlocks(); ++b)
    for (int k = 0; k < Dim; ++k) {
      PECLET_CORE_CHECK_EQ(c.origins()[b][k], dec.origins()[b][k] / ratio[k]);
      PECLET_CORE_CHECK_EQ(c.sizes()[b][k], dec.sizes()[b][k] / ratio[k]);
      PECLET_CORE_CHECK_EQ(c.globalSize()[k], globalSize[k] / ratio[k]);
    }
}

// Coarse-level telescoping primitive: agglomerated(depth) merges ORB siblings by truncating the
// bisection tree. For every depth from the full tree down to the root it must (i) tile the grid
// exactly with ownerOf() agreeing, (ii) map every old block into exactly one new block whose box is
// the exact union of its members (volumes add up, members are a CONTIGUOUS index range, root is the
// lowest member), (iii) be the identity at full depth and one whole-grid block at depth 0, and
// (iv) nest with coarsened() once the merged blocks are even -- the multigrid use.
template <int Dim>
void checkAgglomerated(IVec<Dim> globalSize, std::size_t numBlocks) {
  BlockDecomposer<Dim> dec(numBlocks, globalSize);
  const int depth = dec.treeDepth();
  PECLET_CORE_CHECK(depth >= 0);
  std::size_t prevBlocks = dec.numBlocks() + 1;
  for (int d = depth; d >= 0; --d) {
    std::vector<int> groupOf, rootOf;
    BlockDecomposer<Dim> a = dec.agglomerated(d, &groupOf, &rootOf);
    // (i) tiling + ownerOf
    Index summed = 0;
    for (std::size_t b = 0; b < a.numBlocks(); ++b)
      summed += volume<Dim>(a.sizes()[b]);
    PECLET_CORE_CHECK_EQ(summed, volume<Dim>(globalSize));
    std::vector<Index> count(a.numBlocks(), 0);
    IVec<Dim> bgn{}, end = globalSize;
    forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& g) {
      const int owner = a.ownerOf(g);
      PECLET_CORE_CHECK(owner >= 0 && owner < static_cast<int>(a.numBlocks()));
      const auto& o = a.origins()[owner];
      const auto& sz = a.sizes()[owner];
      for (int i = 0; i < Dim; ++i)
        PECLET_CORE_CHECK(g[i] >= o[i] && g[i] < o[i] + sz[i]);
      ++count[owner];
    });
    for (std::size_t b = 0; b < a.numBlocks(); ++b)
      PECLET_CORE_CHECK_EQ(count[b], volume<Dim>(a.sizes()[b]));
    // (ii) group map: exact union, contiguous members, lowest-member root
    PECLET_CORE_CHECK_EQ(static_cast<Index>(groupOf.size()), static_cast<Index>(dec.numBlocks()));
    PECLET_CORE_CHECK_EQ(static_cast<Index>(rootOf.size()), static_cast<Index>(a.numBlocks()));
    std::vector<Index> memberVol(a.numBlocks(), 0);
    std::vector<int> lo(a.numBlocks(), 1 << 30), hi(a.numBlocks(), -1), cnt(a.numBlocks(), 0);
    for (std::size_t b = 0; b < dec.numBlocks(); ++b) {
      const int g = groupOf[b];
      PECLET_CORE_CHECK(g >= 0 && g < static_cast<int>(a.numBlocks()));
      memberVol[g] += volume<Dim>(dec.sizes()[b]);
      for (int i = 0; i < Dim; ++i) {  // member inside its group box
        PECLET_CORE_CHECK(dec.origins()[b][i] >= a.origins()[g][i]);
        PECLET_CORE_CHECK(dec.origins()[b][i] + dec.sizes()[b][i] <=
                          a.origins()[g][i] + a.sizes()[g][i]);
      }
      lo[g] = std::min(lo[g], static_cast<int>(b));
      hi[g] = std::max(hi[g], static_cast<int>(b));
      ++cnt[g];
    }
    for (std::size_t g = 0; g < a.numBlocks(); ++g) {
      PECLET_CORE_CHECK_EQ(memberVol[g], volume<Dim>(a.sizes()[g]));  // exact union
      PECLET_CORE_CHECK_EQ(static_cast<Index>(hi[g] - lo[g] + 1), static_cast<Index>(cnt[g]));  // contiguous
      PECLET_CORE_CHECK_EQ(static_cast<Index>(rootOf[g]), static_cast<Index>(lo[g]));
    }
    // monotone: merging never increases the block count
    PECLET_CORE_CHECK(a.numBlocks() < prevBlocks);
    prevBlocks = a.numBlocks() + (d == depth ? 1 : 0);
    if (d == depth) {  // (iii) identity
      PECLET_CORE_CHECK_EQ(static_cast<Index>(a.numBlocks()), static_cast<Index>(dec.numBlocks()));
      for (std::size_t b = 0; b < dec.numBlocks(); ++b)
        for (int i = 0; i < Dim; ++i) {
          PECLET_CORE_CHECK_EQ(a.origins()[b][i], dec.origins()[b][i]);
          PECLET_CORE_CHECK_EQ(a.sizes()[b][i], dec.sizes()[b][i]);
        }
    }
    if (d == 0) {
      PECLET_CORE_CHECK_EQ(static_cast<Index>(a.numBlocks()), static_cast<Index>(1));
      for (int i = 0; i < Dim; ++i)
        PECLET_CORE_CHECK_EQ(a.sizes()[0][i], globalSize[i]);
    }
  }
}

// (iv) The multigrid scenario end to end: 8 blocks of 12^3 halve twice in place to 3^3 (odd, stuck);
// the whole-grid agglomeration restores even blocks and coarsened() proceeds to 3^3 on one block.
void checkAgglomeratedUnblocksCoarsening() {
  BlockDecomposer<3> dec(8, {24, 24, 24});
  BlockDecomposer<3> c2 = dec.coarsened({2, 2, 2}).coarsened({2, 2, 2});  // global 6^3, blocks 3^3
  bool anyOdd = false;
  for (std::size_t b = 0; b < c2.numBlocks(); ++b)
    for (int k = 0; k < 3; ++k)
      anyOdd = anyOdd || (c2.sizes()[b][k] % 2) || (c2.origins()[b][k] % 2);
  PECLET_CORE_CHECK(anyOdd);  // the in-place path is genuinely blocked here
  BlockDecomposer<3> one = c2.agglomerated(0);
  PECLET_CORE_CHECK_EQ(static_cast<Index>(one.numBlocks()), static_cast<Index>(1));
  BlockDecomposer<3> c3 = one.coarsened({2, 2, 2});
  PECLET_CORE_CHECK_EQ(c3.sizes()[0][0], static_cast<Index>(3));
  PECLET_CORE_CHECK_EQ(c3.globalSize()[0], static_cast<Index>(3));
  // merging one tree level: 4 blocks; the merged axis is even, the others still odd
  std::vector<int> groupOf;
  BlockDecomposer<3> half = c2.agglomerated(c2.treeDepth() - 1, &groupOf);
  PECLET_CORE_CHECK_EQ(static_cast<Index>(half.numBlocks()), static_cast<Index>(4));
  int evenAxes = 0;
  for (int k = 0; k < 3; ++k) {
    bool even = true;
    for (std::size_t b = 0; b < half.numBlocks(); ++b)
      even = even && !(half.sizes()[b][k] % 2) && !(half.origins()[b][k] % 2);
    evenAxes += even;
  }
  PECLET_CORE_CHECK_EQ(static_cast<Index>(evenAxes), static_cast<Index>(1));
}

int main() {
  // A spread of dimensions, grid sizes (incl. non-powers-of-two) and block counts.
  checkCase<1>({100}, 1);
  checkCase<1>({100}, 7);
  checkCase<2>({64, 64}, 4);
  checkCase<2>({60, 40}, 6);
  checkCase<2>({97, 31}, 5);
  checkCase<3>({32, 32, 32}, 8);
  checkCase<3>({40, 24, 16}, 12);
  checkCase<3>({30, 30, 30}, 3);
  checkCase<3>({17, 19, 23}, 16);

  // Weighted ORB: equal weights must be bit-identical to the unweighted decomposition.
  checkEqualWeightsBitExact<1>({100}, 7);
  checkEqualWeightsBitExact<2>({64, 64}, 4);
  checkEqualWeightsBitExact<2>({97, 31}, 5);
  checkEqualWeightsBitExact<3>({32, 32, 32}, 8);
  checkEqualWeightsBitExact<3>({40, 24, 16}, 12);
  checkEqualWeightsBitExact<3>({17, 19, 23}, 16);

  // Weighted ORB: a skewed weight field is balanced far better than equal-cell ORB.
  checkSkewedBalances<1>({256}, 8);
  checkSkewedBalances<2>({96, 96}, 8);
  checkSkewedBalances<2>({128, 96}, 16);
  checkSkewedBalances<3>({48, 48, 48}, 8);
  checkSkewedBalances<3>({64, 48, 40}, 16);

  // Aligned ORB + nested coarsening (the distributed-multigrid nesting invariant). The channel-like
  // case (x coarsens, z odd -> z never coarsens) is the one whose non-nesting caused the flow MG OOB.
  checkAlignedCoarsen<3>({1508, 240, 503}, 2, {4, 16, 1}, {2, 2, 1});   // production channel structure
  checkAlignedCoarsen<3>({1508, 240, 503}, 4, {4, 16, 1}, {2, 2, 1});
  checkAlignedCoarsen<3>({132, 32, 131}, 2, {4, 16, 1}, {2, 2, 1});     // the local repro (axis-flip)
  checkAlignedCoarsen<3>({132, 32, 131}, 4, {4, 16, 1}, {2, 2, 1});
  checkAlignedCoarsen<3>({64, 64, 64}, 8, {16, 16, 16}, {2, 2, 2});     // cubic, all axes coarsen

  // Coarse-level telescoping (agglomerated): tiling, exact-union groups, identity/whole-grid ends,
  // and the multigrid unblocking scenario. Non-powers-of-two are the interesting cases.
  checkAgglomerated<2>({64, 64}, 4);
  checkAgglomerated<2>({97, 31}, 5);
  checkAgglomerated<3>({32, 32, 32}, 8);
  checkAgglomerated<3>({40, 24, 16}, 12);
  checkAgglomerated<3>({48, 48, 48}, 24);
  checkAgglomerated<3>({17, 19, 23}, 16);
  checkAgglomeratedUnblocksCoarsening();

  PECLET_CORE_RETURN_TEST_RESULT();
}

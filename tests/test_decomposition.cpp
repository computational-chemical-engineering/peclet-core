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

#include "test_util.hpp"
#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"

using namespace tpx;
using tpx::decomp::BlockDecomposer;

template <int Dim>
Index volume(const IVec<Dim>& s) {
  Index v = 1;
  for (int i = 0; i < Dim; ++i) v *= s[i];
  return v;
}

template <int Dim>
void checkCase(IVec<Dim> globalSize, std::size_t numBlocks) {
  BlockDecomposer<Dim> dec(numBlocks, globalSize);

  // Exactly numBlocks leaves.
  TPX_CHECK_EQ(static_cast<Index>(dec.numBlocks()), static_cast<Index>(numBlocks));

  // Block volumes sum to the global volume (no gaps, allowing for rounding distribution).
  Index total = volume<Dim>(globalSize);
  Index summed = 0;
  for (std::size_t b = 0; b < dec.numBlocks(); ++b) summed += volume<Dim>(dec.sizes()[b]);
  TPX_CHECK_EQ(summed, total);

  // Every global cell is owned by exactly one block, and ownerOf agrees with that block's region.
  std::vector<Index> count(dec.numBlocks(), 0);
  IVec<Dim> bgn{}, end{};
  for (int i = 0; i < Dim; ++i) {
    bgn[i] = 0;
    end[i] = globalSize[i];
  }
  forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& g) {
    int owner = dec.ownerOf(g);
    TPX_CHECK(owner >= 0 && owner < static_cast<int>(dec.numBlocks()));
    const auto& o = dec.origins()[owner];
    const auto& s = dec.sizes()[owner];
    for (int i = 0; i < Dim; ++i) {
      TPX_CHECK(g[i] >= o[i] && g[i] < o[i] + s[i]);
    }
    ++count[owner];
  });
  for (std::size_t b = 0; b < dec.numBlocks(); ++b) {
    TPX_CHECK_EQ(count[b], volume<Dim>(dec.sizes()[b]));
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
    for (int i = 0; i < Dim; ++i) end[i] = o[i] + s[i];
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

  TPX_CHECK_EQ(static_cast<Index>(decW.numBlocks()), static_cast<Index>(dec.numBlocks()));
  for (std::size_t b = 0; b < dec.numBlocks(); ++b) {
    for (int i = 0; i < Dim; ++i) {
      TPX_CHECK_EQ(decW.origins()[b][i], dec.origins()[b][i]);
      TPX_CHECK_EQ(decW.sizes()[b][i], dec.sizes()[b][i]);
    }
  }
  // ownerOf agrees at every cell too (covers the implicit tree, not just leaf order).
  IVec<Dim> bgn{}, end{};
  for (int i = 0; i < Dim; ++i) end[i] = globalSize[i];
  forEachInBox<Dim>(bgn, end,
                    [&](const IVec<Dim>& g) { TPX_CHECK_EQ(decW.ownerOf(g), dec.ownerOf(g)); });
}

// A smooth, monotone weight gradient that equal-cell-count ORB balances poorly: the weighted ORB
// must drive the per-block total weight far closer to equal (and stay a valid tiling).
template <int Dim>
void checkSkewedBalances(IVec<Dim> globalSize, std::size_t numBlocks) {
  Index total = volume<Dim>(globalSize);
  std::vector<Real> weights(static_cast<std::size_t>(total), 0.0);
  IVec<Dim> bgn{}, end{};
  for (int i = 0; i < Dim; ++i) end[i] = globalSize[i];

  // weight grows steeply across the diagonal so every axis carries a gradient to balance.
  BlockDecomposer<Dim> probe(1, globalSize);  // just for linearGlobal
  forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& g) {
    double frac = 0.0;
    for (int i = 0; i < Dim; ++i) frac += static_cast<double>(g[i]) / globalSize[i];
    weights[probe.linearGlobal(g)] = 1.0 + 50.0 * frac;
  });

  BlockDecomposer<Dim> decUW(numBlocks, globalSize);
  BlockDecomposer<Dim> decW(numBlocks, globalSize, weights);

  // Weighted ORB is still an exact tiling of the grid (cell-aligned, no gaps/overlap).
  Index summed = 0;
  std::vector<Index> count(decW.numBlocks(), 0);
  for (std::size_t b = 0; b < decW.numBlocks(); ++b) summed += volume<Dim>(decW.sizes()[b]);
  TPX_CHECK_EQ(summed, total);
  forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& g) {
    int owner = decW.ownerOf(g);
    TPX_CHECK(owner >= 0 && owner < static_cast<int>(decW.numBlocks()));
    const auto& o = decW.origins()[owner];
    const auto& s = decW.sizes()[owner];
    for (int i = 0; i < Dim; ++i) TPX_CHECK(g[i] >= o[i] && g[i] < o[i] + s[i]);
    ++count[owner];
  });
  for (std::size_t b = 0; b < decW.numBlocks(); ++b)
    TPX_CHECK_EQ(count[b], volume<Dim>(decW.sizes()[b]));

  double ratioUW = maxOverMin<Dim>(blockWeights<Dim>(decUW, weights));
  double ratioW = maxOverMin<Dim>(blockWeights<Dim>(decW, weights));
  // The weighted split should be much more even, and meaningfully better than equal-cell ORB.
  TPX_CHECK(ratioW < 1.25);
  TPX_CHECK(ratioW < ratioUW);
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

  TPX_RETURN_TEST_RESULT();
}

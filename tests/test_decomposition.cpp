// Serial correctness of the ORB block decomposition:
//  - exactly `numBlocks` leaves are produced,
//  - the blocks tile the global grid exactly (every cell owned once, counts == block volumes),
//  - ownerOf() agrees with the block geometry.
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
  TPX_RETURN_TEST_RESULT();
}

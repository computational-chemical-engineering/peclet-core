// The opt-in balance-aware split-axis rule of the unweighted ORB (SplitAxis::Balanced, flow
// doc/vof_distributed_partition_design.md §5.1, §7 WO-2). Pure function, no MPI. Checks, over a
// census of grids x block counts x three construction forms (the aligned ORB with flow's
// coarsening alignment, the plain ORB, and the coarse-first ORB at depths 4 and 5):
//  - the 5-argument init with SplitAxis::Longest is the 4-argument init (origins, sizes, tree);
//  - (P-i)   whenever the Longest partition is exactly balanced, Balanced builds the identical
//  tree;
//  - (P-ii)  two Balanced builds are identical (pure, deterministic);
//  - (P-iii) Balanced never cuts an aligned axis unaligned where the box allows an aligned cut,
//            and cuts an axis too short for an aligned cut only when it is the Longest axis;
//  - the Balanced blocks tile the grid;
//  - a later default init on the same object is the Longest rule again;
// and exactness on 128x96x64 (align 16) at 3, 12, 24, 32, 96 blocks with four in-place halvings.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::decomp::SplitAxis;
using Dec = BlockDecomposer<3>;

namespace {

bool can(Index d) {
  return d % 2 == 0 && d / 2 >= 2;
}

// flow's CutcellMG::coarsenAlignment: 2^(halvings until odd or < 4), capped at 16
IVec<3> alignOf(const IVec<3>& G) {
  IVec<3> a{1, 1, 1};
  for (int k = 0; k < 3; ++k) {
    Index g = G[k];
    while (can(g)) {
      a[k] *= 2;
      g /= 2;
    }
    a[k] = std::min<Index>(a[k], 16);
  }
  return a;
}
// flow's CutcellMG::refineFactor: the per-axis factor of L-1 halvings
IVec<3> refineFactor(const IVec<3>& G, int L) {
  IVec<3> r{1, 1, 1}, g = G;
  for (int l = 1; l < L; ++l)
    for (int k = 0; k < 3; ++k)
      if (can(g[k])) {
        r[k] *= 2;
        g[k] /= 2;
      }
  return r;
}

Index cells(const IVec<3>& s) {
  return s[0] * s[1] * s[2];
}
bool exact(const Dec& d) {
  Index lo = cells(d.sizes()[0]), hi = lo;
  for (const auto& s : d.sizes()) {
    lo = std::min(lo, cells(s));
    hi = std::max(hi, cells(s));
  }
  return lo == hi;
}
bool same(const Dec& a, const Dec& b) {
  if (a.numBlocks() != b.numBlocks() || a.origins() != b.origins() || a.sizes() != b.sizes())
    return false;
  std::vector<int> da, db;
  std::vector<Index> va, vb;
  a.flattenTree(da, va);
  b.flattenTree(db, vb);
  return da == db && va == vb;
}
int longestAxis(const IVec<3>& s, const IVec<3>& ext) {
  int k = 0;
  for (int j = 1; j < 3; ++j)
    if (s[j] * ext[j] > s[k] * ext[k])
      k = j;
  return k;
}
// (P-iii) by a walk of the flattened tree, reconstructing each node's box
bool alignedCutsOnly(const Dec& d, const IVec<3>& G, const IVec<3>& align, const IVec<3>& ext) {
  std::vector<int> dim;
  std::vector<Index> val;
  d.flattenTree(dim, val);
  struct Box {
    IVec<3> o, s;
  };
  std::vector<Box> box(dim.size());
  box[0] = Box{IVec<3>{0, 0, 0}, G};
  for (std::size_t i = 0; i < dim.size(); ++i) {
    if (dim[i] < 0)
      continue;
    const int k = dim[i];
    const Box b = box[i];
    const Index c = val[i] - b.o[k];
    if (align[k] > 1) {
      if (b.s[k] >= 2 * align[k] && c % align[k] != 0)
        return false;
      if (b.s[k] < 2 * align[k] && k != longestAxis(b.s, ext))
        return false;
    }
    if (2 * i + 2 < dim.size()) {
      box[2 * i + 1] = b;
      box[2 * i + 1].s[k] = c;
      box[2 * i + 2] = b;
      box[2 * i + 2].o[k] += c;
      box[2 * i + 2].s[k] -= c;
    }
  }
  return true;
}
bool tiles(const Dec& d, const IVec<3>& G, std::size_t np) {
  Index v = 0;
  for (const auto& s : d.sizes())
    v += cells(s);
  return d.numBlocks() == np && v == cells(G);
}
// four in-place halvings on every axis that can coarsen
bool nestsFour(const Dec& d0) {
  Dec d = d0;
  for (int l = 0; l < 4; ++l) {
    IVec<3> r{1, 1, 1};
    for (int k = 0; k < 3; ++k)
      if (can(d.globalSize()[k]))
        r[k] = 2;
    for (std::size_t b = 0; b < d.numBlocks(); ++b)
      for (int k = 0; k < 3; ++k)
        if (r[k] == 2 && (d.origins()[b][k] % 2 || d.sizes()[b][k] % 2))
          return false;
    d = d.coarsened(r);
  }
  return true;
}

int nCompared = 0, nExact = 0, nChanged = 0;

void checkForm(const char* form, const IVec<3>& G, std::size_t np, const IVec<3>& align,
               const IVec<3>& ext, const IVec<3>& fineG) {
  Dec L4, L5, B, B2;
  L4.init(np, G, align, ext);
  L5.init(np, G, align, ext, SplitAxis::Longest);
  B.init(np, G, align, ext, SplitAxis::Balanced);
  B2.init(np, G, align, ext, SplitAxis::Balanced);
  ++nCompared;
  const bool longestSame = same(L4, L5);
  const bool det = same(B, B2);
  const bool ex = exact(L4);
  const bool pi = !ex || same(B, L4);
  const bool piii = alignedCutsOnly(B, G, align, ext);
  const bool tl = tiles(B, G, np);
  nExact += ex;
  nChanged += !same(B, L4);
  if (!(longestSame && det && pi && piii && tl))
    std::printf(
        "FAIL %s %lldx%lldx%lld (fine %lldx%lldx%lld) np %zu: Longest5==init4 %d, P-ii %d, "
        "P-i %d, P-iii %d, tiles %d\n",
        form, (long long)G[0], (long long)G[1], (long long)G[2], (long long)fineG[0],
        (long long)fineG[1], (long long)fineG[2], np, longestSame, det, pi, piii, tl);
  PECLET_CORE_CHECK(longestSame);
  PECLET_CORE_CHECK(det);
  PECLET_CORE_CHECK(pi);
  PECLET_CORE_CHECK(piii);
  PECLET_CORE_CHECK(tl);
}

}  // namespace

int main() {
  const std::vector<IVec<3>> grids = {
      {128, 96, 64},   {48, 48, 48},    {64, 64, 64},    {96, 96, 96},    {128, 128, 128},
      {192, 192, 192}, {256, 128, 128}, {384, 288, 192}, {40, 40, 40},    {100, 50, 50},
      {768, 128, 256}, {480, 80, 160},  {30, 30, 30},    {1508, 240, 503}};
  const std::vector<std::size_t> nps = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 24, 32, 48, 96, 192};
  for (const auto& G : grids)
    for (std::size_t np : nps) {
      if ((Index)np > cells(G))
        continue;
      checkForm("aligned", G, np, alignOf(G), IVec<3>{1, 1, 1}, G);
      checkForm("plain", G, np, IVec<3>{1, 1, 1}, IVec<3>{1, 1, 1}, G);
      for (int L : {4, 5}) {
        const IVec<3> r = refineFactor(G, L);
        const IVec<3> Gc{G[0] / r[0], G[1] / r[1], G[2] / r[2]};
        if ((Index)np > cells(Gc))
          continue;
        checkForm(L == 4 ? "coarse-first L4" : "coarse-first L5", Gc, np, IVec<3>{1, 1, 1}, r, G);
      }
    }
  std::printf(
      "census: %d configurations, %d exact under Longest (P-i applies), %d changed by "
      "Balanced\n",
      nCompared, nExact, nChanged);

  // Exactness and nesting on the bubble column grid, aligned (align 16) and coarse-first L5.
  const IVec<3> G{128, 96, 64};
  for (std::size_t np : {3, 12, 24, 32, 96}) {
    Dec A, C;
    A.init(np, G, alignOf(G), IVec<3>{1, 1, 1}, SplitAxis::Balanced);
    const IVec<3> r = refineFactor(G, 5);
    C.init(np, IVec<3>{G[0] / r[0], G[1] / r[1], G[2] / r[2]}, IVec<3>{1, 1, 1}, r,
           SplitAxis::Balanced);
    C = C.refined(r);
    const bool ok = exact(A) && exact(C) && nestsFour(A) && nestsFour(C);
    std::printf(
        "128x96x64 np %zu: Balanced aligned exact %d nests 4 %d; coarse-first L5 exact %d "
        "nests 4 %d -> %s\n",
        np, exact(A), nestsFour(A), exact(C), nestsFour(C), ok ? "ok" : "FAIL");
    PECLET_CORE_CHECK(ok);
  }

  // A later default (Longest) init on the same object forgets the Balanced rule.
  {
    Dec d, ref;
    d.init(24, G, alignOf(G), IVec<3>{1, 1, 1}, SplitAxis::Balanced);
    d.init(24, G, alignOf(G));
    ref.init(24, G, alignOf(G));
    PECLET_CORE_CHECK(same(d, ref));
    d.init(24, G, alignOf(G), IVec<3>{1, 1, 1}, SplitAxis::Balanced);
    d.init(24, G);
    Dec ref2(24, G);
    PECLET_CORE_CHECK(same(d, ref2));
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

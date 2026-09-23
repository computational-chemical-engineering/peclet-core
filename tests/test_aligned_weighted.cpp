// The aligned weighted ORB `BlockDecomposer::init(numBlocks, globalSize, weights, align)` and its
// chooser `chooseAlignedWeighted` (peclet/core/decomp/block_decomposer.hpp;
// amr/docs/amr_mg_core_boundary.md §11.4, step S2a). Pure functions, no MPI; the rank-identity half
// of G-A3 is test_aligned_weighted_mpi.
//
//   G-A1  reduction: init(w, align = 1) == init(w) — origins, sizes and the ORB tree bitwise — over
//         50 weight fields x np in {1..8, 12, 24} x grids 32^3 / 48x32x16 / 96^3.
//   G-A2  nesting, a in {1, 2, 3}: every block origin, size and tree split a multiple of 2^a; the
//         partition IS the refined weighted ORB of the coarse weight sums (coarse-first, never
//         snap-after: coarsened(2^a) equals an independent weighted ORB of the test's own coarse
//         sums, and refines back to itself); for every level j < a the lift j -> j+1 is in place
//         under BOTH flow's per-axis rule and amr's all-axes rule (transcribed from
//         test_stage_target).
//   G-A3  chooser: a > 0 => imbalance (recomputed here) <= 1.05 and the partition is the aligned
//         init at 2^a; every larger feasible a fails the budget (or has an empty block) — "largest
//         a"; a == 0 => today's init(w) bitwise and EVERY feasible a >= 1 fails; forced fallbacks
//         (an odd axis, one dominant cell) return a == 0 bitwise; the caller's aMax caps a.
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "test_util.hpp"

using peclet::core::forEachInBox;
using peclet::core::Index;
using peclet::core::IVec;
using peclet::core::Real;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::decomp::chooseAlignedWeighted;
using Dec = BlockDecomposer<3>;

namespace {

// ---- the lift predicates, transcribed from test_stage_target (flow's and amr's rules)
// ------------
bool can(Index d) {
  return (d % 2 == 0) && (d / 2 >= 2);
}
bool evenOn(const Dec& d, int ax) {
  for (std::size_t b = 0; b < d.sizes().size(); ++b)
    if ((d.origins()[b][ax] % 2) || (d.sizes()[b][ax] % 2))
      return false;
  return true;
}
bool flowLiftable(const Dec& d) {  // flow: even on every axis the level grid can still coarsen
  for (int ax = 0; ax < 3; ++ax)
    if (can(d.globalSize()[ax]) && !evenOn(d, ax))
      return false;
  return true;
}
bool amrLiftable(const Dec& d) {  // amr (amr_mg_depth.md §6.2): even on EVERY axis
  for (int ax = 0; ax < 3; ++ax)
    if (!evenOn(d, ax))
      return false;
  return true;
}

bool sameDec(const Dec& a, const Dec& b) {
  if (a.origins() != b.origins() || a.sizes() != b.sizes() || !(a.globalSize() == b.globalSize()))
    return false;
  std::vector<int> da, db;
  std::vector<Index> va, vb;
  a.flattenTree(da, va);
  b.flattenTree(db, vb);
  return da == db && va == vb;
}

// ---- weight fields: 50 deterministic fields of five shapes, all positive ---------------------
struct Lcg {
  unsigned s;
  Real next() {  // [0, 1)
    s = s * 1664525u + 1013904223u;
    return static_cast<Real>(s >> 8) / static_cast<Real>(1u << 24);
  }
};

std::vector<Real> weightField(const IVec<3>& g, int field) {
  Lcg r{static_cast<unsigned>(12345 + 7919 * field)};
  std::vector<Real> w(static_cast<std::size_t>(g[0] * g[1] * g[2]));
  const int shape = field % 5;
  const Real bed = 0.2 + 0.4 * r.next(), tilt = 0.8 * r.next();
  const Real cx = r.next(), cy = r.next(), cz = r.next();
  std::size_t i = 0;
  for (Index z = 0; z < g[2]; ++z)
    for (Index y = 0; y < g[1]; ++y)
      for (Index x = 0; x < g[0]; ++x, ++i) {
        const Real X = (x + 0.5) / g[0], Y = (y + 0.5) / g[1], Z = (z + 0.5) / g[2];
        const Real u = r.next();
        switch (shape) {
          case 0:  // uniform noise
            w[i] = 0.25 + u;
            break;
          case 1:  // noise + a dense corner (test_stage_target's CFD-DEM shape)
            w[i] = 0.25 + u + ((x < g[0] / 3 && z < g[2] / 2) ? 3.0 : 0.0);
            break;
          case 2:  // flat particle bed: 1 + integer counts below z = bed
            w[i] = 1.0 + (Z < bed ? std::floor(8.0 * u) : 0.0);
            break;
          case 3: {  // heap: bed surface falling in x and y (the probe's tilt)
            const Real zs = bed * (1.0 + tilt * (0.5 - X) + tilt * (0.5 - Y));
            w[i] = 1.0 + (Z < zs ? std::floor(8.0 * u) : 0.0);
            break;
          }
          default: {  // a smooth blob plus heavy-tailed noise
            const Real d2 = (X - cx) * (X - cx) + (Y - cy) * (Y - cy) + (Z - cz) * (Z - cz);
            w[i] = 0.5 + 20.0 * std::exp(-d2 / 0.02) + std::exp(2.0 * u);
            break;
          }
        }
      }
  return w;
}

// The test's own coarse weight sums: per coarse cell, its align-box walked x-fastest from 0.
std::vector<Real> coarseSums(const IVec<3>& g, const std::vector<Real>& w, Index a) {
  const IVec<3> gc{g[0] / a, g[1] / a, g[2] / a};
  std::vector<Real> wc(static_cast<std::size_t>(gc[0] * gc[1] * gc[2]), 0.0);
  for (Index cz = 0; cz < gc[2]; ++cz)
    for (Index cy = 0; cy < gc[1]; ++cy)
      for (Index cx = 0; cx < gc[0]; ++cx) {
        Real s = 0.0;
        for (Index z = cz * a; z < (cz + 1) * a; ++z)
          for (Index y = cy * a; y < (cy + 1) * a; ++y)
            for (Index x = cx * a; x < (cx + 1) * a; ++x)
              s += w[static_cast<std::size_t>(x + g[0] * (y + g[1] * z))];
        wc[static_cast<std::size_t>(cx + gc[0] * (cy + gc[1] * cz))] = s;
      }
  return wc;
}

// The test's own imbalance: max block weight over the mean, block sums x-fastest.
Real imbalanceOf(const Dec& d, const std::vector<Real>& w) {
  const IVec<3> g = d.globalSize();
  double tot = 0.0, hi = 0.0;
  for (std::size_t b = 0; b < d.numBlocks(); ++b) {
    const auto o = d.origins()[b], s = d.sizes()[b];
    double sum = 0.0;
    for (Index z = o[2]; z < o[2] + s[2]; ++z)
      for (Index y = o[1]; y < o[1] + s[1]; ++y)
        for (Index x = o[0]; x < o[0] + s[0]; ++x)
          sum += w[static_cast<std::size_t>(x + g[0] * (y + g[1] * z))];
    tot += sum;
    hi = (b == 0 || sum > hi) ? sum : hi;
  }
  return hi / (tot / static_cast<double>(d.numBlocks()));
}

bool anyEmpty(const Dec& d) {
  for (const auto& s : d.sizes())
    for (int k = 0; k < 3; ++k)
      if (s[k] <= 0)
        return true;
  return false;
}

bool tiles(const Dec& d) {
  const IVec<3> g = d.globalSize();
  std::vector<int> hit(static_cast<std::size_t>(g[0] * g[1] * g[2]), 0);
  for (std::size_t b = 0; b < d.numBlocks(); ++b) {
    IVec<3> e{};
    for (int k = 0; k < 3; ++k)
      e[k] = d.origins()[b][k] + d.sizes()[b][k];
    forEachInBox<3>(d.origins()[b], e,
                    [&](const IVec<3>& c) { ++hit[static_cast<std::size_t>(d.linearGlobal(c))]; });
  }
  for (int h : hit)
    if (h != 1)
      return false;
  return true;
}

// The feasible alignment exponents for (np, g), computed independently of the chooser.
int topAlign(const IVec<3>& g, int np) {
  int top = 0;
  for (int a = 1;; ++a) {
    const Index q = Index(1) << a;
    bool ok = true;
    double cells = 1.0;
    for (int k = 0; k < 3; ++k) {
      if (g[k] % q != 0 || g[k] / q < 2)
        ok = false;
      cells *= static_cast<double>(g[k] / q);
    }
    if (!ok || cells < np)
      return top;
    top = a;
  }
}

Dec alignedInit(int np, const IVec<3>& g, const std::vector<Real>& w, int a) {
  Dec d;
  const Index q = Index(1) << a;
  d.init(static_cast<std::size_t>(np), g, w, IVec<3>{q, q, q});
  return d;
}

const std::vector<IVec<3>> kGrids{{32, 32, 32}, {48, 32, 16}, {96, 96, 96}};
const std::vector<int> kNps{1, 2, 3, 4, 5, 6, 7, 8, 12, 24};
constexpr int kFields = 50;

// ---- G-A1 -------------------------------------------------------------------------------------
void gateA1() {
  long n = 0;
  for (const auto& g : kGrids)
    for (int f = 0; f < kFields; ++f) {
      const std::vector<Real> w = weightField(g, f);
      for (int np : kNps) {
        Dec plain;
        plain.init(static_cast<std::size_t>(np), g, w);
        Dec one;
        one.init(static_cast<std::size_t>(np), g, w, IVec<3>{1, 1, 1});
        PECLET_CORE_CHECK(sameDec(plain, one));
        ++n;
      }
    }
  std::printf("G-A1 reduction: %ld (grid, field, np) cases, init(w, align=1) == init(w) bitwise\n",
              n);
}

// ---- G-A2 -------------------------------------------------------------------------------------
void gateA2() {
  long n = 0, skipped = 0, empty = 0;
  for (const auto& g : kGrids)
    for (int f = 0; f < kFields; ++f) {
      const std::vector<Real> w = weightField(g, f);
      for (int np : kNps)
        for (int a : {1, 2, 3}) {
          if (a > topAlign(g, np)) {
            ++skipped;
            continue;
          }
          const Index q = Index(1) << a;
          const Dec d = alignedInit(np, g, w, a);
          ++n;
          PECLET_CORE_CHECK_EQ(d.numBlocks(), static_cast<std::size_t>(np));
          PECLET_CORE_CHECK(tiles(d));
          // every origin, size and split a multiple of 2^a
          bool mult = true;
          for (std::size_t b = 0; b < d.numBlocks(); ++b)
            for (int k = 0; k < 3; ++k)
              if (d.origins()[b][k] % q || d.sizes()[b][k] % q)
                mult = false;
          // An empty block (size 0, still a multiple) is the existing weighted ORB's behaviour on a
          // coarse grid barely larger than np under a concentrated weight — the plain weighted
          // init does the same at the same granularity. Counted, not failed: the chooser rejects
          // such candidates (G-A3).
          if (anyEmpty(d))
            ++empty;
          std::vector<int> sd;
          std::vector<Index> sv;
          d.flattenTree(sd, sv);
          for (std::size_t i = 0; i < sd.size(); ++i)
            if (sd[i] != -1 && sv[i] % q)
              mult = false;
          PECLET_CORE_CHECK(mult);
          // coarse-first: coarsened(2^a) IS the weighted ORB of the coarse sums, and refines back
          const Dec c = d.coarsened(IVec<3>{q, q, q});
          const IVec<3> gc{g[0] / q, g[1] / q, g[2] / q};
          const Dec ref(static_cast<std::size_t>(np), gc, coarseSums(g, w, q));
          PECLET_CORE_CHECK(sameDec(c, ref));
          PECLET_CORE_CHECK(sameDec(c.refined(IVec<3>{q, q, q}), d));
          // the lift j -> j+1 is in place for every j < a, under both rules
          for (int j = 0; j < a; ++j) {
            const Index r = Index(1) << j;
            const Dec lj = d.coarsened(IVec<3>{r, r, r});
            PECLET_CORE_CHECK(flowLiftable(lj));
            PECLET_CORE_CHECK(amrLiftable(lj));
          }
        }
    }
  std::printf(
      "G-A2 nesting: %ld aligned partitions checked (%ld (a, np, grid) infeasible, %ld with an "
      "empty block)\n",
      n, skipped, empty);
}

// ---- G-A3 -------------------------------------------------------------------------------------
void gateA3() {
  constexpr Real budget = 1.05;
  std::map<std::string, std::map<int, int>> hist;  // "grid np" -> a -> count
  long n = 0;
  for (const auto& g : kGrids)
    for (int f = 0; f < kFields; ++f) {
      const std::vector<Real> w = weightField(g, f);
      for (int np : kNps) {
        const auto r = chooseAlignedWeighted(static_cast<std::size_t>(np), g, w);
        const int top = topAlign(g, np);
        ++n;
        ++hist[std::to_string(g[0]) + "x" + std::to_string(g[1]) + "x" + std::to_string(g[2]) +
               " np" + std::to_string(np)][r.a];
        PECLET_CORE_CHECK(r.a >= 0 && r.a <= top);
        auto fails = [&](int a) {
          const Dec d = alignedInit(np, g, w, a);
          return anyEmpty(d) || !(imbalanceOf(d, w) <= budget);
        };
        if (r.a > 0) {
          PECLET_CORE_CHECK(imbalanceOf(r.dec, w) <= budget);
          PECLET_CORE_CHECK(r.imbalance == imbalanceOf(r.dec, w));
          PECLET_CORE_CHECK(sameDec(r.dec, alignedInit(np, g, w, r.a)));
        } else {
          Dec plain;
          plain.init(static_cast<std::size_t>(np), g, w);
          PECLET_CORE_CHECK(sameDec(r.dec, plain));
        }
        for (int a = r.a + 1; a <= top; ++a)  // nothing larger fits
          PECLET_CORE_CHECK(fails(a));
      }
    }
  std::printf("G-A3 chooser: %ld cases; chosen a per (grid, np) over %d fields:\n", n, kFields);
  for (const auto& [key, h] : hist) {
    std::printf("  %-16s", key.c_str());
    for (const auto& [a, c] : h)
      std::printf("  a=%d:%2d", a, c);
    std::printf("\n");
  }

  // forced fallbacks: an odd axis admits no alignment; one dominant cell no budget
  {
    const IVec<3> g{33, 32, 32};
    const std::vector<Real> w = weightField(g, 3);
    const auto r = chooseAlignedWeighted(std::size_t{8}, g, w);
    Dec plain;
    plain.init(8, g, w);
    PECLET_CORE_CHECK_EQ(r.a, 0);
    PECLET_CORE_CHECK(sameDec(r.dec, plain));
  }
  for (int np : {2, 4, 8, 24}) {
    const IVec<3> g{32, 32, 32};
    std::vector<Real> w(static_cast<std::size_t>(32 * 32 * 32), 1.0);
    w[static_cast<std::size_t>(5 + 32 * (7 + 32 * 9))] = 1.0e7;
    const auto r = chooseAlignedWeighted(static_cast<std::size_t>(np), g, w);
    Dec plain;
    plain.init(static_cast<std::size_t>(np), g, w);
    PECLET_CORE_CHECK_EQ(r.a, 0);
    PECLET_CORE_CHECK(sameDec(r.dec, plain));
    PECLET_CORE_CHECK(r.imbalance > budget);
  }
  // one block: every alignment is perfectly balanced, so the chooser takes the largest feasible
  {
    const IVec<3> g{48, 32, 16};
    const auto r = chooseAlignedWeighted(std::size_t{1}, g, weightField(g, 4));
    PECLET_CORE_CHECK_EQ(r.a, topAlign(g, 1));
  }
  // the caller's aMax caps a, and aMax = 0 is today's partition
  for (int f = 0; f < 10; ++f) {
    const IVec<3> g{32, 32, 32};
    const std::vector<Real> w = weightField(g, f);
    for (int cap : {0, 1, 2}) {
      const auto r = chooseAlignedWeighted(std::size_t{4}, g, w, 1.05, cap);
      PECLET_CORE_CHECK(r.a <= cap);
      const auto u = chooseAlignedWeighted(std::size_t{4}, g, w);
      if (u.a <= cap)
        PECLET_CORE_CHECK(r.a == u.a && sameDec(r.dec, u.dec));
    }
  }
}

}  // namespace

int main() {
  gateA1();
  gateA2();
  gateA3();
  PECLET_CORE_RETURN_TEST_RESULT();
}

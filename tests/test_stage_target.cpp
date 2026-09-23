// chooseStageTarget (peclet/core/decomp/stage_target.hpp) against its reference implementation:
// flow's inline telescope search (flow/src/mac_cutcell_mg.hpp, CutcellMG::initMpi and the
// replicated planner CutcellMG::predict), TRANSCRIBED below — core does not link flow. Every level
// of a whole coarsening ladder is compared, so a stage that fires early changes every row after it:
//
//   A  flow's per-axis rule, maxBlockCells = 0: on every level of every case the policy
//      returns the same decision (stage or not), the same agglomerated decomposition (origins and
//      sizes), the same group map (groupOf) and the same owners (rootOf), and never Replicated
//      (flow's search always admits depth 0). Cases: flow's own telescope-test grids (32^3 forced
//      and 24^3 starved under a plain ORB), the 384^3 FoxBerry ladder at 24…1536 ranks, anisotropic
//      and badly factored grids, plain / aligned / WEIGHTED ORBs, minExtent 0, 2, 4 (flow's
//      default is 4), 1…1536 blocks.
//   B  amr's all-axes rule: the result lifts, has fewer blocks, and no deeper truncation would
//      have; a level grid that is odd on an axis falls through to Replicated.
//   C  the contract edges: InPlace is the identity; a negative maxBlockCells throws; a level grid
//      that is not the decomposition's throws; the Repartition branch (design §11.1-11.2): a
//      merge within the cap is accepted, one above it is replaced by a liftable proportional ORB
//      on np_L = ceil(cells / maxBlockCells) ranks (power-of-two retry), and falls back to the
//      heavy merge when no repartition lifts.
//   D  flow's FORCED telescope (`teleForce_ == L`, test-only) through `shallowestLiftableMerge`
//      called directly, as S4 will: flow's test_telescope_mpi case B verbatim (32^3 on flow's
//      aligned factory partition, nLevels 4, forced at level 1, telescoping on, flow's default
//      minExtent 4) at np = 2 and 4 — where the forced merge must fire — then every forced level
//      0..5 over Part A's grids and partitions, telescoping on and off. Same depth as well.
//   E  G-B1 of the design (§11.6): the six weighted level-0 partitions of flow's weighted-dec0
//   probe
//      (flow tests/study/weighted_dec0_telescope_probe.py + weighted_dec0_results/, flow 0ae29d6;
//      the eight table rows of amr/docs/amr_mg_core_boundary.md §6 — two partitions are run at two
//      depths) as fixtures, rebuilt through core's weighted ORB and checked block for block against
//      the probe's logs. With maxBlockCells = 0 the ladder is the MEASURED one (where the telescope
//      fired and onto how many ranks); with maxBlockCells = the largest level-0 block the stage
//      kinds are §11.1's.
#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/decomp/stage_target.hpp"
#include "test_util.hpp"

using peclet::core::Index;
using peclet::core::IVec;
using peclet::core::Real;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::decomp::chooseStageTarget;
using peclet::core::decomp::largestBlockCells;
using peclet::core::decomp::repartitionTarget;
using peclet::core::decomp::shallowestLiftableMerge;
using peclet::core::decomp::StageKind;
using peclet::core::decomp::StageTarget;
using Dec = BlockDecomposer<3>;

namespace {

// ---- flow's rules, transcribed verbatim (mac_cutcell_mg.hpp) ------------------------------------
bool can(Index d) {
  return (d % 2 == 0) && (d / 2 >= 2);
}
bool evenOn(const Dec& d, int ax) {
  for (std::size_t b = 0; b < d.sizes().size(); ++b)
    if ((d.origins()[b][ax] % 2) || (d.sizes()[b][ax] % 2))
      return false;
  return true;
}
int minExtentOf(const Dec& d) {
  long m = std::numeric_limits<long>::max();
  for (const auto& sz : d.sizes())
    for (int k = 0; k < 3; ++k)
      m = std::min(m, (long)sz[k]);
  return (int)m;
}
IVec<3> coarsenAlignment(Index gx, Index gy, Index gz) {
  IVec<3> gs{gx, gy, gz}, a{1, 1, 1};
  for (bool any = true; any;) {
    any = false;
    for (int k = 0; k < 3; ++k)
      if (can(gs[k])) {
        a[k] *= 2;
        gs[k] /= 2;
        any = true;
      }
  }
  for (int k = 0; k < 3; ++k)
    if (a[k] > 16)
      a[k] = 16;
  return a;
}

// One level's stage decision.
struct Stage {
  bool tele = false;
  int depth = -1;  // the depth the search chose (compared on the forced path)
  Dec dec;
  std::vector<int> groupOf, rootOf;
};

// flow's trigger + depth search (initMpi), verbatim. `forced` is flow's `teleForce_ == L`.
Stage flowReference(const Dec& curDec, const IVec<3>& gs, int teleMinExtent, bool telescope,
                    bool forced) {
  Stage s;
  const bool canAny = can(gs[0]) || can(gs[1]) || can(gs[2]);
  bool blocked = false;
  for (int ax = 0; ax < 3; ++ax)
    if (can(gs[ax]) && !evenOn(curDec, ax))
      blocked = true;
  const bool tooSmall = teleMinExtent > 0 && minExtentOf(curDec) < teleMinExtent;
  const bool doTele =
      canAny && curDec.numBlocks() > 1 && ((telescope && (blocked || tooSmall)) || forced);
  if (!doTele)
    return s;
  for (int d = curDec.treeDepth() - 1; d >= 0; --d) {
    std::vector<int> go, ro;
    Dec cand = curDec.agglomerated(d, &go, &ro);
    bool ok = true;
    for (int ax = 0; ax < 3; ++ax)
      if (can(gs[ax]) && !evenOn(cand, ax))
        ok = false;
    if (ok && teleMinExtent > 0 && cand.numBlocks() > 1 && minExtentOf(cand) < 2 * teleMinExtent)
      ok = false;
    if (ok && cand.numBlocks() < curDec.numBlocks()) {
      s.tele = true;
      s.depth = d;
      s.dec = cand;
      s.groupOf = go;
      s.rootOf = ro;
      return s;
    }
  }
  return s;  // dSel < 0: no telescope
}

bool flowLiftable(const Dec& d) {
  for (int ax = 0; ax < 3; ++ax)
    if (can(d.globalSize()[ax]) && !evenOn(d, ax))
      return false;
  return true;
}

// The same decision through core, composed as S4 will. What stays with the caller is flow's own
// gate (telescoping on, some axis of the level grid still coarsenable) and the forced level, which
// calls the search directly.
Stage viaCore(const Dec& cur, const IVec<3>& gs, int teleMinExtent, bool telescope, bool forced,
              bool& sawReplicated) {
  Stage s;
  const bool canAny = can(gs[0]) || can(gs[1]) || can(gs[2]);
  if (!canAny)
    return s;
  StageTarget<3> t;  // InPlace
  if (telescope)
    t = chooseStageTarget(cur, gs, flowLiftable, teleMinExtent);
  if (t.kind == StageKind::InPlace && forced) {
    if (auto m = shallowestLiftableMerge(cur, flowLiftable, teleMinExtent)) {
      s.tele = true;
      s.depth = m->depth;
      s.dec = m->dec;
      s.groupOf = m->groupOf;
      s.rootOf = m->ownerOf;
    }
    return s;
  }
  if (t.kind == StageKind::Replicated)
    sawReplicated = true;
  if (t.kind == StageKind::InPlace) {
    PECLET_CORE_CHECK(t.groupOf.empty());
    if (telescope)
      PECLET_CORE_CHECK_EQ(t.ownerOf.size(), cur.numBlocks());
    return s;
  }
  if (t.kind == StageKind::SiblingMerge) {
    s.tele = true;
    s.dec = t.dec;
    s.groupOf = t.groupOf;
    s.rootOf = t.ownerOf;
  }
  return s;
}

bool sameDec(const Dec& a, const Dec& b) {
  return a.numBlocks() == b.numBlocks() && a.origins() == b.origins() && a.sizes() == b.sizes() &&
         a.globalSize() == b.globalSize();
}

struct Counts {
  long compared = 0, staged = 0, forcedStaged = 0;
};
Counts gA, gD;

// Walk the whole ladder with both, comparing every level (predict()'s isotropic coarsening). A
// level stages only when L + 1 < nLevels, as in initMpi; `force` is flow's teleForce_ (-1 never).
void compareLadder(const std::string& name, Dec cur, int minExt, Counts& n = gA, int force = -1,
                   int nLevels = 16, bool telescope = true) {
  Dec ref = cur;
  IVec<3> gs = cur.globalSize();
  for (int L = 0; L < nLevels; ++L) {
    bool sawRep = false;
    Stage a, b;
    if (L + 1 < nLevels) {
      a = flowReference(ref, gs, minExt, telescope, force == L);
      b = viaCore(cur, gs, minExt, telescope, force == L, sawRep);
    }
    ++n.compared;
    bool ok = !sawRep && a.tele == b.tele;
    if (ok && a.tele)
      ok = sameDec(a.dec, b.dec) && a.groupOf == b.groupOf && a.rootOf == b.rootOf &&
           (b.depth < 0 || b.depth == a.depth);
    if (!ok) {
      std::fprintf(stderr, "MISMATCH %s minExt=%d L=%d tele %d/%d replicated %d\n", name.c_str(),
                   minExt, L, a.tele, b.tele, sawRep);
      ++::peclet::core::test::g_failures;
      return;
    }
    if (a.tele) {
      ++n.staged;
      if (force == L && b.depth >= 0)
        ++n.forcedStaged;
      ref = a.dec;
      cur = b.dec;
    }
    IVec<3> ratio{1, 1, 1}, next = gs;
    for (int k = 0; k < 3; ++k)
      if (can(gs[k]) && evenOn(ref, k)) {
        ratio[k] = 2;
        next[k] = gs[k] / 2;
      }
    if (next == gs)
      return;
    gs = next;
    ref = ref.coarsened(ratio);
    cur = cur.coarsened(ratio);
  }
}

std::vector<Real> pseudoRandomWeights(const IVec<3>& g, unsigned seed) {
  std::vector<Real> w(static_cast<std::size_t>(g[0] * g[1] * g[2]));
  unsigned s = seed;
  for (auto& v : w) {
    s = s * 1664525u + 1013904223u;
    v = 0.25 + static_cast<Real>(s >> 8) / static_cast<Real>(1u << 24);  // [0.25, 1.25)
  }
  // a dense "bed" in one corner, the CFD-DEM shape that moves the ORB splits off the halvings
  for (Index z = 0; z < g[2] / 2; ++z)
    for (Index y = 0; y < g[1]; ++y)
      for (Index x = 0; x < g[0] / 3; ++x)
        w[static_cast<std::size_t>(x + g[0] * (y + g[1] * z))] += 3.0;
  return w;
}

void partA() {
  const int minExts[] = {0, 2, 4};
  // flow's own telescope test: 32^3 (in place to the bottom at np=2/4) and the STARVED 24^3 under
  // a plain, unaligned ORB; plus badly factored and anisotropic grids.
  const IVec<3> grids[] = {{32, 32, 32}, {24, 24, 24}, {48, 48, 48}, {96, 48, 24},
                           {40, 24, 18}, {64, 32, 16}, {20, 20, 20}, {36, 60, 44}};
  const int nps[] = {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 27, 48, 64};
  for (const auto& g : grids)
    for (int np : nps) {
      if (static_cast<Index>(np) > g[0] * g[1] * g[2] / 8)
        continue;
      for (int me : minExts) {
        const std::string tag = std::to_string(g[0]) + "x" + std::to_string(g[1]) + "x" +
                                std::to_string(g[2]) + " np" + std::to_string(np);
        compareLadder("plain " + tag, Dec((std::size_t)np, g), me);
        compareLadder("aligned " + tag, Dec((std::size_t)np, g, coarsenAlignment(g[0], g[1], g[2])),
                      me);
        compareLadder("weighted " + tag, Dec((std::size_t)np, g, pseudoRandomWeights(g, 7u + np)),
                      me);
      }
    }
  // The FoxBerry 384^3 ladder, 24…1536 ranks (MG_TELESCOPING_PLAN.md): plain and aligned.
  const IVec<3> fox{384, 384, 384};
  for (int np : {24, 48, 96, 192, 384, 768, 1536})
    for (int me : minExts) {
      compareLadder("plain 384^3 np" + std::to_string(np), Dec((std::size_t)np, fox), me);
      compareLadder("aligned 384^3 np" + std::to_string(np),
                    Dec((std::size_t)np, fox, coarsenAlignment(384, 384, 384)), me);
    }
}

// amr's rule (amr_mg_depth.md §6.2): every block even in origin and size on EVERY axis.
bool amrLiftable(const Dec& d) {
  for (int ax = 0; ax < 3; ++ax)
    if (!evenOn(d, ax))
      return false;
  return true;
}

void partB() {
  long checked = 0;
  for (const IVec<3>& g : {IVec<3>{48, 48, 48}, IVec<3>{24, 40, 16}, IVec<3>{30, 30, 30}})
    for (int np : {2, 3, 4, 6, 8, 12, 24}) {
      const Dec cur((std::size_t)np, g, pseudoRandomWeights(g, 11u + np));
      for (int me : {0, 4}) {
        const StageTarget<3> t = chooseStageTarget(cur, g, amrLiftable, me);
        ++checked;
        if (t.kind == StageKind::InPlace) {
          PECLET_CORE_CHECK(amrLiftable(cur));
          continue;
        }
        if (t.kind == StageKind::Replicated) {
          // only when no truncation, not even the whole grid on one block, lifts
          PECLET_CORE_CHECK(!amrLiftable(cur.agglomerated(0)));
          PECLET_CORE_CHECK_EQ(t.dec.numBlocks(), 1u);
          PECLET_CORE_CHECK(t.dec.globalSize() == g);
          PECLET_CORE_CHECK(t.ownerOf == std::vector<int>{0});
          PECLET_CORE_CHECK(t.groupOf.empty());
          continue;
        }
        PECLET_CORE_CHECK(t.kind == StageKind::SiblingMerge);
        PECLET_CORE_CHECK(amrLiftable(t.dec));
        PECLET_CORE_CHECK(t.dec.numBlocks() < cur.numBlocks());
        PECLET_CORE_CHECK_EQ(t.groupOf.size(), cur.numBlocks());
        PECLET_CORE_CHECK_EQ(t.ownerOf.size(), t.dec.numBlocks());
        // the deepest qualifying truncation: none deeper lifts with fewer blocks and fat blocks
        for (int d = cur.treeDepth() - 1; d >= 0; --d) {
          const Dec c = cur.agglomerated(d);
          if (sameDec(c, t.dec))
            break;
          const bool fat = me == 0 || c.numBlocks() == 1 || minExtentOf(c) >= 2 * me;
          PECLET_CORE_CHECK(!(amrLiftable(c) && fat && c.numBlocks() < cur.numBlocks()));
        }
      }
    }
  // 30^3 is odd after one halving: at 15^3 nothing lifts under amr's rule -> Replicated.
  const Dec odd(4, IVec<3>{15, 15, 15});
  const StageTarget<3> t = chooseStageTarget(odd, IVec<3>{15, 15, 15}, amrLiftable, 0);
  PECLET_CORE_CHECK(t.kind == StageKind::Replicated);
  std::printf("  B  amr all-axes rule: %ld targets checked\n", checked);
}

void partC() {
  const Dec one(1, IVec<3>{8, 8, 8});
  auto never = [](const Dec&) { return false; };
  // a single block is never staged, whatever the predicate says
  PECLET_CORE_CHECK(chooseStageTarget(one, IVec<3>{8, 8, 8}, never, 4).kind == StageKind::InPlace);
  const Dec four(4, IVec<3>{32, 32, 32});
  auto always = [](const Dec&) { return true; };
  const StageTarget<3> ip = chooseStageTarget(four, IVec<3>{32, 32, 32}, always, 0);
  PECLET_CORE_CHECK(ip.kind == StageKind::InPlace);
  PECLET_CORE_CHECK(sameDec(ip.dec, four));
  PECLET_CORE_CHECK((ip.ownerOf == std::vector<int>{0, 1, 2, 3}));
  // the economic trigger stages a liftable level whose blocks are thinner than minExtent
  PECLET_CORE_CHECK(chooseStageTarget(four, IVec<3>{32, 32, 32}, always, 32).kind ==
                    StageKind::SiblingMerge);
  bool threw = false;
  try {
    chooseStageTarget(four, IVec<3>{32, 32, 32}, always, 0, /*maxBlockCells=*/-1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  PECLET_CORE_CHECK(threw);
  threw = false;
  try {
    chooseStageTarget(four, IVec<3>{16, 32, 32}, always, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  PECLET_CORE_CHECK(threw);
}

// ---- the weighted level-0 partitions of flow's probe (design §6), as fixtures -------------------
// Each is the per-rank block list the probe printed after `rebalance_by_weights` (flow
// tests/study/weighted_dec0_results/*.txt, flow 0ae29d6). They were reproduced from the probe's
// own weights (bed_weights, numpy seeds 7 / 11) through BlockDecomposer(np, 96^3, w) block for
// block before being copied here. The test rebuilds each through the weighted ORB with a
// synthetic weight — every block the same total, spread uniformly over it — which puts every
// split exactly where the probe's weights put it, so the ORB TREE (which agglomerated() walks) is
// the probe's too; the rebuilt blocks are checked against the fixture.
struct Fixture {
  const char* name;
  std::vector<std::pair<IVec<3>, IVec<3>>> blocks;  // (origin, size), rank order
};
const IVec<3> k96{96, 96, 96};
const Fixture kHeap8{"heap tilt 0.5 np8",
                     {{{0, 0, 0}, {45, 45, 27}},
                      {{0, 0, 27}, {45, 45, 69}},
                      {{0, 45, 0}, {45, 51, 23}},
                      {{0, 45, 23}, {45, 51, 73}},
                      {{45, 0, 0}, {51, 44, 23}},
                      {{45, 0, 23}, {51, 44, 73}},
                      {{45, 44, 0}, {51, 52, 20}},
                      {{45, 44, 20}, {51, 52, 76}}}};
const Fixture kHeap4{"heap tilt 0.5 np4",
                     {{{0, 0, 0}, {45, 45, 96}},
                      {{0, 45, 0}, {45, 51, 96}},
                      {{45, 0, 0}, {51, 44, 96}},
                      {{45, 44, 0}, {51, 52, 96}}}};
const Fixture kFlat8{"flat bed np8",
                     {{{0, 0, 0}, {48, 48, 23}},
                      {{0, 0, 23}, {48, 48, 73}},
                      {{0, 48, 0}, {48, 48, 23}},
                      {{0, 48, 23}, {48, 48, 73}},
                      {{48, 0, 0}, {48, 48, 23}},
                      {{48, 0, 23}, {48, 48, 73}},
                      {{48, 48, 0}, {48, 48, 23}},
                      {{48, 48, 23}, {48, 48, 73}}}};
const Fixture kTilt8{"heap tilt 0.3 seed 11 np8",
                     {{{0, 0, 0}, {46, 46, 25}},
                      {{0, 0, 25}, {46, 46, 71}},
                      {{0, 46, 0}, {46, 50, 23}},
                      {{0, 46, 23}, {46, 50, 73}},
                      {{46, 0, 0}, {50, 46, 23}},
                      {{46, 0, 23}, {50, 46, 73}},
                      {{46, 46, 0}, {50, 50, 21}},
                      {{46, 46, 21}, {50, 50, 75}}}};
const Fixture kTilt4{"heap tilt 0.3 seed 11 np4",
                     {{{0, 0, 0}, {46, 46, 96}},
                      {{0, 46, 0}, {46, 50, 96}},
                      {{46, 0, 0}, {50, 46, 96}},
                      {{46, 46, 0}, {50, 50, 96}}}};
const Fixture kFlat4{"flat bed np4",
                     {{{0, 0, 0}, {48, 48, 96}},
                      {{0, 48, 0}, {48, 48, 96}},
                      {{48, 0, 0}, {48, 48, 96}},
                      {{48, 48, 0}, {48, 48, 96}}}};

Dec fromFixture(const Fixture& f) {
  std::vector<Real> w(static_cast<std::size_t>(k96[0] * k96[1] * k96[2]), 0.0);
  for (const auto& [o, sz] : f.blocks) {
    const Real per = 1.0 / static_cast<Real>(sz[0] * sz[1] * sz[2]);
    for (Index z = o[2]; z < o[2] + sz[2]; ++z)
      for (Index y = o[1]; y < o[1] + sz[1]; ++y)
        for (Index x = o[0]; x < o[0] + sz[0]; ++x)
          w[static_cast<std::size_t>(x + k96[0] * (y + k96[1] * z))] = per;
  }
  const Dec d(f.blocks.size(), k96, w);
  bool same = d.numBlocks() == f.blocks.size();
  for (std::size_t b = 0; same && b < d.numBlocks(); ++b)
    same = d.origins()[b] == f.blocks[b].first && d.sizes()[b] == f.blocks[b].second;
  if (!same) {
    std::fprintf(stderr, "FIXTURE %s: the weighted ORB did not reproduce the probe's blocks\n",
                 f.name);
    ++::peclet::core::test::g_failures;
  }
  return d;
}

bool isProportional(const Dec& d) {
  return sameDec(d, Dec(d.numBlocks(), d.globalSize()));
}

// The Repartition branch (design §11.1-11.2): contract edges on the fixtures, then a sweep that
// checks every returned target against the §11 order.
void partCRepartition() {
  const Dec heap8 = fromFixture(kHeap8), tilt8 = fromFixture(kTilt8);
  const Index cells = 96 * 96 * 96;
  // maxBlockCells = 0 is S1: the heavy merge (the measured d = 0 collapse)
  StageTarget<3> t = chooseStageTarget(heap8, k96, flowLiftable, 4, 0);
  PECLET_CORE_CHECK(t.kind == StageKind::SiblingMerge && t.dec.numBlocks() == 1u);
  // np_L clamps to the current rank count ...
  t = chooseStageTarget(heap8, k96, flowLiftable, 0, 1);
  PECLET_CORE_CHECK(t.kind == StageKind::Repartition && sameDec(t.dec, Dec(8, k96)));
  PECLET_CORE_CHECK((t.ownerOf == std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}) && t.groupOf.empty());
  // ... and to the extent cap prod_k max(1, floor(96 / (2*minExtent))) when minExtent > 0
  t = chooseStageTarget(heap8, k96, flowLiftable, 16, 1);  // cap 3^3 = 27: still 8
  PECLET_CORE_CHECK(t.kind == StageKind::Repartition && t.dec.numBlocks() == 8u);
  t = chooseStageTarget(heap8, k96, flowLiftable, 32, 1);  // cap 1: one rank
  PECLET_CORE_CHECK(t.kind == StageKind::Repartition && t.dec.numBlocks() == 1u);
  // np_L = ceil(cells / maxBlockCells): 7 does not lift (the root split is 41), the retry takes 4
  PECLET_CORE_CHECK(!flowLiftable(Dec(7, k96)));
  t = chooseStageTarget(heap8, k96, flowLiftable, 4, (cells + 6) / 7);
  PECLET_CORE_CHECK(t.kind == StageKind::Repartition && t.dec.numBlocks() == 4u);
  t = chooseStageTarget(heap8, k96, flowLiftable, 4, cells / 6);  // 6 lifts: taken as is
  PECLET_CORE_CHECK(t.kind == StageKind::Repartition && t.dec.numBlocks() == 6u);
  // a merge within the cap is accepted: the whole level is one finest block
  t = chooseStageTarget(heap8, k96, flowLiftable, 4, cells);
  PECLET_CORE_CHECK(t.kind == StageKind::SiblingMerge && t.dec.numBlocks() == 1u);
  // no repartition lifts -> the heavy merge (step 5); nothing lifts at all -> Replicated (step 6)
  auto notProportional = [](const Dec& d) { return flowLiftable(d) && !isProportional(d); };
  t = chooseStageTarget(tilt8, k96, notProportional, 4, 1);
  PECLET_CORE_CHECK(t.kind == StageKind::SiblingMerge && sameDec(t.dec, tilt8.agglomerated(2)));
  PECLET_CORE_CHECK(largestBlockCells(t.dec) > 1);
  PECLET_CORE_CHECK(!repartitionTarget(tilt8, k96, notProportional, 4, 1));
  auto never = [](const Dec&) { return false; };
  t = chooseStageTarget(heap8, k96, never, 4, cells / 8);
  PECLET_CORE_CHECK(t.kind == StageKind::Replicated && t.dec.numBlocks() == 1u);
  bool threw = false;
  try {
    repartitionTarget(heap8, k96, flowLiftable, 4, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  PECLET_CORE_CHECK(threw);

  // The sweep: every target against the §11 order, with np_L and its retry transcribed here.
  long checked = 0, kinds[4] = {0, 0, 0, 0};
  const IVec<3> grids[] = {{48, 48, 48}, {96, 48, 24}, {40, 24, 18}, {64, 64, 64}};
  for (const auto& g : grids)
    for (int np : {2, 3, 4, 5, 6, 7, 8, 12, 16, 24}) {
      const Index gc = g[0] * g[1] * g[2];
      const Dec parts[] = {Dec((std::size_t)np, g),
                           Dec((std::size_t)np, g, pseudoRandomWeights(g, 3u + np))};
      for (const Dec& cur : parts)
        for (int me : {0, 4})
          for (Index cap : {Index{1}, gc / np, 2 * gc / np, gc}) {
            const StageTarget<3> r = chooseStageTarget(cur, g, flowLiftable, me, cap);
            ++checked;
            ++kinds[(int)r.kind];
            const auto S = shallowestLiftableMerge(cur, flowLiftable, me);
            const bool tooSmall = me > 0 && minExtentOf(cur) < me;
            if (r.kind == StageKind::InPlace) {
              PECLET_CORE_CHECK(flowLiftable(cur) && !tooSmall);
              continue;
            }
            PECLET_CORE_CHECK(!(flowLiftable(cur) && !tooSmall));
            // np_L and the retry sequence, from the design's pseudocode
            Index npL = std::clamp<Index>((gc + cap - 1) / cap, 1, np);
            if (me > 0) {
              Index c = 1;
              for (int k = 0; k < 3; ++k)
                c *= std::max<Index>(1, g[k] / (2 * me));
              npL = std::min(npL, c);
            }
            std::vector<Index> tries{npL};
            Index p2 = 1;
            while (2 * p2 <= npL)
              p2 *= 2;
            for (Index n = p2; n >= 1; n /= 2)
              tries.push_back(n);
            Index firstLift = 0;
            for (Index n : tries)
              if (flowLiftable(Dec((std::size_t)n, g))) {
                firstLift = n;
                break;
              }
            const bool sFits = S && largestBlockCells(S->dec) <= cap;
            if (r.kind == StageKind::SiblingMerge) {
              PECLET_CORE_CHECK(S && sameDec(r.dec, S->dec) && r.groupOf == S->groupOf &&
                                r.ownerOf == S->ownerOf);
              PECLET_CORE_CHECK(sFits || firstLift == 0);
            } else if (r.kind == StageKind::Repartition) {
              PECLET_CORE_CHECK(!sFits && firstLift > 0);
              PECLET_CORE_CHECK_EQ(r.dec.numBlocks(), (std::size_t)firstLift);
              PECLET_CORE_CHECK(sameDec(r.dec, Dec((std::size_t)firstLift, g)));
              PECLET_CORE_CHECK(r.groupOf.empty() && r.ownerOf.size() == r.dec.numBlocks());
              for (std::size_t b = 0; b < r.ownerOf.size(); ++b)
                PECLET_CORE_CHECK_EQ(r.ownerOf[b], (int)b);
            } else {
              PECLET_CORE_CHECK(!S && firstLift == 0);
            }
          }
    }
  std::printf(
      "  C  repartition branch: %ld targets checked (%ld in place, %ld merged, %ld repartitioned, "
      "%ld replicated)\n",
      checked, kinds[0], kinds[1], kinds[2], kinds[3]);
  PECLET_CORE_CHECK(kinds[2] > 0);
}

// One ladder under the policy: every stage (level, kind, target block count), flow's gate and
// coarsening as compareLadder has them.
struct StageRec {
  int L;
  StageKind kind;
  std::size_t ranks;
  bool operator==(const StageRec&) const = default;
};
std::vector<StageRec> ladder(Dec cur, int nLevels, int minExt, Index maxBlockCells) {
  std::vector<StageRec> out;
  IVec<3> gs = cur.globalSize();
  for (int L = 0; L < nLevels; ++L) {
    if (L + 1 < nLevels && (can(gs[0]) || can(gs[1]) || can(gs[2]))) {
      StageTarget<3> t = chooseStageTarget(cur, gs, flowLiftable, minExt, maxBlockCells);
      if (t.kind != StageKind::InPlace) {
        out.push_back({L, t.kind, t.dec.numBlocks()});
        cur = t.dec;
      }
    }
    IVec<3> ratio{1, 1, 1}, next = gs;
    for (int k = 0; k < 3; ++k)
      if (can(gs[k]) && evenOn(cur, k)) {
        ratio[k] = 2;
        next[k] = gs[k] / 2;
      }
    if (next == gs)
      break;
    gs = next;
    cur = cur.coarsened(ratio);
  }
  return out;
}
std::string show(const std::vector<StageRec>& v) {
  static const char* nm[] = {"InPlace", "SiblingMerge", "Repartition", "Replicated"};
  std::string s;
  for (const auto& r : v)
    s += " L" + std::to_string(r.L) + ":" + nm[(int)r.kind] + "->" + std::to_string(r.ranks);
  return s.empty() ? " (no stage)" : s;
}

// G-B1: the eight rows of the §6 table (flow's default minExtent 4; 8 requested levels, which the
// 96^3 grid caps at 6, or the default 4).
void partE() {
  using K = StageKind;
  struct Row {
    const Fixture* f;
    int nLevels;
    std::vector<StageRec> measured;  // maxBlockCells = 0: the ladder flow printed ([mg] lines)
    std::vector<StageRec> capped;    // maxBlockCells = the largest level-0 block
  };
  const Row rows[] = {
      {&kHeap8, 8, {{0, K::SiblingMerge, 1}}, {{0, K::Repartition, 5}, {1, K::SiblingMerge, 1}}},
      {&kHeap8, 4, {{0, K::SiblingMerge, 1}}, {{0, K::Repartition, 5}, {1, K::SiblingMerge, 1}}},
      {&kHeap4, 8, {{0, K::SiblingMerge, 1}}, {{0, K::Repartition, 4}, {4, K::SiblingMerge, 1}}},
      {&kHeap4, 4, {{0, K::SiblingMerge, 1}}, {{0, K::Repartition, 4}}},
      {&kFlat8,
       8,
       {{0, K::SiblingMerge, 4}, {4, K::SiblingMerge, 1}},
       {{0, K::Repartition, 6}, {4, K::SiblingMerge, 1}}},
      {&kTilt8,
       8,
       {{0, K::SiblingMerge, 4}, {1, K::SiblingMerge, 1}},
       {{0, K::Repartition, 5}, {1, K::SiblingMerge, 1}}},
      {&kTilt4, 8, {{1, K::SiblingMerge, 1}}, {{1, K::SiblingMerge, 1}}},
      {&kFlat4, 8, {{4, K::SiblingMerge, 1}}, {{4, K::SiblingMerge, 1}}},
  };
  for (const Row& r : rows) {
    const Dec d = fromFixture(*r.f);
    const Index cap = largestBlockCells(d);
    const auto m = ladder(d, r.nLevels, 4, 0), c = ladder(d, r.nLevels, 4, cap);
    std::printf("  E  %-26s depth %d  maxBlockCells %6lld  measured%s  |  capped%s\n", r.f->name,
                r.nLevels, (long long)cap, show(m).c_str(), show(c).c_str());
    if (!(m == r.measured) || !(c == r.capped)) {
      std::fprintf(stderr, "G-B1 MISMATCH %s depth %d\n", r.f->name, r.nLevels);
      ++::peclet::core::test::g_failures;
    }
  }
  // flow's validated proportional ladders are unchanged by the cap (§11.1): the FoxBerry 384^3
  // ladder at 24…1536 ranks, plain and aligned, minExtent 0 / 2 / 4.
  long ladders = 0;
  for (int np : {24, 48, 96, 192, 384, 768, 1536})
    for (int me : {0, 2, 4})
      for (bool aligned : {false, true}) {
        const IVec<3> fox{384, 384, 384};
        const Dec d = aligned ? Dec((std::size_t)np, fox, coarsenAlignment(384, 384, 384))
                              : Dec((std::size_t)np, fox);
        ++ladders;
        if (!(ladder(d, 16, me, 0) == ladder(d, 16, me, largestBlockCells(d)))) {
          std::fprintf(stderr, "G-B1: the 384^3 np%d ladder changed under the cap\n", np);
          ++::peclet::core::test::g_failures;
        }
      }
  std::printf("  E  384^3 ladders (24..1536 ranks) unchanged under the cap: %ld\n", ladders);
}

}  // namespace

// flow's test_telescope_mpi case B, then a sweep of forced levels.
void partD() {
  // Case B verbatim: {"B forced telescope@1 32^3", {32, 32, 32}, 4, 1, true, -1}, dec0 = nullptr
  // -> flow's own factory decomposition(size, 32, 32, 32) (levels = 0: the aligned ORB), minExtent
  // left at flow's default 4. np = 2 and 4 are the gated rank counts.
  const IVec<3> g{32, 32, 32};
  for (int np : {2, 4}) {
    Counts c;
    compareLadder("case B np" + std::to_string(np),
                  Dec((std::size_t)np, g, coarsenAlignment(32, 32, 32)), 4, c, /*force=*/1,
                  /*nLevels=*/4, /*telescope=*/true);
    PECLET_CORE_CHECK_EQ(c.forcedStaged, 1);  // the forced merge fired, through the search
    gD.compared += c.compared;
    gD.staged += c.staged;
    gD.forcedStaged += c.forcedStaged;
  }
  const IVec<3> grids[] = {{32, 32, 32}, {24, 24, 24}, {48, 48, 48}, {96, 48, 24},
                           {40, 24, 18}, {64, 32, 16}, {20, 20, 20}, {36, 60, 44}};
  for (const auto& gr : grids)
    for (int np : {2, 3, 4, 6, 8, 12, 16, 27, 64}) {
      if (static_cast<Index>(np) > gr[0] * gr[1] * gr[2] / 8)
        continue;
      const Dec parts[] = {Dec((std::size_t)np, gr),
                           Dec((std::size_t)np, gr, coarsenAlignment(gr[0], gr[1], gr[2])),
                           Dec((std::size_t)np, gr, pseudoRandomWeights(gr, 7u + np))};
      for (const Dec& p : parts)
        for (int force = 0; force <= 5; ++force)
          for (int me : {0, 4})
            for (bool tele : {true, false})
              compareLadder("forced sweep", p, me, gD, force, 16, tele);
    }
}

int main() {
  partA();
  std::printf("  A  flow's search: %ld ladder levels compared, %ld staged, all identical\n",
              gA.compared, gA.staged);
  PECLET_CORE_CHECK(gA.staged > 0);
  partB();
  partC();
  partCRepartition();
  partD();
  std::printf(
      "  D  forced telescope: %ld ladder levels compared, %ld staged, %ld by the forced search, "
      "all identical\n",
      gD.compared, gD.staged, gD.forcedStaged);
  PECLET_CORE_CHECK(gD.forcedStaged > 0);
  partE();
  PECLET_CORE_RETURN_TEST_RESULT();
}

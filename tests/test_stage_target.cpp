// chooseStageTarget (peclet/core/decomp/stage_target.hpp) against its reference implementation:
// flow's inline telescope search (flow/src/mac_cutcell_mg.hpp, CutcellMG::initMpi and the
// replicated planner CutcellMG::predict), TRANSCRIBED below — core does not link flow. Every level
// of a whole coarsening ladder is compared, so a stage that fires early changes every row after it:
//
//   A  flow's per-axis rule, allowRepartition = false: on every level of every case the policy
//      returns the same decision (stage or not), the same agglomerated decomposition (origins and
//      sizes), the same group map (groupOf) and the same owners (rootOf), and never Replicated
//      (flow's search always admits depth 0). Cases: flow's own telescope-test grids (32^3 forced
//      and 24^3 starved under a plain ORB), the 384^3 FoxBerry ladder at 24…1536 ranks, anisotropic
//      and badly factored grids, plain / aligned / WEIGHTED ORBs, minExtent 0, 2, 4 (flow's
//      default is 4), 1…1536 blocks.
//   B  amr's all-axes rule: the result lifts, has fewer blocks, and no deeper truncation would
//      have; a level grid that is odd on an axis falls through to Replicated.
//   C  the contract edges: InPlace is the identity; allowRepartition = true throws until S2; a
//      level grid that is not the decomposition's throws.
//   D  flow's FORCED telescope (`teleForce_ == L`, test-only) through `shallowestLiftableMerge`
//      called directly, as S4 will: flow's test_telescope_mpi case B verbatim (32^3 on flow's
//      aligned factory partition, nLevels 4, forced at level 1, telescoping on, flow's default
//      minExtent 4) at np = 2 and 4 — where the forced merge must fire — then every forced level
//      0..5 over Part A's grids and partitions, telescoping on and off. Same depth as well.
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
    chooseStageTarget(four, IVec<3>{32, 32, 32}, always, 0, /*allowRepartition=*/true);
  } catch (const std::logic_error&) {
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
  partD();
  std::printf(
      "  D  forced telescope: %ld ladder levels compared, %ld staged, %ld by the forced search, "
      "all identical\n",
      gD.compared, gD.staged, gD.forcedStaged);
  PECLET_CORE_CHECK(gD.forcedStaged > 0);
  PECLET_CORE_RETURN_TEST_RESULT();
}

// core — the TARGET of a coarse-level multigrid stage: onto which decomposition of a level's own
// grid the level moves when it can no longer coarsen in place (suite decision "Coarse-level
// redistribution lives in core; the hierarchies stay in the methods", amr/docs/
// amr_mg_core_boundary.md §4).
//
// A pure function of the current decomposition and the METHOD's lift predicate, replicated on every
// rank with no communication. Core does not know what makes a decomposition coarsenable — flow's
// rule is per-axis (every block even in origin and size on each axis whose global extent can still
// halve), amr's is all-axes — so the caller supplies it as `liftable(dec)`.
//
// The kinds:
//   InPlace      — no stage: the level coarsens on its current decomposition.
//   SiblingMerge — the ORB tree truncated (`BlockDecomposer::agglomerated(d)`): each target block
//                  is the union of a contiguous run of current blocks, owned by the lowest of their
//                  ranks. Moved by a group gather / scatter (stage_comm.hpp,
//                  redistribute_topology.hpp).
//   Repartition  — a fresh proportional ORB of the level grid on the first np_L parent ranks
//                  (repartitionTarget, §11.2 of the design). Moved point-to-point over the box
//                  intersections of the two decompositions (redistribute_topology.hpp).
//   Replicated   — the whole level as one block on EVERY rank. Moved by an Allgatherv.
//
// With `maxBlockCells == 0` the policy is flow's telescope search verbatim
// (flow/src/mac_cutcell_mg.hpp, CutcellMG::initMpi): so that flow's migration onto it is
// byte-identical. `maxBlockCells > 0` (design §11.1) bounds the block a stage may hand a rank.
#ifndef PECLET_CORE_DECOMP_STAGE_TARGET_HPP
#define PECLET_CORE_DECOMP_STAGE_TARGET_HPP

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

namespace peclet::core::decomp {

enum class StageKind { InPlace, SiblingMerge, Repartition, Replicated };

/// Where a level goes. `dec` is a decomposition of the SAME level grid as the current one.
template <int Dim>
struct StageTarget {
  StageKind kind = StageKind::InPlace;
  BlockDecomposer<Dim> dec;
  /// Target block -> the rank that owns it, in the PARENT communicator (whose rank r owns block r
  /// of the current decomposition). InPlace: the identity. SiblingMerge: the lowest rank of each
  /// group (`agglomerated`'s rootOf). Repartition: the identity on [0, np_L) — parent rank b owns
  /// target block b. Replicated: {0} — every rank holds the one block; rank 0 is recorded as its
  /// owner by the same lowest-rank convention.
  std::vector<int> ownerOf;
  /// Current block -> target block. SiblingMerge only (`agglomerated`'s groupOf); empty otherwise.
  std::vector<int> groupOf;
};

/// Smallest block extent over every block and axis — replicated, like the decomposition.
template <int Dim>
inline Index minBlockExtent(const BlockDecomposer<Dim>& d) {
  Index m = std::numeric_limits<Index>::max();
  for (const auto& s : d.sizes())
    for (int k = 0; k < Dim; ++k)
      m = s[k] < m ? s[k] : m;
  return m;
}

/// Cell count of the largest block — replicated, like the decomposition.
template <int Dim>
inline Index largestBlockCells(const BlockDecomposer<Dim>& d) {
  Index m = 0;
  for (const auto& s : d.sizes()) {
    Index c = 1;
    for (int k = 0; k < Dim; ++k)
      c *= s[k];
    m = c > m ? c : m;
  }
  return m;
}

/// A sibling merge the search chose: the tree depth it truncates at, the merged decomposition, and
/// its maps (`agglomerated(depth, &groupOf, &ownerOf)`).
template <int Dim>
struct SiblingMergeChoice {
  int depth = -1;
  BlockDecomposer<Dim> dec;
  std::vector<int> groupOf;  ///< current block -> merged block
  std::vector<int> ownerOf;  ///< merged block -> lowest parent rank of its group
};

/// The sibling-merge SEARCH on its own, with no trigger: the shallowest merge — fewest merges,
/// i.e. the LARGEST depth d in [0, treeDepth) — whose `agglomerated(d)`
///   * lifts (`liftable`, the method's predicate),
///   * has fewer blocks than `cur`, and
///   * when it has more than one block, keeps minBlockExtent >= 2*minExtent (0 disables),
/// or std::nullopt when no depth qualifies. Flow's depth search (CutcellMG::initMpi) verbatim.
///
/// `chooseStageTarget` calls it behind the trigger. A caller that must merge whatever the trigger
/// says calls it directly: that is how flow's test-only forced telescope (`teleForce_ == L`, which
/// runs the search even where in-place coarsening is legal) maps onto core, so the policy's own
/// signature carries no test hook.
template <int Dim, class Liftable>
std::optional<SiblingMergeChoice<Dim>> shallowestLiftableMerge(const BlockDecomposer<Dim>& cur,
                                                               Liftable&& liftable, int minExtent) {
  for (int d = cur.treeDepth() - 1; d >= 0; --d) {
    SiblingMergeChoice<Dim> m;
    m.depth = d;
    m.dec = cur.agglomerated(d, &m.groupOf, &m.ownerOf);
    bool ok = liftable(m.dec);
    // Fat enough to STAY above the threshold after the halving that follows (a single block always
    // qualifies): merging to blocks of 4 that become 2 on the next level would merge again there.
    if (ok && minExtent > 0 && m.dec.numBlocks() > 1 &&
        minBlockExtent(m.dec) < 2 * static_cast<Index>(minExtent))
      ok = false;
    if (ok && m.dec.numBlocks() < cur.numBlocks())
      return m;
  }
  return std::nullopt;
}

/// The Repartition candidate (design §11.2): a fresh proportional, unweighted, unaligned ORB of
/// the level grid on the first np_L parent ranks, or std::nullopt when none lifts. np_L is as many
/// ranks as the level's size justifies at `maxBlockCells` cells per rank, never more than the
/// current rank count and never fewer than one:
///   np_L = clamp(ceil(cells(levelGrid) / maxBlockCells), 1, cur.numBlocks())
/// When the extent rule is on (`minExtent > 0`) np_L is further capped at
/// prod_k max(1, floor(levelGrid[k] / (2*minExtent))) — blocks fat enough to survive the halving
/// that follows; with `minExtent == 0` there is no cap. Then np_L, the largest power of two <=
/// np_L, and its halvings down to 1 are tried in turn, and the first proportional ORB that lifts is
/// returned. That guarantees ONE in-place lift below the stage, not more: a power-of-two count on
/// an even grid keeps lifting, but np_L = 5 or 6 (the §6 heap / flat-bed fixtures of
/// test_stage_target) lifts once and stages again at the next level. Owners are the identity on
/// [0, n); groupOf is empty. A pure function, replicated on every rank.
/// Precondition: maxBlockCells > 0.
template <int Dim, class Liftable>
std::optional<StageTarget<Dim>> repartitionTarget(const BlockDecomposer<Dim>& cur,
                                                  const std::type_identity_t<IVec<Dim>>& levelGrid,
                                                  Liftable&& liftable, int minExtent,
                                                  Index maxBlockCells) {
  if (maxBlockCells <= 0)
    throw std::invalid_argument("repartitionTarget: maxBlockCells must be positive");
  const Index np = static_cast<Index>(cur.numBlocks());
  Index cells = 1;
  for (int k = 0; k < Dim; ++k)
    cells *= levelGrid[k];
  Index npL = (cells + maxBlockCells - 1) / maxBlockCells;
  npL = npL < 1 ? 1 : (npL > np ? np : npL);
  if (minExtent > 0) {
    Index cap = 1;
    for (int k = 0; k < Dim; ++k) {
      const Index f = levelGrid[k] / (2 * static_cast<Index>(minExtent));
      cap *= f > 1 ? f : 1;
    }
    npL = npL < cap ? npL : cap;
  }
  Index p2 = 1;  // the largest power of two <= npL
  while (2 * p2 <= npL)
    p2 *= 2;
  std::vector<Index> tries;  // npL, then p2 (unless it is npL), then its halvings down to 1
  if (npL != p2)
    tries.push_back(npL);
  for (Index n = p2; n >= 1; n /= 2)
    tries.push_back(n);
  for (const Index n : tries) {
    BlockDecomposer<Dim> r(static_cast<std::size_t>(n), levelGrid);
    if (liftable(r)) {
      StageTarget<Dim> t;
      t.kind = StageKind::Repartition;
      t.dec = std::move(r);
      t.ownerOf.resize(static_cast<std::size_t>(n));
      for (Index b = 0; b < n; ++b)
        t.ownerOf[static_cast<std::size_t>(b)] = static_cast<int>(b);
      return t;
    }
  }
  return std::nullopt;
}

/// Choose the stage target of a level whose current decomposition is `cur` over the level grid
/// `levelGrid` (which must equal `cur.globalSize()`).
///
/// `liftable(const BlockDecomposer<Dim>&) -> bool` is the METHOD's predicate: "every block can
/// coarsen in place one more time". `minExtent` is the economic trigger (0 disables it): a level
/// whose smallest block extent is below it is staged even when it could still coarsen in place,
/// and a merged candidate with more than one block must keep an extent of at least 2*minExtent.
/// `maxBlockCells` is the caller's "a rank never holds more cells of a coarse level than it holds
/// of its finest level" — the finest level's largest block, replicated; 0 disables Repartition.
///
/// Order (design §11.1):
///   1. InPlace when `cur` has one block, or when liftable(cur) and no block is below minExtent;
///   2. S := shallowestLiftableMerge(cur, liftable, minExtent);
///   3. SiblingMerge onto S when S exists and (maxBlockCells == 0 or S's largest block has at most
///      maxBlockCells cells);
///   4. else, when maxBlockCells > 0, Repartition onto repartitionTarget(...) if one lifts;
///   5. else SiblingMerge onto S if it exists — legal but heavy, never wrong;
///   6. else Replicated.
/// With maxBlockCells == 0 steps 4 and 5 are never taken, and the policy is flow's trigger +
/// search verbatim (steps 1, 2, 3, 6; step 6 is unreachable under flow's rule, where depth 0
/// always lifts). The one-block candidate agglomerated(0) is excluded by step 3 exactly when the
/// level is larger than one finest-level block.
///
/// Note what stays with the CALLER, because it is the method's rule, not core's: flow stages only
/// when telescoping is enabled and some axis of the level grid can still coarsen at all. With no
/// coarsenable axis flow's predicate is vacuously true, and a level below minExtent would
/// otherwise be sent to step 2. flow's forced telescope calls shallowestLiftableMerge directly.
///
/// What also stays in flow at S4: the outflow ghost-plane gathers (`teleGatherOutflowPlanes` /
/// `teleGatherPlane`, WO-R2) — they move the plane beyond the inner block on the global outflow
/// face, which the inner-cell RedistributeTopology does not describe.
template <int Dim, class Liftable>
StageTarget<Dim> chooseStageTarget(const BlockDecomposer<Dim>& cur,
                                   const std::type_identity_t<IVec<Dim>>& levelGrid,
                                   Liftable&& liftable, int minExtent, Index maxBlockCells = 0) {
  if (levelGrid != cur.globalSize())
    throw std::invalid_argument(
        "chooseStageTarget: levelGrid differs from the current decomposition's global size");
  if (maxBlockCells < 0)
    throw std::invalid_argument("chooseStageTarget: maxBlockCells must be >= 0 (0 disables)");

  const std::size_t nb = cur.numBlocks();
  StageTarget<Dim> t;

  const bool tooSmall = minExtent > 0 && minBlockExtent(cur) < static_cast<Index>(minExtent);
  if (nb <= 1 || (liftable(cur) && !tooSmall)) {
    t.kind = StageKind::InPlace;
    t.dec = cur;
    t.ownerOf.resize(nb);
    for (std::size_t b = 0; b < nb; ++b)
      t.ownerOf[b] = static_cast<int>(b);
    return t;
  }

  auto m = shallowestLiftableMerge(cur, liftable, minExtent);
  auto merge = [&] {
    t.kind = StageKind::SiblingMerge;
    t.dec = std::move(m->dec);
    t.ownerOf = std::move(m->ownerOf);
    t.groupOf = std::move(m->groupOf);
    return std::move(t);
  };
  if (m && (maxBlockCells == 0 || largestBlockCells(m->dec) <= maxBlockCells))
    return merge();

  if (maxBlockCells > 0)
    if (auto r = repartitionTarget(cur, levelGrid, liftable, minExtent, maxBlockCells))
      return std::move(*r);

  if (m)
    return merge();

  t.kind = StageKind::Replicated;
  t.dec.init(1, levelGrid);
  t.ownerOf = {0};
  return t;
}

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_DECOMP_STAGE_TARGET_HPP

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
//   Repartition  — a fresh proportional ORB of the level grid on fewer ranks. Planned; its movement
//                  (and the policy branch that returns it) is step S2 of the design.
//   Replicated   — the whole level as one block on EVERY rank. Moved by an Allgatherv.
//
// With `allowRepartition == false` the policy is flow's telescope search verbatim
// (flow/src/mac_cutcell_mg.hpp, CutcellMG::initMpi): so that flow's migration onto it is
// byte-identical.
#ifndef PECLET_CORE_DECOMP_STAGE_TARGET_HPP
#define PECLET_CORE_DECOMP_STAGE_TARGET_HPP

#include <cstddef>
#include <limits>
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
  /// group (`agglomerated`'s rootOf). Replicated: {0} — every rank holds the one block; rank 0 is
  /// recorded as its owner by the same lowest-rank convention.
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

/// Choose the stage target of a level whose current decomposition is `cur` over the level grid
/// `levelGrid` (which must equal `cur.globalSize()`).
///
/// `liftable(const BlockDecomposer<Dim>&) -> bool` is the METHOD's predicate: "every block can
/// coarsen in place one more time". `minExtent` is the economic trigger (0 disables it): a level
/// whose smallest block extent is below it is staged even when it could still coarsen in place,
/// and a merged candidate with more than one block must keep an extent of at least 2*minExtent.
///
/// Order (allowRepartition == false — flow's search, verbatim):
///   1. InPlace when `cur` has one block, or when liftable(cur) and no block is below minExtent;
///   2. else SiblingMerge onto agglomerated(d) for the LARGEST d in [0, treeDepth) with
///      liftable(agglomerated(d)), fewer blocks than `cur`, and (when it has more than one block)
///      minBlockExtent >= 2*minExtent;
///   3. else Replicated.
/// Note what stays with the CALLER, because it is the method's rule, not core's: flow stages only
/// when telescoping is enabled and some axis of the level grid can still coarsen at all. With no
/// coarsenable axis flow's predicate is vacuously true, and a level below minExtent would
/// otherwise be sent to step 2.
///
/// allowRepartition == true is reserved for S2 (the Repartition kind and its movement); it throws
/// until S2 settles that branch.
template <int Dim, class Liftable>
StageTarget<Dim> chooseStageTarget(const BlockDecomposer<Dim>& cur,
                                   const std::type_identity_t<IVec<Dim>>& levelGrid,
                                   Liftable&& liftable, int minExtent,
                                   bool allowRepartition = false) {
  if (levelGrid != cur.globalSize())
    throw std::invalid_argument(
        "chooseStageTarget: levelGrid differs from the current decomposition's global size");
  if (allowRepartition)
    throw std::logic_error(
        "chooseStageTarget: allowRepartition is not implemented yet (design step S2)");

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

  // Fewest merges (largest tree depth) at which the candidate lifts. depth 0 is the whole grid on
  // one block.
  for (int d = cur.treeDepth() - 1; d >= 0; --d) {
    std::vector<int> go, ro;
    BlockDecomposer<Dim> cand = cur.agglomerated(d, &go, &ro);
    bool ok = liftable(cand);
    // Fat enough to STAY above the threshold after the halving that follows (a single block always
    // qualifies): merging to blocks of 4 that become 2 on the next level would merge again there.
    if (ok && minExtent > 0 && cand.numBlocks() > 1 &&
        minBlockExtent(cand) < 2 * static_cast<Index>(minExtent))
      ok = false;
    if (ok && cand.numBlocks() < nb) {
      t.kind = StageKind::SiblingMerge;
      t.dec = std::move(cand);
      t.ownerOf = std::move(ro);
      t.groupOf = std::move(go);
      return t;
    }
  }

  t.kind = StageKind::Replicated;
  t.dec.init(1, levelGrid);
  t.ownerOf = {0};
  return t;
}

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_DECOMP_STAGE_TARGET_HPP

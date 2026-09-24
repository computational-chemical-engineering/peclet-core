// core — the communicators of a coarse-level multigrid stage (amr/docs/amr_mg_core_boundary.md §4).
//
// Given a `StageTarget` (stage_target.hpp), every rank of the parent communicator learns which
// target block it belongs to, whether it OWNS one (and so continues the hierarchy below the stage),
// and gets two communicators:
//
//   group — the ranks whose current blocks make up this rank's target block, in ascending parent
//           rank; the owner is group rank 0. The movement's gather / scatter runs on it.
//   sub   — the owners only, ranked by target block index (MPI_COMM_NULL on a non-owner). The
//           continued hierarchy runs on it.
//
// SiblingMerge reproduces flow's `Telescope` communicators exactly (flow/src/mac_cutcell_mg.hpp,
// CutcellMG::initMpi): MPI_Comm_split(parent, group, rank) and MPI_Comm_split(parent, owner ? 0 :
// MPI_UNDEFINED, group). Replicated: one group of every rank; every rank is an owner.
// Repartition (design §11.2): there are no groups — every parent rank takes part in the
// point-to-point movement, so `group` IS `parent` (the same handle, not a duplicate) and `members`
// is empty; the owners are parent ranks [0, np_L), `sub` = MPI_Comm_split(parent, rank < np_L ? 0
// : MPI_UNDEFINED, rank), and `myGroup` is -1 (a current block feeds several target blocks).
//
// The parent-rank convention is core's: parent rank r owns block r of the current decomposition.
// The struct OWNS `group` and `sub` (not `parent`, and not `group` when it is `parent`):
// non-copyable, movable, freed on destruction with the same MPI_Finalized guard as GridHalo.
#ifndef PECLET_CORE_DECOMP_STAGE_COMM_HPP
#define PECLET_CORE_DECOMP_STAGE_COMM_HPP

#include <stdexcept>
#include <utility>
#include <vector>

#include "peclet/core/common/mpi.hpp"
#include "peclet/core/decomp/stage_target.hpp"

namespace peclet::core::decomp {

/// The communicators of one coarse-level multigrid stage, made by makeStageComm (the file comment
/// spells out `group` and `sub` per StageKind).
///
/// Ownership: OWNS `group` and `sub` and frees them on destruction (guarded by MPI_Finalized, as
/// GridHalo); does NOT own `parent`, nor `group` when it is `parent` (Repartition). Non-copyable,
/// movable. `parent` must outlive it, and it must outlive every RedistributeTopology built on it —
/// the topology keeps the raw communicator handle, not a copy. Destruction frees communicators,
/// so destroy the StageComm of a stage on every rank of `parent` alike.
struct StageComm {
  MPI_Comm parent = MPI_COMM_NULL;  ///< not owned
  MPI_Comm group = MPI_COMM_NULL;   ///< this rank's target block's members; owner = group rank 0
  MPI_Comm sub = MPI_COMM_NULL;     ///< owners only, key = target block; MPI_COMM_NULL elsewhere
  StageKind kind = StageKind::InPlace;  ///< the kind of the StageTarget this was made for
  bool active = false;                  ///< this rank owns a target block
  int myTargetBlock = -1;               ///< the target block this rank owns, or -1
  int myGroup = -1;  ///< the target block this rank's current block moves into (-1: Repartition)
  std::vector<int>
      members;  ///< parent ranks of this rank's group, in group-comm order (owner first)

  StageComm() = default;
  StageComm(const StageComm&) = delete;
  StageComm& operator=(const StageComm&) = delete;
  /// Takes over `o`'s communicators; `o` is left empty (all MPI_COMM_NULL, nothing to free).
  StageComm(StageComm&& o) noexcept { swap(o); }
  /// Frees this object's own communicators first, then takes over `o`'s.
  StageComm& operator=(StageComm&& o) noexcept {
    if (this != &o) {
      release();
      swap(o);
    }
    return *this;
  }
  ~StageComm() { release(); }

 private:
  void swap(StageComm& o) noexcept {
    std::swap(parent, o.parent);
    std::swap(group, o.group);
    std::swap(sub, o.sub);
    std::swap(kind, o.kind);
    std::swap(active, o.active);
    std::swap(myTargetBlock, o.myTargetBlock);
    std::swap(myGroup, o.myGroup);
    members.swap(o.members);
  }
  void release() noexcept {
    int fin = 0;
    MPI_Finalized(&fin);
    if (!fin) {
      if (group != MPI_COMM_NULL && group != parent)  // Repartition: group IS parent
        MPI_Comm_free(&group);
      if (sub != MPI_COMM_NULL)
        MPI_Comm_free(&sub);
    }
    group = sub = MPI_COMM_NULL;
  }
};

/// Build the stage communicators for `t` on `parent`. Collective on `parent`, whose size must equal
/// the number of current blocks (`t.groupOf.size()` for SiblingMerge); every rank passes the same
/// replicated `t` (chooseStageTarget's result). Parent rank r is taken to own current block r.
///
/// Throws std::invalid_argument for an InPlace target (there is no stage), a SiblingMerge
/// `groupOf` that does not map one block per rank, or a malformed Repartition target (not 1..size
/// blocks, owners not the identity, or a non-empty `groupOf`); std::logic_error when a merge
/// group's owner is not its lowest rank. The std::invalid_argument checks read only `t` and the
/// size of `parent`, so with a replicated `t` they fire on every rank alike, before any
/// communication. The std::logic_error (a broken `agglomerated` invariant, not a caller error)
/// fires after the splits and on the offending group's ranks only: treat it as fatal.
template <int Dim>
StageComm makeStageComm(MPI_Comm parent, const StageTarget<Dim>& t) {
  if (t.kind == StageKind::InPlace)
    throw std::invalid_argument("makeStageComm: an InPlace target has no stage");
  int rank = 0, size = 1;
  MPI_Comm_rank(parent, &rank);
  MPI_Comm_size(parent, &size);

  StageComm c;
  c.parent = parent;
  c.kind = t.kind;
  if (t.kind == StageKind::SiblingMerge) {
    if (t.groupOf.size() != static_cast<std::size_t>(size))
      throw std::invalid_argument(
          "makeStageComm: groupOf must map every block of the current decomposition, one per rank");
    const int myGroup = t.groupOf[static_cast<std::size_t>(rank)];
    const bool owner = (rank == t.ownerOf[static_cast<std::size_t>(myGroup)]);
    MPI_Comm_split(parent, myGroup, rank, &c.group);
    MPI_Comm_split(parent, owner ? 0 : MPI_UNDEFINED, myGroup, &c.sub);
    c.active = owner;
    c.myGroup = myGroup;
    c.myTargetBlock = owner ? myGroup : -1;
    for (int b = 0; b < size; ++b)
      if (t.groupOf[static_cast<std::size_t>(b)] == myGroup)
        c.members.push_back(b);
    if (c.members.front() != t.ownerOf[static_cast<std::size_t>(myGroup)])
      throw std::logic_error("makeStageComm: a group's owner must be its lowest rank");
  } else if (t.kind == StageKind::Repartition) {
    const std::size_t npL = t.dec.numBlocks();
    if (!t.groupOf.empty() || t.ownerOf.size() != npL || npL < 1 ||
        npL > static_cast<std::size_t>(size))
      throw std::invalid_argument(
          "makeStageComm: a Repartition target needs 1..size blocks, ownerOf per block, no "
          "groupOf");
    for (std::size_t b = 0; b < npL; ++b)
      if (t.ownerOf[b] != static_cast<int>(b))
        throw std::invalid_argument(
            "makeStageComm: a Repartition target's owners must be the identity on [0, np_L)");
    const bool owner = static_cast<std::size_t>(rank) < npL;
    c.group = parent;
    MPI_Comm_split(parent, owner ? 0 : MPI_UNDEFINED, rank, &c.sub);
    c.active = owner;
    c.myGroup = -1;
    c.myTargetBlock = owner ? rank : -1;
  } else {  // Replicated
    MPI_Comm_split(parent, 0, rank, &c.group);
    MPI_Comm_split(parent, 0, rank, &c.sub);
    c.active = true;
    c.myGroup = 0;
    c.myTargetBlock = 0;
    c.members.resize(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r)
      c.members[static_cast<std::size_t>(r)] = r;
  }
  return c;
}

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_DECOMP_STAGE_COMM_HPP

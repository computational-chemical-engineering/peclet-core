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
//
// The parent-rank convention is core's: parent rank r owns block r of the current decomposition.
// The struct OWNS `group` and `sub` (not `parent`): non-copyable, movable, freed on destruction
// with the same MPI_Finalized guard as GridHalo.
#ifndef PECLET_CORE_DECOMP_STAGE_COMM_HPP
#define PECLET_CORE_DECOMP_STAGE_COMM_HPP

#include <stdexcept>
#include <utility>
#include <vector>

#include "peclet/core/common/mpi.hpp"
#include "peclet/core/decomp/stage_target.hpp"

namespace peclet::core::decomp {

struct StageComm {
  MPI_Comm parent = MPI_COMM_NULL;  ///< not owned
  MPI_Comm group = MPI_COMM_NULL;   ///< this rank's target block's members; owner = group rank 0
  MPI_Comm sub = MPI_COMM_NULL;     ///< owners only, key = target block; MPI_COMM_NULL elsewhere
  StageKind kind = StageKind::InPlace;
  bool active = false;     ///< this rank owns a target block
  int myTargetBlock = -1;  ///< the target block this rank owns, or -1
  int myGroup = -1;        ///< the target block this rank's current block moves into
  std::vector<int>
      members;  ///< parent ranks of this rank's group, in group-comm order (owner first)

  StageComm() = default;
  StageComm(const StageComm&) = delete;
  StageComm& operator=(const StageComm&) = delete;
  StageComm(StageComm&& o) noexcept { swap(o); }
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
      if (group != MPI_COMM_NULL)
        MPI_Comm_free(&group);
      if (sub != MPI_COMM_NULL)
        MPI_Comm_free(&sub);
    }
    group = sub = MPI_COMM_NULL;
  }
};

/// Build the stage communicators for `t` on `parent`. Collective on `parent`, whose size must equal
/// the number of current blocks (`t.groupOf.size()` for SiblingMerge). Throws for InPlace (there is
/// no stage) and Repartition (design step S2).
template <int Dim>
StageComm makeStageComm(MPI_Comm parent, const StageTarget<Dim>& t) {
  if (t.kind == StageKind::InPlace)
    throw std::invalid_argument("makeStageComm: an InPlace target has no stage");
  if (t.kind == StageKind::Repartition)
    throw std::logic_error("makeStageComm: Repartition is not implemented yet (design step S2)");
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

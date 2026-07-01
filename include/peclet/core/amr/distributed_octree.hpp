// core — distributed adaptive octree over the ORB block decomposition.
//
// The suite keeps the ORB *block* decomposition and gives each block its own
// local octree (block_octree.hpp). DistributedOctree ties the per-rank blocks
// together: ORB (peclet::core::decomp::BlockDecomposer) assigns a brick of global root
// cells to each rank, each rank builds a BlockOctree with the matching world
// geometry, and this class supplies the two pieces of distributed glue:
//
//   * balance()             — cross-block 2:1 (graded) balance to a *global*
//                             fixpoint. Local balance + an owner-based NBX round
//                             (a leaf at level Lf tells the owner of the cell just
//                             across each block face its level; the owner refines
//                             any covering leaf at level >= Lf+2), iterated until
//                             no rank refines. Detection is from the fine side, so
//                             a single probe per face is complete.
//   * faceNeighborGather()  — owner-based ghost exchange: for every local leaf and
//                             face it returns the neighbouring leaf's field value,
//                             local or remote (a two-round NBX request/reply). For
//                             a uniform octree this is exactly the structured-grid
//                             face-neighbour halo; the same call works graded.
//
// Owner lookups go through BlockDecomposer::ownerOf on global root-cell
// coordinates — the same owner-based, no-Cartesian-assumption pattern GridHaloTopology
// uses. Self-addressed messages are handled locally (no MPI send-to-self).
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON; uses the MPI shim (peclet/core/common/mpi.hpp,
// real MPI or the single-rank stub) and the NBX engine.
#ifndef PECLET_CORE_AMR_DISTRIBUTED_OCTREE_HPP
#define PECLET_CORE_AMR_DISTRIBUTED_OCTREE_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "morton/morton.hpp"
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/nbx.hpp"

namespace peclet::core::amr {

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DistributedOctree {
 public:
  using Octree = BlockOctree<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;
  using Coord = typename Octree::Coord;

  static constexpr double kNoNeighbor = -1e300;  ///< gather sentinel for "no neighbour"

  DistributedOctree() = default;

  /// Decompose a `globalRootSize` grid of root cells over the communicator and
  /// build this rank's local block octree (uniform, all leaves at level lmax).
  void init(IVec<Dim> globalRootSize, unsigned lmax, AmrGeometry<Dim> globalGeo,
            std::array<bool, Dim> periodic, MPI_Comm comm = MPI_COMM_WORLD) {
    comm_ = comm;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    globalRootSize_ = globalRootSize;
    lmax_ = lmax;
    globalGeo_ = globalGeo;
    periodic_ = periodic;
    rootSpan_ = Index(1) << lmax_;

    dec_.init(static_cast<std::size_t>(size_), globalRootSize_);
    auto blk = dec_.block(static_cast<std::size_t>(rank_));
    blockOriginRoot_ = blk.origin;
    blockBrick_ = blk.size;
    for (int d = 0; d < Dim; ++d) {
      blockFineOrigin_[d] = blockOriginRoot_[d] * rootSpan_;
      blockFineSize_[d] = blockBrick_[d] * rootSpan_;
      globalFineSize_[d] = globalRootSize_[d] * rootSpan_;
    }
    local_.init(blockBrick_, lmax_, blockOriginRoot_);
  }

  // ---- accessors ---------------------------------------------------------
  Octree& local() { return local_; }
  const Octree& local() const { return local_; }
  int rank() const { return rank_; }
  int size() const { return size_; }
  MPI_Comm comm() const { return comm_; }
  unsigned lmax() const { return lmax_; }
  const IVec<Dim>& blockOriginRoot() const { return blockOriginRoot_; }
  const IVec<Dim>& blockBrick() const { return blockBrick_; }
  const IVec<Dim>& globalRootSize() const { return globalRootSize_; }
  const IVec<Dim>& blockFineOrigin() const { return blockFineOrigin_; }
  const IVec<Dim>& globalFineSize() const { return globalFineSize_; }
  const std::array<bool, Dim>& periodic() const { return periodic_; }
  Index rootSpan() const { return rootSpan_; }
  double h0() const { return globalGeo_.h0; }
  const AmrGeometry<Dim>& globalGeometry() const { return globalGeo_; }

  /// Global root-cell coordinate of local leaf `i` (lmax==0: the leaf is one root cell).
  IVec<Dim> globalRootOf(Index i) const {
    auto o = M::from_code(local_.code(i)).decode();
    IVec<Dim> g{};
    for (int d = 0; d < Dim; ++d) g[d] = static_cast<Index>(o[d]) / rootSpan_ + blockOriginRoot_[d];
    return g;
  }

  /// Local leaf covering global root-cell `g`, or -1 if not owned by this rank
  /// (lmax==0). Used to build the nested fine↔coarse multigrid transfer maps.
  Index findGlobalRoot(const IVec<Dim>& g) const {
    std::array<Coord, Dim> lc{};
    for (int d = 0; d < Dim; ++d) {
      long v = g[d] - blockOriginRoot_[d];
      if (v < 0 || v >= blockBrick_[d]) return -1;
      lc[d] = static_cast<Coord>(v * rootSpan_);
    }
    return local_.find(M::encode(lc).code());
  }

  /// World geometry of this rank's block (the global geometry shifted to its origin).
  AmrGeometry<Dim> localGeometry() const {
    AmrGeometry<Dim> g = globalGeo_;
    for (int d = 0; d < Dim; ++d)
      g.origin[d] = globalGeo_.origin[d] + static_cast<Real>(blockFineOrigin_[d]) * globalGeo_.h0;
    return g;
  }

  /// Global Morton code of local leaf `i` (origin in *global* fine coordinates).
  Code globalCode(Index i) const {
    auto b = local_.bounds(i);
    std::array<Coord, Dim> g{};
    for (int d = 0; d < Dim; ++d) g[d] = static_cast<Coord>(b[0][d] + blockFineOrigin_[d]);
    return M::encode(g).code();
  }

  // ---- cross-block 2:1 balance ------------------------------------------

  /// Bring the whole distributed octree to a 2:1-balanced state. Returns the
  /// number of refinements this rank performed (local + cross-block).
  Index balance() {
    Index grandTotal = 0;
    for (;;) {
      Index work = local_.balance2to1();

      std::map<int, std::vector<char>> sendBufs;
      std::vector<Code> toRefine;
      auto consider = [&](const std::array<Coord, Dim>& gc, int Lf) {
        Index leaf = locateGlobal(gc);
        if (leaf >= 0 && static_cast<int>(local_.level(leaf)) >= Lf + 2)
          toRefine.push_back(local_.code(leaf));
      };

      for (Index i = 0; i < local_.numLeaves(); ++i) {
        const int Lf = static_cast<int>(local_.level(i));
        for (int axis = 0; axis < Dim; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) {
            std::array<Coord, Dim> gc{};
            int owner = -1;
            Index lnb = -1;
            if (neighborInfo(i, axis, dir, gc, owner, lnb) != Remote) continue;
            if (owner == rank_)
              consider(gc, Lf);
            else
              appendQuery(sendBufs[owner], gc, Lf);
          }
      }

      halo::NbxEngine eng(comm_);
      auto it = sendBufs.begin();
      eng.exchange(
          [&](std::vector<char>& out) -> int {
            if (it == sendBufs.end()) return -1;
            out = it->second;
            int dest = it->first;
            ++it;
            return dest;
          },
          [&](int, std::vector<char>& msg) {
            parseQueries(msg, [&](const std::array<Coord, Dim>& gc, int Lf) { consider(gc, Lf); });
          });

      std::sort(toRefine.begin(), toRefine.end());
      toRefine.erase(std::unique(toRefine.begin(), toRefine.end()), toRefine.end());
      Index nref = local_.refineIf(
          [&](Code c, unsigned) { return std::binary_search(toRefine.begin(), toRefine.end(), c); });
      work += nref;
      grandTotal += nref;

      long lw = static_cast<long>(work), gw = 0;
      MPI_Allreduce(&lw, &gw, 1, MPI_LONG, MPI_SUM, comm_);
      if (gw == 0) break;
    }
    return grandTotal;
  }

  // ---- dynamic load re-balancing ----------------------------------------

  /// Re-decompose by per-root-cell *octree-leaf count* and migrate leaves (with
  /// their field columns) to the new owners. The global mesh is unchanged — this
  /// is a pure redistribution of the same leaves, so it is exactly conservative
  /// and every cell's field value is preserved bit-for-bit; only ownership (and
  /// hence this rank's local octree + block geometry) changes. Each `fields[c]`
  /// is a column indexed by leaf slot (length numLeaves()); on return each holds
  /// the migrated/reordered column matching the new local leaf order.
  ///
  /// The caller is responsible for the mesh being 2:1-balanced beforehand (it
  /// stays balanced, since the leaf set is untouched). Returns the number of
  /// leaves this rank migrated away. MPI-optional: a single rank keeps everything.
  Index rebalance(std::vector<std::vector<double>>& fields) {
    const int K = static_cast<int>(fields.size());
    const Index n = local_.numLeaves();

    // 1. Weight grid: number of octree leaves under each *global root cell*,
    //    agreed across ranks (blocks are disjoint, so a SUM-Allreduce of the
    //    zero-padded local counts yields the full global grid).
    std::size_t ncells = 1;
    for (int d = 0; d < Dim; ++d) ncells *= static_cast<std::size_t>(globalRootSize_[d]);
    std::vector<double> localWeight(ncells, 0.0), weight(ncells, 0.0);
    for (Index i = 0; i < n; ++i)
      localWeight[static_cast<std::size_t>(dec_.linearGlobal(globalRootOf(i)))] += 1.0;
    MPI_Allreduce(localWeight.data(), weight.data(), static_cast<int>(ncells), MPI_DOUBLE, MPI_SUM,
                  comm_);

    // 2. Weighted re-decomposition over the same global root grid.
    decomp::BlockDecomposer<Dim> newDec(static_cast<std::size_t>(size_), globalRootSize_, weight);
    auto nblk = newDec.block(static_cast<std::size_t>(rank_));
    const IVec<Dim> newOriginRoot = nblk.origin;
    const IVec<Dim> newBrick = nblk.size;
    IVec<Dim> newFineOrigin{};
    for (int d = 0; d < Dim; ++d) newFineOrigin[d] = newOriginRoot[d] * rootSpan_;

    // 3. Classify every local leaf by its new owner; keep mine, pack the rest.
    //    Leaves are carried by *global* code so the receiver can rebase them.
    std::vector<Code> codes;
    std::vector<std::uint8_t> levels;
    std::vector<std::vector<double>> cols(static_cast<std::size_t>(K));
    std::map<int, std::vector<char>> sendBufs;
    Index migratedOut = 0;
    for (Index i = 0; i < n; ++i) {
      const Code gc = globalCode(i);
      const std::uint8_t lv = static_cast<std::uint8_t>(local_.level(i));
      const int newOwner = newDec.ownerOf(globalRootOf(i));
      if (newOwner == rank_) {
        codes.push_back(gc);
        levels.push_back(lv);
        for (int c = 0; c < K; ++c) cols[static_cast<std::size_t>(c)].push_back(fields[c][static_cast<std::size_t>(i)]);
      } else {
        appendLeaf(sendBufs[newOwner], gc, lv, fields, i, K);
        ++migratedOut;
      }
    }

    // 4. Deliver migrated leaves to their new owners (sparse NBX).
    {
      halo::NbxEngine eng(comm_);
      auto it = sendBufs.begin();
      eng.exchange(
          [&](std::vector<char>& out) -> int {
            if (it == sendBufs.end()) return -1;
            out = it->second;
            int dest = it->first;
            ++it;
            return dest;
          },
          [&](int, std::vector<char>& msg) {
            parseLeaves(msg, K, [&](Code gc, std::uint8_t lv, const double* comps) {
              codes.push_back(gc);
              levels.push_back(lv);
              for (int c = 0; c < K; ++c) cols[static_cast<std::size_t>(c)].push_back(comps[c]);
            });
          });
    }

    // 5. Rebase global codes to the new block origin and sort into Z-order,
    //    carrying the field columns along with the permutation.
    const std::size_t m = codes.size();
    std::vector<Code> localCodes(m);
    for (std::size_t j = 0; j < m; ++j) {
      auto g = M::from_code(codes[j]).decode();
      std::array<Coord, Dim> lc{};
      for (int d = 0; d < Dim; ++d) lc[d] = static_cast<Coord>(g[d] - newFineOrigin[d]);
      localCodes[j] = M::encode(lc).code();
    }
    std::vector<std::size_t> ord(m);
    for (std::size_t j = 0; j < m; ++j) ord[j] = j;
    std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) { return localCodes[a] < localCodes[b]; });

    std::vector<Code> sortedCodes(m);
    std::vector<std::uint8_t> sortedLevels(m);
    std::vector<std::vector<double>> sortedCols(static_cast<std::size_t>(K), std::vector<double>(m));
    for (std::size_t j = 0; j < m; ++j) {
      sortedCodes[j] = localCodes[ord[j]];
      sortedLevels[j] = levels[ord[j]];
      for (int c = 0; c < K; ++c) sortedCols[static_cast<std::size_t>(c)][j] = cols[static_cast<std::size_t>(c)][ord[j]];
    }

    // 6. Install the new decomposition, block geometry and local octree.
    local_.assign(newBrick, lmax_, newOriginRoot, std::move(sortedCodes), std::move(sortedLevels));
    dec_ = newDec;
    blockOriginRoot_ = newOriginRoot;
    blockBrick_ = newBrick;
    blockFineOrigin_ = newFineOrigin;
    for (int d = 0; d < Dim; ++d) blockFineSize_[d] = newBrick[d] * rootSpan_;
    // globalFineSize_ is decomposition-independent and unchanged.
    for (int c = 0; c < K; ++c) fields[c].swap(sortedCols[static_cast<std::size_t>(c)]);
    return migratedOut;
  }

  // ---- owner-based face-neighbour gather (the halo) ---------------------

  /// For each local leaf and each of the 2*Dim faces, the neighbouring leaf's
  /// field value. Layout: out[i*(2*Dim) + 2*axis + (dir>0 ? 0 : 1)]. Domain
  /// boundaries (no neighbour) get `sentinel`. `field` is indexed by leaf slot.
  std::vector<double> faceNeighborGather(const std::vector<double>& field,
                                         double sentinel = kNoNeighbor) const {
    const Index n = local_.numLeaves();
    const int F = 2 * Dim;
    std::vector<double> out(static_cast<std::size_t>(n) * F, sentinel);

    // Phase 1: fill in-block / domain directly; build remote requests.
    std::map<int, std::vector<char>> reqBufs;
    for (Index i = 0; i < n; ++i)
      for (int axis = 0; axis < Dim; ++axis)
        for (int dir = -1; dir <= 1; dir += 2) {
          const int slot = faceSlot(i, axis, dir, F);
          std::array<Coord, Dim> gc{};
          int owner = -1;
          Index lnb = -1;
          NbState st = neighborInfo(i, axis, dir, gc, owner, lnb);
          if (st == InBlock) {
            if (lnb >= 0) out[static_cast<std::size_t>(slot)] = field[static_cast<std::size_t>(lnb)];
          } else if (st == Remote) {
            if (owner == rank_) {
              Index leaf = locateGlobal(gc);
              if (leaf >= 0) out[static_cast<std::size_t>(slot)] = field[static_cast<std::size_t>(leaf)];
            } else {
              appendRequest(reqBufs[owner], gc, static_cast<std::int64_t>(slot));
            }
          }  // DomainNone -> leave sentinel
        }

    // Phase 2: owners answer requests; requesters write replies into `out`.
    std::map<int, std::vector<char>> replyBufs;
    {
      halo::NbxEngine eng(comm_);
      auto it = reqBufs.begin();
      eng.exchange(
          [&](std::vector<char>& outb) -> int {
            if (it == reqBufs.end()) return -1;
            outb = it->second;
            int dest = it->first;
            ++it;
            return dest;
          },
          [&](int src, std::vector<char>& msg) {
            parseRequests(msg, [&](const std::array<Coord, Dim>& gc, std::int64_t reqId) {
              Index leaf = locateGlobal(gc);
              double v = (leaf >= 0) ? field[static_cast<std::size_t>(leaf)] : sentinel;
              appendReply(replyBufs[src], reqId, v);
            });
          },
          /*tag=*/11);  // distinct tag: keep request/reply rounds from aliasing in NBX
    }
    {
      halo::NbxEngine eng(comm_);
      auto it = replyBufs.begin();
      eng.exchange(
          [&](std::vector<char>& outb) -> int {
            if (it == replyBufs.end()) return -1;
            outb = it->second;
            int dest = it->first;
            ++it;
            return dest;
          },
          [&](int, std::vector<char>& msg) {
            parseReplies(msg, [&](std::int64_t reqId, double v) {
              out[static_cast<std::size_t>(reqId)] = v;
            });
          },
          /*tag=*/12);  // reply round: distinct tag from the request round (11)
    }
    return out;
  }

  /// Static topology of the face-neighbour gather: which out-slots are filled from a local leaf
  /// (`directSlot`/`directLeaf`) and which must be gathered from an owner (`remoteCoords` at
  /// `remoteSlot`). This is the per-face `neighborInfo` classification, which depends only on the
  /// decomposition — not the field — so a matvec loop can build it once and reuse it (C1), mirroring
  /// how DistributedFvOperator caches its ghost coords. Slots left in neither list keep the sentinel
  /// (domain boundary / no neighbour). Invalidated by `rebalance` (rebuild after a decomposition
  /// change); the AMR solvers hold the decomposition fixed across a solve.
  struct FaceGatherPlan {
    std::vector<Index> directSlot, directLeaf;        ///< out[directSlot[k]] = field[directLeaf[k]]
    std::vector<std::array<Coord, Dim>> remoteCoords;  ///< owner-gathered coords (incl. owner==rank)
    std::vector<Index> remoteSlot;                     ///< out[remoteSlot[k]] = covered value
    Index nFaces = 0;                                  ///< local leaves * 2*Dim
  };

  FaceGatherPlan buildFaceGatherPlan() const {
    const Index n = local_.numLeaves();
    const int F = 2 * Dim;
    FaceGatherPlan p;
    p.nFaces = n * F;
    for (Index i = 0; i < n; ++i)
      for (int axis = 0; axis < Dim; ++axis)
        for (int dir = -1; dir <= 1; dir += 2) {
          const int slot = faceSlot(i, axis, dir, F);
          std::array<Coord, Dim> gc{};
          int owner = -1;
          Index lnb = -1;
          NbState st = neighborInfo(i, axis, dir, gc, owner, lnb);
          if (st == InBlock) {
            if (lnb >= 0) {
              p.directSlot.push_back(slot);
              p.directLeaf.push_back(lnb);
            }
          } else if (st == Remote) {
            // owner==rank resolves locally inside coverValues just as the un-planned path does, so
            // both remote subcases route through remoteCoords — identical values, identical slots.
            p.remoteCoords.push_back(gc);
            p.remoteSlot.push_back(slot);
          }  // DomainNone -> leave sentinel
        }
    return p;
  }

  /// Face-neighbour gather using a precomputed FaceGatherPlan: only the field values move, the
  /// classification does not re-run. Bit-identical to `faceNeighborGather(field, sentinel)`.
  std::vector<double> faceNeighborGather(const FaceGatherPlan& plan, const std::vector<double>& field,
                                         double sentinel = kNoNeighbor) const {
    std::vector<double> out(static_cast<std::size_t>(plan.nFaces), sentinel);
    for (std::size_t k = 0; k < plan.directSlot.size(); ++k)
      out[static_cast<std::size_t>(plan.directSlot[k])] =
          field[static_cast<std::size_t>(plan.directLeaf[k])];
    std::vector<double> vals = coverValues(plan.remoteCoords, field, sentinel);
    for (std::size_t k = 0; k < plan.remoteSlot.size(); ++k)
      out[static_cast<std::size_t>(plan.remoteSlot[k])] = vals[k];
    return out;
  }

  // ---- device-resident gather halo topology (C2) -------------------------

  /// Flattened, value-only topology for the face-neighbour gather, established ONCE: the per-matvec
  /// exchange then moves only `double` values (no coords, no locateGlobal) over a fixed graph — the
  /// octree analogue of GridHaloTopology::flatten(). A device DistributedGatherHalo mirrors only the
  /// compact send/recv buffers across MPI (à la grid_halo.hpp); the field stays on the device.
  struct GatherHaloTopology {
    // Locally-resolved faces (InBlock + Remote-but-owner==rank): out[localSlot[k]] = field[localLeaf[k]].
    std::vector<Index> localSlot, localLeaf;
    // Owner side of the value exchange: gather these local leaves (−1 ⇒ send sentinel) and send to the
    // requester ranks (concatenated per sendRanks/sendCounts, in received reqId order).
    std::vector<int> sendRanks, sendCounts;
    std::vector<Index> sendLeaf;
    // Requester side: scatter the received values into these out-slots (per recvRanks/recvCounts).
    std::vector<int> recvRanks, recvCounts;
    std::vector<Index> recvSlot;
    Index nFaces = 0;
  };

  /// Build the value-only gather topology from a FaceGatherPlan: classify each remote coord by owner
  /// (owner==rank folds into the local fills), then ONE NBX round in which each owner learns which of
  /// its local leaves to send to each requester (locateGlobal happens here, once — never per matvec).
  GatherHaloTopology buildGatherHaloTopology(const FaceGatherPlan& plan) const {
    GatherHaloTopology t;
    t.nFaces = plan.nFaces;
    t.localSlot = plan.directSlot;  // InBlock direct fills
    t.localLeaf = plan.directLeaf;
    // Classify remote coords; group cross-rank requests by owner with a per-owner reqId = its position.
    std::map<int, std::vector<char>> req;             // owner -> serialized (coord, reqId) requests
    std::map<int, std::vector<Index>> recvSlotByOwner;  // owner -> out-slots in reqId order
    for (std::size_t k = 0; k < plan.remoteCoords.size(); ++k) {
      const std::array<Coord, Dim>& gc = plan.remoteCoords[k];
      const Index slot = plan.remoteSlot[k];
      const int owner = ownerOfFine(gc);
      if (owner == rank_) {
        const Index leaf = locateGlobal(gc);
        if (leaf >= 0) {
          t.localSlot.push_back(slot);
          t.localLeaf.push_back(leaf);
        }  // leaf < 0 ⇒ leave sentinel
      } else {
        const std::int64_t reqId = static_cast<std::int64_t>(recvSlotByOwner[owner].size());
        appendRequest(req[owner], gc, reqId);
        recvSlotByOwner[owner].push_back(slot);
      }
    }
    for (auto& kv : recvSlotByOwner) {  // requester side (deterministic owner order from std::map)
      t.recvRanks.push_back(kv.first);
      t.recvCounts.push_back(static_cast<int>(kv.second.size()));
      for (Index s : kv.second) t.recvSlot.push_back(s);
    }
    // One NBX round: owners receive coord requests and resolve them to local leaves (in reqId order).
    std::map<int, std::vector<Index>> sendLeafBySrc;
    {
      halo::NbxEngine eng(comm_);
      auto it = req.begin();
      eng.exchange(
          [&](std::vector<char>& o) -> int {
            if (it == req.end()) return -1;
            o = it->second;
            int d = it->first;
            ++it;
            return d;
          },
          [&](int src, std::vector<char>& msg) {
            std::vector<Index>& leaves = sendLeafBySrc[src];
            parseRequests(msg, [&](const std::array<Coord, Dim>& gc, std::int64_t reqId) {
              const Index leaf = locateGlobal(gc);
              if (static_cast<std::int64_t>(leaves.size()) <= reqId)
                leaves.resize(static_cast<std::size_t>(reqId) + 1, -1);
              leaves[static_cast<std::size_t>(reqId)] = leaf;
            });
          },
          /*tag=*/31);
    }
    for (auto& kv : sendLeafBySrc) {  // owner side
      t.sendRanks.push_back(kv.first);
      t.sendCounts.push_back(static_cast<int>(kv.second.size()));
      for (Index l : kv.second) t.sendLeaf.push_back(l);
    }
    return t;
  }

  // ---- by-coordinate owner gathers (graded consistent-operator halo) ----

  /// Classification of leaf `i`'s face on (axis,dir): {state, global fine probe
  /// `gc`, `owner`, in-block neighbour `localNb`}. state 0=InBlock,1=Remote,2=None.
  struct FaceInfo {
    int state = 2;
    std::array<Coord, Dim> gc{};
    int owner = -1;
    Index localNb = -1;
  };
  FaceInfo faceAcross(Index i, int axis, int dir) const {
    FaceInfo f;
    f.state = static_cast<int>(neighborInfo(i, axis, dir, f.gc, f.owner, f.localNb));
    return f;
  }

  /// For each global fine coord (already wrapped into the domain), the *level* of
  /// the covering leaf on its owner, or -1 if none. Owner-based request/reply.
  std::vector<int> coverLevels(const std::vector<std::array<Coord, Dim>>& coords) const {
    std::vector<double> out(coords.size(), -1.0);
    std::map<int, std::vector<char>> req;
    for (std::size_t idx = 0; idx < coords.size(); ++idx) {
      int owner = ownerOfFine(coords[idx]);
      if (owner == rank_) {
        Index leaf = locateGlobal(coords[idx]);
        out[idx] = (leaf >= 0) ? static_cast<double>(local_.level(leaf)) : -1.0;
      } else {
        appendRequest(req[owner], coords[idx], static_cast<std::int64_t>(idx));
      }
    }
    requestReply(
        req, [&](const std::array<Coord, Dim>& gc) -> double {
          Index leaf = locateGlobal(gc);
          return (leaf >= 0) ? static_cast<double>(local_.level(leaf)) : -1.0;
        },
        out, /*tagA=*/21, /*tagB=*/22);
    std::vector<int> lv(coords.size());
    for (std::size_t i = 0; i < out.size(); ++i) lv[i] = static_cast<int>(out[i]);
    return lv;
  }

  /// For each global fine coord, the covering leaf's `field` value on its owner
  /// (sentinel if none). Owner-based request/reply — the per-matvec ghost gather.
  std::vector<double> coverValues(const std::vector<std::array<Coord, Dim>>& coords,
                                  const std::vector<double>& field,
                                  double sentinel = kNoNeighbor) const {
    std::vector<double> out(coords.size(), sentinel);
    std::map<int, std::vector<char>> req;
    for (std::size_t idx = 0; idx < coords.size(); ++idx) {
      int owner = ownerOfFine(coords[idx]);
      if (owner == rank_) {
        Index leaf = locateGlobal(coords[idx]);
        if (leaf >= 0) out[idx] = field[static_cast<std::size_t>(leaf)];
      } else {
        appendRequest(req[owner], coords[idx], static_cast<std::int64_t>(idx));
      }
    }
    requestReply(
        req, [&](const std::array<Coord, Dim>& gc) -> double {
          Index leaf = locateGlobal(gc);
          return (leaf >= 0) ? field[static_cast<std::size_t>(leaf)] : sentinel;
        },
        out, /*tagA=*/23, /*tagB=*/24);
    return out;
  }

 private:
  enum NbState { InBlock, Remote, DomainNone };

  int ownerOfFine(const std::array<Coord, Dim>& gc) const {
    IVec<Dim> rootCell{};
    for (int d = 0; d < Dim; ++d) rootCell[d] = static_cast<Index>(gc[d]) / rootSpan_;
    return dec_.ownerOf(rootCell);
  }

  /// Generic owner request/reply: owners answer each requested coord via `respond`,
  /// requesters write replies into out[reqId]. Distinct (tagA,tagB) per use so the
  /// two consecutive NBX rounds never alias.
  template <class Responder>
  void requestReply(std::map<int, std::vector<char>>& req, Responder&& respond,
                    std::vector<double>& out, int tagA, int tagB) const {
    std::map<int, std::vector<char>> rep;
    {
      halo::NbxEngine eng(comm_);
      auto it = req.begin();
      eng.exchange(
          [&](std::vector<char>& o) -> int {
            if (it == req.end()) return -1;
            o = it->second;
            int d = it->first;
            ++it;
            return d;
          },
          [&](int src, std::vector<char>& msg) {
            parseRequests(msg, [&](const std::array<Coord, Dim>& gc, std::int64_t reqId) {
              appendReply(rep[src], reqId, respond(gc));
            });
          },
          tagA);
    }
    {
      halo::NbxEngine eng(comm_);
      auto it = rep.begin();
      eng.exchange(
          [&](std::vector<char>& o) -> int {
            if (it == rep.end()) return -1;
            o = it->second;
            int d = it->first;
            ++it;
            return d;
          },
          [&](int, std::vector<char>& msg) {
            parseReplies(msg,
                         [&](std::int64_t reqId, double v) { out[static_cast<std::size_t>(reqId)] = v; });
          },
          tagB);
    }
  }

  static int faceSlot(Index i, int axis, int dir, int F) {
    return static_cast<int>(i) * F + 2 * axis + (dir > 0 ? 0 : 1);
  }

  /// Classify the neighbour across leaf `i`'s face on `axis`,`dir`. For InBlock,
  /// `localNb` is the covering local leaf; for Remote, `gc` is the global fine
  /// probe point and `owner` its rank.
  NbState neighborInfo(Index i, int axis, int dir, std::array<Coord, Dim>& gc, int& owner,
                       Index& localNb) const {
    auto b = local_.bounds(i);
    const auto& lo = b[0];
    const Coord size = Coord(Coord(1) << local_.level(i));
    const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(size)
                              : static_cast<long>(lo[axis]) - 1;

    if (pc >= 0 && pc < static_cast<long>(blockFineSize_[axis])) {
      std::array<Coord, Dim> p = lo;
      p[axis] = static_cast<Coord>(pc);
      localNb = local_.find(M::encode(p).code());
      return InBlock;
    }

    std::array<long, Dim> g{};
    for (int d = 0; d < Dim; ++d) g[d] = static_cast<long>(blockFineOrigin_[d]) + static_cast<long>(lo[d]);
    g[axis] = static_cast<long>(blockFineOrigin_[axis]) + pc;
    for (int d = 0; d < Dim; ++d) {
      const long gf = static_cast<long>(globalFineSize_[d]);
      if (g[d] < 0 || g[d] >= gf) {
        if (!periodic_[d]) return DomainNone;
        g[d] = ((g[d] % gf) + gf) % gf;
      }
    }
    IVec<Dim> rootCell{};
    for (int d = 0; d < Dim; ++d) {
      gc[d] = static_cast<Coord>(g[d]);
      rootCell[d] = static_cast<Index>(g[d] / static_cast<long>(rootSpan_));
    }
    owner = dec_.ownerOf(rootCell);
    return Remote;
  }

  /// Locate the local leaf covering a *global* fine coordinate that lies in this
  /// rank's block (used by message handlers).
  Index locateGlobal(const std::array<Coord, Dim>& gc) const {
    std::array<Coord, Dim> lc{};
    for (int d = 0; d < Dim; ++d) lc[d] = static_cast<Coord>(gc[d] - blockFineOrigin_[d]);
    return local_.find(M::encode(lc).code());
  }

  // ---- message (de)serialization ----------------------------------------
  static void appendBytes(std::vector<char>& buf, const void* p, std::size_t n) {
    const char* c = static_cast<const char*>(p);
    buf.insert(buf.end(), c, c + n);
  }
  static void appendQuery(std::vector<char>& buf, const std::array<Coord, Dim>& gc, int Lf) {
    for (int d = 0; d < Dim; ++d) {
      std::int64_t v = static_cast<std::int64_t>(gc[d]);
      appendBytes(buf, &v, sizeof(v));
    }
    std::int32_t l = Lf;
    appendBytes(buf, &l, sizeof(l));
  }
  template <class Fn>
  static void parseQueries(const std::vector<char>& buf, Fn&& fn) {
    std::size_t off = 0;
    const std::size_t item = Dim * sizeof(std::int64_t) + sizeof(std::int32_t);
    while (off + item <= buf.size()) {
      std::array<Coord, Dim> gc{};
      for (int d = 0; d < Dim; ++d) {
        std::int64_t v;
        std::memcpy(&v, buf.data() + off, sizeof(v));
        off += sizeof(v);
        gc[d] = static_cast<Coord>(v);
      }
      std::int32_t l;
      std::memcpy(&l, buf.data() + off, sizeof(l));
      off += sizeof(l);
      fn(gc, static_cast<int>(l));
    }
  }
  static void appendRequest(std::vector<char>& buf, const std::array<Coord, Dim>& gc,
                            std::int64_t reqId) {
    for (int d = 0; d < Dim; ++d) {
      std::int64_t v = static_cast<std::int64_t>(gc[d]);
      appendBytes(buf, &v, sizeof(v));
    }
    appendBytes(buf, &reqId, sizeof(reqId));
  }
  template <class Fn>
  static void parseRequests(const std::vector<char>& buf, Fn&& fn) {
    std::size_t off = 0;
    const std::size_t item = (Dim + 1) * sizeof(std::int64_t);
    while (off + item <= buf.size()) {
      std::array<Coord, Dim> gc{};
      for (int d = 0; d < Dim; ++d) {
        std::int64_t v;
        std::memcpy(&v, buf.data() + off, sizeof(v));
        off += sizeof(v);
        gc[d] = static_cast<Coord>(v);
      }
      std::int64_t reqId;
      std::memcpy(&reqId, buf.data() + off, sizeof(reqId));
      off += sizeof(reqId);
      fn(gc, reqId);
    }
  }
  /// Migration record for leaf `i`: global code (int64), level (int32), K field
  /// components (double). Self-contained so the receiver can rebase and re-sort it.
  static void appendLeaf(std::vector<char>& buf, Code gc, std::uint8_t level,
                         const std::vector<std::vector<double>>& fields, Index i, int K) {
    std::int64_t c = static_cast<std::int64_t>(gc);
    appendBytes(buf, &c, sizeof(c));
    std::int32_t l = static_cast<std::int32_t>(level);
    appendBytes(buf, &l, sizeof(l));
    for (int k = 0; k < K; ++k) {
      double v = fields[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)];
      appendBytes(buf, &v, sizeof(v));
    }
  }
  template <class Fn>
  static void parseLeaves(const std::vector<char>& buf, int K, Fn&& fn) {
    std::size_t off = 0;
    const std::size_t item =
        sizeof(std::int64_t) + sizeof(std::int32_t) + static_cast<std::size_t>(K) * sizeof(double);
    std::vector<double> comps(static_cast<std::size_t>(K));
    while (off + item <= buf.size()) {
      std::int64_t c;
      std::memcpy(&c, buf.data() + off, sizeof(c));
      off += sizeof(c);
      std::int32_t l;
      std::memcpy(&l, buf.data() + off, sizeof(l));
      off += sizeof(l);
      for (int k = 0; k < K; ++k) {
        std::memcpy(&comps[static_cast<std::size_t>(k)], buf.data() + off, sizeof(double));
        off += sizeof(double);
      }
      fn(static_cast<Code>(c), static_cast<std::uint8_t>(l), comps.data());
    }
  }
  static void appendReply(std::vector<char>& buf, std::int64_t reqId, double v) {
    appendBytes(buf, &reqId, sizeof(reqId));
    appendBytes(buf, &v, sizeof(v));
  }
  template <class Fn>
  static void parseReplies(const std::vector<char>& buf, Fn&& fn) {
    std::size_t off = 0;
    const std::size_t item = sizeof(std::int64_t) + sizeof(double);
    while (off + item <= buf.size()) {
      std::int64_t reqId;
      std::memcpy(&reqId, buf.data() + off, sizeof(reqId));
      off += sizeof(reqId);
      double v;
      std::memcpy(&v, buf.data() + off, sizeof(v));
      off += sizeof(v);
      fn(reqId, v);
    }
  }

  MPI_Comm comm_ = MPI_COMM_WORLD;
  int rank_ = 0;
  int size_ = 1;
  decomp::BlockDecomposer<Dim> dec_;
  Octree local_;
  AmrGeometry<Dim> globalGeo_{};
  IVec<Dim> globalRootSize_{};
  IVec<Dim> blockOriginRoot_{};
  IVec<Dim> blockBrick_{};
  IVec<Dim> blockFineOrigin_{};
  IVec<Dim> blockFineSize_{};
  IVec<Dim> globalFineSize_{};
  std::array<bool, Dim> periodic_{};
  unsigned lmax_ = 0;
  Index rootSpan_ = 1;
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_DISTRIBUTED_OCTREE_HPP

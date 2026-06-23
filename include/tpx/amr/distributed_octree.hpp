// transport-core — distributed adaptive octree over the ORB block decomposition.
//
// The suite keeps the ORB *block* decomposition and gives each block its own
// local octree (block_octree.hpp). DistributedOctree ties the per-rank blocks
// together: ORB (tpx::decomp::BlockDecomposer) assigns a brick of global root
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
// coordinates — the same owner-based, no-Cartesian-assumption pattern GridHalo
// uses. Self-addressed messages are handled locally (no MPI send-to-self).
//
// Header-only, guarded by TPX_HAVE_MORTON; uses the MPI shim (tpx/common/mpi.hpp,
// real MPI or the single-rank stub) and the NBX engine.
#ifndef TPX_AMR_DISTRIBUTED_OCTREE_HPP
#define TPX_AMR_DISTRIBUTED_OCTREE_HPP

#ifdef TPX_HAVE_MORTON

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "morton/morton.hpp"
#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/common/mpi.hpp"
#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/nbx.hpp"

namespace tpx::amr {

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

 private:
  enum NbState { InBlock, Remote, DomainNone };

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

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DISTRIBUTED_OCTREE_HPP

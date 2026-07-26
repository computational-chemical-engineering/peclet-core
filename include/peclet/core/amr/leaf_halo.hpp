// core — LeafHalo: the ±2 leaf ghost registry + value halo for the distributed AmrFlow.
//
// The distributed flow solver (docs/amr_distributed_flow.md) keeps every per-leaf field in an
// extended array: local leaves [0, nLocal), ghost leaves [nLocal, nLocal + nGhost). Every CSR the
// step reads (momentum, FaceGeom, closure overlays, cf deltas) may reference ghost slots; the
// kernels launch over local rows only and are otherwise unchanged. This header supplies the two
// pieces that make that work:
//
//   LeafHalo (host)         — the ghost REGISTRY + resolve seam the operator builders thread
//     their neighbour probes through. resolve(globalFineCoord) → covering local leaf | ghost
//     slot | kPending. Unknown coords queue as misses; resolveMisses() is ONE collective
//     coverLevels round (owner request/reply) that learns each miss's covering-leaf level, then
//     canonicalizes the coord to the leaf's global ANCHOR (lo corner) so any number of probes
//     into the same remote leaf dedup to ONE ghost slot. Builders run their enumeration pass to
//     a miss-collect fixpoint:  for(;;){ attempt build; if(halo.resolveMisses()==0) break; }
//     — the reach is bounded (±2 cells) so this terminates in ≤3 rounds, and the Allreduce
//     inside resolveMisses keeps the collectives matched across ranks. finalize() then
//     establishes the owner↔ghost value topology ONCE (DistributedOctree::buildGatherHaloTopology
//     — one NBX round does the owner-side locateGlobal; never again per exchange).
//
//   LeafHaloExchange (device, Kokkos-guarded) — the per-use value refresh: device pack of the
//     owner's local values → compact host-staged MPI buffers (GPU-aware opt-in, exactly as
//     GridHalo / DistributedGatherHalo) → device scatter into the ghost tail of the extended
//     field. exchange3 batches the 3 velocity components into one message round (the projection
//     syncs 3-vectors at every point; per-field latency would triple the halo count).
//
// np = 1: every wrapped probe lands back in the block ⇒ zero ghosts, resolve() returns local
// leaves only, and the distributed build path is bit-identical to the single-rank one by
// construction (no code touches a ghost slot that does not exist).
//
// Bit-exactness: ghost values are unmodified copies of the owner's doubles, so any consumer
// whose per-row arithmetic order is decomposition-independent stays bit-exact WORLD==SELF (the
// DistributedFvOperator argument). GPU is tolerance-not-bit-exact vs host (FMA) as documented.
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON. The host part compiles without Kokkos; the
// device exchanger is guarded on KOKKOS_INLINE_FUNCTION (include after a Kokkos-carrying header
// in device TUs, like ghost_projection.hpp).
#ifndef PECLET_CORE_AMR_LEAF_HALO_HPP
#define PECLET_CORE_AMR_LEAF_HALO_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>
#include <map>
#include <stdexcept>
#include <vector>

#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/types.hpp"

// Device-exchanger dependencies, only when the TU already carries Kokkos (the host part of this
// header must stay compilable without it).
#ifdef KOKKOS_INLINE_FUNCTION
#include "peclet/core/common/view.hpp"
#include "peclet/core/halo/grid_halo.hpp"  // halo::detail::gpuAwareMpi() (GPU-aware opt-in)
#endif

namespace peclet::core::amr {

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class LeafHalo {
 public:
  using DO = DistributedOctree<Dim, Bits>;
  using M = typename DO::M;
  using Coord = typename DO::Coord;
  using CoordArr = std::array<Coord, Dim>;

  static constexpr Index kPending = -2;  ///< resolve(): queued as a miss (call resolveMisses)
  static constexpr Index kNone = -1;     ///< resolveGlobal(): exits a non-periodic axis

  void init(DO& d) {
    d_ = &d;
    nLocal_ = d.local().numLeaves();
    for (int a = 0; a < Dim; ++a) {
      fineOrigin_[a] = static_cast<long>(d.blockFineOrigin()[a]);
      fineSize_[a] = static_cast<long>(d.blockBrick()[a]) * static_cast<long>(d.rootSpan());
      globalFine_[a] = static_cast<long>(d.globalFineSize()[a]);
    }
    periodic_ = d.periodic();
    probeSlot_.clear();
    anchorSlot_.clear();
    ghostCoords_.clear();
    ghostLevels_.clear();
    misses_.clear();
    frozen_ = false;
  }

  Index numLocal() const { return nLocal_; }
  Index numGhosts() const { return static_cast<Index>(ghostCoords_.size()); }
  Index extendedSize() const { return nLocal_ + numGhosts(); }
  MPI_Comm comm() const { return d_->comm(); }

  /// Wrap an unbounded global fine probe into the domain (in place); false = the probe exits a
  /// non-periodic axis (no neighbour there).
  bool wrap(std::array<long, Dim>& g) const {
    for (int a = 0; a < Dim; ++a) {
      const long gf = globalFine_[a];
      if (g[a] < 0 || g[a] >= gf) {
        if (!periodic_[a])
          return false;
        g[a] = ((g[a] % gf) + gf) % gf;
      }
    }
    return true;
  }

  /// Resolve a wrapped global fine coordinate to an extended slot: [0, nLocal) = the covering
  /// LOCAL leaf; [nLocal, nLocal+nGhost) = ghost slot; kPending = unknown remote (queued as a
  /// miss — run the collective resolveMisses() and repeat the build pass).
  Index resolve(const CoordArr& gc) {
    bool inBlock = true;
    for (int a = 0; a < Dim; ++a) {
      const long v = static_cast<long>(gc[a]) - fineOrigin_[a];
      if (v < 0 || v >= fineSize_[a]) {
        inBlock = false;
        break;
      }
    }
    if (inBlock) {
      std::array<Coord, Dim> lc{};
      for (int a = 0; a < Dim; ++a)
        lc[a] = static_cast<Coord>(static_cast<long>(gc[a]) - fineOrigin_[a]);
      return d_->local().find(M::encode(lc).code());  // covering local leaf
    }
    auto it = probeSlot_.find(gc);
    if (it != probeSlot_.end())
      return it->second;
    if (frozen_)
      throw std::runtime_error("amr::LeafHalo::resolve: unknown coord after finalize()");
    misses_.emplace(gc, kPending);
    return kPending;
  }

  /// Wrap + resolve from unbounded long coords; kNone on a non-periodic exit.
  Index resolveGlobal(std::array<long, Dim> g) {
    if (!wrap(g))
      return kNone;
    CoordArr gc{};
    for (int a = 0; a < Dim; ++a)
      gc[a] = static_cast<Coord>(g[a]);
    return resolve(gc);
  }

  /// COLLECTIVE (all ranks together, misses or not): one owner coverLevels round resolving every
  /// queued miss to a canonical ghost slot (or a newly-discovered one). Returns the GLOBAL
  /// number of coords that were pending — drive the builders' fixpoint:
  ///   for (;;) { attempt build; if (halo.resolveMisses() == 0) break; }
  long resolveMisses() {
    if (frozen_)
      throw std::runtime_error("amr::LeafHalo::resolveMisses after finalize()");
    long localPending = static_cast<long>(misses_.size()), globalPending = 0;
    MPI_Allreduce(&localPending, &globalPending, 1, MPI_LONG, MPI_SUM, d_->comm());
    if (globalPending == 0)
      return 0;
    std::vector<CoordArr> coords;
    coords.reserve(misses_.size());
    for (const auto& kv : misses_)
      coords.push_back(kv.first);
    // Collective owner request/reply — every rank participates (possibly with zero requests).
    std::vector<int> lv = d_->coverLevels(coords);
    for (std::size_t k = 0; k < coords.size(); ++k) {
      const int L = lv[k];
      if (L < 0)
        throw std::runtime_error(
            "amr::LeafHalo: a wrapped probe has no covering leaf on its owner (octree hole?)");
      CoordArr anchor{};
      for (int a = 0; a < Dim; ++a)
        anchor[a] = static_cast<Coord>((coords[k][a] >> L) << L);  // covering leaf lo corner
      Index g;
      auto it = anchorSlot_.find(anchor);
      if (it != anchorSlot_.end()) {
        g = it->second;
      } else {
        g = static_cast<Index>(ghostCoords_.size());
        anchorSlot_.emplace(anchor, g);
        ghostCoords_.push_back(anchor);
        ghostLevels_.push_back(L);
      }
      probeSlot_[coords[k]] = nLocal_ + g;
      probeSlot_[anchor] = nLocal_ + g;  // a later probe may hit the anchor directly
    }
    misses_.clear();
    return globalPending;
  }

  /// Covering-leaf level of an extended slot (local octree level convention: 0 = finest cell).
  int level(Index slot) const {
    if (slot < nLocal_)
      return static_cast<int>(d_->local().level(slot));
    return ghostLevels_[static_cast<std::size_t>(slot - nLocal_)];
  }
  /// Global fine anchor (lo corner) of ghost `g` in [0, nGhosts) — for SDF sampling / multi-hop
  /// probe construction by the builders.
  const CoordArr& ghostCoord(Index g) const {
    return ghostCoords_[static_cast<std::size_t>(g)];
  }

  /// Freeze the ghost set and establish the owner↔ghost value topology (one NBX round; the
  /// owner-side locateGlobal happens here, once — never per exchange).
  void finalize() {
    typename DO::FaceGatherPlan plan;
    plan.nFaces = extendedSize();
    plan.remoteCoords = ghostCoords_;
    plan.remoteSlot.resize(ghostCoords_.size());
    for (std::size_t g = 0; g < ghostCoords_.size(); ++g)
      plan.remoteSlot[g] = nLocal_ + static_cast<Index>(g);
    topo_ = d_->buildGatherHaloTopology(plan);
    // An out-of-block coord is never self-owned (a rank's ORB region IS its block), so nothing
    // may fold into the local-fill list — that would silently alias a ghost onto a local leaf.
    if (!topo_.localSlot.empty())
      throw std::runtime_error("amr::LeafHalo::finalize: ghost anchor resolved self-owned");
    for (Index l : topo_.sendLeaf)
      if (l < 0)
        throw std::runtime_error(
            "amr::LeafHalo::finalize: an owner cannot locate a requested ghost leaf");
    frozen_ = true;
  }

  const typename DO::GatherHaloTopology& topology() const { return topo_; }

  /// Host exchange: refresh x[nLocal, nLocal+nGhost) from the owners (x sized extendedSize()).
  /// Values are unmodified copies of the owner's doubles (bit-exact by construction).
  void exchangeHost(std::vector<double>& x, int tag = 45) const {
    const auto& t = topo_;
    std::vector<double> sendBuf(t.sendLeaf.size()), recvBuf(t.recvSlot.size());
    for (std::size_t k = 0; k < t.sendLeaf.size(); ++k)
      sendBuf[k] = x[static_cast<std::size_t>(t.sendLeaf[k])];
    std::vector<MPI_Request> reqs;
    reqs.reserve(t.recvRanks.size() + t.sendRanks.size());
    std::size_t off = 0;
    for (std::size_t k = 0; k < t.recvRanks.size(); ++k) {
      reqs.emplace_back();
      MPI_Irecv(recvBuf.data() + off,
                t.recvCounts[k] * static_cast<int>(sizeof(double)), MPI_BYTE, t.recvRanks[k], tag,
                d_->comm(), &reqs.back());
      off += static_cast<std::size_t>(t.recvCounts[k]);
    }
    off = 0;
    for (std::size_t k = 0; k < t.sendRanks.size(); ++k) {
      reqs.emplace_back();
      MPI_Isend(sendBuf.data() + off,
                t.sendCounts[k] * static_cast<int>(sizeof(double)), MPI_BYTE, t.sendRanks[k], tag,
                d_->comm(), &reqs.back());
      off += static_cast<std::size_t>(t.sendCounts[k]);
    }
    if (!reqs.empty())
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    for (std::size_t k = 0; k < t.recvSlot.size(); ++k)
      x[static_cast<std::size_t>(t.recvSlot[k])] = recvBuf[k];
  }

 private:
  DO* d_ = nullptr;
  Index nLocal_ = 0;
  std::array<long, Dim> fineOrigin_{}, fineSize_{}, globalFine_{};
  std::array<bool, Dim> periodic_{};
  std::map<CoordArr, Index> probeSlot_;   // wrapped probe coord → extended slot
  std::map<CoordArr, Index> anchorSlot_;  // canonical covering-leaf anchor → ghost id
  std::map<CoordArr, Index> misses_;      // pending coords (value unused; map for dedup+order)
  std::vector<CoordArr> ghostCoords_;     // ghost id → anchor (global fine lo corner)
  std::vector<int> ghostLevels_;          // ghost id → covering-leaf level
  typename DO::GatherHaloTopology topo_;
  bool frozen_ = false;
};

// ---- device exchanger (Kokkos TUs only; include after a Kokkos-carrying header) ---------------
#ifdef KOKKOS_INLINE_FUNCTION

/// Device-resident value refresh over a finalized LeafHalo: pack the owner's local values as a
/// Kokkos kernel, move only the compact buffers across MPI (host-staged by default, GPU-aware via
/// PECLET_CORE_GPU_AWARE_MPI — the GridHalo/DistributedGatherHalo pattern), scatter into the ghost
/// tail of the extended device field. exchange3 batches three components per message round.
class LeafHaloExchange {
 public:
  template <int Dim, unsigned Bits>
  void init(const LeafHalo<Dim, Bits>& h) {
    const auto& t = h.topology();
    comm_ = h.comm();
    nLocal_ = h.numLocal();
    nSend_ = static_cast<Index>(t.sendLeaf.size());
    nRecv_ = static_cast<Index>(t.recvSlot.size());
    d_sendLeaf_ = toDevice(t.sendLeaf, "lh::sendLeaf");
    d_recvSlot_ = toDevice(t.recvSlot, "lh::recvSlot");
    sendRanks_ = t.sendRanks;
    sendCounts_ = t.sendCounts;
    recvRanks_ = t.recvRanks;
    recvCounts_ = t.recvCounts;
    sendOff_.assign(sendCounts_.size() + 1, 0);
    for (std::size_t k = 0; k < sendCounts_.size(); ++k)
      sendOff_[k + 1] = sendOff_[k] + sendCounts_[k];
    recvOff_.assign(recvCounts_.size() + 1, 0);
    for (std::size_t k = 0; k < recvCounts_.size(); ++k)
      recvOff_[k + 1] = recvOff_[k] + recvCounts_[k];
    // Buffers sized for the batched 3-component exchange (single-field uses the first third).
    d_sendBuf_ = View<double>(Kokkos::view_alloc("lh::sendBuf", Kokkos::WithoutInitializing),
                              static_cast<std::size_t>(nSend_) * 3);
    d_recvBuf_ = View<double>(Kokkos::view_alloc("lh::recvBuf", Kokkos::WithoutInitializing),
                              static_cast<std::size_t>(nRecv_) * 3);
    h_sendBuf_ = Kokkos::create_mirror_view(d_sendBuf_);
    h_recvBuf_ = Kokkos::create_mirror_view(d_recvBuf_);
  }

  Index numGhosts() const { return nRecv_; }

  /// Refresh x[nLocal, nLocal+nGhost) (x is the extended device field, size >= extendedSize()).
  void exchange(View<double> x, int tag = 45) const {
    if (nSend_) {
      IndexView sl = d_sendLeaf_;
      View<double> buf = d_sendBuf_;
      Kokkos::parallel_for(
          "lh::pack", Kokkos::RangePolicy<ExecSpace>(0, nSend_),
          KOKKOS_LAMBDA(const Index p) { buf(p) = x(sl(p)); });
    }
    transfer(1, tag);
    if (nRecv_) {
      IndexView rs = d_recvSlot_;
      View<double> buf = d_recvBuf_;
      Kokkos::parallel_for(
          "lh::scatter", Kokkos::RangePolicy<ExecSpace>(0, nRecv_),
          KOKKOS_LAMBDA(const Index k) { x(rs(k)) = buf(k); });
    }
    Kokkos::fence();
  }

  /// Batched 3-component refresh (one message round for u0,u1,u2 — stride-3 packing).
  void exchange3(View<double> x0, View<double> x1, View<double> x2, int tag = 46) const {
    if (nSend_) {
      IndexView sl = d_sendLeaf_;
      View<double> buf = d_sendBuf_;
      Kokkos::parallel_for(
          "lh::pack3", Kokkos::RangePolicy<ExecSpace>(0, nSend_), KOKKOS_LAMBDA(const Index p) {
            const Index l = sl(p);
            buf(p * 3 + 0) = x0(l);
            buf(p * 3 + 1) = x1(l);
            buf(p * 3 + 2) = x2(l);
          });
    }
    transfer(3, tag);
    if (nRecv_) {
      IndexView rs = d_recvSlot_;
      View<double> buf = d_recvBuf_;
      Kokkos::parallel_for(
          "lh::scatter3", Kokkos::RangePolicy<ExecSpace>(0, nRecv_), KOKKOS_LAMBDA(const Index k) {
            const Index s = rs(k);
            x0(s) = buf(k * 3 + 0);
            x1(s) = buf(k * 3 + 1);
            x2(s) = buf(k * 3 + 2);
          });
    }
    Kokkos::fence();
  }

 private:
  /// Move `width` doubles per topology entry across MPI (host-staged unless GPU-aware).
  void transfer(int width, int tag) const {
    const bool aware = peclet::core::halo::detail::gpuAwareMpi();
    if (nSend_ && !aware)
      Kokkos::deep_copy(h_sendBuf_, d_sendBuf_);
    Kokkos::fence();  // send buffer (host-staged or device) ready before MPI reads it
    double* sendBase = aware ? d_sendBuf_.data() : h_sendBuf_.data();
    double* recvBase = aware ? d_recvBuf_.data() : h_recvBuf_.data();
    std::vector<MPI_Request> reqs;
    reqs.reserve(recvRanks_.size() + sendRanks_.size());
    for (std::size_t k = 0; k < recvRanks_.size(); ++k) {
      reqs.emplace_back();
      MPI_Irecv(recvBase + static_cast<std::size_t>(recvOff_[k]) * width,
                recvCounts_[k] * width * static_cast<int>(sizeof(double)), MPI_BYTE, recvRanks_[k],
                tag, comm_, &reqs.back());
    }
    for (std::size_t k = 0; k < sendRanks_.size(); ++k) {
      reqs.emplace_back();
      MPI_Isend(sendBase + static_cast<std::size_t>(sendOff_[k]) * width,
                sendCounts_[k] * width * static_cast<int>(sizeof(double)), MPI_BYTE, sendRanks_[k],
                tag, comm_, &reqs.back());
    }
    if (!reqs.empty())
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    if (nRecv_ && !aware)
      Kokkos::deep_copy(d_recvBuf_, h_recvBuf_);
  }

  MPI_Comm comm_ = MPI_COMM_NULL;
  Index nLocal_ = 0, nSend_ = 0, nRecv_ = 0;
  IndexView d_sendLeaf_, d_recvSlot_;
  std::vector<int> sendRanks_, sendCounts_, sendOff_, recvRanks_, recvCounts_, recvOff_;
  View<double> d_sendBuf_, d_recvBuf_;
  HostView<double> h_sendBuf_, h_recvBuf_;
};

#endif  // KOKKOS_INLINE_FUNCTION

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_LEAF_HALO_HPP

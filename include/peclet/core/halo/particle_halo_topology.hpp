// core — persistent Lagrangian ghost halo with forward + reverse(accumulate) exchange.
//
// The generic communication machinery behind the standard parallel particle schemes (and the
// Voronoi conservative-flux scheme). Same topology/exchange split as GridHaloTopology: build() establishes a
// persistent owner<->ghost correspondence from particle proximity (rebuilt only when the neighbour
// list rebuilds), then forward()/reverse() are cheap field-agnostic exchanges over it:
//
//   forward(owned -> ghost)            : copy each owner's value into its ghost copies (refresh state)
//   reverse(ghost -> owned, +=)        : accumulate ghost contributions back onto the owner
//   forwardPositions(owned -> ghost)   : forward with the periodic image shift applied (positions only)
//
// The three distributed schemes are compositions of these (see packing-gpu/mpi/README.md):
//   A (frozen/replicate): forwardPositions+forward(state); compute pairs x2; integrate owned.
//   B (Newton-on):        forward(state); each pair x1; reverse(force,sum); integrate owned.
//   C (force-accumulate): each pair x1; reverse(force,sum); forward(totalForce); integrate owned+ghost.
//
// The solver supplies the interaction kernel and the integrator; the core supplies forward/reverse.
#ifndef PECLET_CORE_HALO_PARTICLE_HALO_TOPOLOGY_HPP
#define PECLET_CORE_HALO_PARTICLE_HALO_TOPOLOGY_HPP

#include "peclet/core/common/mpi.hpp"

#include <cstring>
#include <map>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/halo/nbx.hpp"
#include "peclet/core/halo/particle_migrator.hpp"

namespace peclet::core::halo {

template <int Dim>
class ParticleHaloTopology {
 public:
  /// Bind to a migrator (provides the decomposition, domain map, rank and comm).
  void init(const ParticleMigrator<Dim>& mig) { mig_ = &mig; }

  /// (Re)establish the owner<->ghost correspondence: every owned particle within `rcut` of another
  /// rank's block becomes a ghost there. Call after migration / on neighbour-list rebuild.
  ///
  /// `includePeriodicSelf` additionally emits LOCAL periodic self-ghosts: copies of an owned particle
  /// at its own periodic image(s) that fall within `rcut` of THIS rank's own block. They are needed
  /// when a rank borders itself across a periodic face — i.e. an undecomposed periodic axis (a "×1"
  /// ORB axis, e.g. the z of a 2×2×1 layout) or np=1 — where the periodic neighbour is owned by the
  /// same rank and so is never produced by the cross-rank exchange. Off by default => byte-identical
  /// to the cross-rank-only behaviour (no self-ghosts, no MPI self-messages). Self-ghosts occupy the
  /// ghost slots AFTER the received ones ([numReceived, numGhost)) and are filled locally (no MPI).
  void build(const std::vector<Vec<Dim>>& pos, double rcut, bool includePeriodicSelf = false) {
    numOwned_ = pos.size();
    int nranks = 0;
    MPI_Comm_size(mig_->comm(), &nranks);
    const int me = mig_->rank();

    // Sender side: which of my particles each other rank needs, and the periodic shift to apply.
    std::map<int, std::vector<Index>> sendMap;
    std::map<int, std::vector<Vec<Dim>>> shiftMap;
    Vec<Dim> img;
    for (std::size_t i = 0; i < pos.size(); ++i) {
      for (int r = 0; r < nranks; ++r) {
        if (r == me) continue;
        if (!mig_->withinRcutOfBlock(pos[i], r, rcut, img)) continue;
        Vec<Dim> shift;
        for (int d = 0; d < Dim; ++d) shift[d] = img[d] - pos[i][d];
        sendMap[r].push_back(static_cast<Index>(i));
        shiftMap[r].push_back(shift);
      }
    }
    sendRanks_.clear();
    sendIdx_.clear();
    sendRankPos_.clear();
    for (auto& [r, idx] : sendMap) {
      sendRankPos_[r] = sendRanks_.size();
      sendRanks_.push_back(r);
      sendIdx_.push_back(std::move(idx));
    }

    // Exchange the shift vectors (one NBX round) to size the ghost array + matched recv lists.
    recvRanks_.clear();
    recvCount_.clear();
    recvRankPos_.clear();
    shift_.clear();
    NbxEngine nbx(mig_->comm());
    std::size_t k = 0;
    auto packNext = [&](std::vector<char>& out) -> int {
      if (k >= sendRanks_.size()) return -1;
      int dst = sendRanks_[k];
      auto& sh = shiftMap[dst];
      ++k;
      out.resize(sh.size() * sizeof(Vec<Dim>));
      std::memcpy(out.data(), sh.data(), out.size());
      return dst;
    };
    auto onRecv = [&](int src, std::vector<char>& msg) {
      int cnt = static_cast<int>(msg.size() / sizeof(Vec<Dim>));
      recvRankPos_[src] = recvRanks_.size();
      recvRanks_.push_back(src);
      recvCount_.push_back(cnt);
      const Vec<Dim>* sh = reinterpret_cast<const Vec<Dim>*>(msg.data());
      for (int i = 0; i < cnt; ++i) shift_.push_back(sh[i]);
    };
    nbx.exchange(packNext, onRecv, /*tag=*/7501);

    recvOffset_.assign(recvRanks_.size() + 1, 0);
    for (std::size_t i = 0; i < recvCount_.size(); ++i)
      recvOffset_[i + 1] = recvOffset_[i] + recvCount_[i];
    numReceived_ = shift_.size();

    // Local periodic self-ghosts (no MPI): each owned particle's non-identity periodic image(s) that
    // land within rcut of this rank's own block. Appended after the received ghosts; their shift goes
    // in the shift_ tail so the per-ghost position forward applies the wrap uniformly.
    selfIdx_.clear();
    selfShift_.clear();
    if (includePeriodicSelf) {
      std::vector<Vec<Dim>> imgs;
      for (std::size_t i = 0; i < pos.size(); ++i) {
        imgs.clear();
        mig_->imagesWithinRcutOfBlock(pos[i], me, rcut, /*allowIdentity=*/false, imgs);
        for (const auto& sh : imgs) {
          selfIdx_.push_back(static_cast<Index>(i));
          selfShift_.push_back(sh);
        }
      }
    }
    for (const auto& sh : selfShift_) shift_.push_back(sh);
    numGhost_ = shift_.size();

    // Populate the initial ghost positions (= owner position + shift), received + self.
    ghostPos_.assign(numGhost_, Vec<Dim>{});
    forwardPositions(pos.data(), ghostPos_.data());
  }

  std::size_t numOwned() const { return numOwned_; }
  std::size_t numGhost() const { return numGhost_; }
  const std::vector<Vec<Dim>>& ghostPositions() const { return ghostPos_; }

  /// owned[N] -> ghost[G], with the periodic image shift added (use for positions).
  /// Direct point-to-point over the topology fixed by build() (no NBX consensus) -- this is the
  /// per-iteration hot path, so it must not pay the dynamic-discovery cost every call.
  void forwardPositions(const Vec<Dim>* owned, Vec<Dim>* ghost) {
    forwardDirect<Vec<Dim>>(owned, /*tag=*/7502, [&](std::size_t p, const Vec<Dim>* in) {
      Index off = recvOffset_[p];
      for (int i = 0; i < recvCount_[p]; ++i)
        for (int d = 0; d < Dim; ++d) ghost[off + i][d] = in[i][d] + shift_[off + i][d];
    });
    // Local periodic self-ghosts: owner position + wrap shift (no MPI).
    for (std::size_t j = 0; j < selfIdx_.size(); ++j)
      for (int d = 0; d < Dim; ++d)
        ghost[numReceived_ + j][d] = owned[selfIdx_[j]][d] + shift_[numReceived_ + j][d];
  }

  /// owned[N] -> ghost[G], verbatim (translation-invariant fields: velocity, id, radius, ...).
  template <typename T>
  void forward(const T* owned, T* ghost) {
    forwardDirect<T>(owned, /*tag=*/7503, [&](std::size_t p, const T* in) {
      Index off = recvOffset_[p];
      for (int i = 0; i < recvCount_[p]; ++i) ghost[off + i] = in[i];
    });
    // Local periodic self-ghosts: owner value, verbatim (no MPI).
    for (std::size_t j = 0; j < selfIdx_.size(); ++j) ghost[numReceived_ + j] = owned[selfIdx_[j]];
  }

  /// ghost[G] -> owned[N], accumulated (T must have operator+=). Use for forces/torques/fluxes:
  /// each ghost's partial contribution is summed onto its owner. Owner array is added to in place.
  template <typename T>
  void reverse(const T* ghost, T* owned) {
    // Mirror of forwardDirect: the receivers (owners) post recvs sized by sendIdx_, the ghost-holders
    // send their contiguous ghost slices back; accumulate on arrival.
    const int ns = static_cast<int>(sendRanks_.size());
    const int nr = static_cast<int>(recvRanks_.size());
    std::vector<std::vector<T>> rbuf(ns), sbuf(nr);
    std::vector<MPI_Request> rreq(ns, MPI_REQUEST_NULL), sreq(nr, MPI_REQUEST_NULL);
    for (int k = 0; k < ns; ++k) {
      rbuf[k].resize(sendIdx_[k].size());
      MPI_Irecv(rbuf[k].data(), static_cast<int>(rbuf[k].size() * sizeof(T)), MPI_BYTE,
                sendRanks_[k], 7504, mig_->comm(), &rreq[k]);
    }
    for (int p = 0; p < nr; ++p) {
      Index off = recvOffset_[p];
      sbuf[p].assign(ghost + off, ghost + off + recvCount_[p]);
      MPI_Isend(sbuf[p].data(), static_cast<int>(sbuf[p].size() * sizeof(T)), MPI_BYTE,
                recvRanks_[p], 7504, mig_->comm(), &sreq[p]);
    }
    MPI_Waitall(ns, rreq.data(), MPI_STATUSES_IGNORE);
    for (int k = 0; k < ns; ++k) {
      auto& idx = sendIdx_[k];
      for (std::size_t i = 0; i < idx.size(); ++i) owned[idx[i]] += rbuf[k][i];
    }
    // Local periodic self-ghosts accumulate straight onto their (same-rank) owner.
    for (std::size_t j = 0; j < selfIdx_.size(); ++j) owned[selfIdx_[j]] += ghost[numReceived_ + j];
    MPI_Waitall(nr, sreq.data(), MPI_STATUSES_IGNORE);
  }

 private:
  // Direct (persistent-topology) forward: owners send their gathered sendIdx_ slices, ghost-holders
  // receive into contiguous slots and apply `store(recvRankPos, buf)`. No NBX consensus -- the
  // neighbour set + message sizes are fixed by build(), so plain Irecv/Isend/Waitall is correct.
  template <typename T, typename Store>
  void forwardDirect(const T* owned, int tag, Store&& store) {
    const int ns = static_cast<int>(sendRanks_.size());
    const int nr = static_cast<int>(recvRanks_.size());
    std::vector<std::vector<T>> sbuf(ns), rbuf(nr);
    std::vector<MPI_Request> sreq(ns, MPI_REQUEST_NULL), rreq(nr, MPI_REQUEST_NULL);
    for (int p = 0; p < nr; ++p) {
      rbuf[p].resize(recvCount_[p]);
      MPI_Irecv(rbuf[p].data(), static_cast<int>(rbuf[p].size() * sizeof(T)), MPI_BYTE,
                recvRanks_[p], tag, mig_->comm(), &rreq[p]);
    }
    for (int k = 0; k < ns; ++k) {
      auto& idx = sendIdx_[k];
      sbuf[k].resize(idx.size());
      for (std::size_t i = 0; i < idx.size(); ++i) sbuf[k][i] = owned[idx[i]];
      MPI_Isend(sbuf[k].data(), static_cast<int>(sbuf[k].size() * sizeof(T)), MPI_BYTE,
                sendRanks_[k], tag, mig_->comm(), &sreq[k]);
    }
    MPI_Waitall(nr, rreq.data(), MPI_STATUSES_IGNORE);
    for (int p = 0; p < nr; ++p) store(static_cast<std::size_t>(p), rbuf[p].data());
    MPI_Waitall(ns, sreq.data(), MPI_STATUSES_IGNORE);
  }

 public:
  MPI_Comm comm() const { return mig_->comm(); }

  // The send/recv topology in a flat, device-friendly form, so a CUDA consumer can drive the
  // exchange with on-device gather kernels + device-pointer MPI (see packing-gpu's device-resident
  // pack). recvOffsets index the contiguous [0,numGhost) ghost array (in recvRanks order); shift is
  // per-ghost (for position forwards). Rebuild after each build().
  struct FlatTopo {
    std::vector<int> sendRanks;         // neighbour ranks I send owned copies to
    std::vector<Index> sendIdx;         // concatenated owned indices to send (all ranks)
    std::vector<int> sendCounts;        // per send rank
    std::vector<int> sendOffsets;       // prefix sum into sendIdx (size sendRanks+1)
    std::vector<int> recvRanks;         // neighbour ranks I receive ghosts from
    std::vector<int> recvCounts;        // per recv rank
    std::vector<Index> recvOffsets;     // per recv rank: start in the [0,numReceived) ghost array
    std::vector<Vec<Dim>> shift;        // per ghost: periodic image offset (add to forwarded position)
    std::vector<Index> selfIdx;         // owned index of each LOCAL periodic self-ghost (size numSelf)
    Index numReceived = 0;              // ghost slots [0,numReceived) are cross-rank (MPI), the rest
                                        // [numReceived, numGhost) are local self-ghosts gathered from selfIdx
  };
  FlatTopo flatten() const {
    FlatTopo t;
    t.sendRanks = sendRanks_;
    t.sendOffsets.push_back(0);
    for (const auto& idx : sendIdx_) {
      t.sendCounts.push_back(static_cast<int>(idx.size()));
      for (Index id : idx) t.sendIdx.push_back(id);
      t.sendOffsets.push_back(static_cast<int>(t.sendIdx.size()));
    }
    t.recvRanks = recvRanks_;
    t.recvCounts = recvCount_;
    t.recvOffsets.assign(recvOffset_.begin(),
                         recvOffset_.begin() + static_cast<std::ptrdiff_t>(recvRanks_.size()));
    t.shift = shift_;
    t.selfIdx = selfIdx_;
    t.numReceived = static_cast<Index>(numReceived_);
    return t;
  }

 private:
  const ParticleMigrator<Dim>* mig_ = nullptr;
  std::size_t numOwned_ = 0, numGhost_ = 0;

  // Owner side: particles I send as ghosts to each neighbour rank.
  std::vector<int> sendRanks_;
  std::vector<std::vector<Index>> sendIdx_;
  std::map<int, std::size_t> sendRankPos_;

  // Ghost side: ghosts I receive from each neighbour rank (contiguous slots, in recvRanks_ order).
  std::vector<int> recvRanks_;
  std::vector<int> recvCount_;
  std::vector<Index> recvOffset_;
  std::map<int, std::size_t> recvRankPos_;

  // Local periodic self-ghosts (undecomposed periodic axis / np=1): appended after the received ones.
  std::vector<Index> selfIdx_;       // owned index for each self-ghost (ghost slot numReceived_ + j)
  std::vector<Vec<Dim>> selfShift_;  // matching periodic wrap shift
  std::size_t numReceived_ = 0;      // count of cross-rank received ghosts (= start of the self tail)

  std::vector<Vec<Dim>> shift_;     // per ghost: periodic image offset (img - owner pos); received then self
  std::vector<Vec<Dim>> ghostPos_;  // per ghost: current image position
};

}  // namespace peclet::core::halo

#endif  // PECLET_CORE_HALO_PARTICLE_HALO_TOPOLOGY_HPP

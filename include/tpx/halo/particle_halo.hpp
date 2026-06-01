// transport-core — persistent Lagrangian ghost halo with forward + reverse(accumulate) exchange.
//
// The generic communication machinery behind the standard parallel particle schemes (and the
// Voronoi conservative-flux scheme). Same topology/exchange split as GridHalo: build() establishes a
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
#ifndef TPX_HALO_PARTICLE_HALO_HPP
#define TPX_HALO_PARTICLE_HALO_HPP

#include <mpi.h>

#include <cstring>
#include <map>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/halo/nbx.hpp"
#include "tpx/halo/particle_migrator.hpp"

namespace tpx::halo {

template <int Dim>
class ParticleHalo {
 public:
  /// Bind to a migrator (provides the decomposition, domain map, rank and comm).
  void init(const ParticleMigrator<Dim>& mig) { mig_ = &mig; }

  /// (Re)establish the owner<->ghost correspondence: every owned particle within `rcut` of another
  /// rank's block becomes a ghost there. Call after migration / on neighbour-list rebuild.
  void build(const std::vector<Vec<Dim>>& pos, double rcut) {
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
    numGhost_ = shift_.size();

    // Populate the initial ghost positions (= owner position + shift).
    ghostPos_.assign(numGhost_, Vec<Dim>{});
    forwardPositions(pos.data(), ghostPos_.data());
  }

  std::size_t numOwned() const { return numOwned_; }
  std::size_t numGhost() const { return numGhost_; }
  const std::vector<Vec<Dim>>& ghostPositions() const { return ghostPos_; }

  /// owned[N] -> ghost[G], with the periodic image shift added (use for positions).
  void forwardPositions(const Vec<Dim>* owned, Vec<Dim>* ghost) {
    NbxEngine nbx(mig_->comm());
    std::size_t k = 0;
    auto packNext = [&](std::vector<char>& out) -> int {
      if (k >= sendRanks_.size()) return -1;
      auto& idx = sendIdx_[k];
      int dst = sendRanks_[k];
      ++k;
      out.resize(idx.size() * sizeof(Vec<Dim>));
      Vec<Dim>* o = reinterpret_cast<Vec<Dim>*>(out.data());
      for (std::size_t i = 0; i < idx.size(); ++i) o[i] = owned[idx[i]];
      return dst;
    };
    auto onRecv = [&](int src, std::vector<char>& msg) {
      std::size_t p = recvRankPos_.at(src);
      Index off = recvOffset_[p];
      const Vec<Dim>* in = reinterpret_cast<const Vec<Dim>*>(msg.data());
      for (int i = 0; i < recvCount_[p]; ++i) {
        for (int d = 0; d < Dim; ++d) ghost[off + i][d] = in[i][d] + shift_[off + i][d];
      }
    };
    nbx.exchange(packNext, onRecv, /*tag=*/7502);
  }

  /// owned[N] -> ghost[G], verbatim (translation-invariant fields: velocity, id, radius, ...).
  template <typename T>
  void forward(const T* owned, T* ghost) {
    NbxEngine nbx(mig_->comm());
    std::size_t k = 0;
    auto packNext = [&](std::vector<char>& out) -> int {
      if (k >= sendRanks_.size()) return -1;
      auto& idx = sendIdx_[k];
      int dst = sendRanks_[k];
      ++k;
      out.resize(idx.size() * sizeof(T));
      T* o = reinterpret_cast<T*>(out.data());
      for (std::size_t i = 0; i < idx.size(); ++i) o[i] = owned[idx[i]];
      return dst;
    };
    auto onRecv = [&](int src, std::vector<char>& msg) {
      std::size_t p = recvRankPos_.at(src);
      Index off = recvOffset_[p];
      const T* in = reinterpret_cast<const T*>(msg.data());
      for (int i = 0; i < recvCount_[p]; ++i) ghost[off + i] = in[i];
    };
    nbx.exchange(packNext, onRecv, /*tag=*/7503);
  }

  /// ghost[G] -> owned[N], accumulated (T must have operator+=). Use for forces/torques/fluxes:
  /// each ghost's partial contribution is summed onto its owner. Owner array is added to in place.
  template <typename T>
  void reverse(const T* ghost, T* owned) {
    NbxEngine nbx(mig_->comm());
    std::size_t k = 0;
    auto packNext = [&](std::vector<char>& out) -> int {
      if (k >= recvRanks_.size()) return -1;
      Index off = recvOffset_[k];
      int cnt = recvCount_[k];
      int dst = recvRanks_[k];
      ++k;
      out.resize(cnt * sizeof(T));
      T* o = reinterpret_cast<T*>(out.data());
      for (int i = 0; i < cnt; ++i) o[i] = ghost[off + i];
      return dst;
    };
    auto onRecv = [&](int src, std::vector<char>& msg) {
      auto& idx = sendIdx_[sendRankPos_.at(src)];
      const T* in = reinterpret_cast<const T*>(msg.data());
      for (std::size_t i = 0; i < idx.size(); ++i) owned[idx[i]] += in[i];
    };
    nbx.exchange(packNext, onRecv, /*tag=*/7504);
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

  std::vector<Vec<Dim>> shift_;     // per ghost: periodic image offset (img - owner pos)
  std::vector<Vec<Dim>> ghostPos_;  // per ghost: current image position
};

}  // namespace tpx::halo

#endif  // TPX_HALO_PARTICLE_HALO_HPP

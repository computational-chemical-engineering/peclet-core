// transport-core — Lagrangian particle migration over the same block decomposition.
//
// After particles move, each one may now belong to a different rank's block. migrate() reassigns
// every particle to the rank that owns the block containing its (periodically wrapped) position,
// shipping departing particles to their new owner via the NBX engine — the dynamic/sparse exchange
// the consensus protocol is built for. Positions and an opaque fixed-stride payload travel together,
// so the caller can carry velocity, orientation, id, etc. without this layer knowing the schema.
//
// This is the Lagrangian counterpart to tpx::halo::GridHalo (Eulerian ghost cells): same
// decomposition, same async engine, different payload — the field-agnostic design in action.
#ifndef TPX_HALO_PARTICLE_MIGRATOR_HPP
#define TPX_HALO_PARTICLE_MIGRATOR_HPP

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/nbx.hpp"

namespace tpx::halo {

/// Physical-space layout that maps a position to a global cell of the decomposition.
template <int Dim>
struct DomainMap {
  Vec<Dim> origin{};                  ///< physical coordinate of global cell (0,...)
  Vec<Dim> cellSize{};                ///< physical size of one global cell
  std::array<bool, Dim> periodic{};   ///< per-axis periodicity
};

template <int Dim>
class ParticleMigrator {
 public:
  void init(const decomp::BlockDecomposer<Dim>& dec, int rank, DomainMap<Dim> map,
            MPI_Comm comm = MPI_COMM_WORLD) {
    dec_ = &dec;
    rank_ = rank;
    map_ = map;
    comm_ = comm;
  }

  /// Rank that owns the block containing position x (after periodic wrap / boundary clamp).
  int ownerOf(const Vec<Dim>& x) const {
    IVec<Dim> g{};
    const IVec<Dim>& gsize = dec_->globalSize();
    for (int i = 0; i < Dim; ++i) {
      double rel = (x[i] - map_.origin[i]) / map_.cellSize[i];
      Index c = static_cast<Index>(std::floor(rel));
      if (map_.periodic[i]) {
        c = wrap(c, gsize[i]);
      } else {
        c = std::clamp<Index>(c, 0, gsize[i] - 1);
      }
      g[i] = c;
    }
    return dec_->ownerOf(g);
  }

  /// Wrap a position into the periodic box (no-op on non-periodic axes).
  Vec<Dim> wrapPosition(Vec<Dim> x) const {
    const IVec<Dim>& gsize = dec_->globalSize();
    for (int i = 0; i < Dim; ++i) {
      if (!map_.periodic[i]) continue;
      double L = map_.cellSize[i] * static_cast<double>(gsize[i]);
      double rel = x[i] - map_.origin[i];
      rel = std::fmod(std::fmod(rel, L) + L, L);
      x[i] = map_.origin[i] + rel;
    }
    return x;
  }

  /// Reassign every particle to its owning rank. `pos` and `payload` (stride bytes per particle) are
  /// rewritten in place to hold exactly this rank's particles after migration. Returns the new
  /// local count. Last-migrated stats are available via lastSent()/lastReceived().
  std::size_t migrate(std::vector<Vec<Dim>>& pos, std::vector<char>& payload, std::size_t stride) {
    const std::size_t n = pos.size();
    const std::size_t posBytes = Dim * sizeof(double);
    const std::size_t recBytes = posBytes + stride;

    std::map<int, std::vector<char>> outbox;
    std::size_t keep = 0;
    sent_ = 0;
    for (std::size_t i = 0; i < n; ++i) {
      Vec<Dim> wx = wrapPosition(pos[i]);
      int owner = ownerOf(wx);
      if (owner == rank_) {
        pos[keep] = wx;
        if (stride) std::memmove(&payload[keep * stride], &payload[i * stride], stride);
        ++keep;
      } else {
        auto& buf = outbox[owner];
        std::size_t off = buf.size();
        buf.resize(off + recBytes);
        std::memcpy(buf.data() + off, wx.data(), posBytes);
        if (stride) std::memcpy(buf.data() + off + posBytes, &payload[i * stride], stride);
        ++sent_;
      }
    }
    pos.resize(keep);
    payload.resize(keep * stride);

    // Ship departing particles to their owners and absorb arrivals (NBX).
    std::vector<std::pair<int, const std::vector<char>*>> queue;
    queue.reserve(outbox.size());
    for (auto& [r, buf] : outbox) queue.emplace_back(r, &buf);

    received_ = 0;
    NbxEngine nbx(comm_);
    std::size_t q = 0;
    auto packNext = [&](std::vector<char>& out) -> int {
      if (q >= queue.size()) return -1;
      int dest = queue[q].first;
      out = *queue[q].second;
      ++q;
      return dest;
    };
    auto onRecv = [&](int /*src*/, std::vector<char>& msg) {
      std::size_t cnt = msg.size() / recBytes;
      for (std::size_t i = 0; i < cnt; ++i) {
        const char* rec = msg.data() + i * recBytes;
        Vec<Dim> x{};
        std::memcpy(x.data(), rec, posBytes);
        pos.push_back(x);
        if (stride) {
          std::size_t off = payload.size();
          payload.resize(off + stride);
          std::memcpy(&payload[off], rec + posBytes, stride);
        }
        ++received_;
      }
    };
    nbx.exchange(packNext, onRecv, /*tag=*/7401);
    return pos.size();
  }

  std::size_t lastSent() const { return sent_; }
  std::size_t lastReceived() const { return received_; }

 private:
  const decomp::BlockDecomposer<Dim>* dec_ = nullptr;
  int rank_ = 0;
  DomainMap<Dim> map_{};
  MPI_Comm comm_ = MPI_COMM_WORLD;
  std::size_t sent_ = 0, received_ = 0;
};

}  // namespace tpx::halo

#endif  // TPX_HALO_PARTICLE_MIGRATOR_HPP

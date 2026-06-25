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

#include "tpx/common/mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <utility>
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

  /// Global decomposition cell containing position x (after periodic wrap / boundary clamp).
  /// This is the binning index for weighted load-balancing — `ownerOf(x) ==
  /// decomposer().ownerOf(cellOf(x))`.
  IVec<Dim> cellOf(const Vec<Dim>& x) const {
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
    return g;
  }

  /// Rank that owns the block containing position x (after periodic wrap / boundary clamp).
  int ownerOf(const Vec<Dim>& x) const { return dec_->ownerOf(cellOf(x)); }

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

  int rank() const { return rank_; }
  MPI_Comm comm() const { return comm_; }
  const decomp::BlockDecomposer<Dim>& decomposer() const { return *dec_; }

  /// True if `x` (via its closest periodic image) comes within `rcut` of block `r`'s physical AABB.
  /// On success `img` is that closest image — the position to ship so rank `r` has correct coords.
  bool withinRcutOfBlock(const Vec<Dim>& x, int r, double rcut, Vec<Dim>& img) const {
    const auto& o = dec_->origins()[r];
    const auto& s = dec_->sizes()[r];
    const IVec<Dim>& gsize = dec_->globalSize();
    double d2 = 0.0;
    for (int i = 0; i < Dim; ++i) {
      double lo = map_.origin[i] + o[i] * map_.cellSize[i];
      double hi = map_.origin[i] + (o[i] + s[i]) * map_.cellSize[i];
      double L = map_.cellSize[i] * static_cast<double>(gsize[i]);
      double cands[3];
      int nc = 0;
      cands[nc++] = x[i];
      if (map_.periodic[i]) {
        cands[nc++] = x[i] - L;
        cands[nc++] = x[i] + L;
      }
      double bestGap = std::numeric_limits<double>::infinity();
      double bestImg = x[i];
      for (int c = 0; c < nc; ++c) {
        double p = cands[c];
        double gap = (p < lo) ? (lo - p) : (p > hi) ? (p - hi) : 0.0;
        if (gap < bestGap) {
          bestGap = gap;
          bestImg = p;
        }
      }
      img[i] = bestImg;
      d2 += bestGap * bestGap;
    }
    return d2 < rcut * rcut;
  }

  /// Append the shift (= image − x) of EVERY periodic image of `x` that comes within `rcut` of block
  /// `r`'s AABB (enumerating the up-to-3^Dim images: 0 and ±L per periodic axis). Unlike
  /// `withinRcutOfBlock` (which keeps only the single nearest image), this yields all qualifying
  /// images — the owner-based view a ghost layer needs when one rank borders ITSELF across a periodic
  /// face (an undecomposed periodic axis, or np=1). `allowIdentity=false` drops the un-shifted image,
  /// so r==self yields only the periodic self-wrap copies (a particle is never its own ghost).
  void imagesWithinRcutOfBlock(const Vec<Dim>& x, int r, double rcut, bool allowIdentity,
                               std::vector<Vec<Dim>>& outShifts) const {
    const auto& o = dec_->origins()[r];
    const auto& s = dec_->sizes()[r];
    const IVec<Dim>& gsize = dec_->globalSize();
    double lo[Dim], hi[Dim], cand[Dim][3];
    int nc[Dim];
    for (int d = 0; d < Dim; ++d) {
      lo[d] = map_.origin[d] + o[d] * map_.cellSize[d];
      hi[d] = map_.origin[d] + (o[d] + s[d]) * map_.cellSize[d];
      const double L = map_.cellSize[d] * static_cast<double>(gsize[d]);
      cand[d][0] = 0.0;
      nc[d] = 1;
      if (map_.periodic[d]) { cand[d][1] = -L; cand[d][2] = L; nc[d] = 3; }
    }
    int total = 1;
    for (int d = 0; d < Dim; ++d) total *= nc[d];
    const double rc2 = rcut * rcut;
    for (int idx = 0; idx < total; ++idx) {
      Vec<Dim> shift{};
      int t = idx;
      bool identity = true;
      double d2 = 0.0;
      for (int d = 0; d < Dim; ++d) {
        const int k = t % nc[d];
        t /= nc[d];
        shift[d] = cand[d][k];
        if (k != 0) identity = false;
        const double p = x[d] + shift[d];
        const double gap = (p < lo[d]) ? (lo[d] - p) : (p > hi[d]) ? (p - hi[d]) : 0.0;
        d2 += gap * gap;
      }
      if (d2 < rc2 && (allowIdentity || !identity)) outShifts.push_back(shift);
    }
  }

  /// Gather ghost copies of particles within `rcut` of this rank's block boundary. For each particle
  /// this rank owns, a copy (the periodic image closest to the target block) is sent to every OTHER
  /// rank whose block comes within `rcut`. Received ghosts are written to ghostPos/ghostPayload
  /// (cleared first). Returns the number of ghosts received. This is the Lagrangian halo: after it,
  /// each rank's neighbour search runs locally over its owned + ghost particles.
  std::size_t gatherGhosts(const std::vector<Vec<Dim>>& pos, const std::vector<char>& payload,
                           std::size_t stride, double rcut, std::vector<Vec<Dim>>& ghostPos,
                           std::vector<char>& ghostPayload) {
    ghostPos.clear();
    ghostPayload.clear();
    const std::size_t posBytes = Dim * sizeof(double);
    const std::size_t recBytes = posBytes + stride;
    int nranks = 0;
    MPI_Comm_size(comm_, &nranks);

    std::map<int, std::vector<char>> outbox;
    Vec<Dim> img;
    for (std::size_t i = 0; i < pos.size(); ++i) {
      for (int r = 0; r < nranks; ++r) {
        if (r == rank_) continue;
        if (!withinRcutOfBlock(pos[i], r, rcut, img)) continue;
        auto& buf = outbox[r];
        std::size_t off = buf.size();
        buf.resize(off + recBytes);
        std::memcpy(buf.data() + off, img.data(), posBytes);
        if (stride) std::memcpy(buf.data() + off + posBytes, &payload[i * stride], stride);
      }
    }

    std::vector<std::pair<int, const std::vector<char>*>> queue;
    queue.reserve(outbox.size());
    for (auto& kv : outbox) queue.emplace_back(kv.first, &kv.second);
    std::size_t q = 0, recv = 0;
    NbxEngine nbx(comm_);
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
        ghostPos.push_back(x);
        if (stride) {
          std::size_t off = ghostPayload.size();
          ghostPayload.resize(off + stride);
          std::memcpy(&ghostPayload[off], rec + posBytes, stride);
        }
        ++recv;
      }
    };
    nbx.exchange(packNext, onRecv, /*tag=*/7402);
    return recv;
  }

 private:
  const decomp::BlockDecomposer<Dim>* dec_ = nullptr;
  int rank_ = 0;
  DomainMap<Dim> map_{};
  MPI_Comm comm_ = MPI_COMM_WORLD;
  std::size_t sent_ = 0, received_ = 0;
};

}  // namespace tpx::halo

#endif  // TPX_HALO_PARTICLE_MIGRATOR_HPP

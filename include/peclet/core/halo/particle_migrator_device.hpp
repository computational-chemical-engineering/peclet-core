// transport-core — device-resident Lagrangian particle migration (D1).
//
// ParticleMigrator::migrate (particle_migrator.hpp) is 100% host: a consumer holding its particle SoA
// on the device must download it, migrate on the host, and re-upload — every migration. This class
// keeps the SoA on the device: the per-particle periodic wrap + owner lookup (the ORB tree, flattened
// to Views) run as a Kokkos kernel (device binning); departing particles are gathered into compact
// per-rank buffers on the device (device pack) and only those compact buffers are host-staged for the
// NBX consensus exchange (which stays host — it is sparse control logic). Arrivals are unpacked back
// into the device SoA. The bulk payload never round-trips; only the compact migrating records cross.
//
// Correctness contract (matching test_particle_migration): count is conserved, every surviving particle
// is owned by this rank (ownerOf == rank), and the global id multiset is preserved. The device wrap /
// cellOf / ownerOf reproduce the host ParticleMigrator math exactly (bit-identical on OpenMP), so the
// per-particle destination is identical; the NBX exchange itself is order-nondeterministic by design, so
// only the resulting set is well-defined (as for the host migrator).
//
// Requires a Kokkos build + MPI.
#ifndef PECLET_CORE_HALO_PARTICLE_MIGRATOR_DEVICE_HPP
#define PECLET_CORE_HALO_PARTICLE_MIGRATOR_DEVICE_HPP

#include "peclet/core/common/mpi.hpp"

#include <cstring>
#include <map>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/nbx.hpp"
#include "peclet/core/halo/particle_migrator.hpp"  // DomainMap<Dim>

namespace peclet::core::halo {

/// Device counterpart of ParticleMigrator. Particles live in flat device Views: `pos` is
/// (capacity·Dim) particle-major (pos(i·Dim+d)), `payload` is (capacity·stride) opaque bytes. migrate()
/// rewrites [0, newCount) in place and returns the new local count.
template <int Dim>
class ParticleMigratorDevice {
 public:
  void init(const decomp::BlockDecomposer<Dim>& dec, int rank, DomainMap<Dim> map,
            MPI_Comm comm = MPI_COMM_WORLD) {
    rank_ = rank;
    comm_ = comm;
    std::vector<int> sd;
    std::vector<Index> sv;
    dec.flattenTree(sd, sv);
    splitDim_ = toDevice(sd, "pmd::splitDim");
    splitVal_ = toDevice(sv, "pmd::splitVal");
    const IVec<Dim>& gs = dec.globalSize();
    for (int d = 0; d < Dim; ++d) {
      origin_[d] = map.origin[d];
      cellSize_[d] = map.cellSize[d];
      gsize_[d] = gs[d];
      periodic_[d] = map.periodic[d] ? 1 : 0;
    }
  }

  std::size_t lastSent() const { return sent_; }
  std::size_t lastReceived() const { return received_; }

  /// Reassign every particle (i ∈ [0,n)) to its owning rank, in place. Returns the new local count
  /// (≤ pos.extent(0)/Dim). Caller must size the Views for the worst-case arrival count.
  Index migrate(View<double> pos, View<char> payload, Index n, std::size_t stride) {
    const std::size_t posBytes = Dim * sizeof(double);
    const std::size_t recBytes = posBytes + stride;

    // ---- 1. device: periodic-wrap each position in place + compute its destination rank ----
    View<int> sd = splitDim_;
    View<Index> sv = splitVal_;
    double org[Dim], csz[Dim];
    Index gsz[Dim];
    char per[Dim];
    for (int d = 0; d < Dim; ++d) {
      org[d] = origin_[d];
      csz[d] = cellSize_[d];
      gsz[d] = gsize_[d];
      per[d] = periodic_[d];
    }
    View<int> dest(Kokkos::view_alloc("pmd::dest", Kokkos::WithoutInitializing),
                   static_cast<std::size_t>(n));
    Kokkos::parallel_for(
        "pmd::wrap_bin", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const Index i) {
          double x[Dim];
          for (int d = 0; d < Dim; ++d) x[d] = pos(i * Dim + d);
          for (int d = 0; d < Dim; ++d)
            if (per[d]) {
              const double L = csz[d] * static_cast<double>(gsz[d]);
              double rel = x[d] - org[d];
              rel -= L * Kokkos::floor(rel / L);  // into [0,L)
              x[d] = org[d] + rel;
            }
          for (int d = 0; d < Dim; ++d) pos(i * Dim + d) = x[d];  // store wrapped position
          Index g[Dim];
          for (int d = 0; d < Dim; ++d) {
            const double rel = (x[d] - org[d]) / csz[d];
            Index c = static_cast<Index>(Kokkos::floor(rel));
            if (per[d])
              c = ((c % gsz[d]) + gsz[d]) % gsz[d];
            else
              c = c < 0 ? 0 : (c >= gsz[d] ? gsz[d] - 1 : c);
            g[d] = c;
          }
          Index node = 0;
          while (sd(node) != -1) node = (g[sd(node)] < sv(node)) ? 2 * node + 1 : 2 * node + 2;
          dest(i) = static_cast<int>(sv(node));
        });

    // ---- 2. host consensus: split into kept indices + per-rank departing indices ----
    auto hdest = Kokkos::create_mirror_view(dest);
    Kokkos::deep_copy(hdest, dest);
    std::vector<Index> keepIdx;
    keepIdx.reserve(static_cast<std::size_t>(n));
    std::map<int, std::vector<Index>> sendByRank;
    for (Index i = 0; i < n; ++i) {
      if (hdest(i) == rank_)
        keepIdx.push_back(i);
      else
        sendByRank[hdest(i)].push_back(i);
    }
    const Index keep = static_cast<Index>(keepIdx.size());
    std::vector<int> sendRanks, sendCounts;
    std::vector<Index> sendIdxFlat;
    for (auto& kv : sendByRank) {
      sendRanks.push_back(kv.first);
      sendCounts.push_back(static_cast<int>(kv.second.size()));
      for (Index s : kv.second) sendIdxFlat.push_back(s);
    }
    const Index nSend = static_cast<Index>(sendIdxFlat.size());
    sent_ = static_cast<std::size_t>(nSend);

    // ---- 3. device: gather departing particles into compact send buffers; host-stage. MUST happen
    // BEFORE the compaction below, which overwrites pos[0,keep) (the departing indices are original). ----
    std::vector<double> hSendPos(static_cast<std::size_t>(nSend) * Dim);
    std::vector<char> hSendPay(static_cast<std::size_t>(nSend) * stride);
    if (nSend) {
      IndexView si = toDevice(sendIdxFlat, "pmd::sendIdx");
      View<double> sp(Kokkos::view_alloc("pmd::sendPos", Kokkos::WithoutInitializing),
                      static_cast<std::size_t>(nSend) * Dim);
      View<char> spay(Kokkos::view_alloc("pmd::sendPay", Kokkos::WithoutInitializing),
                      static_cast<std::size_t>(nSend) * stride);
      Kokkos::parallel_for(
          "pmd::pack", Kokkos::RangePolicy<ExecSpace>(0, nSend), KOKKOS_LAMBDA(const Index j) {
            const Index s = si(j);
            for (int d = 0; d < Dim; ++d) sp(j * Dim + d) = pos(s * Dim + d);
            for (std::size_t b = 0; b < stride; ++b) spay(j * stride + b) = payload(s * stride + b);
          });
      Kokkos::fence();
      auto hsp = Kokkos::create_mirror_view(sp);
      Kokkos::deep_copy(hsp, sp);
      for (std::size_t k = 0; k < hSendPos.size(); ++k) hSendPos[k] = hsp(k);
      if (stride) {
        auto hspay = Kokkos::create_mirror_view(spay);
        Kokkos::deep_copy(hspay, spay);
        for (std::size_t k = 0; k < hSendPay.size(); ++k) hSendPay[k] = hspay(k);
      }
    }

    // ---- 4. device: compact kept particles to the front via a scratch gather (overwrites pos[0,keep)). ----
    ensureScratch(pos.extent(0), payload.extent(0));
    if (keep) {
      IndexView ki = toDevice(keepIdx, "pmd::keepIdx");
      View<double> tp = tmpPos_;
      View<char> tpay = tmpPay_;
      Kokkos::parallel_for(
          "pmd::compact", Kokkos::RangePolicy<ExecSpace>(0, keep), KOKKOS_LAMBDA(const Index k) {
            const Index s = ki(k);
            for (int d = 0; d < Dim; ++d) tp(k * Dim + d) = pos(s * Dim + d);
            for (std::size_t b = 0; b < stride; ++b) tpay(k * stride + b) = payload(s * stride + b);
          });
      Kokkos::deep_copy(
          Kokkos::subview(pos, std::pair<std::size_t, std::size_t>(0, static_cast<std::size_t>(keep) * Dim)),
          Kokkos::subview(tmpPos_, std::pair<std::size_t, std::size_t>(0, static_cast<std::size_t>(keep) * Dim)));
      if (stride)
        Kokkos::deep_copy(
            Kokkos::subview(payload, std::pair<std::size_t, std::size_t>(0, static_cast<std::size_t>(keep) * stride)),
            Kokkos::subview(tmpPay_, std::pair<std::size_t, std::size_t>(0, static_cast<std::size_t>(keep) * stride)));
    }

    // ---- 5. host: NBX exchange of the compact records (pos bytes + payload bytes per particle) ----
    std::vector<std::vector<char>> outbox(sendRanks.size());
    {
      Index base = 0;
      for (std::size_t r = 0; r < sendRanks.size(); ++r) {
        const Index cnt = sendCounts[r];
        std::vector<char>& buf = outbox[r];
        buf.resize(static_cast<std::size_t>(cnt) * recBytes);
        for (Index j = 0; j < cnt; ++j) {
          char* rec = buf.data() + static_cast<std::size_t>(j) * recBytes;
          std::memcpy(rec, &hSendPos[(static_cast<std::size_t>(base) + j) * Dim], posBytes);
          if (stride) std::memcpy(rec + posBytes, &hSendPay[(static_cast<std::size_t>(base) + j) * stride], stride);
        }
        base += cnt;
      }
    }
    std::vector<double> hRecvPos;
    std::vector<char> hRecvPay;
    received_ = 0;
    {
      NbxEngine nbx(comm_);
      std::size_t q = 0;
      auto packNext = [&](std::vector<char>& out) -> int {
        if (q >= outbox.size()) return -1;
        int d = sendRanks[q];
        out = outbox[q];
        ++q;
        return d;
      };
      auto onRecv = [&](int /*src*/, std::vector<char>& msg) {
        const std::size_t cnt = msg.size() / recBytes;
        for (std::size_t i = 0; i < cnt; ++i) {
          const char* rec = msg.data() + i * recBytes;
          std::size_t po = hRecvPos.size();
          hRecvPos.resize(po + Dim);
          std::memcpy(&hRecvPos[po], rec, posBytes);
          if (stride) {
            std::size_t yo = hRecvPay.size();
            hRecvPay.resize(yo + stride);
            std::memcpy(&hRecvPay[yo], rec + posBytes, stride);
          }
          ++received_;
        }
      };
      nbx.exchange(packNext, onRecv, /*tag=*/7411);
    }

    // ---- 6. device: append arrivals to the compacted SoA at [keep, keep+recv) ----
    const Index recv = static_cast<Index>(received_);
    if (recv) {
      View<double> rp = toDevice(hRecvPos, "pmd::recvPos");
      Kokkos::parallel_for(
          "pmd::unpack_pos", Kokkos::RangePolicy<ExecSpace>(0, recv), KOKKOS_LAMBDA(const Index j) {
            for (int d = 0; d < Dim; ++d) pos((keep + j) * Dim + d) = rp(j * Dim + d);
          });
      if (stride) {
        View<char> ry = toDevice(hRecvPay, "pmd::recvPay");
        Kokkos::parallel_for(
            "pmd::unpack_pay", Kokkos::RangePolicy<ExecSpace>(0, recv), KOKKOS_LAMBDA(const Index j) {
              for (std::size_t b = 0; b < stride; ++b) payload((keep + j) * stride + b) = ry(j * stride + b);
            });
      }
    }
    Kokkos::fence();
    return keep + recv;
  }

 private:
  void ensureScratch(std::size_t posLen, std::size_t payLen) {
    if (tmpPos_.extent(0) < posLen)
      tmpPos_ = View<double>(Kokkos::view_alloc("pmd::tmpPos", Kokkos::WithoutInitializing), posLen);
    if (tmpPay_.extent(0) < payLen)
      tmpPay_ = View<char>(Kokkos::view_alloc("pmd::tmpPay", Kokkos::WithoutInitializing), payLen);
  }

  int rank_ = 0;
  MPI_Comm comm_ = MPI_COMM_WORLD;
  View<int> splitDim_;
  View<Index> splitVal_;
  double origin_[Dim] = {}, cellSize_[Dim] = {};
  Index gsize_[Dim] = {};
  char periodic_[Dim] = {};
  View<double> tmpPos_;
  View<char> tmpPay_;
  std::size_t sent_ = 0, received_ = 0;
};

}  // namespace peclet::core::halo

#endif  // PECLET_CORE_HALO_PARTICLE_MIGRATOR_DEVICE_HPP

// transport-core — portable (Kokkos) device driver for the persistent Lagrangian ghost halo.
//
// Kokkos counterpart of the device-resident particle gather that packing-gpu hand-rolls
// (gatherFloat4 / device-pointer MPI): it drives ParticleHalo<Dim>'s forward/reverse exchanges with
// on-device gather/scatter kernels, so per-particle attribute arrays stay on the GPU and only the
// compact send/recv buffers are host-staged for MPI (or handed straight to a GPU-aware MPI via
// TPX_GPU_AWARE_MPI). Topology comes from ParticleHalo::flatten(); results match the CPU exchange.
//
//   forward<T>(owned -> ghost)        : copy each owner's value into its ghost copies (verbatim).
//   reverse<T>(ghost -> owned, +=)    : accumulate ghost contributions back onto owners (atomic).
//
// The periodic position-shift forward (forwardPositions) is payload-specific (e.g. packing's float4)
// and lives in the consumer; this primitive is the field-agnostic core.
#ifndef TPX_HALO_PARTICLE_HALO_KOKKOS_HPP
#define TPX_HALO_PARTICLE_HALO_KOKKOS_HPP

#include "tpx/common/mpi.hpp"

#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/common/view.hpp"
#include "tpx/halo/grid_halo_kokkos.hpp"  // detail::gpuAwareMpi()
#include "tpx/halo/particle_halo.hpp"

namespace tpx::halo {

template <int Dim>
class DeviceParticleHaloKokkos {
 public:
  DeviceParticleHaloKokkos() = default;
  DeviceParticleHaloKokkos(const DeviceParticleHaloKokkos&) = delete;
  DeviceParticleHaloKokkos& operator=(const DeviceParticleHaloKokkos&) = delete;

  /// Capture the (already-built) host halo's topology in device-friendly form.
  void init(const ParticleHalo<Dim>& halo) {
    auto t = halo.flatten();
    comm_ = halo.comm();
    sendRanks_ = t.sendRanks;
    sendCounts_ = t.sendCounts;
    sendOff_ = t.sendOffsets;  // prefix sum into sendIdx, size sendRanks+1
    recvRanks_ = t.recvRanks;
    recvCounts_ = t.recvCounts;
    recvOff_.assign(t.recvOffsets.begin(), t.recvOffsets.end());  // per recv rank start in [0,numReceived)
    nSend_ = static_cast<Index>(t.sendIdx.size());
    numGhost_ = static_cast<Index>(halo.numGhost());
    numReceived_ = t.numReceived;             // cross-rank ghosts [0,numReceived); self-ghosts after
    numSelf_ = numGhost_ - numReceived_;
    d_sendIdx_ = toDevice(t.sendIdx, "tpx::halo::p_sendIdx");
    d_selfIdx_ = toDevice(t.selfIdx, "tpx::halo::p_selfIdx");
  }

  /// owned[N] -> ghost[G], verbatim. Both live on the device.
  template <class T>
  void forward(const View<T>& owned, const View<T>& ghost, int tag = 7603) {
    const bool aware = detail::gpuAwareMpi();
    View<T> sendBuf(Kokkos::view_alloc("tpx::halo::p_sendBuf", Kokkos::WithoutInitializing),
                    static_cast<std::size_t>(nSend_));
    if (nSend_) {
      View<T> o = owned;
      IndexView idx = d_sendIdx_;
      View<T> buf = sendBuf;
      Kokkos::parallel_for(
          "tpx::halo::p_gather", Kokkos::RangePolicy<ExecSpace>(0, nSend_),
          KOKKOS_LAMBDA(const Index i) { buf(i) = o(idx(i)); });
    }

    // MPI fills only the cross-rank received slots [0,numReceived); the self tail is gathered locally.
    std::vector<T> hSend, hRecv;
    T* sendBase;
    T* recvBase;
    if (aware) {
      Kokkos::fence();
      sendBase = sendBuf.data();
      recvBase = ghost.data();
    } else {
      hSend.resize(static_cast<std::size_t>(nSend_));
      hRecv.resize(static_cast<std::size_t>(numReceived_));
      copyToHost(sendBuf, hSend);
      sendBase = hSend.data();
      recvBase = hRecv.data();
    }

    std::vector<MPI_Request> reqs;
    postRecv(recvBase, recvRanks_, recvOff_, recvCounts_, tag, reqs);
    postSend(sendBase, sendRanks_, sendOff_, sendCounts_, tag, reqs);
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    if (!aware && numReceived_) {
      auto sub = Kokkos::subview(
          ghost, std::pair<std::size_t, std::size_t>(0, static_cast<std::size_t>(numReceived_)));
      Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(
          hRecv.data(), hRecv.size());
      Kokkos::deep_copy(sub, hv);
    }
    // Local periodic self-ghosts: gather owner -> ghost tail on device (no MPI). Needed when a rank
    // borders itself across a periodic face (undecomposed axis / np=1).
    if (numSelf_) {
      View<T> o = owned;
      IndexView sidx = d_selfIdx_;
      View<T> g = ghost;
      const Index base = numReceived_;
      Kokkos::parallel_for(
          "tpx::halo::p_selfGather", Kokkos::RangePolicy<ExecSpace>(0, numSelf_),
          KOKKOS_LAMBDA(const Index j) { g(base + j) = o(sidx(j)); });
    }
    Kokkos::fence();
  }

  /// ghost[G] -> owned[N], accumulated (owned += contributions). T must support Kokkos::atomic_add.
  template <class T>
  void reverse(const View<T>& ghost, const View<T>& owned, int tag = 7604) {
    const bool aware = detail::gpuAwareMpi();

    std::vector<T> hGhost, hRecv;
    T* ghostBase;
    T* recvBase;
    View<T> recvBuf(Kokkos::view_alloc("tpx::halo::p_recvBuf", Kokkos::WithoutInitializing),
                    static_cast<std::size_t>(nSend_));  // owners receive sendIdx-many contributions
    if (aware) {
      Kokkos::fence();
      ghostBase = ghost.data();
      recvBase = recvBuf.data();
    } else {
      hGhost.resize(static_cast<std::size_t>(numGhost_));
      hRecv.resize(static_cast<std::size_t>(nSend_));
      copyToHost(ghost, hGhost);
      ghostBase = hGhost.data();
      recvBase = hRecv.data();
    }

    // Mirror of forward: ghost-holders send ghost slices, owners receive into sendIdx-shaped buffer.
    std::vector<MPI_Request> reqs;
    postRecv(recvBase, sendRanks_, sendOff_, sendCounts_, tag, reqs);
    postSend(ghostBase, recvRanks_, recvOff_, recvCounts_, tag, reqs);
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    if (!aware) copyToDevice(hRecv, recvBuf);
    if (nSend_) {
      View<T> o = owned;
      IndexView idx = d_sendIdx_;
      View<T> buf = recvBuf;
      // Duplicate owned indices (a particle is a ghost on several ranks) => accumulate atomically.
      Kokkos::parallel_for(
          "tpx::halo::p_scatter", Kokkos::RangePolicy<ExecSpace>(0, nSend_),
          KOKKOS_LAMBDA(const Index i) { Kokkos::atomic_add(&o(idx(i)), buf(i)); });
    }
    // Local periodic self-ghosts accumulate straight onto their (same-rank) owner.
    if (numSelf_) {
      View<T> o = owned;
      IndexView sidx = d_selfIdx_;
      View<T> g = ghost;
      const Index base = numReceived_;
      Kokkos::parallel_for(
          "tpx::halo::p_selfScatter", Kokkos::RangePolicy<ExecSpace>(0, numSelf_),
          KOKKOS_LAMBDA(const Index j) { Kokkos::atomic_add(&o(sidx(j)), g(base + j)); });
    }
    Kokkos::fence();
  }

  Index numGhost() const { return numGhost_; }

 private:
  template <class T>
  static void copyToHost(const View<T>& d, std::vector<T>& h) {
    if (h.empty()) return;
    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(h.data(),
                                                                                    h.size());
    Kokkos::deep_copy(hv, d);
  }
  template <class T>
  static void copyToDevice(const std::vector<T>& h, const View<T>& d) {
    if (h.empty()) return;
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(h.data(),
                                                                                          h.size());
    Kokkos::deep_copy(d, hv);
  }

  // Post one Irecv per neighbour rank into `base` at the given per-rank offsets (in elements).
  template <class T>
  void postRecv(T* base, const std::vector<int>& ranks, const std::vector<int>& off,
                const std::vector<int>& cnt, int tag, std::vector<MPI_Request>& reqs) {
    for (std::size_t k = 0; k < ranks.size(); ++k) {
      reqs.emplace_back();
      MPI_Irecv(base + off[k], cnt[k] * static_cast<int>(sizeof(T)), MPI_BYTE, ranks[k], tag, comm_,
                &reqs.back());
    }
  }
  template <class T>
  void postSend(T* base, const std::vector<int>& ranks, const std::vector<int>& off,
                const std::vector<int>& cnt, int tag, std::vector<MPI_Request>& reqs) {
    for (std::size_t k = 0; k < ranks.size(); ++k) {
      reqs.emplace_back();
      MPI_Isend(base + off[k], cnt[k] * static_cast<int>(sizeof(T)), MPI_BYTE, ranks[k], tag, comm_,
                &reqs.back());
    }
  }

  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<int> sendRanks_, sendCounts_, sendOff_;
  std::vector<int> recvRanks_, recvCounts_, recvOff_;
  Index nSend_ = 0, numGhost_ = 0, numReceived_ = 0, numSelf_ = 0;
  IndexView d_sendIdx_, d_selfIdx_;
};

}  // namespace tpx::halo

#endif  // TPX_HALO_PARTICLE_HALO_KOKKOS_HPP

// transport-core — portable (Kokkos) GPU-resident ghost-layer exchange.
//
// Portable (Kokkos) GPU-resident grid halo: the field lives on the device as a
// peclet::core::View<T>; pack (gather send cells), unpack (scatter into ghost cells) and the periodic
// self-copy run as Kokkos::parallel_for on the default execution space (CUDA / HIP / OpenMP), so the
// full field never crosses the bus — only the compact halo buffers are staged to the host for MPI.
// A GPU-aware-MPI path (hand device pointers straight to MPI) is opt-in via the env var
// PECLET_CORE_GPU_AWARE_MPI (the legacy PECLET_CORE_CUDA_AWARE_MPI is still honoured); host-staging is the portable
// default. Topology comes from a host-built GridHaloTopology<Dim>::flatten(), exactly like the CUDA version,
// and the result is bit-for-bit identical to the CPU exchange.
#ifndef PECLET_CORE_HALO_GRID_HALO_HPP
#define PECLET_CORE_HALO_GRID_HALO_HPP

#include "peclet/core/common/mpi.hpp"

#include <cstdlib>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"

namespace peclet::core::halo {

namespace detail {
/// Whether to hand DEVICE pointers straight to MPI (GPU-aware MPI) instead of host-staging. Gated on
/// an env var (read once) rather than MPIX_Query_cuda_support(), which under-reports on some stacks.
inline bool gpuAwareMpi() {
  static const bool v = [] {
    const char* e = std::getenv("PECLET_CORE_GPU_AWARE_MPI");
    if (!e) e = std::getenv("PECLET_CORE_CUDA_AWARE_MPI");
    return e && std::atoi(e) != 0;
  }();
  return v;
}
}  // namespace detail

/// GPU ghost-layer exchange for a contiguous device field `peclet::core::View<T>` (one element per
/// extended-block cell). Build once from a host GridHaloTopology via init(); exchange() runs every step.
template <class T>
class GridHalo {
 public:
  GridHalo() = default;
  GridHalo(const GridHalo&) = delete;
  GridHalo& operator=(const GridHalo&) = delete;

  template <int Dim>
  void init(const GridHaloTopology<Dim>& halo) {
    auto t = halo.flatten();
    comm_ = t.comm;
    sendRanks_ = t.sendRanks;
    recvRanks_ = t.recvRanks;
    sendCounts_ = t.sendCounts;
    recvCounts_ = t.recvCounts;

    sendOff_.assign(sendCounts_.size() + 1, 0);
    for (std::size_t k = 0; k < sendCounts_.size(); ++k)
      sendOff_[k + 1] = sendOff_[k] + sendCounts_[k];
    recvOff_.assign(recvCounts_.size() + 1, 0);
    for (std::size_t k = 0; k < recvCounts_.size(); ++k)
      recvOff_[k + 1] = recvOff_[k] + recvCounts_[k];

    nSend_ = static_cast<Index>(t.sendIdx.size());
    nRecv_ = static_cast<Index>(t.recvIdx.size());
    nSelf_ = static_cast<Index>(t.selfSrc.size());

    d_sendIdx_ = toDevice(t.sendIdx, "peclet::core::halo::sendIdx");
    d_recvIdx_ = toDevice(t.recvIdx, "peclet::core::halo::recvIdx");
    d_selfSrc_ = toDevice(t.selfSrc, "peclet::core::halo::selfSrc");
    d_selfDst_ = toDevice(t.selfDst, "peclet::core::halo::selfDst");
    d_sendBuf_ = View<T>(Kokkos::view_alloc("peclet::core::halo::sendBuf", Kokkos::WithoutInitializing),
                         static_cast<std::size_t>(nSend_));
    d_recvBuf_ = View<T>(Kokkos::view_alloc("peclet::core::halo::recvBuf", Kokkos::WithoutInitializing),
                         static_cast<std::size_t>(nRecv_));
    h_sendBuf_ = Kokkos::create_mirror_view(d_sendBuf_);
    h_recvBuf_ = Kokkos::create_mirror_view(d_recvBuf_);
  }

  /// Exchange ghost layers of the device field `field`. Blocking.
  void exchange(const View<T>& field, int tag = 0) {
    const bool aware = detail::gpuAwareMpi();

    // Periodic copy within our own block (read inner cell, write ghost cell).
    if (nSelf_) {
      View<T> f = field;
      IndexView src = d_selfSrc_, dst = d_selfDst_;
      Kokkos::parallel_for(
          "peclet::core::halo::selfCopy", Kokkos::RangePolicy<ExecSpace>(0, nSelf_),
          KOKKOS_LAMBDA(const Index i) { f(dst(i)) = f(src(i)); });
    }
    // Gather the cells we send into the contiguous send buffer.
    if (nSend_) {
      View<T> f = field;
      IndexView idx = d_sendIdx_;
      View<T> buf = d_sendBuf_;
      Kokkos::parallel_for(
          "peclet::core::halo::pack", Kokkos::RangePolicy<ExecSpace>(0, nSend_),
          KOKKOS_LAMBDA(const Index i) { buf(i) = f(idx(i)); });
      if (!aware) Kokkos::deep_copy(h_sendBuf_, d_sendBuf_);
    }
    // The send buffer (host-staged, or device for the aware path) must be ready before MPI reads it.
    Kokkos::fence();

    T* sendBase = aware ? d_sendBuf_.data() : h_sendBuf_.data();
    T* recvBase = aware ? d_recvBuf_.data() : h_recvBuf_.data();
    std::vector<MPI_Request> reqs;
    reqs.reserve(recvRanks_.size() + sendRanks_.size());
    for (std::size_t k = 0; k < recvRanks_.size(); ++k) {
      reqs.emplace_back();
      MPI_Irecv(recvBase + recvOff_[k], recvCounts_[k] * static_cast<int>(sizeof(T)), MPI_BYTE,
                recvRanks_[k], tag, comm_, &reqs.back());
    }
    for (std::size_t k = 0; k < sendRanks_.size(); ++k) {
      reqs.emplace_back();
      MPI_Isend(sendBase + sendOff_[k], sendCounts_[k] * static_cast<int>(sizeof(T)), MPI_BYTE,
                sendRanks_[k], tag, comm_, &reqs.back());
    }
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    // Scatter received halo cells back into the field's ghost region.
    if (nRecv_) {
      if (!aware) Kokkos::deep_copy(d_recvBuf_, h_recvBuf_);
      View<T> f = field;
      IndexView idx = d_recvIdx_;
      View<T> buf = d_recvBuf_;
      Kokkos::parallel_for(
          "peclet::core::halo::unpack", Kokkos::RangePolicy<ExecSpace>(0, nRecv_),
          KOKKOS_LAMBDA(const Index i) { f(idx(i)) = buf(i); });
    }
    Kokkos::fence();
  }

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<int> sendRanks_, recvRanks_, sendCounts_, recvCounts_;
  std::vector<int> sendOff_, recvOff_;
  Index nSend_ = 0, nRecv_ = 0, nSelf_ = 0;
  IndexView d_sendIdx_, d_recvIdx_, d_selfSrc_, d_selfDst_;
  View<T> d_sendBuf_, d_recvBuf_;
  HostView<T> h_sendBuf_, h_recvBuf_;
};

}  // namespace peclet::core::halo

#endif  // PECLET_CORE_HALO_GRID_HALO_HPP

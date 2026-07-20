// core — portable (Kokkos) GPU-resident ghost-layer exchange.
//
// Portable (Kokkos) GPU-resident grid halo: the field lives on the device as a
// peclet::core::View<T>; pack (gather send cells), unpack (scatter into ghost cells) and the
// periodic self-copy run as Kokkos::parallel_for on the default execution space (CUDA / HIP /
// OpenMP), so the full field never crosses the bus — only the compact halo buffers are staged to
// the host for MPI. The GPU-aware-MPI path (hand device pointers straight to MPI, dropping both
// staging copies) is the DEFAULT on device builds when a loopback probe confirms MPI accepts
// device pointers; env PECLET_CORE_GPU_AWARE_MPI=0 forces host staging (kill switch for stacks
// that segfault instead of erroring), =1 forces the device path without probing (legacy
// PECLET_CORE_CUDA_AWARE_MPI honoured). exchangeBegin/exchangeEnd expose the split so callers can
// overlap interior compute with the messages in flight; exchange() is the blocking composition.
// Topology comes from a host-built GridHaloTopology<Dim>::flatten(), and the result is bit-for-bit
// identical to the CPU exchange.
#ifndef PECLET_CORE_HALO_GRID_HALO_HPP
#define PECLET_CORE_HALO_GRID_HALO_HPP

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "peclet/core/common/mpi.hpp"
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>  // OpenMPI: MPIX_CUDA_AWARE_SUPPORT / MPIX_Query_cuda_support
#endif
#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"

namespace peclet::core::halo {

namespace detail {
/// Does the MPI library itself claim CUDA awareness (OpenMPI's MPIX_Query_cuda_support)? Some
/// stacks under-report (measured: user-space OpenMPI 5.0.7 + UCX 1.20.1 with working cuda_ipc
/// returns 0 — see core/docs/cuda-aware-mpi.md), so this is only used to AUTO-enable, never to
/// veto an explicit env request.
inline bool mpiReportsCudaAware() {
#if defined(MPIX_CUDA_AWARE_SUPPORT)
#if MPIX_CUDA_AWARE_SUPPORT
  return MPIX_Query_cuda_support() == 1;
#else
  return false;
#endif
#else
  return false;  // no query available: no signal
#endif
}

/// Probe whether MPI actually accepts DEVICE pointers (GPU-aware MPI): checksummed loopback
/// Sendrecv on MPI_COMM_SELF with device buffers, errors returned (not fatal) and the payload
/// verified on device. Necessary-but-not-sufficient (same-node cuda_ipc/cuda_copy can work while
/// inter-node GPUDirect is broken). CAUTION: a stack with NO cuda transport SEGFAULTS inside UCX
/// rather than returning an error (measured on the system OpenMPI here), so this probe is only
/// run once the MPI has claimed CUDA awareness — it is the validation gate, not the discovery
/// mechanism.
inline bool probeGpuAwareMpi() {
#ifdef KOKKOS_ENABLE_SERIAL
  if (std::is_same_v<ExecSpace, Kokkos::Serial>)
    return false;  // host backend: device pointers are host pointers; staging copies alias anyway
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  if (std::is_same_v<ExecSpace, Kokkos::OpenMP>)
    return false;
#endif
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited)
    return false;
  MPI_Comm self;
  if (MPI_Comm_dup(MPI_COMM_SELF, &self) != MPI_SUCCESS)
    return false;
  MPI_Comm_set_errhandler(self, MPI_ERRORS_RETURN);
  constexpr int N = 64;
  View<int> src(Kokkos::view_alloc("peclet::core::halo::probeSrc", Kokkos::WithoutInitializing),
                N);
  View<int> dst("peclet::core::halo::probeDst", N);
  Kokkos::parallel_for(
      "peclet::core::halo::probeFill", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { src(i) = 0x5EC0DE + i * 7919; });
  Kokkos::fence();
  const int rc = MPI_Sendrecv(src.data(), N, MPI_INT, 0, 77, dst.data(), N, MPI_INT, 0, 77, self,
                              MPI_STATUS_IGNORE);
  MPI_Comm_free(&self);
  if (rc != MPI_SUCCESS)
    return false;
  int bad = 1;
  Kokkos::parallel_reduce(
      "peclet::core::halo::probeCheck", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i, int& b) {
        if (dst(i) != 0x5EC0DE + i * 7919)
          b += 1;
      },
      bad);
  return bad == 0;
}

/// Whether to hand DEVICE pointers straight to MPI (GPU-aware MPI) instead of host-staging.
/// Resolution (decided once per process): env PECLET_CORE_GPU_AWARE_MPI (legacy
/// PECLET_CORE_CUDA_AWARE_MPI) set to 0 forces host staging, nonzero forces the device-pointer
/// path (trusted, no probe — MPIX under-reports on some working stacks); UNSET auto-enables the
/// device path when the MPI REPORTS CUDA awareness AND the loopback probe validates it, so a
/// properly built GPU-aware MPI is used by default with no configuration. Stacks that work but
/// under-report (see core/docs/cuda-aware-mpi.md) still need the env=1 opt-in — blind probing is
/// not survivable there because an unsupported transport segfaults instead of erroring.
inline bool gpuAwareMpi() {
  static const bool v = [] {
    const char* e = std::getenv("PECLET_CORE_GPU_AWARE_MPI");
    if (!e)
      e = std::getenv("PECLET_CORE_CUDA_AWARE_MPI");
    const bool aware = e ? (std::atoi(e) != 0) : (mpiReportsCudaAware() && probeGpuAwareMpi());
    if (std::getenv("PECLET_CORE_HALO_VERBOSE"))
      std::fprintf(stderr, "peclet.core halo: %s MPI buffers (%s)\n",
                   aware ? "DEVICE" : "host-staged", e ? "env-forced" : "auto");
    return aware;
  }();
  return v;
}
}  // namespace detail

/// GPU ghost-layer exchange for a contiguous device field `peclet::core::View<T>` (one element per
/// extended-block cell). Build once from a host GridHaloTopology via init(); exchange() runs every
/// step.
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
    d_sendBuf_ =
        View<T>(Kokkos::view_alloc("peclet::core::halo::sendBuf", Kokkos::WithoutInitializing),
                static_cast<std::size_t>(nSend_));
    d_recvBuf_ =
        View<T>(Kokkos::view_alloc("peclet::core::halo::recvBuf", Kokkos::WithoutInitializing),
                static_cast<std::size_t>(nRecv_));
    h_sendBuf_ = Kokkos::create_mirror_view(d_sendBuf_);
    h_recvBuf_ = Kokkos::create_mirror_view(d_recvBuf_);
  }

  /// Exchange ghost layers of the device field `field`. Blocking (== exchangeBegin+exchangeEnd).
  void exchange(const View<T>& field, int tag = 0) {
    exchangeBegin(field, tag);
    exchangeEnd(field);
  }

  /// Start an exchange: periodic self-copy, device pack, post Isend/Irecv, return with the
  /// messages in flight. The caller may run kernels that touch only INNER (non-ghost, non-send)
  /// cells before exchangeEnd — that overlap is the point of the split. One exchange may be in
  /// flight per GridHalo instance.
  void exchangeBegin(const View<T>& field, int tag = 0) {
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
      if (!aware)
        Kokkos::deep_copy(h_sendBuf_, d_sendBuf_);
    }
    // The send buffer (host-staged, or device for the aware path) must be ready before MPI reads
    // it.
    Kokkos::fence();

    T* sendBase = aware ? d_sendBuf_.data() : h_sendBuf_.data();
    T* recvBase = aware ? d_recvBuf_.data() : h_recvBuf_.data();
    reqs_.clear();
    reqs_.reserve(recvRanks_.size() + sendRanks_.size());
    for (std::size_t k = 0; k < recvRanks_.size(); ++k) {
      reqs_.emplace_back();
      MPI_Irecv(recvBase + recvOff_[k], recvCounts_[k] * static_cast<int>(sizeof(T)), MPI_BYTE,
                recvRanks_[k], tag, comm_, &reqs_.back());
    }
    for (std::size_t k = 0; k < sendRanks_.size(); ++k) {
      reqs_.emplace_back();
      MPI_Isend(sendBase + sendOff_[k], sendCounts_[k] * static_cast<int>(sizeof(T)), MPI_BYTE,
                sendRanks_[k], tag, comm_, &reqs_.back());
    }
    inFlightAware_ = aware;
  }

  /// Complete the exchange started by exchangeBegin: wait, then scatter the received halo cells
  /// into `field`'s ghost region (must be the same field).
  void exchangeEnd(const View<T>& field) {
    if (!reqs_.empty())
      MPI_Waitall(static_cast<int>(reqs_.size()), reqs_.data(), MPI_STATUSES_IGNORE);
    reqs_.clear();
    if (nRecv_) {
      if (!inFlightAware_)
        Kokkos::deep_copy(d_recvBuf_, h_recvBuf_);
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
  std::vector<MPI_Request> reqs_;  // in-flight requests between exchangeBegin and exchangeEnd
  bool inFlightAware_ = false;     // staging mode of the in-flight exchange
  Index nSend_ = 0, nRecv_ = 0, nSelf_ = 0;
  IndexView d_sendIdx_, d_recvIdx_, d_selfSrc_, d_selfDst_;
  View<T> d_sendBuf_, d_recvBuf_;
  HostView<T> h_sendBuf_, h_recvBuf_;
};

}  // namespace peclet::core::halo

#endif  // PECLET_CORE_HALO_GRID_HALO_HPP

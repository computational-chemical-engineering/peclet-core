// transport-core — GPU-resident ghost-layer exchange (host-staged).
//
// The field lives on the device. Pack (gather send cells), unpack (scatter into ghost cells), and the
// periodic self-copy all run as CUDA kernels, so the *full* field never crosses PCIe — only the
// compact halo buffers are staged through the host for MPI. (Direct device-pointer MPI would avoid
// even that, but requires a CUDA-aware MPI runtime with a working GPUDirect/UCX path; this staged
// version is the portable fallback and is selected automatically.)
//
// Topology comes from a host-built tpx::halo::GridHalo<Dim> via flatten(); this class only moves
// payload, mirroring the field-agnostic CPU design.
#ifndef TPX_HALO_GRID_HALO_CUDA_CUH
#define TPX_HALO_GRID_HALO_CUDA_CUH

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/halo/grid_halo.hpp"

namespace tpx::halo::detail {
// Whether to hand DEVICE pointers straight to MPI (CUDA-aware MPI) instead of host-staging. Gated on
// the env var TPX_CUDA_AWARE_MPI (read once) rather than MPIX_Query_cuda_support(), which wrongly
// reports 0 with OpenMPI+UCX on this box even though device-pointer transfers work (see
// docs/cuda-aware-mpi.md). Set TPX_CUDA_AWARE_MPI=1 when running against a CUDA-aware MPI.
inline bool cudaAwareMpi() {
  static const bool v = [] {
    const char* e = std::getenv("TPX_CUDA_AWARE_MPI");
    return e && std::atoi(e) != 0;
  }();
  return v;
}
}  // namespace tpx::halo::detail

namespace tpx::halo {

namespace detail {

template <typename T>
__global__ void packKernel(const T* __restrict__ field, const Index* __restrict__ idx,
                           T* __restrict__ buf, Index n) {
  Index i = blockIdx.x * static_cast<Index>(blockDim.x) + threadIdx.x;
  if (i < n) buf[i] = field[idx[i]];
}

template <typename T>
__global__ void unpackKernel(T* __restrict__ field, const Index* __restrict__ idx,
                             const T* __restrict__ buf, Index n) {
  Index i = blockIdx.x * static_cast<Index>(blockDim.x) + threadIdx.x;
  if (i < n) field[idx[i]] = buf[i];
}

template <typename T>
__global__ void selfCopyKernel(T* __restrict__ field, const Index* __restrict__ src,
                               const Index* __restrict__ dst, Index n) {
  Index i = blockIdx.x * static_cast<Index>(blockDim.x) + threadIdx.x;
  if (i < n) field[dst[i]] = field[src[i]];
}

inline Index gridFor(Index n, int block) { return (n + block - 1) / block; }

}  // namespace detail

/// GPU ghost-layer exchange for a contiguous device field of `T` (one per extended-block cell).
template <typename T>
class DeviceGridExchange {
 public:
  DeviceGridExchange() = default;
  DeviceGridExchange(const DeviceGridExchange&) = delete;
  DeviceGridExchange& operator=(const DeviceGridExchange&) = delete;

  template <int Dim>
  void init(const GridHalo<Dim>& halo) {
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

    uploadIdx(&d_sendIdx_, t.sendIdx);
    uploadIdx(&d_recvIdx_, t.recvIdx);
    uploadIdx(&d_selfSrc_, t.selfSrc);
    uploadIdx(&d_selfDst_, t.selfDst);
    if (nSend_) cudaMalloc(&d_sendBuf_, nSend_ * sizeof(T));
    if (nRecv_) cudaMalloc(&d_recvBuf_, nRecv_ * sizeof(T));
    h_sendBuf_.resize(static_cast<std::size_t>(nSend_));
    h_recvBuf_.resize(static_cast<std::size_t>(nRecv_));
  }

  /// Exchange ghost layers of the device field `d_field` (host-staged). Blocking.
  void exchange(T* d_field, int tag = 0) {
    const int blk = 256;
    const bool aware = detail::cudaAwareMpi();  // pass device pointers straight to MPI?
    if (nSelf_) {
      detail::selfCopyKernel<T><<<detail::gridFor(nSelf_, blk), blk>>>(d_field, d_selfSrc_,
                                                                       d_selfDst_, nSelf_);
    }
    if (nSend_) {
      detail::packKernel<T><<<detail::gridFor(nSend_, blk), blk>>>(d_field, d_sendIdx_, d_sendBuf_,
                                                                   nSend_);
      if (!aware)
        cudaMemcpy(h_sendBuf_.data(), d_sendBuf_, nSend_ * sizeof(T), cudaMemcpyDeviceToHost);
    }
    // CUDA-aware path hands MPI the device buffers, so the pack kernel must finish first (the
    // host-staged path's cudaMemcpy already synchronises).
    if (aware) cudaDeviceSynchronize();

    T* sendBase = aware ? d_sendBuf_ : h_sendBuf_.data();
    T* recvBase = aware ? d_recvBuf_ : h_recvBuf_.data();
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

    if (nRecv_) {
      if (!aware)
        cudaMemcpy(d_recvBuf_, h_recvBuf_.data(), nRecv_ * sizeof(T), cudaMemcpyHostToDevice);
      detail::unpackKernel<T><<<detail::gridFor(nRecv_, blk), blk>>>(d_field, d_recvIdx_, d_recvBuf_,
                                                                     nRecv_);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "tpx::halo::DeviceGridExchange CUDA error: %s\n",
                   cudaGetErrorString(err));
    }
  }

  ~DeviceGridExchange() {
    cudaFree(d_sendIdx_);
    cudaFree(d_recvIdx_);
    cudaFree(d_selfSrc_);
    cudaFree(d_selfDst_);
    cudaFree(d_sendBuf_);
    cudaFree(d_recvBuf_);
  }

 private:
  static void uploadIdx(Index** dptr, const std::vector<Index>& v) {
    if (v.empty()) {
      *dptr = nullptr;
      return;
    }
    cudaMalloc(dptr, v.size() * sizeof(Index));
    cudaMemcpy(*dptr, v.data(), v.size() * sizeof(Index), cudaMemcpyHostToDevice);
  }

  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<int> sendRanks_, recvRanks_, sendCounts_, recvCounts_;
  std::vector<int> sendOff_, recvOff_;
  Index nSend_ = 0, nRecv_ = 0, nSelf_ = 0;
  Index* d_sendIdx_ = nullptr;
  Index* d_recvIdx_ = nullptr;
  Index* d_selfSrc_ = nullptr;
  Index* d_selfDst_ = nullptr;
  T* d_sendBuf_ = nullptr;
  T* d_recvBuf_ = nullptr;
  std::vector<T> h_sendBuf_, h_recvBuf_;
};

}  // namespace tpx::halo

#endif  // TPX_HALO_GRID_HALO_CUDA_CUH

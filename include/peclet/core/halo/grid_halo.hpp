// core — portable (Kokkos) GPU-resident ghost-layer exchange.
//
// Portable (Kokkos) GPU-resident grid halo: the field lives on the device as a
// peclet::core::View<T>; the gather (periodic self-copy + pack of the send cells, one launch) and
// the unpack (scatter into ghost cells) run as Kokkos::parallel_for on the default execution space
// (CUDA / HIP / OpenMP) — on a host memory space a copy of at most kHaloSerialCopy elements runs on
// the calling thread instead (no parallel region) — so the full field never crosses the bus — only
// the compact halo buffers are staged to the host for MPI. The GPU-aware-MPI path (hand device
// pointers straight to MPI, dropping both staging copies) is the DEFAULT on device builds when a
// loopback probe confirms MPI accepts device pointers; env PECLET_CORE_GPU_AWARE_MPI=0 forces host
// staging (kill switch for stacks that segfault instead of erroring), =1 forces the device path
// without probing (legacy PECLET_CORE_CUDA_AWARE_MPI honoured). exchangeBegin/exchangeEnd expose
// the split so callers can overlap interior compute with the messages in flight; exchange() is the
// blocking composition. Topology comes from a host-built GridHaloTopology<Dim>::flatten(), and the
// result is bit-for-bit identical to the CPU exchange.
//
// On a host memory space, partners on the same node exchange through MPI-3 shared memory (flow
// doc/vof_distributed_host_comm_design.md §5.5, WO-K5): init() allocates, per instance and node,
// a shared window that holds every rank's receive slots (two parities per on-node sender) and the
// per-sender flags seq[2] / consumed; the gather writes an on-node payload straight into the
// receiver's slot and publishes it with a release store of the exchange's sequence number, so the
// data lands during the sender's pack and the begin/interior/end overlap is real. Off-node
// partners keep the persistent MPI requests. The bytes moved are identical. PECLET_CORE_HALO_SHM=0
// is the debug kill switch (bitwise-neutral: every partner then goes through MPI, as before K5);
// PECLET_CORE_HALO_SHM_GROUP=k splits each node into groups of k ranks, so one node runs the mixed
// on-node / off-node path of a multi-node job (debug, bitwise-neutral). Both must be the same on
// every rank: init checks that with one allreduce and throws on a mismatch.
//
// GridHalo init and destruction are COLLECTIVE over the communicator on a host memory space with
// more than one rank (node split, MPI_Win_allocate_shared, MPI_Win_free): create, re-init and
// destroy every instance on every rank, in the same order. From Python, release solvers on every
// rank (del + an explicit teardown at the same point of the program); do not rely on the garbage
// collector, whose timing differs between ranks.
#ifndef PECLET_CORE_HALO_GRID_HALO_HPP
#define PECLET_CORE_HALO_GRID_HALO_HPP

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
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
#ifdef KOKKOS_ENABLE_THREADS
  if (std::is_same_v<ExecSpace, Kokkos::Threads>)
    return false;  // host backend, same as the two above — a wheel without a usable OpenMP gets
                   // this
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
  View<int> src(Kokkos::view_alloc("peclet::core::halo::probeSrc", Kokkos::WithoutInitializing), N);
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

/// PECLET_CORE_HALO_TIMEOUT=<seconds>: turn every exchangeEnd into a deadline wait that names the
/// pending messages and aborts (see GridHalo::waitAllDiagnosed). 0 / unset = plain MPI_Waitall.
inline double haloTimeout() {
  static const double v = [] {
    const char* e = std::getenv("PECLET_CORE_HALO_TIMEOUT");
    return e ? std::atof(e) : 0.0;
  }();
  return v;
}

/// PECLET_CORE_HALO_SHM=0: debug kill switch for the on-node shared-memory transport (WO-K5), read
/// once per process. Bitwise-neutral: the same bytes then travel through the persistent MPI
/// requests. It must be the same on every rank (mpirun forwards the launcher's environment).
inline bool haloShmEnabled() {
  static const bool v = [] {
    const char* e = std::getenv("PECLET_CORE_HALO_SHM");
    return !(e && std::atoi(e) == 0);
  }();
  return v;
}

/// PECLET_CORE_HALO_SHM_GROUP=k (k > 0): debug sub-split of the on-node shared-memory transport,
/// read once per process. Each node communicator is split into groups of k consecutive node ranks
/// (MPI_Comm_split(node, nodeRank / k)); only partners in the same group use shared memory, the
/// rest stay on the persistent MPI requests. It lets one node exercise the mixed on-node /
/// off-node path of a multi-node job. Transport only, bitwise-neutral; 0 / unset = no sub-split.
/// It must be the same on every rank.
inline int haloShmGroup() {
  static const int v = [] {
    const char* e = std::getenv("PECLET_CORE_HALO_SHM_GROUP");
    const int k = e ? std::atoi(e) : 0;
    return k > 0 ? k : 0;
  }();
  return v;
}
}  // namespace detail

/// Host memory space only: a gather (self-copy + pack) or an unpack of at most this many elements
/// runs as a plain loop on the calling thread, not as a parallel region. A few hundred halo
/// elements cost far less to copy than one fork/join of the OpenMP pool. Bitwise-neutral (a copy
/// is a copy); device builds always launch.
inline constexpr Index kHaloSerialCopy = 2048;

/// Host memory space, on-node shared-memory transport: a wait on a partner's flag polls this many
/// times (with a CPU pause) between yields of the time slice. Bitwise-neutral (it only decides
/// how a wait spends its time).
inline constexpr int kHaloShmSpin = 256;

/// GPU ghost-layer exchange for a contiguous device field `peclet::core::View<T>` (one element per
/// extended-block cell). Build once from a host GridHaloTopology via init(); exchange() runs every
/// step.
template <class T>
class GridHalo {
 public:
  GridHalo() = default;
  GridHalo(const GridHalo&) = delete;
  GridHalo& operator=(const GridHalo&) = delete;
  ~GridHalo() {
    freePersistent();
    freeShm();
  }

  /// Build the exchange from a host topology. COLLECTIVE over the topology's communicator when
  /// the on-node transport is possible (host memory space, more than one rank): every rank of it
  /// calls init for this instance, as for GridHaloTopology::buildTopology, and destroys or
  /// re-inits the instance at the same point of the program (the shared window is freed
  /// collectively).
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
    // The on-node shared-memory slots first (they decide which partners stay on MPI), then the
    // persistent requests for the rest, on the buffers the staging choice selects (cached here; a
    // later flip re-inits them in exchangeBegin).
    initShm();
    initPersistent(detail::gpuAwareMpi());
  }

  /// Name printed by the PECLET_CORE_HALO_TIMEOUT diagnostic (e.g. "mg L4 g1"), so a stuck
  /// exchange can be told apart from the others that share a communicator.
  void setLabel(std::string label) { label_ = std::move(label); }

  /// Number of this rank's partners (send + receive) served by the on-node shared-memory
  /// transport; 0 on a device memory space, at one rank, or with PECLET_CORE_HALO_SHM=0
  /// (PECLET_CORE_HALO_SHM_GROUP=k: only the partners in this rank's group of k node ranks).
  int numShmPartners() const { return static_cast<int>(shmSend_.size() + shmRecv_.size()); }

  /// Exchange ghost layers of the device field `field`. Blocking (== exchangeBegin+exchangeEnd).
  void exchange(const View<T>& field, int tag = 0) {
    exchangeBegin(field, tag);
    exchangeEnd(field);
  }

  /// Start an exchange: periodic self-copy and pack (ONE gather), start the receives and sends,
  /// return with the messages in flight. The caller may run kernels that touch only INNER
  /// (non-ghost, non-send) cells before exchangeEnd — that overlap is the point of the split. One
  /// exchange may be in flight per GridHalo instance. `tag == 0` (every caller today) starts the
  /// persistent requests built at init and, on a host memory space, the on-node shared-memory
  /// transport; any other tag posts Isend/Irecv on that tag to every partner.
  void exchangeBegin(const View<T>& field, int tag = 0) {
    const bool aware = detail::gpuAwareMpi();
    const bool shm = tag == 0 && shm_;
    // Exchange s of this instance (on-node transport): its payloads go into the parity-(s & 1)
    // slots, which exchange s - 2 used, so each on-node receiver must have consumed s - 2 first.
    std::uint64_t s = 0;
    if (shm) {
      s = ++seq_;
      for (const ShmPeer& p : shmSend_)
        shmWait([&] { return load(p.consumed) + 2 >= s; }, "CONSUMED by", sendRanks_[p.k], s);
    }
    // The gather: the periodic self-copy (read inner cell, write ghost cell) and the pack of the
    // send cells — into the contiguous send buffer, or for an on-node receiver straight into its
    // slot — as ONE launch over [0, nSelf_ + nSend_). Race-free for any ghost width: the self-copy
    // writes ghost cells only and the pack reads inner cells only. On a host memory space at most
    // kHaloSerialCopy elements are copied by the calling thread. (Inline, not a private helper:
    // nvcc rejects an extended lambda in a private member function.)
    if (nSelf_ + nSend_) {
      const Index nSelf = nSelf_, nAll = nSelf_ + nSend_;
      View<T> f = field;
      IndexView src = d_selfSrc_, dst = d_selfDst_, idx = d_sendIdx_;
      View<T> buf = d_sendBuf_;
      if constexpr (kHostMemory) {
        if (shm) {
          T* const* out = sendDst_[s & 1].data();  // per send element: its slot or buffer entry
          if (nAll <= kHaloSerialCopy) {
            Kokkos::fence();  // the caller's kernels on `field` are complete before the thread
                              // reads
            for (Index i = 0; i < nSelf; ++i)
              f(dst(i)) = f(src(i));
            for (Index i = nSelf; i < nAll; ++i)
              *out[i - nSelf] = f(idx(i - nSelf));
          } else {
            Kokkos::parallel_for(
                "peclet::core::halo::gather", Kokkos::RangePolicy<ExecSpace>(0, nAll),
                KOKKOS_LAMBDA(const Index i) {
                  if (i < nSelf)
                    f(dst(i)) = f(src(i));
                  else
                    *out[i - nSelf] = f(idx(i - nSelf));
                });
          }
        }
      }
      if (!shm) {
        if (kHostMemory && nAll <= kHaloSerialCopy) {
          Kokkos::fence();  // the caller's kernels on `field` are complete before the thread reads
          for (Index i = 0; i < nSelf; ++i)
            f(dst(i)) = f(src(i));
          for (Index i = nSelf; i < nAll; ++i)
            buf(i - nSelf) = f(idx(i - nSelf));
        } else {
          Kokkos::parallel_for(
              "peclet::core::halo::gather", Kokkos::RangePolicy<ExecSpace>(0, nAll),
              KOKKOS_LAMBDA(const Index i) {
                if (i < nSelf)
                  f(dst(i)) = f(src(i));
                else
                  buf(i - nSelf) = f(idx(i - nSelf));
              });
        }
      }
    }
    // The send buffer (host-staged, or device for the aware path) must be ready before MPI reads
    // it, and on the aware path the previous exchange's unpack must be done reading the receive
    // buffer MPI is about to fill. Host staging: the two-argument deep_copy fences. With no MPI
    // traffic (np 1: self-copy only) nothing outside the execution space reads either buffer —
    // the gather is stream-ordered before the caller's next kernel — so there is no fence. The
    // same fence completes the gather into the on-node slots before their release below.
    if (nSend_ && !aware)
      Kokkos::deep_copy(h_sendBuf_, d_sendBuf_);
    if (!recvRanks_.empty() || !sendRanks_.empty())
      Kokkos::fence("peclet::core::halo::beforeMpi");
    if (shm) {
      // Publish exchange s to every on-node receiver: its payload is in the parity-(s & 1) slot.
      std::atomic_thread_fence(std::memory_order_release);
      for (const ShmPeer& p : shmSend_)
        store(p.seq[s & 1], s);
    }

    inFlightAware_ = aware;
    inFlightShm_ = shm;
    if (tag == 0 && usePersistent_ && aware != preqAware_)
      initPersistent(aware);
    if (tag == 0 && usePersistent_) {
      if (!preqs_.empty())
        MPI_Startall(static_cast<int>(preqs_.size()), preqs_.data());
      inFlightPersistent_ = true;
      return;
    }
    inFlightPersistent_ = false;
    T* sendBase = aware ? d_sendBuf_.data() : h_sendBuf_.data();
    T* recvBase = aware ? d_recvBuf_.data() : h_recvBuf_.data();
    reqs_.clear();
    reqs_.reserve(recvRanks_.size() + sendRanks_.size());
    reqPeer_.clear();
    for (std::size_t k = 0; k < recvRanks_.size(); ++k) {
      if (shm && recvShm_[k])
        continue;  // (persistent requests unavailable) the on-node payloads come through the slots
      reqs_.emplace_back();
      reqPeer_.push_back(static_cast<int>(k));
      MPI_Irecv(recvBase + recvOff_[k], recvCounts_[k] * static_cast<int>(sizeof(T)), MPI_BYTE,
                recvRanks_[k], tag, comm_, &reqs_.back());
    }
    for (std::size_t k = 0; k < sendRanks_.size(); ++k) {
      if (shm && sendShm_[k])
        continue;
      reqs_.emplace_back();
      reqPeer_.push_back(static_cast<int>(recvRanks_.size() + k));
      MPI_Isend(sendBase + sendOff_[k], sendCounts_[k] * static_cast<int>(sizeof(T)), MPI_BYTE,
                sendRanks_[k], tag, comm_, &reqs_.back());
    }
  }

  /// Complete the exchange started by exchangeBegin: wait, then scatter the received halo cells
  /// into `field`'s ghost region (must be the same field).
  void exchangeEnd(const View<T>& field) {
    std::vector<MPI_Request>& r = inFlightPersistent_ ? preqs_ : reqs_;
    const std::vector<int>& peer = inFlightPersistent_ ? preqPeer_ : reqPeer_;
    if (!r.empty()) {
      if (detail::haloTimeout() > 0)
        waitAllDiagnosed(r, peer, detail::haloTimeout());
      else
        MPI_Waitall(static_cast<int>(r.size()), r.data(), MPI_STATUSES_IGNORE);
    }
    reqs_.clear();
    inFlightPersistent_ = false;
    // On-node transport: after the MPI completion (so no request of this exchange needs this rank
    // inside MPI while it polls), wait until every on-node sender has published exchange s.
    const bool shm = inFlightShm_;
    inFlightShm_ = false;
    const std::uint64_t s = seq_;
    if (shm)
      for (const ShmPeer& p : shmRecv_)
        shmWait([&] { return load(p.seq[s & 1]) == s; }, "DATA from", recvRanks_[p.k], s);
    if (nRecv_) {
      if (!inFlightAware_)
        Kokkos::deep_copy(d_recvBuf_, h_recvBuf_);
      View<T> f = field;
      IndexView idx = d_recvIdx_;
      View<T> buf = d_recvBuf_;
      bool done = false;
      if constexpr (kHostMemory) {
        if (shm) {
          const T* const* in = recvSrc_[s & 1].data();  // per receive element: slot or buffer
          if (nRecv_ <= kHaloSerialCopy) {
            for (Index i = 0; i < nRecv_; ++i)
              f(idx(i)) = *in[i];
          } else {
            Kokkos::parallel_for(
                "peclet::core::halo::unpack", Kokkos::RangePolicy<ExecSpace>(0, nRecv_),
                KOKKOS_LAMBDA(const Index i) { f(idx(i)) = *in[i]; });
            Kokkos::fence("peclet::core::halo::afterShmUnpack");  // slots read before released
          }
          done = true;
        } else if (nRecv_ <= kHaloSerialCopy) {
          for (Index i = 0; i < nRecv_; ++i)
            f(idx(i)) = buf(i);
          done = true;
        }
      }
      if (!done) {
        Kokkos::parallel_for(
            "peclet::core::halo::unpack", Kokkos::RangePolicy<ExecSpace>(0, nRecv_),
            KOKKOS_LAMBDA(const Index i) { f(idx(i)) = buf(i); });
        // The unpack is stream-ordered before the caller's next kernel and is not fenced. The
        // aware path keeps one fence: a caller may hand the exchanged device field straight to
        // MPI. (Host staging: the receive buffer was copied by a fencing deep_copy above.)
        if (inFlightAware_)
          Kokkos::fence("peclet::core::halo::afterUnpack");
      }
    }
    // Hand the parity-(s & 1) slots back to their senders (exchange s + 2 may refill them).
    if (shm)
      for (const ShmPeer& p : shmRecv_)
        store(p.consumed, s);
  }

 private:
  static constexpr bool kHostMemory = std::is_same_v<MemSpace, Kokkos::HostSpace>;
  static constexpr std::size_t kLine = 64;  // one flag per cache line; slots line-aligned
  static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free,
                "the on-node flags need lock-free 64-bit atomics (address-free across processes)");

  /// One on-node partner of the shared-memory transport. For a receiver (shmSend_) the flags and
  /// slots live in ITS segment of the window; for a sender (shmRecv_) in this rank's own segment.
  struct ShmPeer {
    int k = 0;                                   // index into sendRanks_ / recvRanks_
    std::uint64_t* seq[2] = {nullptr, nullptr};  // last exchange published in parity 0 / 1
    std::uint64_t* consumed = nullptr;           // last exchange the receiver has unpacked
  };

  static std::uint64_t load(std::uint64_t* p) {
    return std::atomic_ref<std::uint64_t>(*p).load(std::memory_order_acquire);
  }
  static void store(std::uint64_t* p, std::uint64_t v) {
    std::atomic_ref<std::uint64_t>(*p).store(v, std::memory_order_release);
  }
  static void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
  }

  /// Wait for a partner's flag: bounded spin (kHaloShmSpin polls), then yield the time slice and
  /// spin again. With PECLET_CORE_HALO_TIMEOUT set, a wait past the deadline names the partner
  /// and the exchange and aborts the job, as waitAllDiagnosed does for the MPI requests.
  template <class Done>
  void shmWait(Done done, const char* what, int partner, std::uint64_t s) {
    if (done())
      return;
    const double timeout = detail::haloTimeout();
    const double t0 = timeout > 0 ? MPI_Wtime() : 0.0;
    for (int n = 1;; ++n) {
      if (done())
        return;
      if (n < kHaloShmSpin) {
        cpuRelax();
        continue;
      }
      n = 0;
      std::this_thread::yield();
      if (timeout > 0 && MPI_Wtime() - t0 >= timeout) {
        std::string msg = timeoutHeader(timeout) + "   pending shared-memory " + what + " comm " +
                          std::to_string(partner) + " (world " +
                          std::to_string(worldRank(partner)) + "), exchange " + std::to_string(s) +
                          "\n";
        std::fputs(msg.c_str(), stderr);
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 77);
      }
    }
  }

  /// The on-node shared-memory transport (§5.5). Collective over comm_ (the node split) and over
  /// the node communicator (window, layout exchange); skipped entirely on a device memory space,
  /// at one rank and under PECLET_CORE_HALO_SHM=0, which leaves every partner on MPI.
  void initShm() {
    freeShm();
    sendShm_.assign(sendRanks_.size(), 0);
    recvShm_.assign(recvRanks_.size(), 0);
    if constexpr (kHostMemory) {
      int csize = 1;
      MPI_Comm_size(comm_, &csize);
      if (csize < 2)
        return;
      checkShmEnvCollective();
      if (!detail::haloShmEnabled())
        return;
      MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_);
      if (const int k = detail::haloShmGroup(); k > 0) {  // debug sub-split (mixed transport)
        int nodeRank = 0;
        MPI_Comm_rank(node_, &nodeRank);
        MPI_Comm group = MPI_COMM_NULL;
        MPI_Comm_split(node_, nodeRank / k, nodeRank, &group);
        MPI_Comm_free(&node_);
        node_ = group;
      }
      int nsize = 1;
      MPI_Comm_size(node_, &nsize);
      if (nsize < 2) {  // every member of this node communicator sees the same size
        MPI_Comm_free(&node_);
        return;
      }
      // Which partners share the node: their rank in node_ (MPI_UNDEFINED when off-node).
      std::vector<int> sendNode(sendRanks_.size(), MPI_UNDEFINED);
      std::vector<int> recvNode(recvRanks_.size(), MPI_UNDEFINED);
      {
        MPI_Group cg, ng;
        MPI_Comm_group(comm_, &cg);
        MPI_Comm_group(node_, &ng);
        if (!sendRanks_.empty())
          MPI_Group_translate_ranks(cg, static_cast<int>(sendRanks_.size()), sendRanks_.data(), ng,
                                    sendNode.data());
        if (!recvRanks_.empty())
          MPI_Group_translate_ranks(cg, static_cast<int>(recvRanks_.size()), recvRanks_.data(), ng,
                                    recvNode.data());
        MPI_Group_free(&cg);
        MPI_Group_free(&ng);
      }
      // This rank's segment: per on-node sender, three flag lines (seq[0], seq[1], consumed) and
      // two parity slots of its payload, every piece line-aligned.
      auto slotBytes = [](int count) {
        return (static_cast<std::size_t>(count) * sizeof(T) + kLine - 1) / kLine * kLine;
      };
      std::vector<std::size_t> recvBlock(recvRanks_.size(), 0);
      std::size_t bytes = 0;
      for (std::size_t k = 0; k < recvRanks_.size(); ++k)
        if (recvNode[k] != MPI_UNDEFINED) {
          recvBlock[k] = bytes;
          bytes += 3 * kLine + 2 * slotBytes(recvCounts_[k]);
        }
      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Info_set(info, "alloc_shared_noncontig", "true");
      void* base = nullptr;
      MPI_Win_allocate_shared(static_cast<MPI_Aint>(bytes + kLine), 1, info, node_, &base, &win_);
      MPI_Info_free(&info);
      // The unified memory model (load/store = the window's public copy), on every member.
      int* model = nullptr;
      int flag = 0;
      MPI_Win_get_attr(win_, MPI_WIN_MODEL, &model, &flag);
      int unified = (flag && *model == MPI_WIN_UNIFIED) ? 1 : 0, allUnified = 0;
      MPI_Allreduce(&unified, &allUnified, 1, MPI_INT, MPI_MIN, node_);
      if (!allUnified) {
        MPI_Win_free(&win_);
        MPI_Comm_free(&node_);
        return;
      }
      MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
      winLocked_ = true;
      // Line-align the segment in this rank's view (the mapping is page-aligned in every process,
      // so the same offsets are line-aligned in the senders' views too); zero it, so every flag
      // starts at 0 (no exchange published, none consumed) before any partner reads it.
      const std::size_t pad = (kLine - reinterpret_cast<std::uintptr_t>(base) % kLine) % kLine;
      char* own = static_cast<char*>(base) + pad;
      std::memset(own, 0, bytes);
      MPI_Win_sync(win_);
      MPI_Barrier(node_);
      // Tell every on-node sender where its block sits in this segment, and the count expected.
      std::vector<std::int64_t> out(2 * static_cast<std::size_t>(nsize), -1),
          in(2 * static_cast<std::size_t>(nsize), -1);
      for (std::size_t k = 0; k < recvRanks_.size(); ++k)
        if (recvNode[k] != MPI_UNDEFINED) {
          out[2 * static_cast<std::size_t>(recvNode[k])] =
              static_cast<std::int64_t>(pad + recvBlock[k]);
          out[2 * static_cast<std::size_t>(recvNode[k]) + 1] = recvCounts_[k];
        }
      MPI_Alltoall(out.data(), 2, MPI_INT64_T, in.data(), 2, MPI_INT64_T, node_);

      auto peerAt = [&](char* blk, int k) {
        ShmPeer p;
        p.k = k;
        p.seq[0] = reinterpret_cast<std::uint64_t*>(blk);
        p.seq[1] = reinterpret_cast<std::uint64_t*>(blk + kLine);
        p.consumed = reinterpret_cast<std::uint64_t*>(blk + 2 * kLine);
        return p;
      };
      for (int p = 0; p < 2; ++p) {
        sendDst_[p].resize(static_cast<std::size_t>(nSend_));
        recvSrc_[p].resize(static_cast<std::size_t>(nRecv_));
      }
      for (std::size_t k = 0; k < sendRanks_.size(); ++k) {
        T* priv = d_sendBuf_.data() + sendOff_[k];
        T* slot[2] = {priv, priv};
        if (sendNode[k] != MPI_UNDEFINED) {
          const std::size_t q = static_cast<std::size_t>(sendNode[k]);
          if (in[2 * q] < 0 || in[2 * q + 1] != sendCounts_[k]) {
            std::fprintf(stderr,
                         "peclet.core halo [%s]: on-node layout mismatch with node rank %d (offset "
                         "%lld, count %lld, expected %d)\n",
                         label_.c_str(), sendNode[k], static_cast<long long>(in[2 * q]),
                         static_cast<long long>(in[2 * q + 1]), sendCounts_[k]);
            MPI_Abort(MPI_COMM_WORLD, 78);
          }
          MPI_Aint sz = 0;
          int du = 1;
          void* rb = nullptr;
          MPI_Win_shared_query(win_, sendNode[k], &sz, &du, &rb);
          char* blk = static_cast<char*>(rb) + in[2 * q];
          shmSend_.push_back(peerAt(blk, static_cast<int>(k)));
          sendShm_[k] = 1;
          slot[0] = reinterpret_cast<T*>(blk + 3 * kLine);
          slot[1] = reinterpret_cast<T*>(blk + 3 * kLine + slotBytes(sendCounts_[k]));
        }
        for (int p = 0; p < 2; ++p)
          for (int j = 0; j < sendCounts_[k]; ++j)
            sendDst_[p][static_cast<std::size_t>(sendOff_[k] + j)] = slot[p] + j;
      }
      for (std::size_t k = 0; k < recvRanks_.size(); ++k) {
        const T* priv = d_recvBuf_.data() + recvOff_[k];
        const T* slot[2] = {priv, priv};
        if (recvNode[k] != MPI_UNDEFINED) {
          char* blk = own + recvBlock[k];
          shmRecv_.push_back(peerAt(blk, static_cast<int>(k)));
          recvShm_[k] = 1;
          slot[0] = reinterpret_cast<const T*>(blk + 3 * kLine);
          slot[1] = reinterpret_cast<const T*>(blk + 3 * kLine + slotBytes(recvCounts_[k]));
        }
        for (int p = 0; p < 2; ++p)
          for (int j = 0; j < recvCounts_[k]; ++j)
            recvSrc_[p][static_cast<std::size_t>(recvOff_[k] + j)] = slot[p] + j;
      }
      shm_ = true;
    }
  }

  /// PECLET_CORE_HALO_SHM and PECLET_CORE_HALO_SHM_GROUP decide which collectives initShm and the
  /// destructor run, so a rank that disagrees would hang its partners: one allreduce over comm_
  /// at init checks that every rank sees the same values, and throws on every rank otherwise.
  void checkShmEnvCollective() const {
    const int en = detail::haloShmEnabled() ? 1 : 0, grp = detail::haloShmGroup();
    int v[4] = {en, -en, grp, -grp};
    MPI_Allreduce(MPI_IN_PLACE, v, 4, MPI_INT, MPI_MAX, comm_);
    if (v[0] != -v[1] || v[2] != -v[3])
      throw std::runtime_error(
          "peclet.core halo [" + label_ +
          "]: PECLET_CORE_HALO_SHM / PECLET_CORE_HALO_SHM_GROUP differ across the ranks of the "
          "communicator (shared memory on " + std::to_string(-v[1]) + ".." + std::to_string(v[0]) +
          ", group " + std::to_string(-v[3]) + ".." + std::to_string(v[2]) +
          "); set them identically on every rank (mpirun -x)");
  }

  /// Free the window and the node communicator (collective over it; see init). Skipped after
  /// MPI_Finalize, like the persistent requests.
  void freeShm() {
    shm_ = false;
    inFlightShm_ = false;
    seq_ = 0;
    shmSend_.clear();
    shmRecv_.clear();
    for (int p = 0; p < 2; ++p) {
      sendDst_[p].clear();
      recvSrc_[p].clear();
    }
    if (win_ == MPI_WIN_NULL && node_ == MPI_COMM_NULL)
      return;
    int fin = 0;
    MPI_Finalized(&fin);
    if (!fin) {
      if (win_ != MPI_WIN_NULL) {
        if (winLocked_)
          MPI_Win_unlock_all(win_);
        MPI_Win_free(&win_);
      }
      if (node_ != MPI_COMM_NULL)
        MPI_Comm_free(&node_);
    }
    winLocked_ = false;
    win_ = MPI_WIN_NULL;
    node_ = MPI_COMM_NULL;
  }

  /// (Re)build the persistent requests (MPI_Recv_init / MPI_Send_init per neighbour off the
  /// on-node transport, tag 0) on the host-staging or the device buffers. A stack that rejects one
  /// (R-3: device pointers) leaves usePersistent_ false, and every exchange takes the Isend/Irecv
  /// path.
  void initPersistent(bool aware) {
    freePersistent();
    preqAware_ = aware;
    T* sendBase = aware ? d_sendBuf_.data() : h_sendBuf_.data();
    T* recvBase = aware ? d_recvBuf_.data() : h_recvBuf_.data();
    preqs_.clear();
    preqPeer_.clear();
    bool ok = true;
    for (std::size_t k = 0; k < recvRanks_.size(); ++k) {
      if (shm_ && recvShm_[k])
        continue;
      preqs_.push_back(MPI_REQUEST_NULL);
      preqPeer_.push_back(static_cast<int>(k));
      ok &= MPI_Recv_init(recvBase + recvOff_[k], recvCounts_[k] * static_cast<int>(sizeof(T)),
                          MPI_BYTE, recvRanks_[k], 0, comm_, &preqs_.back()) == MPI_SUCCESS;
    }
    const std::size_t nr = recvRanks_.size();
    for (std::size_t k = 0; k < sendRanks_.size(); ++k) {
      if (shm_ && sendShm_[k])
        continue;
      preqs_.push_back(MPI_REQUEST_NULL);
      preqPeer_.push_back(static_cast<int>(nr + k));
      ok &= MPI_Send_init(sendBase + sendOff_[k], sendCounts_[k] * static_cast<int>(sizeof(T)),
                          MPI_BYTE, sendRanks_[k], 0, comm_, &preqs_.back()) == MPI_SUCCESS;
    }
    if (!ok)
      freePersistent();
    usePersistent_ = ok;
  }

  void freePersistent() {
    usePersistent_ = false;
    if (preqs_.empty())
      return;
    int fin = 0;
    MPI_Finalized(&fin);
    if (!fin)
      for (auto& r : preqs_)
        if (r != MPI_REQUEST_NULL)
          MPI_Request_free(&r);
    preqs_.clear();
    preqPeer_.clear();
  }

  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<int> sendRanks_, recvRanks_, sendCounts_, recvCounts_;
  std::vector<int> sendOff_, recvOff_;
  std::vector<MPI_Request> reqs_;        // in-flight Isend/Irecv (tag != 0) between begin and end
  std::vector<MPI_Request> preqs_;       // persistent: receives then sends, on preqAware_'s buffers
  std::vector<int> reqPeer_, preqPeer_;  // per request: recv neighbour k, or nRecvRanks + send k
  bool usePersistent_ = false;           // preqs_ built (false: init failed, or no init yet)
  bool preqAware_ = false;               // staging mode preqs_ were built for
  bool inFlightPersistent_ = false;      // the in-flight exchange runs on preqs_
  bool inFlightAware_ = false;           // staging mode of the in-flight exchange
  Index nSend_ = 0, nRecv_ = 0, nSelf_ = 0;
  std::string label_;
  IndexView d_sendIdx_, d_recvIdx_, d_selfSrc_, d_selfDst_;

  // On-node shared-memory transport (host memory space only; empty otherwise).
  MPI_Comm node_ = MPI_COMM_NULL;        // the ranks of comm_ on this node (MPI_COMM_TYPE_SHARED)
  MPI_Win win_ = MPI_WIN_NULL;           // every node rank's receive segment
  bool winLocked_ = false;               // the passive-target epoch (lock_all) of win_ is open
  bool shm_ = false;                     // on-node partners use the window
  bool inFlightShm_ = false;             // the in-flight exchange uses it
  std::uint64_t seq_ = 0;                // exchanges of this instance on the window (s of the last)
  std::vector<char> sendShm_, recvShm_;  // per send / receive neighbour: on the window?
  std::vector<ShmPeer> shmSend_, shmRecv_;  // the on-node receivers / senders
  std::vector<T*> sendDst_[2];              // per parity, per send element: slot or sendBuf entry
  std::vector<const T*> recvSrc_[2];        // per parity, per receive element: slot or recvBuf

  int worldRank(int r) const {
    MPI_Group g, wg;
    MPI_Comm_group(comm_, &g);
    MPI_Comm_group(MPI_COMM_WORLD, &wg);
    int w = -1;
    MPI_Group_translate_ranks(g, 1, &r, wg, &w);
    MPI_Group_free(&g);
    MPI_Group_free(&wg);
    return w;
  }
  std::string timeoutHeader(double timeout) const {
    int rank = -1, size = -1, wrank = -1, wsize = -1;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    return "peclet.core halo TIMEOUT after " + std::to_string(int(timeout)) + " s [" + label_ +
           "] world " + std::to_string(wrank) + "/" + std::to_string(wsize) + " comm " +
           std::to_string(rank) + "/" + std::to_string(size) + ": " +
           std::to_string(recvRanks_.size()) + " recv + " + std::to_string(sendRanks_.size()) +
           " send partners (" + std::to_string(shmRecv_.size()) + " + " +
           std::to_string(shmSend_.size()) + " on shared memory)\n";
  }

  /// MPI_Waitall with a deadline (PECLET_CORE_HALO_TIMEOUT seconds). On expiry, print every
  /// request still pending — direction, partner (communicator AND world rank), byte count — and
  /// abort the job. A hang in a halo exchange is otherwise invisible (every rank sits in
  /// MPI_Waitall and a stack sample cannot say WHICH message never arrived); this names it.
  /// `peer[k]` is request k's neighbour: a receive neighbour index, or nRecvRanks + a send one.
  void waitAllDiagnosed(std::vector<MPI_Request>& reqs, const std::vector<int>& peer,
                        double timeout) {
    const double t0 = MPI_Wtime();
    const int n = static_cast<int>(reqs.size());
    int done = 0;
    while (!done) {
      MPI_Testall(n, reqs.data(), &done, MPI_STATUSES_IGNORE);
      if (done || MPI_Wtime() - t0 < timeout)
        continue;
      std::string msg = timeoutHeader(timeout);
      const std::size_t nr = recvRanks_.size();
      for (std::size_t k = 0; k < reqs.size(); ++k) {
        int flag = 0;
        MPI_Request_get_status(reqs[k], &flag, MPI_STATUS_IGNORE);
        if (flag)
          continue;
        const std::size_t m = static_cast<std::size_t>(peer[k]);
        const bool recv = m < nr;
        const int q = recv ? recvRanks_[m] : sendRanks_[m - nr];
        const int c = recv ? recvCounts_[m] : sendCounts_[m - nr];
        msg += std::string("   pending ") + (recv ? "RECV from" : "SEND to  ") + " comm " +
               std::to_string(q) + " (world " + std::to_string(worldRank(q)) + ") " +
               std::to_string(c * static_cast<long>(sizeof(T))) + " bytes\n";
      }
      std::fputs(msg.c_str(), stderr);
      std::fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 77);
    }
  }
  View<T> d_sendBuf_, d_recvBuf_;
  HostView<T> h_sendBuf_, h_recvBuf_;
};

}  // namespace peclet::core::halo

#endif  // PECLET_CORE_HALO_GRID_HALO_HPP

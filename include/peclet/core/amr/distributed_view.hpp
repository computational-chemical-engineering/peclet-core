// transport-core — device-resident distributed AMR Poisson + multigrid (C2).
//
// The host DistributedPoisson / DistributedMultigrid (distributed_poisson.hpp) run every per-cell
// apply / jacobi / residual / restrict / prolong on host std::vector, gathering cross-block ghosts by
// re-issuing an owner request/reply each matvec. Only the MPI byte exchange truly needs the host; the
// compute is embarrassingly parallel per cell. This header keeps the field on the device and runs all
// of it as Kokkos kernels, mirroring only the compact ghost buffer across MPI — the octree analogue of
// grid_halo.hpp's GridHalo.
//
//   DistributedGatherHalo — value-only face-neighbour gather over a topology established ONCE
//     (DistributedOctree::buildGatherHaloTopology): device pack/scatter kernels + a host-staged compact
//     MPI values exchange. No per-matvec coord exchange / locateGlobal.
//   DistributedPoissonView — apply / jacobi / residual as parallel_for over the device field.
//   DistributedMultigridView — the V-cycle: device jacobi + the (local) child-CSR restrict /
//     piecewise-constant prolong reused from multigrid.hpp (bit-exact, deterministic).
//
// Bit-exactness contract (the suite's distributed lock): the SOLUTION is bit-for-bit identical across
// rank counts and to the single-block MPI_COMM_SELF reference — apply/jacobi are per-cell, restrict is a
// fixed-order child-CSR gather (no atomics), so floating-point order is preserved. (The residual L2 norm
// uses a parallel reduction whose order differs, but the norm is only a convergence scalar, never the
// solution.) GPU is tolerance-not-bit-exact (FMA) by the documented convention.
//
// Requires a Kokkos build + MPI + the morton checkout (PECLET_CORE_HAVE_MORTON).
#ifndef PECLET_CORE_AMR_DISTRIBUTED_VIEW_HPP
#define PECLET_CORE_AMR_DISTRIBUTED_VIEW_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <cmath>
#include <memory>
#include <vector>

#include "peclet/core/amr/distributed_poisson.hpp"  // DistributedOctree, AmrGeometry, DistributedMultigrid (ref)
#include "peclet/core/amr/multigrid.hpp"            // restrictField / prolongAdd (bit-exact, reused)
#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/halo/grid_halo.hpp"  // detail::gpuAwareMpi() (PECLET_CORE_GPU_AWARE_MPI opt-in, H1)

namespace peclet::core::amr {

/// Value-only, device-resident face-neighbour gather over a fixed topology (C2). Built once from a
/// DistributedOctree::GatherHaloTopology; thereafter `gather(x, g)` moves only the compact send/recv
/// `double` buffers across MPI (host-staged), with device pack/scatter kernels.
template <int Dim, unsigned Bits>
class DistributedGatherHalo {
 public:
  using DO = DistributedOctree<Dim, Bits>;

  void init(const DO& d, const typename DO::GatherHaloTopology& t) {
    comm_ = d.comm();
    nFaces_ = t.nFaces;
    nLocal_ = static_cast<Index>(t.localSlot.size());
    nSend_ = static_cast<Index>(t.sendLeaf.size());
    nRecv_ = static_cast<Index>(t.recvSlot.size());
    d_localSlot_ = toDevice(t.localSlot, "dgh::localSlot");
    d_localLeaf_ = toDevice(t.localLeaf, "dgh::localLeaf");
    d_sendLeaf_ = toDevice(t.sendLeaf, "dgh::sendLeaf");
    d_recvSlot_ = toDevice(t.recvSlot, "dgh::recvSlot");
    sendRanks_ = t.sendRanks;
    sendCounts_ = t.sendCounts;
    recvRanks_ = t.recvRanks;
    recvCounts_ = t.recvCounts;
    sendOff_.assign(sendCounts_.size() + 1, 0);
    for (std::size_t k = 0; k < sendCounts_.size(); ++k) sendOff_[k + 1] = sendOff_[k] + sendCounts_[k];
    recvOff_.assign(recvCounts_.size() + 1, 0);
    for (std::size_t k = 0; k < recvCounts_.size(); ++k) recvOff_[k + 1] = recvOff_[k] + recvCounts_[k];
    d_sendBuf_ = View<double>(Kokkos::view_alloc("dgh::sendBuf", Kokkos::WithoutInitializing),
                              static_cast<std::size_t>(nSend_));
    d_recvBuf_ = View<double>(Kokkos::view_alloc("dgh::recvBuf", Kokkos::WithoutInitializing),
                              static_cast<std::size_t>(nRecv_));
    h_sendBuf_ = Kokkos::create_mirror_view(d_sendBuf_);
    h_recvBuf_ = Kokkos::create_mirror_view(d_recvBuf_);
  }

  Index nFaces() const { return nFaces_; }

  /// g (size nFaces) := the face-neighbour values of x. Local faces fill on device; cross-rank faces
  /// cross MPI as a compact host-staged values buffer (no coords). Bit-identical to the host
  /// faceNeighborGather over the same field (periodic ⇒ every slot is filled; no sentinels).
  void gather(View<const double> x, View<double> g, int tag = 41) const {
    const double sentinel = -1e300;  // == DistributedOctree::kNoNeighbor; unused in the periodic case
    // GPU-aware MPI (H1): when enabled (PECLET_CORE_GPU_AWARE_MPI) hand the device send/recv buffer pointers
    // straight to MPI — the field never touches the host even for the compact buffers. Default is the
    // portable host-staged path (deep_copy to a host mirror), exactly as GridHalo does.
    const bool aware = peclet::core::halo::detail::gpuAwareMpi();
    Kokkos::deep_copy(g, sentinel);
    if (nLocal_) {
      IndexView ls = d_localSlot_, ll = d_localLeaf_;
      View<double> gv = g;
      Kokkos::parallel_for(
          "dgh::local", Kokkos::RangePolicy<ExecSpace>(0, nLocal_),
          KOKKOS_LAMBDA(const Index k) { gv(ls(k)) = x(ll(k)); });
    }
    if (nSend_) {
      IndexView sl = d_sendLeaf_;
      View<double> buf = d_sendBuf_;
      Kokkos::parallel_for(
          "dgh::pack", Kokkos::RangePolicy<ExecSpace>(0, nSend_),
          KOKKOS_LAMBDA(const Index p) { buf(p) = (sl(p) >= 0) ? x(sl(p)) : sentinel; });
      if (!aware) Kokkos::deep_copy(h_sendBuf_, d_sendBuf_);
    }
    Kokkos::fence();  // send buffer (host-staged or device) ready before MPI reads it
    double* sendBase = aware ? d_sendBuf_.data() : h_sendBuf_.data();
    double* recvBase = aware ? d_recvBuf_.data() : h_recvBuf_.data();
    std::vector<MPI_Request> reqs;
    reqs.reserve(recvRanks_.size() + sendRanks_.size());
    for (std::size_t k = 0; k < recvRanks_.size(); ++k) {
      reqs.emplace_back();
      MPI_Irecv(recvBase + recvOff_[k], recvCounts_[k] * static_cast<int>(sizeof(double)), MPI_BYTE,
                recvRanks_[k], tag, comm_, &reqs.back());
    }
    for (std::size_t k = 0; k < sendRanks_.size(); ++k) {
      reqs.emplace_back();
      MPI_Isend(sendBase + sendOff_[k], sendCounts_[k] * static_cast<int>(sizeof(double)), MPI_BYTE,
                sendRanks_[k], tag, comm_, &reqs.back());
    }
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    if (nRecv_) {
      if (!aware) Kokkos::deep_copy(d_recvBuf_, h_recvBuf_);
      IndexView rs = d_recvSlot_;
      View<double> buf = d_recvBuf_, gv = g;
      Kokkos::parallel_for(
          "dgh::scatter", Kokkos::RangePolicy<ExecSpace>(0, nRecv_),
          KOKKOS_LAMBDA(const Index k) { gv(rs(k)) = buf(k); });
    }
    Kokkos::fence();
  }

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  Index nFaces_ = 0, nLocal_ = 0, nSend_ = 0, nRecv_ = 0;
  IndexView d_localSlot_, d_localLeaf_, d_sendLeaf_, d_recvSlot_;
  std::vector<int> sendRanks_, sendCounts_, sendOff_, recvRanks_, recvCounts_, recvOff_;
  View<double> d_sendBuf_, d_recvBuf_;
  HostView<double> h_sendBuf_, h_recvBuf_;
};

/// Device-resident distributed plain-Laplacian Poisson operator: y = ∇²x with cross-block neighbours
/// from the device gather halo. apply / jacobi / residual run as Kokkos kernels over the device field.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DistributedPoissonView {
 public:
  void init(DistributedOctree<Dim, Bits>& d, double h0) {
    d_ = &d;
    h0_ = h0;
    auto plan = d.buildFaceGatherPlan();
    halo_.init(d, d.buildGatherHaloTopology(plan));
    n_ = d.local().numLeaves();
    g_ = View<double>(Kokkos::view_alloc("dpd::g", Kokkos::WithoutInitializing),
                      static_cast<std::size_t>(n_) * (2 * Dim));
    scratch_ = View<double>(Kokkos::view_alloc("dpd::scratch", Kokkos::WithoutInitializing),
                            static_cast<std::size_t>(n_));
  }

  Index numLeaves() const { return n_; }
  MPI_Comm comm() const { return d_->comm(); }

  /// y = L x (L = ∇², periodic).
  void apply(View<const double> x, View<double> y) const {
    halo_.gather(x, g_);
    const int F = 2 * Dim;
    const double inv = 1.0 / (h0_ * h0_);
    View<double> gv = g_;
    Kokkos::parallel_for(
        "dpd::apply", Kokkos::RangePolicy<ExecSpace>(0, n_), KOKKOS_LAMBDA(const Index i) {
          double s = 0.0;
          for (int f = 0; f < F; ++f) s += gv(i * F + f) - x(i);
          y(i) = inv * s;
        });
  }

  /// `sweeps` damped-Jacobi relaxations of L u = b (in place). Reads only the previous iterate.
  void jacobi(View<double> x, View<const double> b, int sweeps, double omega = 0.8) const {
    const int F = 2 * Dim;
    const double inv = 1.0 / (h0_ * h0_), diag = F * inv;
    View<double> lx = scratch_;
    for (int s = 0; s < sweeps; ++s) {
      apply(View<const double>(x), lx);
      Kokkos::parallel_for(
          "dpd::jacobi_update", Kokkos::RangePolicy<ExecSpace>(0, n_),
          KOKKOS_LAMBDA(const Index i) { x(i) += omega * (lx(i) - b(i)) / diag; });
    }
  }

  /// res = b − L x (device); returns the global L2 norm (the only host scalar).
  double residual(View<const double> x, View<const double> b, View<double> res) const {
    apply(x, scratch_);
    View<double> ax = scratch_;
    double s = 0.0;
    Kokkos::parallel_reduce(
        "dpd::residual", Kokkos::RangePolicy<ExecSpace>(0, n_),
        KOKKOS_LAMBDA(const Index i, double& acc) {
          double r = b(i) - ax(i);
          res(i) = r;
          acc += r * r;
        },
        s);
    double g = 0.0;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, d_->comm());
    return std::sqrt(g);
  }

  double residualNorm(View<const double> x, View<const double> b) const {
    apply(x, scratch_);
    View<double> ax = scratch_;
    double s = 0.0;
    Kokkos::parallel_reduce(
        "dpd::resnorm", Kokkos::RangePolicy<ExecSpace>(0, n_),
        KOKKOS_LAMBDA(const Index i, double& acc) {
          double r = b(i) - ax(i);
          acc += r * r;
        },
        s);
    double g = 0.0;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, d_->comm());
    return std::sqrt(g);
  }

 private:
  DistributedOctree<Dim, Bits>* d_ = nullptr;
  double h0_ = 1.0;
  Index n_ = 0;
  DistributedGatherHalo<Dim, Bits> halo_;
  mutable View<double> g_, scratch_;
};

/// Device-resident distributed geometric multigrid for the plain Laplacian (the device analogue of the
/// host DistributedMultigrid). Hierarchy + nesting maps mirror the host build; smoother is the device
/// distributed Jacobi; restriction (child-CSR gather) and prolongation (piecewise-constant) are the
/// local, bit-exact device kernels from multigrid.hpp.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DistributedMultigridView {
 public:
  using Octree = DistributedOctree<Dim, Bits>;

  void build(const IVec<Dim>& g0, const AmrGeometry<Dim>& geo, const std::array<bool, Dim>& periodic,
             MPI_Comm comm) {
    levels_.clear();
    int size = 1;
    MPI_Comm_size(comm, &size);
    IVec<Dim> g = g0;
    double h = geo.h0;
    for (;;) {
      auto lvl = std::make_unique<Level>();
      AmrGeometry<Dim> lg = geo;
      lg.h0 = h;
      lvl->d.init(g, /*lmax=*/0, lg, periodic, comm);
      lvl->op.init(lvl->d, h);
      lvl->n = lvl->d.local().numLeaves();
      lvl->x = View<double>("dmg::x", static_cast<std::size_t>(lvl->n));
      lvl->b = View<double>("dmg::b", static_cast<std::size_t>(lvl->n));
      lvl->res = View<double>("dmg::res", static_cast<std::size_t>(lvl->n));
      levels_.push_back(std::move(lvl));
      bool ok = true;
      long prod = 1;
      IVec<Dim> ng{};
      for (int d = 0; d < Dim; ++d) {
        if (g[d] % 2 != 0 || g[d] / 2 < 2) ok = false;
        ng[d] = g[d] / 2;
        prod *= ng[d];
      }
      if (!ok || prod < size) break;
      g = ng;
      h *= 2.0;
    }
    // Nested fine→coarse maps + the coarse→children CSR (fine-index order ⇒ the restrict gather sums in
    // the same order as the host serial accumulation, hence bit-exact). All local (ORB nests).
    for (std::size_t L = 0; L + 1 < levels_.size(); ++L) {
      auto& fine = levels_[L]->d;
      auto& coarse = levels_[L + 1]->d;
      const Index nf = fine.local().numLeaves();
      const Index nc = coarse.local().numLeaves();
      std::vector<Index> c2p(static_cast<std::size_t>(nf), -1);
      std::vector<Index> cnt(static_cast<std::size_t>(nc), 0);
      for (Index i = 0; i < nf; ++i) {
        IVec<Dim> gf = fine.globalRootOf(i), gc{};
        for (int d = 0; d < Dim; ++d) gc[d] = gf[d] / 2;
        Index p = coarse.findGlobalRoot(gc);
        c2p[static_cast<std::size_t>(i)] = p;
        if (p >= 0) ++cnt[static_cast<std::size_t>(p)];
      }
      std::vector<Index> start(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p)
        start[static_cast<std::size_t>(p) + 1] = start[static_cast<std::size_t>(p)] + cnt[static_cast<std::size_t>(p)];
      std::vector<Index> idx(static_cast<std::size_t>(start[static_cast<std::size_t>(nc)]));
      std::vector<Index> cur(start.begin(), start.end() - 1);
      for (Index i = 0; i < nf; ++i) {  // increasing fine index ⇒ children listed in fine order
        Index p = c2p[static_cast<std::size_t>(i)];
        if (p >= 0) idx[static_cast<std::size_t>(cur[static_cast<std::size_t>(p)]++)] = i;
      }
      levels_[L]->c2p = toDevice(c2p, "dmg::c2p");
      levels_[L]->childStart = toDevice(start, "dmg::cstart");
      levels_[L]->childIdx = toDevice(idx, "dmg::cidx");
    }
  }

  std::size_t numLevels() const { return levels_.size(); }
  Index numLeaves(std::size_t L = 0) const { return levels_[L]->n; }
  DistributedPoissonView<Dim, Bits>& op(std::size_t L = 0) { return levels_[L]->op; }
  Octree& octree(std::size_t L = 0) { return levels_[L]->d; }
  View<double> x(std::size_t L = 0) { return levels_[L]->x; }
  View<double> b(std::size_t L = 0) { return levels_[L]->b; }

  /// One V-cycle of A x = b on level L (correction scheme); entirely device-resident bar the compact
  /// gather buffers + the residual-norm Allreduce.
  void vcycle(View<double> x, View<const double> b, int pre = 2, int post = 2, int bottom = 30,
              std::size_t L = 0) {
    Level& lv = *levels_[L];
    if (L + 1 == levels_.size()) {
      lv.op.jacobi(x, b, bottom);
      return;
    }
    lv.op.jacobi(x, b, pre);
    lv.op.residual(View<const double>(x), b, lv.res);
    Level& cl = *levels_[L + 1];
    restrictField(View<const Index>(lv.childStart), View<const Index>(lv.childIdx),
                   View<const double>(lv.res), cl.b, cl.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(cl.x, View<const double>(cl.b), pre, post, bottom, L + 1);
    prolongAdd(View<const Index>(lv.c2p), View<const double>(cl.x), x, lv.n);
    lv.op.jacobi(x, b, post);
  }

 private:
  struct Level {
    Octree d;
    DistributedPoissonView<Dim, Bits> op;
    Index n = 0;
    View<double> x, b, res;
    View<Index> c2p, childStart, childIdx;
  };
  std::vector<std::unique_ptr<Level>> levels_;
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_DISTRIBUTED_VIEW_HPP

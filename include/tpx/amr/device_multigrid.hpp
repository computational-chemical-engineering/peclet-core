// transport-core — device (Kokkos) geometric-multigrid V-cycle on a BlockOctree.
//
// The device compute path's capstone: a full MG V-cycle for the plain Laplacian
// running entirely in device kernels over the leaf Views. The hierarchy is the
// octree coarsened uniformly one level at a time (same as the host AmrMultigrid),
// uploaded as a stack of DeviceBlockOctrees; each level carries its scratch Views
// and the fine↔coarse transfer maps as device Views.
//
// Every stage is deterministic / order-independent, so the result is bit-identical
// to the same Jacobi V-cycle on the host (validated in test_amr_device_multigrid_kokkos):
//   * smoother   — weighted Jacobi (deviceJacobiSweep): reads only the prev iterate;
//   * restriction— each *coarse* cell sums its children in a fixed (CSR) order, so the
//                  summation order matches the host exactly (no atomics);
//   * prolongation — piecewise-constant, per fine cell.
//
// Scope: uniform plain Laplacian (openness-free), homogeneous-Neumann block edges —
// the same operator as device_poisson.hpp; openness / graded / distributed-on-device
// build on this. Requires a Kokkos build + the morton checkout (TPX_HAVE_MORTON).
#ifndef TPX_AMR_DEVICE_MULTIGRID_HPP
#define TPX_AMR_DEVICE_MULTIGRID_HPP

#ifdef TPX_HAVE_MORTON

#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/block_octree_kokkos.hpp"
#include "tpx/amr/device_poisson.hpp"
#include "tpx/common/view.hpp"

namespace tpx::amr {

/// res(i) = b(i) − A x(i), with A = −∇²  ⇒  res = b + ∇²x  (device).
template <int Dim, unsigned Bits>
void deviceResidual(DeviceBlockOctree<Dim, Bits> dev, View<double> x, View<const double> b,
                    View<double> res, double inv) {
  const Index n = dev.numLeaves();
  Kokkos::parallel_for(
      "amr::device_residual", n, KOKKOS_LAMBDA(const Index i) {
        double s = 0.0;
        for (int axis = 0; axis < Dim; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) {
            Index j = dev.faceNeighbor(i, axis, dir);
            if (j >= 0) s += x(j) - x(i);
          }
        res(i) = b(i) + inv * s;  // b − A x = b + ∇²x
      });
}

/// Restrict: coarse(p) = mean over p's children (CSR fixed order ⇒ deterministic).
inline void deviceRestrict(View<const Index> childStart, View<const Index> childIdx,
                           View<const double> fine, View<double> coarse, Index nCoarse) {
  Kokkos::parallel_for(
      "amr::device_restrict", nCoarse, KOKKOS_LAMBDA(const Index p) {
        const Index a = childStart(p), z = childStart(p + 1);
        double s = 0.0;
        for (Index k = a; k < z; ++k) s += fine(childIdx(k));
        coarse(p) = (z > a) ? s / static_cast<double>(z - a) : 0.0;
      });
}

/// Prolong (piecewise-constant) + correct: fine(i) += coarse(c2p(i)).
inline void deviceProlongAdd(View<const Index> c2p, View<const double> coarse, View<double> fine,
                             Index nFine) {
  Kokkos::parallel_for(
      "amr::device_prolong", nFine, KOKKOS_LAMBDA(const Index i) {
        const Index p = c2p(i);
        if (p >= 0) fine(i) += coarse(p);
      });
}

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DeviceMultigrid {
 public:
  using Octree = BlockOctree<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;

  /// Build + upload the hierarchy from a finest octree (uniform coarsening). `h0` is
  /// the finest spacing; level L uses inv = 1/(h0·2^L)².
  void build(const Octree& finest, double h0) {
    levels_.clear();
    std::vector<Octree> host;
    host.push_back(finest);
    for (;;) {
      Octree c = host.back();
      Index merged = c.coarsenIf([](Code, unsigned) { return true; });
      if (merged == 0 || c.numLeaves() == host.back().numLeaves()) break;
      host.push_back(c);
      if (c.numLeaves() == 1) break;
    }
    levels_.resize(host.size());
    for (std::size_t L = 0; L < host.size(); ++L) {
      Level& lv = levels_[L];
      lv.n = host[L].numLeaves();
      double hL = h0 * static_cast<double>(Index(1) << L);
      lv.inv = 1.0 / (hL * hL);
      lv.dev.upload(host[L]);
      lv.x = View<double>("mg_x", static_cast<std::size_t>(lv.n));
      lv.b = View<double>("mg_b", static_cast<std::size_t>(lv.n));
      lv.res = View<double>("mg_res", static_cast<std::size_t>(lv.n));
      lv.ax = View<double>("mg_ax", static_cast<std::size_t>(lv.n));
    }
    // fine→coarse map + CSR coarse→children, per fine/coarse pair.
    for (std::size_t L = 0; L + 1 < host.size(); ++L) {
      const Octree& f = host[L];
      const Octree& c = host[L + 1];
      const Index nf = f.numLeaves(), nc = c.numLeaves();
      std::vector<Index> c2p(static_cast<std::size_t>(nf));
      std::vector<Index> cnt(static_cast<std::size_t>(nc), 0);
      for (Index i = 0; i < nf; ++i) {
        Code parent = M::from_code(f.code(i)).ancestor(f.level(i) + 1).code();
        Index p = c.find(parent);
        c2p[static_cast<std::size_t>(i)] = p;
        if (p >= 0) ++cnt[static_cast<std::size_t>(p)];
      }
      std::vector<Index> start(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p) start[static_cast<std::size_t>(p) + 1] = start[static_cast<std::size_t>(p)] + cnt[static_cast<std::size_t>(p)];
      std::vector<Index> idx(static_cast<std::size_t>(nf));
      std::vector<Index> cur(start.begin(), start.end() - 1);
      for (Index i = 0; i < nf; ++i) {  // ascending i ⇒ children stored in leaf order
        Index p = c2p[static_cast<std::size_t>(i)];
        if (p >= 0) idx[static_cast<std::size_t>(cur[static_cast<std::size_t>(p)]++)] = i;
      }
      levels_[L].c2p = toDevice(c2p, "mg_c2p");
      levels_[L].childStart = toDevice(start, "mg_cstart");
      levels_[L].childIdx = toDevice(idx, "mg_cidx");
    }
  }

  std::size_t numLevels() const { return levels_.size(); }
  Index numLeaves(std::size_t L = 0) const { return levels_[L].n; }
  View<double> x(std::size_t L = 0) { return levels_[L].x; }
  View<double> b(std::size_t L = 0) { return levels_[L].b; }

  /// One V-cycle on level `L` (default finest), correction scheme.
  void vcycle(int pre = 2, int post = 2, int bottom = 30, double omega = 0.8, std::size_t L = 0) {
    Level& lv = levels_[L];
    View<const double> bc(lv.b);
    if (L + 1 == levels_.size()) {
      for (int s = 0; s < bottom; ++s)
        deviceJacobiSweep<Dim, Bits>(lv.dev, lv.x, bc, lv.ax, lv.inv, omega);
      return;
    }
    for (int s = 0; s < pre; ++s) deviceJacobiSweep<Dim, Bits>(lv.dev, lv.x, bc, lv.ax, lv.inv, omega);
    deviceResidual<Dim, Bits>(lv.dev, lv.x, bc, lv.res, lv.inv);
    Level& cl = levels_[L + 1];
    deviceRestrict(lv.childStart, lv.childIdx, View<const double>(lv.res), cl.b, cl.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(pre, post, bottom, omega, L + 1);
    deviceProlongAdd(lv.c2p, View<const double>(cl.x), lv.x, lv.n);
    for (int s = 0; s < post; ++s) deviceJacobiSweep<Dim, Bits>(lv.dev, lv.x, bc, lv.ax, lv.inv, omega);
  }

 private:
  struct Level {
    DeviceBlockOctree<Dim, Bits> dev;
    Index n = 0;
    double inv = 1.0;
    View<double> x, b, res, ax;
    View<Index> c2p, childStart, childIdx;  // describe L → L+1 (unused on coarsest)
  };
  std::vector<Level> levels_;
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_MULTIGRID_HPP

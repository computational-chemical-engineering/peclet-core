// transport-core — device (Kokkos) Poisson operator + smoother on a DeviceBlockOctree.
//
// The device compute path for the AMR octree: the Laplacian matvec and a weighted-
// Jacobi smoother run as Kokkos::parallel_for over the leaf Views, using the
// device-callable face-neighbour walk of DeviceBlockOctree (block_octree_kokkos.hpp).
// Same arithmetic as the host BlockOctree::faceNeighbor, so the result is identical
// to the host operator (validated bit-for-bit in tests/test_amr_device_poisson_kokkos).
//
// Scope: the plain (uniform, openness-free) Laplacian over the leaves; a face with
// no neighbour (block edge) is skipped (homogeneous Neumann). Openness / graded /
// distributed-on-device build on this + the device halo.
//
// Requires a Kokkos build (MORTON_ENABLE_KOKKOS ⇒ MORTON_HD = KOKKOS_FUNCTION) and
// the morton checkout (TPX_HAVE_MORTON).
#ifndef TPX_AMR_DEVICE_POISSON_HPP
#define TPX_AMR_DEVICE_POISSON_HPP

#ifdef TPX_HAVE_MORTON

#include "tpx/amr/block_octree_kokkos.hpp"
#include "tpx/common/view.hpp"

namespace tpx::amr {

/// y = inv · Σ_faces (x_nb − x_i)  (= ∇² in spacing h0 with inv = 1/h0²), on device.
template <int Dim, unsigned Bits>
void deviceLaplacian(DeviceBlockOctree<Dim, Bits> dev, View<double> x, View<double> y, double inv) {
  const Index n = dev.numLeaves();
  Kokkos::parallel_for(
      "amr::device_laplacian", n, KOKKOS_LAMBDA(const Index i) {
        double s = 0.0;
        for (int axis = 0; axis < Dim; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) {
            Index j = dev.faceNeighbor(i, axis, dir);
            if (j >= 0) s += x(j) - x(i);
          }
        y(i) = inv * s;
      });
}

/// One weighted-Jacobi sweep of A x = b with A = −∇² (diagonal 2·Dim·inv), on device.
/// `ax` is scratch (Ax). x updated in place.
template <int Dim, unsigned Bits>
void deviceJacobiSweep(DeviceBlockOctree<Dim, Bits> dev, View<double> x, View<const double> b,
                       View<double> ax, double inv, double omega) {
  const Index n = dev.numLeaves();
  const double diag = 2.0 * Dim * inv;
  // ax = A x = −∇² x
  Kokkos::parallel_for(
      "amr::device_jacobi_apply", n, KOKKOS_LAMBDA(const Index i) {
        double s = 0.0;
        for (int axis = 0; axis < Dim; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) {
            Index j = dev.faceNeighbor(i, axis, dir);
            if (j >= 0) s += x(j) - x(i);
          }
        ax(i) = -inv * s;
      });
  Kokkos::parallel_for(
      "amr::device_jacobi_update", n,
      KOKKOS_LAMBDA(const Index i) { x(i) += omega * (b(i) - ax(i)) / diag; });
}

// ===========================================================================
// Consistent FV (graded) operator as a precomputed face CSR.
//
// The plain operator above ignores 2:1-interface geometry (it treats every face
// as same-level, spacing h0). The *consistent* conservative FV Laplacian is
//   (L u)_i = invVol_i · Σ_{f ∈ [start_i, start_{i+1})} w_f · (u[nbr_f] − u_i),
// with w_f = openness · A_f/d_f and the coarse side of a 2:1 interface carrying
// one face per fine sub-neighbour — exactly the host AmrPoisson enumeration
// (forEachFaceNeighbor). The CSR is built once on the host in that same face
// order, so the device operator is bit-identical to the host operator (and the
// graded V-cycle converges instead of stalling on the inconsistent plain op).
// ===========================================================================
struct DeviceFvOp {
  View<double> invVol;     ///< 1/V_i, size n
  View<Index> faceStart;   ///< CSR row offsets, size n+1
  View<Index> faceNbr;     ///< neighbour leaf per face, size nFaces
  View<double> faceW;      ///< w_f = openness·A_f/d_f per face, size nFaces
  Index n = 0;
};

/// Lu = L u (consistent conservative FV Laplacian, negative-definite).
inline void deviceApplyFv(const DeviceFvOp& op, View<const double> u, View<double> Lu) {
  auto invVol = op.invVol;
  auto fs = op.faceStart;
  auto fn = op.faceNbr;
  auto fw = op.faceW;
  Kokkos::parallel_for(
      "amr::fv_apply", op.n, KOKKOS_LAMBDA(const Index i) {
        const double ui = u(i);
        double acc = 0.0;
        for (Index k = fs(i); k < fs(i + 1); ++k) acc += fw(k) * (u(fn(k)) - ui);
        Lu(i) = invVol(i) * acc;
      });
}

/// res = rhs − L u.
inline void deviceResidualFv(const DeviceFvOp& op, View<const double> u, View<const double> rhs,
                             View<double> res) {
  auto invVol = op.invVol;
  auto fs = op.faceStart;
  auto fn = op.faceNbr;
  auto fw = op.faceW;
  Kokkos::parallel_for(
      "amr::fv_residual", op.n, KOKKOS_LAMBDA(const Index i) {
        const double ui = u(i);
        double acc = 0.0;
        for (Index k = fs(i); k < fs(i + 1); ++k) acc += fw(k) * (u(fn(k)) - ui);
        res(i) = rhs(i) - invVol(i) * acc;
      });
}

/// One weighted-Jacobi sweep of L u = rhs (in place). `tmp` is scratch (size n).
/// Mirrors the host point solve u_i ← (Σ w u_j − V_i rhs_i)/Σ w with damping ω;
/// pass 1 reads only the previous iterate (into tmp), pass 2 updates — so the
/// sweep is order-independent / bit-reproducible.
inline void deviceJacobiFv(const DeviceFvOp& op, View<double> u, View<const double> rhs,
                           View<double> tmp, double omega) {
  auto invVol = op.invVol;
  auto fs = op.faceStart;
  auto fn = op.faceNbr;
  auto fw = op.faceW;
  Kokkos::parallel_for(
      "amr::fv_jacobi_compute", op.n, KOKKOS_LAMBDA(const Index i) {
        double sumOff = 0.0, diag = 0.0;
        for (Index k = fs(i); k < fs(i + 1); ++k) {
          sumOff += fw(k) * u(fn(k));
          diag += fw(k);
        }
        // V_i rhs_i = rhs_i / invVol_i ; point solve of L u = rhs for u_i.
        tmp(i) = (diag != 0.0) ? (sumOff - rhs(i) / invVol(i)) / diag : u(i);
      });
  Kokkos::parallel_for(
      "amr::fv_jacobi_update", op.n,
      KOKKOS_LAMBDA(const Index i) { u(i) = (1.0 - omega) * u(i) + omega * tmp(i); });
}

/// dq = (L_quad − L_std) u, the quadratic coarse-fine correction as its own SpMV
/// over a precomputed CSR (built from AmrPoisson::coarseStar). Used for deferred
/// correction: solve L_std u = rhs − dq with dq lagged ⇒ 2nd-order at 2:1 faces.
inline void deviceQuadDelta(View<const Index> qStart, View<const Index> qSlot,
                            View<const double> qCoef, View<const double> u, View<double> dq,
                            Index n) {
  Kokkos::parallel_for(
      "amr::fv_quad_delta", n, KOKKOS_LAMBDA(const Index i) {
        double acc = 0.0;
        for (Index k = qStart(i); k < qStart(i + 1); ++k) acc += qCoef(k) * u(qSlot(k));
        dq(i) = acc;
      });
}

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_POISSON_HPP

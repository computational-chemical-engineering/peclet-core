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

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_POISSON_HPP

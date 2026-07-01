// transport-core — device (Kokkos) Poisson operator + smoother on a BlockOctreeView.
//
// The device compute path for the AMR octree: the Laplacian matvec and a weighted-
// Jacobi smoother run as Kokkos::parallel_for over the leaf Views, using the
// device-callable face-neighbour walk of BlockOctreeView (block_octree_kokkos.hpp).
// Same arithmetic as the host BlockOctree::faceNeighbor, so the result is identical
// to the host operator (validated bit-for-bit in tests/test_amr_device_poisson_kokkos).
//
// Scope: the plain (uniform, openness-free) Laplacian over the leaves; a face with
// no neighbour (block edge) is skipped (homogeneous Neumann). Openness / graded /
// distributed-on-device build on this + the device halo.
//
// Requires a Kokkos build (MORTON_ENABLE_KOKKOS ⇒ MORTON_HD = KOKKOS_FUNCTION) and
// the morton checkout (PECLET_CORE_HAVE_MORTON).
#ifndef PECLET_CORE_AMR_FV_OP_HPP
#define PECLET_CORE_AMR_FV_OP_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include "peclet/core/amr/block_octree_view.hpp"
#include "peclet/core/amr/face_csr.hpp"  // shared host+device FV (weight-CSR) row kernels
#include "peclet/core/common/view.hpp"

namespace peclet::core::amr {

/// y = inv · Σ_faces (x_nb − x_i)  (= ∇² in spacing h0 with inv = 1/h0²), on device.
template <int Dim, unsigned Bits>
void laplacian(BlockOctreeView<Dim, Bits> dev, View<double> x, View<double> y, double inv) {
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

/// One weighted-Jacobi sweep of L u = b with L = ∇² (negative-definite, diagonal
/// −2·Dim·inv), on device. The update u_i += ω (L u_i − b_i)/diag, diag = 2·Dim·inv
/// (= −L_ii), has the right sign for the negative-definite L. `lx` is scratch (L x).
template <int Dim, unsigned Bits>
void jacobiSweep(BlockOctreeView<Dim, Bits> dev, View<double> x, View<const double> b,
                       View<double> lx, double inv, double omega) {
  const Index n = dev.numLeaves();
  const double diag = 2.0 * Dim * inv;
  // lx = L x = ∇² x
  Kokkos::parallel_for(
      "amr::device_jacobi_apply", n, KOKKOS_LAMBDA(const Index i) {
        double s = 0.0;
        for (int axis = 0; axis < Dim; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) {
            Index j = dev.faceNeighbor(i, axis, dir);
            if (j >= 0) s += x(j) - x(i);
          }
        lx(i) = inv * s;
      });
  Kokkos::parallel_for(
      "amr::device_jacobi_update", n,
      KOKKOS_LAMBDA(const Index i) { x(i) += omega * (lx(i) - b(i)) / diag; });
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
struct FvOp {
  View<double> invVol;     ///< 1/V_i, size n
  View<Index> faceStart;   ///< CSR row offsets, size n+1
  View<Index> faceNbr;     ///< neighbour leaf per face, size nFaces
  View<double> faceW;      ///< w_f = openness·A_f/d_f per face, size nFaces
  View<double> bcDiag;     ///< Dirichlet boundary diagonal per cell (0 if periodic), size n
  Index n = 0;
  // Helmholtz generalisation: the applied operator is H = c0·I + cD·L (c0=0, cD=1 ⇒ the
  // pure Laplacian L, the default — bit-exact unchanged). Setting c0=idiag, cD=−μ turns the
  // existing hierarchy into the momentum operator idiag·I − μ∇², so Multigrid can be
  // used as a (non-singular when c0≠0) preconditioner for the momentum BiCGStab. The L path
  // is bit-exact preserved: c0·u+cD·(L) with c0=0,cD=1 is L exactly in IEEE arithmetic, and
  // the Jacobi point-solve branches on (c0==0 && cD==1) to keep the original expression.
  double c0 = 0.0;
  double cD = 1.0;
};

/// View the assembled FV operator through the shared backend-agnostic FvCsrOpT, so the device
/// kernels and the host AmrPoisson run the *same* row arithmetic (face_csr.hpp).
inline FvCsrOpT<View<const double>, View<const Index>> fvView(const FvOp& op) {
  FvCsrOpT<View<const double>, View<const Index>> v;
  v.n = op.n;
  v.invVol = op.invVol;
  v.coef = op.faceW;
  v.start = op.faceStart;
  v.nbr = op.faceNbr;
  v.bcDiag = op.bcDiag;
  v.c0 = op.c0;
  v.cD = op.cD;
  return v;
}

/// Hu = (c0·I + cD·L) u (consistent conservative FV Laplacian, c0=0/cD=1 ⇒ pure L). A
/// non-zero bcDiag adds the homogeneous-Dirichlet boundary term −bcDiag·u_i to L.
inline void applyFv(const FvOp& op, View<const double> u, View<double> Lu) {
  const auto A = fvView(op);
  Kokkos::parallel_for(
      "amr::fv_apply", op.n, KOKKOS_LAMBDA(const Index i) { Lu(i) = fvApplyRow(A, i, u); });
}

/// res = rhs − H u.
inline void residualFv(const FvOp& op, View<const double> u, View<const double> rhs,
                             View<double> res) {
  const auto A = fvView(op);
  Kokkos::parallel_for(
      "amr::fv_residual", op.n,
      KOKKOS_LAMBDA(const Index i) { res(i) = rhs(i) - fvApplyRow(A, i, u); });
}

/// One weighted-Jacobi sweep of H u = rhs (in place). `tmp` is scratch (size n).
/// Mirrors the host point solve u_i ← (Σ w u_j − V_i rhs_i)/Σ w with damping ω;
/// pass 1 reads only the previous iterate (into tmp), pass 2 updates — so the
/// sweep is order-independent / bit-reproducible. The pure-L path (c0=0, cD=1)
/// keeps the exact original expression (bit-exact); the Helmholtz path uses the
/// point solve of (c0·I + cD·L) u = rhs.
inline void jacobiFv(const FvOp& op, View<double> u, View<const double> rhs,
                           View<double> tmp, double omega) {
  const auto A = fvView(op);
  Kokkos::parallel_for(
      "amr::fv_jacobi_compute", op.n,
      KOKKOS_LAMBDA(const Index i) { tmp(i) = fvPointSolve(A, i, u, rhs(i), u(i)); });
  Kokkos::parallel_for(
      "amr::fv_jacobi_update", op.n,
      KOKKOS_LAMBDA(const Index i) { u(i) = (1.0 - omega) * u(i) + omega * tmp(i); });
}

/// Project `u` to volume-weighted-mean-zero over the ACTIVE (fluid) cells of the operator —
/// the constant-nullspace removal for the singular periodic (pure-Neumann) operator. A cell is
/// active when its diagonal (Σw + bc) > 0; fully-closed (solid) cells are excluded. Mirrors
/// sdflow CutcellMG::removeMean (sum over cells with AC > 1e-30). Applied at every V-cycle level
/// so the multigrid preconditioner does not drift / amplify the nullspace.
inline void removeMeanFv(const FvOp& op, View<double> u) {
  auto invVol = op.invVol;
  auto fs = op.faceStart;
  auto fw = op.faceW;
  auto bc = op.bcDiag;
  double sx = 0.0, sv = 0.0;
  // Numerator and denominator share the same active-cell diagonal d = bc + Σfw; fold them into a
  // single multi-output reduction so that diagonal is walked once instead of twice. Each output keeps
  // its original term verbatim (u/invVol, 1/invVol) and its own Sum reducer over the same range, so
  // the result is bit-identical to the two separate reduces. (The subtract pass below recomputes d a
  // third time but is a parallel_for, where reloading a cached mask would cost the same as the walk.)
  Kokkos::parallel_reduce(
      "amr::fv_rmean", op.n,
      KOKKOS_LAMBDA(const Index i, double& an, double& ad) {
        double d = bc(i);
        for (Index k = fs(i); k < fs(i + 1); ++k) d += fw(k);
        if (d > 1e-30) {
          an += u(i) / invVol(i);  // V_i = 1/invVol_i
          ad += 1.0 / invVol(i);
        }
      },
      sx, sv);
  if (sv <= 0.0) return;
  const double m = sx / sv;
  Kokkos::parallel_for(
      "amr::fv_rmean_sub", op.n, KOKKOS_LAMBDA(const Index i) {
        double d = bc(i);
        for (Index k = fs(i); k < fs(i + 1); ++k) d += fw(k);
        if (d > 1e-30) u(i) -= m;
      });
}

/// dq = (L_quad − L_std) u, the quadratic coarse-fine correction as its own SpMV
/// over a precomputed CSR (built from AmrPoisson::coarseStar). Used for deferred
/// correction: solve L_std u = rhs − dq with dq lagged ⇒ 2nd-order at 2:1 faces.
inline void quadDelta(View<const Index> qStart, View<const Index> qSlot,
                            View<const double> qCoef, View<const double> u, View<double> dq,
                            Index n) {
  Kokkos::parallel_for(
      "amr::fv_quad_delta", n, KOKKOS_LAMBDA(const Index i) {
        double acc = 0.0;
        for (Index k = qStart(i); k < qStart(i + 1); ++k) acc += qCoef(k) * u(qSlot(k));
        dq(i) = acc;
      });
}

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_FV_OP_HPP

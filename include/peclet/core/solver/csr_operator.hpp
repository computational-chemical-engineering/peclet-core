// core — an assembled face-CSR operator on the device (Kokkos): matvec, residual, weighted-Jacobi
// and symmetric multicolour Gauss–Seidel sweeps.
//
// Lifted verbatim from the AMR tree (peclet/core/amr/momentum.hpp, 2026-09-10; QUALITY_PLAN G.2),
// where it is the cut-cell momentum operator A = (ρ/dt)I − μ∇² (+ implicit-FOU advection + the
// ξ-polynomial Dirichlet overlay on cut cells) assembled as a per-cell diagonal + general face CSR,
// uploaded once and applied / smoothed entirely in device kernels. Nothing in it is specific to
// momentum or to an octree: `MomentumOp` is any diagonal + face-CSR operator (with an optional
// second CSR), which is why voro's optimal-transport optimiser drives the same code. The names keep
// their AMR spelling — the AMR package and voro alias them, and a rename is not a structural move.
//
// The row arithmetic itself lives in face_csr.hpp, shared with the host serial oracle so the two
// cannot drift. A is generally NON-symmetric (row scaling, upwind advection), so the smoother is
// weighted Jacobi (parallel, deterministic: reads only the previous iterate) or the multicolour GS
// below, and the Krylov accelerator is BiCGStab (csr_bicgstab.hpp) rather than CG.
#ifndef PECLET_CORE_SOLVER_CSR_OPERATOR_HPP
#define PECLET_CORE_SOLVER_CSR_OPERATOR_HPP

#include <cstddef>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/coloring.hpp"
#include "peclet/core/solver/face_csr.hpp"

namespace peclet::core::solver {

/// Assembled momentum operator on the device: (A u)_i = diag_i u_i + Σ coef·u[nbr], with an
/// optional implicit-FOU advection part (rebuilt each step from the lagged velocity): a
/// per-cell outflow diagonal `advDiag` + per-face inflow coefficients over a second
/// (face-geometry) CSR. hasAdv=false ⇒ the pure cut-cell operator, bit-exact unchanged.
struct MomentumOp {
  View<double> diag;      ///< size n
  View<Index> faceStart;  ///< CSR row offsets, size n+1
  View<Index> faceNbr;    ///< neighbour leaf per off-diagonal, size nnz
  View<double> faceCoef;  ///< off-diagonal coefficient, size nnz
  Index n = 0;
  // Optional implicit-FOU advection (over the face-geometry CSR):
  bool hasAdv = false;
  View<double> advDiag;  ///< per-cell outflow (diagonal) advection weight, size n
  View<Index> advStart;  ///< face-geom CSR row offsets, size n+1
  View<Index> advNbr;    ///< face-geom neighbour per face, size nFaces
  View<double> advCoef;  ///< per-face inflow advection coefficient (0 on outflow/solid faces)
};

/// View the assembled momentum operator through the shared, backend-agnostic FaceCsrOpT, so the
/// device kernels and the host serial solver (cut_cell.hpp) run the *same* row arithmetic
/// (face_csr.hpp) and cannot drift. Non-const Views convert to their const accessor form
/// implicitly; the advection arrays are empty (and untouched) when hasAdv is false.
inline FaceCsrOpT<View<const double>, View<const Index>> momView(const MomentumOp& op) {
  FaceCsrOpT<View<const double>, View<const Index>> v;
  v.n = op.n;
  v.diag = op.diag;
  v.coef = op.faceCoef;
  v.start = op.faceStart;
  v.nbr = op.faceNbr;
  v.hasAdv = op.hasAdv;
  v.advDiag = op.advDiag;
  v.advCoef = op.advCoef;
  v.advStart = op.advStart;
  v.advNbr = op.advNbr;
  return v;
}

/// Au = A u (cut-cell operator + optional implicit-FOU advection).
inline void applyMom(const MomentumOp& op, View<const double> u, View<double> Au) {
  const auto A = momView(op);
  Kokkos::parallel_for(
      "amr::mom_apply", op.n, KOKKOS_LAMBDA(const Index i) { Au(i) = faceCsrApplyRow(A, i, u); });
}

/// res = b − A u.
inline void residualMom(const MomentumOp& op, View<const double> u, View<const double> b,
                        View<double> res) {
  const auto A = momView(op);
  Kokkos::parallel_for(
      "amr::mom_residual", op.n,
      KOKKOS_LAMBDA(const Index i) { res(i) = b(i) - faceCsrApplyRow(A, i, u); });
}

/// One weighted-Jacobi sweep of A u = b (in place). `tmp` is scratch (size n). Pass 1
/// reads only the previous iterate, pass 2 updates ⇒ order-independent / deterministic.
inline void jacobiMom(const MomentumOp& op, View<double> u, View<const double> b, View<double> tmp,
                      double omega) {
  const auto A = momView(op);
  Kokkos::parallel_for(
      "amr::mom_jacobi_compute", op.n, KOKKOS_LAMBDA(const Index i) {
        double off, d;
        faceCsrOffDiag(A, i, u, off, d);
        tmp(i) = (d != 0.0) ? (b(i) - off) / d : u(i);
      });
  Kokkos::parallel_for(
      "amr::mom_jacobi_update", op.n,
      KOKKOS_LAMBDA(const Index i) { u(i) = (1.0 - omega) * u(i) + omega * tmp(i); });
}

/// One **symmetric** multicolour Gauss–Seidel sweep of A u = b in place (momentum operator: diag +
/// face CSR + optional implicit-FOU advection): a forward pass over colours 0…C-1 followed by a
/// reverse pass C-1…0. Each colour is a parallel_for over its cells doing the GS point update
/// reading the current (already-updated) neighbours; cells of one colour share no edge ⇒ race-free.
///
/// The forward+reverse pairing makes the smoother **symmetric**, which matters when the MG V-cycle
/// is used as a *preconditioner* for BiCGStab (the momentum path): a forward-only GS V-cycle is a
/// non-symmetric, non-normal operator that breaks BiCGStab's bi-orthogonal recurrence on the larger
/// non-symmetric 64³ system (false convergence to NaN), whereas the symmetric (SGS) V-cycle keeps
/// it robust — the textbook remedy, and the behaviour flow gets from its RB-GS / MG-as-solver path.
inline void multicolorGSMom(const MomentumOp& op, View<double> u, View<const double> b,
                            const Coloring& col, double omega) {
  const auto A = momView(op);
  auto idx = col.idx;
  auto colorPass = [&](int c) {
    const Index a0 = col.hStart[static_cast<std::size_t>(c)];
    const Index a1 = col.hStart[static_cast<std::size_t>(c) + 1];
    Kokkos::parallel_for(
        "amr::gs_mom", Kokkos::RangePolicy<ExecSpace>(a0, a1), KOKKOS_LAMBDA(const Index k) {
          const Index i = idx(k);
          double off, d;
          faceCsrOffDiag(A, i, u, off, d);
          u(i) = faceCsrPointUpdate(b(i), off, d, u(i), omega);
        });
  };
  for (int c = 0; c < col.nColors; ++c)
    colorPass(c);  // forward
  for (int c = col.nColors - 2; c >= 0; --c)
    colorPass(c);  // reverse (last colour not repeated)
}

}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_CSR_OPERATOR_HPP

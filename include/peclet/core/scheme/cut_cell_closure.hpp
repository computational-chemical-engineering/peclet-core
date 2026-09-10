/// @file
/// @brief core — the shared cut-cell IBM boundary-distance closure polynomials (flow <-> amr).
///
/// QUALITY_PLAN §3.G.2 ("What IS shared"): the one-sided Dirichlet closure that a cut cell's
/// Shortley-Weller ghost value satisfies,
///
///     poly_D(th) * u_ghost = 2*u_bc + poly_Nc(th)*u_near + poly_N_nb(th)*u_far
///     th = sdf_near / (sdf_near - sdf_ghost), clamped to (0, 1]
///
/// (plus its sandwiched-axis and cell-average siblings) was written independently twice: in
/// `flow/src/cut_cell_ibm.hpp` (float, `KOKKOS_INLINE_FUNCTION`, the momentum IBM's Robust-Scaled
/// overlay) and in `amr/include/peclet/amr/cut_cell.hpp`'s `cc::` namespace (double,
/// `MORTON_HD`, the octree's Dirichlet operator). Both are pure functions of local crossing
/// fractions with no data layout in them, so this ONE templated copy replaces both: flow
/// instantiates it at `Real = float` (or `double` under `-DPECLET_FLOW_OPERATOR_DOUBLE`, G.6),
/// amr at `Real = double`. Verified identical between the two pre-lift copies (same formulas,
/// same literal constants, same operation order) before this header existed; a unit test
/// (`tests/test_cut_cell_closure.cpp`) pins every polynomial against the literal expressions at a
/// few theta values.
///
/// amr's `cc::` namespace used only the point-value (SCHEME 0) subset — `poly_D`, `poly_N_nb`,
/// `poly_Nc`, `poly_Nbc`, `poly_D_sandwich`, `poly_N_c_sandwich`, `poly_Nbc_pp_sw`,
/// `poly_Nbc_mp_sw` — plus its own `poly_abs` (a host/device `std::fabs` substitute, not part of
/// the closure family, so it is NOT lifted here; amr keeps it local). This header also carries
/// flow's cell-average (SCHEME 1) siblings (`poly_D_avg`, `poly_Nnb_avg`, `poly_Nc_avg`,
/// `poly_Nbc_avg`, `poly_D_sandwich_avg`, `poly_N_c_sandwich_avg`, `poly_Nbc_pp_sw_avg`,
/// `poly_Nbc_mp_sw_avg`) and the Navier-slip generalization (`poly_D_slip`, `poly_N_nb_slip`,
/// `poly_Nc_slip`, WO-V6b) that amr does not use but flow does -- one file, every scheme.
///
/// Host/device: `PECLET_CORE_CC_HD` is `KOKKOS_INLINE_FUNCTION` when Kokkos_Core.hpp has been
/// included BEFORE this header, plain `inline` otherwise (host-only oracle builds) -- the same
/// convention `scheme/ghost_closure.hpp` uses.
#ifndef PECLET_CORE_SCHEME_CUT_CELL_CLOSURE_HPP
#define PECLET_CORE_SCHEME_CUT_CELL_CLOSURE_HPP

#ifdef KOKKOS_INLINE_FUNCTION
#define PECLET_CORE_CC_HD KOKKOS_INLINE_FUNCTION
#else
#define PECLET_CORE_CC_HD inline
#endif

namespace peclet::core::scheme {

// ---- boundary-distance polynomials, SCHEME 0 (point-value) ----
template <class Real>
PECLET_CORE_CC_HD Real poly_D(Real xi) {
  return xi * (Real(1) + xi);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_N_nb(Real xi) {
  return xi * (Real(1) - xi);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nc(Real xi) {
  return Real(2) * (xi * xi - Real(1));
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nbc(Real) {
  return Real(2);
}

// ---- SCHEME 1 (cell-average) siblings ----
template <class Real>
PECLET_CORE_CC_HD Real poly_D_avg(Real xi) {
  return xi * (Real(1) + xi) - Real(1) / Real(12);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nnb_avg(Real xi) {
  return xi * (Real(1) - xi) + Real(1) / Real(12);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nc_avg(Real xi) {
  return Real(2) * (xi * xi - Real(1)) - Real(1) / Real(6);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nbc_avg(Real) {
  return Real(2);
}

// ---- Navier-slip generalization of the SCHEME 0 polynomials (flow WO-V6b; amr does not use
// these -- point-shell wall models are a flow/dem convention). See flow/src/cut_cell_ibm.hpp for
// the derivation (P = theta + lam, Q = theta^2 + 2*lam*theta); lam = 0 reproduces poly_D /
// poly_N_nb / poly_Nc identically.
template <class Real>
PECLET_CORE_CC_HD Real poly_D_slip(Real xi, Real lam) {
  return poly_D(xi) + lam * (Real(1) + Real(2) * xi);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_N_nb_slip(Real xi, Real lam) {
  return poly_N_nb(xi) + lam * (Real(1) - Real(2) * xi);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nc_slip(Real xi, Real lam) {
  return poly_Nc(xi) + Real(4) * lam * xi;
}

// ---- sandwiched axis (both neighbours solid = a one-cell fluid gap), SCHEME 0 then SCHEME 1 ----
template <class Real>
PECLET_CORE_CC_HD Real poly_D_sandwich(Real xi_m, Real xi_p) {
  return xi_m * xi_p;
}
template <class Real>
PECLET_CORE_CC_HD Real poly_N_c_sandwich(Real xi_m, Real xi_p) {
  return (xi_m + Real(1)) * (xi_p - Real(1));
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nbc_pp_sw(Real xi_m, Real xi_p) {
  return (xi_m / (xi_m + xi_p)) * (Real(1) + xi_m);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nbc_mp_sw(Real xi_m, Real xi_p) {
  return (xi_p / (xi_m + xi_p)) * (Real(1) - xi_p);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_D_sandwich_avg(Real xi_m, Real xi_p) {
  return xi_m * xi_p - Real(1) / Real(12);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_N_c_sandwich_avg(Real xi_m, Real xi_p) {
  return (xi_m + Real(1)) * (xi_p - Real(1)) - Real(1) / Real(12);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nbc_pp_sw_avg(Real xi_m, Real xi_p) {
  return (xi_m / (xi_m + xi_p)) * (Real(1) + xi_m) - Real(1) / Real(12);
}
template <class Real>
PECLET_CORE_CC_HD Real poly_Nbc_mp_sw_avg(Real xi_m, Real xi_p) {
  return (xi_p / (xi_m + xi_p)) * (Real(1) - xi_p) + Real(1) / Real(12);
}

}  // namespace peclet::core::scheme

#endif  // PECLET_CORE_SCHEME_CUT_CELL_CLOSURE_HPP

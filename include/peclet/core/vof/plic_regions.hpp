/// @file
/// @brief core — PLIC area fractions of the planar MAC stress regions: a unit face (box × box), a
/// shear-link kernel (hat × box) and a cell midsection, all on the fluid side of a PLIC plane.
///
/// Design: flow `doc/vof_sharp_viscous_stress.md` §5.2 (WO-1). The sharp conservative viscous
/// stress at a VoF interface needs, for every stress sample in the interface band, the colour-1
/// fraction of a planar region taken from the owner cell's PLIC plane: the arithmetic face mean μ̄
/// of a MAC face (box × box), the series resistance R̄ of a shear link (a hat of half-width one cell
/// times a unit box) and the c-midsection of a cell (box × box through the cell centre).
///
/// **Container-free, like `plic.hpp`.** Scalars and the 3-vector `m` only; no View, no grid
/// indexing. Each fraction is a thin composition of `plicVolume` and is exact for a plane:
///
/// - A region lying in the plane `x_p = xiP`, spanning `x_q ∈ [q0, q0+1]` and `x_b ∈ [b0, b0+1]`,
///   is the unit square of `(s, t)` with `x_q = q0 + s`, `x_b = b0 + t`. On it the plane is
///   `m_q s + m_b t < alpha − m_p xiP − m_q q0 − m_b b0`, i.e. `plicVolume(m_q, m_b, 0, ·)`.
/// - The hat kernel on `x_q ∈ [qc − 1, qc + 1]` (weight 1 − |x_q − qc|) is the convolution of two
///   unit boxes, `x_q = qc − 1 + s + u` with `s, u` uniform on [0, 1]; so the hat × box region is
///   the unit cube of `(s, u, t)` and the fraction is `plicVolume(m_q, m_q, m_b, ·)`.
///
/// Coordinates are the owner cell's LOCAL index coordinates (its unit cube [0,1]^3 at the low
/// corner); a region may extend beyond that cube, the plane is then extrapolated. Like every
/// volume-fraction operation in `plic.hpp`, these are metric-free (index space): see `VofMetric`.
/// The fluid (colour C = 1) side is `m . x < alpha`, the `plic.hpp` convention; `m` may have any
/// scale and zero components.
///
/// Axis convention: `q` and `b` are the two axes other than the plane axis `p`, taken in INCREASING
/// order for the box × box region (so `q0` belongs to the lower one); for the hat × box region `q`
/// is the given hat axis and `b` the remaining one.
#ifndef PECLET_CORE_VOF_PLIC_REGIONS_HPP
#define PECLET_CORE_VOF_PLIC_REGIONS_HPP

#include <Kokkos_Core.hpp>

#include "peclet/core/vof/plic.hpp"

namespace peclet::core::vof {

/// Colour-1 fraction of the unit square in the plane `x_axis = xiP` spanning
/// `x_q ∈ [q0, q0+1]`, `x_b ∈ [b0, b0+1]` (q < b the two other axes), box × box.
KOKKOS_INLINE_FUNCTION double plicFaceFraction(const double m[3], double alpha, int axis,
                                               double xiP, double q0, double b0) {
  const int q = (axis == 0) ? 1 : 0;
  const int b = (axis == 2) ? 1 : 2;
  return plicVolume(m[q], m[b], 0.0, alpha - m[axis] * xiP - m[q] * q0 - m[b] * b0);
}

/// Colour-1 fraction of the hat × box shear-link kernel in the plane `x_planeAxis = xiP`: a hat of
/// half-width 1 centred at `x_hatAxis = qc` (support [qc − 1, qc + 1]) times the unit box
/// `x_b ∈ [b0, b0+1]` on the remaining axis b. `hatAxis != planeAxis`.
KOKKOS_INLINE_FUNCTION double plicHatBoxFraction(const double m[3], double alpha, int planeAxis,
                                                 double xiP, int hatAxis, double qc, double b0) {
  const int b = 3 - planeAxis - hatAxis;
  const double mq = m[hatAxis];
  return plicVolume(mq, mq, m[b], alpha - m[planeAxis] * xiP - mq * (qc - 1.0) - m[b] * b0);
}

/// Colour-1 fraction of the cell's own c-midsection: the unit square `x_c = 1/2`,
/// `x_a, x_b ∈ [0, 1]` (the box × box face fraction at `xiP = 1/2`, `q0 = b0 = 0`).
KOKKOS_INLINE_FUNCTION double plicMidsectionFraction(const double m[3], double alpha, int c) {
  return plicFaceFraction(m, alpha, c, 0.5, 0.0, 0.0);
}

}  // namespace peclet::core::vof

#endif  // PECLET_CORE_VOF_PLIC_REGIONS_HPP

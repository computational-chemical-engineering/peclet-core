/// @file
/// @brief core — the probe-flux wall closure: probe distance, trilinear probe stencil, the
/// four-rung fallback ladder, and the per-facet wall / interface conductances.
///
/// Design: flow `doc/scalar_ibm_design.md` §1.4 (wall closure), §3 (probe distance D6, stencil,
/// ladder, precision D7). A wall facet with centroid x_w and unit normal n (into the phase being
/// probed) reads the field at the probe point p = x_w + s n by trilinear interpolation over the
/// 8 cells around p; the wall flux is then eliminated per facet as G (u_p − g) with the
/// conductance G of `wallConductance` / `interfaceConductance`.
///
/// Container-free: the caller's `Lookup` answers "is the cell at lattice offset (dx, dy, dz) an
/// unknown of this phase?" and "what is the phase-signed SDF at index-space point ξ?". Lattice
/// OFFSETS (relative to the facet's cell) are returned, never DOF indices — flow converts them to
/// linear block offsets, amr maps them to leaf DOFs (§7.1).
///
/// Ladder (§3.3), S = S(n) = ½ Σ_a |n_a| h_a (the cell's support half-width along n):
///
/// - R0:  s₀ = 1.1 S, trilinear (all 8); accept if V1 ∧ V2 ∧ V3.
/// - R1a: only when R0 failed V1 alone; s = 1.5 S, trilinear (all 8); accept if V1 ∧ V2 ∧ V3.
/// - R1b: gap estimate w = s₀ + φ(p₀); if w > 0.2 S, s = ½ w, trilinear with the cells that are
///   not unknowns dropped and the rest renormalized; accept if Σ kept weights ≥ 0.5 ∧ V3.
/// - R2:  s = 0.5 S, the cell itself with weight 1 (last resort, first order); always.
///
/// V1: every stencil cell with weight > 1e-12 is an unknown of the phase. V2 (clearance):
/// φ_phase(p) ≥ ½ s. V3: reach ≤ 2 on every axis — evaluated on all 8 stencil cells FIRST, so the
/// `Lookup` is never asked about a cell (or a φ sample) beyond the G = 2 halo. R1b needs φ(p₀),
/// so it is attempted only when R0's stencil passed V3.
///
/// Host/device: `PECLET_CORE_CC_HD` as in `scheme/cut_cell_closure.hpp`. Everything is double (D7).
#ifndef PECLET_CORE_SCHEME_PROBE_FLUX_HPP
#define PECLET_CORE_SCHEME_PROBE_FLUX_HPP

#include <cmath>

#include "peclet/core/scheme/cut_cell_closure.hpp"  // PECLET_CORE_CC_HD

namespace peclet::core::scheme {

/// Ladder rungs, in `ProbeResult::rung` (flow stores them as uint8).
inline constexpr int kProbeR0 = 0;
inline constexpr int kProbeR1a = 1;
inline constexpr int kProbeR1b = 2;
inline constexpr int kProbeR2 = 3;

/// Probe distance factors on S(n) (§3.1, §3.3): R0 = 1.1, R1a = 1.5, R2 = 0.5.
inline constexpr double kProbeSigma = 1.1;
inline constexpr double kProbeSigmaLong = 1.5;
inline constexpr double kProbeSigmaLast = 0.5;
/// V1 ignores stencil cells whose weight is at most this.
inline constexpr double kProbeWeightEps = 1e-12;
/// V2: φ_phase(p) ≥ kProbeClearance · s.
inline constexpr double kProbeClearance = 0.5;
/// V3: every stencil cell within ±kProbeReach cells on each axis (the G = 2 halo).
inline constexpr int kProbeReach = 2;
/// R1b is attempted only when the estimated gap w exceeds kProbeGapMin · S(n) ...
inline constexpr double kProbeGapMin = 0.2;
/// ... and accepted only when the kept weights sum to at least kProbeMinKeptWeight.
inline constexpr double kProbeMinKeptWeight = 0.5;

/// One facet's probe: the rung, the probe distance s, and n stencil cells (lattice offsets
/// relative to the facet's cell, weights summing to 1). R0 / R1a: n = 8 in trilinear corner order
/// (bit 0 = +x, bit 1 = +y, bit 2 = +z from the base cell); R1b: the kept cells, compacted in
/// that order; R2: n = 1, the cell itself.
struct ProbeResult {
  int rung;
  double s;
  int n;
  int off[8][3];
  double w[8];
};

/// S(n) = ½ Σ_a |n_a| h_a: the support half-width of a cell of size h along the unit normal n
/// (§3.1). A cell whose centre lies at signed distance ≥ −S(n) from a plane meets its positive
/// half-space.
PECLET_CORE_CC_HD double probeSupport(const double n[3], const double h[3]) {
  const double ax = n[0] < 0.0 ? -n[0] : n[0];
  const double ay = n[1] < 0.0 ? -n[1] : n[1];
  const double az = n[2] < 0.0 ? -n[2] : n[2];
  return 0.5 * (ax * h[0] + ay * h[1] + az * h[2]);
}

namespace detail {

PECLET_CORE_CC_HD double probeFloor(double x) {
#ifdef KOKKOS_INLINE_FUNCTION
  return Kokkos::floor(x);
#else
  return std::floor(x);
#endif
}

}  // namespace detail

/// Trilinear stencil of the index-space point ξ (cell centres at integer ξ, relative to the
/// facet's cell): base = floor(ξ), and w[k] for the cell base + (k & 1, (k >> 1) & 1, k >> 2),
/// w[k] = Π_a (bit_a ? f_a : 1 − f_a) with f = ξ − base. Reproduces linear fields exactly.
PECLET_CORE_CC_HD void trilinearStencil(const double xi[3], int base[3], double w[8]) {
  double f[3];
  for (int a = 0; a < 3; ++a) {
    const double b = detail::probeFloor(xi[a]);
    base[a] = static_cast<int>(b);
    f[a] = xi[a] - b;
  }
  for (int k = 0; k < 8; ++k) {
    const double wx = (k & 1) ? f[0] : 1.0 - f[0];
    const double wy = ((k >> 1) & 1) ? f[1] : 1.0 - f[1];
    const double wz = ((k >> 2) & 1) ? f[2] : 1.0 - f[2];
    w[k] = wx * wy * wz;
  }
}

namespace detail {

// The candidate probe at distance s: index-space point xi, its trilinear stencil, and V3.
PECLET_CORE_CC_HD bool probeCandidate(const double xw[3], const double n[3], const double h[3],
                                      double s, double xi[3], int base[3], double w[8]) {
  for (int a = 0; a < 3; ++a)
    xi[a] = (xw[a] + s * n[a]) / h[a];
  trilinearStencil(xi, base, w);
  for (int a = 0; a < 3; ++a)
    if (base[a] < -kProbeReach || base[a] + 1 > kProbeReach)
      return false;  // V3
  return true;
}

// V1: every cell with weight > kProbeWeightEps is an unknown of the phase.
template <class Lookup>
PECLET_CORE_CC_HD bool probeAllUnknown(const int base[3], const double w[8], const Lookup& L) {
  for (int k = 0; k < 8; ++k)
    if (w[k] > kProbeWeightEps &&
        !L.unknown(base[0] + (k & 1), base[1] + ((k >> 1) & 1), base[2] + ((k >> 2) & 1)))
      return false;
  return true;
}

PECLET_CORE_CC_HD void probeFillAll(ProbeResult& r, int rung, double s, const int base[3],
                                    const double w[8]) {
  r.rung = rung;
  r.s = s;
  r.n = 8;
  for (int k = 0; k < 8; ++k) {
    r.off[k][0] = base[0] + (k & 1);
    r.off[k][1] = base[1] + ((k >> 1) & 1);
    r.off[k][2] = base[2] + ((k >> 2) & 1);
    r.w[k] = w[k];
  }
}

}  // namespace detail

/// Build the probe of one facet by the ladder of §3.3. `xw` is the facet centroid relative to its
/// cell centre and `n` the unit normal INTO the probed phase (the solid side passes −n, the solid
/// flags and −φ through its own `Lookup`); both in internal lengths, h the cell size. `Lookup`
/// provides `bool unknown(int dx, int dy, int dz) const` (lattice offset from the facet's cell)
/// and `double phi(const double xi[3]) const` (the phase-signed SDF at the index-space point ξ
/// relative to the cell, in internal lengths).
template <class Lookup>
PECLET_CORE_CC_HD ProbeResult buildProbe(const double xw[3], const double n[3], const double h[3],
                                         const Lookup& L) {
  ProbeResult r;
  const double S = probeSupport(n, h);
  double xi[3], w[8];
  int base[3];

  // R0: s0 = 1.1 S, trilinear, V1 ∧ V2 ∧ V3.
  const double s0 = kProbeSigma * S;
  const bool reach0 = detail::probeCandidate(xw, n, h, s0, xi, base, w);
  double phi0 = 0.0;
  if (reach0) {
    phi0 = L.phi(xi);
    const bool v1 = detail::probeAllUnknown(base, w, L);
    const bool v2 = phi0 >= kProbeClearance * s0;
    if (v1 && v2) {
      detail::probeFillAll(r, kProbeR0, s0, base, w);
      return r;
    }
    // R1a: R0 failed V1 only -> a longer probe, s = 1.5 S.
    if (v2) {
      const double s1 = kProbeSigmaLong * S;
      if (detail::probeCandidate(xw, n, h, s1, xi, base, w) &&
          detail::probeAllUnknown(base, w, L) && L.phi(xi) >= kProbeClearance * s1) {
        detail::probeFillAll(r, kProbeR1a, s1, base, w);
        return r;
      }
    }
    // R1b: probe the estimated gap midpoint, s = ½ w with w = s0 + φ(p0), dropping the cells
    // that are not unknowns and renormalizing.
    const double gap = s0 + phi0;
    if (gap > kProbeGapMin * S) {
      const double s1 = 0.5 * gap;
      if (detail::probeCandidate(xw, n, h, s1, xi, base, w)) {
        int m = 0;
        double sum = 0.0;
        for (int k = 0; k < 8; ++k) {
          const int o0 = base[0] + (k & 1), o1 = base[1] + ((k >> 1) & 1), o2 = base[2] + (k >> 2);
          if (!L.unknown(o0, o1, o2))
            continue;
          r.off[m][0] = o0;
          r.off[m][1] = o1;
          r.off[m][2] = o2;
          r.w[m] = w[k];
          sum += w[k];
          ++m;
        }
        if (sum >= kProbeMinKeptWeight) {
          for (int k = 0; k < m; ++k)
            r.w[k] /= sum;
          r.rung = kProbeR1b;
          r.s = s1;
          r.n = m;
          return r;
        }
      }
    }
  }

  // R2 (last resort, first order): the cell itself at s = 0.5 S.
  r.rung = kProbeR2;
  r.s = kProbeSigmaLast * S;
  r.n = 1;
  r.off[0][0] = r.off[0][1] = r.off[0][2] = 0;
  r.w[0] = 1.0;
  return r;
}

/// Wall conductance of a Robin facet (§1.4): G = 1 / (s/L + 1/k), the flux out of the fluid per
/// area being G (u_p − g). k = 0 is Neumann (G = 0), k = +inf Dirichlet (G = L/s). s is the probe
/// distance and L the phase conductivity (Λ_f = D).
PECLET_CORE_CC_HD double wallConductance(double s, double L, double k) {
  constexpr double kDoubleMax = 1.7976931348623157e308;
  if (!(k > 0.0))
    return 0.0;
  if (k > kDoubleMax)
    return L / s;
  return 1.0 / (s / L + 1.0 / k);
}

/// Conjugate interface conductance (§1.4): the series resistance of the fluid probe leg, the
/// contact resistance and the solid probe leg, G_c = 1 / (s_f/Λ_f + R_c + s_s/Λ_s), so that the
/// flux from solid to fluid is G_c (u_ps − u_pf) in the ψ form.
PECLET_CORE_CC_HD double interfaceConductance(double sf, double Lf, double Rc, double ss,
                                              double Ls) {
  return 1.0 / (sf / Lf + Rc + ss / Ls);
}

}  // namespace peclet::core::scheme

#endif  // PECLET_CORE_SCHEME_PROBE_FLUX_HPP

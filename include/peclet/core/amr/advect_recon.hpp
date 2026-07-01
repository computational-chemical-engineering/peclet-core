// transport-core — shared, backend-agnostic high-order advection face reconstruction.
//
// The SOU / Koren-TVD reconstruction of the advected face value from the two upwind cells (upup, up)
// and the downwind cell (down) is the numerically delicate part of the advection scheme — exactly the
// kind of formula that must not drift between the serial host solver (AmrFlow::hoFace, flow.hpp) and
// the Kokkos device solver (deferredSou / advectExplicit, flow.hpp). This is that
// formula, written ONCE as a MORTON_HD function (host- and device-callable; empty MORTON_HD in the
// pure-C++ build). min/max/abs are expressed with branches rather than std::/Kokkos:: math so the body
// is identical on both backends (and bit-identical to the previous std::fmin/fmax form for the finite
// velocity/scalar data the solver produces).
#ifndef PECLET_CORE_AMR_ADVECT_RECON_HPP
#define PECLET_CORE_AMR_ADVECT_RECON_HPP

#if defined(PECLET_CORE_HAVE_MORTON)
#include <morton/morton.hpp>
#endif
#ifndef MORTON_HD
#define MORTON_HD
#endif

namespace peclet::core::amr {

/// High-order advected face value from the two upwind cells (`upup`, `up`) and the downwind cell
/// (`down`). `scheme` 0 = second-order upwind (SOU = 1.5·up − 0.5·upup); else Koren TVD limiter.
MORTON_HD inline double hoFaceValue(double upup, double up, double down, int scheme) {
  if (scheme == 0) return 1.5 * up - 0.5 * upup;  // SOU
  const double den = down - up;
  const double aden = (den < 0.0) ? -den : den;
  const double r = (aden < 1e-10) ? 0.0 : (up - upup) / den;
  // psi = max(0, min(2r, (1+2r)/3, 2)) — the Koren limiter.
  const double t = (1.0 + 2.0 * r) / 3.0;
  double m = (2.0 * r < t) ? (2.0 * r) : t;
  if (m > 2.0) m = 2.0;
  const double psi = (m > 0.0) ? m : 0.0;
  return up + 0.5 * psi * den;  // Koren TVD
}

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_AMR_ADVECT_RECON_HPP

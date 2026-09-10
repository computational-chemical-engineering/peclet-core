// peclet::core::scheme cut-cell closure polynomials (peclet/core/scheme/cut_cell_closure.hpp).
//
// QUALITY_PLAN §3.G.2 poly lift: the ONE templated copy that replaces flow's
// (cut_cell_ibm.hpp, float) and amr's (cut_cell.hpp cc::, double) independently-written closure
// polynomials. Pins every function against the literal formula (not a re-derivation -- the
// point is to catch a transcription slip in the lift itself) at several theta values, at BOTH
// Real = float and Real = double, and checks that Real = float reproduces the exact bits a plain
// float expression would give (the bit-exactness flow's default build depends on).
#include <cmath>
#include <cstdio>

#include "peclet/core/scheme/cut_cell_closure.hpp"
#include "test_util.hpp"

using namespace peclet::core::scheme;

template <class Real>
static void checkClose(Real a, Real b, Real tol, const char* what) {
  if (!(std::fabs(a - b) <= tol)) {
    std::fprintf(stderr, "CHECK_CLOSE failed (%s): %.17g vs %.17g\n", what, (double)a, (double)b);
    ++::peclet::core::test::g_failures;
  }
}

template <class Real>
static void checkPolynomials(const char* tag) {
  const Real thetas[] = {Real(0.1), Real(0.37), Real(0.5), Real(0.9), Real(1.0)};
  for (Real th : thetas) {
    checkClose<Real>(poly_D(th), th * (Real(1) + th), Real(0), tag);
    checkClose<Real>(poly_N_nb(th), th * (Real(1) - th), Real(0), tag);
    checkClose<Real>(poly_Nc(th), Real(2) * (th * th - Real(1)), Real(0), tag);
    checkClose<Real>(poly_Nbc(th), Real(2), Real(0), tag);

    checkClose<Real>(poly_D_avg(th), th * (Real(1) + th) - Real(1) / Real(12), Real(0), tag);
    checkClose<Real>(poly_Nnb_avg(th), th * (Real(1) - th) + Real(1) / Real(12), Real(0), tag);
    checkClose<Real>(poly_Nc_avg(th), Real(2) * (th * th - Real(1)) - Real(1) / Real(6), Real(0),
                     tag);
    checkClose<Real>(poly_Nbc_avg(th), Real(2), Real(0), tag);

    // Navier-slip: lam = 0 reproduces the no-slip polynomials exactly.
    checkClose<Real>(poly_D_slip(th, Real(0)), poly_D(th), Real(0), tag);
    checkClose<Real>(poly_N_nb_slip(th, Real(0)), poly_N_nb(th), Real(0), tag);
    checkClose<Real>(poly_Nc_slip(th, Real(0)), poly_Nc(th), Real(0), tag);
    // lam > 0 against the literal formula.
    const Real lam = Real(0.2);
    checkClose<Real>(poly_D_slip(th, lam), th * (Real(1) + th) + lam * (Real(1) + Real(2) * th),
                     Real(0), tag);
    checkClose<Real>(poly_N_nb_slip(th, lam), th * (Real(1) - th) + lam * (Real(1) - Real(2) * th),
                     Real(0), tag);
    checkClose<Real>(poly_Nc_slip(th, lam), Real(2) * (th * th - Real(1)) + Real(4) * lam * th,
                     Real(0), tag);
  }

  const Real pairs[][2] = {{Real(0.3), Real(0.7)}, {Real(0.05), Real(0.95)}, {Real(1), Real(1)}};
  for (auto& p : pairs) {
    const Real xm = p[0], xp = p[1];
    checkClose<Real>(poly_D_sandwich(xm, xp), xm * xp, Real(0), tag);
    checkClose<Real>(poly_N_c_sandwich(xm, xp), (xm + Real(1)) * (xp - Real(1)), Real(0), tag);
    checkClose<Real>(poly_Nbc_pp_sw(xm, xp), (xm / (xm + xp)) * (Real(1) + xm), Real(0), tag);
    checkClose<Real>(poly_Nbc_mp_sw(xm, xp), (xp / (xm + xp)) * (Real(1) - xp), Real(0), tag);

    checkClose<Real>(poly_D_sandwich_avg(xm, xp), xm * xp - Real(1) / Real(12), Real(0), tag);
    checkClose<Real>(poly_N_c_sandwich_avg(xm, xp),
                     (xm + Real(1)) * (xp - Real(1)) - Real(1) / Real(12), Real(0), tag);
    checkClose<Real>(poly_Nbc_pp_sw_avg(xm, xp),
                     (xm / (xm + xp)) * (Real(1) + xm) - Real(1) / Real(12), Real(0), tag);
    checkClose<Real>(poly_Nbc_mp_sw_avg(xm, xp),
                     (xp / (xm + xp)) * (Real(1) - xp) + Real(1) / Real(12), Real(0), tag);
  }
}

int main() {
  checkPolynomials<float>("float");
  checkPolynomials<double>("double");

  // amr's cc:: namespace (pre-lift) used ONLY these eight at double: cross-check against amr's own
  // formula spelling once more, verbatim, so a future edit to either side is caught here too.
  {
    const double xi = 0.42;
    if (poly_D<double>(xi) != xi * (1.0 + xi))
      ++::peclet::core::test::g_failures;
    if (poly_N_nb<double>(xi) != xi * (1.0 - xi))
      ++::peclet::core::test::g_failures;
    if (poly_Nc<double>(xi) != 2.0 * (xi * xi - 1.0))
      ++::peclet::core::test::g_failures;
    if (poly_Nbc<double>(xi) != 2.0)
      ++::peclet::core::test::g_failures;
    const double xm = 0.3, xp = 0.6;
    if (poly_D_sandwich<double>(xm, xp) != xm * xp)
      ++::peclet::core::test::g_failures;
    if (poly_N_c_sandwich<double>(xm, xp) != (xm + 1.0) * (xp - 1.0))
      ++::peclet::core::test::g_failures;
    if (poly_Nbc_pp_sw<double>(xm, xp) != (xm / (xm + xp)) * (1.0 + xm))
      ++::peclet::core::test::g_failures;
    if (poly_Nbc_mp_sw<double>(xm, xp) != (xp / (xm + xp)) * (1.0 - xp))
      ++::peclet::core::test::g_failures;
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

// Quadratic coarse-fine flux + refluxing (peclet::core::amr::AmrPoisson / AmrMultigrid):
// on a half-coarse/half-fine periodic mesh with a manufactured solution whose
// tangential gradient is non-zero at the 2:1 interface,
//   (1) the standard two-point C/F flux degrades the solution toward 1st order
//       near the interface, while the quadratic flux stays 2nd order (L-infinity
//       order across two refinements);
//   (2) the quadratic operator is still conservative (sum_i V_i (L_quad u)_i ~ 0),
//       i.e. the coarse face flux equals the summed fine sub-face fluxes (reflux).
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <cstdint>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

using BO = BlockOctree<2, 32>;
using Code = BO::Code;
using M = BO::M;

// Half-coarse (left) / half-fine (right) periodic mesh on [0,1)^2; finest cell
// width 1/(2*Nr). Returns the L-infinity solution error for the standard and the
// quadratic C/F operator, and (out) the quadratic conservation residual.
struct Out {
  double stdLinf, quadLinf, conserv;
};
Out solve(long Nr) {
  BO t(IVec<2>{Nr, Nr}, 1);  // root cells at level 1 (size 2 fine)
  const double h0 = 1.0 / static_cast<double>(Nr * 2);
  t.refineIf([&](Code c, unsigned L) {
    auto o = M::from_code(c).decode();
    double s = static_cast<double>(1 << L);
    return (static_cast<double>(o[0]) + 0.5 * s) * h0 > 0.5;  // refine right half
  });
  t.balance2to1();

  AmrMultigrid<2, 32> mg;
  mg.build(t, h0);
  const AmrPoisson<2, 32>& P = mg.op();
  const Index n = t.numLeaves();

  const double k = 2.0 * M_PI;
  auto ex = [&](Index i) {
    auto b = t.bounds(i);
    double s = static_cast<double>(1 << t.level(i));
    double cx = (static_cast<double>(b[0][0]) + 0.5 * s) * h0;
    double cy = (static_cast<double>(b[0][1]) + 0.5 * s) * h0;
    return std::cos(k * cx) * std::cos(k * cy);  // nonzero tangential grad at x=0.5
  };
  std::vector<double> uex(static_cast<std::size_t>(n)), rhs(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    uex[static_cast<std::size_t>(i)] = ex(i);
    rhs[static_cast<std::size_t>(i)] = -2.0 * k * k * uex[static_cast<std::size_t>(i)];
  }
  auto linf = [&](std::vector<double>& u) {
    double mu = 0, me = 0, vol = 0;
    for (Index i = 0; i < n; ++i) {
      mu += P.cellVolume(i) * u[static_cast<std::size_t>(i)];
      me += P.cellVolume(i) * uex[static_cast<std::size_t>(i)];
      vol += P.cellVolume(i);
    }
    mu /= vol;
    me /= vol;
    double e = 0;
    for (Index i = 0; i < n; ++i)
      e = std::max(e, std::fabs((u[static_cast<std::size_t>(i)] - mu) -
                                (uex[static_cast<std::size_t>(i)] - me)));
    return e;
  };

  std::vector<double> us(static_cast<std::size_t>(n), 0.0), res;
  for (int c = 0; c < 80; ++c) {
    mg.vcycle(0, us, rhs);
    if (P.residual(us, rhs, res) < 1e-12)
      break;
  }
  std::vector<double> uq(static_cast<std::size_t>(n), 0.0);
  mg.solveQuad(uq, rhs, 300, 1);

  // conservation of the quadratic operator on an arbitrary field.
  std::vector<double> rnd(static_cast<std::size_t>(n)), lq;
  std::uint64_t s = 7;
  for (auto& x : rnd) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    x = static_cast<double>((s >> 11) & 0xFFFFF) / static_cast<double>(0x100000) - 0.5;
  }
  P.applyLaplacianQuad(rnd, lq);
  double integral = 0, scale = 0;
  for (Index i = 0; i < n; ++i) {
    integral += P.cellVolume(i) * lq[static_cast<std::size_t>(i)];
    scale += P.cellVolume(i) * std::fabs(lq[static_cast<std::size_t>(i)]);
  }
  return {linf(us), linf(uq), std::fabs(integral) / (scale + 1e-30)};
}

void run() {
  Out b = solve(16);
  Out c = solve(32);

  // (1) order: quadratic ~2nd order (ratio ~4); standard degraded (< 2.6).
  double quadOrder = b.quadLinf / c.quadLinf;
  double stdOrder = b.stdLinf / c.stdLinf;
  PECLET_CORE_CHECK(quadOrder > 3.0);         // ~2nd order in L-infinity
  PECLET_CORE_CHECK(stdOrder < 2.6);          // standard two-point degraded near C/F
  PECLET_CORE_CHECK(c.quadLinf < c.stdLinf);  // quadratic is more accurate at the finest

  // (2) conservation / refluxing.
  PECLET_CORE_CHECK(b.conserv < 1e-9);
  PECLET_CORE_CHECK(c.conserv < 1e-9);
}

}  // namespace

int main() {
  run();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping quadratic C/F test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

// Cut-cell openness folded into the operator + multigrid coarsening, together with
// the quadratic coarse-fine flux (peclet::core::amr::AmrPoisson / AmrMultigrid):
//   (1) regression — openness ≡ 1 reproduces the no-openness quadratic solve
//       bit-for-bit (the openness path is a pure generalisation);
//   (2) conservation — the openness-weighted quadratic operator still conserves
//       (sum_i V_i (L u)_i ~ 0): each face's flux is openness-weighted identically
//       from both sides, including across 2:1 interfaces;
//   (3) consistency — a variable-openness Poisson solve converges with the
//       area-averaged (coarsened) openness on every multigrid level, i.e. the
//       rediscretized coarse operators are consistent with the fine one.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
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

// Half-coarse / half-fine periodic mesh on [0,1)^2 (exercises 2:1 interfaces).
BO makeMesh(long Nr, double& h0) {
  BO t(IVec<2>{Nr, Nr}, 1);
  h0 = 1.0 / static_cast<double>(Nr * 2);
  t.refineIf([&](Code c, unsigned L) {
    auto o = M::from_code(c).decode();
    double s = static_cast<double>(1 << L);
    return (static_cast<double>(o[0]) + 0.5 * s) * h0 > 0.5;
  });
  t.balance2to1();
  return t;
}

// Mean-zero RHS so the periodic system is consistent.
std::vector<double> makeRhs(const BO& t, double h0, const AmrPoisson<2, 32>& P) {
  const Index n = t.numLeaves();
  const double k = 2.0 * M_PI;
  std::vector<double> rhs(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    auto b = t.bounds(i);
    double s = static_cast<double>(1 << t.level(i));
    double cx = (static_cast<double>(b[0][0]) + 0.5 * s) * h0;
    double cy = (static_cast<double>(b[0][1]) + 0.5 * s) * h0;
    rhs[static_cast<std::size_t>(i)] = std::sin(k * cx) * std::sin(k * cy);
  }
  double m = 0, vol = 0;
  for (Index i = 0; i < n; ++i) {
    m += P.cellVolume(i) * rhs[static_cast<std::size_t>(i)];
    vol += P.cellVolume(i);
  }
  m /= vol;
  for (Index i = 0; i < n; ++i) rhs[static_cast<std::size_t>(i)] -= m;
  return rhs;
}

void run() {
  double h0 = 0;
  BO t = makeMesh(16, h0);
  const Index n = t.numLeaves();

  // (1) openness ≡ 1 == no-openness, bit-for-bit.
  {
    AmrMultigrid<2, 32> mg0, mg1;
    mg0.build(t, h0);
    mg1.build(t, h0);
    mg1.setOpenness([](const Vec<2>&, int) { return 1.0; });
    std::vector<double> rhs = makeRhs(t, h0, mg0.op());
    std::vector<double> u0(static_cast<std::size_t>(n), 0.0), u1(static_cast<std::size_t>(n), 0.0);
    mg0.solveQuad(u0, rhs, 80, 1);
    mg1.solveQuad(u1, rhs, 80, 1);
    double dmax = 0;
    for (Index i = 0; i < n; ++i)
      dmax = std::max(dmax, std::fabs(u0[static_cast<std::size_t>(i)] - u1[static_cast<std::size_t>(i)]));
    PECLET_CORE_CHECK(dmax < 1e-12);
  }

  // A smooth, strictly-positive openness field (variable-coefficient Poisson).
  auto openFn = [](const Vec<2>& fc, int) {
    return 0.6 + 0.3 * std::sin(2.0 * M_PI * fc[0]) * std::cos(2.0 * M_PI * fc[1]);
  };

  // (2) conservation of the openness-weighted quadratic operator.
  {
    AmrMultigrid<2, 32> mg;
    mg.build(t, h0);
    mg.setOpenness(openFn);
    const AmrPoisson<2, 32>& P = mg.op();
    std::vector<double> u(static_cast<std::size_t>(n)), lq;
    for (Index i = 0; i < n; ++i) u[static_cast<std::size_t>(i)] = std::sin(0.1 * i) + 0.3 * std::cos(0.07 * i);
    P.applyLaplacianQuad(u, lq);
    double integ = 0, scale = 0;
    for (Index i = 0; i < n; ++i) {
      integ += P.cellVolume(i) * lq[static_cast<std::size_t>(i)];
      scale += P.cellVolume(i) * std::fabs(lq[static_cast<std::size_t>(i)]);
    }
    PECLET_CORE_CHECK(std::fabs(integ) < 1e-9 * (scale + 1e-30));
  }

  // (3) variable-openness solve converges with coarsened openness on every level.
  {
    AmrMultigrid<2, 32> mg;
    mg.build(t, h0);
    mg.setOpenness(openFn);
    const AmrPoisson<2, 32>& P = mg.op();
    std::vector<double> rhs = makeRhs(t, h0, P);
    std::vector<double> u(static_cast<std::size_t>(n), 0.0), res;
    double r0 = P.residualQuad(u, rhs, res);
    double r = mg.solveQuad(u, rhs, 60, 1);
    PECLET_CORE_CHECK(r0 / r > 1e8);
  }
}

}  // namespace

int main() {
  run();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping AMR openness test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

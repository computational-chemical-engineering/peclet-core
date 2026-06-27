// ABC/Basilisk divergence-free FACE field on the host oracle AmrFlow. The collocated projection only
// makes the *cell* field approximately divergence-free (O(h²)); the face field uf_f = ½(u_i+u_j) −
// (φ₊−φ₋)/d is divergence-free to the pressure-solve residual, because L = D·G_face on the same
// (sub)faces ⇒ D(uf) = D u* − Lφ. This must hold across 2:1 interfaces too (the coarse cell sums its
// fine sub-faces). Guarded by TPX_HAVE_MORTON.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/flow_oracle.hpp"
#include "tpx/amr/refine.hpp"
#include "tpx/common/types.hpp"

using namespace tpx;
using namespace tpx::amr;
using BO = BlockOctree<3, 21>;
using Code = BO::Code;

namespace {

// Run a Stokes sphere to steady, return (divNorm cell, divNorm face).
std::pair<double, double> run(BO& t, double R, Vec<3> c, int presIters) {
  const double mu = 0.1, f = 1e-3, dt = 60.0;
  oracle::AmrFlow<21> fl;
  fl.init(t, 1.0, Vec<3>{0, 0, 0});
  fl.setDensity(1.0); fl.setViscosity(mu); fl.setDt(dt); fl.setBodyForce(f, 0, 0); fl.setAdvection(false);
  fl.setSolid([&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - R;
  });
  // The face field's divergence-free property is algebraic (div(uf) = div(u*) − Lφ), so it holds at
  // any state — a handful of steps is enough to exercise it without running to full steady state.
  for (int it = 0; it < 25; ++it) fl.step(60, presIters, 2);
  return {fl.divNormL2(fl.velocityRef()), fl.divNormFace()};
}

void run_test() {
  // (1) uniform finest N=16: the face field is far cleaner than the cell field, and tightening the
  //     pressure solve shrinks it (it is the solve residual, not a fixed O(h²) error).
  const unsigned L = 4; const long N = 1L << L;
  const double R = std::pow(0.125 * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * N, cc = N / 2.0;
  BO t6(IVec<3>{1, 1, 1}, L); for (unsigned k = 0; k < L; ++k) t6.refineIf([](Code, unsigned) { return true; });
  auto [dCell6, dFace6] = run(t6, R, Vec<3>{cc, cc, cc}, 6);
  BO t30(IVec<3>{1, 1, 1}, L); for (unsigned k = 0; k < L; ++k) t30.refineIf([](Code, unsigned) { return true; });
  auto [dCell30, dFace30] = run(t30, R, Vec<3>{cc, cc, cc}, 30);
  TPX_CHECK(dFace6 < 0.05 * dCell6);     // face field ≥20× more divergence-free than the cell field
  TPX_CHECK(dFace30 < 0.05 * dCell30);   // ditto at the tighter solve
  TPX_CHECK(dFace30 < dFace6);            // tightening the pressure solve shrinks the face divergence
  //         (the cell-field divergence is the fixed O(h²) approximate-projection error, ~unchanged)

  // (2) graded 2:1 grid: the face field stays divergence-free across the coarse–fine interfaces, where
  //     the cell field's divergence is actually larger.
  const unsigned lmax = 4; const long Nf = 1L << lmax;
  BO tg(IVec<3>{2, 2, 2}, lmax);
  AmrGeometry<3> geo; geo.h0 = 1.0;
  const double Rg = std::pow(0.125 * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * (2 * Nf), cg = (2.0 * Nf) / 2.0;
  refineToSdf(tg, geo, [&](const Vec<3>& p) {
    double dx = p[0] - cg, dy = p[1] - cg, dz = p[2] - cg;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - Rg;
  }, /*target_level=*/0, /*band=*/3.0, /*balance=*/true);
  TPX_CHECK(tg.isBalanced());
  auto [dCellG, dFaceG] = run(tg, Rg, Vec<3>{cg, cg, cg}, 30);
  TPX_CHECK(dFaceG < 0.01 * dCellG);     // across 2:1: face field ≥100× cleaner than the cell field
}

}  // namespace

int main() {
  run_test();
  if (tpx::test::g_failures == 0) std::printf("OK\n");
  return tpx::test::g_failures == 0 ? 0 : 1;
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping face-field test\n");
  return 0;
}
#endif

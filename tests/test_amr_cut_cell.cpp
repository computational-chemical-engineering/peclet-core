// Robust-Scaled cut-cell Dirichlet operator (tpx::amr::AmrCutCell) — the port of
// sdflow's ξ-polynomial sub-cell BC onto the octree:
//   (1) 2nd-order accuracy on an embedded-Dirichlet problem with an exact
//       solution: fluid = inside a sphere, u = 0 on the surface, u = R^2 - r^2
//       (so lap u = -6); the L2 error quarters when the grid halves;
//   (2) the cell volume fraction κ integrates to the sphere volume (4/3 π R^3);
//   (3) solid cells are held at the wall value u_bc.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/cut_cell.hpp"
#include "tpx/common/types.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

constexpr double kLbox = 1.0;  // domain [-1,1]^3
constexpr double kR = 0.6;     // sphere radius

BO uniformFine(unsigned L) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k) t.refineIf([](Code, unsigned) { return true; });
  return t;
}

struct Result {
  double err;
  double fluidVol;
  long nfluid;
};

Result solveSphere(unsigned L) {
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 2.0 * kLbox / static_cast<double>(N);
  AmrCutCell<21> cc;
  cc.init(t, h0, Vec<3>{-kLbox, -kLbox, -kLbox});
  auto sdf = [&](const Vec<3>& p) {
    return kR - std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  };
  cc.build(sdf, /*idiag=*/0.0, /*beta=*/1.0, /*nsub=*/4);

  const Index n = t.numLeaves();
  std::vector<double> src(static_cast<std::size_t>(n), 0.0), uex(static_cast<std::size_t>(n), 0.0);
  auto center = [&](Index i, int d) {
    auto b = t.bounds(i);
    double s = static_cast<double>(1 << t.level(i));
    return -kLbox + (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
  };
  for (Index i = 0; i < n; ++i) {
    if (!cc.isFluid(i)) continue;
    double r2 = 0;
    for (int d = 0; d < 3; ++d) r2 += center(i, d) * center(i, d);
    uex[static_cast<std::size_t>(i)] = kR * kR - r2;
    src[static_cast<std::size_t>(i)] = 6.0 * h0 * h0;  // A u = -h^2 f, f = lap u = -6
  }
  std::vector<double> b = cc.makeRhs(src, /*u_bc=*/0.0);
  std::vector<double> u(static_cast<std::size_t>(n), 0.0), res;
  double r0 = cc.residual(u, b, res), r = r0;
  for (int it = 0; it < 200000; ++it) {
    cc.gaussSeidel(u, b, 20);
    r = cc.residual(u, b, res);
    if (r < r0 * 1e-9) break;
  }

  double e = 0, vol = 0;
  long nf = 0;
  for (Index i = 0; i < n; ++i) {
    double V = h0 * h0 * h0;  // uniform
    vol += cc.kappa(i) * V;
    if (cc.isFluid(i)) {
      double d = u[static_cast<std::size_t>(i)] - uex[static_cast<std::size_t>(i)];
      e += d * d;
      ++nf;
    }
  }
  // a solid cell is held at u_bc (= 0)
  bool solidHeld = true;
  for (Index i = 0; i < n; ++i)
    if (!cc.isFluid(i) && std::fabs(u[static_cast<std::size_t>(i)]) > 1e-9) solidHeld = false;
  TPX_CHECK(solidHeld);

  return {std::sqrt(e / nf), vol, nf};
}

void run() {
  Result a = solveSphere(4);  // 16^3
  Result b = solveSphere(5);  // 32^3

  // (1) 2nd-order convergence.
  double order = a.err / b.err;
  TPX_CHECK(order > 3.3);

  // (2) κ integrates to the sphere volume.
  const double exactVol = 4.0 / 3.0 * M_PI * kR * kR * kR;
  TPX_CHECK(std::fabs(b.fluidVol - exactVol) < 0.03 * exactVol);
  TPX_CHECK(b.nfluid > a.nfluid);
}

}  // namespace

int main() {
  run();
  TPX_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping cut-cell test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

// Stokes drag of a simple-cubic sphere array vs Zick & Homsy (1982) — the same
// external ground truth sdflow validates against (so matching Z&H == matching
// sdflow). Replicates sdflow's metric exactly (validate_zick_homsy_sdflow.py):
//   K = f N^3 / (6 pi mu R U_sup),  U_sup = mean(u_x) over the whole cell.
//
// FINDING (documented in docs/AMR.md + the amr-octree memory): the collocated AMR
// Stokes solver CONVERGES to the Z&H drag but at ~1st order — the cut-cell
// *diffusion* is 2nd order (test_amr_cut_cell), but the α-weighted ("porosity")
// cut-cell *pressure* operator here is only 1st order at the boundary (unlike
// sdflow's aperture/centroid buildCutcellOp). So this test asserts the CONVERGENCE
// TREND toward Z&H (error shrinks as the grid refines, correct sign, right
// ballpark) — not a tight match. A 2nd-order match needs the proper cut-cell
// pressure operator (and, for graded meshes, a C/F-consistent momentum operator).
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/flow.hpp"
#include "tpx/common/types.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

BO uniformFine(unsigned L) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k) t.refineIf([](Code, unsigned) { return true; });
  return t;
}

// SC sphere drag factor K for solid fraction phi on an N^3 grid (dx=1, sdflow's setup).
double dragK(unsigned L, double phi) {
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double R = std::pow(phi * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * static_cast<double>(N);
  const double mu = 0.1, f = 1e-3, dt = 60.0, c = N / 2.0;
  AmrFlow<21> fl;
  fl.init(t, 1.0, Vec<3>{0, 0, 0});  // grid units
  fl.setDensity(1.0);
  fl.setViscosity(mu);
  fl.setDt(dt);
  fl.setBodyForce(f, 0, 0);
  fl.setAdvection(false);  // Stokes
  fl.setSolid([&](const Vec<3>& p) {
    double dx = p[0] - c, dy = p[1] - c, dz = p[2] - c;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - R;  // <0 inside sphere (solid)
  });
  const Index n = t.numLeaves();
  auto mean = [&]() {
    double s = 0;
    const auto& u = fl.velocity(0);
    for (Index i = 0; i < n; ++i) s += u[static_cast<std::size_t>(i)];
    return s / n;
  };
  double prev = 0;
  for (int it = 0; it < 120; ++it) {
    fl.step(/*momSweeps=*/150, /*presIters=*/6, 2);
    if (it % 5 == 4) {
      double m = mean();
      if (it > 10 && std::fabs(m - prev) < 1e-6 * (std::fabs(m) + 1e-30)) break;
      prev = m;
    }
  }
  double m = mean();
  return f * N * N * N / (6.0 * M_PI * mu * R * m);
}

void run() {
  const double phi = 0.125;
  const double kZH = 4.292;  // Zick & Homsy (1982), SC, phi=0.125
  double k8 = dragK(3, phi);   // N=8
  double k16 = dragK(4, phi);  // N=16
  double e8 = std::fabs(k8 - kZH) / kZH;
  double e16 = std::fabs(k16 - kZH) / kZH;

  TPX_CHECK(k8 > 0 && k16 > 0);
  TPX_CHECK(k8 < kZH && k16 < kZH);  // undershoots (consistent sign)
  TPX_CHECK(e16 < e8);               // converges toward Z&H as the grid refines
  TPX_CHECK(e16 < 0.30);             // right ballpark at N=16 (~ -22%)
}

}  // namespace

int main() {
  run();
  TPX_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping Stokes drag test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

// Stokes drag of a simple-cubic sphere array vs Zick & Homsy (1982) — the same
// external ground truth sdflow validates against (matching Z&H == matching sdflow).
// Replicates sdflow's metric exactly (validate_zick_homsy_sdflow.py):
//   K = f N^3 / (6 pi mu R U_sup),  U_sup = mean(u_x) over the whole cell.
//
// With the incremental ROTATIONAL projection (p += (rho/dt)phi - mu div(u*)) and
// sdflow's gradient-normalised openness, the collocated AMR Stokes solver matches
// Z&H to ~1% across resolutions and solid fractions (dt-independent):
//   phi=0.125: N=8 -0.8%, N=16 +0.4%, N=32 +0.3%;  phi=0.064 N=16 -0.7%;
//   phi=0.216 N=16 -0.03%.   (An earlier non-incremental Chorin projection gave an
//   O(dt) splitting error -> ~ -11% at N=32; the rotational term fixed it.)
// This test runs the cheapest resolved point and asserts a tight match.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/flow.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/amr/refine.hpp"
#include "tpx/common/types.hpp"
#include "tpx/geom/sdf.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

// SC sphere drag factor K for solid fraction phi on an N^3 grid (dx=1, sdflow's setup).
double dragK(unsigned L, double phi) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k) t.refineIf([](Code, unsigned) { return true; });
  const long N = 1L << L;
  const double R = std::pow(phi * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * static_cast<double>(N);
  const double mu = 0.1, f = 1e-3, dt = 60.0, c = N / 2.0;
  oracle::AmrFlow<21> fl;
  fl.init(t, 1.0, Vec<3>{0, 0, 0});  // grid units (dx=1)
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
  for (int it = 0; it < 100; ++it) fl.step(/*momSweeps=*/120, /*presIters=*/6, 2);  // -> steady
  double s = 0;
  const auto& u = fl.velocity(0);
  for (Index i = 0; i < n; ++i) s += u[static_cast<std::size_t>(i)];
  double umean = s / n;
  return f * N * N * N / (6.0 * M_PI * mu * R * umean);
}

void run() {
  const double phi = 0.125;
  const double kZH = 4.292;  // Zick & Homsy (1982), SC, phi=0.125
  double k = dragK(3, phi);  // N=8 (cheapest); finer N tightens further (see header)
  double err = std::fabs(k - kZH) / kZH;
  TPX_CHECK(k > 0);
  TPX_CHECK(err < 0.03);  // tight match to Z&H (== sdflow); N=8 is ~ -0.8%

  // --- GRADED mesh: sphere band refined to the finest level, far field coarse ---
  // With the C/F-consistent momentum diffusion + FV divergence/ABC gradient, the
  // graded flow is STABLE (it diverged before the C/F fix) and matches Z&H within a
  // few percent at a fraction of the uniform cell count, converging as the band
  // widens. (finest 16, band 2.5 -> ~82% cells, ~ +2%.)
  const long brick = 8;
  const unsigned lmax = 1;
  const long Nf = brick * (1L << lmax);  // finest = 16
  const double R = std::pow(phi * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * static_cast<double>(Nf);
  const double mu = 0.1, f = 1e-3, c = Nf / 2.0;
  BO t(IVec<3>{brick, brick, brick}, lmax);
  AmrGeometry<3> geo;
  geo.h0 = 1.0;
  tpx::geom::Sphere sph{{c, c, c}, R};
  refineToSdf(t, geo, [&](const Vec<3>& p) { return -sph.eval(p); }, /*target*/ 0, /*band*/ 2.5, true);
  TPX_CHECK(t.numLeaves() < Nf * Nf * Nf);  // genuinely coarsened (graded)

  oracle::AmrFlow<21> fl;
  fl.init(t, 1.0, Vec<3>{0, 0, 0});
  fl.setDensity(1.0);
  fl.setViscosity(mu);
  fl.setDt(60.0);
  fl.setBodyForce(f, 0, 0);
  fl.setAdvection(false);
  fl.setSolid([&](const Vec<3>& p) {
    double dx = p[0] - c, dy = p[1] - c, dz = p[2] - c;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - R;
  });
  const long nuni = Nf * Nf * Nf;
  for (int it = 0; it < 45; ++it) fl.step(70, 5, 2);  // -> near steady (finest 16)
  double usup = 0;
  const auto& u = fl.velocity(0);
  for (Index i = 0; i < t.numLeaves(); ++i) {
    double w = static_cast<double>(1L << t.level(i));
    usup += u[static_cast<std::size_t>(i)] * w * w * w;  // volume-weighted over the cell
  }
  usup /= nuni;
  TPX_CHECK(std::isfinite(usup) && std::fabs(usup) < 1.0);  // STABLE (no blow-up)
  double kg = f * nuni / (6.0 * M_PI * mu * R * usup);
  TPX_CHECK(std::fabs(kg - kZH) / kZH < 0.10);  // graded drag within ~10% of Z&H
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

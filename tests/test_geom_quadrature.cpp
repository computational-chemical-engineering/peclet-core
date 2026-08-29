// Layer 2 gate: IMPLICIT QUADRATURE apertures (geom/quadrature.hpp) against the closed form.
//
// The reference is the analytic disk-rectangle overlap that flow/scripts/exact_apertures_spheres.py
// uses (validated there to 4e-16): a sphere cut by a coordinate plane is a disk, so the fluid area
// of a grid face is (face area) - (disk-rectangle overlap), in closed form.
//
// Three claims:
//   1. ACCURACY on a single sphere -- max and mean error over every cut face of a bed, against the
//      closed form, with the SAMPLED-SDF linear estimator alongside so the improvement is visible
//      rather than asserted.
//   2. WHERE THE ACCURACY COMES FROM -- the same faces with and without splitting the outer Gauss
//      rule at the points where the interface leaves the face. Without the split the error sits
//      near 1e-3 AT EVERY RESOLUTION: an aperture is a FRACTION, so a kink in a fixed normalized
//      integrand costs a fixed amount and refining h does not help. With it, machine precision.
//   3. HOW IT CONVERGES -- in the NODE COUNT, not in h.
//   4. ROBUSTNESS -- results stay in [0,1] and stay consistent where the height-graph premise
//      fails (a torus: a line through the hole crosses four times), and no order is claimed there.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "peclet/core/geom/primitives.hpp"
#include "peclet/core/geom/quadrature.hpp"
#include "peclet/core/geom/scene_builder.hpp"
#include "peclet/core/geom/scene_query.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;

// --- closed form: area of {u < x, v < y, u^2 + v^2 < r^2} for a disk at the origin -------------
static double G(double t, double r) {
  t = std::fmax(-r, std::fmin(r, t));
  return 0.5 * (t * std::sqrt(std::fmax(r * r - t * t, 0.0)) + r * r * std::asin(t / r));
}

static double Fquad(double x, double y, double r) {
  if (y <= -r)
    return 0.0;
  if (y >= r)
    return 2.0 * (G(x, r) - G(-r, r));
  const double yc = std::fmax(-r, std::fmin(r, y));
  const double xc = std::fmax(-r, std::fmin(r, x));
  const double us = std::sqrt(std::fmax(r * r - yc * yc, 0.0));
  const double aM = std::fmax(-r, -us), bM = std::fmin(xc, us);
  double F = bM > aM ? yc * (bM - aM) + (G(bM, r) - G(aM, r)) : 0.0;
  if (y >= 0.0) {
    const double a1 = -r, b1 = std::fmin(xc, -us);
    if (b1 > a1)
      F += 2.0 * (G(b1, r) - G(a1, r));
    const double a2 = us, b2 = std::fmax(xc, us);
    if (b2 > a2)
      F += 2.0 * (G(b2, r) - G(a2, r));
  }
  return F;
}

/// Exact SOLID area fraction of the square [y0,y0+h] x [z0,z0+h] cut by a disk of radius rho
/// centred at (cy, cz).
static double diskSquareOverlap(double cy, double cz, double rho, double y0, double z0, double h) {
  const double a = y0 - cy, b = a + h, c = z0 - cz, d = c + h;
  return Fquad(b, d, rho) - Fquad(a, d, rho) - Fquad(b, c, rho) - -Fquad(a, c, rho) * 0.0 +
         Fquad(a, c, rho) - Fquad(a, c, rho) * 0.0 - 0.0;
}

int main() {
  // --- a single sphere, faces of a uniform grid, against the closed form --------------------
  const double R = 0.3;
  const Vec3<double> C{0.5, 0.5, 0.5};
  auto sphere = [&](Vec3<double> p) {
    const double dx = p.x - C.x, dy = p.y - C.y, dz = p.z - C.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - R;  // >0 fluid
  };
  // exact OPEN fraction of the square [y0,y0+h]x[z0,z0+h] on the plane x, by inclusion-exclusion
  auto refOpenFace = [&](double x, double y0, double z0, double h) {
    const double d = x - C.x, rho2 = R * R - d * d;
    if (rho2 <= 0.0)
      return 1.0;
    const double rho = std::sqrt(rho2);
    const double a = y0 - C.y, b = a + h, c = z0 - C.z, e = c + h;
    const double solid = Fquad(b, e, rho) - Fquad(a, e, rho) - Fquad(b, c, rho) + Fquad(a, c, rho);
    return 1.0 - solid / (h * h);
  };

  std::printf("  implicit quadrature vs the closed-form disk-rectangle overlap (sphere R=%.2f)\n",
              R);
  std::vector<double> errs, errsNoSplit;
  const std::vector<int> Ns{16, 32, 64};
  for (int N : Ns) {
    const double h = 1.0 / N;
    double worstQ = 0, sumQ = 0, sumNo = 0, worstLin = 0, sumLin = 0;
    long nCut = 0;
    for (int i = 0; i <= N; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k) {
          const double x = i * h, y0 = j * h, z0 = k * h;
          const double ref = refOpenFace(x, y0, z0, h);
          if (ref <= 1e-12 || ref >= 1.0 - 1e-12)
            continue;  // uncut faces are exact for every method; measure the CUT ones
          ++nCut;
          const Vec3<double> o{x, y0, z0};
          const double eq = std::fabs(faceAperture<double>(sphere, o, 0, h, h, 5, 8) - ref);
          worstQ = std::fmax(worstQ, eq);
          sumQ += eq;
          sumNo += std::fabs(faceAperture<double>(sphere, o, 0, h, h, 5, 8, false) - ref);
          // the sampled-SDF linear estimator the CFD otherwise uses
          const double lin = std::fmax(
              0.0, std::fmin(1.0, 0.5 + sphere(Vec3<double>{x, y0 + 0.5 * h, z0 + 0.5 * h}) / h));
          worstLin = std::fmax(worstLin, std::fabs(lin - ref));
          sumLin += std::fabs(lin - ref);
        }
    errs.push_back(sumQ / (double)nCut);
    errsNoSplit.push_back(sumNo / (double)nCut);
    std::printf("    N=%3d  %5ld cut faces   mean err: quadrature %.3e   unsplit %.3e   "
                "linear %.3e   (worst quad %.3e, worst linear %.3e)\n",
                N, nCut, sumQ / nCut, sumNo / nCut, sumLin / nCut, worstQ, worstLin);
    PECLET_CORE_CHECK(worstQ < 1e-7);         // machine-ish even at the worst cut face
    PECLET_CORE_CHECK(sumQ < 1e-6 * sumLin);  // >= 10^6 better than the linear estimator
    PECLET_CORE_CHECK(sumQ < 1e-5 * sumNo);   // and splitting at the kinks is what buys it
  }
  std::printf("    -> splitting the outer rule at the face-exit kinks buys %.0e (N=16) to %.0e "
              "(N=64). The UNSPLIT error barely moves with h (%.2e -> %.2e), which is the point: "
              "an aperture is a fraction, so a fixed kink costs a fixed amount.\n",
              errsNoSplit[0] / errs[0], errsNoSplit.back() / errs.back(), errsNoSplit[0],
              errsNoSplit.back());

  // --- convergence is in the NODE COUNT, not in h ---------------------------------------------
  {
    const int N = 32;
    const double h = 1.0 / N;
    std::printf("  convergence in the NODE COUNT (N=%d, all cut x-faces):\n", N);
    double prev = 0;
    for (int ord : {2, 3, 4, 5, 8}) {
      double sum = 0;
      long n = 0;
      for (int i = 0; i <= N; ++i)
        for (int j = 0; j < N; ++j)
          for (int k = 0; k < N; ++k) {
            const double x = i * h, y0 = j * h, z0 = k * h;
            const double ref = refOpenFace(x, y0, z0, h);
            if (ref <= 1e-12 || ref >= 1.0 - 1e-12)
              continue;
            ++n;
            sum += std::fabs(
                faceAperture<double>(sphere, Vec3<double>{x, y0, z0}, 0, h, h, ord, 8) - ref);
          }
      std::printf("    order=%d  mean err %.3e\n", ord, sum / n);
      if (prev > 0)
        PECLET_CORE_CHECK(sum / n <= prev * 2.0);  // monotone down to the root-finder floor
      prev = sum / n;
    }
  }

  // --- cell volume fractions: the 2-D outer rule has kink CURVES, so no subdivision helps ------
  {
    const int N = 24;
    const double h = 1.0 / N;
    double vol = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k)
          vol += cellVolumeFraction<double>(sphere, Vec3<double>{i * h, j * h, k * h},
                                            Vec3<double>{h, h, h}, 5, 8) *
                 h * h * h;
    const double exact = 1.0 - 4.0 / 3.0 * M_PI * R * R * R;
    std::printf("  cell volume fractions sum to %.10f, exact fluid volume %.10f (rel %.2e) -- "
                "weaker than the faces because the 2-D outer rule's kinks are CURVES\n",
                vol, exact, std::fabs(vol - exact) / exact);
    PECLET_CORE_CHECK(std::fabs(vol - exact) / exact < 1e-5);
  }

  // --- robustness where the height-graph premise fails -----------------------------------------
  {
    auto torus = [](Vec3<double> p) {
      const double x = p.x - 0.5, y = p.y - 0.5, z = p.z - 0.5;
      const double q = std::sqrt(x * x + z * z) - 0.25;
      return std::sqrt(q * q + y * y) - 0.08;
    };
    const int N = 24;
    const double h = 1.0 / N;
    int bad = 0, cut = 0;
    double vol = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k) {
          const double a =
              faceAperture<double>(torus, Vec3<double>{i * h, j * h, k * h}, 1, h, h, 5, 12);
          if (!(a >= 0.0 && a <= 1.0))
            ++bad;
          if (a > 1e-9 && a < 1.0 - 1e-9)
            ++cut;
          vol += cellVolumeFraction<double>(torus, Vec3<double>{i * h, j * h, k * h},
                                            Vec3<double>{h, h, h}, 5, 12) *
                 h * h * h;
        }
    const double exact = 1.0 - 2.0 * M_PI * M_PI * 0.25 * 0.08 * 0.08;  // 1 - 2 pi^2 R r^2
    PECLET_CORE_CHECK(bad == 0);
    std::printf("  torus (4 crossings per line; NO order claimed): %d cut faces, %d out of [0,1]; "
                "volume %.8f vs exact %.8f (rel %.2e)\n",
                cut, bad, vol, exact, std::fabs(vol - exact) / exact);
    PECLET_CORE_CHECK(std::fabs(vol - exact) / exact < 5e-3);
  }

  // --- a SceneQueryView drives it just as well (the intended consumer path) ---------------------
  {
    SceneBuilder<double> b;
    const int s0 = b.addLeaf(kSphere, {0.3});
    b.addInstance(s0, Transform<double>{Vec3<double>{0.5, 0.5, 0.5}});
    const SceneView<double> sv = b.view();
    SceneQueryView<double> q;
    q.scene = sv;
    const double h = 1.0 / 32;
    double worst = 0;
    for (int j = 0; j < 32; ++j)
      for (int k = 0; k < 32; ++k) {
        const Vec3<double> o{0.5, j * h, k * h};
        worst = std::fmax(worst,
                          std::fabs(faceAperture<double>(
                                        [&](Vec3<double> p) { return q.eval(p); }, o, 0, h, h, 5, 8) -
                                    faceAperture<double>(sphere, o, 0, h, h, 5, 8)));
      }
    PECLET_CORE_CHECK(worst < 1e-14);
    std::printf("  SceneQueryView vs a raw lambda: worst face difference %.2e\n", worst);
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

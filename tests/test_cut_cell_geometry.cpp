// peclet::core::scheme cut-cell geometric record (peclet/core/scheme/cut_cell_geometry.hpp).
//
// Gate G-geom (a)-(c) of flow doc/scalar_ibm_design.md §11 (WO-1); (d), the probe ladder, is
// tests/test_probe_flux.cpp.
//   (a) 1000 random planar cuts (isotropic cells, and again on random anisotropic cells):
//       |κ − plicVolume| ≤ 1e-14, apertures within 1e-14 of the exact plane–square fractions, the
//       facet area vector within 1e-13·h² and the centroid within 1e-13·h of plicPolygon +
//       polygonAreaCentroid (core vof/curvature.hpp);
//   (b) random non-planar samples: the PL closure |Σ ± a A e − areaPL| ≤ 1e-14·max A, κ ∈ [0, 1],
//       κ = 0 ⇒ all apertures 0 (with exact-zero samples mixed in);
//   (c) a gap (two planes 0.3h apart, fluid between) → `gap`, 2 facets, opposite normals; a slab
//       0.3h thick → `thinSolid`.
// Plus the fraction formulas' unit checks and the bitwise identity of a face aperture computed from
// both cells sharing the face (the fixed corner summation order of §2.2).
//
// Kokkos is included first so the header compiles on its KOKKOS_INLINE_FUNCTION path and the vof
// oracles are available; test_probe_flux.cpp covers the plain-inline path.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>

#include "peclet/core/scheme/cut_cell_geometry.hpp"
#include "peclet/core/vof/curvature.hpp"
#include "peclet/core/vof/plic.hpp"
#include "test_util.hpp"

using namespace peclet::core::scheme;
namespace vof = peclet::core::vof;

namespace {

struct Samples {
  double corner[8];
  double face[6];
  double centre;
};

// Samples of the plane φ(x) = n·x − d (fluid on the +n side) at the 15 PL points.
Samples planeSamples(const double n[3], double d, const double h[3]) {
  Samples s;
  for (int c = 0; c < 8; ++c) {
    double x[3];
    for (int a = 0; a < 3; ++a)
      x[a] = (((c >> a) & 1) ? 0.5 : -0.5) * h[a];
    s.corner[c] = n[0] * x[0] + n[1] * x[1] + n[2] * x[2] - d;
  }
  for (int f = 0; f < 6; ++f) {
    const int a = f >> 1;
    s.face[f] = n[a] * ((f & 1) ? 0.5 : -0.5) * h[a] - d;
  }
  s.centre = -d;
  return s;
}

double maxAbs3(const double v[3]) {
  return std::fmax(std::fabs(v[0]), std::fmax(std::fabs(v[1]), std::fabs(v[2])));
}

void randomUnit(std::mt19937_64& rng, double n[3]) {
  std::normal_distribution<double> g(0.0, 1.0);
  double q = 0.0;
  do {
    for (int a = 0; a < 3; ++a)
      n[a] = g(rng);
    q = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  } while (q < 1e-3);
  for (int a = 0; a < 3; ++a)
    n[a] /= q;
}

// G-geom (a): one batch of `count` random planar cuts. Returns the max errors in `err`
// (κ, aperture, area vector / h², centroid / h).
void planarBatch(std::mt19937_64& rng, int count, bool aniso, double err[4]) {
  std::uniform_real_distribution<double> uh(0.5, 2.0), u01(0.0, 1.0);
  for (int i = 0; i < 4; ++i)
    err[i] = 0.0;
  for (int t = 0; t < count; ++t) {
    double h[3] = {1.0, 1.0, 1.0};
    if (aniso)
      for (int a = 0; a < 3; ++a)
        h[a] = uh(rng);
    double n[3];
    randomUnit(rng, n);
    // d uniform over the offsets at which the plane meets the cell: |d| < S(n).
    const double S =
        0.5 * (std::fabs(n[0]) * h[0] + std::fabs(n[1]) * h[1] + std::fabs(n[2]) * h[2]);
    const double d = (2.0 * u01(rng) - 1.0) * S;
    const Samples s = planeSamples(n, d, h);
    const CutCellGeometry g = cutCellGeometryFanTet(s.corner, s.face, s.centre, h);

    // Oracle in the unit-cube frame u = x/h + ½: fluid ⇔ m·u < alpha, m = −n∘h.
    const double m[3] = {-n[0] * h[0], -n[1] * h[1], -n[2] * h[2]};
    const double alpha = -d - 0.5 * (n[0] * h[0] + n[1] * h[1] + n[2] * h[2]);
    const double kEx = vof::plicVolume(m[0], m[1], m[2], alpha);
    err[0] = std::fmax(err[0], std::fabs(g.kappa - kEx));
    for (int f = 0; f < 6; ++f) {
      const int a = f >> 1;
      double mm[3] = {m[0], m[1], m[2]};
      mm[a] = 0.0;  // the face's square: the plane restricted to u_a = 0 or 1
      const double aEx = vof::plicVolume(mm[0], mm[1], mm[2], alpha - ((f & 1) ? m[a] : 0.0));
      err[1] = std::fmax(err[1], std::fabs(g.aperture[f] - aEx));
    }
    double v[8][3];
    const int nv = vof::plicPolygon(m[0], m[1], m[2], alpha, v);
    for (int k = 0; k < nv; ++k)
      for (int a = 0; a < 3; ++a)
        v[k][a] = (v[k][a] - 0.5) * h[a];
    double ctr[3], area = 0.0;
    vof::polygonAreaCentroid(v, nv, ctr, area);
    const double hmax = std::fmax(h[0], std::fmax(h[1], h[2]));
    double dA[3];
    for (int a = 0; a < 3; ++a)
      dA[a] = g.areaPL[a] - area * n[a];
    err[2] = std::fmax(err[2], maxAbs3(dA) / (hmax * hmax));
    if (g.nFacet == 1) {
      PECLET_CORE_CHECK(g.kind == CutCellKind::single);
      double dF[3], dC[3];
      for (int a = 0; a < 3; ++a) {
        dF[a] = g.facetArea[0][a] - area * n[a];
        dC[a] = g.facetCentroid[0][a] - ctr[a];
      }
      err[2] = std::fmax(err[2], maxAbs3(dF) / (hmax * hmax));
      err[3] = std::fmax(err[3], maxAbs3(dC) / hmax);
    } else {
      // No facet is acceptable only when the exact cut is itself below the drop threshold.
      PECLET_CORE_CHECK(g.nFacet == 0 && area < 1e-13 * hmax * hmax);
    }
  }
}

void checkFractions() {
  // §2.2's check value.
  PECLET_CORE_CHECK(std::fabs(tetPositiveFraction(1.0, 1.0, -1.0, -1.0) - 0.5) <= 1e-16);
  PECLET_CORE_CHECK(tetPositiveFraction(1.0, 2.0, 3.0, 4.0) == 1.0);
  PECLET_CORE_CHECK(tetPositiveFraction(0.0, -2.0, 0.0, -4.0) == 0.0);  // φ = 0 is solid
  PECLET_CORE_CHECK(triPositiveFraction(0.0, 0.0, 0.0) == 0.0);
  PECLET_CORE_CHECK(triPositiveFraction(1.0, 0.0, 0.0) == 1.0);  // positive a.e.
  // Complement: F(v) + F(−v) = 1 for nonzero values, in every sign pattern.
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  double e = 0.0;
  for (int t = 0; t < 20000; ++t) {
    const double a = u(rng), b = u(rng), c = u(rng), d = u(rng);
    e = std::fmax(
        e, std::fabs(tetPositiveFraction(a, b, c, d) + tetPositiveFraction(-a, -b, -c, -d) - 1.0));
    e = std::fmax(e,
                  std::fabs(triPositiveFraction(a, b, c) + triPositiveFraction(-a, -b, -c) - 1.0));
  }
  std::printf("  fractions: max |F(v) + F(-v) - 1| = %.3e\n", e);
  PECLET_CORE_CHECK(e <= 1e-14);
  // Scale invariance incl. the underflow fallback.
  const double f1 = tetPositiveFraction(0.3, 0.7, -0.2, -0.9);
  const double f2 = tetPositiveFraction(0.3e-200, 0.7e-200, -0.2e-200, -0.9e-200);
  PECLET_CORE_CHECK(std::fabs(f1 - f2) <= 1e-15);
}

// G-geom (b).
void checkNonPlanar() {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> u(-1.0, 1.0), uh(0.5, 2.0), u01(0.0, 1.0);
  double eClose = 0.0;
  int nCut = 0, nZeroK = 0;
  for (int t = 0; t < 20000; ++t) {
    double h[3] = {1.0, 1.0, 1.0};
    if (t % 2)
      for (int a = 0; a < 3; ++a)
        h[a] = uh(rng);
    Samples s;
    const double pz = (t % 4 == 3) ? 0.3 : 0.0;  // a quarter of the trials carry exact zeros
    auto draw = [&]() { return u01(rng) < pz ? 0.0 : u(rng) * h[0]; };
    for (int c = 0; c < 8; ++c)
      s.corner[c] = draw();
    for (int f = 0; f < 6; ++f)
      s.face[f] = draw();
    s.centre = draw();
    const CutCellGeometry g = cutCellGeometryFanTet(s.corner, s.face, s.centre, h);
    double impl[3];
    facetAreaVector(g.aperture, h, impl);
    const double Amax = std::fmax(h[1] * h[2], std::fmax(h[0] * h[2], h[0] * h[1]));
    double dv[3];
    for (int a = 0; a < 3; ++a)
      dv[a] = impl[a] - g.areaPL[a];
    eClose = std::fmax(eClose, maxAbs3(dv) / Amax);
    PECLET_CORE_CHECK(g.kappa >= 0.0 && g.kappa <= 1.0);
    for (int f = 0; f < 6; ++f)
      PECLET_CORE_CHECK(g.aperture[f] >= 0.0 && g.aperture[f] <= 1.0);
    if (g.kappa == 0.0) {
      ++nZeroK;
      for (int f = 0; f < 6; ++f)
        PECLET_CORE_CHECK(g.aperture[f] == 0.0);
    }
    if (g.nFacet > 0)
      ++nCut;
    PECLET_CORE_CHECK((g.nFacet == 0) == (g.kind == CutCellKind::none));
  }
  std::printf(
      "  (b) 20000 random samples (%d cut, %d with kappa = 0): max PL closure / max A = %.3e\n",
      nCut, nZeroK, eClose);
  PECLET_CORE_CHECK(eClose <= 1e-14);
}

// A face aperture is the same bits from both cells sharing it, given the §2.2 sample assembly.
void checkSharedFace() {
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const int NX = 4, NY = 3, NZ = 3;
  for (int t = 0; t < 2000; ++t) {
    double sdf[NZ][NY][NX];
    for (int k = 0; k < NZ; ++k)
      for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
          sdf[k][j][i] = u(rng);
    auto cell = [&](int i, int j, int k, Samples& s) {
      for (int c = 0; c < 8; ++c) {
        // corner keyed to its lowest cell (i0, j0, k0)
        const int i0 = i - 1 + (c & 1), j0 = j - 1 + ((c >> 1) & 1), k0 = k - 1 + (c >> 2);
        auto S = [&](int di, int dj, int dk) { return sdf[k0 + dk][j0 + dj][i0 + di]; };
        s.corner[c] = (((S(0, 0, 0) + S(1, 0, 0)) + (S(0, 1, 0) + S(1, 1, 0))) +
                       ((S(0, 0, 1) + S(1, 0, 1)) + (S(0, 1, 1) + S(1, 1, 1)))) *
                      0.125;
      }
      s.face[0] = 0.5 * (sdf[k][j][i - 1] + sdf[k][j][i]);
      s.face[1] = 0.5 * (sdf[k][j][i] + sdf[k][j][i + 1]);
      s.face[2] = 0.5 * (sdf[k][j - 1][i] + sdf[k][j][i]);
      s.face[3] = 0.5 * (sdf[k][j][i] + sdf[k][j + 1][i]);
      s.face[4] = 0.5 * (sdf[k - 1][j][i] + sdf[k][j][i]);
      s.face[5] = 0.5 * (sdf[k][j][i] + sdf[k + 1][j][i]);
      s.centre = sdf[k][j][i];
    };
    const double h[3] = {1.0, 1.0, 1.0};
    Samples a, b;
    cell(1, 1, 1, a);
    cell(2, 1, 1, b);
    const CutCellGeometry ga = cutCellGeometryFanTet(a.corner, a.face, a.centre, h);
    const CutCellGeometry gb = cutCellGeometryFanTet(b.corner, b.face, b.centre, h);
    PECLET_CORE_CHECK(ga.aperture[1] == gb.aperture[0]);
  }
}

// G-geom (c): an x-normal gap / slab of width 0.3h centred at x = c, PL samples of
// φ = ±(0.15h − |x − c|).
void checkGapSlab(double c, double sign, CutCellKind expect, const double n[3]) {
  const double h[3] = {1.0, 1.0, 1.0};
  auto phi = [&](const double x[3]) {
    const double xn = n[0] * x[0] + n[1] * x[1] + n[2] * x[2];
    return sign * (0.15 - std::fabs(xn - c));
  };
  Samples s;
  for (int k = 0; k < 8; ++k) {
    const double x[3] = {((k & 1) ? 0.5 : -0.5), (((k >> 1) & 1) ? 0.5 : -0.5),
                         ((k >> 2) ? 0.5 : -0.5)};
    s.corner[k] = phi(x);
  }
  for (int f = 0; f < 6; ++f) {
    double x[3] = {0.0, 0.0, 0.0};
    x[f >> 1] = (f & 1) ? 0.5 : -0.5;
    s.face[f] = phi(x);
  }
  const double x0[3] = {0.0, 0.0, 0.0};
  s.centre = phi(x0);
  const CutCellGeometry g = cutCellGeometryFanTet(s.corner, s.face, s.centre, h);
  const double a0 =
      std::sqrt(g.facetArea[0][0] * g.facetArea[0][0] + g.facetArea[0][1] * g.facetArea[0][1] +
                g.facetArea[0][2] * g.facetArea[0][2]);
  const double a1 =
      std::sqrt(g.facetArea[1][0] * g.facetArea[1][0] + g.facetArea[1][1] * g.facetArea[1][1] +
                g.facetArea[1][2] * g.facetArea[1][2]);
  const double cosn =
      (g.facetArea[0][0] * g.facetArea[1][0] + g.facetArea[0][1] * g.facetArea[1][1] +
       g.facetArea[0][2] * g.facetArea[1][2]) /
      (a0 * a1);
  std::printf(
      "  (c) %s c=%+.2f n=(%.3f,%.3f,%.3f): kind %d, nFacet %d, kappa %.6f, |A0| %.6f, "
      "|A1| %.6f, cos(n0,n1) %.12f\n",
      sign > 0 ? "gap " : "slab", c, n[0], n[1], n[2], static_cast<int>(g.kind), g.nFacet, g.kappa,
      a0, a1, cosn);
  PECLET_CORE_CHECK(g.kind == expect);
  PECLET_CORE_CHECK(g.nFacet == 2);
  PECLET_CORE_CHECK(cosn < -0.999);
}

}  // namespace

int main() {
  std::mt19937_64 rng(20261002);
  double eIso[4], eAni[4];
  planarBatch(rng, 1000, false, eIso);
  planarBatch(rng, 1000, true, eAni);
  std::printf(
      "  (a) 1000 planar cuts, isotropic : |dkappa| %.3e  |dap| %.3e  |dA|/h^2 %.3e  "
      "|dx|/h %.3e\n",
      eIso[0], eIso[1], eIso[2], eIso[3]);
  std::printf(
      "  (a) 1000 planar cuts, anisotropic: |dkappa| %.3e  |dap| %.3e  |dA|/h^2 %.3e  "
      "|dx|/h %.3e\n",
      eAni[0], eAni[1], eAni[2], eAni[3]);
  for (const double* e : {eIso, eAni}) {
    PECLET_CORE_CHECK(e[0] <= 1e-14);
    PECLET_CORE_CHECK(e[1] <= 1e-14);
    PECLET_CORE_CHECK(e[2] <= 1e-13);
    PECLET_CORE_CHECK(e[3] <= 1e-13);
  }

  checkFractions();
  checkNonPlanar();
  checkSharedFace();

  const double ex[3] = {1.0, 0.0, 0.0};
  checkGapSlab(0.0, +1.0, CutCellKind::gap, ex);
  checkGapSlab(0.0, -1.0, CutCellKind::thinSolid, ex);
  // Off-centre and oblique variants of the same two configurations.
  const double ob[3] = {0.9, 0.3, std::sqrt(1.0 - 0.81 - 0.09)};
  checkGapSlab(0.1, +1.0, CutCellKind::gap, ob);
  checkGapSlab(-0.1, -1.0, CutCellKind::thinSolid, ob);

  // Snapping at both ends (§2.3).
  PECLET_CORE_CHECK(snapAperture(0.0009) == 0.0);
  PECLET_CORE_CHECK(snapAperture(0.001) == 0.001);
  PECLET_CORE_CHECK(snapAperture(0.9995) == 1.0);
  PECLET_CORE_CHECK(snapAperture(0.5) == 0.5);
  PECLET_CORE_CHECK(1.0 - snapAperture(0.9995) == snapAperture(1.0 - 0.9995));

  PECLET_CORE_RETURN_TEST_RESULT();
}

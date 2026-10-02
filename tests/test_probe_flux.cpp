// peclet::core::scheme probe-flux closure (peclet/core/scheme/probe_flux.hpp).
//
// Gate G-geom (d) of flow doc/scalar_ibm_design.md §11 (WO-1), the probe-ladder unit cases:
//   - all valid → R0 (1000 random planar facets from cutCellGeometryFanTet, isotropic cells);
//   - one stencil cell covered → R1a;
//   - opposite wall at 0.8 s₀ → R1b;
//   - no valid cell → R2;
//   - the weights reproduce linear fields to 1e-14.
// Plus: V3 (reach) on a strongly anisotropic cell, the Lookup is never queried beyond ±2, the
// support function S(n), and the wall / interface conductances of §1.4.
//
// Host-only, no Kokkos: this binary also compiles both scheme headers on their plain-inline path.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "peclet/core/scheme/cut_cell_geometry.hpp"
#include "peclet/core/scheme/probe_flux.hpp"
#include "test_util.hpp"

using namespace peclet::core::scheme;

namespace {

// Records the widest lattice offset / φ sample the ladder asked about.
struct Reach {
  mutable int maxOff = 0;
  mutable double maxXi = 0.0;
  void off(int dx, int dy, int dz) const {
    maxOff = std::max(maxOff, std::max(std::abs(dx), std::max(std::abs(dy), std::abs(dz))));
  }
  void xi(const double x[3]) const {
    for (int a = 0; a < 3; ++a)
      maxXi = std::fmax(maxXi, std::fabs(x[a]));
  }
};

// A planar wall through xw with unit normal n (fluid on +n): φ(x) = n·(x − xw). A cell is an
// unknown iff it meets the fluid half-space (centre distance > −S(n)), minus an optional covered
// cell.
struct PlaneLookup : Reach {
  double n[3], xw[3], h[3];
  bool cover = false;
  int cov[3] = {0, 0, 0};
  double dist(const double x[3]) const {
    return n[0] * (x[0] - xw[0]) + n[1] * (x[1] - xw[1]) + n[2] * (x[2] - xw[2]);
  }
  bool unknown(int dx, int dy, int dz) const {
    off(dx, dy, dz);
    if (cover && dx == cov[0] && dy == cov[1] && dz == cov[2])
      return false;
    const double c[3] = {dx * h[0], dy * h[1], dz * h[2]};
    return dist(c) > -probeSupport(n, h);
  }
  double phi(const double x[3]) const {
    xi(x);
    const double p[3] = {x[0] * h[0], x[1] * h[1], x[2] * h[2]};
    return dist(p);
  }
};

// A fluid slab between x = 0 and x = W (unit cells): φ = min(x, W − x); unknown iff the cell's
// x-extent meets (0, W).
struct SlabLookup : Reach {
  double W;
  bool unknown(int dx, int dy, int dz) const {
    off(dx, dy, dz);
    return dx + 0.5 > 0.0 && dx - 0.5 < W;
  }
  double phi(const double x[3]) const {
    xi(x);
    return std::fmin(x[0], W - x[0]);
  }
};

// Nothing (or only the cell itself) is an unknown.
struct NoneLookup : PlaneLookup {
  bool self = false;
  bool unknown(int dx, int dy, int dz) const {
    off(dx, dy, dz);
    return self && dx == 0 && dy == 0 && dz == 0;
  }
};

// Max |Σ_k w_k L(x_k) − L(p)| for linear fields L(x) = c + g·x, relative to max |L| on the stencil.
double linearDefect(const ProbeResult& r, const double xw[3], const double n[3], const double h[3],
                    std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  double e = 0.0;
  for (int t = 0; t < 4; ++t) {
    const double c = u(rng), g[3] = {u(rng), u(rng), u(rng)};
    double sum = 0.0, scale = std::fabs(c);
    for (int k = 0; k < r.n; ++k) {
      const double L =
          c + g[0] * r.off[k][0] * h[0] + g[1] * r.off[k][1] * h[1] + g[2] * r.off[k][2] * h[2];
      sum += r.w[k] * L;
      scale = std::fmax(scale, std::fabs(L));
    }
    const double Lp =
        c + g[0] * (xw[0] + r.s * n[0]) + g[1] * (xw[1] + r.s * n[1]) + g[2] * (xw[2] + r.s * n[2]);
    e = std::fmax(e, std::fabs(sum - Lp) / scale);
  }
  return e;
}

double weightSum(const ProbeResult& r) {
  double s = 0.0;
  for (int k = 0; k < r.n; ++k)
    s += r.w[k];
  return s;
}

}  // namespace

int main() {
  std::mt19937_64 rng(20261002);
  std::uniform_real_distribution<double> u(-1.0, 1.0), u01(0.0, 1.0);

  // S(n).
  {
    const double h[3] = {1.0, 2.0, 4.0};
    const double nx[3] = {-1.0, 0.0, 0.0}, nd[3] = {0.6, 0.0, -0.8};
    PECLET_CORE_CHECK(probeSupport(nx, h) == 0.5);
    PECLET_CORE_CHECK(std::fabs(probeSupport(nd, h) - 0.5 * (0.6 + 3.2)) <= 1e-15);
  }

  // Trilinear stencil: weights sum to 1 and reproduce linear fields (random ξ in (−1.9, 1.9)³).
  double eTri = 0.0;
  for (int t = 0; t < 10000; ++t) {
    const double xi[3] = {1.9 * u(rng), 1.9 * u(rng), 1.9 * u(rng)};
    int base[3];
    double w[8];
    trilinearStencil(xi, base, w);
    const double c = u(rng), g[3] = {u(rng), u(rng), u(rng)};
    double sum = 0.0, ws = 0.0;
    for (int k = 0; k < 8; ++k) {
      const double x[3] = {double(base[0] + (k & 1)), double(base[1] + ((k >> 1) & 1)),
                           double(base[2] + (k >> 2))};
      sum += w[k] * (c + g[0] * x[0] + g[1] * x[1] + g[2] * x[2]);
      ws += w[k];
      PECLET_CORE_CHECK(w[k] >= 0.0);
    }
    eTri = std::fmax(eTri, std::fabs(sum - (c + g[0] * xi[0] + g[1] * xi[1] + g[2] * xi[2])));
    eTri = std::fmax(eTri, std::fabs(ws - 1.0));
  }
  std::printf("  trilinear stencil: max linear defect %.3e (10000 points)\n", eTri);
  PECLET_CORE_CHECK(eTri <= 1e-14);

  // (d.1) All valid → R0, on facets from the fan-tet record of random planar cuts.
  {
    const double h[3] = {1.0, 1.0, 1.0};
    int nR0 = 0, nTot = 0, maxOff = 0;
    double eLin = 0.0, sMin = 1e9, sMax = 0.0;
    for (int t = 0; t < 1000; ++t) {
      double n[3];
      double q = 0.0;
      do {
        for (int a = 0; a < 3; ++a)
          n[a] = u(rng);
        q = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      } while (q < 1e-2 || q > 1.0);
      for (int a = 0; a < 3; ++a)
        n[a] /= q;
      const double S = probeSupport(n, h);
      const double d = (2.0 * u01(rng) - 1.0) * S;
      double corner[8], face[6];
      for (int c = 0; c < 8; ++c) {
        double xc = 0.0;
        for (int a = 0; a < 3; ++a)
          xc += n[a] * (((c >> a) & 1) ? 0.5 : -0.5) * h[a];
        corner[c] = xc - d;
      }
      for (int f = 0; f < 6; ++f)
        face[f] = n[f >> 1] * ((f & 1) ? 0.5 : -0.5) * h[f >> 1] - d;
      const CutCellGeometry g = cutCellGeometryFanTet(corner, face, -d, h);
      if (g.nFacet != 1)
        continue;  // a cut below the drop threshold has no facet to probe
      ++nTot;
      const double a = std::sqrt(g.areaPL[0] * g.areaPL[0] + g.areaPL[1] * g.areaPL[1] +
                                 g.areaPL[2] * g.areaPL[2]);
      const double nf[3] = {g.areaPL[0] / a, g.areaPL[1] / a, g.areaPL[2] / a};
      PlaneLookup L;
      for (int k = 0; k < 3; ++k) {
        L.n[k] = nf[k];
        L.xw[k] = g.facetCentroid[0][k];
        L.h[k] = h[k];
      }
      const ProbeResult r = buildProbe(g.facetCentroid[0], nf, h, L);
      if (r.rung == kProbeR0)
        ++nR0;
      PECLET_CORE_CHECK(r.n == 8);
      eLin = std::fmax(eLin, linearDefect(r, g.facetCentroid[0], nf, h, rng));
      eLin = std::fmax(eLin, std::fabs(weightSum(r) - 1.0));
      sMin = std::fmin(sMin, r.s);
      sMax = std::fmax(sMax, r.s);
      maxOff = std::max(maxOff, L.maxOff);
    }
    std::printf(
        "  (d) all valid: %d / %d facets R0, s in [%.4f, %.4f] h, max |offset| queried %d, "
        "max linear defect %.3e\n",
        nR0, nTot, sMin, sMax, maxOff, eLin);
    PECLET_CORE_CHECK(nR0 == nTot && nTot > 990);
    PECLET_CORE_CHECK(maxOff <= 2);
    PECLET_CORE_CHECK(eLin <= 1e-14);
    PECLET_CORE_CHECK(sMin >= 0.55 - 1e-12 && sMax <= 1.1 * std::sqrt(3.0) / 2.0 + 1e-12);
  }

  // (d.2) One stencil cell covered → R1a. Wall x = 0.35, n = +x; R0's point (0.9, 0.1, 0.2) has
  // cell (0, 1, 0) at weight 0.008; R1a's point (1.1, 0.1, 0.2) has a stencil without it.
  {
    const double h[3] = {1.0, 1.0, 1.0}, n[3] = {1.0, 0.0, 0.0}, xw[3] = {0.35, 0.1, 0.2};
    PlaneLookup L;
    for (int k = 0; k < 3; ++k) {
      L.n[k] = n[k];
      L.xw[k] = xw[k];
      L.h[k] = h[k];
    }
    PlaneLookup L0 = L;
    PECLET_CORE_CHECK(buildProbe(xw, n, h, L0).rung == kProbeR0);  // uncovered control
    L.cover = true;
    L.cov[1] = 1;
    const ProbeResult r = buildProbe(xw, n, h, L);
    const double eLin = linearDefect(r, xw, n, h, rng);
    std::printf(
        "  (d) one cell covered: rung %d (R1a = %d), s = %.4f, base x %d, linear defect %.3e\n",
        r.rung, kProbeR1a, r.s, r.off[0][0], eLin);
    PECLET_CORE_CHECK(r.rung == kProbeR1a);
    PECLET_CORE_CHECK(std::fabs(r.s - 0.75) <= 1e-15 && r.n == 8 && r.off[0][0] == 1);
    PECLET_CORE_CHECK(eLin <= 1e-14);
    PECLET_CORE_CHECK(L.maxOff <= 2);
  }

  // (d.3) Opposite wall at 0.8 s₀ → R1b. Fluid slab x ∈ (0, 0.44) (s₀ = 0.55), facet at
  // (0, 0.1, 0.2) with n = +x: R0 fails V2, the gap estimate is w = 0.44, s = 0.22, and only the
  // x = 0 layer is kept (weight 0.78), renormalized to the bilinear weights in (y, z).
  {
    const double h[3] = {1.0, 1.0, 1.0}, n[3] = {1.0, 0.0, 0.0}, xw[3] = {0.0, 0.1, 0.2};
    SlabLookup L;
    L.W = 0.8 * kProbeSigma * 0.5;
    const ProbeResult r = buildProbe(xw, n, h, L);
    std::printf(
        "  (d) opposite wall at 0.8 s0: rung %d (R1b = %d), s = %.4f, n = %d, sum w = %.17g\n",
        r.rung, kProbeR1b, r.s, r.n, weightSum(r));
    PECLET_CORE_CHECK(r.rung == kProbeR1b);
    PECLET_CORE_CHECK(std::fabs(r.s - 0.22) <= 1e-15 && r.n == 4);
    PECLET_CORE_CHECK(std::fabs(weightSum(r) - 1.0) <= 1e-15);
    const double wy[2] = {0.9, 0.1}, wz[2] = {0.8, 0.2};
    for (int k = 0; k < r.n; ++k) {
      PECLET_CORE_CHECK(r.off[k][0] == 0);
      PECLET_CORE_CHECK(std::fabs(r.w[k] - wy[r.off[k][1]] * wz[r.off[k][2]]) <= 1e-15);
    }
    PECLET_CORE_CHECK(L.maxOff <= 2);
  }

  // (d.4) No valid cell → R2 (nothing an unknown; and only the cell itself an unknown).
  for (int self = 0; self < 2; ++self) {
    const double h[3] = {1.0, 1.0, 1.0}, n[3] = {1.0, 0.0, 0.0}, xw[3] = {0.0, 0.1, 0.2};
    NoneLookup L;
    for (int k = 0; k < 3; ++k) {
      L.n[k] = n[k];
      L.xw[k] = xw[k];
      L.h[k] = h[k];
    }
    L.self = self != 0;
    const ProbeResult r = buildProbe(xw, n, h, L);
    std::printf("  (d) no valid cell (self %d): rung %d (R2 = %d), s = %.4f, n = %d\n", self,
                r.rung, kProbeR2, r.s, r.n);
    PECLET_CORE_CHECK(r.rung == kProbeR2 && r.n == 1 && r.w[0] == 1.0);
    PECLET_CORE_CHECK(r.off[0][0] == 0 && r.off[0][1] == 0 && r.off[0][2] == 0);
    PECLET_CORE_CHECK(std::fabs(r.s - 0.25) <= 1e-15);
  }

  // V3: on h = (1, 1, 10) with n = (0.8, 0, 0.6), s₀ = 3.74 reaches ξ_x = 2.99 — beyond the G = 2
  // halo. The ladder must fall to R2 without querying the Lookup at all.
  {
    const double h[3] = {1.0, 1.0, 10.0}, n[3] = {0.8, 0.0, 0.6}, xw[3] = {0.0, 0.0, 0.0};
    PlaneLookup L;
    for (int k = 0; k < 3; ++k) {
      L.n[k] = n[k];
      L.xw[k] = xw[k];
      L.h[k] = h[k];
    }
    const ProbeResult r = buildProbe(xw, n, h, L);
    std::printf("  V3 reach: rung %d, max |offset| queried %d, max |xi| sampled %.3f\n", r.rung,
                L.maxOff, L.maxXi);
    PECLET_CORE_CHECK(r.rung == kProbeR2);
    PECLET_CORE_CHECK(L.maxOff == 0 && L.maxXi == 0.0);
  }

  // Conductances (§1.4).
  {
    const double s = 0.6, L = 0.3, k = 2.0;
    PECLET_CORE_CHECK(wallConductance(s, L, 0.0) == 0.0);         // Neumann
    PECLET_CORE_CHECK(wallConductance(s, L, INFINITY) == L / s);  // Dirichlet
    PECLET_CORE_CHECK(wallConductance(s, L, k) == 1.0 / (s / L + 1.0 / k));
    // The Robin elimination: c_Γ = (L u_p + k s g)/(L + k s) and q_in = k (g − c_Γ) = G (g − u_p).
    const double up = 0.7, g = 0.2;
    const double cG = (L * up + k * s * g) / (L + k * s);
    PECLET_CORE_CHECK(std::fabs(k * (g - cG) - wallConductance(s, L, k) * (g - up)) <= 1e-15);
    PECLET_CORE_CHECK(std::fabs(L * (up - cG) / s - k * (cG - g)) <= 1e-15);
    // Series resistance: R_c = 0, equal legs reduce to two Dirichlet legs in series.
    PECLET_CORE_CHECK(interfaceConductance(s, L, 0.0, s, L) == 1.0 / (s / L + 0.0 + s / L));
    PECLET_CORE_CHECK(std::fabs(interfaceConductance(s, L, 0.0, s, L) - 0.5 * L / s) <= 1e-15);
    PECLET_CORE_CHECK(interfaceConductance(0.4, 0.3, 1.5, 0.7, 2.0) ==
                      1.0 / (0.4 / 0.3 + 1.5 + 0.7 / 2.0));
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

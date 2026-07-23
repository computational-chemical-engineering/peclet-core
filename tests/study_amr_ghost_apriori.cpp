// A-priori defect measurement for the AMR collocated projection (step 0 of the ghost-projection
// port — see flow/tests/study/ghost_collocated_apriori.py for the uniform-grid original).
//
// Measures, on the REAL AmrPoisson operators (uniform unrefined octree == structured grid):
//   [A1] the ABC cell-gradient gradOf (the -grad(P) predictor / cell-correction operator) on a
//        smooth pressure at cut cells: it reads the DECOUPLED p=0 of solid-centered neighbours
//        through partially-open faces (alpha > 1e-12), so it is expected O(1)-or-worse and
//        gauge-dependent. Ladder contrast: the directional ghost gradient (gpCenterGrad analog —
//        central where both axis-neighbour centers are fluid, 2nd-order one-sided else), expected
//        O(h^2) and exactly gauge-invariant.
//   [A2] the openness-weighted 1/2-1/2 face-average divergence (the projection constraint) on the
//        exact solenoidal Stokes field around the sphere: expected O(1) local truncation at cut
//        faces, O(h^2) in the bulk (fixed shell).
//
// This is a STUDY (measurement, printed tables + gates), not a regression test — the gates
// document the defect so the fix (the ghost overlay) has a measured baseline.
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef PECLET_CORE_HAVE_MORTON
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

constexpr double kPi = 3.14159265358979323846;
const Vec<3> C0{0.513, 0.493, 0.504};  // off-lattice center (no symmetry luck)
constexpr double R0 = 0.30;

double sdfSphere(const Vec<3>& p) {
  double dx = p[0] - C0[0], dy = p[1] - C0[1], dz = p[2] - C0[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz) - R0;
}

double phiMan(const Vec<3>& p) {
  const double tp = 2.0 * kPi;
  return std::sin(tp * p[0]) * std::cos(tp * p[1]) + std::sin(tp * p[1]) * std::cos(tp * p[2]) +
         std::sin(tp * p[2]) * std::cos(tp * p[0]);
}
Vec<3> phiManGrad(const Vec<3>& p) {
  const double tp = 2.0 * kPi;
  return {tp * std::cos(tp * p[0]) * std::cos(tp * p[1]) -
              tp * std::sin(tp * p[2]) * std::sin(tp * p[0]),
          -tp * std::sin(tp * p[0]) * std::sin(tp * p[1]) +
              tp * std::cos(tp * p[1]) * std::cos(tp * p[2]),
          -tp * std::sin(tp * p[1]) * std::sin(tp * p[2]) +
              tp * std::cos(tp * p[2]) * std::cos(tp * p[0])};
}

/// Exact Stokes flow past the sphere, U_inf = (1,0,0): solenoidal, no-slip at r=R0.
Vec<3> stokesU(const Vec<3>& p) {
  double dx = p[0] - C0[0], dy = p[1] - C0[1], dz = p[2] - C0[2];
  double r2 = dx * dx + dy * dy + dz * dz;
  double r = std::sqrt(r2);
  double A = 3.0 * R0 / (4.0 * r);
  double B = R0 * R0 * R0 / (4.0 * r2 * r);
  return {1.0 - A * (1.0 + dx * dx / r2) - B * (1.0 - 3.0 * dx * dx / r2),
          -(A - 3.0 * B) * dx * dy / r2, -(A - 3.0 * B) * dx * dz / r2};
}

// flow ccFractionCore aperture — verbatim copy of oracle::AmrFlow::faceFrac.
double faceFrac(const Vec<3>& fc, int axis, double h0) {
  double sd = sdfSphere(fc);
  if (sd <= 0.0)
    return 0.0;
  Vec<3> g{};
  for (int d = 0; d < 3; ++d) {
    Vec<3> pp = fc, pm = fc;
    pp[d] += h0;
    pm[d] -= h0;
    g[d] = (sdfSphere(pp) - sdfSphere(pm)) / (2.0 * h0);
  }
  double gmag = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
  if (gmag < 1e-6)
    gmag = 1e-6;
  int t1 = (axis + 1) % 3, t2 = (axis + 2) % 3;
  double denom = (std::fabs(g[t1]) + std::fabs(g[t2])) / gmag * h0;
  if (denom < 1e-9)
    denom = 1e-9;
  double frac = 0.5 + sd / denom;
  return frac < 0.0 ? 0.0 : (frac > 1.0 ? 1.0 : frac);
}

struct Geo {
  BO t;
  AmrPoisson<3, 21> pres;
  double h;
  Index n;
  std::vector<char> fluid;   // cell-center classification (sdf > 0)
  std::vector<Vec<3>> cen;   // world cell centers
};

Geo buildGeo(long N) {
  Geo g;
  unsigned L = 0;
  while ((1L << L) < N)
    ++L;
  g.t = BO(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k)
    g.t.refineIf([](Code, unsigned) { return true; });
  g.h = 1.0 / static_cast<double>(N);
  g.pres.init(g.t, g.h);
  g.pres.setOrigin(Vec<3>{0, 0, 0});
  const double h = g.h;
  g.pres.buildOpenness([h](const Vec<3>& fc, int axis) { return faceFrac(fc, axis, h); });
  g.n = g.t.numLeaves();
  g.fluid.resize(static_cast<std::size_t>(g.n));
  g.cen.resize(static_cast<std::size_t>(g.n));
  for (Index i = 0; i < g.n; ++i) {
    auto b = g.t.bounds(i);
    double s = static_cast<double>(Index(1) << g.t.level(i));
    Vec<3> c{};
    for (int d = 0; d < 3; ++d)
      c[d] = (static_cast<double>(b[0][d]) + 0.5 * s) * g.h;
    g.cen[static_cast<std::size_t>(i)] = c;
    g.fluid[static_cast<std::size_t>(i)] = sdfSphere(c) > 0.0 ? 1 : 0;
  }
  return g;
}

// ABC cell-gradient — verbatim copy of oracle::AmrFlow::gradOf (the operator under test).
double gradOfAbc(const Geo& g, const std::vector<double>& fld, Index i, int c) {
  const double pi = fld[static_cast<std::size_t>(i)];
  double gp = 0, gm = 0;
  int np = 0, nm = 0;
  g.pres.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double alpha) {
    if (axis != c || alpha <= 1e-12)
      return;
    double gg = (dir > 0) ? (fld[static_cast<std::size_t>(j)] - pi) / dist
                          : (pi - fld[static_cast<std::size_t>(j)]) / dist;
    if (dir > 0) {
      gp += gg;
      ++np;
    } else {
      gm += gg;
      ++nm;
    }
  });
  double gpa = np ? gp / np : 0.0, gma = nm ? gm / nm : 0.0;
  return 0.5 * (gpa + gma);
}

// Directional ghost gradient (gpCenterGrad analog on the octree, same-level uniform): central
// where both axis-neighbour CENTERS are fluid; 2nd-order one-sided toward the fluid else
// (2-point fallback when the +/-2 cell is solid too); 0 when sandwiched. Never reads solid p.
double gradOfGhost(const Geo& g, const std::vector<double>& fld, Index i, int c) {
  auto F = [&](Index j) { return fld[static_cast<std::size_t>(j)]; };
  auto fl = [&](Index j) { return j >= 0 && g.fluid[static_cast<std::size_t>(j)]; };
  const Index jp = g.pres.periodicNeighbor(i, c, +1);
  const Index jm = g.pres.periodicNeighbor(i, c, -1);
  const bool ap = fl(jp), am = fl(jm);
  const double h = g.h;
  if (am && ap)
    return (F(jp) - F(jm)) / (2.0 * h);
  if (ap) {
    const Index jpp = g.pres.periodicNeighbor(jp, c, +1);
    return fl(jpp) ? (-3.0 * F(i) + 4.0 * F(jp) - F(jpp)) / (2.0 * h) : (F(jp) - F(i)) / h;
  }
  if (am) {
    const Index jmm = g.pres.periodicNeighbor(jm, c, -1);
    return fl(jmm) ? (3.0 * F(i) - 4.0 * F(jm) + F(jmm)) / (2.0 * h) : (F(i) - F(jm)) / h;
  }
  return 0.0;
}

// Openness-weighted FV divergence — verbatim copy of oracle::AmrFlow::divergence.
double divergenceAbc(const Geo& g, const std::array<std::vector<double>, 3>& vel, Index i) {
  double d = 0.0;
  g.pres.forEachFaceFull(i, [&](Index j, int axis, int dir, double area, double, double alpha) {
    double ui = vel[axis][static_cast<std::size_t>(i)];
    double uj = vel[axis][static_cast<std::size_t>(j)];
    d += alpha * area * dir * 0.5 * (ui + uj);
  });
  return d / g.pres.cellVolume(i);
}

double orderOf(double prev, double cur, long Nprev, long N) {
  return std::log2(prev / cur) / std::log2(static_cast<double>(N) / Nprev);
}

// ---- [A1] gradient ladder at cut cells + gauge test --------------------------------------------
struct A1Result {
  double ordAbc = 0, ordGhost = 0, gaugeAbc = 0, gaugeGhost = 0;
};

A1Result testGradLadder(const std::vector<long>& Ns) {
  std::printf("\n[A1] cell-gradient operators on a smooth P at CUT cells (fluid center, solid\n");
  std::printf("     axis-neighbour); max error vs analytic grad, physical units. gauge = +5 added\n");
  std::printf("     to P on FLUID cells only (solid rows stay decoupled 0, as in the solver).\n");
  std::printf("%5s %12s %6s %12s %6s %12s %12s %10s\n", "N", "abc(gradOf)", "ord", "ghost", "ord",
              "gauge_abc", "gauge_ghost", "openSolid");
  A1Result r;
  double prevA = 0, prevG = 0;
  long prevN = 0;
  for (long N : Ns) {
    Geo g = buildGeo(N);
    std::vector<double> P(static_cast<std::size_t>(g.n), 0.0), Pg = P;
    for (Index i = 0; i < g.n; ++i)
      if (g.fluid[static_cast<std::size_t>(i)]) {
        P[static_cast<std::size_t>(i)] = phiMan(g.cen[static_cast<std::size_t>(i)]);
        Pg[static_cast<std::size_t>(i)] = P[static_cast<std::size_t>(i)] + 5.0;
      }
    double eA = 0, eG = 0, dA = 0, dG = 0;
    long nOpenSolid = 0;  // faces with alpha>1e-12 whose neighbour center is solid
    for (Index i = 0; i < g.n; ++i) {
      if (!g.fluid[static_cast<std::size_t>(i)])
        continue;
      g.pres.forEachFaceFull(i, [&](Index j, int, int, double, double, double alpha) {
        if (alpha > 1e-12 && !g.fluid[static_cast<std::size_t>(j)])
          ++nOpenSolid;
      });
      const Vec<3> ge = phiManGrad(g.cen[static_cast<std::size_t>(i)]);
      for (int c = 0; c < 3; ++c) {
        const bool cutAxis = !g.fluid[static_cast<std::size_t>(g.pres.periodicNeighbor(i, c, +1))] ||
                             !g.fluid[static_cast<std::size_t>(g.pres.periodicNeighbor(i, c, -1))];
        const double ga = gradOfAbc(g, P, i, c);
        const double gg = gradOfGhost(g, P, i, c);
        if (cutAxis) {
          eA = std::max(eA, std::fabs(ga - ge[c]));
          eG = std::max(eG, std::fabs(gg - ge[c]));
        }
        dA = std::max(dA, std::fabs(gradOfAbc(g, Pg, i, c) - ga));
        dG = std::max(dG, std::fabs(gradOfGhost(g, Pg, i, c) - gg));
      }
    }
    const double oA = prevN ? orderOf(prevA, eA, prevN, N) : 0.0;
    const double oG = prevN ? orderOf(prevG, eG, prevN, N) : 0.0;
    std::printf("%5ld %12.3e %6.2f %12.3e %6.2f %12.1e %12.1e %10ld\n", N, eA, oA, eG, oG, dA, dG,
                nOpenSolid);
    prevA = eA;
    prevG = eG;
    prevN = N;
    r.ordAbc = oA;
    r.ordGhost = oG;
    r.gaugeAbc = dA;
    r.gaugeGhost = dG;
  }
  return r;
}

// ---- [A2] constraint truncation on the exact Stokes field --------------------------------------
struct A2Result {
  double ordIb = 0, ordShell = 0;
};

A2Result testConstraint(const std::vector<long>& Ns) {
  std::printf("\n[A2] openness-weighted 1/2-1/2 divergence of the exact solenoidal Stokes field\n");
  std::printf("     (masked to 0 at solid centers). near-IB = fluid cells with a cut/solid face;\n");
  std::printf("     bulk gated on the fixed shell r in [%.2f,%.2f], interior only.\n", R0 + 0.10,
              R0 + 0.22);
  std::printf("%5s %12s %6s %12s %6s\n", "N", "near-IB", "ord", "shell", "ord");
  A2Result r;
  double prevI = 0, prevS = 0;
  long prevN = 0;
  for (long N : Ns) {
    Geo g = buildGeo(N);
    std::array<std::vector<double>, 3> u;
    for (int c = 0; c < 3; ++c)
      u[c].assign(static_cast<std::size_t>(g.n), 0.0);
    for (Index i = 0; i < g.n; ++i)
      if (g.fluid[static_cast<std::size_t>(i)]) {
        Vec<3> v = stokesU(g.cen[static_cast<std::size_t>(i)]);
        for (int c = 0; c < 3; ++c)
          u[c][static_cast<std::size_t>(i)] = v[c];
      }
    double eI = 0, eS = 0;
    for (Index i = 0; i < g.n; ++i) {
      if (!g.fluid[static_cast<std::size_t>(i)])
        continue;
      const Vec<3>& c = g.cen[static_cast<std::size_t>(i)];
      // stokesU is not periodic: keep clear of the box boundary (2h margin).
      bool interior = true;
      for (int d = 0; d < 3; ++d)
        interior = interior && c[d] > 2.0 * g.h && c[d] < 1.0 - 2.0 * g.h;
      if (!interior)
        continue;
      bool nearIb = false;
      g.pres.forEachFaceFull(i, [&](Index j, int, int, double, double, double alpha) {
        if (alpha < 1.0 - 1e-12 || !g.fluid[static_cast<std::size_t>(j)])
          nearIb = true;
      });
      const double d = divergenceAbc(g, u, i);
      if (nearIb) {
        eI = std::max(eI, std::fabs(d));
      } else {
        double dx = c[0] - C0[0], dy = c[1] - C0[1], dz = c[2] - C0[2];
        double rr = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (rr >= R0 + 0.10 && rr < R0 + 0.22)
          eS = std::max(eS, std::fabs(d));
      }
    }
    const double oI = prevN ? orderOf(prevI, eI, prevN, N) : 0.0;
    const double oS = prevN ? orderOf(prevS, eS, prevN, N) : 0.0;
    std::printf("%5ld %12.3e %6.2f %12.3e %6.2f\n", N, eI, oI, eS, oS);
    prevI = eI;
    prevS = eS;
    prevN = N;
    r.ordIb = oI;
    r.ordShell = oS;
  }
  return r;
}

}  // namespace

int main() {
  const std::vector<long> Ns{16, 32, 64, 128};
  A1Result a1 = testGradLadder(Ns);
  A2Result a2 = testConstraint(Ns);

  std::printf("\n==== gates (documenting the measured defect + the candidate fix) ====\n");
  struct Gate {
    const char* name;
    bool ok;
  } gates[] = {
      {"A1 abc gradOf cut-cell order      <= 0.5  (defect: O(1) or worse)", a1.ordAbc <= 0.5},
      {"A1 abc gradOf gauge-DEPENDENT     (reads decoupled solid p)", a1.gaugeAbc > 1e-3},
      {"A1 ghost gradient cut-cell order  >= 1.7", a1.ordGhost >= 1.7},
      {"A1 ghost gradient gauge-exact     (< 1e-12)", a1.gaugeGhost < 1e-12},
      {"A2 constraint near-IB order       <= 0.5  (defect: O(1) truncation)", a2.ordIb <= 0.5},
      {"A2 constraint bulk shell order    >= 1.7", a2.ordShell >= 1.7},
  };
  int npass = 0;
  for (const auto& gt : gates) {
    std::printf("  %s  %s\n", gt.ok ? "PASS" : "FAIL", gt.name);
    npass += gt.ok ? 1 : 0;
  }
  std::printf("%d/6 gates passed\n", npass);
  return npass == 6 ? 0 : 1;
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping\n");
  return 0;
}
#endif

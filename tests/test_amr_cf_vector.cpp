// C/F interface schemes for the collocated flow operators (cf_scheme.hpp): a-priori truncation
// on a graded (2:1) mesh with smooth manufactured fields, measured at the level-boundary rows.
//
//  (1) the L delta (buildCfLapDelta, factor 1) must reproduce the VALIDATED
//      applyLaplacianQuad − applyLaplacian to round-off (anti-drift lock vs P5b);
//  (2) D (½/½ face-average divergence): standard is low-order at C/F rows (the average is
//      normally offset from the face); the scheme's distance-weighted + tangential-corrected
//      value restores ~2nd order. The quad D stays exactly conservative (Σ V·D = 0, periodic).
//  (3) G (ABC cell gradient): standard ½/½ side recombination misses the cell center at C/F
//      rows; the scheme's reweighting + coarse* substitution restores ~2nd order.
//  (4) momentum ∇² (the velocity operator): the tangential-only flux correction restores
//      ~2nd order at C/F rows (the P5b result, now on the velocity path).
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/cf_scheme.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

constexpr double kPi = 3.14159265358979323846;

struct Geo {
  BO t;
  AmrPoisson<3, 21> ap;
  double h;
  Index n;
  std::vector<Vec<3>> cen;
  std::vector<char> cfRow;  // 1 = adjacent to a 2:1 face
};

Geo buildGeo(long N) {
  Geo g;
  unsigned L = 0;
  while ((1L << L) < N)
    ++L;
  g.t = BO(IVec<3>{1, 1, 1}, L);
  // uniform to level 1, then the lower octant to level 0 (finest), 2:1 balanced.
  for (unsigned k = 0; k + 1 < L; ++k)
    g.t.refineIf([](Code, unsigned l) { return l > 1; });
  const long half = N / 2;
  g.t.refineIf([&](Code c, unsigned l) {
    if (l != 1)
      return false;
    auto o = BO::M::from_code(c).decode();
    return o[0] < half && o[1] < half && o[2] < half;
  });
  g.t.balance2to1();
  g.h = 1.0 / static_cast<double>(N);
  g.ap.init(g.t, g.h);
  g.n = g.t.numLeaves();
  g.cen.resize(static_cast<std::size_t>(g.n));
  g.cfRow.assign(static_cast<std::size_t>(g.n), 0);
  for (Index i = 0; i < g.n; ++i) {
    auto b = g.t.bounds(i);
    double s = static_cast<double>(Index(1) << g.t.level(i));
    for (int d = 0; d < 3; ++d)
      g.cen[static_cast<std::size_t>(i)][d] = (static_cast<double>(b[0][d]) + 0.5 * s) * g.h;
    const unsigned Li = g.t.level(i);
    g.ap.forEachFaceFull(i, [&](Index j, int, int, double, double, double) {
      if (g.t.level(j) != Li)
        g.cfRow[static_cast<std::size_t>(i)] = 1;
    });
  }
  return g;
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
double phiManLap(const Vec<3>& p) {
  return -2.0 * (2.0 * kPi) * (2.0 * kPi) * phiMan(p);
}
Vec<3> velMan(const Vec<3>& p) {
  const double tp = 2.0 * kPi;
  return {std::sin(tp * p[0]) * std::cos(tp * p[1]), std::sin(tp * p[1]) * std::cos(tp * p[2]),
          std::sin(tp * p[2]) * std::cos(tp * p[0])};
}
double velManDiv(const Vec<3>& p) {
  const double tp = 2.0 * kPi;
  return tp * (std::cos(tp * p[0]) * std::cos(tp * p[1]) +
               std::cos(tp * p[1]) * std::cos(tp * p[2]) +
               std::cos(tp * p[2]) * std::cos(tp * p[0]));
}

// Standard ½/½ face-average divergence (the oracle::AmrFlow::divergence form, α=1).
void divStd(const Geo& g, const std::array<std::vector<double>, 3>& u, std::vector<double>& d) {
  d.assign(static_cast<std::size_t>(g.n), 0.0);
  for (Index i = 0; i < g.n; ++i) {
    double acc = 0.0;
    g.ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double area, double, double) {
      acc += area * dir * 0.5 *
             (u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(i)] +
              u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(j)]);
    });
    d[static_cast<std::size_t>(i)] = acc / g.ap.cellVolume(i);
  }
}

// Standard ABC cell gradient (the oracle::AmrFlow::gradOf form, α=1).
double gradStd(const Geo& g, const std::vector<double>& f, Index i, int c) {
  const double fi = f[static_cast<std::size_t>(i)];
  double gp = 0, gm = 0;
  int np = 0, nm = 0;
  g.ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double) {
    if (axis != c)
      return;
    double gg = (dir > 0) ? (f[static_cast<std::size_t>(j)] - fi) / dist
                          : (fi - f[static_cast<std::size_t>(j)]) / dist;
    if (dir > 0) {
      gp += gg;
      ++np;
    } else {
      gm += gg;
      ++nm;
    }
  });
  return 0.5 * ((np ? gp / np : 0.0) + (nm ? gm / nm : 0.0));
}

double orderOf(double prev, double cur, long Nprev, long N) {
  return std::log2(prev / cur) / std::log2(static_cast<double>(N) / Nprev);
}

void run() {
  auto all = [](Index) { return true; };
  std::printf("%5s | %10s %6s %10s %6s | %10s %6s %10s %6s | %10s %6s %10s %6s\n", "N", "D_std",
              "ord", "D_quad", "ord", "G_std", "ord", "G_quad", "ord", "L_std", "ord", "L_quad",
              "ord");
  double pDs = 0, pDq = 0, pGs = 0, pGq = 0, pLs = 0, pLq = 0;
  long pN = 0;
  double oDq = 0, oGq = 0, oLq = 0, oDs = 0, oGs = 0, oLs = 0;
  for (long N : {16L, 32L, 64L}) {
    Geo g = buildGeo(N);
    // (1) L delta == applyLaplacianQuad − applyLaplacian (bit-parity vs the validated P5b impl).
    std::vector<double> phi(static_cast<std::size_t>(g.n));
    for (Index i = 0; i < g.n; ++i)
      phi[static_cast<std::size_t>(i)] = phiMan(g.cen[static_cast<std::size_t>(i)]);
    {
      std::vector<double> ls, lq, dl(static_cast<std::size_t>(g.n), 0.0);
      g.ap.applyLaplacian(phi, ls);
      g.ap.applyLaplacianQuad(phi, lq);
      CfCsr cs = buildCfLapDelta(g.ap, g.t, 1.0, all, all, CfScheme::quadratic);
      cfApplyHost(cs, phi, dl);
      double scale = 0.0, dmax = 0.0;
      for (Index i = 0; i < g.n; ++i) {
        scale = std::max(scale, std::fabs(lq[static_cast<std::size_t>(i)]));
        dmax = std::max(dmax, std::fabs(dl[static_cast<std::size_t>(i)] -
                                        (lq[static_cast<std::size_t>(i)] -
                                         ls[static_cast<std::size_t>(i)])));
      }
      PECLET_CORE_CHECK(dmax < 1e-12 * scale);  // the delta IS the P5b quad correction
    }
    // fields
    std::array<std::vector<double>, 3> u;
    for (int c = 0; c < 3; ++c) {
      u[static_cast<std::size_t>(c)].resize(static_cast<std::size_t>(g.n));
      for (Index i = 0; i < g.n; ++i)
        u[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] =
            velMan(g.cen[static_cast<std::size_t>(i)])[c];
    }
    // (2) divergence
    std::vector<double> ds;
    divStd(g, u, ds);
    std::vector<double> dq = ds;
    CfCompCsr dd = buildCfDivDelta(g.ap, g.t, all, all, CfScheme::quadratic);
    cfApplyCompHost(dd, u, dq);
    double eDs = 0, eDq = 0, consQ = 0;
    for (Index i = 0; i < g.n; ++i) {
      const double ex = velManDiv(g.cen[static_cast<std::size_t>(i)]);
      consQ += g.ap.cellVolume(i) * dq[static_cast<std::size_t>(i)];
      if (!g.cfRow[static_cast<std::size_t>(i)])
        continue;
      eDs = std::max(eDs, std::fabs(ds[static_cast<std::size_t>(i)] - ex));
      eDq = std::max(eDq, std::fabs(dq[static_cast<std::size_t>(i)] - ex));
    }
    PECLET_CORE_CHECK(std::fabs(consQ) < 1e-10);  // quad D exactly conservative (periodic)
    // (3) gradient
    std::array<CfCsr, 3> gd = buildCfGradDelta(g.ap, g.t, all, all, CfScheme::quadratic);
    std::array<std::vector<double>, 3> gq;
    for (int c = 0; c < 3; ++c) {
      gq[static_cast<std::size_t>(c)].assign(static_cast<std::size_t>(g.n), 0.0);
      cfApplyHost(gd[static_cast<std::size_t>(c)], phi, gq[static_cast<std::size_t>(c)]);
    }
    double eGs = 0, eGq = 0;
    for (Index i = 0; i < g.n; ++i) {
      if (!g.cfRow[static_cast<std::size_t>(i)])
        continue;
      const Vec<3> ge = phiManGrad(g.cen[static_cast<std::size_t>(i)]);
      for (int c = 0; c < 3; ++c) {
        const double gs = gradStd(g, phi, i, c);
        eGs = std::max(eGs, std::fabs(gs - ge[c]));
        eGq = std::max(eGq, std::fabs(gs + gq[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] -
                                      ge[c]));
      }
    }
    // (4) momentum ∇² (the velocity operator): SOLUTION-level order for the Helmholtz solve
    // (I − ∇²)u = f, manufactured. The quad flux is conservative with an O(1) POINTWISE row
    // truncation at C/F faces (the sample point of the two-point C/F gradient is normally
    // offset from the face) — the flux error telescopes and the SOLUTION is 2nd order (the P5b
    // result); pointwise operator truncation is the wrong metric here.
    CfCsr lm = buildCfLapDelta(g.ap, g.t, 1.0, all, all, CfScheme::quadratic);
    auto solveHelmholtz = [&](bool quad) {
      // u solved by unpreconditioned-BiCGStab-with-Jacobi on (I − L[ − Δ]) u = b.
      const std::size_t ns = static_cast<std::size_t>(g.n);
      std::vector<double> b(ns);
      for (Index i = 0; i < g.n; ++i)
        b[static_cast<std::size_t>(i)] = phiMan(g.cen[static_cast<std::size_t>(i)]) -
                                         phiManLap(g.cen[static_cast<std::size_t>(i)]);
      auto applyA = [&](const std::vector<double>& x, std::vector<double>& y) {
        g.ap.applyLaplacian(x, y);
        if (quad)
          cfApplyHost(lm, x, y);
        for (std::size_t p = 0; p < ns; ++p)
          y[p] = x[p] - y[p];
      };
      auto dot = [&](const std::vector<double>& a, const std::vector<double>& c) {
        double s = 0;
        for (std::size_t p = 0; p < ns; ++p)
          s += a[p] * c[p];
        return s;
      };
      std::vector<double> x(ns, 0.0), r(ns), rh(ns), p(ns, 0.0), v(ns, 0.0), s(ns), tt(ns);
      applyA(x, r);
      for (std::size_t q = 0; q < ns; ++q)
        r[q] = b[q] - r[q];
      rh = r;
      const double r0 = std::sqrt(dot(r, r));
      double rho = 1, alpha = 1, omega = 1;
      for (int it = 0; it < 4000; ++it) {
        const double rhoN = dot(rh, r);
        if (rhoN == 0)
          break;
        const double beta = (rhoN / rho) * (alpha / omega);
        for (std::size_t q = 0; q < ns; ++q)
          p[q] = r[q] + beta * (p[q] - omega * v[q]);
        applyA(p, v);
        alpha = rhoN / dot(rh, v);
        for (std::size_t q = 0; q < ns; ++q)
          s[q] = r[q] - alpha * v[q];
        applyA(s, tt);
        const double t2 = dot(tt, tt);
        omega = (t2 != 0) ? dot(tt, s) / t2 : 0;
        for (std::size_t q = 0; q < ns; ++q) {
          x[q] += alpha * p[q] + omega * s[q];
          r[q] = s[q] - omega * tt[q];
        }
        if (std::sqrt(dot(r, r)) < 1e-11 * r0)
          break;
        rho = rhoN;
        if (omega == 0)
          break;
      }
      return x;
    };
    std::vector<double> uS = solveHelmholtz(false), uQ = solveHelmholtz(true);
    double eLs = 0, eLq = 0;
    for (Index i = 0; i < g.n; ++i) {
      if (!g.cfRow[static_cast<std::size_t>(i)])
        continue;
      const double ex = phiMan(g.cen[static_cast<std::size_t>(i)]);
      eLs = std::max(eLs, std::fabs(uS[static_cast<std::size_t>(i)] - ex));
      eLq = std::max(eLq, std::fabs(uQ[static_cast<std::size_t>(i)] - ex));
    }
    // (5) uf face field: value truncation at the 2:1 sub-faces. uf_k = ½(u_i+u_j) − (φ₊−φ₋)/d
    // samples the face value; standard is normally offset at C/F (O(h)); the scheme's
    // distance-weighted + coarse*-substituted value is ~2nd order at the sub-face centroid.
    {
      CfUfDelta ufd = buildCfUfDelta(g.ap, g.t, all, CfScheme::quadratic);
      Index nSlots = 0;
      for (Index i = 0; i < g.n; ++i)
        g.ap.forEachFaceFull(i, [&](Index, int, int, double, double, double) { ++nSlots; });
      std::vector<double> ufS(static_cast<std::size_t>(nSlots));
      std::vector<Vec<3>> fc(static_cast<std::size_t>(nSlots));
      std::vector<int8_t> fAxis(static_cast<std::size_t>(nSlots));
      std::vector<char> fCf(static_cast<std::size_t>(nSlots), 0);
      Index slot = 0;
      for (Index i = 0; i < g.n; ++i) {
        const unsigned Li = g.t.level(i);
        g.ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double) {
          const double ui = u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(i)];
          const double uj = u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(j)];
          const double gphi = (dir > 0) ? (phi[static_cast<std::size_t>(j)] -
                                           phi[static_cast<std::size_t>(i)]) /
                                              dist
                                        : (phi[static_cast<std::size_t>(i)] -
                                           phi[static_cast<std::size_t>(j)]) /
                                              dist;
          ufS[static_cast<std::size_t>(slot)] = 0.5 * (ui + uj) - gphi;
          const unsigned Lj = g.t.level(j);
          fCf[static_cast<std::size_t>(slot)] = (Lj != Li) ? 1 : 0;
          fAxis[static_cast<std::size_t>(slot)] = static_cast<int8_t>(axis);
          // sub-face centroid = the FINER cell's face center toward the other cell.
          const Index fine = (Lj < Li) ? j : i;
          const double sgn = (fine == i) ? 1.0 : -1.0;  // dir points i→j
          Vec<3> c = g.cen[static_cast<std::size_t>(fine)];
          c[axis] += sgn * dir * 0.5 * g.ap.cellWidth(fine);
          fc[static_cast<std::size_t>(slot)] = c;
          ++slot;
        });
      }
      // The φ-gradient part keeps the P5b flux form (sampled at the 2-point midpoint, an O(h)
      // normal offset at C/F — conservative, telescoping, and VANISHING at steady state where
      // φ→0). So the whole uf is O(h) at C/F faces during transients, while the face-AVERAGE
      // part — the steady advecting velocity — must be ~2nd order. Gate both separately.
      std::vector<double> ufQ = ufS;
      cfApplyCompHost(ufd.vel, u, ufQ);
      cfApplyHost(ufd.phi, phi, ufQ);
      std::vector<double> avS(static_cast<std::size_t>(nSlots), 0.0), avQ;
      {
        Index k = 0;
        for (Index i = 0; i < g.n; ++i)
          g.ap.forEachFaceFull(i, [&](Index j, int axis, int, double, double, double) {
            avS[static_cast<std::size_t>(k++)] =
                0.5 * (u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(i)] +
                       u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(j)]);
          });
        avQ = avS;
        cfApplyCompHost(ufd.vel, u, avQ);
      }
      double eUs = 0, eUq = 0, eAs = 0, eAq = 0;
      for (Index k = 0; k < nSlots; ++k) {
        if (!fCf[static_cast<std::size_t>(k)])
          continue;
        const Vec<3>& c = fc[static_cast<std::size_t>(k)];
        const int a = fAxis[static_cast<std::size_t>(k)];
        const double exA = velMan(c)[a];
        const double ex = exA - phiManGrad(c)[a];
        eUs = std::max(eUs, std::fabs(ufS[static_cast<std::size_t>(k)] - ex));
        eUq = std::max(eUq, std::fabs(ufQ[static_cast<std::size_t>(k)] - ex));
        eAs = std::max(eAs, std::fabs(avS[static_cast<std::size_t>(k)] - exA));
        eAq = std::max(eAq, std::fabs(avQ[static_cast<std::size_t>(k)] - exA));
      }
      static double pUq = 0, pAq = 0, pAs = 0;
      static double oUq = 0, oAq = 0, oAs = 0;
      oUq = pN ? orderOf(pUq, eUq, pN, N) : 0;
      oAq = pN ? orderOf(pAq, eAq, pN, N) : 0;
      oAs = pN ? orderOf(pAs, eAs, pN, N) : 0;
      std::printf("      | uf: whole std %.3e quad %.3e ord %5.2f | avg std %.3e ord %5.2f "
                  "quad %.3e ord %5.2f\n",
                  eUs, eUq, oUq, eAs, oAs, eAq, oAq);
      pUq = eUq;
      pAq = eAq;
      pAs = eAs;
      if (N == 64) {
        PECLET_CORE_CHECK(oAq >= 1.7);         // steady advecting velocity ~2nd order
        PECLET_CORE_CHECK(oAs <= oAq - 0.5);   // standard average is lower order
        PECLET_CORE_CHECK(oUq >= 0.9);         // whole uf ≥ O(h) (φ part, transient-only)
      }
    }
    oDs = pN ? orderOf(pDs, eDs, pN, N) : 0;
    oDq = pN ? orderOf(pDq, eDq, pN, N) : 0;
    oGs = pN ? orderOf(pGs, eGs, pN, N) : 0;
    oGq = pN ? orderOf(pGq, eGq, pN, N) : 0;
    oLs = pN ? orderOf(pLs, eLs, pN, N) : 0;
    oLq = pN ? orderOf(pLq, eLq, pN, N) : 0;
    std::printf("%5ld | %10.3e %6.2f %10.3e %6.2f | %10.3e %6.2f %10.3e %6.2f | %10.3e %6.2f "
                "%10.3e %6.2f\n",
                N, eDs, oDs, eDq, oDq, eGs, oGs, eGq, oGq, eLs, oLs, eLq, oLq);
    pDs = eDs;
    pDq = eDq;
    pGs = eGs;
    pGq = eGq;
    pLs = eLs;
    pLq = eLq;
    pN = N;
  }
  // Gates: the scheme restores ~2nd order at C/F rows for all three operators; the standard
  // treatment is measurably lower order there.
  PECLET_CORE_CHECK(oDq >= 1.7);
  PECLET_CORE_CHECK(oGq >= 1.7);
  PECLET_CORE_CHECK(oLq >= 1.7);
  PECLET_CORE_CHECK(oDs <= oDq - 0.5);
  PECLET_CORE_CHECK(oGs <= oGq - 0.5);
  PECLET_CORE_CHECK(oLs <= oLq - 0.5);
}

}  // namespace

int main() {
  run();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping cf-scheme test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

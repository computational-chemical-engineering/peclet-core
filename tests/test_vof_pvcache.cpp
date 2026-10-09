// Design G (flow doc/vof_curvature_cost_design.md §5.1, §6 WO-1): the cost kernels of tier 3 of the
// curvature cascade, on the default execution space (CUDA / HIP / OpenMP) and on the host space.
//
//   T1  pvFitTerm (now pvTermNormal -> plicPolygon -> pvTermPolygon) == a frozen verbatim copy of
//       the pre-split pvFitTerm, BITWISE (ok; w, B, s[6] when ok), 10^5 cases.
//   T2  the cache pattern: kernel 1 builds every case's PvPolygon into a View, kernel 2 evaluates
//       pvOutsideSupport -> pvTermNormal -> pvTermPolygon from it. BITWISE against T1's reference;
//       prefilter false skips (skipped although the reference accepted) must be 0, and it must skip
//       >= 50 % of the cases the reference rejects for w == 0 (outside the support).
//   T3  pvFitAccumLower and the 27-lane pvFitAccumEntry team kernel against pvFitAccum: lower
//       triangle, diagonal, b, npoly bitwise; pvFitSolve bitwise on full-rank systems and on
//       systems forced rank deficient (npoly 3-5, the reduced model).
//   T4  the moment terms (pvMomentsBuild + pvTermMoments, "V6") against the reference: accept flags
//       agree away from the thresholds (r = dW, s0 = 1e-14, cj = cosMin; within 1e-9 relative is
//       "near" and reported); on accepted cases |ds_k| <= 1e-13 (|s_k| + s0 3.5^deg k),
//       |dB| <= 1e-13 (|B| + 3.5 s0), |dw| <= 1e-14. pvFrameMoments against curvFallbackFrame:
//       frame vectors bitwise, |d org| <= 1e-14 (physical).
//   T5  analytic moments: the unit square, the same plane on the metric h = (2, 1, 0.5), and a
//       tilted plane against a fine triangulated quadrature of its polygon.
//
// The terms of the moment form carry the orientation sign of nothing (s0 = cj a > 0), while
// pvFitTerm's Green's-theorem moments carry the projected polygon's orientation; the quadratic
// normal equations cancel a global sign exactly (polygonMoments2d's note), so T4 compares s and B
// after multiplying the reference by sign(s0_ref) sign(s0_v6).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/vof/curvature.hpp"

using namespace peclet::core::vof;

namespace {

// --- the pre-split pvFitTerm (core ecedb9b), verbatim apart from its name ------------------------
KOKKOS_INLINE_FUNCTION bool pvFitTermReference(PvTerm& t, double mx, double my, double mz,
                                               double alpha, const double off[3],
                                               const double org[3], const double t1[3],
                                               const double t2[3], const double nn[3], double dW,
                                               double cosMin, const VofMetric& g) {
  t.ok = false;
  const double n2 = mx * mx + my * my + mz * mz;
  if (!(n2 > 0.0))
    return false;
  // Phase 3 (V2.4): the polygon's own normal enters the fit as a PHYSICAL direction (the frame is
  // physical), so it is pulled back through H^-1 and renormalized. At g.h = {1,1,1} the pullback
  // is `m/1.0` and `invn` is the same reciprocal square root as before — bitwise.
  const double mi[3] = {mx / g.h[0], my / g.h[1], mz / g.h[2]};
  const double invn = 1.0 / Kokkos::sqrt(mi[0] * mi[0] + mi[1] * mi[1] + mi[2] * mi[2]);
  const double np[3] = {(mi[0] * t1[0] + mi[1] * t1[1] + mi[2] * t1[2]) * invn,
                        (mi[0] * t2[0] + mi[1] * t2[1] + mi[2] * t2[2]) * invn,
                        (mi[0] * nn[0] + mi[1] * nn[1] + mi[2] * nn[2]) * invn};
  if (!(np[2] > cosMin))
    return false;

  double v[8][3];
  const int nv = plicPolygon(mx, my, mz, alpha, v);
  if (nv < 3)
    return false;

  // cell-local [0,1]^3 -> target-centred cell units -> the fit frame
  double xy[8][2];
  double zc = 0.0;
  double px = 0.0, py = 0.0;
  for (int k = 0; k < nv; ++k) {
    const double Xi[3] = {off[0] + v[k][0] - 0.5 - org[0], off[1] + v[k][1] - 0.5 - org[1],
                          off[2] + v[k][2] - 0.5 - org[2]};
    double X[3];
    g.toPhys(Xi, X);  // index displacement -> physical (identity at g.h = {1,1,1})
    xy[k][0] = X[0] * t1[0] + X[1] * t1[1] + X[2] * t1[2];
    xy[k][1] = X[0] * t2[0] + X[1] * t2[1] + X[2] * t2[2];
    const double z = X[0] * nn[0] + X[1] * nn[1] + X[2] * nn[2];
    px += xy[k][0];
    py += xy[k][1];
    zc += z;
  }
  const double invv = 1.0 / static_cast<double>(nv);
  px *= invv;
  py *= invv;
  zc *= invv;

  double s[6];
  polygonMoments2d(xy, nv, s);
  if (!(Kokkos::fabs(s[0]) > 1e-14))
    return false;

  // the polygon's own plane in the fit frame: z' = b0 + b1 x' + b2 y'
  const double b1 = -np[0] / np[2], b2 = -np[1] / np[2];
  const double b0 = zc - b1 * px - b2 * py;

  const double r = Kokkos::sqrt(px * px + py * py + zc * zc);
  const double w = wendlandWeight(r, dW);
  if (!(w > 0.0))
    return false;

  const double B = b0 * s[0] + b1 * s[1] + b2 * s[2];
  t.w = w;
  t.B = B;
  for (int i = 0; i < 6; ++i)
    t.s[i] = s[i];
  t.ok = true;
  return true;
}

// --- flow's curvFallbackFrame (flow src/vof/curvature_field.hpp, fedadb2), verbatim -------------
KOKKOS_INLINE_FUNCTION bool curvFallbackFrameReference(double m0, double m1, double m2,
                                                       double alpha, const VofMetric& g,
                                                       double nn[3], double t1[3], double t2[3],
                                                       double org[3]) {
  const double mi[3] = {m0, m1, m2};
  nn[0] = nn[1] = nn[2] = 0.0;
  if (!(vofPhysNormalInv(mi, g, nn) > 0.0))
    return false;
  curvFrame(nn, t1, t2);
  double v[8][3], ctr[3], area;
  const int nv = plicPolygon(m0, m1, m2, alpha, v);
  polygonAreaCentroid(v, nv, ctr, area);
  org[0] = ctr[0] - 0.5;
  org[1] = ctr[1] - 0.5;
  org[2] = ctr[2] - 0.5;
  return true;
}

// --- the reference's rejection reason and threshold quantities (classification only; the same
// expressions as above, never compared bitwise) -----------------------------------------------
struct Probe {
  int reason;  // 0 accepted, 1 zero normal, 2 cosMin, 3 nv < 3, 4 s0, 5 w == 0
  double cj, s0, r;
};
KOKKOS_INLINE_FUNCTION Probe probeReference(double mx, double my, double mz, double alpha,
                                            const double off[3], const double org[3],
                                            const double t1[3], const double t2[3],
                                            const double nn[3], double dW, double cosMin,
                                            const VofMetric& g) {
  Probe p{0, 0.0, 0.0, 0.0};
  const double n2 = mx * mx + my * my + mz * mz;
  if (!(n2 > 0.0)) {
    p.reason = 1;
    return p;
  }
  const double mi[3] = {mx / g.h[0], my / g.h[1], mz / g.h[2]};
  const double invn = 1.0 / Kokkos::sqrt(mi[0] * mi[0] + mi[1] * mi[1] + mi[2] * mi[2]);
  p.cj = (mi[0] * nn[0] + mi[1] * nn[1] + mi[2] * nn[2]) * invn;
  if (!(p.cj > cosMin)) {
    p.reason = 2;
    return p;
  }
  double v[8][3];
  const int nv = plicPolygon(mx, my, mz, alpha, v);
  if (nv < 3) {
    p.reason = 3;
    return p;
  }
  double xy[8][2];
  double zc = 0.0, px = 0.0, py = 0.0;
  for (int k = 0; k < nv; ++k) {
    const double Xi[3] = {off[0] + v[k][0] - 0.5 - org[0], off[1] + v[k][1] - 0.5 - org[1],
                          off[2] + v[k][2] - 0.5 - org[2]};
    double X[3];
    g.toPhys(Xi, X);
    xy[k][0] = X[0] * t1[0] + X[1] * t1[1] + X[2] * t1[2];
    xy[k][1] = X[0] * t2[0] + X[1] * t2[1] + X[2] * t2[2];
    zc += X[0] * nn[0] + X[1] * nn[1] + X[2] * nn[2];
    px += xy[k][0];
    py += xy[k][1];
  }
  const double invv = 1.0 / static_cast<double>(nv);
  px *= invv;
  py *= invv;
  zc *= invv;
  double s[6];
  polygonMoments2d(xy, nv, s);
  p.s0 = Kokkos::fabs(s[0]);
  p.r = Kokkos::sqrt(px * px + py * py + zc * zc);
  if (!(p.s0 > 1e-14)) {
    p.reason = 4;
    return p;
  }
  if (!(wendlandWeight(p.r, dW) > 0.0))
    p.reason = 5;
  return p;
}

constexpr int kSide = 5, kCells = kSide * kSide * kSide, kStencils = 800;
constexpr int kCases = kStencils * kCells;  // 100000

struct Stencil {
  double org[3], t1[3], t2[3], nn[3], dW, cosMin;
  VofMetric g;
};
struct Cell {
  double mx, my, mz, alpha, off[3];
};

struct Rng {  // splitmix64: fixed seed, identical sequence on every host
  std::uint64_t s;
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double uni() { return (next() >> 11) * 0x1.0p-53; }  // [0,1)
  double sym() { return 2.0 * uni() - 1.0; }           // [-1,1)
};

void unit(double v[3]) {
  const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  for (int a = 0; a < 3; ++a)
    v[a] /= n;
}

// The random-case generator of test_vof_pvfit.cpp (planted rejections, random metrics), verbatim.
void makeCases(std::vector<Stencil>& st, std::vector<Cell>& ce) {
  Rng r{20261002ULL};
  st.resize(kStencils);
  ce.resize(kCases);
  for (int s = 0; s < kStencils; ++s) {
    Stencil& S = st[s];
    double nn[3] = {r.sym(), r.sym(), r.sym()};
    unit(nn);
    // t1 = any unit vector orthogonal to nn, t2 = nn x t1
    double a[3] = {1, 0, 0};
    if (std::fabs(nn[0]) > 0.6)
      a[0] = 0, a[1] = 1;
    const double d = a[0] * nn[0] + a[1] * nn[1] + a[2] * nn[2];
    double t1[3] = {a[0] - d * nn[0], a[1] - d * nn[1], a[2] - d * nn[2]};
    unit(t1);
    const double t2[3] = {nn[1] * t1[2] - nn[2] * t1[1], nn[2] * t1[0] - nn[0] * t1[2],
                          nn[0] * t1[1] - nn[1] * t1[0]};
    for (int k = 0; k < 3; ++k) {
      S.nn[k] = nn[k];
      S.t1[k] = t1[k];
      S.t2[k] = t2[k];
      S.org[k] = 0.5 * r.sym();
    }
    S.dW = (s % 50 == 7) ? 0.0 : 1.5 + 2.0 * r.uni();  // a few unweighted stencils (d <= 0)
    const double cm[4] = {0.0, 0.2, 0.5, -1.0};
    S.cosMin = cm[s % 4];
    if (s % 3 != 0)  // two thirds anisotropic
      for (int k = 0; k < 3; ++k)
        S.g.h[k] = 0.5 + 1.5 * r.uni();
    for (int k = 0; k < kCells; ++k) {
      Cell& C = ce[s * kCells + k];
      C.off[0] = k % kSide - 2;
      C.off[1] = (k / kSide) % kSide - 2;
      C.off[2] = k / (kSide * kSide) - 2;
      const int kind = static_cast<int>(r.next() % 20);
      if (kind == 0) {  // degenerate normal
        C.mx = C.my = C.mz = 0.0;
      } else if (kind <= 3) {  // random direction (many fall below cosMin)
        C.mx = r.sym(), C.my = r.sym(), C.mz = r.sym();
      } else {  // near the target normal, in index space
        C.mx = nn[0] * S.g.h[0] + 0.3 * r.sym();
        C.my = nn[1] * S.g.h[1] + 0.3 * r.sym();
        C.mz = nn[2] * S.g.h[2] + 0.3 * r.sym();
      }
      if (kind == 1) {  // plane well outside the cube: no polygon
        C.alpha = 3.0 + r.uni();
      } else {  // plane through a random point of the cell
        const double p[3] = {r.uni(), r.uni(), r.uni()};
        C.alpha = C.mx * p[0] + C.my * p[1] + C.mz * p[2];
      }
    }
  }
}

template <class V>
auto toHost(const V& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  using T = typename V::non_const_value_type;
  return std::vector<T>(h.data(), h.data() + h.extent(0));
}

// The reference's mathematics in long double (host only), from the same double polygon: the
// "exact" value T4 measures the moment terms against where the double reference itself is
// ill-conditioned (b1 = -np0/np2 amplifies its rounding by ~1/cj as the polygon turns edge-on).
struct ExactTerm {
  bool ok;
  long double w, B, s[6];
};
ExactTerm exactTerm(const Cell& c, const Stencil& T) {
  using R = long double;
  ExactTerm e{false, 0, 0, {0, 0, 0, 0, 0, 0}};
  const VofMetric& g = T.g;
  const R mi[3] = {R(c.mx) / g.h[0], R(c.my) / g.h[1], R(c.mz) / g.h[2]};
  const R n2 = mi[0] * mi[0] + mi[1] * mi[1] + mi[2] * mi[2];
  if (!(n2 > 0))
    return e;
  const R invn = 1 / std::sqrt(n2);
  R np[3];
  for (int q = 0; q < 3; ++q) {
    const double* ax = q == 0 ? T.t1 : (q == 1 ? T.t2 : T.nn);
    np[q] = (mi[0] * ax[0] + mi[1] * ax[1] + mi[2] * ax[2]) * invn;
  }
  double v[8][3];
  const int nv = plicPolygon(c.mx, c.my, c.mz, c.alpha, v);
  if (nv < 3)
    return e;
  R xy[8][2], zc = 0, px = 0, py = 0;
  for (int k = 0; k < nv; ++k) {
    R X[3];
    for (int a = 0; a < 3; ++a)
      X[a] = (R(c.off[a]) + v[k][a] - R(0.5) - T.org[a]) * g.h[a];
    xy[k][0] = X[0] * T.t1[0] + X[1] * T.t1[1] + X[2] * T.t1[2];
    xy[k][1] = X[0] * T.t2[0] + X[1] * T.t2[1] + X[2] * T.t2[2];
    zc += X[0] * T.nn[0] + X[1] * T.nn[1] + X[2] * T.nn[2];
    px += xy[k][0];
    py += xy[k][1];
  }
  px /= nv;
  py /= nv;
  zc /= nv;
  R* s = e.s;
  for (int k = 0; k < nv; ++k) {  // polygonMoments2d in long double
    const int k1 = (k + 1 == nv) ? 0 : k + 1;
    const R x0 = xy[k][0], y0 = xy[k][1], x1 = xy[k1][0], y1 = xy[k1][1];
    const R dx = x1 - x0, dy = y1 - y0;
    if (dy == 0)
      continue;
    s[0] += dy * R(0.5) * (x0 + x1);
    s[1] += dy * (x0 * x0 + x0 * x1 + x1 * x1) / 6;
    s[2] += dy * (2 * x0 * y0 + x0 * y1 + x1 * y0 + 2 * x1 * y1) / 6;
    s[3] += dy * (x0 + x1) * (x0 * x0 + x1 * x1) / 12;
    s[4] += dy * R(0.5) *
            (x0 * x0 * y0 + R(0.5) * x0 * x0 * dy + x0 * dx * y0 + R(2) / 3 * x0 * dx * dy +
             dx * dx * y0 / 3 + R(0.25) * dx * dx * dy);
    s[5] += dy * (x0 * y0 * y0 + x0 * y0 * dy + x0 * dy * dy / 3 + R(0.5) * dx * y0 * y0 +
                  R(2) / 3 * dx * y0 * dy + R(0.25) * dx * dy * dy);
  }
  const R b1 = -np[0] / np[2], b2 = -np[1] / np[2];
  const R b0 = zc - b1 * px - b2 * py;
  e.B = b0 * s[0] + b1 * s[1] + b2 * s[2];
  e.w = wendlandWeight(static_cast<double>(std::sqrt(px * px + py * py + zc * zc)), T.dW);
  e.ok = true;
  return e;
}

struct Frame {
  double nn[3], t1[3], t2[3], org[3];
  int ok;
};

struct Result {
  std::vector<PvTerm> ref, split, cache, mom;
  std::vector<Probe> probe;
  std::vector<int> skip;
  std::vector<PvFit> full, lower, entry, fullR, lowerR, entryR;  // R: rank-deficient prefixes
  std::vector<Frame> frRef, frMom;
};

// the number of accepted terms kept by stencil s's forced rank-deficient fit
KOKKOS_INLINE_FUNCTION int deficientCount(int s) {
  return 3 + s % 3;
}

template <class Exec>
Result run(const std::vector<Stencil>& st, const std::vector<Cell>& ce) {
  using Mem = typename Exec::memory_space;
  Kokkos::View<Stencil*, Mem> S("st", kStencils);
  Kokkos::View<Cell*, Mem> C("ce", kCases);
  {
    auto hs = Kokkos::create_mirror_view(S);
    auto hc = Kokkos::create_mirror_view(C);
    for (int i = 0; i < kStencils; ++i)
      hs(i) = st[i];
    for (int i = 0; i < kCases; ++i)
      hc(i) = ce[i];
    Kokkos::deep_copy(S, hs);
    Kokkos::deep_copy(C, hc);
  }
  Kokkos::View<PvTerm*, Mem> ref("ref", kCases), split("split", kCases), cache("cache", kCases),
      mom("mom", kCases);
  Kokkos::View<Probe*, Mem> probe("probe", kCases);
  Kokkos::View<int*, Mem> skip("skip", kCases);
  Kokkos::View<PvPolygon*, Mem> poly("poly", kCases);
  Kokkos::View<PvMoments*, Mem> mcache("mcache", kCases);
  Kokkos::View<Frame*, Mem> frRef("frRef", kCases), frMom("frMom", kCases);

  // T1: the frozen reference and the split composition, separate kernels
  Kokkos::parallel_for(
      "pvcache_ref", Kokkos::RangePolicy<Exec>(0, kCases), KOKKOS_LAMBDA(const int i) {
        const Stencil& T = S(i / kCells);
        const Cell& c = C(i);
        PvTerm t;
        pvFitTermReference(t, c.mx, c.my, c.mz, c.alpha, c.off, T.org, T.t1, T.t2, T.nn, T.dW,
                           T.cosMin, T.g);
        ref(i) = t;
        probe(i) = probeReference(c.mx, c.my, c.mz, c.alpha, c.off, T.org, T.t1, T.t2, T.nn, T.dW,
                                  T.cosMin, T.g);
      });
  Kokkos::parallel_for(
      "pvcache_split", Kokkos::RangePolicy<Exec>(0, kCases), KOKKOS_LAMBDA(const int i) {
        const Stencil& T = S(i / kCells);
        const Cell& c = C(i);
        PvTerm t;
        pvFitTerm(t, c.mx, c.my, c.mz, c.alpha, c.off, T.org, T.t1, T.t2, T.nn, T.dW, T.cosMin,
                  T.g);
        split(i) = t;
      });
  // T2: kernel 1 builds the cache, kernel 2 consumes it
  Kokkos::parallel_for(
      "pvcache_build", Kokkos::RangePolicy<Exec>(0, kCases), KOKKOS_LAMBDA(const int i) {
        const Cell& c = C(i);
        pvPolygonBuild(c.mx, c.my, c.mz, c.alpha, poly(i));
      });
  Kokkos::parallel_for(
      "pvcache_use", Kokkos::RangePolicy<Exec>(0, kCases), KOKKOS_LAMBDA(const int i) {
        const Stencil& T = S(i / kCells);
        const Cell& c = C(i);
        const PvPolygon& P = poly(i);
        PvTerm t;
        t.ok = false;
        skip(i) = 0;
        if (pvOutsideSupport(P, c.off, T.org, T.dW, T.g)) {
          skip(i) = 1;
        } else {
          double np[3];
          if (pvTermNormal(c.mx, c.my, c.mz, T.t1, T.t2, T.nn, T.cosMin, T.g, np) && P.nv >= 3)
            pvTermPolygon(t, np, P.v, P.nv, c.off, T.org, T.t1, T.t2, T.nn, T.dW, T.g);
        }
        cache(i) = t;
      });
  // T3: full, lower and entry-parallel accumulation of the reference terms, and the forced
  // rank-deficient prefixes (the first deficientCount(s) accepted terms)
  Kokkos::View<PvFit*, Mem> full("full", kStencils), lower("lower", kStencils),
      entry("entry", kStencils), fullR("fullR", kStencils), lowerR("lowerR", kStencils),
      entryR("entryR", kStencils);
  Kokkos::parallel_for(
      "pvcache_accum", Kokkos::RangePolicy<Exec>(0, kStencils), KOKKOS_LAMBDA(const int s) {
        PvFit f, l, fr, lr;
        pvFitInit(f);
        pvFitInit(l);
        pvFitInit(fr);
        pvFitInit(lr);
        for (int k = 0; k < kCells; ++k) {
          const PvTerm& t = ref(s * kCells + k);
          if (!t.ok)
            continue;
          pvFitAccum(f, t);
          pvFitAccumLower(l, t);
          if (fr.npoly < deficientCount(s)) {
            pvFitAccum(fr, t);
            pvFitAccumLower(lr, t);
          }
        }
        full(s) = f;
        lower(s) = l;
        fullR(s) = fr;
        lowerR(s) = lr;
      });
  {
    using Policy = Kokkos::TeamPolicy<Exec>;
    Kokkos::parallel_for(
        "pvcache_entry", Policy(kStencils, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename Policy::member_type& tm) {
          const int s = tm.league_rank();
          // two prefixes: every accepted term, and the first deficientCount(s) of them
          for (int pass = 0; pass < 2; ++pass) {
            int npoly = 0;
            for (int k = 0; k < kCells; ++k)
              if (ref(s * kCells + k).ok && (pass == 0 || npoly < deficientCount(s)))
                ++npoly;
            Kokkos::View<PvFit*, Mem> out = pass == 0 ? entry : entryR;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, kPvEntries), [&](const int e) {
              double a = 0.0;
              int used = 0;
              for (int k = 0; k < kCells && used < npoly; ++k) {
                const PvTerm& t = ref(s * kCells + k);
                if (!t.ok)
                  continue;
                pvFitAccumEntry(a, t, e);
                ++used;
              }
              // stage the entry in the output's own storage, rebuilt below by one lane
              if (e < 21) {
                int i = 0;
                while ((i + 1) * (i + 2) / 2 <= e)
                  ++i;
                out(s).A[i][e - i * (i + 1) / 2] = a;
              } else {
                out(s).b[e - 21] = a;
              }
            });
            tm.team_barrier();
            Kokkos::single(Kokkos::PerTeam(tm), [&]() {
              double ent[kPvEntries];
              for (int i = 0; i < 6; ++i) {
                ent[21 + i] = out(s).b[i];
                for (int j = 0; j <= i; ++j)
                  ent[i * (i + 1) / 2 + j] = out(s).A[i][j];
              }
              PvFit f;
              pvFitFromEntries(f, ent, npoly);
              out(s) = f;
            });
            tm.team_barrier();
          }
        });
  }
  // T4: the moment cache and its terms; the frames
  Kokkos::parallel_for(
      "pvcache_mbuild", Kokkos::RangePolicy<Exec>(0, kCases), KOKKOS_LAMBDA(const int i) {
        const Stencil& T = S(i / kCells);
        const Cell& c = C(i);
        pvMomentsBuild(c.mx, c.my, c.mz, c.alpha, T.g, mcache(i));
      });
  Kokkos::parallel_for(
      "pvcache_muse", Kokkos::RangePolicy<Exec>(0, kCases), KOKKOS_LAMBDA(const int i) {
        const Stencil& T = S(i / kCells);
        const Cell& c = C(i);
        double orgP[3];
        T.g.toPhys(T.org, orgP);
        PvTerm t;
        pvTermMoments(t, mcache(i), c.off, orgP, T.t1, T.t2, T.nn, T.dW, T.cosMin, T.g);
        mom(i) = t;
        // every case cell as a target of its own: the two frames
        Frame a, b;
        a.ok = curvFallbackFrameReference(c.mx, c.my, c.mz, c.alpha, T.g, a.nn, a.t1, a.t2, a.org);
        b.ok = pvFrameMoments(mcache(i), T.g, b.nn, b.t1, b.t2, b.org);
        frRef(i) = a;
        frMom(i) = b;
      });
  Kokkos::fence();

  Result R;
  R.ref = toHost(ref);
  R.split = toHost(split);
  R.cache = toHost(cache);
  R.mom = toHost(mom);
  R.probe = toHost(probe);
  R.skip = toHost(skip);
  R.full = toHost(full);
  R.lower = toHost(lower);
  R.entry = toHost(entry);
  R.fullR = toHost(fullR);
  R.lowerR = toHost(lowerR);
  R.entryR = toHost(entryR);
  R.frRef = toHost(frRef);
  R.frMom = toHost(frMom);
  return R;
}

bool sameTerm(const PvTerm& a, const PvTerm& b) {
  if (a.ok != b.ok)
    return false;
  if (!a.ok)
    return true;
  return std::memcmp(&a.w, &b.w, sizeof(double)) == 0 &&
         std::memcmp(&a.B, &b.B, sizeof(double)) == 0 && std::memcmp(a.s, b.s, sizeof a.s) == 0;
}

// lower triangle + diagonal + b + npoly
bool sameLower(const PvFit& a, const PvFit& b) {
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j <= i; ++j)
      if (std::memcmp(&a.A[i][j], &b.A[i][j], sizeof(double)) != 0)
        return false;
  return std::memcmp(a.b, b.b, sizeof a.b) == 0 && a.npoly == b.npoly;
}

struct Solve {
  bool ok, red;
  double a[6];
};
Solve solve(const PvFit& f) {
  Solve s;
  s.ok = pvFitSolve(f, s.a, s.red);
  return s;
}
bool sameSolve(const Solve& x, const Solve& y) {
  return x.ok == y.ok && x.red == y.red && std::memcmp(x.a, y.a, sizeof x.a) == 0;
}

int check(const char* name, const Result& R, const std::vector<Stencil>& st,
          const std::vector<Cell>& ce) {
  int bad = 0;
  // ---- T1
  int t1bad = 0, accepted = 0;
  for (int i = 0; i < kCases; ++i) {
    t1bad += !sameTerm(R.ref[i], R.split[i]);
    accepted += R.ref[i].ok;
  }
  const bool t1cov = accepted > kCases / 5 && kCases - accepted > kCases / 20;
  std::printf("[%s] T1 split == frozen: %d cases, %d accepted, mismatches %d%s\n", name, kCases,
              accepted, t1bad, t1cov ? "" : "  FAIL: coverage");
  bad += t1bad + !t1cov;
  // ---- T2
  int t2bad = 0, falseSkip = 0, planted = 0, plantedSkipped = 0, skips = 0;
  for (int i = 0; i < kCases; ++i) {
    t2bad += !sameTerm(R.ref[i], R.cache[i]);
    skips += R.skip[i];
    falseSkip += R.skip[i] && R.ref[i].ok;
    if (R.probe[i].reason == 5) {
      ++planted;
      plantedSkipped += R.skip[i];
    }
  }
  const bool t2cov = planted > 1000 && 2 * plantedSkipped >= planted;
  std::printf(
      "[%s] T2 cache pattern == frozen: mismatches %d; prefilter skips %d, false skips %d; "
      "outside-support rejections %d, skipped %d (%.1f %%)%s\n",
      name, t2bad, skips, falseSkip, planted, plantedSkipped,
      planted ? 100.0 * plantedSkipped / planted : 0.0, t2cov ? "" : "  FAIL: coverage");
  bad += t2bad + falseSkip + !t2cov;
  // ---- T3
  int t3acc = 0, t3solve = 0, nFull = 0, nRed = 0, nRedOk = 0;
  for (int s = 0; s < kStencils; ++s) {
    t3acc += !sameLower(R.full[s], R.lower[s]) + !sameLower(R.full[s], R.entry[s]) +
             !sameLower(R.fullR[s], R.lowerR[s]) + !sameLower(R.fullR[s], R.entryR[s]);
    const Solve a = solve(R.full[s]), b = solve(R.lower[s]), c = solve(R.entry[s]);
    const Solve ar = solve(R.fullR[s]), br = solve(R.lowerR[s]), cr = solve(R.entryR[s]);
    t3solve += !sameSolve(a, b) + !sameSolve(a, c) + !sameSolve(ar, br) + !sameSolve(ar, cr);
    nFull += a.ok && !a.red;
    nRed += R.fullR[s].npoly >= 3 && R.fullR[s].npoly <= 5;
    nRedOk += ar.ok && ar.red;
  }
  const bool t3cov = nFull > kStencils / 2 && nRedOk > kStencils / 2;
  std::printf(
      "[%s] T3 lower/entry == full: accumulator mismatches %d, solve mismatches %d; %d full-rank "
      "solves, %d forced-deficient systems (%d solved by the reduced model)%s\n",
      name, t3acc, t3solve, nFull, nRed, nRedOk, t3cov ? "" : "  FAIL: coverage");
  bad += t3acc + t3solve + !t3cov;
  // ---- T4: terms
  auto nearRel = [](double x, double thr, double scale) {
    return std::fabs(x - thr) <= 1e-9 * scale;
  };
  // Against the double reference with the stated tolerances, HARD-GATED on every accepted case
  // with cj > 0.2 (flow's cosMin: the regime tier 3 ever evaluates). Below that — the generator's
  // cosMin = 0 and -1 stencils, nearly edge-on polygons — the stated tolerance is not attainable by
  // ANY double evaluation: the projected quantities scale with cj while their rounding does not
  // (the reference's b1 = -np0/np2 amplifies it by 1/cj, and cj itself carries eps/|cj|). Those
  // cases are reported (INFO), with the moment term's excess over a long-double evaluation of the
  // reference's mathematics.
  int flagBad = 0, nearThr = 0, cmpd = 0, cmpProd = 0, outTol = 0, outProd = 0, outExact = 0;
  double worstS = 0.0, worstB = 0.0, worstW = 0.0, worstX = 0.0, minCjOut = 1.0;
  const int deg[6] = {0, 1, 1, 2, 2, 2};
  // max over the six s, B and w of |d| / tol; s0 is the |projected area|
  auto excess = [&](const double sv[6], double Bv, double wv, const long double sr[6],
                    long double Br, long double wr, double sg, double ws[3]) {
    const double s0 = static_cast<double>(std::fabs(sr[0]));
    ws[0] = ws[1] = ws[2] = 0.0;
    for (int k = 0; k < 6; ++k) {
      const double d = static_cast<double>(std::fabs(sg * sr[k] - sv[k]));
      const double tol =
          1e-13 * (static_cast<double>(std::fabs(sr[k])) + s0 * std::pow(3.5, deg[k]));
      ws[0] = std::fmax(ws[0], d / tol);
    }
    ws[1] = static_cast<double>(std::fabs(sg * Br - Bv)) /
            (1e-13 * (static_cast<double>(std::fabs(Br)) + 3.5 * s0));
    ws[2] = static_cast<double>(std::fabs(wr - wv)) / 1e-14;
    return std::fmax(ws[0], std::fmax(ws[1], ws[2]));
  };
  for (int i = 0; i < kCases; ++i) {
    const Stencil& T = st[i / kCells];
    const Probe& p = R.probe[i];
    const PvTerm& a = R.ref[i];
    const PvTerm& b = R.mom[i];
    const bool nr =
        (p.reason >= 2 && nearRel(p.cj, T.cosMin, std::fmax(std::fabs(T.cosMin), 1.0))) ||
        (p.reason >= 4 && nearRel(p.s0, 1e-14, 1e-14)) ||
        (p.reason >= 4 && T.dW > 0.0 && nearRel(p.r, T.dW, T.dW));
    if (a.ok != b.ok) {
      if (nr)
        ++nearThr;
      else
        ++flagBad;
      continue;
    }
    if (!a.ok)
      continue;
    ++cmpd;
    cmpProd += p.cj > 0.2;
    const double sg = (a.s[0] < 0.0) != (b.s[0] < 0.0) ? -1.0 : 1.0;
    const long double ar[6] = {a.s[0], a.s[1], a.s[2], a.s[3], a.s[4], a.s[5]};
    double ws[3];
    const double x = excess(b.s, b.B, b.w, ar, a.B, a.w, sg, ws);
    if (p.cj > 0.2) {  // the gated regime's worst excess
      worstS = std::fmax(worstS, ws[0]);
      worstB = std::fmax(worstB, ws[1]);
      worstW = std::fmax(worstW, ws[2]);
    }
    if (x <= 1.0)
      continue;
    ++outTol;
    minCjOut = std::fmin(minCjOut, std::fabs(p.cj));
    if (p.cj > 0.2) {
      ++outProd;
      continue;
    }
    const ExactTerm e = exactTerm(ce[i], T);
    const double sge = (e.s[0] < 0) != (b.s[0] < 0.0) ? -1.0 : 1.0;
    double we[3];
    const double xe = e.ok ? excess(b.s, b.B, b.w, e.s, e.B, e.w, sge, we) : 1e30;
    worstX = std::fmax(worstX, xe);
    outExact += !(xe <= 1.0);
    if (!(xe <= 1.0))
      std::printf(
          "  [%s] case %d off the long-double reference: cj %.3e cosMin %.2f |s0| %.3e "
          "excess s %.3f B %.3f w %.3f (vs double ref: s %.3f B %.3f w %.3f)\n",
          name, i, p.cj, T.cosMin, std::fabs(a.s[0]), we[0], we[1], we[2], ws[0], ws[1], ws[2]);
  }
  std::printf(
      "[%s] T4 moment terms vs frozen: flag mismatches %d (+%d within 1e-9 of a threshold); %d "
      "accepted compared, %d with cj > 0.2: worst/tol s %.3f B %.3f w %.3f, out of tolerance %d "
      "(gated)\n",
      name, flagBad, nearThr, cmpd, cmpProd, worstS, worstB, worstW, outProd);
  std::printf(
      "[%s] T4 INFO cj <= 0.2: %d out of tolerance vs the double reference (min |cj| %.3e); vs "
      "the long-double reference %d out, worst/tol %.3f\n",
      name, outTol - outProd, minCjOut, outExact, worstX);
  bad += flagBad + outProd + (cmpd > kCases / 5 && cmpProd > kCases / 10 ? 0 : 1);
  // ---- T4: frames
  int frBad = 0, framed = 0;
  double worstOrg = 0.0;
  for (int i = 0; i < kCases; ++i) {
    const Frame& a = R.frRef[i];
    const Frame& b = R.frMom[i];
    if (a.ok != b.ok) {
      ++frBad;
      continue;
    }
    if (!a.ok)
      continue;
    ++framed;
    frBad += std::memcmp(a.nn, b.nn, sizeof a.nn) != 0 ||
             std::memcmp(a.t1, b.t1, sizeof a.t1) != 0 || std::memcmp(a.t2, b.t2, sizeof a.t2) != 0;
    double op[3];
    st[i / kCells].g.toPhys(a.org, op);
    for (int k = 0; k < 3; ++k)
      worstOrg = std::fmax(worstOrg, std::fabs(op[k] - b.org[k]));
  }
  const bool orgOk = worstOrg <= 1e-14;
  std::printf("[%s] T4 frames: %d framed, frame mismatches %d, max |d org| %.3e%s\n", name, framed,
              frBad, worstOrg, orgOk ? "" : "  FAIL");
  bad += frBad + !orgOk + (framed > kCases / 2 ? 0 : 1);
  return bad;
}

// ---- T5: analytic moments ----------------------------------------------------------------------
template <class Exec>
int checkAnalytic(const char* name) {
  using Mem = typename Exec::memory_space;
  Kokkos::View<PvMoments*, Mem> out("an", 3);
  Kokkos::parallel_for(
      "pvcache_analytic", Kokkos::RangePolicy<Exec>(0, 3), KOKKOS_LAMBDA(const int k) {
        VofMetric g;
        if (k == 1) {
          g.h[0] = 2.0;
          g.h[1] = 1.0;
          g.h[2] = 0.5;
        }
        if (k < 2)
          pvMomentsBuild(0.0, 0.0, 1.0, 0.5, g, out(k));
        else
          pvMomentsBuild(0.3, -0.5, 0.8, 0.31, g, out(k));
      });
  Kokkos::fence();
  const std::vector<PvMoments> M = toHost(out);
  int bad = 0;
  auto chk = [&](const char* what, double got, double want, double tol) {
    const bool ok = std::fabs(got - want) <= tol;
    if (!ok)
      std::printf("[%s] T5 %s = %.17g, want %.17g (tol %.1e)  FAIL\n", name, what, got, want, tol);
    bad += !ok;
  };
  // the unit square at z = 0.5 of the unit cell, cell-centred: a = 1, m1 = 0, m2 =
  // diag(1/12,1/12,0)
  chk("square a", M[0].a, 1.0, 1e-15);
  for (int a = 0; a < 3; ++a)
    chk("square m1", M[0].m1[a], 0.0, 1e-15);
  chk("square m2xx", M[0].m2[0], 1.0 / 12.0, 1e-15);
  chk("square m2yy", M[0].m2[1], 1.0 / 12.0, 1e-15);
  for (int q = 2; q < 6; ++q)
    chk("square m2 (zero entries)", M[0].m2[q], 0.0, 1e-15);
  // the same plane on h = (2, 1, 0.5): a 2 x 1 rectangle: a = 2, m2xx = 2 * 4/12, m2yy = 2 * 1/12
  chk("metric a", M[1].a, 2.0, 1e-15);
  chk("metric m2xx", M[1].m2[0], 2.0 / 3.0, 1e-15);
  chk("metric m2yy", M[1].m2[1], 1.0 / 6.0, 1e-15);
  // a tilted plane against a fine triangulated quadrature of its polygon: each fan triangle split
  // into L^2 congruent sub-triangles, each integrated by the edge-midpoint rule (exact for
  // quadratics), accumulated in long double — independent of the fan formulas above
  {
    double v[8][3];
    const int nv = plicPolygon(0.3, -0.5, 0.8, 0.31, v);
    long double qa = 0, q1[3] = {0, 0, 0}, q2[6] = {0, 0, 0, 0, 0, 0};
    const int L = 32;
    const int ip[6] = {0, 1, 2, 0, 0, 1}, iq[6] = {0, 1, 2, 1, 2, 2};
    for (int k = 1; k + 1 < nv; ++k) {
      long double P0[3], e1[3], e2[3];
      for (int a = 0; a < 3; ++a) {
        P0[a] = static_cast<long double>(v[0][a]) - 0.5L;
        e1[a] = static_cast<long double>(v[k][a]) - v[0][a];
        e2[a] = static_cast<long double>(v[k + 1][a]) - v[0][a];
      }
      const long double cr[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                                 e1[0] * e2[1] - e1[1] * e2[0]};
      const long double At = 0.5L * std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
      const long double dA = At / (static_cast<long double>(L) * L);
      auto pt = [&](long double u, long double w, long double Y[3]) {
        for (int a = 0; a < 3; ++a)
          Y[a] = P0[a] + u / L * e1[a] + w / L * e2[a];
      };
      for (int i = 0; i < L; ++i)
        for (int j = 0; i + j < L; ++j)
          for (int up = 0; up < 2; ++up) {
            if (up == 1 && i + j + 1 >= L)
              continue;
            // corners in (u, w) lattice units: up = 0 (i,j)(i+1,j)(i,j+1); up = 1 the flipped one
            const long double cu[3] = {up ? i + 1.0L : i + 0.0L, up ? i + 0.0L : i + 1.0L,
                                       up ? i + 1.0L : i + 0.0L};
            const long double cw[3] = {up ? j + 1.0L : j + 0.0L, up ? j + 1.0L : j + 0.0L,
                                       up ? j + 0.0L : j + 1.0L};
            for (int m = 0; m < 3; ++m) {
              const int n = (m + 1) % 3;
              long double Y[3];
              pt(0.5L * (cu[m] + cu[n]), 0.5L * (cw[m] + cw[n]), Y);
              const long double wq = dA / 3;
              qa += wq;
              for (int a = 0; a < 3; ++a)
                q1[a] += wq * Y[a];
              for (int q = 0; q < 6; ++q)
                q2[q] += wq * Y[ip[q]] * Y[iq[q]];
            }
          }
    }
    chk("tilted a", M[2].a, static_cast<double>(qa), 1e-12);
    for (int a = 0; a < 3; ++a)
      chk("tilted m1", M[2].m1[a], static_cast<double>(q1[a]), 1e-12);
    for (int q = 0; q < 6; ++q)
      chk("tilted m2", M[2].m2[q], static_cast<double>(q2[q]), 1e-12);
    std::printf("[%s] T5 tilted polygon nv %d: a %.15f (quadrature %.15Lf)\n", name, nv, M[2].a,
                qa);
  }
  std::printf("[%s] T5 analytic moments: %s\n", name, bad ? "FAIL" : "ok");
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int bad = 0;
  {
    std::vector<Stencil> st;
    std::vector<Cell> ce;
    makeCases(st, ce);
    bad += check(Kokkos::DefaultExecutionSpace::name(), run<Kokkos::DefaultExecutionSpace>(st, ce),
                 st, ce);
    bad += check(Kokkos::DefaultHostExecutionSpace::name(),
                 run<Kokkos::DefaultHostExecutionSpace>(st, ce), st, ce);
    bad += checkAnalytic<Kokkos::DefaultExecutionSpace>(Kokkos::DefaultExecutionSpace::name());
    bad +=
        checkAnalytic<Kokkos::DefaultHostExecutionSpace>(Kokkos::DefaultHostExecutionSpace::name());
  }
  Kokkos::finalize();
  std::printf(bad ? "FAIL\n" : "PASS\n");
  return bad ? 1 : 0;
}

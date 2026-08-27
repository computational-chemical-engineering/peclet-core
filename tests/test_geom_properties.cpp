// Layer 0 rung 2 gate (suite/docs/ANALYTIC_SDF_GEOMETRY.md): a-priori property tests over the full
// leaf vocabulary. There is no bit-exact oracle at this rung -- these primitives are NEW numerics,
// not a relocation -- so correctness is established by measuring the properties an SDF must have:
//
//   1. SIGN        eval(p) < 0 exactly where an INDEPENDENT inside/outside oracle says solid.
//                  The oracle is written from the shape's defining inequality, not from eval.
//   2. EIKONAL     |grad eval| == 1 away from kinks, for every leaf claiming exact_distance.
//                  This is the sharp, sampling-error-free test of exactness: a field that is not a
//                  true distance fails it immediately.
//   3. GRADIENT    the closed-form grad() agrees with central differences of that leaf's own
//                  eval(), for every leaf claiming analytic_gradient.
//   4. LIPSCHITZ   |f(a) - f(b)| <= |a - b|. Necessary for any distance or distance-bound, and the
//                  property Layer 2's bracket-and-root-find actually relies on.
//   5. METRIC      |eval| vs a brute-force nearest point over a dense parametric sampling of the
//                  true surface. For exact leaves this must agree; for bound-only leaves the ratio
//                  is REPORTED, so whether the estimate under- or over-shoots is measured rather
//                  than assumed. Over-claiming exactness costs Layer 2 correctness, so a leaf is
//                  marked exact only if it earns it here and in (2).
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "peclet/core/geom/primitives.hpp"
#include "test_util.hpp"

using namespace peclet::core;
namespace prim = peclet::core::geom::prim;

using V = Vec3<double>;
using Surface = std::vector<V>;
using Inside = std::function<bool(V)>;

struct Rng {
  unsigned long long s = 0x9E3779B97F4A7C15ull;
  double u(double lo, double hi) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return lo + (hi - lo) * (double)((s >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
  }
};

static double len(V a) {
  return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
static V subv(V a, V b) {
  return V{a.x - b.x, a.y - b.y, a.z - b.z};
}

// Brute-force distance from p to the nearest sampled surface point. This is an UPPER bound on the
// true distance (the nearest true point may lie between samples), by O(spacing^2 / distance) -- so
// it is used with a tolerance, and the sharp exactness evidence comes from the eikonal test.
static double nearestSurface(V p, const Surface& surf) {
  double best = 1e300;
  for (const V& q : surf) {
    const double d = len(subv(p, q));
    if (d < best)
      best = d;
  }
  return best;
}

struct Stats {
  double eikonalMaxErr = 0;    // max | |grad| - 1 |
  double gradMaxErr = 0;       // max |closed-form grad - FD grad|
  double lipMax = 0;           // max |f(a)-f(b)| / |a-b|
  double metricMaxErr = 0;     // max | |eval| - nearestSurface |
  double ratioMin = 1e300;     // min |eval| / nearestSurface
  double ratioMax = 0;         // max |eval| / nearestSurface
  int signFailures = 0;
  int eikonalSamples = 0, metricSamples = 0;
};

template <class Shape>
static Stats measure(const Shape& sh, const Inside& inside, const Surface& surf, double box,
                     Rng& rng) {
  Stats st;
  const double h = 1e-6;

  for (int i = 0; i < 20000; ++i) {
    const V p{rng.u(-box, box), rng.u(-box, box), rng.u(-box, box)};
    const double f = sh.eval(p);

    // (1) sign, skipping a thin shell around the surface where the oracle and eval legitimately
    // disagree on which side a point within rounding of the boundary falls.
    if (std::fabs(f) > 1e-6) {
      if ((f < 0) != inside(p))
        ++st.signFailures;
    }

    // (2)+(3) gradient tests, away from kinks. A kink is where the FD stencil straddles a
    // non-smooth feature; detect it by requiring the three central differences to be mutually
    // consistent with a smooth field (second difference small).
    const V gfd = prim::gradient(sh, p, h);
    const double gl = len(gfd);
    const bool smooth = std::fabs(sh.eval(V{p.x + h, p.y, p.z}) + sh.eval(V{p.x - h, p.y, p.z}) -
                                  2 * f) < 1e-6 &&
                        std::fabs(sh.eval(V{p.x, p.y + h, p.z}) + sh.eval(V{p.x, p.y - h, p.z}) -
                                  2 * f) < 1e-6 &&
                        std::fabs(sh.eval(V{p.x, p.y, p.z + h}) + sh.eval(V{p.x, p.y, p.z - h}) -
                                  2 * f) < 1e-6;
    if (smooth && gl > 0.1) {
      if (Shape::exact_distance) {
        st.eikonalMaxErr = std::fmax(st.eikonalMaxErr, std::fabs(gl - 1.0));
        ++st.eikonalSamples;
      }
      if (Shape::analytic_gradient) {
        const V ga = sh.grad(p);
        st.gradMaxErr = std::fmax(st.gradMaxErr, len(subv(ga, gfd)));
      }
    }
  }

  // (4) Lipschitz over random pairs.
  for (int i = 0; i < 20000; ++i) {
    const V a{rng.u(-box, box), rng.u(-box, box), rng.u(-box, box)};
    const V b{rng.u(-box, box), rng.u(-box, box), rng.u(-box, box)};
    const double d = len(subv(a, b));
    if (d > 1e-9)
      st.lipMax = std::fmax(st.lipMax, std::fabs(sh.eval(a) - sh.eval(b)) / d);
  }

  // (5) metric check against the sampled surface. Restricted to OUTSIDE points at a healthy
  // distance, where the sampled-surface upper bound is tightest.
  for (int i = 0; i < 300; ++i) {
    const V p{rng.u(-box, box), rng.u(-box, box), rng.u(-box, box)};
    const double f = sh.eval(p);
    if (f <= 0.05)
      continue;
    const double t = nearestSurface(p, surf);
    if (t < 0.05)
      continue;
    st.metricMaxErr = std::fmax(st.metricMaxErr, std::fabs(f - t));
    st.ratioMin = std::fmin(st.ratioMin, f / t);
    st.ratioMax = std::fmax(st.ratioMax, f / t);
    ++st.metricSamples;
  }
  return st;
}

template <class Shape>
static void report(const char* name, const Shape& sh, const Inside& inside, const Surface& surf,
                   double box, Rng& rng) {
  const Stats st = measure(sh, inside, surf, box, rng);
  std::printf("  %-20s exact=%d agrad=%d | sign_fail=%-4d eikonal=%.2e grad=%.2e lip=%.4f "
              "metric=%.2e ratio=[%.4f,%.4f]\n",
              name, (int)Shape::exact_distance, (int)Shape::analytic_gradient, st.signFailures,
              st.eikonalMaxErr, st.gradMaxErr, st.lipMax, st.metricMaxErr, st.ratioMin,
              st.ratioMax);

  PECLET_CORE_CHECK(st.signFailures == 0);
  // 1-Lipschitz is a THEOREM about true distance fields, so it is asserted only for the leaves
  // claiming exactness. For a bound-only leaf it is a measured property, not a requirement:
  // contract 2 promises those are sign-correct bounds, and Layer 2's bracket-and-root-find needs
  // only the sign. The constant is still reported (and recorded in the design note) because it is
  // what decides whether a consumer may sphere-trace the field; a leaf far above 1 must not be.
  if (Shape::exact_distance)
    PECLET_CORE_CHECK(st.lipMax <= 1.0 + 1e-9);
  else
    PECLET_CORE_CHECK(st.lipMax < 2.0);  // sanity ceiling: catches a blown-up estimate
  if (Shape::exact_distance) {
    PECLET_CORE_CHECK(st.eikonalSamples > 500);
    PECLET_CORE_CHECK(st.eikonalMaxErr < 1e-4);  // sharp: a non-distance fails this outright
    PECLET_CORE_CHECK(st.metricSamples > 50);
    PECLET_CORE_CHECK(st.metricMaxErr < 5e-3);  // loose: sampled-surface slack
  }
  if (Shape::analytic_gradient)
    PECLET_CORE_CHECK(st.gradMaxErr < 1e-4);
}

// --- parametric surface samplers -------------------------------------------------------------
static void addSphere(Surface& s, double R, int n) {
  for (int i = 0; i <= n; ++i)
    for (int j = 0; j < 2 * n; ++j) {
      const double th = M_PI * i / n, ph = M_PI * j / n;
      s.push_back(
          V{R * std::sin(th) * std::cos(ph), R * std::cos(th), R * std::sin(th) * std::sin(ph)});
    }
}

int main() {
  Rng rng;
  std::printf("rung-2 property measurements "
             "(sign / eikonal / analytic-grad / Lipschitz / metric)\n");

  // --- Sphere -------------------------------------------------------------------------------
  {
    const double R = 1.1;
    Surface surf;
    addSphere(surf, R, 160);
    report("Sphere", prim::Sphere<double>{R}, [&](V p) { return len(p) < R; }, surf, 2.5, rng);
  }

  // --- Box ----------------------------------------------------------------------------------
  {
    const double hx = 0.8, hy = 1.1, hz = 0.5;
    Surface surf;
    const int n = 90;
    for (int i = 0; i <= n; ++i)
      for (int j = 0; j <= n; ++j) {
        const double a = -1 + 2.0 * i / n, b = -1 + 2.0 * j / n;
        surf.push_back(V{hx, hy * a, hz * b});
        surf.push_back(V{-hx, hy * a, hz * b});
        surf.push_back(V{hx * a, hy, hz * b});
        surf.push_back(V{hx * a, -hy, hz * b});
        surf.push_back(V{hx * a, hy * b, hz});
        surf.push_back(V{hx * a, hy * b, -hz});
      }
    report("Box", prim::Box<double>{hx, hy, hz},
           [&](V p) {
             return std::fabs(p.x) < hx && std::fabs(p.y) < hy && std::fabs(p.z) < hz;
           },
           surf, 2.5, rng);
  }

  // --- HollowCylinder (dem form, y axis) ----------------------------------------------------
  {
    const double ro = 1.0, hgt = 1.4, th = 0.35;
    const double rMid = ro - th * 0.5, ri = ro - th;
    Surface surf;
    const int na = 400, nv = 60;
    for (int i = 0; i < na; ++i) {
      const double a = 2 * M_PI * i / na, c = std::cos(a), s2 = std::sin(a);
      for (int j = 0; j <= nv; ++j) {
        const double y = -hgt / 2 + hgt * j / nv;
        surf.push_back(V{ro * c, y, ro * s2});
        surf.push_back(V{ri * c, y, ri * s2});
      }
      for (int j = 0; j <= 12; ++j) {
        const double r = ri + (ro - ri) * j / 12.0;
        surf.push_back(V{r * c, hgt / 2, r * s2});
        surf.push_back(V{r * c, -hgt / 2, r * s2});
      }
    }
    report("HollowCylinder", prim::HollowCylinder<double>{ro, hgt, th},
           [&](V p) {
             const double r = std::sqrt(p.x * p.x + p.z * p.z);
             return std::fabs(r - rMid) < th * 0.5 && std::fabs(p.y) < hgt * 0.5;
           },
           surf, 2.5, rng);
  }

  // --- HollowCylinderShell (max form, z axis) -- bound-only, ratio is the point ---------------
  {
    const double ro = 1.0, ri = 0.6, hgt = 1.4;
    Surface surf;
    const int na = 400, nv = 60;
    for (int i = 0; i < na; ++i) {
      const double a = 2 * M_PI * i / na, c = std::cos(a), s2 = std::sin(a);
      for (int j = 0; j <= nv; ++j) {
        const double z = -hgt / 2 + hgt * j / nv;
        surf.push_back(V{ro * c, ro * s2, z});
        surf.push_back(V{ri * c, ri * s2, z});
      }
      for (int j = 0; j <= 12; ++j) {
        const double r = ri + (ro - ri) * j / 12.0;
        surf.push_back(V{r * c, r * s2, hgt / 2});
        surf.push_back(V{r * c, r * s2, -hgt / 2});
      }
    }
    report("HollowCylinderShell", prim::HollowCylinderShell<double>{ro, ri, hgt},
           [&](V p) {
             const double r = std::sqrt(p.x * p.x + p.y * p.y);
             return r < ro && r > ri && std::fabs(p.z) < hgt * 0.5;
           },
           surf, 2.5, rng);
  }

  // --- Capsule ------------------------------------------------------------------------------
  {
    const double rad = 0.45, hl = 0.7;
    Surface surf;
    const int na = 300, nv = 60;
    for (int i = 0; i < na; ++i) {
      const double a = 2 * M_PI * i / na, c = std::cos(a), s2 = std::sin(a);
      for (int j = 0; j <= nv; ++j)
        surf.push_back(V{rad * c, -hl + 2 * hl * j / nv, rad * s2});
      for (int j = 1; j <= 40; ++j) {  // hemispherical caps
        const double t = 0.5 * M_PI * j / 40.0;
        surf.push_back(V{rad * std::cos(t) * c, hl + rad * std::sin(t), rad * std::cos(t) * s2});
        surf.push_back(V{rad * std::cos(t) * c, -hl - rad * std::sin(t), rad * std::cos(t) * s2});
      }
    }
    report("Capsule", prim::Capsule<double>{rad, hl},
           [&](V p) {
             const double qy = p.y - std::fmin(std::fmax(p.y, -hl), hl);
             return std::sqrt(p.x * p.x + qy * qy + p.z * p.z) < rad;
           },
           surf, 2.5, rng);
  }

  // --- Torus --------------------------------------------------------------------------------
  {
    const double R = 0.9, r = 0.3;
    Surface surf;
    const int nu = 300, nv = 120;
    for (int i = 0; i < nu; ++i)
      for (int j = 0; j < nv; ++j) {
        const double u = 2 * M_PI * i / nu, v = 2 * M_PI * j / nv;
        const double rr = R + r * std::cos(v);
        surf.push_back(V{rr * std::cos(u), r * std::sin(v), rr * std::sin(u)});
      }
    report("Torus", prim::Torus<double>{R, r},
           [&](V p) {
             const double q = std::sqrt(p.x * p.x + p.z * p.z) - R;
             return q * q + p.y * p.y < r * r;
           },
           surf, 2.5, rng);
  }

  // --- Cone (frustum) -----------------------------------------------------------------------
  {
    const double rb = 0.9, rt = 0.35, hh = 0.8;
    auto radiusAt = [&](double y) { return rb + (rt - rb) * (y + hh) / (2 * hh); };
    Surface surf;
    const int na = 400, nv = 80;
    for (int i = 0; i < na; ++i) {
      const double a = 2 * M_PI * i / na, c = std::cos(a), s2 = std::sin(a);
      for (int j = 0; j <= nv; ++j) {
        const double y = -hh + 2 * hh * j / nv, rr = radiusAt(y);
        surf.push_back(V{rr * c, y, rr * s2});
      }
      for (int j = 0; j <= 20; ++j) {
        surf.push_back(V{rb * c * j / 20.0, -hh, rb * s2 * j / 20.0});
        surf.push_back(V{rt * c * j / 20.0, hh, rt * s2 * j / 20.0});
      }
    }
    report("Cone", prim::Cone<double>{rb, rt, hh},
           [&](V p) {
             return std::fabs(p.y) < hh && std::sqrt(p.x * p.x + p.z * p.z) < radiusAt(p.y);
           },
           surf, 2.5, rng);
  }

  // --- Ellipsoid (bound-only: the ratio band is the measurement) -----------------------------
  {
    const double rx = 1.0, ry = 0.65, rz = 0.4;
    Surface surf;
    const int n = 200;
    for (int i = 0; i <= n; ++i)
      for (int j = 0; j < 2 * n; ++j) {
        const double th = M_PI * i / n, ph = M_PI * j / n;
        surf.push_back(V{rx * std::sin(th) * std::cos(ph), ry * std::cos(th),
                         rz * std::sin(th) * std::sin(ph)});
      }
    report("Ellipsoid", prim::Ellipsoid<double>{rx, ry, rz},
           [&](V p) {
             const double a = p.x / rx, b = p.y / ry, c = p.z / rz;
             return a * a + b * b + c * c < 1.0;
           },
           surf, 2.5, rng);
  }

  // --- Superquadric (bound-only) ------------------------------------------------------------
  {
    const double rx = 0.9, ry = 0.7, rz = 0.5, e = 4.0;
    auto sgnpow = [](double v, double p) {
      return (v < 0 ? -1.0 : 1.0) * std::pow(std::fabs(v), p);
    };
    Surface surf;
    const int n = 200;
    for (int i = 0; i <= n; ++i)
      for (int j = 0; j < 2 * n; ++j) {
        const double th = -0.5 * M_PI + M_PI * i / n, ph = -M_PI + 2 * M_PI * j / (2 * n);
        const double ct = sgnpow(std::cos(th), 2.0 / e), st = sgnpow(std::sin(th), 2.0 / e);
        const double cp = sgnpow(std::cos(ph), 2.0 / e), sp = sgnpow(std::sin(ph), 2.0 / e);
        surf.push_back(V{rx * ct * cp, ry * st, rz * ct * sp});
      }
    report("Superquadric e=4", prim::Superquadric<double>{rx, ry, rz, e},
           [&](V p) {
             return std::pow(std::fabs(p.x) / rx, e) + std::pow(std::fabs(p.y) / ry, e) +
                        std::pow(std::fabs(p.z) / rz, e) <
                    1.0;
           },
           surf, 2.5, rng);
  }

  // --- Rounded box: offsetting an exact field must stay exact ---------------------------------
  {
    const double hx = 0.6, hy = 0.5, hz = 0.4, rr = 0.2;
    Surface surf;
    prim::Box<double> inner{hx, hy, hz};
    // The rounded box is the Minkowski sum of the box with a ball of radius rr, so its surface has
    // THREE pieces: flat faces (offset along one normal), quarter-cylinder edges (the normal sweeps
    // 90 degrees about the edge line), and octant-sphere corners. Offsetting face points along the
    // face normal alone -- the obvious thing -- builds a box with SHARP offset edges lying outside
    // the true rounded surface, which is what the first run of this gate caught (metric err 0.10
    // against an eikonal of 4e-9, i.e. the field was right and the sampler was wrong).
    const int n = 60;
    for (int i = 0; i <= n; ++i)
      for (int j = 0; j <= n; ++j) {
        const double a = -1 + 2.0 * i / n, b = -1 + 2.0 * j / n;
        // faces
        surf.push_back(V{hx + rr, hy * a, hz * b});
        surf.push_back(V{-hx - rr, hy * a, hz * b});
        surf.push_back(V{hx * a, hy + rr, hz * b});
        surf.push_back(V{hx * a, -hy - rr, hz * b});
        surf.push_back(V{hx * a, hy * b, hz + rr});
        surf.push_back(V{hx * a, hy * b, -hz - rr});
      }
    const int ne = 90;
    for (int i = 0; i <= ne; ++i) {
      const double t = -1 + 2.0 * i / ne;                 // position along the edge
      for (int j = 0; j <= 24; ++j) {
        const double a = 0.5 * M_PI * j / 24.0, c = rr * std::cos(a), s2 = rr * std::sin(a);
        for (int sa = -1; sa <= 1; sa += 2)
          for (int sb = -1; sb <= 1; sb += 2) {
            // edges parallel to each axis, one quarter-cylinder per (sa, sb) sign pair
            surf.push_back(V{hx * t, sa * (hy + c), sb * (hz + s2)});
            surf.push_back(V{sa * (hx + c), hy * t, sb * (hz + s2)});
            surf.push_back(V{sa * (hx + c), sb * (hy + s2), hz * t});
          }
      }
    }
    for (int i = 0; i <= 30; ++i)
      for (int j = 0; j <= 30; ++j) {
        const double th = 0.5 * M_PI * i / 30.0, ph = 0.5 * M_PI * j / 30.0;
        const double ux = std::sin(th) * std::cos(ph), uy = std::cos(th);
        const double uz = std::sin(th) * std::sin(ph);
        for (int sx = -1; sx <= 1; sx += 2)
          for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2)
              surf.push_back(V{sx * (hx + rr * ux), sy * (hy + rr * uy), sz * (hz + rr * uz)});
      }
    prim::Rounded<prim::Box<double>> rb{inner, rr};
    report("Rounded(Box)", rb, [&](V p) { return inner.eval(p) - rr < 0; }, surf, 2.5, rng);
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

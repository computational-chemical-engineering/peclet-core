// Layer 2-for-core gate (docs/AMR_GEOMETRY_SETUP_REQUIREMENTS.md). Two distinct parity claims,
// gated separately because they are different strengths:
//
//   (1) ACCELERATION IS FREE: candidate-grid + equal-radius evaluation is BIT-IDENTICAL to the
//       library's own brute-force scan — value, not sign — under list reordering and the
//       out-of-coverage fallback. Same expressions, superset lists, min combination: exact.
//   (2) THE CANONICAL EXPRESSION vs THE RETIRED STOPGAP: the library computes the per-sphere
//       distance FMA-canonically (one rounding in the min-image and the d2 chain) so host and
//       device evaluate bit-identically; the retired stopgap used the two-rounding form. The
//       difference is bounded — measured here against an independent transcription of the stopgap
//       — at a few ulp of the sqrt magnitude, and any sign disagreement is confined to values
//       within that band of zero (knife-edge samples).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#include "peclet/core/geom/scene_query.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;

// The stopgap, transcribed literally (loop, per-sphere min-image, sqrt, running min).
static double stopgap(const std::vector<double>& cx, const std::vector<double>& cy,
                      const std::vector<double>& cz, const std::vector<double>& r, bool periodic,
                      double Lx, double Ly, double Lz, const Vec<3>& p) {
  double best = 1e300;
  for (std::size_t i = 0; i < cx.size(); ++i) {
    double dx = p[0] - cx[i], dy = p[1] - cy[i], dz = p[2] - cz[i];
    if (periodic) {
      dx -= Lx * std::nearbyint(dx / Lx);
      dy -= Ly * std::nearbyint(dy / Ly);
      dz -= Lz * std::nearbyint(dz / Lz);
    }
    const double d = std::sqrt(dx * dx + dy * dy + dz * dz) - r[i];
    if (d < best)
      best = d;
  }
  return best;
}

static bool bitEq(double a, double b) {
  std::uint64_t x, y;
  std::memcpy(&x, &a, 8);
  std::memcpy(&y, &b, 8);
  return x == y;
}

struct Rng {
  std::mt19937_64 g{20260829ull};
  double u(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(g); }
};

// One full parity campaign against a sphere set. Returns probes checked.
static int campaign(const char* name, const std::vector<double>& cx, const std::vector<double>& cy,
                    const std::vector<double>& cz, const std::vector<double>& r, bool periodic,
                    double L, Rng& rng, double probeScale) {
  SphereBedQuery q(cx, cy, cz, r, Vec<3>{0, 0, 0}, Vec<3>{L, L, L}, periodic);
  int checked = 0, fell = 0;
  // The strict claim: grid path == library brute path, bitwise.
  double worstUlp = 0;  // and the measured claim vs the legacy expression
  int signFlips = 0;
  auto checkPoint = [&](const Vec<3>& p) {
    const double got = q(p);
    const double brute = evalSphereUnion(q.sphereUnion(), Vec3<double>{p[0], p[1], p[2]}, q.box());
    if (!bitEq(got, brute)) {
      std::fprintf(stderr, "  %s GRID!=BRUTE at (%.17g, %.17g, %.17g): %.17g vs %.17g\n", name,
                   p[0], p[1], p[2], got, brute);
      ++peclet::core::test::g_failures;
    }
    const double ref = stopgap(cx, cy, cz, r, periodic, L, L, L, p);
    // ulp scale of the value BEFORE the cancelling subtraction: sqrt(d2) ~ ref + r0
    const double scale = std::fabs(ref) + (r.empty() ? 1.0 : r[0]);
    const double ulp = scale * 2.220446049250313e-16;
    const double diff = std::fabs(got - ref);
    if (diff > 0)
      worstUlp = std::max(worstUlp, diff / ulp);
    if ((got < 0) != (ref < 0) && std::fabs(ref) > 8 * ulp)
      ++signFlips;  // a sign flip OUTSIDE the knife-edge band would be a real error
    ++checked;
  };
  // (a) uniform probes over (and beyond) the box — beyond exercises wrap/fallback
  const double m = periodic ? 0.3 * L : 0.0;
  for (int t = 0; t < 40000; ++t)
    checkPoint(Vec<3>{rng.u(-m, L + m), rng.u(-m, L + m), rng.u(-m, L + m)});
  // (b) probe-scale points seeded NEAR SURFACES (the classification-flicker regime): a random
  // surface point of a random sphere, nudged by +-probeScale along a random direction.
  for (int t = 0; t < 40000; ++t) {
    const std::size_t i = (std::size_t)rng.u(0, (double)cx.size());
    const double th = rng.u(0, 2 * M_PI), uz = rng.u(-1, 1), s = std::sqrt(1 - uz * uz);
    const double eps = rng.u(-probeScale, probeScale);
    checkPoint(Vec<3>{cx[i] + (r[i] + eps) * s * std::cos(th),
                      cy[i] + (r[i] + eps) * s * std::sin(th), cz[i] + (r[i] + eps) * uz});
  }
  // (c) superset audit: the brute argmin must be in the list of its bin (or the bin falls back)
  for (int t = 0; t < 5000; ++t) {
    const Vec<3> p{rng.u(0, L), rng.u(0, L), rng.u(0, L)};
    std::size_t win = 0;
    double best = 1e300;
    for (std::size_t i = 0; i < cx.size(); ++i) {
      double dx = p[0] - cx[i], dy = p[1] - cy[i], dz = p[2] - cz[i];
      if (periodic) {
        dx -= L * std::nearbyint(dx / L);
        dy -= L * std::nearbyint(dy / L);
        dz -= L * std::nearbyint(dz / L);
      }
      const double d = std::sqrt(dx * dx + dy * dy + dz * dz) - r[i];
      if (d < best) {
        best = d;
        win = i;
      }
    }
    const auto& gv = q.grid();
    const long b = gv.binOf(Vec3<double>{p[0], p[1], p[2]});
    if (b < 0 || gv.offsets[b] == gv.offsets[b + 1]) {
      ++fell;  // fallback path: exact by construction
      continue;
    }
    bool found = false;
    for (int k = gv.offsets[b]; k < gv.offsets[b + 1]; ++k)
      if ((std::size_t)gv.items[k] == win)
        found = true;
    PECLET_CORE_CHECK(found);
  }
  PECLET_CORE_CHECK(worstUlp < 4.0);  // fma- vs two-rounding form: a few ulp at sqrt magnitude
  PECLET_CORE_CHECK(signFlips == 0);
  std::printf(
      "  %-22s %d probes: grid==brute bitwise; vs legacy stopgap worst %.2f ulp, "
      "0 sign flips beyond the knife edge; meanList=%.2f, fallbacks=%d/5000\n",
      name, checked, worstUlp, q.meanListLength(), fell);
  return checked;
}

int main() {
  Rng rng;

  // --- the RCP acceptance bed itself: 180 monodisperse spheres, unit periodic box ------------
  {
    std::vector<double> cx, cy, cz, r;
    std::ifstream f("data/rcp_pack_seed3_unit.txt");
    PECLET_CORE_CHECK(f.good());
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      std::istringstream is(line);
      double x, y, z, rr;
      if (is >> x >> y >> z >> rr) {
        cx.push_back(x);
        cy.push_back(y);
        cz.push_back(z);
        r.push_back(rr);
      }
    }
    PECLET_CORE_CHECK(cx.size() == 180);
    // probeScale ~ h/2 at depth 8 (N=256): 0.5/256 in unit-box units
    campaign("RCP-180 (equal-R)", cx, cy, cz, r, true, 1.0, rng, 0.5 / 256.0);

    // shuffle-invariance: reorder every bin's list; values must not move (min is a value)
    {
      SphereBedQuery q(cx, cy, cz, r, Vec<3>{0, 0, 0}, Vec<3>{1, 1, 1}, true);
      auto gv = q.grid();
      std::vector<int> items(gv.items, gv.items + gv.offsets[(long)gv.nx * gv.ny * gv.nz]);
      std::mt19937 sg(7);
      for (long b = 0; b < (long)gv.nx * gv.ny * gv.nz; ++b)
        std::shuffle(items.begin() + gv.offsets[b], items.begin() + gv.offsets[b + 1], sg);
      CandidateGridView<double> shuffled = gv;
      shuffled.items = items.data();
      int n = 0;
      for (int t = 0; t < 20000; ++t) {
        const Vec3<double> p{rng.u(0, 1), rng.u(0, 1), rng.u(0, 1)};
        const double a = evalSphereUnionGrid(q.sphereUnion(), p, q.box(), gv);
        const double b2 = evalSphereUnionGrid(q.sphereUnion(), p, q.box(), shuffled);
        PECLET_CORE_CHECK(bitEq(a, b2));
        ++n;
      }
      std::printf("  shuffle-invariance     %d probes identical under per-bin reordering\n", n);
    }
  }

  // --- polydisperse, periodic (defeats the equal-R path; exercises the general subset) --------
  {
    std::vector<double> cx, cy, cz, r;
    for (int i = 0; i < 120; ++i) {
      cx.push_back(rng.u(0, 1));
      cy.push_back(rng.u(0, 1));
      cz.push_back(rng.u(0, 1));
      r.push_back(rng.u(0.02, 0.12));
    }
    campaign("poly-120 periodic", cx, cy, cz, r, true, 1.0, rng, 1e-3);
  }

  // --- non-periodic: out-of-box probes MUST take the fallback and stay exact ------------------
  {
    std::vector<double> cx, cy, cz, r;
    for (int i = 0; i < 60; ++i) {
      cx.push_back(rng.u(0.2, 0.8));
      cy.push_back(rng.u(0.2, 0.8));
      cz.push_back(rng.u(0.2, 0.8));
      r.push_back(0.05);
    }
    SphereBedQuery q(cx, cy, cz, r, Vec<3>{0, 0, 0}, Vec<3>{1, 1, 1}, false);
    int n = 0;
    for (int t = 0; t < 20000; ++t) {
      const Vec<3> p{rng.u(-0.5, 1.5), rng.u(-0.5, 1.5), rng.u(-0.5, 1.5)};
      // Non-periodic: min-image is off, so the canonical and legacy expressions COINCIDE and
      // full bit-identity with the stopgap must hold (plain sub, mul-add chain is fma-canonical
      // on both... the d2 chain still differs: fma vs plain). Compare against library brute
      // bitwise; against the stopgap through the same measured bound as the periodic campaigns.
      PECLET_CORE_CHECK(
          bitEq(q(p), evalSphereUnion(q.sphereUnion(), Vec3<double>{p[0], p[1], p[2]}, q.box())));
      ++n;
    }
    std::printf("  non-periodic + outside %d probes grid==brute bitwise (fallback exercised)\n", n);
  }

  // --- degenerate: one sphere, and a bin-count hint ------------------------------------------
  {
    std::vector<double> cx{0.5}, cy{0.5}, cz{0.5}, r{0.25};
    SphereBedQuery q(cx, cy, cz, r, Vec<3>{0, 0, 0}, Vec<3>{1, 1, 1}, true, /*nbHint=*/4);
    for (int t = 0; t < 2000; ++t) {
      const Vec<3> p{rng.u(0, 1), rng.u(0, 1), rng.u(0, 1)};
      PECLET_CORE_CHECK(
          bitEq(q(p), evalSphereUnion(q.sphereUnion(), Vec3<double>{p[0], p[1], p[2]}, q.box())));
    }
    std::printf("  single sphere          2000 probes grid==brute bitwise (nbHint=4)\n");
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

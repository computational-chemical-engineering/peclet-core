// Layer 3 rung 1 gate (suite/docs/ANALYTIC_SDF_GEOMETRY.md): CUT OWNERSHIP.
//
// `SceneQueryView::owner(p)` names the instance whose surface is nearest to p. Moving geometry
// reads its wall velocity off that instance, and resolved CFD-DEM posts the hydrodynamic force
// back to it, so an owner that disagreed with `eval`'s argmin would attribute wall motion — or a
// drag force — to the wrong body.
//
// Three claims, each gated separately:
//   (1) ACCELERATION-INVARIANCE. The candidate-grid owner equals an INDEPENDENT brute-force
//       argmin transcription, and does not move when every bin's list is shuffled. The superset
//       argument that makes `eval` bit-identical under acceleration carries to the argmin, but
//       only because the tie-break is by index and not by scan order — so this also gates that.
//   (2) VALUE CONSISTENCY. eval(p) is bitwise the value of the owner's own contribution. Any
//       drift between the two code paths shows up here.
//   (3) TIE DETERMINISM. Exactly coincident and exactly equidistant bodies resolve to the LOWEST
//       index through every path (grid, full scan, always-list composition).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#include "peclet/core/geom/scene_builder.hpp"
#include "peclet/core/geom/scene_query.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;

static bool bitEq(double a, double b) {
  std::uint64_t x, y;
  std::memcpy(&x, &a, 8);
  std::memcpy(&y, &b, 8);
  return x == y;
}

// Independent transcription of "nearest sphere surface", ascending scan with strict <, so the
// lowest index wins a tie. Deliberately written out rather than calling sphereDist2: the point is
// to have a second expression of the same formula.
static int refOwner(const std::vector<double>& cx, const std::vector<double>& cy,
                    const std::vector<double>& cz, const std::vector<double>& r, bool periodic,
                    double L, const Vec3<double>& p) {
  double best = 1e300;
  int bestI = -1;
  for (std::size_t i = 0; i < cx.size(); ++i) {
    double dx = p.x - cx[i], dy = p.y - cy[i], dz = p.z - cz[i];
    if (periodic) {
      dx = std::fma(-std::nearbyint(dx / L), L, dx);
      dy = std::fma(-std::nearbyint(dy / L), L, dy);
      dz = std::fma(-std::nearbyint(dz / L), L, dz);
    }
    const double d = std::sqrt(std::fma(dz, dz, std::fma(dy, dy, dx * dx))) - r[i];
    if (d < best) {
      best = d;
      bestI = static_cast<int>(i);
    }
  }
  return bestI;
}

struct Rng {
  std::mt19937_64 g{20260830ull};
  double u(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(g); }
};

// Build the SceneQueryView a consumer would capture for a sphere bed.
struct BedQ {
  SphereUnionView<double> u{};
  PeriodicBox<double> box{};
  CandidateGrid<double> grid;
  SceneQueryView<double> q{};
};

static BedQ makeBed(const std::vector<double>& cx, const std::vector<double>& cy,
                    const std::vector<double>& cz, const std::vector<double>& r, bool periodic,
                    double L) {
  BedQ b;
  b.box = PeriodicBox<double>{L, L, L, periodic};
  b.u.cx = cx.data();
  b.u.cy = cy.data();
  b.u.cz = cz.data();
  b.u.r = r.data();
  b.u.n = static_cast<int>(cx.size());
  b.u.equalR = true;
  for (std::size_t i = 1; i < r.size(); ++i)
    if (r[i] != r[0])
      b.u.equalR = false;
  b.u.r0 = r[0];
  b.grid = buildSphereCandidateGrid(b.u, Vec3<double>{0, 0, 0}, Vec3<double>{L, L, L}, b.box);
  b.q.u = b.u;
  b.q.box = b.box;
  b.q.grid = b.grid.view();
  return b;
}

static void bedCampaign(const char* name, const std::vector<double>& cx,
                        const std::vector<double>& cy, const std::vector<double>& cz,
                        const std::vector<double>& r, bool periodic, double L, Rng& rng,
                        int nProbe) {
  BedQ b = makeBed(cx, cy, cz, r, periodic, L);
  int mismatched = 0, valueMismatch = 0, fallbackMismatch = 0;
  for (int t = 0; t < nProbe; ++t) {
    const Vec3<double> p{rng.u(-0.1 * L, 1.1 * L), rng.u(-0.1 * L, 1.1 * L),
                         rng.u(-0.1 * L, 1.1 * L)};
    int coOwn = -1;
    const double coVal = b.q.evalOwner(p, coOwn);
    const int got = b.q.owner(p);
    const int want = refOwner(cx, cy, cz, r, periodic, L, p);
    if (got != want || coOwn != want)
      ++mismatched;
    // (2) the owner's own contribution IS eval's value, bitwise -- and the one-traversal
    // evalOwner returns bitwise the same number as eval, which is what lets a consumer sample the
    // field and the attribution together.
    const double own = std::sqrt(sphereDist2(b.u, got, p, b.box)) - r[(std::size_t)got];
    if (!bitEq(own, b.q.eval(p)) || !bitEq(coVal, b.q.eval(p)))
      ++valueMismatch;
    // grid vs brute-force owner path
    if (sphereUnionOwner(b.u, p, b.box) != got)
      ++fallbackMismatch;
  }
  PECLET_CORE_CHECK(mismatched == 0);
  PECLET_CORE_CHECK(valueMismatch == 0);
  PECLET_CORE_CHECK(fallbackMismatch == 0);
  std::printf("  %-22s %6d probes: owner==brute %d bad, eval==owner-value %d bad, grid==scan %d bad\n",
              name, nProbe, mismatched, valueMismatch, fallbackMismatch);
}

int main() {
  Rng rng;

  // --- the RCP acceptance bed (180 monodisperse spheres, unit periodic box) -------------------
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
    bedCampaign("RCP-180 equal-R", cx, cy, cz, r, true, 1.0, rng, 60000);

    // (1) shuffle-invariance of the OWNER (not just the value): the tie-break is by index.
    BedQ b = makeBed(cx, cy, cz, r, true, 1.0);
    auto gv = b.grid.view();
    std::vector<int> items(gv.items, gv.items + gv.offsets[(long)gv.nx * gv.ny * gv.nz]);
    std::mt19937 sg(11);
    for (long bb = 0; bb < (long)gv.nx * gv.ny * gv.nz; ++bb)
      std::shuffle(items.begin() + gv.offsets[bb], items.begin() + gv.offsets[bb + 1], sg);
    SceneQueryView<double> shuffled = b.q;
    CandidateGridView<double> sg2 = gv;
    sg2.items = items.data();
    shuffled.grid = sg2;
    int bad = 0;
    for (int t = 0; t < 30000; ++t) {
      const Vec3<double> p{rng.u(0, 1), rng.u(0, 1), rng.u(0, 1)};
      if (b.q.owner(p) != shuffled.owner(p))
        ++bad;
    }
    PECLET_CORE_CHECK(bad == 0);
    std::printf("  shuffle-invariance     30000 probes, %d owner changes under per-bin reorder\n",
                bad);
  }

  // --- polydisperse (defeats the equal-R path) ------------------------------------------------
  {
    std::vector<double> cx, cy, cz, r;
    for (int i = 0; i < 120; ++i) {
      cx.push_back(rng.u(0, 1));
      cy.push_back(rng.u(0, 1));
      cz.push_back(rng.u(0, 1));
      r.push_back(rng.u(0.02, 0.12));
    }
    bedCampaign("poly-120 periodic", cx, cy, cz, r, true, 1.0, rng, 40000);
  }

  // --- the 4-sphere flow bed (flow_probe geometry, N=32 cell units) ---------------------------
  {
    const double N = 32.0, R = 0.18 * N;
    std::vector<double> cx{0.25 * N, 0.75 * N, 0.75 * N, 0.25 * N};
    std::vector<double> cy{0.25 * N, 0.75 * N, 0.25 * N, 0.75 * N};
    std::vector<double> cz{0.25 * N, 0.25 * N, 0.75 * N, 0.75 * N};
    std::vector<double> r{R, R, R, R};
    bedCampaign("flow 4-sphere bed", cx, cy, cz, r, true, N, rng, 40000);
  }

  // --- (3) tie determinism ---------------------------------------------------------------------
  {
    // two coincident spheres: every probe must name index 0, through grid and full scan alike
    std::vector<double> cx{0.5, 0.5}, cy{0.5, 0.5}, cz{0.5, 0.5}, r{0.2, 0.2};
    BedQ b = makeBed(cx, cy, cz, r, true, 1.0);
    int bad = 0;
    for (int t = 0; t < 5000; ++t) {
      const Vec3<double> p{rng.u(0, 1), rng.u(0, 1), rng.u(0, 1)};
      if (b.q.owner(p) != 0 || sphereUnionOwner(b.u, p, b.box) != 0)
        ++bad;
    }
    PECLET_CORE_CHECK(bad == 0);
    // exact mid-plane between two equal spheres: x is the tie axis, so ANY probe on x=0.5 ties
    std::vector<double> mx{0.25, 0.75}, my{0.5, 0.5}, mz{0.5, 0.5}, mr{0.1, 0.1};
    BedQ m = makeBed(mx, my, mz, mr, false, 1.0);
    int badMid = 0;
    for (int t = 0; t < 2000; ++t) {
      const Vec3<double> p{0.5, rng.u(0.2, 0.8), rng.u(0.2, 0.8)};
      if (m.q.owner(p) != 0)
        ++badMid;
    }
    PECLET_CORE_CHECK(badMid == 0);
    std::printf("  tie determinism        coincident %d bad, exact mid-plane %d bad (lowest index)\n",
                bad, badMid);
  }

  // --- general scenes: always-list composition + fallback ---------------------------------------
  {
    SceneBuilder<double> b;
    const int sph = b.addLeaf(kSphere, {0.09});
    const int box = b.addLeaf(kBox, {0.07, 0.05, 0.06});
    const int ell = b.addLeaf(kEllipsoid, {0.10, 0.06, 0.08});  // NOT certified -> always-list
    Rng r2;
    for (int i = 0; i < 14; ++i)
      b.addInstance(sph, Transform<double>{Vec3<double>{r2.u(0, 1), r2.u(0, 1), r2.u(0, 1)}});
    for (int i = 0; i < 6; ++i)
      b.addInstance(box, Transform<double>{Vec3<double>{r2.u(0, 1), r2.u(0, 1), r2.u(0, 1)}});
    b.addInstance(ell, Transform<double>{Vec3<double>{0.5, 0.5, 0.5}});
    const SceneView<double> sv = b.view();
    for (int per = 0; per < 2; ++per) {
      const PeriodicBox<double> pb{1.0, 1.0, 1.0, per != 0};
      CandidateGrid<double> g =
          buildSceneCandidateGrid(sv, Vec3<double>{0, 0, 0}, Vec3<double>{1, 1, 1}, pb);
      const auto gv = g.view();
      const double* br = g.instBoundR.empty() ? nullptr : g.instBoundR.data();
      SceneQueryView<double> q;
      q.scene = sv;
      q.box = pb;
      q.grid = gv;
      q.instBoundR = br;
      PECLET_CORE_CHECK(gv.alwaysCount == 1);  // exactly the ellipsoid
      int bad = 0, valBad = 0;
      for (int t = 0; t < 20000; ++t) {
        const Vec3<double> p{r2.u(-0.2, 1.2), r2.u(-0.2, 1.2), r2.u(-0.2, 1.2)};
        int coOwn = -1;
        const double coVal = q.evalOwner(p, coOwn);
        const int got = q.owner(p);
        const int want = sceneOwnerPeriodic(sv, p, pb, br);
        if (got != want || coOwn != want)
          ++bad;
        // the owner's own instance value must be bitwise eval's, and so must evalOwner's
        if (!bitEq(evalInstancePeriodic(sv, got, p, pb, br ? br[got] : -1.0), q.eval(p)) ||
            !bitEq(coVal, q.eval(p)))
          ++valBad;
      }
      PECLET_CORE_CHECK(bad == 0);
      PECLET_CORE_CHECK(valBad == 0);
      std::printf("  general %-8s      20000 probes: grid owner==full scan %d bad, value %d bad\n",
                  per ? "periodic" : "open", bad, valBad);
    }
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

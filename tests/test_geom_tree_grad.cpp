// Gate for evalTreeGrad: the runtime tree's ANALYTIC gradient (value + gradient, one traversal).
//
// Claims:
//   1. VALUE PARITY -- the returned value is BITWISE evalTree's, over a battery of random trees
//      with nested transforms and CSG. Same expressions in the same order; if this ever drifts,
//      consumers using the pair would disagree with consumers using eval alone.
//   2. CHAIN-RULE EXACTNESS -- for a single leaf under a rotated + scaled + translated transform,
//      the tree gradient equals rotate(q, prim.grad(canonical point)) EXACTLY (the scale cancels
//      analytically; the code path is the same two operations).
//   3. FD AGREEMENT away from ridges -- central differences of evalTree converge to the analytic
//      gradient at O(h^2) on smooth regions of a composed tree.
//   4. THE RIDGE -- at the edge of a CSG difference (a drilled box), the analytic gradient IS the
//      active face's exact normal at any distance from the edge, while the central difference
//      SMEARS the two faces once the ridge is inside its stencil: measured as an angle error that
//      grows as the probe approaches the edge. This is the contact-normal failure mode for
//      composed particles with sharp features, and the reason the analytic gradient exists.
//   5. COST sanity -- evalTreeGrad is cheaper than the 6-extra-eval central difference it
//      replaces (reported, not asserted; wall-clock on the host build).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "peclet/core/geom/scene_builder.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;

static bool bitEq(double a, double b) {
  std::uint64_t x, y;
  std::memcpy(&x, &a, 8);
  std::memcpy(&y, &b, 8);
  return x == y;
}

template <class F>
static Vec3<double> fdGrad(const F& f, Vec3<double> p, double h) {
  return Vec3<double>{(f(Vec3<double>{p.x + h, p.y, p.z}) - f(Vec3<double>{p.x - h, p.y, p.z})) /
                          (2 * h),
                      (f(Vec3<double>{p.x, p.y + h, p.z}) - f(Vec3<double>{p.x, p.y - h, p.z})) /
                          (2 * h),
                      (f(Vec3<double>{p.x, p.y, p.z + h}) - f(Vec3<double>{p.x, p.y, p.z - h})) /
                          (2 * h)};
}

static double norm3(Vec3<double> v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

static double angleDeg(Vec3<double> a, Vec3<double> b) {
  const double d = (a.x * b.x + a.y * b.y + a.z * b.z) / (norm3(a) * norm3(b));
  return std::acos(std::fmin(1.0, std::fmax(-1.0, d))) * 180.0 / M_PI;
}

int main() {
  std::mt19937_64 rng(20260830ull);
  auto u = [&](double lo, double hi) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
  };
  auto randQ = [&]() {
    Quat<double> q{u(-1, 1), u(-1, 1), u(-1, 1), u(-1, 1)};
    const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return Quat<double>{q.x / n, q.y / n, q.z / n, q.w / n};
  };

  // --- a composed test tree: (rotated box  U  scaled torus)  minus  rotated capsule -----------
  SceneBuilder<double> b;
  Transform<double> tb;
  tb.translation = Vec3<double>{0.1, -0.05, 0.02};
  tb.rotation = randQ();
  const int box = b.addLeaf(kBox, {0.4, 0.3, 0.25}, tb);
  Transform<double> tt;
  tt.translation = Vec3<double>{-0.2, 0.15, 0.0};
  tt.rotation = randQ();
  tt.scale = 1.4;
  const int tor = b.addLeaf(kTorus, {0.3, 0.1}, tt);
  Transform<double> tc;
  tc.translation = Vec3<double>{0.05, 0.0, 0.15};
  tc.rotation = randQ();
  tc.scale = 0.8;
  const int cap = b.addLeaf(kCapsule, {0.12, 0.3}, tc);
  const int uni = b.addUnion(box, tor);
  Transform<double> troot;
  troot.rotation = randQ();
  troot.scale = 1.2;
  const int root = b.addDifference(uni, cap, troot);
  const SceneView<double> sv = b.view();
  auto evalV = [&](Vec3<double> p) {
    return evalTree<double>(TablePtr<ShapeNode<double>>{sv.nodes}, sv.nodeCount, root, p,
                            TablePtr<GridDesc<double>>{sv.grids}, PoolPtr<float>{sv.samples});
  };
  auto evalG = [&](Vec3<double> p, Vec3<double>& g) {
    return evalTreeGrad<double>(TablePtr<ShapeNode<double>>{sv.nodes}, sv.nodeCount, root, p,
                                TablePtr<GridDesc<double>>{sv.grids}, PoolPtr<float>{sv.samples},
                                g);
  };

  // 1. value parity, bitwise, over 50k probes
  {
    int bad = 0;
    for (int t = 0; t < 50000; ++t) {
      const Vec3<double> p{u(-1.2, 1.2), u(-1.2, 1.2), u(-1.2, 1.2)};
      Vec3<double> g;
      if (!bitEq(evalG(p, g), evalV(p)))
        ++bad;
    }
    PECLET_CORE_CHECK(bad == 0);
    std::printf("  value parity           50000 probes, %d bitwise mismatches vs evalTree\n", bad);
  }

  // 2. chain-rule exactness on a single transformed leaf
  {
    SceneBuilder<double> b2;
    Transform<double> tr;
    tr.translation = Vec3<double>{0.3, -0.2, 0.1};
    tr.rotation = randQ();
    tr.scale = 1.7;
    const int leaf = b2.addLeaf(kCone, {0.25, 0.1, 0.3}, tr);
    const SceneView<double> s2 = b2.view();
    double worst = 0;
    for (int t = 0; t < 20000; ++t) {
      const Vec3<double> p{u(-1, 1), u(-1, 1), u(-1, 1)};
      Vec3<double> g;
      (void)evalTreeGrad<double>(TablePtr<ShapeNode<double>>{s2.nodes}, s2.nodeCount, leaf, p,
                                 TablePtr<GridDesc<double>>{s2.grids}, PoolPtr<float>{s2.samples},
                                 g);
      const Vec3<double> q = toCanonical(tr, p);
      const Vec3<double> ref = rotate(tr.rotation, prim::Cone<double>{0.25, 0.1, 0.3}.grad(q));
      worst = std::fmax(worst, std::fabs(g.x - ref.x) + std::fabs(g.y - ref.y) +
                                   std::fabs(g.z - ref.z));
    }
    PECLET_CORE_CHECK(worst == 0.0);  // same two operations, so exactly equal
    std::printf("  chain-rule exactness   rotated+scaled cone: worst |diff| = %.1e (exact)\n",
                worst);
  }

  // 3. FD agreement away from ridges: O(h^2)
  {
    double worst4 = 0, worst5 = 0;
    int used = 0;
    for (int t = 0; t < 4000 && used < 1500; ++t) {
      const Vec3<double> p{u(-1.0, 1.0), u(-1.0, 1.0), u(-1.0, 1.0)};
      Vec3<double> g;
      (void)evalG(p, g);
      // stay away from ridges: both FD steps must see the same smooth branch. Filter by
      // comparing FD at two step sizes -- near a ridge they disagree at O(1).
      const Vec3<double> g4 = fdGrad(evalV, p, 1e-4), g5 = fdGrad(evalV, p, 1e-5);
      if (angleDeg(g4, g5) > 0.01)
        continue;  // ridge (or grid-clamp) inside the stencil; case 4 covers those
      ++used;
      worst4 = std::fmax(worst4, norm3(Vec3<double>{g4.x - g.x, g4.y - g.y, g4.z - g.z}));
      worst5 = std::fmax(worst5, norm3(Vec3<double>{g5.x - g.x, g5.y - g.y, g5.z - g.z}));
    }
    // the residual is the FD REFERENCE's truncation, so the check is the O(h^2) ratio between
    // the two step sizes (100x for h 1e-4 -> 1e-5), plus a loose absolute lid
    PECLET_CORE_CHECK(worst4 < 2e-5);
    PECLET_CORE_CHECK(worst5 < worst4 / 50.0);
    std::printf("  FD agreement           %d smooth probes: |fd - exact| %.1e at h=1e-4, %.1e at "
                "h=1e-5 (O(h^2))\n",
                used, worst4, worst5);
  }

  // 4. THE RIDGE: a drilled box (box minus cylinder-ish capsule through the top face). Probe at
  // distance delta from the rim edge, on the flat top face: the true normal is EXACTLY +z.
  {
    SceneBuilder<double> b3;
    const int bx = b3.addLeaf(kBox, {0.5, 0.5, 0.2});
    Transform<double> th;
    th.rotation = Quat<double>{std::sin(M_PI / 4), 0, 0, std::cos(M_PI / 4)};  // capsule y -> z
    const int hole = b3.addLeaf(kCapsule, {0.15, 0.5}, th);
    const int drilled = b3.addDifference(bx, hole);
    const SceneView<double> s3 = b3.view();
    auto v3 = [&](Vec3<double> p) {
      return evalTree<double>(TablePtr<ShapeNode<double>>{s3.nodes}, s3.nodeCount, drilled, p,
                              TablePtr<GridDesc<double>>{s3.grids}, PoolPtr<float>{s3.samples});
    };
    std::printf("  ridge (drilled box rim, true normal +z; FD step 1e-4):\n");
    std::printf("    %-12s %-18s %-18s\n", "dist to rim", "analytic err(deg)", "FD err(deg)");
    const Vec3<double> ez{0, 0, 1};
    double fdWorst = 0;
    for (double delta : {1e-2, 1e-3, 1e-4, 1e-5}) {
      const Vec3<double> p{0.15 + delta, 0.0, 0.2};  // ON the top face, delta outside the rim
      Vec3<double> g;
      (void)evalTreeGrad<double>(TablePtr<ShapeNode<double>>{s3.nodes}, s3.nodeCount, drilled, p,
                                 TablePtr<GridDesc<double>>{s3.grids}, PoolPtr<float>{s3.samples},
                                 g);
      const Vec3<double> gf = fdGrad(v3, p, 1e-4);
      const double ea = angleDeg(g, ez), ef = angleDeg(gf, ez);
      std::printf("    %-12.0e %-18.6f %-18.3f\n", delta, ea, ef);
      PECLET_CORE_CHECK(ea < 1e-6);  // the active face's EXACT normal at every distance
      fdWorst = std::fmax(fdWorst, ef);
    }
    PECLET_CORE_CHECK(fdWorst > 10.0);  // the smear is real: tens of degrees inside the stencil
  }

  // 5. cost: one evalTreeGrad vs eval + 6-eval central difference (what dem ran before)
  {
    volatile double sink = 0;
    const int N = 200000;
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < N; ++t) {
      const Vec3<double> p{u(-1, 1), u(-1, 1), u(-1, 1)};
      Vec3<double> g;
      sink += evalG(p, g) + g.x;
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int t = 0; t < N; ++t) {
      const Vec3<double> p{u(-1, 1), u(-1, 1), u(-1, 1)};
      const Vec3<double> g = fdGrad(evalV, p, 1e-4);
      sink += evalV(p) + g.x;
    }
    auto t2 = std::chrono::steady_clock::now();
    const double a = std::chrono::duration<double>(t1 - t0).count();
    const double f = std::chrono::duration<double>(t2 - t1).count();
    std::printf("  cost                   analytic %.0f ns/probe vs eval+FD %.0f ns/probe "
                "(%.1fx)\n",
                a / N * 1e9, f / N * 1e9, f / a);
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

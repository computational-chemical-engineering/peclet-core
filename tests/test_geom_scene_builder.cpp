// Layer 0 rung 5 gate (suite/docs/ANALYTIC_SDF_GEOMETRY.md): host scene assembly + the flat
// encoding the Python bindings will speak.
//
// The load-bearing property is a ROUND TRIP: a scene composed with SceneBuilder, encoded to flat
// int/Real arrays, decoded back, and re-evaluated must give BIT-IDENTICAL distances. That is what
// makes the encoding a contract rather than a convention -- a binding that writes numpy arrays in
// this layout gets exactly the geometry the C++ side would have built.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "peclet/core/geom/scene_builder.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;
using V = Vec3<double>;

static bool bitEq(double a, double b) {
  std::uint64_t x, y;
  std::memcpy(&x, &a, 8);
  std::memcpy(&y, &b, 8);
  return x == y;
}

struct Rng {
  std::uint64_t s = 0x243F6A8885A308D3ull;
  double u(double lo, double hi) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return lo + (hi - lo) * (double)((s >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
  }
};

int main() {
  Rng rng;

  // ---------------------------------------------------------------------------------------
  // A scene a real consumer would build: a spinning stirrer (shaft tube UNION two blades, one
  // rotated), a static container the fluid sits inside (complement-style difference), and a
  // sampled-grid body -- three instances, mixed leaf kinds, CSG, transforms, velocities.
  // ---------------------------------------------------------------------------------------
  SceneBuilder<double> b;
  const int shaft = b.addLeaf(kHollowCylinder, {0.30, 3.00, 0.30});
  const int bladeA = b.addLeaf(kBox, {0.90, 0.10, 0.25}, Transform<double>{V{1.0, 0.5, 0.0}});
  const double a45 = 0.7853981633974483;
  const int bladeB = b.addLeaf(
      kBox, {0.90, 0.10, 0.25},
      Transform<double>{V{-1.0, -0.5, 0.0}, Quat<double>{0, std::sin(a45), 0, std::cos(a45)}, 1.0});
  const int blades = b.addUnion(bladeA, bladeB);
  const int stirrer = b.addUnion(shaft, blades);

  const int ball = b.addLeaf(kSphere, {0.8});
  const int cut = b.addLeaf(kTorus, {0.5, 0.2});
  const int carved = b.addDifference(ball, cut);

  const int G = 6;
  std::vector<float> samples((std::size_t)G * G * G);
  for (int k = 0; k < G; ++k)
    for (int j = 0; j < G; ++j)
      for (int i = 0; i < G; ++i) {
        const float x = -1.0f + 0.4f * i, y = -1.0f + 0.4f * j, z = -1.0f + 0.4f * k;
        samples[i + G * (j + G * k)] = std::sqrt(x * x + y * y + z * z) - 0.55f;
      }
  const int grid =
      b.addGrid(samples, G, G, G, V{-1, -1, -1}, V{0.4, 0.4, 0.4}, GridExtension::kObject);

  b.addInstance(stirrer, Transform<double>{V{0.0, 0.0, 0.0}}, V{0, 0, 0}, V{0, 10.0, 0}, V{0, 0, 0},
                /*materialId=*/7);
  b.addInstance(carved, Transform<double>{V{2.5, 0.0, 0.0}, Quat<double>{0, 0, 0, 1}, 0.75});
  b.addInstance(grid, Transform<double>{V{-2.5, 0.0, 0.0}});

  PECLET_CORE_CHECK(b.nodes().size() == 9);
  PECLET_CORE_CHECK(b.instances().size() == 3);
  PECLET_CORE_CHECK(b.grids().size() == 1);
  PECLET_CORE_CHECK(b.samples().size() == (std::size_t)G * G * G);

  // ---------------------------------------------------------------------------------------
  // (1) ROUND TRIP: encode -> decode -> re-evaluate, bit-identical everywhere.
  // ---------------------------------------------------------------------------------------
  std::vector<int> ni, ii;
  std::vector<double> nr, ir;
  b.encode(ni, nr, ii, ir);
  PECLET_CORE_CHECK(ni.size() == b.nodes().size() * kNodeIntStride);
  PECLET_CORE_CHECK(nr.size() == b.nodes().size() * kNodeRealStride);
  PECLET_CORE_CHECK(ii.size() == b.instances().size() * kInstanceIntStride);
  PECLET_CORE_CHECK(ir.size() == b.instances().size() * kInstanceRealStride);

  SceneBuilder<double> b2 = SceneBuilder<double>::decode(ni, nr, ii, ir, b.grids(), b.samples());
  const SceneView<double> s1 = b.view(), s2 = b2.view();
  PECLET_CORE_CHECK(s2.nodeCount == s1.nodeCount && s2.instanceCount == s1.instanceCount);

  int probes = 0, mismatches = 0;
  for (int t = 0; t < 20000; ++t) {
    const V p{rng.u(-4, 4), rng.u(-4, 4), rng.u(-4, 4)};
    if (!bitEq(evalScene(s1, p), evalScene(s2, p)))
      ++mismatches;
    for (int i = 0; i < s1.instanceCount; ++i)
      if (!bitEq(evalInstance(s1, i, p), evalInstance(s2, i, p)))
        ++mismatches;
    ++probes;
  }
  PECLET_CORE_CHECK(probes == 20000);
  PECLET_CORE_CHECK(mismatches == 0);

  // ---------------------------------------------------------------------------------------
  // (2) The scene means what it should: each instance is solid where its shape is, the scene is
  //     the union, and the instance transform (translate + isotropic scale) really applies.
  // ---------------------------------------------------------------------------------------
  {
    // shaft interior, in the stirrer instance at the origin
    PECLET_CORE_CHECK(evalInstance(s1, 0, V{0.15, 0.0, 0.0}) < 0);
    // the carved ball instance sits at x = 2.5 scaled by 0.75 -> radius 0.6
    PECLET_CORE_CHECK(evalInstance(s1, 1, V{2.5, 0.0, 0.0}) < 0);
    PECLET_CORE_CHECK(evalInstance(s1, 1, V{2.5 + 0.7, 0.0, 0.0}) > 0);
    // the torus cut is subtracted: a point on the cut ring must be VOID inside the ball
    PECLET_CORE_CHECK(evalInstance(s1, 1, V{2.5 + 0.75 * 0.5, 0.0, 0.0}) > 0);
    // union: the scene is negative wherever any instance is
    for (int t = 0; t < 4000; ++t) {
      const V p{rng.u(-4, 4), rng.u(-4, 4), rng.u(-4, 4)};
      double m = 1e9;
      for (int i = 0; i < s1.instanceCount; ++i)
        m = std::fmin(m, evalInstance(s1, i, p));
      PECLET_CORE_CHECK(bitEq(evalScene(s1, p), m));
    }
  }

  // ---------------------------------------------------------------------------------------
  // (3) Candidate lists (contract 9) must agree with the full sweep when they list everything,
  //     and be the partial union when they do not -- this is the hook acceleration plugs into.
  // ---------------------------------------------------------------------------------------
  {
    const int all[3] = {0, 1, 2};
    const int one[1] = {1};
    for (int t = 0; t < 4000; ++t) {
      const V p{rng.u(-4, 4), rng.u(-4, 4), rng.u(-4, 4)};
      PECLET_CORE_CHECK(bitEq(evalCandidates(s1, p, all, 3), evalScene(s1, p)));
      PECLET_CORE_CHECK(bitEq(evalCandidates(s1, p, one, 1), evalInstance(s1, 1, p)));
    }
    const int none[1] = {0};
    PECLET_CORE_CHECK(evalCandidates(s1, V{0, 0, 0}, none, 0) == 1e9);  // empty list = "far"
  }

  // ---------------------------------------------------------------------------------------
  // (4) Rigid-body surface velocity (contract 7) — what Layer 3's moving-wall BC reads.
  // ---------------------------------------------------------------------------------------
  {
    const Instance<double>& stir = b.instances()[0];  // angVel = (0, 10, 0) about the origin
    const V v = instanceVelocity(stir, V{1.0, 0.0, 0.0});
    PECLET_CORE_CHECK(std::fabs(v.x - 0.0) < 1e-12);
    PECLET_CORE_CHECK(std::fabs(v.y - 0.0) < 1e-12);
    PECLET_CORE_CHECK(std::fabs(v.z + 10.0) < 1e-12);  // omega_y x r_x = -10 z
    const V vs = instanceVelocity(b.instances()[1], V{9, 9, 9});
    PECLET_CORE_CHECK(vs.x == 0.0 && vs.y == 0.0 && vs.z == 0.0);  // static instance
  }

  // ---------------------------------------------------------------------------------------
  // (5) The builder rejects malformed scenes rather than producing a corrupt one.
  // ---------------------------------------------------------------------------------------
  {
    SceneBuilder<double> bad;
    bool threw = false;
    try {
      bad.addUnion(0, 1);  // no nodes yet
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    PECLET_CORE_CHECK(threw);
    threw = false;
    try {
      bad.addLeaf(kUnion, {1.0});  // an operator is not a leaf
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    PECLET_CORE_CHECK(threw);
    threw = false;
    try {
      bad.addGrid(std::vector<float>(7), 2, 2, 2, V{0, 0, 0}, V{1, 1, 1}, GridExtension::kClamp);
    } catch (const std::invalid_argument&) {
      threw = true;  // 7 != 2*2*2
    }
    PECLET_CORE_CHECK(threw);
    threw = false;
    try {
      SceneBuilder<double>::decode({1, 2}, {}, {}, {}, {}, {});  // ragged node arrays
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    PECLET_CORE_CHECK(threw);
  }

  // ---------------------------------------------------------------------------------------
  // (6) The GENERAL surface-point generator (Layer 1): SDF-driven, so it works for shapes that
  //     have no hand-written generator, including CSG results that are not star-shaped.
  // ---------------------------------------------------------------------------------------
  {
    struct Case {
      const char* name;
      int root;
      double lo, hi;
      double area;  // analytic surface area, for a sanity check on the point count
    };
    SceneBuilder<double> g;
    const int sph = g.addLeaf(kSphere, {0.5});
    const int tor = g.addLeaf(kTorus, {0.5, 0.15});
    const int bx = g.addLeaf(kBox, {0.4, 0.3, 0.2});
    const int sp2 = g.addLeaf(kSphere, {0.3}, Transform<double>{V{0.25, 0, 0}});
    const int csg = g.addDifference(bx, sp2);  // a box with a bite taken out: not star-shaped
    g.addInstance(sph);
    const SceneView<double> gv = g.view();

    const Case cases[] = {{"sphere r=0.5", sph, -0.7, 0.7, 4 * M_PI * 0.25},
                          {"torus R=.5 r=.15", tor, -0.75, 0.75, 4 * M_PI * M_PI * 0.5 * 0.15},
                          {"box .4x.3x.2", bx, -0.55, 0.55, 2 * (0.8 * 0.6 + 0.8 * 0.4 + 0.6 * 0.4)},
                          {"box MINUS sphere", csg, -0.6, 0.7, 0.0}};
    for (const Case& c : cases) {
      const double sp = 0.05;
      const std::vector<V> pts =
          surfacePoints<double>(gv, c.root, sp, V{c.lo, c.lo, c.lo}, V{c.hi, c.hi, c.hi});
      // every returned point must actually lie on the zero level set
      double worst = 0;
      for (const V& p : pts)
        worst = std::fmax(worst, std::fabs(evalTree<double>(TablePtr<ShapeNode<double>>{gv.nodes},
                                                            gv.nodeCount, c.root, p,
                                                            TablePtr<GridDesc<double>>{gv.grids},
                                                            PoolPtr<float>{gv.samples})));
      std::printf("  surfacePoints %-18s %5zu pts, worst |sdf| = %.2e\n", c.name, pts.size(),
                  worst);
      PECLET_CORE_CHECK(pts.size() > 100);
      PECLET_CORE_CHECK(worst < sp * 0.05);
      if (c.area > 0) {  // point count should track area / spacing^2 within a small factor
        const double expect = c.area / (sp * sp);
        PECLET_CORE_CHECK(pts.size() > 0.2 * expect && pts.size() < 5.0 * expect);
      }
    }
  }

  std::printf("scene round-trip: %d probes x %d instances, 0 mismatches\n", probes,
              s1.instanceCount);
  PECLET_CORE_RETURN_TEST_RESULT();
}

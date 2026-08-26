// Layer 0 rung 0 gate (suite/docs/ANALYTIC_SDF_GEOMETRY.md): the shared device-callable leaf
// primitives in peclet/core/geom/primitives.hpp reproduce their origins EXACTLY.
//
// What this test does and does not prove:
//   * The `float` leaves are compared BIT-FOR-BIT against reference functions transcribed verbatim
//     from dem/src/dem_portable.hpp:79-103 into this file (Kokkos::sqrt/fabs/fmax/fmin -> std::,
//     which is the same libm call on host). This catches any drift in the leaf bodies — association
//     order, fmax-vs-`<`, a "simplified" expression. It does NOT link dem's compiled code; the
//     cross-check against dem itself is rung 3's gate (dem's own ctest suite, host + CUDA).
//   * The `double` leaves are compared against the existing host geom/sdf.hpp shapes where the
//     formulas coincide (Sphere, Box, and HollowCylinderShell vs geom::HollowCylinder).
//   * Building at all proves the host-only path: this TU never includes Kokkos, so PECLET_HD must
//     degrade to plain `inline`.
#include <cmath>
#include <cstdint>
#include <cstring>

#include "peclet/core/common/portable.hpp"
#include "peclet/core/geom/primitives.hpp"
#include "peclet/core/geom/sdf.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;
namespace prim = peclet::core::geom::prim;

// --- Reference: verbatim transcription of dem/src/dem_portable.hpp:79-103 -------------------
// Kept textually as close to the origin as C++ allows so a reader can diff the two side by side.

static float ref_sdfSphere(float px, float py, float pz, float radius) {
  return std::sqrt(px * px + py * py + pz * pz) - radius;
}

static float ref_sdfHollowCylinder(float px, float py, float pz, float r_outer, float h,
                                   float thick) {
  const float r = std::sqrt(px * px + pz * pz);
  const float r_mid = r_outer - thick * 0.5f;
  const float dx = std::fabs(r - r_mid) - thick * 0.5f;
  const float dy = std::fabs(py) - h * 0.5f;
  const float ox = std::fmax(dx, 0.0f), oy = std::fmax(dy, 0.0f);
  const float outside = std::sqrt(ox * ox + oy * oy);
  const float inside = std::fmin(std::fmax(dx, dy), 0.0f);
  return outside + inside;
}

static float ref_sdfBox(float px, float py, float pz, float hx, float hy, float hz) {
  const float dx = std::fabs(px) - hx;
  const float dy = std::fabs(py) - hy;
  const float dz = std::fabs(pz) - hz;
  const float ox = std::fmax(dx, 0.0f), oy = std::fmax(dy, 0.0f), oz = std::fmax(dz, 0.0f);
  const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
  const float inside = std::fmin(std::fmax(dx, std::fmax(dy, dz)), 0.0f);
  return outside + inside;
}

// --- Bit-exact float comparison (catches -0.0 vs 0.0 and any NaN payload difference) ----------
static bool bitEqual(float a, float b) {
  std::uint32_t ba, bb;
  std::memcpy(&ba, &a, sizeof ba);
  std::memcpy(&bb, &b, sizeof bb);
  return ba == bb;
}

// Deterministic point cloud: a coarse lattice covering inside / surface / edge / corner / far
// field in every octant, plus an LCG tail for coverage the lattice misses. Fixed seed — this test
// must never be flaky.
struct PointCloud {
  static constexpr int kLatticeN = 13;
  static constexpr int kRandom = 4000;
  std::uint32_t state = 12345u;

  float nextUniform(float lo, float hi) {
    state = state * 1664525u + 1013904223u;
    const float u = static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0x1000000u);
    return lo + (hi - lo) * u;
  }
};

int main() {
  PointCloud pc;

  // Parameters chosen so the lattice below straddles every feature: the sphere surface, the box
  // faces/edges/corners, and the tube's inner wall, outer wall, mid-surface and end caps.
  const float R = 1.25f;
  const float BX = 0.75f, BY = 1.10f, BZ = 0.40f;
  const float CO = 1.30f, CH = 1.60f, CT = 0.50f;

  prim::Sphere<float> sphF{R};
  prim::Box<float> boxF{BX, BY, BZ};
  prim::HollowCylinder<float> hcF{CO, CH, CT};

  int checked = 0;

  auto compareAt = [&](float x, float y, float z) {
    const Vec3<float> p{x, y, z};
    if (!bitEqual(sphF.eval(p), ref_sdfSphere(x, y, z, R))) {
      std::fprintf(stderr, "Sphere mismatch at (%.9g, %.9g, %.9g): leaf=%.9g ref=%.9g\n", x, y, z,
                   sphF.eval(p), ref_sdfSphere(x, y, z, R));
      ++peclet::core::test::g_failures;
    }
    if (!bitEqual(boxF.eval(p), ref_sdfBox(x, y, z, BX, BY, BZ))) {
      std::fprintf(stderr, "Box mismatch at (%.9g, %.9g, %.9g): leaf=%.9g ref=%.9g\n", x, y, z,
                   boxF.eval(p), ref_sdfBox(x, y, z, BX, BY, BZ));
      ++peclet::core::test::g_failures;
    }
    if (!bitEqual(hcF.eval(p), ref_sdfHollowCylinder(x, y, z, CO, CH, CT))) {
      std::fprintf(stderr, "HollowCylinder mismatch at (%.9g, %.9g, %.9g): leaf=%.9g ref=%.9g\n", x,
                   y, z, hcF.eval(p), ref_sdfHollowCylinder(x, y, z, CO, CH, CT));
      ++peclet::core::test::g_failures;
    }
    ++checked;
  };

  // (a) lattice over [-2, 2]^3 — hits interiors, exteriors and (by construction of the extents
  //     above) points very near each surface.
  for (int i = 0; i < PointCloud::kLatticeN; ++i)
    for (int j = 0; j < PointCloud::kLatticeN; ++j)
      for (int k = 0; k < PointCloud::kLatticeN; ++k) {
        const float t = 4.0f / static_cast<float>(PointCloud::kLatticeN - 1);
        compareAt(-2.0f + t * i, -2.0f + t * j, -2.0f + t * k);
      }

  // (b) exact-feature points: origin, on-surface, on box faces/edges/corners, tube walls and caps,
  //     and the degenerate axis point where the cylinder's r = 0.
  const float feat[][3] = {
      {0, 0, 0},        {R, 0, 0},         {0, R, 0},          {0, 0, R},
      {-R, 0, 0},       {BX, 0, 0},        {BX, BY, 0},        {BX, BY, BZ},
      {-BX, -BY, -BZ},  {BX, BY, 2.0f},    {CO, 0, 0},         {CO - CT, 0, 0},
      {CO - CT * 0.5f, 0, 0},              {0, CH * 0.5f, 0},  {CO, CH * 0.5f, 0},
      {0, 0, 0.0f},     {0, CH, 0},        {0, -CH, 0},        {1e-8f, 0, 1e-8f},
  };
  for (const auto& f : feat)
    compareAt(f[0], f[1], f[2]);

  // (c) pseudo-random tail, including a far-field band so the `outside` branch dominates.
  for (int n = 0; n < PointCloud::kRandom; ++n) {
    const float x = pc.nextUniform(-3.0f, 3.0f);
    const float y = pc.nextUniform(-3.0f, 3.0f);
    compareAt(x, y, pc.nextUniform(-3.0f, 3.0f));
  }

  PECLET_CORE_CHECK(checked > 2000);

  // --- double leaves vs the existing host geom/sdf.hpp shapes (formulas coincide) --------------
  {
    prim::Sphere<double> sD{2.0};
    geom::Sphere hostS{{0, 0, 0}, 2.0};
    prim::Box<double> bD{1.0, 1.0, 1.0};
    geom::Box hostB{{0, 0, 0}, {1, 1, 1}};
    // geom::HollowCylinder is the max-of-halfspaces form -> HollowCylinderShell, not
    // HollowCylinder.
    prim::HollowCylinderShell<double> shD{3.0, 1.0, 4.0};
    geom::HollowCylinder hostHC{{0, 0, 0}, 3.0, 1.0, 4.0, 2};

    const double pts[][3] = {{0, 0, 0},   {3, 0, 0},        {2, 0, 0},      {0.5, 0.5, 0.5},
                             {1, 0, 0},   {2, 0, 3},        {5, 0, 0},      {-1.3, 2.2, -0.7},
                             {0, 0, 2.0}, {2.9, 0.1, 1.99}, {1.0, 1.0, 1.0}};
    for (const auto& q : pts) {
      const Vec3<double> p{q[0], q[1], q[2]};
      const Vec<3> hp{q[0], q[1], q[2]};
      PECLET_CORE_CHECK(sD.eval(p) == hostS.eval(hp));
      PECLET_CORE_CHECK(bD.eval(p) == hostB.eval(hp));
      PECLET_CORE_CHECK(shD.eval(p) == hostHC.eval(hp));
    }
  }

  // --- the TRAP: the two cylinder forms are genuinely different functions -----------------------
  // Guards against a future "consolidation" quietly collapsing them. Same tube (rOuter 1.3,
  // rInner 0.8, height 1.6), a point diagonally outside the top rim: the distance-exact form must
  // report a strictly LARGER distance than the halfspace-max lower bound.
  {
    prim::HollowCylinder<double> exact{1.3, 1.6, 0.5};
    prim::HollowCylinderShell<double> shell{1.3, 0.8, 1.6};
    const Vec3<double> pExactFrame{1.8, 1.4, 0.0};  // exact form: tube about y
    const Vec3<double> pShellFrame{1.8, 0.0, 1.4};  // shell form: tube about z
    const double a = exact.eval(pExactFrame), b = shell.eval(pShellFrame);
    PECLET_CORE_CHECK(a > 0 && b > 0);
    PECLET_CORE_CHECK(a > b + 1e-9);  // halfspace max is a strict under-estimate off the rim
  }

  // --- sign convention + Complement --------------------------------------------------------
  {
    prim::Sphere<double> s{2.0};
    prim::Complement<prim::Sphere<double>> c{s};
    const Vec3<double> inside{0, 0, 0}, outside{3, 0, 0};
    PECLET_CORE_CHECK(s.eval(inside) < 0 && s.eval(outside) > 0);
    PECLET_CORE_CHECK(c.eval(inside) > 0 && c.eval(outside) < 0);
    // finite-difference gradient is outward and unit-length for a distance-exact leaf
    const Vec3<double> g = prim::gradient(s, outside, 1e-6);
    PECLET_CORE_CHECK(std::fabs(std::sqrt(g.x * g.x + g.y * g.y + g.z * g.z) - 1.0) < 1e-6);
    PECLET_CORE_CHECK(g.x > 0.9);
  }

  // exact_distance annotations are what Layer 2's crossing/aperture code will branch on.
  static_assert(prim::Sphere<double>::exact_distance, "sphere is a true distance");
  static_assert(prim::Box<double>::exact_distance, "box is a true distance");
  static_assert(prim::HollowCylinder<double>::exact_distance, "dem tube is a true distance");
  static_assert(!prim::HollowCylinderShell<double>::exact_distance, "halfspace max: sign only");

  std::printf("compared %d points bit-exact against the dem reference\n", checked);
  PECLET_CORE_RETURN_TEST_RESULT();
}

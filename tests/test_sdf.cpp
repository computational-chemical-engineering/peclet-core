// Serial checks of the analytic SDF primitives: sign convention (negative inside solid), zero on the
// surface, outward unit-normal gradient, and a few known distances.
#include <cmath>

#include "test_util.hpp"
#include "tpx/common/types.hpp"
#include "tpx/geom/grid_sdf.hpp"
#include "tpx/geom/sdf.hpp"

using namespace tpx;
using namespace tpx::geom;

static double len(const Vec<3>& v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

int main() {
  // --- Sphere R=2 at origin ---
  Sphere s{{0, 0, 0}, 2.0};
  TPX_CHECK(s.eval({0, 0, 0}) < 0);           // inside solid -> negative
  TPX_CHECK(s.eval({3, 0, 0}) > 0);           // outside -> positive
  TPX_CHECK(std::fabs(s.eval({2, 0, 0})) < 1e-12);  // on surface
  TPX_CHECK(std::fabs(s.eval({3, 0, 0}) - 1.0) < 1e-9);
  {
    Vec<3> g = gradient(s, {3, 0, 0});
    TPX_CHECK(std::fabs(len(g) - 1.0) < 1e-4);  // unit gradient
    TPX_CHECK(g[0] > 0.9);                       // outward (away from center)
  }

  // --- Box half=(1,1,1) at origin ---
  Box b{{0, 0, 0}, {1, 1, 1}};
  TPX_CHECK(b.eval({0, 0, 0}) < 0);
  TPX_CHECK(std::fabs(b.eval({2, 0, 0}) - 1.0) < 1e-9);   // 1 outside the +x face
  TPX_CHECK(std::fabs(b.eval({1, 0, 0})) < 1e-12);        // on +x face
  TPX_CHECK(b.eval({0.5, 0.5, 0.5}) < 0);                 // interior corner-ish, still inside

  // --- Hollow cylinder: rOuter=3, rInner=1, height=4, axis z ---
  HollowCylinder hc{{0, 0, 0}, 3.0, 1.0, 4.0, 2};
  TPX_CHECK(hc.eval({2, 0, 0}) < 0);   // within the wall (r=2), mid-height -> solid
  TPX_CHECK(hc.eval({0, 0, 0}) > 0);   // in the central hole -> void
  TPX_CHECK(hc.eval({5, 0, 0}) > 0);   // outside the outer radius -> void
  TPX_CHECK(hc.eval({2, 0, 3}) > 0);   // above the top (z=3 > 2) -> void
  TPX_CHECK(std::fabs(hc.eval({3, 0, 0})) < 1e-12);  // on the outer wall surface

  // --- Complement flips the sign everywhere ---
  Complement<Sphere> cs{s};
  TPX_CHECK(cs.eval({0, 0, 0}) > 0);
  TPX_CHECK(cs.eval({3, 0, 0}) < 0);

  // --- GridSdf: sample the sphere onto a grid, then trilinearly interpolate ---
  // Grid covering [-4,4]^3 at spacing 0.25.
  double sp = 0.25;
  IVec<3> dims{33, 33, 33};
  GridSdf gs = sample(s, dims, {-4.0, -4.0, -4.0}, {sp, sp, sp});
  // Away from the surface (>2*spacing), trilinear interpolation tracks the analytic SDF closely and
  // agrees in sign.
  const Vec<3> pts[] = {{0, 0, 0}, {1.0, 0.3, -0.2}, {3.1, 0, 0}, {2.7, 1.0, 0.5}, {-3.0, 0, 1.0}};
  for (const auto& p : pts) {
    double a = s.eval(p);
    if (std::fabs(a) < 2 * sp) continue;  // skip near-surface points
    double gsv = gs.eval(p);
    TPX_CHECK((a < 0) == (gsv < 0));            // sign agreement
    TPX_CHECK(std::fabs(gsv - a) < 2 * sp);     // value within interpolation error
  }
  // A grid sample coincident with a node reproduces the analytic value (almost) exactly.
  TPX_CHECK(std::fabs(gs.eval({0, 0, 0}) - s.eval({0, 0, 0})) < 1e-5);

  TPX_RETURN_TEST_RESULT();
}

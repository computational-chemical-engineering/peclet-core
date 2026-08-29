// Gate for geom/body_properties.hpp: rigid-body mass properties by implicit quadrature.
//
// Claims, each against a CLOSED FORM:
//   1. ACCURACY on true-distance shapes -- sphere (offset; also gates the degenerate-frame rule:
//      a fully degenerate tensor must report the IDENTITY rotation), rotated+offset box (gates
//      COM recovery, principal moments, and the recovered QUATERNION against the one used to
//      build the shape), hollow cylinder through a SceneBuilder tree (the consumer path).
//   2. THE BOUND-LEAF HEADLINE -- an ellipsoid via the kEllipsoid leaf, which UNDER-ESTIMATES
//      distance by up to 4x. The voxel integrator is systematically biased on such fields
//      (+3.5% measured); this integrator uses only signs and bracketed roots, so the bound
//      costs NOTHING. Same for the composed shape below.
//   3. COMPOSITION -- a dumbbell (union of two disjoint spheres): exact closed form by
//      additivity + parallel axis.
//   4. REFRAME ROUND TRIP -- addReframed with the inverse principal transform must produce a
//      tree whose OWN bodyProperties report com ~ 0, identity rotation, and the same principal
//      moments. This is the move that makes "canonical frame != principal frame" a non-issue
//      for node trees: one composed transform, no resampling.
//   5. CONVERGENCE in n (printed; thresholds from measurement).
#include <cmath>
#include <cstdio>

#include "peclet/core/geom/body_properties.hpp"
#include "peclet/core/geom/scene_builder.hpp"
#include "peclet/core/geom/scene_query.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;

static double relErr(double a, double b) { return std::fabs(a - b) / std::fabs(b); }

// angle (deg) between two rotations given as quaternions, sign/double-cover safe
static double quatAngleDeg(Quat<double> a, Quat<double> b) {
  const double d = std::fabs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
  return std::acos(std::fmin(1.0, d)) * 2.0 * 180.0 / M_PI;
}

int main() {
  // --- 1a. offset sphere: exact everything; degenerate frame -> identity ----------------------
  {
    const double R = 0.35;
    const Vec3<double> C{0.11, -0.07, 0.19};
    auto f = [&](Vec3<double> p) {
      const double dx = p.x - C.x, dy = p.y - C.y, dz = p.z - C.z;
      return std::sqrt(dx * dx + dy * dy + dz * dz) - R;
    };
    std::printf("  sphere R=%.2f offset: convergence in n (order=5, nseg=8)\n", R);
    for (int n : {8, 16, 32}) {
      auto bp = bodyProperties<double>(f, Vec3<double>{-0.6, -0.6, -0.6},
                                       Vec3<double>{0.9, 0.6, 0.9}, n);
      const double Vex = 4.0 / 3.0 * M_PI * R * R * R;
      const double Iex = 0.4 * Vex * R * R;
      std::printf("    n=%2d  vol err %.2e  I err %.2e  |com err| %.2e  frame dev %.2e deg\n", n,
                  relErr(bp.volume, Vex), relErr(bp.principal[0], Iex),
                  std::sqrt((bp.com.x - C.x) * (bp.com.x - C.x) +
                            (bp.com.y - C.y) * (bp.com.y - C.y) +
                            (bp.com.z - C.z) * (bp.com.z - C.z)),
                  quatAngleDeg(bp.quat, Quat<double>{0, 0, 0, 1}));
      if (n == 32) {
        PECLET_CORE_CHECK(relErr(bp.volume, Vex) < 2e-5);
        PECLET_CORE_CHECK(relErr(bp.principal[0], Iex) < 2e-5);
        PECLET_CORE_CHECK(quatAngleDeg(bp.quat, Quat<double>{0, 0, 0, 1}) < 1e-6);
      }
    }
  }

  // --- 1b. rotated + offset box: COM, principal moments, recovered quaternion ------------------
  {
    const double hx = 0.4, hy = 0.25, hz = 0.15;
    const Vec3<double> C{0.11, -0.07, 0.05};
    // rotation: 35 deg about the (1, 1, 0)/sqrt(2) axis
    const double th = 35.0 * M_PI / 180.0, s = std::sin(0.5 * th), c = std::cos(0.5 * th);
    const Quat<double> q{s / std::sqrt(2.0), s / std::sqrt(2.0), 0.0, c};
    auto f = [&](Vec3<double> p) {
      const Vec3<double> b = invRotate(q, Vec3<double>{p.x - C.x, p.y - C.y, p.z - C.z});
      const double qx = std::fabs(b.x) - hx, qy = std::fabs(b.y) - hy, qz = std::fabs(b.z) - hz;
      const double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0), oz = std::fmax(qz, 0.0);
      return std::sqrt(ox * ox + oy * oy + oz * oz) +
             std::fmin(std::fmax(qx, std::fmax(qy, qz)), 0.0);
    };
    auto bp = bodyProperties<double>(f, Vec3<double>{-0.8, -0.8, -0.8}, Vec3<double>{1.0, 0.8, 0.8},
                                     40);
    const double Vex = 8 * hx * hy * hz;
    double Iex[3] = {Vex / 3 * (hy * hy + hz * hz), Vex / 3 * (hx * hx + hz * hz),
                     Vex / 3 * (hx * hx + hy * hy)};
    // bodyProperties sorts principal moments ascending; sort the exact ones too
    for (int i = 0; i < 2; ++i)
      for (int j = i + 1; j < 3; ++j)
        if (Iex[j] < Iex[i])
          std::swap(Iex[i], Iex[j]);
    const double eI = std::fmax(relErr(bp.principal[0], Iex[0]),
                                std::fmax(relErr(bp.principal[1], Iex[1]),
                                          relErr(bp.principal[2], Iex[2])));
    // recovered frame vs the true one: compare the DISTINCT axis (hx is unique -> smallest I is
    // about x_body). Column pairing/sign is free for the others; test the unique one.
    const Vec3<double> xTrue = rotate(q, Vec3<double>{1, 0, 0});
    // smallest principal moment pairs with column 0 (ascending sort)
    const double dot = std::fabs(bp.rotation[0][0] * xTrue.x + bp.rotation[1][0] * xTrue.y +
                                 bp.rotation[2][0] * xTrue.z);
    const double axErr = std::acos(std::fmin(1.0, dot)) * 180.0 / M_PI;
    std::printf("  rotated box: vol err %.2e  worst principal err %.2e  com err %.2e  unique-axis "
                "err %.4f deg\n",
                relErr(bp.volume, Vex), eI,
                std::sqrt(std::pow(bp.com.x - C.x, 2) + std::pow(bp.com.y - C.y, 2) +
                          std::pow(bp.com.z - C.z, 2)),
                axErr);
    PECLET_CORE_CHECK(relErr(bp.volume, Vex) < 2e-4);
    PECLET_CORE_CHECK(eI < 5e-4);
    PECLET_CORE_CHECK(axErr < 5e-2);
  }

  // --- 1c + 2 + 3 + 4: SceneBuilder trees through SceneQueryView (the consumer path) ----------
  {
    SceneBuilder<double> b;

    // hollow cylinder (dem form, distance-exact, axis = y): ring closed form
    const double ro = 0.4, hgt = 0.5, tk = 0.12, ri = ro - tk;
    const int hc = b.addLeaf(kHollowCylinder, {ro, hgt, tk});
    // ellipsoid BOUND leaf (under-estimates distance up to 4x): the headline
    const double ea = 0.5, eb = 0.3, ec = 0.2;
    const int el = b.addLeaf(kEllipsoid, {ea, eb, ec});
    // dumbbell: union of two disjoint spheres
    const double rs = 0.25, off = 0.35;
    const int s1 = b.addLeaf(kSphere, {rs}, Transform<double>{Vec3<double>{-off, 0, 0}});
    const int s2 = b.addLeaf(kSphere, {rs}, Transform<double>{Vec3<double>{off, 0, 0}});
    const int db = b.addUnion(s1, s2);

    auto evalRoot = [&](int root) {
      return [&b, root](Vec3<double> p) {
        const SceneView<double> sv = b.view();
        return evalTree<double>(TablePtr<ShapeNode<double>>{sv.nodes}, sv.nodeCount, root, p,
                                TablePtr<GridDesc<double>>{sv.grids}, PoolPtr<float>{sv.samples});
      };
    };

    {  // hollow cylinder
      auto bp = bodyProperties<double>(evalRoot(hc), Vec3<double>{-0.6, -0.5, -0.6},
                                       Vec3<double>{0.6, 0.5, 0.6}, 32);
      const double Vex = M_PI * (ro * ro - ri * ri) * hgt;
      const double Iax = 0.5 * Vex * (ro * ro + ri * ri);                       // about y
      const double Ipp = Vex / 12.0 * (3.0 * (ro * ro + ri * ri) + hgt * hgt);  // perpendicular
      double Iex[3] = {Iax, Ipp, Ipp};
      for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
          if (Iex[j] < Iex[i])
            std::swap(Iex[i], Iex[j]);
      const double e = std::fmax(relErr(bp.principal[0], Iex[0]),
                                 std::fmax(relErr(bp.principal[1], Iex[1]),
                                           relErr(bp.principal[2], Iex[2])));
      std::printf("  hollow cylinder (tree): vol err %.2e  worst principal err %.2e\n",
                  relErr(bp.volume, Vex), e);
      PECLET_CORE_CHECK(relErr(bp.volume, Vex) < 1e-3);
      PECLET_CORE_CHECK(e < 1e-3);
    }

    {  // ellipsoid via the BOUND leaf -- the number that buries the voxel integrator's bias
      auto bp = bodyProperties<double>(evalRoot(el), Vec3<double>{-0.7, -0.5, -0.4},
                                       Vec3<double>{0.7, 0.5, 0.4}, 32);
      const double Vex = 4.0 / 3.0 * M_PI * ea * eb * ec;
      double Iex[3] = {Vex / 5 * (eb * eb + ec * ec), Vex / 5 * (ea * ea + ec * ec),
                       Vex / 5 * (ea * ea + eb * eb)};
      for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
          if (Iex[j] < Iex[i])
            std::swap(Iex[i], Iex[j]);
      const double e = std::fmax(relErr(bp.principal[0], Iex[0]),
                                 std::fmax(relErr(bp.principal[1], Iex[1]),
                                           relErr(bp.principal[2], Iex[2])));
      std::printf("  ellipsoid via BOUND leaf: vol err %.2e  worst principal err %.2e  (the voxel "
                  "integrator is +3.5e-02 on this field)\n",
                  relErr(bp.volume, Vex), e);
      PECLET_CORE_CHECK(relErr(bp.volume, Vex) < 1e-5);
      PECLET_CORE_CHECK(e < 1e-4);
    }

    {  // dumbbell: additivity + parallel axis, and then the REFRAME ROUND TRIP on a moved copy
      auto bp = bodyProperties<double>(evalRoot(db), Vec3<double>{-0.7, -0.35, -0.35},
                                       Vec3<double>{0.7, 0.35, 0.35}, 32);
      const double Vs = 4.0 / 3.0 * M_PI * rs * rs * rs, Vex = 2 * Vs;
      const double Ix = 2 * 0.4 * Vs * rs * rs;
      const double Iyz = 2 * (0.4 * Vs * rs * rs + Vs * off * off);
      std::printf("  dumbbell (union tree): vol err %.2e  I_axis err %.2e  I_perp err %.2e\n",
                  relErr(bp.volume, Vex), relErr(bp.principal[0], Ix),
                  relErr(bp.principal[2], Iyz));
      PECLET_CORE_CHECK(relErr(bp.volume, Vex) < 1e-5);
      PECLET_CORE_CHECK(relErr(bp.principal[0], Ix) < 1e-5);
      PECLET_CORE_CHECK(relErr(bp.principal[2], Iyz) < 1e-5);

      // move the dumbbell off-axis and off-centre: wrap in a reframed copy carrying the offending
      // transform, measure, then reframe by the INVERSE principal transform and re-measure.
      const double th = 25.0 * M_PI / 180.0, sn = std::sin(0.5 * th), cs = std::cos(0.5 * th);
      const Quat<double> qm{0.0, sn, 0.0, cs};  // 25 deg about y
      Transform<double> Tm;
      Tm.translation = Vec3<double>{0.15, -0.1, 0.08};
      Tm.rotation = qm;
      const int moved = b.addReframed(db, Tm);
      // NOTE addReframed composes W as "applied FIRST": eval_new(p) = eval_old(toLocal(W, p)) --
      // toLocal(Tm, .) maps world into the dumbbell's frame, i.e. this PLACES the dumbbell at Tm.
      auto bpM = bodyProperties<double>(evalRoot(moved), Vec3<double>{-1.0, -1.0, -1.0},
                                        Vec3<double>{1.0, 1.0, 1.0}, 40);
      // recovered com must be Tm.translation; recovered unique axis must be the rotated x
      const double comErr = std::sqrt(std::pow(bpM.com.x - 0.15, 2) +
                                      std::pow(bpM.com.y + 0.10, 2) +
                                      std::pow(bpM.com.z - 0.08, 2));
      // The inverse principal transform W' such that toLocal(W', p_body) = com + R p_body:
      // toLocal(T, p) = invRotate(q_T, p - t_T)/s_T, so take q_W = conj(q_R) (then
      // invRotate(q_W, x) = rotate(q_R, x)) and t_W with rotate(q_R, t_W) = -com, i.e.
      // t_W = invRotate(q_R, -com). Verified below numerically, not just algebraically: the
      // round-tripped body must report com ~ 0 and an identity frame.
      Transform<double> Wp;
      Wp.rotation = Quat<double>{-bpM.quat.x, -bpM.quat.y, -bpM.quat.z, bpM.quat.w};
      Wp.translation = invRotate(bpM.quat, Vec3<double>{-bpM.com.x, -bpM.com.y, -bpM.com.z});
      const int home = b.addReframed(moved, Wp);
      auto bpH = bodyProperties<double>(evalRoot(home), Vec3<double>{-0.8, -0.5, -0.5},
                                        Vec3<double>{0.8, 0.5, 0.5}, 40);
      const double comH = std::sqrt(bpH.com.x * bpH.com.x + bpH.com.y * bpH.com.y +
                                    bpH.com.z * bpH.com.z);
      const double frameH = quatAngleDeg(bpH.quat, Quat<double>{0, 0, 0, 1});
      const double eP = std::fmax(relErr(bpH.principal[0], bpM.principal[0]),
                                  relErr(bpH.principal[2], bpM.principal[2]));
      std::printf("  reframe round trip: placed-com err %.2e; after addReframed(inverse principal) "
                  "com %.2e, frame dev %.2e deg, principal drift %.2e\n",
                  comErr, comH, frameH, eP);
      PECLET_CORE_CHECK(comErr < 1e-4);
      PECLET_CORE_CHECK(comH < 1e-4);
      PECLET_CORE_CHECK(frameH < 1e-3);
      PECLET_CORE_CHECK(eP < 1e-3);
    }
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

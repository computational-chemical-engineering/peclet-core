// core — device-callable analytic SDF leaf primitives: THE single source of the formulas.
//
// Layer 0 rung 0 of suite/docs/ANALYTIC_SDF_GEOMETRY.md. These leaves are the only place a signed
// distance formula is written in the suite; the runtime ShapeNode/SceneView layer (geom/scene.hpp)
// dispatches to them and contains no formulas of its own, and voro's compile-time provider template
// consumes them directly. dem's dem_portable.hpp and voro's voro/sdf.hpp are ported onto them in
// rungs 3-4.
//
// CONVENTIONS
//   * Sign: sdf < 0 inside solid, > 0 in the void, 0 on the surface (docs/CONVENTIONS.md).
//   * Leaves are ORIGIN-CENTRED in their canonical frame. Placement (translation, rotation,
//     isotropic scale) is the caller's / the scene transform's job — never a primitive parameter.
//     This is dem's contract (canonical body space + per-particle pos/quat/scale) and is why the
//     dem port in rung 3 is a relocation rather than a rewrite.
//   * Canonical axes are ADOPTED FROM dem VERBATIM (HollowCylinder about y). Canonicalising to a
//     single axis would insert a rotation whose float rounding breaks the bit-exact dem port.
//   * Exactness is per-primitive and load-bearing for consumers — see the `exact_distance` flag on
//     each leaf. Sphere/Box/HollowCylinder are true Euclidean distances; HollowCylinderShell is
//     sign-correct only (a lower bound in magnitude). Consumers needing wall crossings must bracket
//     and root-find on eval() along the segment, which needs only sign correctness.
//
// FAITHFULNESS
//   Every formula below is transcribed operation-for-operation (including association order and the
//   fmax/fmin vs `<` distinction) from its origin, so a float instantiation reproduces the origin
//   bit-for-bit. Origins are cited per leaf. Do not "simplify" these expressions.
//
// ANNOTATIONS (rung 2). Every leaf carries two compile-time flags, and they mean exactly this:
//   exact_distance    — eval() is the true Euclidean signed distance, not merely sign-correct.
//                       Layer 2's crossing/aperture code branches on this. UNDER-claiming costs a
//                       little accuracy; OVER-claiming costs correctness, so when in doubt it is
//                       false and the gate measures the actual bound behaviour.
//   analytic_gradient — grad() is closed form rather than finite differences.
//
//   grad() IS ALWAYS THE GRADIENT OF THAT LEAF'S OWN eval(), never of some idealised distance the
//   eval only approximates. That single invariant is what lets the gate check every primitive the
//   same way (grad ~= FD(eval) everywhere off the kinks) instead of special-casing the
//   approximate leaves. Gradients are NOT normalised — |grad| = 1 is a property the exact leaves
//   have and the gate verifies, not something imposed by construction.
//
// C++17-clean so it can be pulled into CUDA translation units.
#ifndef PECLET_CORE_GEOM_PRIMITIVES_HPP
#define PECLET_CORE_GEOM_PRIMITIVES_HPP

#include "peclet/core/common/portable.hpp"

// NAMESPACE: the leaves live in peclet::core::geom::prim, NOT peclet::core::geom, because
// geom/sdf.hpp already owns non-template host shapes of the same names (Sphere, Box,
// HollowCylinder, Complement) that existing consumers -- refineToSdf, the AMR/diffusion tests, the
// VTI path -- depend on. Those stay as the host concept + I/O layer and delegate INTO these leaves;
// see docs/ANALYTIC_SDF_GEOMETRY.md Layer 0.
namespace peclet::core::geom::prim {

using peclet::core::Vec3;

/// Generic outward normal by central differences — the fallback for any leaf until rung 2 supplies
/// closed-form grad(), and permanently for sampled (grid) fields. NOT normalised: callers that need
/// a unit normal divide by the length, as dem's narrow-phase does today.
template <class Shape, class Real>
PECLET_HD Vec3<Real> gradient(const Shape& s, Vec3<Real> p, Real h) {
  const Real den = Real(2) * h;
  const Real gx = s.eval(Vec3<Real>{p.x + h, p.y, p.z}) - s.eval(Vec3<Real>{p.x - h, p.y, p.z});
  const Real gy = s.eval(Vec3<Real>{p.x, p.y + h, p.z}) - s.eval(Vec3<Real>{p.x, p.y - h, p.z});
  const Real gz = s.eval(Vec3<Real>{p.x, p.y, p.z + h}) - s.eval(Vec3<Real>{p.x, p.y, p.z - h});
  return Vec3<Real>{gx / den, gy / den, gz / den};
}

/// Solid ball of radius `radius`, centred on the origin. Exact Euclidean distance.
/// Origin: dem `sdfSphere` (dem/src/dem_portable.hpp:79-81).
template <class Real>
struct Sphere {
  using value_type = Real;
  Real radius = Real(1);
  static constexpr bool exact_distance = true;
  static constexpr bool analytic_gradient = true;

  PECLET_HD Real eval(Vec3<Real> p) const {
    // len3(p) - radius, with dot3's left-to-right association preserved.
    return detail::hdSqrt(p.x * p.x + p.y * p.y + p.z * p.z) - radius;
  }

  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    const Real l = detail::hdSqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    // At the centre the gradient is undefined (the medial axis is the single point p=0); pick +x
    // so callers never see a NaN. Any direction is equally wrong there.
    return (l > Real(0)) ? Vec3<Real>{p.x / l, p.y / l, p.z / l} : Vec3<Real>{1, 0, 0};
  }
};

/// Axis-aligned solid box of half-extents (hx, hy, hz), centred on the origin. Exact Euclidean
/// distance. Origin: dem `sdfBox` (dem/src/dem_portable.hpp:95-103).
template <class Real>
struct Box {
  using value_type = Real;
  Real hx = Real(1), hy = Real(1), hz = Real(1);
  static constexpr bool exact_distance = true;
  static constexpr bool analytic_gradient = true;

  PECLET_HD Real eval(Vec3<Real> p) const {
    const Real dx = detail::hdAbs(p.x) - hx;
    const Real dy = detail::hdAbs(p.y) - hy;
    const Real dz = detail::hdAbs(p.z) - hz;
    const Real ox = detail::hdMax(dx, Real(0));
    const Real oy = detail::hdMax(dy, Real(0));
    const Real oz = detail::hdMax(dz, Real(0));
    const Real outside = detail::hdSqrt(ox * ox + oy * oy + oz * oz);
    // Nesting order hdMax(dx, hdMax(dy, dz)) is dem's — keep it.
    const Real inside = detail::hdMin(detail::hdMax(dx, detail::hdMax(dy, dz)), Real(0));
    return outside + inside;
  }

  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    const Real sx = p.x < Real(0) ? Real(-1) : Real(1);
    const Real sy = p.y < Real(0) ? Real(-1) : Real(1);
    const Real sz = p.z < Real(0) ? Real(-1) : Real(1);
    const Real dx = detail::hdAbs(p.x) - hx;
    const Real dy = detail::hdAbs(p.y) - hy;
    const Real dz = detail::hdAbs(p.z) - hz;
    if (dx > Real(0) || dy > Real(0) || dz > Real(0)) {
      // Outside: the gradient is the direction to the nearest face/edge/corner.
      const Real ox = detail::hdMax(dx, Real(0));
      const Real oy = detail::hdMax(dy, Real(0));
      const Real oz = detail::hdMax(dz, Real(0));
      const Real l = detail::hdSqrt(ox * ox + oy * oy + oz * oz);
      return (l > Real(0)) ? Vec3<Real>{sx * ox / l, sy * oy / l, sz * oz / l}
                           : Vec3<Real>{sx, 0, 0};
    }
    // Inside: eval reduces to max(dx,dy,dz), so the gradient is the unit axis of the LEAST
    // negative term -- the nearest face.
    if (dx >= dy && dx >= dz)
      return Vec3<Real>{sx, 0, 0};
    if (dy >= dz)
      return Vec3<Real>{0, sy, 0};
    return Vec3<Real>{0, 0, sz};
  }
};

/// Solid tube wall about the **y** axis: outer radius `rOuter`, full height `height`, wall
/// thickness `thickness` (the solid is the shell, the bore is void). Exact Euclidean distance — it
/// is the 2-D box distance in the (r, y) cross-section.
///
/// Origin: dem `sdfHollowCylinder` (dem/src/dem_portable.hpp:83-93). NOTE the axis is y and the
/// parameterisation is (rOuter, height, thickness) — NOT the (rOuter, rInner, height, axis) of
/// HollowCylinderShell below. The two are different functions that shared a name across the suite;
/// see the TRAP section of docs/ANALYTIC_SDF_GEOMETRY.md.
template <class Real>
struct HollowCylinder {
  using value_type = Real;
  Real rOuter = Real(1), height = Real(1), thickness = Real(0.5);
  static constexpr bool exact_distance = true;
  static constexpr bool analytic_gradient = true;

  PECLET_HD Real eval(Vec3<Real> p) const {
    const Real r = detail::hdSqrt(p.x * p.x + p.z * p.z);
    const Real rMid = rOuter - thickness * Real(0.5);
    const Real dx = detail::hdAbs(r - rMid) - thickness * Real(0.5);
    const Real dy = detail::hdAbs(p.y) - height * Real(0.5);
    const Real ox = detail::hdMax(dx, Real(0));
    const Real oy = detail::hdMax(dy, Real(0));
    const Real outside = detail::hdSqrt(ox * ox + oy * oy);
    const Real inside = detail::hdMin(detail::hdMax(dx, dy), Real(0));
    return outside + inside;
  }

  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    // The tube is a 2-D box in the (r, y) half-plane; differentiate there and map the radial
    // component back through dr/dp = (x/r, 0, z/r).
    const Real r = detail::hdSqrt(p.x * p.x + p.z * p.z);
    const Real rMid = rOuter - thickness * Real(0.5);
    const Real a = detail::hdAbs(r - rMid) - thickness * Real(0.5);
    const Real b = detail::hdAbs(p.y) - height * Real(0.5);
    const Real sr = (r - rMid) < Real(0) ? Real(-1) : Real(1);  // outward across the wall
    const Real sy = p.y < Real(0) ? Real(-1) : Real(1);
    Real gr, gy;
    if (a > Real(0) || b > Real(0)) {
      const Real oa = detail::hdMax(a, Real(0)), ob = detail::hdMax(b, Real(0));
      const Real l = detail::hdSqrt(oa * oa + ob * ob);
      gr = (l > Real(0)) ? oa / l : Real(1);
      gy = (l > Real(0)) ? ob / l : Real(0);
    } else if (a >= b) {
      gr = Real(1);
      gy = Real(0);
    } else {
      gr = Real(0);
      gy = Real(1);
    }
    // On the axis (r = 0) the radial direction is undefined; fall back to +x.
    const Real ux = (r > Real(0)) ? p.x / r : Real(1);
    const Real uz = (r > Real(0)) ? p.z / r : Real(0);
    return Vec3<Real>{ux * gr * sr, gy * sy, uz * gr * sr};
  }
};

/// Solid tube wall about the **z** axis as a CSG intersection of half-spaces:
/// max(r - rOuter, rInner - r, |z| - height/2).
///
/// SIGN-EXACT ONLY — this is NOT a distance: the magnitude is wrong at the edges/corners and inside
/// the shell (it is a lower bound on |distance|). Kept as a distinct primitive from HollowCylinder
/// above because core and voro both compute geometry with it today and swapping either for the
/// other would silently move their numerics.
///
/// Origin: core `geom::Sphere`-family `HollowCylinder::eval` (geom/sdf.hpp) and voro
/// `SdfHollowCylinder` (voro/include/peclet/voro/sdf.hpp), which agree with each other.
template <class Real>
struct HollowCylinderShell {
  using value_type = Real;
  Real rOuter = Real(1), rInner = Real(0.5), height = Real(1);
  static constexpr bool exact_distance = false;
  static constexpr bool analytic_gradient = true;

  /// Cross-section form: `r` is the in-plane radius, `z` the axial coordinate. Lets an
  /// axis-parameterised host wrapper permute coordinates and reuse this body unchanged.
  PECLET_HD Real evalRZ(Real r, Real z) const {
    // `<`-ordered max (std::max / voro's ternaries), NOT fmax — see detail::cmpMax.
    Real m = r - rOuter;
    m = detail::cmpMax(m, rInner - r);
    m = detail::cmpMax(m, detail::hdAbs(z) - height * Real(0.5));
    return m;
  }

  PECLET_HD Real eval(Vec3<Real> p) const {
    return evalRZ(detail::hdSqrt(p.x * p.x + p.y * p.y), p.z);
  }

  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    // eval is a max of three half-space terms: the gradient is that of whichever is active.
    const Real r = detail::hdSqrt(p.x * p.x + p.y * p.y);
    const Real tOuter = r - rOuter;
    const Real tInner = rInner - r;
    const Real tAxial = detail::hdAbs(p.z) - height * Real(0.5);
    const Real ux = (r > Real(0)) ? p.x / r : Real(1);
    const Real uy = (r > Real(0)) ? p.y / r : Real(0);
    if (tOuter >= tInner && tOuter >= tAxial)
      return Vec3<Real>{ux, uy, 0};  // outward radial
    if (tInner >= tAxial)
      return Vec3<Real>{-ux, -uy, 0};  // inward radial (into the bore)
    return Vec3<Real>{0, 0, p.z < Real(0) ? Real(-1) : Real(1)};
  }
};

/// Solid capsule about the **y** axis: a segment from (0,-halfLength,0) to (0,+halfLength,0)
/// swept by `radius`. Exact Euclidean distance — the distance to a swept sphere is exactly the
/// distance to its spine minus the radius, for any spine.
template <class Real>
struct Capsule {
  using value_type = Real;
  Real radius = Real(0.5), halfLength = Real(0.5);
  static constexpr bool exact_distance = true;
  static constexpr bool analytic_gradient = true;

  PECLET_HD Real eval(Vec3<Real> p) const {
    const Real qy = p.y - detail::hdMin(detail::hdMax(p.y, -halfLength), halfLength);
    return detail::hdSqrt(p.x * p.x + qy * qy + p.z * p.z) - radius;
  }

  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    const Real qy = p.y - detail::hdMin(detail::hdMax(p.y, -halfLength), halfLength);
    const Real l = detail::hdSqrt(p.x * p.x + qy * qy + p.z * p.z);
    return (l > Real(0)) ? Vec3<Real>{p.x / l, qy / l, p.z / l} : Vec3<Real>{1, 0, 0};
  }
};

/// Solid torus with its axis along **y**: tube of minor radius `minorRadius` swept around a circle
/// of radius `majorRadius` in the xz-plane. Exact Euclidean distance (again a swept sphere).
template <class Real>
struct Torus {
  using value_type = Real;
  Real majorRadius = Real(1), minorRadius = Real(0.25);
  static constexpr bool exact_distance = true;
  static constexpr bool analytic_gradient = true;

  PECLET_HD Real eval(Vec3<Real> p) const {
    const Real lr = detail::hdSqrt(p.x * p.x + p.z * p.z);
    const Real qx = lr - majorRadius;
    return detail::hdSqrt(qx * qx + p.y * p.y) - minorRadius;
  }

  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    const Real lr = detail::hdSqrt(p.x * p.x + p.z * p.z);
    const Real qx = lr - majorRadius;
    const Real l = detail::hdSqrt(qx * qx + p.y * p.y);
    if (l <= Real(0))  // on the spine circle: gradient undefined
      return Vec3<Real>{1, 0, 0};
    const Real ux = (lr > Real(0)) ? p.x / lr : Real(1);
    const Real uz = (lr > Real(0)) ? p.z / lr : Real(0);
    const Real gr = qx / l;
    return Vec3<Real>{ux * gr, p.y / l, uz * gr};
  }
};

/// Solid capped cone (frustum) about the **y** axis: radius `rBottom` at y = -halfHeight, radius
/// `rTop` at y = +halfHeight. Exact Euclidean distance — the two candidate feet are the nearest
/// point on a cap annulus and the nearest point on the lateral segment, and the true distance is
/// the smaller. Set rTop = 0 for a plain cone, rTop = rBottom for a solid cylinder.
///
/// Derived from the standard exact capped-cone construction; its exactness is not asserted on
/// authority — the rung-2 gate measures |grad| = 1 and compares against a brute-force nearest
/// point on a dense parametric sampling of the surface.
template <class Real>
struct Cone {
  using value_type = Real;
  Real rBottom = Real(1), rTop = Real(0.5), halfHeight = Real(1);
  static constexpr bool exact_distance = true;
  static constexpr bool analytic_gradient = false;  // FD fallback: the piecewise form is not worth
                                                    // a hand-derived gradient at this rung

  PECLET_HD Real eval(Vec3<Real> p) const {
    const Real qx = detail::hdSqrt(p.x * p.x + p.z * p.z);
    const Real qy = p.y;
    // (a) nearest point on the end caps, clamped to the annulus of the relevant radius
    const Real rCap = (qy < Real(0)) ? rBottom : rTop;
    const Real ax = qx - detail::hdMin(qx, rCap);
    const Real ay = detail::hdAbs(qy) - halfHeight;
    // (b) nearest point on the lateral edge, as a clamped projection onto the slant segment
    const Real k1x = rTop, k1y = halfHeight;
    const Real k2x = rTop - rBottom, k2y = Real(2) * halfHeight;
    const Real d2k2 = k2x * k2x + k2y * k2y;
    const Real tRaw = ((k1x - qx) * k2x + (k1y - qy) * k2y) / ((d2k2 > Real(0)) ? d2k2 : Real(1));
    const Real t = detail::hdMin(detail::hdMax(tRaw, Real(0)), Real(1));
    const Real bx = qx - k1x + k2x * t;
    const Real by = qy - k1y + k2y * t;
    // Inside iff we are on the solid side of BOTH the lateral surface and the caps.
    const Real sgn = (bx < Real(0) && ay < Real(0)) ? Real(-1) : Real(1);
    const Real da = ax * ax + ay * ay;
    const Real db = bx * bx + by * by;
    return sgn * detail::hdSqrt(detail::hdMin(da, db));
  }

  /// Finite differences over this leaf's own eval (the piecewise capped-cone form has no
  /// gradient worth hand-deriving at this rung). h is dem's 1e-4 probe convention.
  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    return gradient(*this, p, Real(1e-4));
  }
};

/// Solid ellipsoid with semi-axes (rx, ry, rz), centred on the origin.
///
/// NOT an exact distance, and no closed form exists — this is the standard first-order normalised
/// implicit estimate k0*(k0-1)/k1. The rung-2 gate MEASURES whether it under- or over-estimates
/// rather than assuming; see the gate's report. Use `Sphere` under a conformal transform whenever
/// the body is actually a sphere: that path is exact, this one is not.
template <class Real>
struct Ellipsoid {
  using value_type = Real;
  Real rx = Real(1), ry = Real(1), rz = Real(1);
  static constexpr bool exact_distance = false;
  static constexpr bool analytic_gradient = false;

  PECLET_HD Real eval(Vec3<Real> p) const {
    const Real ax = p.x / rx, ay = p.y / ry, az = p.z / rz;
    const Real k0 = detail::hdSqrt(ax * ax + ay * ay + az * az);
    const Real bx = p.x / (rx * rx), by = p.y / (ry * ry), bz = p.z / (rz * rz);
    const Real k1 = detail::hdSqrt(bx * bx + by * by + bz * bz);
    if (k1 <= Real(0))  // the centre: return the inradius, negative (inside)
      return -detail::hdMin(rx, detail::hdMin(ry, rz));
    return k0 * (k0 - Real(1)) / k1;
  }

  /// Finite differences over this leaf's own eval (the normalised implicit has no gradient worth
  /// hand-deriving at this rung). h is dem's 1e-4 probe convention.
  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    return gradient(*this, p, Real(1e-4));
  }
};

/// Solid superquadric with semi-axes (rx, ry, rz) and exponent `e`:
/// the implicit |x/rx|^e + |y/ry|^e + |z/rz|^e = 1. e = 2 is an ellipsoid, large e approaches a
/// box, e < 2 gives the pinched/octahedral family. The DEM literature's workhorse non-spherical
/// grain.
///
/// NOT an exact distance: the implicit is normalised by its own gradient (the Taubin estimate),
/// which is first-order correct near the surface and degrades away from it. Bound direction is
/// MEASURED by the rung-2 gate, not assumed.
template <class Real>
struct Superquadric {
  using value_type = Real;
  Real rx = Real(1), ry = Real(1), rz = Real(1), e = Real(2);
  static constexpr bool exact_distance = false;
  static constexpr bool analytic_gradient = false;

  PECLET_HD Real eval(Vec3<Real> p) const {
    const Real ux = detail::hdAbs(p.x) / rx;
    const Real uy = detail::hdAbs(p.y) / ry;
    const Real uz = detail::hdAbs(p.z) / rz;
    const Real f = detail::hdPow(ux, e) + detail::hdPow(uy, e) + detail::hdPow(uz, e) - Real(1);
    // |grad f| = e * sqrt( sum_i ( u_i^(e-1) / r_i )^2 )
    const Real gx = detail::hdPow(ux, e - Real(1)) / rx;
    const Real gy = detail::hdPow(uy, e - Real(1)) / ry;
    const Real gz = detail::hdPow(uz, e - Real(1)) / rz;
    const Real gn = e * detail::hdSqrt(gx * gx + gy * gy + gz * gz);
    const Real inradius = detail::hdMin(rx, detail::hdMin(ry, rz));
    if (!(gn > Real(0)))  // the centre (and any point where the implicit gradient vanishes)
      return -inradius;
    // CLAMP AT THE INRADIUS. |grad f| -> 0 as p -> 0, so the raw quotient f/|grad f| dives to -inf
    // near the centre: the rung-2 gate measured a Lipschitz constant of 779 and estimates 4x too
    // deep before this clamp. No interior point can be farther inside than the largest ball that
    // fits, so clamping there is both a hard geometric bound and what restores sane behaviour.
    return detail::hdMax(f / gn, -inradius);
  }

  /// Finite differences over this leaf's own eval (the Taubin estimate has no gradient worth
  /// hand-deriving at this rung). h is dem's 1e-4 probe convention.
  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    return gradient(*this, p, Real(1e-4));
  }
};

/// Offset (rounded) shape: the inner solid grown outward by `radius`. Exactness is INHERITED --
/// offsetting a true distance field by a constant is still a true distance field, which is why
/// rounding is the cheapest way to get an exact rounded box or rounded cone.
template <class Shape>
struct Rounded {
  using value_type = typename Shape::value_type;
  Shape shape;
  value_type radius = value_type(0);
  static constexpr bool exact_distance = Shape::exact_distance;
  static constexpr bool analytic_gradient = Shape::analytic_gradient;

  template <class Real>
  PECLET_HD Real eval(Vec3<Real> p) const {
    return shape.eval(p) - radius;
  }
  template <class Real>
  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    return shape.grad(p);  // a constant offset does not change the gradient
  }
};

/// Negation: solid and void swap. Composes with any leaf (a container is the complement of a body).
template <class Shape>
struct Complement {
  Shape shape;
  static constexpr bool exact_distance = Shape::exact_distance;
  static constexpr bool analytic_gradient = Shape::analytic_gradient;

  template <class Real>
  PECLET_HD Real eval(Vec3<Real> p) const {
    return -shape.eval(p);
  }
  template <class Real>
  PECLET_HD Vec3<Real> grad(Vec3<Real> p) const {
    const Vec3<Real> g = shape.grad(p);
    return Vec3<Real>{-g.x, -g.y, -g.z};
  }
};


}  // namespace peclet::core::geom::prim

#endif  // PECLET_CORE_GEOM_PRIMITIVES_HPP

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
//   Rung 0 ships eval() only. Closed-form per-primitive grad() and the extended vocabulary
//   (ellipsoid, superquadric, capsule, cone, torus, rounded/offset) are rung 2; until then use the
//   generic finite-difference gradient() at the bottom of this header.
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

/// Solid ball of radius `radius`, centred on the origin. Exact Euclidean distance.
/// Origin: dem `sdfSphere` (dem/src/dem_portable.hpp:79-81).
template <class Real>
struct Sphere {
  Real radius = Real(1);
  static constexpr bool exact_distance = true;

  PECLET_HD Real eval(Vec3<Real> p) const {
    // len3(p) - radius, with dot3's left-to-right association preserved.
    return detail::hdSqrt(p.x * p.x + p.y * p.y + p.z * p.z) - radius;
  }
};

/// Axis-aligned solid box of half-extents (hx, hy, hz), centred on the origin. Exact Euclidean
/// distance. Origin: dem `sdfBox` (dem/src/dem_portable.hpp:95-103).
template <class Real>
struct Box {
  Real hx = Real(1), hy = Real(1), hz = Real(1);
  static constexpr bool exact_distance = true;

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
  Real rOuter = Real(1), height = Real(1), thickness = Real(0.5);
  static constexpr bool exact_distance = true;

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
  Real rOuter = Real(1), rInner = Real(0.5), height = Real(1);
  static constexpr bool exact_distance = false;

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
};

/// Negation: solid and void swap. Composes with any leaf (a container is the complement of a body).
template <class Shape>
struct Complement {
  Shape shape;
  static constexpr bool exact_distance = Shape::exact_distance;

  template <class Real>
  PECLET_HD Real eval(Vec3<Real> p) const {
    return -shape.eval(p);
  }
};

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

}  // namespace peclet::core::geom::prim

#endif  // PECLET_CORE_GEOM_PRIMITIVES_HPP

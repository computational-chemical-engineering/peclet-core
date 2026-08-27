// core — runtime-dispatched SDF scene nodes: the layer that turns the compile-time leaves into
// geometry a solver can be handed at run time.
//
// Layer 0 rung 1 of suite/docs/ANALYTIC_SDF_GEOMETRY.md. This header contains NO signed-distance
// formulas: every leaf case dispatches into peclet::core::geom::prim (geom/primitives.hpp), and the
// only arithmetic here is the conformal transform stack and the CSG combinators. voro keeps its
// zero-overhead compile-time provider path over the leaves; dem and flow get the runtime scene
// their Python-composed, per-particle heterogeneity requires.
//
// WHY RUNTIME DISPATCH IS FREE ENOUGH: dem already pays a per-probe `switch` on shape kind in its
// hottest kernel today (narrowphase.hpp sdfEvalShape), and a simulation carries few distinct
// shapes, so the branch is warp-coherent. flow's geometry derivation is setup-phase, where dispatch
// cost is irrelevant.
//
// TRANSFORMS ARE CONFORMAL ONLY (contract 1): translation + quaternion + ISOTROPIC scale. A
// non-uniform scale destroys the signed-distance property (d_world = s * d_canonical holds only for
// conformal maps); anisotropy belongs in primitive parameters (box half-extents, ellipsoid
// semi-axes). This is exactly dem's per-particle (pos, quat, scale x globalScale) contract, which
// is what makes the rung-3 port a relocation.
//
// C++17-clean so it can be pulled into CUDA translation units.
#ifndef PECLET_CORE_GEOM_SCENE_HPP
#define PECLET_CORE_GEOM_SCENE_HPP

#include "peclet/core/common/portable.hpp"
#include "peclet/core/geom/primitives.hpp"

namespace peclet::core::geom {

/// Node kind tag. The first four values are dem's ShapeKind numbering VERBATIM
/// (dem_portable.hpp:77: SHAPE_GRID_SDF=0, SPHERE=1, HOLLOW_CYLINDER=2, BOX=3) so dem's existing
/// Python `shape_type` integers keep working through the rung-3 port with no remapping table.
/// Everything past 3 is new. CSG operators start at kCsgBase so `kind >= kCsgBase` is the
/// leaf/operator test.
enum ShapeKind : int {
  kGrid = 0,                 ///< sampled field; aux0 = index into the GridDesc table
  kSphere = 1,               ///< params[0] = radius
  kHollowCylinder = 2,       ///< params[0..2] = rOuter, height, thickness (distance-exact, y axis)
  kBox = 3,                  ///< params[0..2] = hx, hy, hz
  kHollowCylinderShell = 4,  ///< params[0..2] = rOuter, rInner, height (sign-exact, z axis)
  kCapsule = 5,              ///< params[0..1] = radius, halfLength (exact, y axis)
  kTorus = 6,                ///< params[0..1] = majorRadius, minorRadius (exact, y axis)
  kCone = 7,                 ///< params[0..2] = rBottom, rTop, halfHeight (exact, y axis)
  kEllipsoid = 8,            ///< params[0..2] = rx, ry, rz (BOUND, under-estimates)
  kSuperquadric = 9,         ///< params[0..3] = rx, ry, rz, exponent (BOUND, under-estimates)

  kCsgBase = 32,
  kUnion = 32,         ///< min(L, R)          — aux0/aux1 = child node indices
  kIntersection = 33,  ///< max(L, R)
  kDifference = 34,    ///< max(L, -R)         — L minus R
};

/// NOT YET A NODE KIND: prim::Rounded (offset/rounding) stays a compile-time wrapper for now.
/// Folding a rounding radius into every node would add an unconditional subtract to every leaf
/// eval, and no scene consumer needs it yet; add it with the rung-5 encoding if one does.

/// Off-grid extension policy for a sampled field (contract 6). The trilinear sample is clamped into
/// the lattice and the distance from the query to that clamped point is then added or subtracted:
///   kObject    (+residual) — a BODY. Outside the stored box means further from the body, so the
///                            far field must grow POSITIVE. dem `sampleGridSdf`.
///   kContainer (-residual) — a CONTAINER whose void is bounded. Outside the stored box is
///                            wall-side, so it must read ever more NEGATIVE. dem `sampleWallSdf`.
/// Getting this backwards is not cosmetic: dem's history records a 180k-bead pile losing 71k grains
/// through a distributor because a container read "clear" beyond its own grid face.
enum class GridExtension : int { kObject = 0, kContainer = 1 };

/// Descriptor for one sampled field inside a shared sample pool.
template <class Real>
struct GridDesc {
  int nx = 0, ny = 0, nz = 0;
  int offset = 0;  ///< start index of this field's samples in the pool
  Vec3<Real> origin{0, 0, 0};
  Vec3<Real> invSpacing{0, 0, 0};  ///< 1 / node spacing, per axis (dem stores the inverse too)
  GridExtension extension = GridExtension::kObject;
};

/// Conformal placement: canonical -> parent frame. Identity by default.
template <class Real>
struct Transform {
  Vec3<Real> translation{0, 0, 0};
  Quat<Real> rotation{0, 0, 0, 1};
  Real scale = Real(1);
};

/// Map a point from the parent frame into this transform's canonical frame.
/// Operation order is dem's narrow-phase verbatim (narrowphase.hpp: invRotateVector(q, p - pos),
/// then multiply by 1/effScale) so a transformed leaf reproduces a dem contact probe bit-for-bit.
template <class Real>
PECLET_HD Vec3<Real> toCanonical(const Transform<Real>& tr, Vec3<Real> p) {
  return scale(invRotate(tr.rotation, sub(p, tr.translation)), Real(1) / tr.scale);
}

/// One node of a flat shape tree. POD, trivially copyable, safe in a Kokkos View.
template <class Real>
struct ShapeNode {
  int kind = kSphere;
  int aux0 = -1;  ///< CSG: left child index. kGrid: GridDesc table index.
  int aux1 = -1;  ///< CSG: right child index.
  Real params[8] = {Real(1), 0, 0, 0, 0, 0, 0, 0};
  Transform<Real> transform{};
};

/// Adaptors so a raw pointer can stand in for a Kokkos View (Views already provide operator()),
/// letting host tests and device kernels share one evaluator. PoolPtr returns by value (sample
/// data); TablePtr returns by reference (node / descriptor records, which are larger).
template <class T>
struct PoolPtr {
  const T* data = nullptr;
  PECLET_HD T operator()(long i) const { return data[i]; }
};
template <class T>
struct TablePtr {
  const T* data = nullptr;
  PECLET_HD const T& operator()(long i) const { return data[i]; }
};

/// Trilinear sample of a grid field, with the clamp-plus-residual off-grid extension selected by
/// `d.extension`. Verbatim transcription of dem `sampleGridSdf` (narrowphase.hpp:104) and
/// `sampleWallSdf`, which are the same body differing only in the residual's sign.
template <class Real, class Pool>
PECLET_HD Real sampleGrid(Vec3<Real> p, const GridDesc<Real>& d, const Pool& pool) {
  const Real fx = (p.x - d.origin.x) * d.invSpacing.x;
  const Real fy = (p.y - d.origin.y) * d.invSpacing.y;
  const Real fz = (p.z - d.origin.z) * d.invSpacing.z;
  const Real cx = detail::hdMin(detail::hdMax(fx, Real(0)), Real(d.nx - 1));
  const Real cy = detail::hdMin(detail::hdMax(fy, Real(0)), Real(d.ny - 1));
  const Real cz = detail::hdMin(detail::hdMax(fz, Real(0)), Real(d.nz - 1));
  const int ix = (int)cx, iy = (int)cy, iz = (int)cz;
  const int ix1 = ix < d.nx - 1 ? ix + 1 : ix;
  const int iy1 = iy < d.ny - 1 ? iy + 1 : iy;
  const int iz1 = iz < d.nz - 1 ? iz + 1 : iz;
  const Real tx = cx - ix, ty = cy - iy, tz = cz - iz;
  const long nxny = (long)d.nx * d.ny;
  const int off = d.offset;
  const auto at = [&](int x, int y, int z) {
    return (Real)pool(off + (long)z * nxny + (long)y * d.nx + x);
  };
  const Real c00 = at(ix, iy, iz) * (1 - tx) + at(ix1, iy, iz) * tx;
  const Real c10 = at(ix, iy1, iz) * (1 - tx) + at(ix1, iy1, iz) * tx;
  const Real c01 = at(ix, iy, iz1) * (1 - tx) + at(ix1, iy, iz1) * tx;
  const Real c11 = at(ix, iy1, iz1) * (1 - tx) + at(ix1, iy1, iz1) * tx;
  const Real c0 = c00 * (1 - ty) + c10 * ty;
  const Real c1 = c01 * (1 - ty) + c11 * ty;
  const Real val = c0 * (1 - tz) + c1 * tz;
  const Real rx = (d.invSpacing.x > Real(0)) ? (fx - cx) / d.invSpacing.x : Real(0);
  const Real ry = (d.invSpacing.y > Real(0)) ? (fy - cy) / d.invSpacing.y : Real(0);
  const Real rz = (d.invSpacing.z > Real(0)) ? (fz - cz) / d.invSpacing.z : Real(0);
  const Real residual = detail::hdSqrt(rx * rx + ry * ry + rz * rz);
  return (d.extension == GridExtension::kObject) ? val + residual : val - residual;
}

/// Evaluate a single LEAF node at a point already in that node's canonical frame. No transform, no
/// recursion — the formulas all live in geom::prim.
template <class Real, class Grids, class Pool>
PECLET_HD Real evalLeaf(const ShapeNode<Real>& n, Vec3<Real> p, const Grids& grids,
                        const Pool& pool) {
  switch (n.kind) {
    case kSphere:
      return prim::Sphere<Real>{n.params[0]}.eval(p);
    case kBox:
      return prim::Box<Real>{n.params[0], n.params[1], n.params[2]}.eval(p);
    case kHollowCylinder:
      return prim::HollowCylinder<Real>{n.params[0], n.params[1], n.params[2]}.eval(p);
    case kHollowCylinderShell:
      return prim::HollowCylinderShell<Real>{n.params[0], n.params[1], n.params[2]}.eval(p);
    case kCapsule:
      return prim::Capsule<Real>{n.params[0], n.params[1]}.eval(p);
    case kTorus:
      return prim::Torus<Real>{n.params[0], n.params[1]}.eval(p);
    case kCone:
      return prim::Cone<Real>{n.params[0], n.params[1], n.params[2]}.eval(p);
    case kEllipsoid:
      return prim::Ellipsoid<Real>{n.params[0], n.params[1], n.params[2]}.eval(p);
    case kSuperquadric:
      return prim::Superquadric<Real>{n.params[0], n.params[1], n.params[2], n.params[3]}.eval(p);
    case kGrid:
      return sampleGrid(p, grids(n.aux0), pool);
    default:
      return Real(1e9);  // unknown kind: same "infinitely far" sentinel dem's sdfEval returns
  }
}

/// Maximum shape-tree depth the iterative evaluator supports. A stirrer (shaft union blades) is
/// depth 2-3; 16 is far above anything a scene should need, and it bounds the on-stack frame array
/// so the evaluator allocates nothing and stays device-safe (no recursion — contract 5).
constexpr int kMaxTreeDepth = 16;

/// Evaluate the shape tree rooted at `root` at point `p` (in the root's parent frame).
///
/// Iterative post-order walk over the flat node array with an explicit frame stack. A node's
/// children are evaluated at the node's CANONICAL point and their results scaled by the node's
/// scale on the way out, so `d_parent = s * d_canonical` composes correctly down the tree.
///
/// Returns 1e9 if the tree exceeds kMaxTreeDepth or references a child out of range — a malformed
/// scene reads as "infinitely far away" rather than corrupting memory.
template <class Real, class Nodes, class Grids, class Pool>
PECLET_HD Real evalTree(const Nodes& nodes, int nodeCount, int root, Vec3<Real> p,
                        const Grids& grids, const Pool& pool) {
  struct Frame {
    int node;
    int stage;  ///< 0 = entered, 1 = left child done, 2 = both done
    Vec3<Real> can;
    Real scale;
    Real left;
  };
  Frame stack[kMaxTreeDepth];
  int sp = 0;
  Real result = Real(1e9);

  if (root < 0 || root >= nodeCount)
    return result;

  {
    const auto& n = nodes(root);
    stack[0] = Frame{root, 0, toCanonical(n.transform, p), n.transform.scale, Real(0)};
    sp = 1;
  }

  while (sp > 0) {
    Frame& f = stack[sp - 1];
    const auto& n = nodes(f.node);

    if (n.kind < kCsgBase) {  // leaf: evaluate and unwind
      result = f.scale * evalLeaf(n, f.can, grids, pool);
      --sp;
    } else if (f.stage == 0) {  // descend into the left child
      if (n.aux0 < 0 || n.aux0 >= nodeCount || sp >= kMaxTreeDepth)
        return Real(1e9);
      f.stage = 1;
      const auto& c = nodes(n.aux0);
      stack[sp] = Frame{n.aux0, 0, toCanonical(c.transform, f.can), c.transform.scale, Real(0)};
      ++sp;
      continue;
    } else if (f.stage == 1) {  // left value in hand; descend into the right child
      if (n.aux1 < 0 || n.aux1 >= nodeCount || sp >= kMaxTreeDepth)
        return Real(1e9);
      f.left = result;
      f.stage = 2;
      const auto& c = nodes(n.aux1);
      stack[sp] = Frame{n.aux1, 0, toCanonical(c.transform, f.can), c.transform.scale, Real(0)};
      ++sp;
      continue;
    } else {  // both children evaluated: combine
      const Real l = f.left, r = result;
      Real combined;
      if (n.kind == kUnion)
        combined = detail::hdMin(l, r);
      else if (n.kind == kIntersection)
        combined = detail::hdMax(l, r);
      else  // kDifference: L minus R
        combined = detail::hdMax(l, -r);
      result = f.scale * combined;
      --sp;
    }
  }
  return result;
}

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_SCENE_HPP

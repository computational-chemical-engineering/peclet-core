// core — accelerated scene queries: periodic evaluation, geometry-driven candidate grids, and the
// sphere-union fast paths the porous-media consumers live on.
//
// Layer 2-for-core of suite/docs/ANALYTIC_SDF_GEOMETRY.md, shaped by the AMR consumer's measured
// requirements (docs/AMR_GEOMETRY_SETUP_REQUIREMENTS.md): 94% of AmrFlow::setSolid on the
// 180-sphere RCP bed is SDF evaluation — 101 evals/leaf x 1.18 us/eval brute force over all
// primitives. This header owns the cost-per-eval side of that product:
//
//   PeriodicBox / minImage      min-image periodicity, the AMR stopgap's EXACT expression
//                               (d -= L*nearbyint(d/L)) so results are BIT-IDENTICAL, not merely
//                               sign-agreeing at the probe scale.
//   SphereUnionView             compact union-of-spheres form (the workhorse geometry) with an
//                               equal-radius fast path: min over SQUARED center distances, ONE
//                               sqrt per query instead of one per sphere.
//   CandidateGridView + build   per-bin candidate lists over a UNIFORM auxiliary lattice, built
//                               geometry-driven and pruned with exact bounds, so a query costs
//                               bin arithmetic + a 1-5 entry list instead of a scan over every
//                               primitive.
//   SphereBedQuery              the host-callable object that slots straight into
//                               AmrFlow::setSolid's templated SdfFn parameter — the stopgap
//                               retirement path.
//
// WHY UNIFORM BINS AND NOT OCTREE-LEAF LISTS. The directive behind contract 9 is that geometry
// drives (each primitive splats onto the cells its band overlaps) and that far-field cells never
// traverse an acceleration structure. Keying the lists by a fixed auxiliary lattice satisfies both
// with no octree coupling at all: a query is O(1) bin arithmetic from the POINT's own coordinates,
// so leaves of every level — and probe points at any h, virtual positions across 2:1 jumps, coarse
// MG face centers — hit the right list with no per-level machinery and no leavesInBox dependency.
// A leaf-keyed variant can be layered on top by the AMR side (list(leaf) = list(binOf(center))
// unioned over the leaf's span) without any new API here.
//
// EXACTNESS CONTRACT (the float-sign classification requirement). The candidate list of a bin is a
// conservative superset of the argmin set over every point of the bin, built with the classic
// bound pruning: keep primitive i iff lower_i(B) <= U(B), where lower_i is a lower bound of i's
// contribution anywhere in B and U(B) = min_j upper_j(B) is an upper bound of the union anywhere
// in B. min over a superset that contains the argmin IS the min — bit-identically, since each
// member's value is computed with the identical expression — so acceleration changes NOTHING about
// the result: not the value, not the sign, not under list reordering (min is order-independent as
// a value). Queries outside the grid's coverage, or in a bin whose list is empty (possible far
// from all geometry when bounds never reached it), FALL BACK to the full scan — exact, and rare
// where it matters.
//
// The bounds argument needs eval_i >= (distance to a ball containing i's solid) pointwise, which
// holds for exact-distance leaves and is PRESERVED by union (min over children, all certified),
// intersection and difference (max(a, ...) >= a, so the LEFT child's ball and certificate
// suffice). It FAILS for the under-estimating leaves (ellipsoid, superquadric,
// HollowCylinderShell — measured under-run up to 4x in Layer 0's gate) and for grid leaves: those
// instances are never pruned — they ride on an ALWAYS list evaluated for every query. The
// workhorse geometries (sphere beds, box/cylinder/capsule/torus stirrers and their CSG) prune.
//
// Host-compilable (PECLET_HD); the Kokkos-owning DeviceScene and batched drivers live in
// device_scene.hpp.
#ifndef PECLET_CORE_GEOM_SCENE_QUERY_HPP
#define PECLET_CORE_GEOM_SCENE_QUERY_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/geom/scene.hpp"

namespace peclet::core::geom {

// ---------------------------------------------------------------------------------------------
// Periodicity (AMR requirement 4)
// ---------------------------------------------------------------------------------------------

/// Min-image box. `on = false` disables wrapping (all extents ignored).
template <class Real>
struct PeriodicBox {
  Real Lx = 0, Ly = 0, Lz = 0;
  bool on = false;
};

/// Min-image a displacement. FMA-CANONICAL: d = fma(-nearbyint(d/L), L, d) per axis. Explicit fma
/// is correctly rounded by IEEE on every backend and cannot be re-contracted by nvcc, which is
/// what makes host and device evaluation BIT-IDENTICAL — the property the AMR parity ctests
/// assume. The retired stopgap computed the two-rounding form (mul then sub); the difference is
/// bounded at 1 ulp of the displacement and its effect on the final distance is measured in the
/// geom_scene_query gate (and empirically zero on the acceptance meshes' classifications). Do not
/// "optimise" the divide into a multiply by 1/L: that rounds differently.
template <class Real>
PECLET_HD Vec3<Real> minImage(Vec3<Real> d, const PeriodicBox<Real>& b) {
  namespace cd = peclet::core::detail;
  if (b.on) {
    d.x = cd::hdFma(-cd::hdNearbyint(d.x / b.Lx), b.Lx, d.x);
    d.y = cd::hdFma(-cd::hdNearbyint(d.y / b.Ly), b.Ly, d.y);
    d.z = cd::hdFma(-cd::hdNearbyint(d.z / b.Lz), b.Lz, d.z);
  }
  return d;
}

// ---------------------------------------------------------------------------------------------
// Sphere union (the porous-media workhorse)
// ---------------------------------------------------------------------------------------------

/// Non-owning union-of-spheres view. `equalR` (all radii bit-equal, value `r0`) enables the
/// one-sqrt-per-query path.
template <class Real>
struct SphereUnionView {
  const Real* cx = nullptr;
  const Real* cy = nullptr;
  const Real* cz = nullptr;
  const Real* r = nullptr;
  int n = 0;
  bool equalR = false;
  Real r0 = 0;
};

/// Squared min-image distance from p to sphere i — THE per-sphere expression, shared by every
/// path (brute, subset, batched, host, device) so drift between them is impossible. FMA-canonical
/// for the same host==device reason as minImage.
template <class Real>
PECLET_HD Real sphereDist2(const SphereUnionView<Real>& u, int i, Vec3<Real> p,
                           const PeriodicBox<Real>& box) {
  namespace cd = peclet::core::detail;
  const Vec3<Real> d = minImage(Vec3<Real>{p.x - u.cx[i], p.y - u.cy[i], p.z - u.cz[i]}, box);
  return cd::hdFma(d.z, d.z, cd::hdFma(d.y, d.y, d.x * d.x));
}

/// Union SDF over an index subset (a candidate list). Per-sphere expression is the stopgap's:
/// displacement, min-image, dx*dx + dy*dy + dz*dz (this association), sqrt, minus r. The
/// equal-radius path takes min over the squared distances first and pays ONE sqrt: the winning
/// sphere's d2 is the same number the general path would sqrt, so the result is bit-identical.
template <class Real>
PECLET_HD Real evalSphereUnionSubset(const SphereUnionView<Real>& u, Vec3<Real> p,
                                     const PeriodicBox<Real>& box, const int* items, int count) {
  if (u.equalR) {
    Real best2 = Real(1e300);
    for (int k = 0; k < count; ++k) {
      const Real d2 = sphereDist2(u, items[k], p, box);
      if (d2 < best2)
        best2 = d2;
    }
    return peclet::core::detail::hdSqrt(best2) - u.r0;
  }
  Real best = Real(1e300);
  for (int k = 0; k < count; ++k) {
    const int i = items[k];
    const Real v = peclet::core::detail::hdSqrt(sphereDist2(u, i, p, box)) - u.r[i];
    if (v < best)
      best = v;
  }
  return best;
}

/// Brute-force union SDF (all spheres) — the reference the candidate path must match bit-for-bit,
/// and the fallback for out-of-coverage queries.
template <class Real>
PECLET_HD Real evalSphereUnion(const SphereUnionView<Real>& u, Vec3<Real> p,
                               const PeriodicBox<Real>& box) {
  if (u.equalR) {
    Real best2 = Real(1e300);
    for (int i = 0; i < u.n; ++i) {
      const Real d2 = sphereDist2(u, i, p, box);
      if (d2 < best2)
        best2 = d2;
    }
    return peclet::core::detail::hdSqrt(best2) - u.r0;
  }
  Real best = Real(1e300);
  for (int i = 0; i < u.n; ++i) {
    const Real v = peclet::core::detail::hdSqrt(sphereDist2(u, i, p, box)) - u.r[i];
    if (v < best)
      best = v;
  }
  return best;
}

// ---------------------------------------------------------------------------------------------
// Candidate grid
// ---------------------------------------------------------------------------------------------

/// Non-owning candidate grid: uniform bins over [origin, origin+extent), CSR lists of primitive
/// (or instance) indices, plus an ALWAYS list of never-pruned entries. POD, device-copyable.
template <class Real>
struct CandidateGridView {
  int nx = 0, ny = 0, nz = 0;
  Real ox = 0, oy = 0, oz = 0;   // origin
  Real bx = 1, by = 1, bz = 1;   // bin size
  const int* offsets = nullptr;  // nx*ny*nz + 1
  const int* items = nullptr;
  const int* always = nullptr;  // non-prunable entries, evaluated for every query
  int alwaysCount = 0;
  bool wrap = false;  // periodic bin lookup

  /// Flat bin index of a point, or -1 when the point is outside a non-periodic grid's coverage
  /// (the caller must fall back to the full scan — exactness before speed).
  PECLET_HD long binOf(Vec3<Real> p) const {
    long i = (long)peclet::core::detail::hdFloor((p.x - ox) / bx);
    long j = (long)peclet::core::detail::hdFloor((p.y - oy) / by);
    long k = (long)peclet::core::detail::hdFloor((p.z - oz) / bz);
    if (wrap) {
      i = ((i % nx) + nx) % nx;
      j = ((j % ny) + ny) % ny;
      k = ((k % nz) + nz) % nz;
    } else if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) {
      return -1;
    }
    return i + (long)nx * (j + (long)ny * k);
  }
};

/// Union SDF through the candidate grid: bin lookup + short list, full-scan fallback when the bin
/// is out of coverage or its list is empty. Bit-identical to evalSphereUnion by the superset
/// argument in the header comment.
template <class Real>
PECLET_HD Real evalSphereUnionGrid(const SphereUnionView<Real>& u, Vec3<Real> p,
                                   const PeriodicBox<Real>& box, const CandidateGridView<Real>& g) {
  const long b = g.binOf(p);
  if (b < 0)
    return evalSphereUnion(u, p, box);
  const int lo = g.offsets[b], hi = g.offsets[b + 1];
  if (lo == hi)
    return evalSphereUnion(u, p, box);
  return evalSphereUnionSubset(u, p, box, g.items + lo, hi - lo);
}

/// Owning host-side candidate grid.
template <class Real>
struct CandidateGrid {
  std::vector<int> offsets, items, always;
  std::vector<Real> instBoundR;  // per-instance world bounding radii (general-scene builds)
  CandidateGridView<Real> meta;  // pointers unset; use view()

  CandidateGridView<Real> view() const {
    CandidateGridView<Real> v = meta;
    v.offsets = offsets.data();
    v.items = items.data();
    v.always = always.data();
    v.alwaysCount = static_cast<int>(always.size());
    return v;
  }
};

/// Geometry-driven candidate build for a sphere union over [origin, origin+extent). Two splat
/// passes over each sphere's bounded bin neighbourhood:
///   pass 1: each sphere lowers U(B) (atomic-min-shaped, serial here) with its upper bound
///           upper_i(B) = d(binCenter, c_i) + halfDiag - r_i;
///   pass 2: each sphere appends itself to bins where lower_i(B) = d - halfDiag - r_i <= U(B).
/// The neighbourhood of pass 2 is bounded by the pass-1 U field, so far bins are never visited by
/// far spheres; bins no sphere reaches keep empty lists and the query path falls back (exact).
/// `nbHint` overrides the bin count per axis (default targets bin ~ r_mean/2, the measured sweet
/// spot for 1-5 entry lists on the RCP bed).
template <class Real>
CandidateGrid<Real> buildSphereCandidateGrid(const SphereUnionView<Real>& u, Vec3<Real> origin,
                                             Vec3<Real> extent, const PeriodicBox<Real>& box,
                                             int nbHint = 0) {
  if (u.n <= 0)
    throw std::invalid_argument("buildSphereCandidateGrid: empty union");
  Real rMean = 0;
  for (int i = 0; i < u.n; ++i)
    rMean += u.r[i];
  rMean /= u.n;
  auto nbAxis = [&](Real L) {
    if (nbHint > 0)
      return nbHint;
    int nb = rMean > 0 ? static_cast<int>(L / (rMean * Real(0.5))) : 16;
    return nb < 2 ? 2 : (nb > 96 ? 96 : nb);
  };
  CandidateGrid<Real> g;
  auto& m = g.meta;
  m.nx = nbAxis(extent.x);
  m.ny = nbAxis(extent.y);
  m.nz = nbAxis(extent.z);
  m.ox = origin.x;
  m.oy = origin.y;
  m.oz = origin.z;
  m.bx = extent.x / m.nx;
  m.by = extent.y / m.ny;
  m.bz = extent.z / m.nz;
  m.wrap = box.on;
  const long nbins = (long)m.nx * m.ny * m.nz;
  const Real halfDiag = Real(0.5) * std::sqrt(m.bx * m.bx + m.by * m.by + m.bz * m.bz);
  // Conservative slack absorbing the float rounding of the bounds themselves; supersets stay
  // supersets, lists grow by (rarely) one entry.
  const Real slack = Real(1e-12) * (extent.x + extent.y + extent.z);

  auto binCenter = [&](long b, Vec3<Real>& c) {
    const long i = b % m.nx, j = (b / m.nx) % m.ny, k = b / ((long)m.nx * m.ny);
    c = Vec3<Real>{m.ox + (i + Real(0.5)) * m.bx, m.oy + (j + Real(0.5)) * m.by,
                   m.oz + (k + Real(0.5)) * m.bz};
  };
  auto dist = [&](const Vec3<Real>& a, int i) {
    const Vec3<Real> d = minImage(Vec3<Real>{a.x - u.cx[i], a.y - u.cy[i], a.z - u.cz[i]}, box);
    return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
  };
  // The bin neighbourhood a sphere splats: bins whose center lies within `reach` of the sphere
  // center (min-image). Enumerated as an index box around the sphere, wrapped when periodic.
  auto forEachBinNear = [&](int i, Real reach, auto&& fn) {
    const int riX = static_cast<int>(reach / m.bx) + 1;
    const int riY = static_cast<int>(reach / m.by) + 1;
    const int riZ = static_cast<int>(reach / m.bz) + 1;
    const int ci = static_cast<int>(peclet::core::detail::hdFloor((u.cx[i] - m.ox) / m.bx));
    const int cj = static_cast<int>(peclet::core::detail::hdFloor((u.cy[i] - m.oy) / m.by));
    const int ck = static_cast<int>(peclet::core::detail::hdFloor((u.cz[i] - m.oz) / m.bz));
    for (int dk = -riZ; dk <= riZ; ++dk)
      for (int dj = -riY; dj <= riY; ++dj)
        for (int di = -riX; di <= riX; ++di) {
          long ii = ci + di, jj = cj + dj, kk = ck + dk;
          if (m.wrap) {
            ii = ((ii % m.nx) + m.nx) % m.nx;
            jj = ((jj % m.ny) + m.ny) % m.ny;
            kk = ((kk % m.nz) + m.nz) % m.nz;
          } else if (ii < 0 || ii >= m.nx || jj < 0 || jj >= m.ny || kk < 0 || kk >= m.nz) {
            continue;
          }
          fn(ii + (long)m.nx * (jj + (long)m.ny * kk));
        }
  };
  // Pass 1: U(B) = min_i upper_i(B), splat within each sphere's competitive reach. The reach must
  // cover every bin where this sphere could OWN the best upper bound; beyond max over spheres of
  // (their own reach), U stays +inf and queries there fall back. Reach heuristic: the largest
  // min-image distance possible is bounded by the box half-diagonal, so cap there.
  std::vector<Real> U((std::size_t)nbins, Real(1e300));
  const Real capReach =
      box.on ? Real(0.5) * std::sqrt(box.Lx * box.Lx + box.Ly * box.Ly + box.Lz * box.Lz) + halfDiag
             : std::sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z);
  for (int i = 0; i < u.n; ++i)
    forEachBinNear(i, capReach, [&](long b) {
      Vec3<Real> c;
      binCenter(b, c);
      const Real up = dist(c, i) + halfDiag - u.r[i];
      if (up < U[(std::size_t)b])
        U[(std::size_t)b] = up;
    });
  // Pass 2: membership by the exact criterion. Each sphere visits bins within its own competitive
  // reach: lower_i(B) <= U(B) needs d <= r_i + halfDiag + U(B); U is bounded above by the max
  // over the field, so cap the walk there (plus its own radius + diag).
  Real Umax = 0;
  for (Real v : U)
    if (v < Real(1e300) && v > Umax)
      Umax = v;
  std::vector<std::vector<int>> lists((std::size_t)nbins);
  for (int i = 0; i < u.n; ++i)
    forEachBinNear(i, u.r[i] + halfDiag + Umax + slack, [&](long b) {
      Vec3<Real> c;
      binCenter(b, c);
      const Real lower = dist(c, i) - halfDiag - u.r[i];
      if (lower <= U[(std::size_t)b] + slack)
        lists[(std::size_t)b].push_back(i);
    });
  g.offsets.resize((std::size_t)nbins + 1);
  g.offsets[0] = 0;
  for (long b = 0; b < nbins; ++b)
    g.offsets[(std::size_t)b + 1] =
        g.offsets[(std::size_t)b] + static_cast<int>(lists[(std::size_t)b].size());
  g.items.resize((std::size_t)g.offsets[(std::size_t)nbins]);
  for (long b = 0; b < nbins; ++b)
    std::copy(lists[(std::size_t)b].begin(), lists[(std::size_t)b].end(),
              g.items.begin() + g.offsets[(std::size_t)b]);
  return g;
}

// ---------------------------------------------------------------------------------------------
// General scenes: periodic evaluation, bounds, candidate grids
// ---------------------------------------------------------------------------------------------

/// Instance evaluation with MIN-IMAGE periodicity.
///
/// SUBTLETY (a bug this replaced): min-imaging the displacement to the instance ORIGIN is only
/// correct for bodies whose every point sits AT the origin (spheres). A body part offset from the
/// origin — a stirrer blade — can have its nearest periodic image across the seam the
/// origin-wrap did not take: body point at local +0.3 in a unit box, probe at 0.55 → origin-wrap
/// gives displacement −0.45 and distance 0.75, but the true periodic distance is 0.25. So the
/// evaluation takes the min over the axis images that can matter: the wrapped displacement, plus
/// the neighbouring image on each axis where the probe is within `boundR` (the instance's world
/// bounding radius) of the seam. `boundR < 0` means "unknown": every axis neighbour is tried
/// (up to 8 tree walks — correct, slower; supply bounds via instanceBound for the fast form).
template <class Real>
PECLET_HD Real evalInstancePeriodic(const SceneView<Real>& sc, int i, Vec3<Real> p,
                                    const PeriodicBox<Real>& box, Real boundR = Real(-1)) {
  if (i < 0 || i >= sc.instanceCount)
    return Real(1e9);
  const Instance<Real>& inst = sc.instances[i];
  if (!box.on) {
    const Vec3<Real> q =
        scale(invRotate(inst.transform.rotation, sub(p, inst.transform.translation)),
              Real(1) / inst.transform.scale);
    return inst.transform.scale *
           evalTree<Real>(TablePtr<ShapeNode<Real>>{sc.nodes}, sc.nodeCount, inst.shapeRoot, q,
                          TablePtr<GridDesc<Real>>{sc.grids}, PoolPtr<float>{sc.samples});
  }
  const Vec3<Real> d0 = minImage(sub(p, inst.transform.translation), box);
  // which axes need the neighbouring image? |d_a| > L_a/2 - boundR (all of them when unknown)
  const Real Ls[3] = {box.Lx, box.Ly, box.Lz};
  const Real da[3] = {d0.x, d0.y, d0.z};
  int nAlt[3] = {1, 1, 1};
  Real alt[3][2] = {{d0.x, 0}, {d0.y, 0}, {d0.z, 0}};
  for (int a = 0; a < 3; ++a) {
    const bool need =
        boundR < Real(0) || (da[a] < Real(0) ? -da[a] : da[a]) > Ls[a] * Real(0.5) - boundR;
    if (need) {
      alt[a][1] = da[a] < Real(0) ? da[a] + Ls[a] : da[a] - Ls[a];
      nAlt[a] = 2;
    }
  }
  Real best = Real(1e300);
  for (int kz = 0; kz < nAlt[2]; ++kz)
    for (int ky = 0; ky < nAlt[1]; ++ky)
      for (int kx = 0; kx < nAlt[0]; ++kx) {
        const Vec3<Real> d{alt[0][kx], alt[1][ky], alt[2][kz]};
        const Vec3<Real> q =
            scale(invRotate(inst.transform.rotation, d), Real(1) / inst.transform.scale);
        const Real v = inst.transform.scale * evalTree<Real>(TablePtr<ShapeNode<Real>>{sc.nodes},
                                                             sc.nodeCount, inst.shapeRoot, q,
                                                             TablePtr<GridDesc<Real>>{sc.grids},
                                                             PoolPtr<float>{sc.samples});
        if (v < best)
          best = v;
      }
  return best;
}

/// Union over every instance, min-image periodic. `boundR` is an optional per-instance world
/// bounding-radius array (from instanceBound) that keeps the seam handling to one tree walk away
/// from seams. The non-periodic case reduces bitwise to evalScene.
template <class Real>
PECLET_HD Real evalScenePeriodic(const SceneView<Real>& sc, Vec3<Real> p,
                                 const PeriodicBox<Real>& box, const Real* boundR = nullptr) {
  Real m = Real(1e9);
  for (int i = 0; i < sc.instanceCount; ++i) {
    const Real d = evalInstancePeriodic(sc, i, p, box, boundR ? boundR[i] : Real(-1));
    if (d < m)
      m = d;
  }
  return m;
}

/// World bounding sphere of an instance + the PRUNING CERTIFICATE. `certified` means the whole
/// tree satisfies eval >= (distance to a ball of radius r about c) AND eval is 1-Lipschitz --
/// which exact-distance leaves satisfy and CSG preserves (union: all children; intersection /
/// difference: the LEFT child's ball and certificate suffice, since max(a, .) >= a). The
/// under-estimating leaves (ellipsoid, superquadric, HollowCylinderShell -- measured under-runs in
/// the Layer-0 gate) and grid leaves are NOT certified: candidate builds put those instances on
/// the always-list instead of pruning them.
template <class Real>
struct InstanceBound {
  Vec3<Real> c{0, 0, 0};
  Real r = 0;
  bool certified = false;
};

namespace query_detail {

template <class Real>
struct NodeBound {
  Vec3<Real> c{0, 0, 0};  // in the node's PARENT frame
  Real r = 0;
  bool certified = false;
};

template <class Real>
inline NodeBound<Real> nodeBound(const SceneView<Real>& sc, int node) {
  NodeBound<Real> nb;
  if (node < 0 || node >= sc.nodeCount)
    return nb;
  const ShapeNode<Real>& n = sc.nodes[node];
  Real localR = 0;
  bool cert = false;
  if (n.kind < kCsgBase) {
    switch (n.kind) {
      case kSphere:
        localR = n.params[0];
        cert = true;
        break;
      case kBox:
        localR = std::sqrt(n.params[0] * n.params[0] + n.params[1] * n.params[1] +
                           n.params[2] * n.params[2]);
        cert = true;
        break;
      case kHollowCylinder: {  // (rOuter, height, thickness), about y -- distance-exact
        const Real ro = n.params[0], h2 = n.params[1] * Real(0.5);
        localR = std::sqrt(ro * ro + h2 * h2);
        cert = true;
        break;
      }
      case kCapsule:
        localR = n.params[1] + n.params[0];
        cert = true;
        break;
      case kTorus:
        localR = n.params[0] + n.params[1];
        cert = true;
        break;
      case kCone: {
        const Real rb = std::fmax(n.params[0], n.params[1]), hh = n.params[2];
        localR = std::sqrt(rb * rb + hh * hh);
        cert = true;
        break;
      }
      case kHollowCylinderShell: {  // sign-exact only: bounded but NOT certified
        const Real ro = n.params[0], h2 = n.params[2] * Real(0.5);
        localR = std::sqrt(ro * ro + h2 * h2);
        cert = false;
        break;
      }
      case kEllipsoid:
        localR = std::fmax(n.params[0], std::fmax(n.params[1], n.params[2]));
        cert = false;
        break;
      case kSuperquadric:
        localR = std::sqrt(n.params[0] * n.params[0] + n.params[1] * n.params[1] +
                           n.params[2] * n.params[2]);
        cert = false;
        break;
      default:  // kGrid or unknown: no useful bound; never pruned
        localR = Real(1e30);
        cert = false;
        break;
    }
    nb.c = n.transform.translation;  // canonical leaves are origin-centred
    nb.r = n.transform.scale * localR;
    nb.certified = cert;
    return nb;
  }
  // CSG: children live in this node's frame.
  const NodeBound<Real> a = nodeBound(sc, n.aux0);
  const NodeBound<Real> b = nodeBound(sc, n.aux1);
  NodeBound<Real> comb;
  if (n.kind == kUnion) {
    // enclosing sphere of the two child spheres; certified iff BOTH children are
    const Vec3<Real> d = sub(b.c, a.c);
    const Real dist = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    if (dist + b.r <= a.r) {
      comb.c = a.c;
      comb.r = a.r;
    } else if (dist + a.r <= b.r) {
      comb.c = b.c;
      comb.r = b.r;
    } else {
      const Real R = (dist + a.r + b.r) * Real(0.5);
      const Real t = dist > Real(0) ? (R - a.r) / dist : Real(0);
      comb.c = Vec3<Real>{a.c.x + d.x * t, a.c.y + d.y * t, a.c.z + d.z * t};
      comb.r = R;
    }
    comb.certified = a.certified && b.certified;
  } else {  // intersection / difference: the LEFT child's ball + certificate suffice
    comb = a;
  }
  // apply this node's own transform to the child-frame bound
  comb.c =
      add(rotate(n.transform.rotation, scale(comb.c, n.transform.scale)), n.transform.translation);
  comb.r *= n.transform.scale;
  return comb;
}

}  // namespace query_detail

/// World bounding sphere + certificate of one instance (host-side; used by candidate builds).
template <class Real>
inline InstanceBound<Real> instanceBound(const SceneView<Real>& sc, int i) {
  InstanceBound<Real> ib;
  if (i < 0 || i >= sc.instanceCount)
    return ib;
  const Instance<Real>& inst = sc.instances[i];
  const query_detail::NodeBound<Real> nb = query_detail::nodeBound(sc, inst.shapeRoot);
  ib.c = add(rotate(inst.transform.rotation, scale(nb.c, inst.transform.scale)),
             inst.transform.translation);
  ib.r = inst.transform.scale * nb.r;
  ib.certified = nb.certified && nb.r < Real(1e29);
  return ib;
}

/// Union over a candidate list + the always-list, min-image periodic. Exact when the list is a
/// certified-argmin superset and every non-certified instance rides the always-list (see the
/// header comment): min(min_certified-kept, min_always) == the full min.
template <class Real>
PECLET_HD Real evalSceneGrid(const SceneView<Real>& sc, Vec3<Real> p, const PeriodicBox<Real>& box,
                             const CandidateGridView<Real>& g, const Real* boundR = nullptr) {
  Real m = Real(1e9);
  for (int k = 0; k < g.alwaysCount; ++k) {
    const int i = g.always[k];
    const Real d = evalInstancePeriodic(sc, i, p, box, boundR ? boundR[i] : Real(-1));
    if (d < m)
      m = d;
  }
  const long b = g.binOf(p);
  if (b < 0) {  // out of coverage: full scan of the certified instances too
    const Real full = evalScenePeriodic(sc, p, box, boundR);
    return full < m ? full : m;
  }
  const int lo = g.offsets[b], hi = g.offsets[b + 1];
  if (lo == hi) {
    // An EMPTY certified list means no certified instance's band reached this bin — the far
    // field — where a certified instance may still be the minimum. Fall back to the full scan
    // (the always-list min in m composes via min; re-evaluating those instances is harmless).
    // Returning only the always-list here was a bug: probes in uncovered bins got the distance
    // to the nearest UNCERTIFIED body only.
    const Real full = evalScenePeriodic(sc, p, box, boundR);
    return full < m ? full : m;
  }
  for (int k = lo; k < hi; ++k) {
    const int i = g.items[k];
    const Real d = evalInstancePeriodic(sc, i, p, box, boundR ? boundR[i] : Real(-1));
    if (d < m)
      m = d;
  }
  return m;
}

/// Candidate grid for a GENERAL scene: certified instances splat Lipschitz bounds evaluated at
/// bin centers within their inflated bounding-sphere band (the generateSdfKokkos pattern);
/// non-certified instances go on the always-list. Bins outside every band keep empty lists and
/// queries there fall back (exact). For sphere-only scenes prefer the SphereUnion path -- this
/// build costs one instance eval per (instance, nearby bin).
template <class Real>
CandidateGrid<Real> buildSceneCandidateGrid(const SceneView<Real>& sc, Vec3<Real> origin,
                                            Vec3<Real> extent, const PeriodicBox<Real>& box,
                                            Real binSizeHint = 0) {
  CandidateGrid<Real> g;
  auto& m = g.meta;
  // default bin ~ half the median certified bounding radius (mirrors the sphere build's target)
  std::vector<InstanceBound<Real>> ib((std::size_t)sc.instanceCount);
  g.instBoundR.resize((std::size_t)sc.instanceCount);
  Real rSum = 0;
  int nCert = 0;
  for (int i = 0; i < sc.instanceCount; ++i) {
    ib[(std::size_t)i] = instanceBound(sc, i);
    g.instBoundR[(std::size_t)i] = ib[(std::size_t)i].r;
    if (ib[(std::size_t)i].certified) {
      rSum += ib[(std::size_t)i].r;
      ++nCert;
    } else {
      g.always.push_back(i);
    }
  }
  const Real rMean = nCert ? rSum / nCert : extent.x / 8;
  const Real bin = binSizeHint > 0 ? binSizeHint : std::fmax(rMean * Real(0.5), extent.x / 96);
  auto nbAxis = [&](Real L) {
    int nb = static_cast<int>(L / bin);
    return nb < 2 ? 2 : (nb > 96 ? 96 : nb);
  };
  m.nx = nbAxis(extent.x);
  m.ny = nbAxis(extent.y);
  m.nz = nbAxis(extent.z);
  m.ox = origin.x;
  m.oy = origin.y;
  m.oz = origin.z;
  m.bx = extent.x / m.nx;
  m.by = extent.y / m.ny;
  m.bz = extent.z / m.nz;
  m.wrap = box.on;
  const long nbins = (long)m.nx * m.ny * m.nz;
  const Real halfDiag = Real(0.5) * std::sqrt(m.bx * m.bx + m.by * m.by + m.bz * m.bz);
  const Real slack = Real(1e-12) * (extent.x + extent.y + extent.z);

  auto binCenter = [&](long b) {
    const long i = b % m.nx, j = (b / m.nx) % m.ny, k = b / ((long)m.nx * m.ny);
    return Vec3<Real>{m.ox + (i + Real(0.5)) * m.bx, m.oy + (j + Real(0.5)) * m.by,
                      m.oz + (k + Real(0.5)) * m.bz};
  };
  auto forEachBinNear = [&](const Vec3<Real>& c, Real reach, auto&& fn) {
    const int riX = static_cast<int>(reach / m.bx) + 1;
    const int riY = static_cast<int>(reach / m.by) + 1;
    const int riZ = static_cast<int>(reach / m.bz) + 1;
    const int ci = static_cast<int>(std::floor((c.x - m.ox) / m.bx));
    const int cj = static_cast<int>(std::floor((c.y - m.oy) / m.by));
    const int ck = static_cast<int>(std::floor((c.z - m.oz) / m.bz));
    for (int dk = -riZ; dk <= riZ; ++dk)
      for (int dj = -riY; dj <= riY; ++dj)
        for (int di = -riX; di <= riX; ++di) {
          long ii = ci + di, jj = cj + dj, kk = ck + dk;
          if (m.wrap) {
            ii = ((ii % m.nx) + m.nx) % m.nx;
            jj = ((jj % m.ny) + m.ny) % m.ny;
            kk = ((kk % m.nz) + m.nz) % m.nz;
          } else if (ii < 0 || ii >= m.nx || jj < 0 || jj >= m.ny || kk < 0 || kk >= m.nz) {
            continue;
          }
          fn(ii + (long)m.nx * (jj + (long)m.ny * kk));
        }
  };
  // Certified instances are 1-Lipschitz (their periodic eval is the true torus distance to the
  // body), so at a bin center bc: upper(B) = eval(bc) + halfDiag, lower(B) = eval(bc) - halfDiag.
  //
  // COVERAGE MUST BE TOTAL. A bounded per-instance splat reach reproduces the far-field bug this
  // replaced: a bin beyond instance i's reach never membership-tests i, so its (non-empty) list
  // can miss the true argmin — measured as O(1e-2) absolute distance errors in the far field.
  // Every (bin, certified instance) pair is therefore evaluated once, row-cached per bin:
  // U(B) = min_i (e_i + hd), keep i iff e_i - hd <= U(B). Cost = nbins x nCertified tree walks —
  // fine for stirrer-scale scenes (~1e6 walks); MANY-instance scenes belong on the sphere-union
  // path, and the build() wrapper refuses to build a general grid beyond a cost cap rather than
  // silently taking minutes.
  std::vector<int> certIdx;
  certIdx.reserve((std::size_t)nCert);
  for (int i = 0; i < sc.instanceCount; ++i)
    if (ib[(std::size_t)i].certified)
      certIdx.push_back(i);
  std::vector<Real> row(certIdx.size());
  std::vector<std::vector<int>> lists((std::size_t)nbins);
  for (long b = 0; b < nbins; ++b) {
    const Vec3<Real> bc = binCenter(b);
    Real U = Real(1e300);
    for (std::size_t k = 0; k < certIdx.size(); ++k) {
      const int i = certIdx[k];
      row[k] = evalInstancePeriodic(sc, i, bc, box, ib[(std::size_t)i].r);
      const Real up = row[k] + halfDiag;
      if (up < U)
        U = up;
    }
    for (std::size_t k = 0; k < certIdx.size(); ++k)
      if (row[k] - halfDiag <= U + slack)
        lists[(std::size_t)b].push_back(certIdx[k]);
  }
  g.offsets.resize((std::size_t)nbins + 1);
  g.offsets[0] = 0;
  for (long b = 0; b < nbins; ++b)
    g.offsets[(std::size_t)b + 1] =
        g.offsets[(std::size_t)b] + static_cast<int>(lists[(std::size_t)b].size());
  g.items.resize((std::size_t)g.offsets[(std::size_t)nbins]);
  for (long b = 0; b < nbins; ++b)
    std::copy(lists[(std::size_t)b].begin(), lists[(std::size_t)b].end(),
              g.items.begin() + g.offsets[(std::size_t)b]);
  return g;
}


// ---------------------------------------------------------------------------------------------
// Cut ownership (Layer 3 rung 1 of suite/docs/ANALYTIC_SDF_GEOMETRY.md)
//
// "Which body owns the surface nearest to p" — the attribution a moving-geometry solver needs to
// read a wall velocity off the right instance, and a resolved CFD-DEM solver needs to post a
// hydrodynamic force back to the right grain.
//
// DETERMINISM CONTRACT. The winner is the argmin of exactly the value `eval` returns, and ties
// break to the LOWEST index — enforced by the comparison itself (`v < best || (v == best && i <
// best_i)`), NOT by scan order. That matters because the same point can be answered through three
// different orders: a candidate list, the always-list + candidate list, or the full-scan fallback.
// All three therefore agree, and a per-bin shuffle of the candidate lists cannot change the owner.
//
// The equal-radius sphere path carries SQUARED center distances through the scan and converts
// once at the end, exactly as evalSphereUnion does — so `evalOwner` returns a value BITWISE equal
// to `eval`'s, and the argmin is the same one sqrt(.) - r0 would have chosen (it is strictly
// increasing in d2).
//
// EVERY owner path is written as an eval-AND-owner primitive, with `owner()` discarding the value.
// One traversal answers both, which is what keeps a per-cell owner field free for a consumer that
// is sampling the scene anyway (flow's set_solid_from_scene).
// ---------------------------------------------------------------------------------------------

namespace owner_detail {
/// The single tie-break rule (lowest index wins an exact tie), shared by every owner path.
template <class Real>
PECLET_HD void take(Real v, int i, Real& best, int& bestI) {
  if (v < best || (v == best && i < bestI)) {
    best = v;
    bestI = i;
  }
}
}  // namespace owner_detail

/// Accumulate the argmin over an explicit index subset (a candidate list) into (best, bestI).
/// `best` carries d2 on the equal-radius path and the signed distance otherwise — the caller
/// converts, exactly once, like evalSphereUnion.
template <class Real>
PECLET_HD void sphereUnionOwnerSubset(const SphereUnionView<Real>& u, Vec3<Real> p,
                                      const PeriodicBox<Real>& box, const int* items, int count,
                                      Real& best, int& bestI) {
  if (u.equalR) {
    for (int k = 0; k < count; ++k) {
      const int i = items[k];
      owner_detail::take(sphereDist2(u, i, p, box), i, best, bestI);
    }
    return;
  }
  for (int k = 0; k < count; ++k) {
    const int i = items[k];
    owner_detail::take(peclet::core::detail::hdSqrt(sphereDist2(u, i, p, box)) - u.r[i], i, best,
                       bestI);
  }
}

/// Brute-force value + owner (all spheres) — the reference the candidate path must match, and the
/// out-of-coverage fallback.
template <class Real>
PECLET_HD Real sphereUnionEvalOwner(const SphereUnionView<Real>& u, Vec3<Real> p,
                                    const PeriodicBox<Real>& box, int& own) {
  Real best = Real(1e300);
  int bestI = -1;
  if (u.equalR) {
    for (int i = 0; i < u.n; ++i)
      owner_detail::take(sphereDist2(u, i, p, box), i, best, bestI);
    own = bestI;
    return peclet::core::detail::hdSqrt(best) - u.r0;
  }
  for (int i = 0; i < u.n; ++i)
    owner_detail::take(peclet::core::detail::hdSqrt(sphereDist2(u, i, p, box)) - u.r[i], i, best,
                       bestI);
  own = bestI;
  return best;
}

/// Value + owner through the candidate grid. Same superset argument as evalSphereUnionGrid: the
/// list contains the argmin, so the argmin over the list IS the global argmin.
template <class Real>
PECLET_HD Real sphereUnionEvalOwnerGrid(const SphereUnionView<Real>& u, Vec3<Real> p,
                                        const PeriodicBox<Real>& box,
                                        const CandidateGridView<Real>& g, int& own) {
  const long b = g.binOf(p);
  if (b < 0)
    return sphereUnionEvalOwner(u, p, box, own);
  const int lo = g.offsets[b], hi = g.offsets[b + 1];
  if (lo == hi)
    return sphereUnionEvalOwner(u, p, box, own);
  Real best = Real(1e300);
  int bestI = -1;
  sphereUnionOwnerSubset(u, p, box, g.items + lo, hi - lo, best, bestI);
  own = bestI;
  return u.equalR ? peclet::core::detail::hdSqrt(best) - u.r0 : best;
}

/// Value + owner over every instance of a general scene, min-image periodic.
template <class Real>
PECLET_HD Real sceneEvalOwnerPeriodic(const SceneView<Real>& sc, Vec3<Real> p,
                                      const PeriodicBox<Real>& box, int& own,
                                      const Real* boundR = nullptr) {
  Real best = Real(1e9);
  int bestI = -1;
  for (int i = 0; i < sc.instanceCount; ++i)
    owner_detail::take(evalInstancePeriodic(sc, i, p, box, boundR ? boundR[i] : Real(-1)), i, best,
                       bestI);
  own = bestI;
  return best;
}

/// Value + owner through a general-scene candidate grid: always-list ∪ candidate list, with the
/// same empty-list / out-of-coverage full-scan fallback evalSceneGrid uses (an owner that
/// disagreed with the value would attribute a wall velocity to a body that is not the nearest).
template <class Real>
PECLET_HD Real sceneEvalOwnerGrid(const SceneView<Real>& sc, Vec3<Real> p,
                                  const PeriodicBox<Real>& box, const CandidateGridView<Real>& g,
                                  int& own, const Real* boundR = nullptr) {
  Real best = Real(1e9);
  int bestI = -1;
  for (int k = 0; k < g.alwaysCount; ++k) {
    const int i = g.always[k];
    owner_detail::take(evalInstancePeriodic(sc, i, p, box, boundR ? boundR[i] : Real(-1)), i, best,
                       bestI);
  }
  const long b = g.binOf(p);
  const int lo = b < 0 ? 0 : g.offsets[b], hi = b < 0 ? 0 : g.offsets[b + 1];
  if (b < 0 || lo == hi) {
    for (int i = 0; i < sc.instanceCount; ++i)
      owner_detail::take(evalInstancePeriodic(sc, i, p, box, boundR ? boundR[i] : Real(-1)), i,
                         best, bestI);
    own = bestI;
    return best;
  }
  for (int k = lo; k < hi; ++k) {
    const int i = g.items[k];
    owner_detail::take(evalInstancePeriodic(sc, i, p, box, boundR ? boundR[i] : Real(-1)), i, best,
                       bestI);
  }
  own = bestI;
  return best;
}

/// Owner-only wrappers (the value is discarded by the compiler on every backend).
template <class Real>
PECLET_HD int sphereUnionOwner(const SphereUnionView<Real>& u, Vec3<Real> p,
                               const PeriodicBox<Real>& box) {
  int own = -1;
  (void)sphereUnionEvalOwner(u, p, box, own);
  return own;
}
template <class Real>
PECLET_HD int sphereUnionOwnerGrid(const SphereUnionView<Real>& u, Vec3<Real> p,
                                   const PeriodicBox<Real>& box, const CandidateGridView<Real>& g) {
  int own = -1;
  (void)sphereUnionEvalOwnerGrid(u, p, box, g, own);
  return own;
}
template <class Real>
PECLET_HD int sceneOwnerPeriodic(const SceneView<Real>& sc, Vec3<Real> p,
                                 const PeriodicBox<Real>& box, const Real* boundR = nullptr) {
  int own = -1;
  (void)sceneEvalOwnerPeriodic(sc, p, box, own, boundR);
  return own;
}
template <class Real>
PECLET_HD int sceneOwnerGrid(const SceneView<Real>& sc, Vec3<Real> p, const PeriodicBox<Real>& box,
                             const CandidateGridView<Real>& g, const Real* boundR = nullptr) {
  int own = -1;
  (void)sceneEvalOwnerGrid(sc, p, box, g, own, boundR);
  return own;
}

/// The one POD every consumer captures: sphere-union fast path when the scene is a plain sphere
/// union, general scene otherwise, candidate-accelerated in both modes, min-image periodic.
/// Mode selection happens ONCE at build (SceneQueryDevice / SceneQueryHost), so numerics are
/// deterministic per scene; the sphere path is fma-canonical (bitwise host==device), the general
/// tree walk is judged by the sign+ULP rule across backends.
template <class Real>
struct SceneQueryView {
  SphereUnionView<Real> u{};       // u.n > 0  => sphere fast path
  SceneView<Real> scene{};         // otherwise: general scene
  CandidateGridView<Real> grid{};  // offsets == nullptr => no acceleration
  PeriodicBox<Real> box{};
  const Real* instBoundR = nullptr;  // per-instance bounds (general periodic seam handling)

  PECLET_HD Real eval(Vec3<Real> p) const {
    if (u.n > 0)
      return grid.offsets ? evalSphereUnionGrid(u, p, box, grid) : evalSphereUnion(u, p, box);
    return grid.offsets ? evalSceneGrid(scene, p, box, grid, instBoundR)
                        : evalScenePeriodic(scene, p, box, instBoundR);
  }

  /// eval(p) AND the index of the instance (sphere-union mode: the sphere) whose surface is
  /// nearest — the argmin behind eval's min, in ONE traversal. `own` is -1 for an empty scene.
  /// Ties break to the lowest index; see the determinism contract above. The returned value is
  /// bitwise eval's, so a consumer that needs both never has to reconcile two numbers.
  PECLET_HD Real evalOwner(Vec3<Real> p, int& own) const {
    if (u.n > 0)
      return grid.offsets ? sphereUnionEvalOwnerGrid(u, p, box, grid, own)
                          : sphereUnionEvalOwner(u, p, box, own);
    return grid.offsets ? sceneEvalOwnerGrid(scene, p, box, grid, own, instBoundR)
                        : sceneEvalOwnerPeriodic(scene, p, box, own, instBoundR);
  }

  /// Owner only.
  PECLET_HD int owner(Vec3<Real> p) const {
    int own = -1;
    (void)evalOwner(p, own);
    return own;
  }
};

/// Try to view a scene as a PLAIN sphere union: every instance a single kSphere leaf with a
/// bitwise-identity node transform and a translation-only, scale-1, identity-rotation instance
/// transform. Returns the centers/radii (world) or empty when any instance fails the test.
template <class Real>
inline bool extractSphereUnion(const SceneView<Real>& sc, std::vector<Real>& cx,
                               std::vector<Real>& cy, std::vector<Real>& cz, std::vector<Real>& r) {
  cx.clear();
  cy.clear();
  cz.clear();
  r.clear();
  auto identity = [](const Transform<Real>& t, bool allowTranslate) {
    const bool q = t.rotation.x == Real(0) && t.rotation.y == Real(0) && t.rotation.z == Real(0) &&
                   t.rotation.w == Real(1) && t.scale == Real(1);
    const bool tr = allowTranslate || (t.translation.x == Real(0) && t.translation.y == Real(0) &&
                                       t.translation.z == Real(0));
    return q && tr;
  };
  for (int i = 0; i < sc.instanceCount; ++i) {
    const Instance<Real>& inst = sc.instances[i];
    if (inst.shapeRoot < 0 || inst.shapeRoot >= sc.nodeCount)
      return false;
    const ShapeNode<Real>& n = sc.nodes[inst.shapeRoot];
    if (n.kind != kSphere || !identity(n.transform, false) || !identity(inst.transform, true))
      return false;
    cx.push_back(inst.transform.translation.x);
    cy.push_back(inst.transform.translation.y);
    cz.push_back(inst.transform.translation.z);
    r.push_back(n.params[0]);
  }
  return !cx.empty();
}

// ---------------------------------------------------------------------------------------------
// The callable that retires the AMR stopgap
// ---------------------------------------------------------------------------------------------

/// Owning, host-side accelerated union-of-spheres query. operator()(Vec<3>) slots directly into
/// AmrFlow::setSolid's templated SdfFn parameter (and inlines there — it is a template, not a
/// std::function), returning values BIT-IDENTICAL to the retired brute-force callback.
class SphereBedQuery {
 public:
  SphereBedQuery(std::vector<double> cx, std::vector<double> cy, std::vector<double> cz,
                 std::vector<double> r, Vec<3> origin, Vec<3> extent, bool periodic, int nbHint = 0)
      : cx_(std::move(cx)), cy_(std::move(cy)), cz_(std::move(cz)), r_(std::move(r)) {
    const int n = static_cast<int>(cx_.size());
    if (n == 0 || cy_.size() != cx_.size() || cz_.size() != cx_.size() || r_.size() != cx_.size())
      throw std::invalid_argument("SphereBedQuery: centers/radii sizes disagree");
    box_.on = periodic;
    box_.Lx = extent[0];
    box_.Ly = extent[1];
    box_.Lz = extent[2];
    union_.cx = cx_.data();
    union_.cy = cy_.data();
    union_.cz = cz_.data();
    union_.r = r_.data();
    union_.n = n;
    union_.equalR = true;
    for (int i = 1; i < n; ++i)
      if (r_[i] != r_[0])
        union_.equalR = false;
    union_.r0 = r_[0];
    grid_ = buildSphereCandidateGrid(union_, Vec3<double>{origin[0], origin[1], origin[2]},
                                     Vec3<double>{extent[0], extent[1], extent[2]}, box_, nbHint);
    gview_ = grid_.view();
  }

  double operator()(const Vec<3>& p) const {
    return evalSphereUnionGrid(union_, Vec3<double>{p[0], p[1], p[2]}, box_, gview_);
  }

  const SphereUnionView<double>& sphereUnion() const { return union_; }
  const PeriodicBox<double>& box() const { return box_; }
  const CandidateGridView<double>& grid() const { return gview_; }
  /// Mean candidate-list length (diagnostics for the acceptance report).
  double meanListLength() const {
    return grid_.offsets.empty() ? 0.0
                                 : double(grid_.items.size()) / double(grid_.offsets.size() - 1);
  }

 private:
  std::vector<double> cx_, cy_, cz_, r_;
  SphereUnionView<double> union_;
  PeriodicBox<double> box_{};
  CandidateGrid<double> grid_;
  CandidateGridView<double> gview_{};
};

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_SCENE_QUERY_HPP

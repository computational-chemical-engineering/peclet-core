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

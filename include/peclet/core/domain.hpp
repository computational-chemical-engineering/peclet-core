// core — the physical domain: the box a method solves in, and the uniform grid that resolves it.
//
// This is the one place in the suite that says what "the domain" is, so that flow, pnm, the AMR
// octree, voro and the coupling drivers all describe the same world the same way. It implements
// the `Domain` concept of suite/docs/INTERFACES.md §1 (`dim()`, `length()`, `origin()`,
// `periodic(axis)`), and `UniformGrid` adds the Eulerian `resolution()`.
//
// Units are the caller's, and consistent (suite/docs/CONVENTIONS.md §7): the domain carries no unit
// system and no dimension checking. Spacing is derived, never given:
//
//     spacing[a] = length[a] / cells[a]
//
// Per-axis spacings may differ (anisotropic cells); the consumers that only support isotropic cells
// today assert `isIsotropic()` themselves rather than this header forbidding it.
//
// Conventions honoured here: axis order is x-fastest, cell (i,j,k) spans
// [origin + i*h, origin + (i+1)*h) per axis, so its centre is at origin + (i+0.5)*h. A face centre
// on axis `a` at index i is at origin[a] + i*h[a] in that axis and cell-centred in the others —
// the staggered MAC convention flow uses.
//
// Header-only, C++17-clean (so it can be pulled into device translation units).
#ifndef PECLET_CORE_DOMAIN_HPP
#define PECLET_CORE_DOMAIN_HPP

#include <cmath>
#include <cstddef>

#include "peclet/core/common/types.hpp"

namespace peclet::core {

/// A physical, axis-aligned box with per-axis periodicity. Continuous — no cells.
template <int Dim = 3>
struct Box {
  Vec<Dim> origin{};                    ///< lower corner
  Vec<Dim> length{};                    ///< extent per axis (> 0)
  std::array<bool, Dim> periodic{};     ///< per-axis periodicity

  static constexpr int dim() { return Dim; }

  /// Upper corner: origin + length.
  Vec<Dim> upper() const {
    Vec<Dim> u{};
    for (int a = 0; a < Dim; ++a)
      u[a] = origin[a] + length[a];
    return u;
  }

  Vec<Dim> centre() const {
    Vec<Dim> c{};
    for (int a = 0; a < Dim; ++a)
      c[a] = origin[a] + 0.5 * length[a];
    return c;
  }

  Real volume() const {
    Real v = 1.0;
    for (int a = 0; a < Dim; ++a)
      v *= length[a];
    return v;
  }

  /// Is the point inside [origin, origin+length) on every axis?
  bool contains(const Vec<Dim>& p) const {
    for (int a = 0; a < Dim; ++a) {
      if (!(p[a] >= origin[a] && p[a] < origin[a] + length[a]))
        return false;
    }
    return true;
  }

  /// Fold a point into the box along the periodic axes; non-periodic axes are left untouched.
  Vec<Dim> wrap(const Vec<Dim>& p) const {
    Vec<Dim> q = p;
    for (int a = 0; a < Dim; ++a) {
      if (!periodic[a] || !(length[a] > 0.0))
        continue;
      Real t = (q[a] - origin[a]) / length[a];
      t -= std::floor(t);
      // std::floor can return t == 1 for tiny negative inputs after rounding; clamp to [0,1).
      if (!(t < 1.0))
        t = 0.0;
      q[a] = origin[a] + t * length[a];
    }
    return q;
  }

  /// Minimum-image separation b - a, shortened across the periodic axes.
  Vec<Dim> minImage(const Vec<Dim>& a_, const Vec<Dim>& b) const {
    Vec<Dim> d{};
    for (int a = 0; a < Dim; ++a) {
      d[a] = b[a] - a_[a];
      if (!periodic[a] || !(length[a] > 0.0))
        continue;
      d[a] -= length[a] * std::round(d[a] / length[a]);
    }
    return d;
  }
};

/// A box resolved by a regular cell grid. Satisfies `Domain` plus `resolution()`.
///
/// Constructed from cells + a box; `spacing()` is derived. `UniformGrid::cellUnits(cells)` builds
/// the historical "cell units" grid (extent == cells, origin 0, spacing 1) that peclet solvers used
/// before physical domains existed.
template <int Dim = 3>
struct UniformGrid {
  Box<Dim> box{};
  IVec<Dim> cells{};

  UniformGrid() = default;
  UniformGrid(const IVec<Dim>& c, const Box<Dim>& b) : box(b), cells(c) {}

  /// The historical cell-unit grid: extent == cells, origin 0, spacing exactly 1.
  static UniformGrid cellUnits(const IVec<Dim>& c, const std::array<bool, Dim>& per = {}) {
    Box<Dim> b{};
    for (int a = 0; a < Dim; ++a) {
      b.origin[a] = 0.0;
      b.length[a] = static_cast<Real>(c[a]);
    }
    b.periodic = per;
    return UniformGrid(c, b);
  }

  // --- Domain concept ---
  static constexpr int dim() { return Dim; }
  const Vec<Dim>& length() const { return box.length; }
  const Vec<Dim>& origin() const { return box.origin; }
  bool periodic(int axis) const { return box.periodic[axis]; }
  const IVec<Dim>& resolution() const { return cells; }

  /// Cell size per axis: length / cells. Zero cells on an axis yields 0 (degenerate; callers check).
  Vec<Dim> spacing() const {
    Vec<Dim> h{};
    for (int a = 0; a < Dim; ++a)
      h[a] = (cells[a] > 0) ? box.length[a] / static_cast<Real>(cells[a]) : 0.0;
    return h;
  }

  /// The smallest cell size — the reference length the solvers non-dimensionalise with.
  Real minSpacing() const {
    const Vec<Dim> h = spacing();
    Real m = h[0];
    for (int a = 1; a < Dim; ++a)
      m = (h[a] < m) ? h[a] : m;
    return m;
  }

  /// True when every axis has the same cell size to within `tol` relative.
  bool isIsotropic(Real tol = 1e-12) const {
    const Vec<Dim> h = spacing();
    for (int a = 1; a < Dim; ++a) {
      const Real d = std::fabs(h[a] - h[0]);
      const Real s = std::fabs(h[0]) > 0.0 ? std::fabs(h[0]) : 1.0;
      if (d > tol * s)
        return false;
    }
    return true;
  }

  Index numCells() const {
    Index n = 1;
    for (int a = 0; a < Dim; ++a)
      n *= cells[a];
    return n;
  }

  Real cellVolume() const {
    const Vec<Dim> h = spacing();
    Real v = 1.0;
    for (int a = 0; a < Dim; ++a)
      v *= h[a];
    return v;
  }

  /// Physical centre of cell `idx`: origin + (idx + 1/2) * h.
  Vec<Dim> cellCentre(const IVec<Dim>& idx) const {
    const Vec<Dim> h = spacing();
    Vec<Dim> p{};
    for (int a = 0; a < Dim; ++a)
      p[a] = box.origin[a] + (static_cast<Real>(idx[a]) + 0.5) * h[a];
    return p;
  }

  /// Physical centre of the `axis`-normal face at index `idx` (MAC staggering): the face coordinate
  /// is on the node in `axis` and cell-centred in the other axes.
  Vec<Dim> faceCentre(int axis, const IVec<Dim>& idx) const {
    const Vec<Dim> h = spacing();
    Vec<Dim> p{};
    for (int a = 0; a < Dim; ++a) {
      const Real off = (a == axis) ? 0.0 : 0.5;
      p[a] = box.origin[a] + (static_cast<Real>(idx[a]) + off) * h[a];
    }
    return p;
  }

  /// Physical position of grid node `idx` (cell corner): origin + idx * h.
  Vec<Dim> nodePosition(const IVec<Dim>& idx) const {
    const Vec<Dim> h = spacing();
    Vec<Dim> p{};
    for (int a = 0; a < Dim; ++a)
      p[a] = box.origin[a] + static_cast<Real>(idx[a]) * h[a];
    return p;
  }

  /// Continuous index coordinate of a physical point: xi = (p - origin) / h. The inverse of
  /// `physicalOf`; cell centre (i+1/2) maps back to itself.
  Vec<Dim> indexOf(const Vec<Dim>& p) const {
    const Vec<Dim> h = spacing();
    Vec<Dim> xi{};
    for (int a = 0; a < Dim; ++a)
      xi[a] = (h[a] > 0.0) ? (p[a] - box.origin[a]) / h[a] : 0.0;
    return xi;
  }

  /// Physical point of a continuous index coordinate: p = origin + xi * h. Inverse of `indexOf`.
  Vec<Dim> physicalOf(const Vec<Dim>& xi) const {
    const Vec<Dim> h = spacing();
    Vec<Dim> p{};
    for (int a = 0; a < Dim; ++a)
      p[a] = box.origin[a] + xi[a] * h[a];
    return p;
  }

  /// Index of the cell containing `p` (floor of the continuous index). No bounds check.
  IVec<Dim> cellOf(const Vec<Dim>& p) const {
    const Vec<Dim> xi = indexOf(p);
    IVec<Dim> idx{};
    for (int a = 0; a < Dim; ++a)
      idx[a] = static_cast<Index>(std::floor(xi[a]));
    return idx;
  }

  bool contains(const Vec<Dim>& p) const { return box.contains(p); }
  Vec<Dim> wrap(const Vec<Dim>& p) const { return box.wrap(p); }
  Vec<Dim> minImage(const Vec<Dim>& a, const Vec<Dim>& b) const { return box.minImage(a, b); }
};

}  // namespace peclet::core

#endif  // PECLET_CORE_DOMAIN_HPP

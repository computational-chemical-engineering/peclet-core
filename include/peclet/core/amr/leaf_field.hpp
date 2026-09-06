// core — leaf-indexed fields and world geometry for a BlockOctree.
//
// A BlockOctree addresses leaves in integer *fine units* relative to the block
// origin. AmrGeometry maps those to world coordinates (origin + h0 * fine), and
// LeafField<T> is a value-per-leaf array in the octree's leaf (Z-order) slot
// order — the simulation field the device kernels and the VTU writer operate on.
// The on-device counterpart is simply a peclet::core::View<T> of length numLeaves(); this
// host container is the I/O / setup form.
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON (so it tracks block_octree.hpp).
#ifndef PECLET_CORE_AMR_LEAF_FIELD_HPP
#define PECLET_CORE_AMR_LEAF_FIELD_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>
#include <cstddef>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::amr {

namespace detail {
/// `Vec<Dim>` with every component set to `v` (C++ has no aggregate-fill initializer).
template <int Dim>
inline Vec<Dim> filledVec(Real v) {
  Vec<Dim> x{};
  for (int d = 0; d < Dim; ++d)
    x[d] = v;
  return x;
}
}  // namespace detail

/// World-space placement of a block-local octree: fine coordinate (0,..,0) sits at
/// `origin`, and one level-0 fine cell is `h0[d]` wide on axis `d`.
///
/// **`h0` is PER AXIS** (Phase 3 of `suite/docs/PHYSICAL_UNITS_PLAN.md`; the design is
/// `docs/amr_anisotropic.md`). The octree refines by 2 on every axis, so a level-`l` leaf is
/// `h0[d] * 2^l` wide on axis `d` and **every level inherits the root aspect ratio** — the Morton
/// codes, `bounds()`, `level()`, the halo, the ORB and the 2:1 balance are integer fine-unit
/// arithmetic and know nothing about it.
///
/// `leafSize()` and `lowerCorner()` are the two funnels every world-coordinate computation goes
/// through; making `h0` a `Vec` breaks every scalar caller at COMPILE time, which is the inventory
/// mechanism the port relies on. Cubic geometry is `AmrGeometry::isotropicAt(origin, h)`.
template <int Dim>
struct AmrGeometry {
  Vec<Dim> origin{};
  Vec<Dim> h0 = detail::filledVec<Dim>(1.0);

  /// Cubic-cell convenience: the pre-Phase-3 `{origin, h}` construction.
  static AmrGeometry isotropicAt(const Vec<Dim>& o, Real h) {
    AmrGeometry g;
    g.origin = o;
    g.h0 = detail::filledVec<Dim>(h);
    return g;
  }
  /// Set all axes to one spacing (the cubic case), in place.
  void setIsotropic(Real h) { h0 = detail::filledVec<Dim>(h); }

  Real hMin() const {
    Real m = h0[0];
    for (int d = 1; d < Dim; ++d)
      m = h0[d] < m ? h0[d] : m;
    return m;
  }
  Real hMax() const {
    Real m = h0[0];
    for (int d = 1; d < Dim; ++d)
      m = h0[d] > m ? h0[d] : m;
    return m;
  }
  bool isotropic() const {
    for (int d = 1; d < Dim; ++d)
      if (h0[d] != h0[0])
        return false;
    return true;
  }
  /// World volume of a level-0 fine cell.
  Real cellVolume() const {
    Real v = h0[0];
    for (int d = 1; d < Dim; ++d)
      v *= h0[d];
    return v;
  }

  /// World width of a leaf at `level` on axis `d` (it covers 2^level fine cells).
  Real leafSize(unsigned level, int d) const {
    return h0[d] * static_cast<Real>(Index(1) << level);
  }
  /// World widths of a leaf at `level`, per axis.
  Vec<Dim> leafSize(unsigned level) const {
    Vec<Dim> s{};
    const Real f = static_cast<Real>(Index(1) << level);
    for (int d = 0; d < Dim; ++d)
      s[d] = h0[d] * f;
    return s;
  }

  /// World coordinate of a leaf's lower corner, given its integer lower bound (fine units).
  template <class Coord>
  Vec<Dim> lowerCorner(const std::array<Coord, Dim>& lo) const {
    Vec<Dim> p{};
    for (int d = 0; d < Dim; ++d)
      p[d] = origin[d] + static_cast<Real>(lo[d]) * h0[d];
    return p;
  }

  /// World coordinate of a leaf centre, from its integer bounds [lo,hi] (inclusive, fine units).
  template <class Coord>
  Vec<Dim> center(const std::array<std::array<Coord, Dim>, 2>& b) const {
    Vec<Dim> p{};
    for (int d = 0; d < Dim; ++d)
      p[d] = origin[d] + (static_cast<Real>(b[0][d]) +
                          0.5 * (static_cast<Real>(b[1][d] - b[0][d]) + 1.0)) *
                             h0[d];
    return p;
  }
};

/// A value per leaf, in the octree's Z-order leaf slot order.
template <class T>
struct LeafField {
  std::vector<T> values;

  LeafField() = default;
  explicit LeafField(Index n, T init = T{}) : values(static_cast<std::size_t>(n), init) {}

  template <int Dim, unsigned Bits>
  explicit LeafField(const BlockOctree<Dim, Bits>& t, T init = T{})
      : values(static_cast<std::size_t>(t.numLeaves()), init) {}

  Index size() const { return static_cast<Index>(values.size()); }
  T& operator[](Index i) { return values[static_cast<std::size_t>(i)]; }
  const T& operator[](Index i) const { return values[static_cast<std::size_t>(i)]; }
  T* data() { return values.data(); }
  const T* data() const { return values.data(); }
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_LEAF_FIELD_HPP

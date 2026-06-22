// transport-core — leaf-indexed fields and world geometry for a BlockOctree.
//
// A BlockOctree addresses leaves in integer *fine units* relative to the block
// origin. AmrGeometry maps those to world coordinates (origin + h0 * fine), and
// LeafField<T> is a value-per-leaf array in the octree's leaf (Z-order) slot
// order — the simulation field the device kernels and the VTU writer operate on.
// The on-device counterpart is simply a tpx::View<T> of length numLeaves(); this
// host container is the I/O / setup form.
//
// Header-only, guarded by TPX_HAVE_MORTON (so it tracks block_octree.hpp).
#ifndef TPX_AMR_LEAF_FIELD_HPP
#define TPX_AMR_LEAF_FIELD_HPP

#ifdef TPX_HAVE_MORTON

#include <array>
#include <cstddef>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/common/types.hpp"

namespace tpx::amr {

/// World-space placement of a block-local octree: fine coordinate (0,..,0) sits at
/// `origin`, and one level-0 fine cell is `h0` wide in every axis.
template <int Dim>
struct AmrGeometry {
  Vec<Dim> origin{};
  Real h0 = 1.0;

  /// World width of a leaf at `level` (covers 2^level fine cells).
  Real leafSize(unsigned level) const { return h0 * static_cast<Real>(Index(1) << level); }

  /// World coordinate of a leaf's lower corner, given its integer lower bound (fine units).
  template <class Coord>
  Vec<Dim> lowerCorner(const std::array<Coord, Dim>& lo) const {
    Vec<Dim> p{};
    for (int d = 0; d < Dim; ++d) p[d] = origin[d] + static_cast<Real>(lo[d]) * h0;
    return p;
  }

  /// World coordinate of a leaf centre, from its integer bounds [lo,hi] (inclusive, fine units).
  template <class Coord>
  Vec<Dim> center(const std::array<std::array<Coord, Dim>, 2>& b) const {
    Vec<Dim> p{};
    for (int d = 0; d < Dim; ++d)
      p[d] = origin[d] + (static_cast<Real>(b[0][d]) + 0.5 * (static_cast<Real>(b[1][d] - b[0][d]) + 1.0)) * h0;
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

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_LEAF_FIELD_HPP

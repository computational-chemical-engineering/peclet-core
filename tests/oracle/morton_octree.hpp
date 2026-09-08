// Test-local reference oracle: a std::map linear octree on the morton library.
//
// Formerly morton/octree (`morton_octree::Octree`, the "early scaffold" that was
// going to become its own project). Removed from morton on 2026-09-08 — the
// suite's AMR uses the morton library (`morton/morton.hpp`) directly and its own
// peclet::core::amr::BlockOctree; the scaffold's only consumer was
// core/tests/test_block_octree.cpp, which uses it as an independent oracle for
// the leaf set. The last morton tree that carried it is commit b502598
// (also reachable at the morton tag `pre-legacy-removal`, one comment line older).
//
// The namespace `morton_octree` is kept so the test body is unchanged. This
// header is NOT part of peclet_core's include path or its public API.
//
// Original header comment follows.
//
// A linear octree/quadtree built on the peclet-morton library. Leaves are
// stored in a std::map keyed by their Morton origin, so the map is already in
// Z-order. Point location, neighbour finding and refinement are expressed with
// the core's O(1) arithmetic and hierarchy helpers rather than decode/encode.
//
// It depends only on the standalone morton library (morton::Morton).
//
// Conventions
// -----------
// A cell at `level` L covers a 2^L block per axis; its key is the cell origin,
// which has its low L*Dim Morton bits zero. The stored leaves are expected to
// tile the domain without overlap (a valid octree partition); point location
// relies on that.
//
// SPDX-License-Identifier: MIT

#ifndef PECLET_CORE_TESTS_ORACLE_MORTON_OCTREE_HPP
#define PECLET_CORE_TESTS_ORACLE_MORTON_OCTREE_HPP

#include <array>
#include <cstdint>
#include <map>
#include <vector>

#include "morton/morton.hpp"

namespace morton_octree {

using morton::Morton;

template <unsigned Dim, unsigned Bits, typename T>
class Octree {
 public:
  using key_type = Morton<Dim, Bits>;
  using coord_type = typename key_type::coord_type;
  static constexpr unsigned octants = key_type::octants;

  struct Cell {
    unsigned level;  // cell covers a 2^level block per axis
    T value;
  };

  using map_type = std::map<key_type, Cell>;
  using iterator = typename map_type::iterator;
  using const_iterator = typename map_type::const_iterator;

  // ---- capacity / iteration ---------------------------------------------

  std::size_t size() const { return leaves_.size(); }
  bool empty() const { return leaves_.empty(); }
  void clear() { leaves_.clear(); }
  const_iterator begin() const { return leaves_.begin(); }
  const_iterator end() const { return leaves_.end(); }
  iterator begin() { return leaves_.begin(); }
  iterator end() { return leaves_.end(); }

  // ---- insertion ---------------------------------------------------------

  /// Insert (or overwrite) a leaf identified by an origin Morton code and a
  /// level. The origin is normalised to the cell boundary.
  iterator insert(key_type origin, unsigned level, T value) {
    key_type key = origin.ancestor(level);
    auto r = leaves_.insert_or_assign(key, Cell{level, value});
    return r.first;
  }

  /// Insert a leaf from integer coordinates (any point inside the cell).
  iterator insert(const std::array<coord_type, Dim>& coords, unsigned level, T value) {
    return insert(key_type::encode(coords), level, value);
  }

  // ---- point location ----------------------------------------------------

  /// Leaf containing the point `p`, or end() if no leaf covers it.
  const_iterator find(key_type p) const {
    auto it = leaves_.upper_bound(p);
    if (it == leaves_.begin())
      return leaves_.end();
    --it;
    // p is in this leaf iff clearing the leaf's low bits gives its origin.
    if (p.ancestor(it->second.level) == it->first)
      return it;
    return leaves_.end();
  }

  const_iterator find(const std::array<coord_type, Dim>& coords) const {
    return find(key_type::encode(coords));
  }

  // ---- neighbour finding -------------------------------------------------

  /// The leaf adjacent to `leaf` across its face in direction `dir` (±1) on
  /// `axis`, or end() if outside the stored domain. Uses arithmetic to step
  /// one cell-width to a point just across the face, then locates it.
  const_iterator face_neighbor(const_iterator leaf, unsigned axis, int dir) const {
    if (leaf == leaves_.end())
      return leaves_.end();
    const coord_type step = coord_type(coord_type(1) << leaf->second.level);
    key_type probe = leaf->first;
    if (dir >= 0) {
      // step to the far side of this cell
      if (!probe.try_add(axis, step))
        return leaves_.end();  // domain edge
    } else {
      if (!probe.try_sub(axis, 1))
        return leaves_.end();  // one past lower face
    }
    return find(probe);
  }

  // ---- geometry ----------------------------------------------------------

  /// Inclusive integer bounds [lo, hi] of a leaf (hi = lo + 2^level - 1).
  std::array<std::array<coord_type, Dim>, 2> bounds(const_iterator leaf) const {
    std::array<std::array<coord_type, Dim>, 2> b{};
    auto origin = leaf->first.decode();
    coord_type extent = coord_type((coord_type(1) << leaf->second.level) - 1);
    for (unsigned d = 0; d < Dim; ++d) {
      b[0][d] = origin[d];
      b[1][d] = coord_type(origin[d] + extent);
    }
    return b;
  }

  // ---- refinement --------------------------------------------------------

  /// Replace a leaf with its 2^Dim children one level finer. `make_child`
  /// maps (childOrigin, octantIndex) -> child value.
  template <typename MakeChild>
  void refine(const_iterator leaf, MakeChild&& make_child) {
    if (leaf == leaves_.end() || leaf->second.level == 0)
      return;
    key_type parent = leaf->first;
    unsigned plevel = leaf->second.level;
    leaves_.erase(leaf->first);
    for (unsigned oct = 0; oct < octants; ++oct) {
      key_type child = parent.child(plevel, oct);
      leaves_.insert_or_assign(child, Cell{plevel - 1, make_child(child, oct)});
    }
  }

 private:
  map_type leaves_;
};

}  // namespace morton_octree

#endif  // PECLET_CORE_TESTS_ORACLE_MORTON_OCTREE_HPP

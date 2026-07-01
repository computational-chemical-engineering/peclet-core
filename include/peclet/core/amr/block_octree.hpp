// core — per-block adaptive octree with block-local Morton codes.
//
// The suite's AMR primitive. Unlike the textbook linear octree (one global
// space-filling curve, partitioned by index ranges à la p4est/Dendro), the suite
// keeps the ORB *block* decomposition (peclet::core::decomp::BlockDecomposer) and gives
// *each block its own local Morton coordinate system*. BlockOctree is that
// per-block octree: leaves are addressed by `morton::Morton<Dim,Bits>` codes
// relative to the block origin, so codes stay narrow and the block is
// self-contained. The distributed glue (cross-block 2:1 balance + an owner-based
// ghost halo) is layered on top later (DistributedOctree / AmrHalo); a uniform,
// unrefined BlockOctree is bit-identical to the existing structured block grid.
//
// Conventions (see also morton/octree and ../../../docs)
// -----------------------------------------------------------------------------
//  * Coordinates are in *fine units*: 1 fine unit = a leaf at level 0. A leaf at
//    `level` L covers a 2^L block per axis; its origin Morton code has the low
//    L*Dim bits zero (morton's level convention).
//  * Root cells sit at `level = lmax`; refinement decreases level toward 0
//    (finest). A block is a "root brick" of brick[i] root cells per axis, so it
//    spans brick[i]*2^lmax fine units; require 2^Bits >= brick[i]*2^lmax.
//  * Leaves tile the block without overlap and are stored sorted by code, i.e. in
//    Z-order — the same order morton's curve and the device leaf arrays use.
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON (set by CMake when the morton sibling
// checkout is present). The query helpers carry morton's MORTON_HD, so they are
// device-callable exactly when morton is (KOKKOS_FUNCTION under a Kokkos build);
// topology mutation (refine/coarsen/balance) is host-side and rebuilds the leaf
// arrays the device queries read — mirroring grid_halo.hpp (host) vs
// grid_halo_kokkos.hpp (device).
#ifndef PECLET_CORE_AMR_BLOCK_OCTREE_HPP
#define PECLET_CORE_AMR_BLOCK_OCTREE_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "morton/morton.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::amr {

/// Locate the leaf containing Morton code `p` in sorted leaf arrays.
///
/// `codes` (ascending, length `n`) are leaf origin codes; `levels[i]` the level
/// of leaf i. Returns the leaf index whose cell covers `p`, or -1 if none does.
/// Device-callable: a hand-rolled binary search (no std::upper_bound) so it
/// compiles in a Kokkos device pass. O(log n).
template <class M>
MORTON_HD inline Index amrLocate(const typename M::code_type* codes, const std::uint8_t* levels,
                                 Index n, typename M::code_type p) {
  // First index with codes[idx] > p, then step back one.
  Index lo = 0, hi = n;
  while (lo < hi) {
    Index mid = lo + ((hi - lo) >> 1);
    if (codes[mid] <= p)
      lo = mid + 1;
    else
      hi = mid;
  }
  Index idx = lo - 1;
  if (idx < 0) return -1;
  // p is inside leaf idx iff clearing the leaf's low bits gives its origin.
  if (M::from_code(p).ancestor(levels[idx]).code() == codes[idx]) return idx;
  return -1;
}

/// Per-block adaptive octree over block-local Morton codes.
///
/// `Bits` is the per-axis code width; the default keeps codes in a built-in
/// 64-bit integer (3D: 21 bits/axis, 2D: 32). A block must satisfy
/// brick[i]*2^lmax <= 2^Bits.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class BlockOctree {
 public:
  using M = morton::Morton<Dim, Bits>;
  using Code = typename M::code_type;
  using Coord = typename M::coord_type;
  static constexpr unsigned octants = M::octants;  // children per node (2^Dim)

  BlockOctree() = default;

  /// Build a uniform block: a `brick` (in root cells) of leaves at level `lmax`.
  /// `globalOrigin` is the block's lower corner in the *global coarse grid's*
  /// root-cell coordinates (metadata used by the distributed layer; the local
  /// codes here are origin-relative and independent of it).
  BlockOctree(IVec<Dim> brick, unsigned lmax, IVec<Dim> globalOrigin = IVec<Dim>{}) {
    init(brick, lmax, globalOrigin);
  }

  void init(IVec<Dim> brick, unsigned lmax, IVec<Dim> globalOrigin = IVec<Dim>{}) {
    brick_ = brick;
    lmax_ = lmax;
    globalOrigin_ = globalOrigin;
    const Coord rootSpan = Coord(Coord(1) << lmax_);  // fine units per root cell
    codes_.clear();
    levels_.clear();
    IVec<Dim> bgn{};
    IVec<Dim> end = brick_;
    forEachInBox<Dim>(bgn, end, [&](const IVec<Dim>& r) {
      std::array<Coord, Dim> o{};
      for (int i = 0; i < Dim; ++i) o[i] = static_cast<Coord>(r[i]) * rootSpan;
      codes_.push_back(M::encode(o).code());
      levels_.push_back(static_cast<std::uint8_t>(lmax_));
    });
    sortByCode();
  }

  /// Replace the entire leaf set directly (used by load-balancing migration, which
  /// rebuilds a block from leaves received from other ranks). `codes` are
  /// block-local origin codes and `levels` their parallel levels; they need not be
  /// sorted — assign() restores the Z-order invariant. `brick`/`lmax`/`globalOrigin`
  /// describe the (possibly new) block geometry the codes are relative to.
  void assign(IVec<Dim> brick, unsigned lmax, IVec<Dim> globalOrigin, std::vector<Code> codes,
              std::vector<std::uint8_t> levels) {
    brick_ = brick;
    lmax_ = lmax;
    globalOrigin_ = globalOrigin;
    codes_ = std::move(codes);
    levels_ = std::move(levels);
    sortByCode();
  }

  // ---- introspection -----------------------------------------------------
  Index numLeaves() const { return static_cast<Index>(codes_.size()); }
  unsigned lmax() const { return lmax_; }
  static constexpr unsigned bits() { return Bits; }
  const IVec<Dim>& brick() const { return brick_; }
  const IVec<Dim>& globalOrigin() const { return globalOrigin_; }
  Code code(Index i) const { return codes_[static_cast<std::size_t>(i)]; }
  unsigned level(Index i) const { return levels_[static_cast<std::size_t>(i)]; }
  const std::vector<Code>& codes() const { return codes_; }
  const std::vector<std::uint8_t>& levels() const { return levels_; }

  // ---- queries -----------------------------------------------------------

  /// Leaf containing Morton code `p`, or -1. Host wrapper over amrLocate.
  Index find(Code p) const {
    return amrLocate<M>(codes_.data(), levels_.data(), numLeaves(), p);
  }
  /// Leaf containing fine-unit coordinates, or -1.
  Index find(const std::array<Coord, Dim>& fine) const { return find(M::encode(fine).code()); }

  /// Inclusive integer bounds [lo, hi] of leaf `i` in fine units.
  std::array<std::array<Coord, Dim>, 2> bounds(Index i) const {
    std::array<std::array<Coord, Dim>, 2> b{};
    auto o = M::from_code(codes_[static_cast<std::size_t>(i)]).decode();
    Coord extent = Coord((Coord(1) << levels_[static_cast<std::size_t>(i)]) - 1);
    for (int d = 0; d < Dim; ++d) {
      b[0][d] = o[d];
      b[1][d] = Coord(o[d] + extent);
    }
    return b;
  }

  /// The leaf across leaf `i`'s face on `axis` in direction `dir` (±1), or -1 if
  /// it lies outside the block. With 2:1 balance this lands in a same-, one-
  /// coarser-, or one-finer-level neighbour (the corner-most, for the finer case).
  Index faceNeighbor(Index i, int axis, int dir) const {
    M probe = M::from_code(codes_[static_cast<std::size_t>(i)]);
    const Coord step = Coord(Coord(1) << levels_[static_cast<std::size_t>(i)]);
    if (dir >= 0) {
      if (!probe.try_add(static_cast<unsigned>(axis), step)) return -1;  // block edge
    } else {
      if (!probe.try_sub(static_cast<unsigned>(axis), 1)) return -1;  // one past lower face
    }
    return find(probe.code());
  }

  // ---- refinement / coarsening ------------------------------------------

  /// Split every leaf for which `pred(code, level)` is true (and level > 0) into
  /// its 2^Dim children one level finer. Returns the number of leaves split.
  /// A single forward pass: children of a parent occupy the parent's code range
  /// in ascending octant == ascending code order, so the sorted invariant holds
  /// without a re-sort.
  template <class Pred>
  Index refineIf(Pred&& pred) {
    std::vector<Code> nc;
    std::vector<std::uint8_t> nl;
    nc.reserve(codes_.size());
    nl.reserve(levels_.size());
    Index nsplit = 0;
    for (std::size_t k = 0; k < codes_.size(); ++k) {
      const unsigned L = levels_[k];
      const Code c = codes_[k];
      if (L > 0 && pred(c, L)) {
        M parent = M::from_code(c);
        for (unsigned oct = 0; oct < octants; ++oct) {
          nc.push_back(parent.child(L, oct).code());
          nl.push_back(static_cast<std::uint8_t>(L - 1));
        }
        ++nsplit;
      } else {
        nc.push_back(c);
        nl.push_back(static_cast<std::uint8_t>(L));
      }
    }
    codes_.swap(nc);
    levels_.swap(nl);
    return nsplit;
  }

  /// Refine a single leaf by index. Returns true if it was split.
  bool refineLeaf(Index i) {
    const Code target = codes_[static_cast<std::size_t>(i)];
    return refineIf([&](Code c, unsigned) { return c == target; }) > 0;
  }

  /// Merge complete sibling groups (all 2^Dim children present, all at the same
  /// level) whose parent satisfies `pred(parentCode, parentLevel)`. Returns the
  /// number of groups merged. Siblings are consecutive in the sorted array.
  template <class Pred>
  Index coarsenIf(Pred&& pred) {
    std::vector<Code> nc;
    std::vector<std::uint8_t> nl;
    nc.reserve(codes_.size());
    nl.reserve(levels_.size());
    Index nmerged = 0;
    std::size_t k = 0;
    while (k < codes_.size()) {
      const unsigned L = levels_[k];
      bool merged = false;
      if (L < lmax_ && k + octants <= codes_.size()) {
        const Code parent = M::from_code(codes_[k]).ancestor(L + 1).code();
        bool full = (codes_[k] == parent);  // octant-0 child shares the parent origin
        for (unsigned oct = 0; full && oct < octants; ++oct) {
          full = levels_[k + oct] == L &&
                 M::from_code(codes_[k + oct]).ancestor(L + 1).code() == parent &&
                 M::from_code(parent).child(L + 1, oct).code() == codes_[k + oct];
        }
        if (full && pred(parent, L + 1)) {
          nc.push_back(parent);
          nl.push_back(static_cast<std::uint8_t>(L + 1));
          k += octants;
          ++nmerged;
          merged = true;
        }
      }
      if (!merged) {
        nc.push_back(codes_[k]);
        nl.push_back(static_cast<std::uint8_t>(L));
        ++k;
      }
    }
    codes_.swap(nc);
    levels_.swap(nl);
    return nmerged;
  }

  /// Enforce 2:1 (graded) balance within this block: no two face-adjacent leaves
  /// differ by more than one level. Iterates to a fixpoint; returns the total
  /// number of refinements performed. Detection is from the *fine* side — a leaf
  /// at level Lf probes one cell across each face; that point always lands inside
  /// a coarser neighbour, so any neighbour with level >= Lf+2 is found and split.
  Index balance2to1() {
    Index total = 0;
    for (;;) {
      std::vector<Code> toRefine;  // origins of leaves to split this round
      for (std::size_t k = 0; k < codes_.size(); ++k) {
        const unsigned Lf = levels_[k];
        for (int axis = 0; axis < Dim; ++axis) {
          for (int dir = -1; dir <= 1; dir += 2) {
            Index nb = faceNeighbor(static_cast<Index>(k), axis, dir);
            if (nb >= 0 && levels_[static_cast<std::size_t>(nb)] >= Lf + 2)
              toRefine.push_back(codes_[static_cast<std::size_t>(nb)]);
          }
        }
      }
      if (toRefine.empty()) break;
      std::sort(toRefine.begin(), toRefine.end());
      toRefine.erase(std::unique(toRefine.begin(), toRefine.end()), toRefine.end());
      total += refineIf([&](Code c, unsigned) {
        return std::binary_search(toRefine.begin(), toRefine.end(), c);
      });
    }
    return total;
  }

  /// True iff the leaves form a valid 2:1-balanced partition: every face
  /// neighbour differs by at most one level. (Test/debug helper.)
  bool isBalanced() const {
    for (std::size_t k = 0; k < codes_.size(); ++k) {
      const unsigned Lf = levels_[k];
      for (int axis = 0; axis < Dim; ++axis)
        for (int dir = -1; dir <= 1; dir += 2) {
          Index nb = faceNeighbor(static_cast<Index>(k), axis, dir);
          if (nb >= 0) {
            unsigned Ln = levels_[static_cast<std::size_t>(nb)];
            unsigned hi = Lf > Ln ? Lf : Ln, lo = Lf > Ln ? Ln : Lf;
            if (hi - lo > 1) return false;
          }
        }
    }
    return true;
  }

 private:
  void sortByCode() {
    std::vector<Index> ord(codes_.size());
    for (std::size_t i = 0; i < ord.size(); ++i) ord[i] = static_cast<Index>(i);
    std::sort(ord.begin(), ord.end(),
              [&](Index a, Index b) { return codes_[a] < codes_[b]; });
    std::vector<Code> nc(codes_.size());
    std::vector<std::uint8_t> nl(levels_.size());
    for (std::size_t i = 0; i < ord.size(); ++i) {
      nc[i] = codes_[static_cast<std::size_t>(ord[i])];
      nl[i] = levels_[static_cast<std::size_t>(ord[i])];
    }
    codes_.swap(nc);
    levels_.swap(nl);
  }

  IVec<Dim> brick_{};
  IVec<Dim> globalOrigin_{};
  unsigned lmax_ = 0;
  std::vector<Code> codes_;        // leaf origin codes, ascending (Z-order)
  std::vector<std::uint8_t> levels_;  // parallel to codes_
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_BLOCK_OCTREE_HPP

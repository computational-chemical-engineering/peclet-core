// core — Morton / Z-order cell indexing (via the suite `morton` primitive).
//
// The complement to BlockIndexer's x-fastest linear order: MortonIndexer maps a
// block's cells onto a Z-order (Morton) space-filling curve, so a cell's spatial
// neighbourhood is near it in memory. This is the "morton/Z-order cell indexing
// (via morton)" piece of the decomposition module described in
// ../../../docs/ARCHITECTURE.md. The x-fastest convention in peclet/core/common/types.hpp
// is unchanged — this is an *alternative* ordering for cache-friendly traversal /
// spatial sorting, not a replacement.
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON (set by CMake when the morton sibling
// checkout is present). Methods carry morton's MORTON_HD, so they are device-
// callable exactly when morton is (KOKKOS_FUNCTION under a Kokkos build with
// MORTON_ENABLE_KOKKOS, __host__ __device__ under nvcc/hipcc, nothing on host).
#ifndef PECLET_CORE_DECOMP_MORTON_INDEXER_HPP
#define PECLET_CORE_DECOMP_MORTON_INDEXER_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include "morton/morton.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::decomp {

/// Z-order (Morton) indexer over an axis-origin-relative cell block.
///
/// `Bits` is the per-axis code width; the default keeps the code in a built-in
/// 64-bit integer (3D: 21 bits/axis → 96-bit codes use __uint128, so we cap the
/// default at 21 for a 64-bit `code_type` in 3D and 32 in 2D). A block extent must
/// satisfy `size[i] <= 2^Bits`. Codes are *origin-relative*: codeOf subtracts the
/// origin so the curve starts at the block corner.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class MortonIndexer {
 public:
  using M = morton::Morton<Dim, Bits>;
  using Code = typename M::code_type;
  using Coord = typename M::coord_type;

  MortonIndexer() = default;
  MortonIndexer(IVec<Dim> origin, IVec<Dim> size) { init(origin, size); }

  void init(IVec<Dim> origin, IVec<Dim> size) {
    origin_ = origin;
    size_ = size;
  }

  static constexpr unsigned bits() { return Bits; }
  const IVec<Dim>& origin() const { return origin_; }
  const IVec<Dim>& size() const { return size_; }

  /// Global multi-index → Z-order code (origin-relative). Precondition: every
  /// `g[i] - origin[i]` is in [0, 2^Bits).
  MORTON_HD Code codeOf(const IVec<Dim>& g) const {
    M m;
    for (int i = 0; i < Dim; ++i)
      m.set(static_cast<unsigned>(i), localCoord(g, i));
    return m.code();
  }

  /// Z-order code → global multi-index (inverse of codeOf).
  MORTON_HD IVec<Dim> multiIndex(Code c) const {
    M m = M::from_code(c);
    IVec<Dim> g{};
    for (int i = 0; i < Dim; ++i)
      g[i] = static_cast<Index>(m.get(static_cast<unsigned>(i))) + origin_[i];
    return g;
  }

  /// Neighbour code one cell along ±`axis`, computed directly in Morton space
  /// (the headline O(1) morton arithmetic — no decode/re-encode). `dir>=0` → +1.
  /// Wraps mod 2^Bits per axis, like morton's `neighbor`.
  MORTON_HD Code neighborCode(Code c, int axis, int dir) const {
    return M::from_code(c).neighbor(static_cast<unsigned>(axis), dir >= 0 ? 1 : -1).code();
  }

 private:
  MORTON_HD Coord localCoord(const IVec<Dim>& g, int i) const {
    return static_cast<Coord>(g[i] - origin_[i]);
  }

  IVec<Dim> origin_{};
  IVec<Dim> size_{};
};

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_DECOMP_MORTON_INDEXER_HPP

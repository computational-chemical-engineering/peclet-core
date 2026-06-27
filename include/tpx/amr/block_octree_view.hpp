// transport-core — device-resident view of a BlockOctree (portable Kokkos).
//
// BlockOctree (block_octree.hpp) owns the topology on the host: refine / coarsen /
// 2:1 balance rebuild its sorted leaf arrays. BlockOctreeView mirrors those
// arrays into Kokkos Views and exposes the *queries* (point location, and the
// face-neighbour step) as KOKKOS_INLINE_FUNCTIONs over the same code, so field
// kernels and the (later) octree multigrid can locate / walk leaves on device on
// any backend (CUDA / HIP / OpenMP). This mirrors the host-vs-device split the
// halo already uses (grid_halo.hpp vs grid_halo_kokkos.hpp): topology on host,
// the per-leaf hot path on device, bit-for-bit identical results.
//
// Including this header requires Kokkos (build with -DTPX_ENABLE_KOKKOS=ON) and
// the morton sibling checkout (TPX_HAVE_MORTON, with MORTON_ENABLE_KOKKOS so
// MORTON_HD == KOKKOS_FUNCTION).
#ifndef TPX_AMR_BLOCK_OCTREE_VIEW_HPP
#define TPX_AMR_BLOCK_OCTREE_VIEW_HPP

#ifdef TPX_HAVE_MORTON

#include <cstdint>

#include "tpx/amr/block_octree.hpp"
#include "tpx/common/view.hpp"

namespace tpx::amr {

/// Device mirror of a BlockOctree's leaf arrays + device-callable queries.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
struct BlockOctreeView {
  using Host = BlockOctree<Dim, Bits>;
  using M = typename Host::M;
  using Code = typename Host::Code;
  using Coord = typename Host::Coord;

  View<Code> codes;            // leaf origin codes, ascending (Z-order)
  View<std::uint8_t> levels;   // parallel to codes
  Index n = 0;

  /// (Re)upload the host octree's current leaf set to the device.
  void upload(const Host& t) {
    codes = toDevice(t.codes(), "amr_codes");
    levels = toDevice(t.levels(), "amr_levels");
    n = t.numLeaves();
  }

  Index numLeaves() const { return n; }

  /// Leaf containing Morton code `p`, or -1. Callable in a device kernel.
  KOKKOS_INLINE_FUNCTION Index locate(Code p) const {
    return amrLocate<M>(codes.data(), levels.data(), n, p);
  }

  /// Leaf across leaf `i`'s face on `axis` in direction `dir` (±1), or -1 if it
  /// lies outside the block. Same arithmetic as BlockOctree::faceNeighbor.
  KOKKOS_INLINE_FUNCTION Index faceNeighbor(Index i, int axis, int dir) const {
    M probe = M::from_code(codes(i));
    const Coord step = Coord(Coord(1) << levels(i));
    if (dir >= 0) {
      if (!probe.try_add(static_cast<unsigned>(axis), step)) return -1;
    } else {
      if (!probe.try_sub(static_cast<unsigned>(axis), 1)) return -1;
    }
    return locate(probe.code());
  }
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_BLOCK_OCTREE_VIEW_HPP

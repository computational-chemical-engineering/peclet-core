// transport-core — device assembly of the FV (pressure) operator (D2), built on the S1 CSR primitive.
//
// AmrPoisson::assembleFv walks `forEachFaceNeighbor` on the host (serial) to produce the weight-CSR
// (invVol / start / nbr / coef=openness·A_f/d_f / bcDiag) that the device MG then applies. For a STATIC
// geometry that host walk runs once; for a DYNAMIC geometry (moving SDF / adapt) it would run every
// step and force a host round-trip. This header reproduces that walk as a device functor + a per-cell
// diagonal kernel, so the FvOp is assembled entirely on the device — no host round-trip — feeding the
// same shared face_csr.hpp apply kernels.
//
// Bit-exactness: the device functor emits faces in the exact same axis/dir/sub-face order as the host
// `forEachFaceNeighbor`, each cell fills its own CSR slice (S1, atomic-free), and every coefficient
// uses the identical double arithmetic. So on the OpenMP backend the device-assembled FvOp is
// bit-for-bit equal to the host `assembleFv` / `buildFaceCsr` (the cross-backend anti-drift lock in
// tests/test_amr_device_assembly_kokkos). GPU is tolerance-not-bit-exact (FMA), per the convention.
//
// Requires a Kokkos build + the morton checkout (TPX_HAVE_MORTON ⇒ MORTON_HD == KOKKOS_FUNCTION).
#ifndef TPX_AMR_DEVICE_ASSEMBLY_HPP
#define TPX_AMR_DEVICE_ASSEMBLY_HPP

#ifdef TPX_HAVE_MORTON

#include <array>

#include "tpx/amr/block_octree_view.hpp"
#include "tpx/amr/device_csr.hpp"
#include "tpx/amr/fv_op.hpp"
#include "tpx/amr/poisson.hpp"
#include "tpx/common/view.hpp"

namespace tpx::amr {

/// Device-callable reproduction of AmrPoisson's per-cell face walk + geometry. Trivially copyable
/// (Views are handles, the rest are scalars), so it captures by value into Kokkos kernels. Drives both
/// the CSR emit (operator()) and the per-cell diagonal (cellVolume/bcDiag).
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
struct FvFaceEmit {
  using View_t = BlockOctreeView<Dim, Bits>;
  using M = typename View_t::M;
  using Code = typename View_t::Code;
  using Coord = typename View_t::Coord;

  View_t ov;                       ///< device octree (codes/levels/locate)
  View<const double> alpha;        ///< per-leaf·(2·Dim) openness, or empty when !hasOpen
  Coord fineExt[Dim] = {};         ///< periodic-wrap modulus per axis
  Real h0 = 1.0;                   ///< finest cell width
  bool hasOpen = false;
  bool periodic = true;
  bool immersedWall = false;

  static constexpr int kFaces = 2 * Dim;
  KOKKOS_INLINE_FUNCTION static int faceIndex(int axis, int dir) {
    return 2 * axis + (dir > 0 ? 0 : 1);
  }
  KOKKOS_INLINE_FUNCTION double openness(Index leaf, int axis, int dir) const {
    if (!hasOpen) return 1.0;
    return alpha(static_cast<std::size_t>(leaf) * kFaces + faceIndex(axis, dir));
  }
  KOKKOS_INLINE_FUNCTION Real areaOf(Coord s) const {
    Real a = 1;
    for (int d = 0; d < Dim - 1; ++d) a *= static_cast<Real>(s) * h0;
    return a;
  }
  KOKKOS_INLINE_FUNCTION Real coeff(Coord si, Coord sj) const {
    const Real dist = 0.5 * (static_cast<Real>(si) + static_cast<Real>(sj)) * h0;
    return areaOf(si < sj ? si : sj) / dist;
  }
  KOKKOS_INLINE_FUNCTION Coord wrap(long c, int axis) const {
    const long e = static_cast<long>(fineExt[axis]);
    return static_cast<Coord>(((c % e) + e) % e);
  }
  KOKKOS_INLINE_FUNCTION Real cellWidth(Index i) const {
    return h0 * static_cast<Real>(Index(1) << ov.levels(i));
  }
  KOKKOS_INLINE_FUNCTION Real cellVolume(Index i) const {
    Real w = cellWidth(i), v = 1;
    for (int d = 0; d < Dim; ++d) v *= w;
    return v;
  }

  /// Σ over leaf i's Dirichlet-wall faces of A_f/(½·cellWidth) — mirrors AmrPoisson::boundaryDiag.
  KOKKOS_INLINE_FUNCTION double bcDiag(Index i) const {
    if (periodic && !immersedWall) return 0.0;
    std::array<Coord, Dim> lo = M::from_code(ov.codes(i)).decode();
    const Coord si = Coord(Coord(1) << ov.levels(i));
    const double wall = areaOf(si) / (0.5 * static_cast<Real>(si) * h0);
    double s = 0.0;
    for (int axis = 0; axis < Dim; ++axis)
      for (int dir = -1; dir <= 1; dir += 2) {
        const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                  : static_cast<long>(lo[axis]) - 1;
        const bool domainBoundary = !periodic && (pc < 0 || pc >= static_cast<long>(fineExt[axis]));
        if (domainBoundary)
          s += openness(i, axis, dir) * wall;
        else if (immersedWall)
          s += (1.0 - openness(i, axis, dir)) * wall;
      }
    return s;
  }

  /// Emit each face of leaf i as sink(neighbourLeaf, coef=openness·A_f/d_f) — the exact enumeration
  /// (axis-major, dir −1 then +1, 2:1 sub-faces in bit order) and weights of forEachFaceNeighbor.
  template <class Sink>
  KOKKOS_INLINE_FUNCTION void operator()(Index i, Sink& sink) const {
    std::array<Coord, Dim> lo = M::from_code(ov.codes(i)).decode();
    const unsigned Li = ov.levels(i);
    const Coord si = Coord(Coord(1) << Li);
    for (int axis = 0; axis < Dim; ++axis)
      for (int dir = -1; dir <= 1; dir += 2) {
        const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                  : static_cast<long>(lo[axis]) - 1;
        if (!periodic && (pc < 0 || pc >= static_cast<long>(fineExt[axis]))) continue;
        std::array<Coord, Dim> p = lo;
        p[axis] = wrap(pc, axis);
        const Index j = ov.locate(M::encode(p).code());
        const unsigned Lj = ov.levels(j);
        if (Lj >= Li) {
          sink(j, openness(i, axis, dir) * coeff(si, Coord(Coord(1) << Lj)));
        } else {
          const Coord sj = Coord(si >> 1);
          const int nsub = 1 << (Dim - 1);
          for (int k = 0; k < nsub; ++k) {
            std::array<Coord, Dim> q = lo;
            q[axis] = wrap(pc, axis);
            int bit = 0;
            for (int t = 0; t < Dim; ++t) {
              if (t == axis) continue;
              const Coord off = ((k >> bit) & 1) ? sj : Coord(0);
              q[t] = wrap(static_cast<long>(lo[t]) + static_cast<long>(off), t);
              ++bit;
            }
            const Index jj = ov.locate(M::encode(q).code());
            sink(jj, openness(jj, axis, -dir) * coeff(si, sj));
          }
        }
      }
  }
};

/// Assemble the FvOp entirely on the device from a host AmrPoisson's geometry (h0, periodicity,
/// immersed-wall mode, fine extent, and openness) + a device BlockOctreeView. The openness array is
/// staged to the device once (it is the only host input — the expensive face walk + weight build runs
/// on device). Result equals the host AmrPoisson::assembleFv CSR bit-for-bit on OpenMP.
template <int Dim, unsigned Bits>
FvOp deviceAssembleFv(const AmrPoisson<Dim, Bits>& ap, const BlockOctreeView<Dim, Bits>& ov) {
  FvFaceEmit<Dim, Bits> emit;
  emit.ov = ov;
  emit.h0 = ap.h0();
  emit.periodic = ap.periodic();
  emit.immersedWall = ap.immersedWall();
  emit.hasOpen = ap.hasOpenness();
  for (int d = 0; d < Dim; ++d) emit.fineExt[d] = ap.fineExt()[d];
  if (ap.hasOpenness()) emit.alpha = toDevice(ap.opennessRaw(), "amr::alpha");

  DeviceCsr csr = deviceBuildFaceCsr(ov.numLeaves(), emit);

  FvOp op;
  op.n = ov.numLeaves();
  op.faceStart = csr.start;
  op.faceNbr = csr.nbr;
  op.faceW = csr.coef;
  op.invVol = View<double>(Kokkos::view_alloc("amr::invVol", Kokkos::WithoutInitializing),
                           static_cast<std::size_t>(op.n));
  op.bcDiag = View<double>(Kokkos::view_alloc("amr::bcDiag", Kokkos::WithoutInitializing),
                           static_cast<std::size_t>(op.n));
  View<double> invVol = op.invVol;
  View<double> bcDiag = op.bcDiag;
  const FvFaceEmit<Dim, Bits> e = emit;
  Kokkos::parallel_for(
      "amr::fv_diag", op.n, KOKKOS_LAMBDA(const Index i) {
        invVol(i) = 1.0 / e.cellVolume(i);
        bcDiag(i) = e.bcDiag(i);
      });
  return op;
}

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_ASSEMBLY_HPP

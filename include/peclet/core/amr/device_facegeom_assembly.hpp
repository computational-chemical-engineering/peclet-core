// transport-core — device assembly of the collocated-projection FACE-GEOMETRY tables (D4).
//
// buildFaceGeom (flow.hpp) walks AmrPoisson::forEachFaceFull on the host to produce the per-(sub)face
// geometry CSR the collocated divergence / gradient / ABC face-field / FOU advection kernels consume:
// neighbour, axis, dir, α·area, raw area, distance, openness α, and the two SOU upstream-probe leaves.
// It is pure per-face geometry — embarrassingly parallel — so for a moving boundary it should rebuild on
// device rather than re-walk the host and re-upload. This header does that, reusing the D2 FvFaceEmit
// traversal helpers (decode / wrap / areaOf / openness / locate) and the S1 offsets primitive.
//
// Bit-exactness: same forEachFaceFull enumeration (axis-major, dir −1/+1, 2:1 sub-faces), same per-face
// formulas, each cell fills its own slice — so on OpenMP the device FaceGeom == host buildFaceGeom
// bit-for-bit (test_amr_device_facegeom). GPU is tolerance-not-bit-exact (FMA), per the convention.
//
// Requires a Kokkos build + the morton checkout (PECLET_CORE_HAVE_MORTON).
#ifndef PECLET_CORE_AMR_DEVICE_FACEGEOM_ASSEMBLY_HPP
#define PECLET_CORE_AMR_DEVICE_FACEGEOM_ASSEMBLY_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>

#include "peclet/core/amr/block_octree_view.hpp"
#include "peclet/core/amr/device_assembly.hpp"  // FvFaceEmit (shared geometry traversal helpers)
#include "peclet/core/amr/csr.hpp"       // deviceScanOffsets
#include "peclet/core/amr/face_geom.hpp"        // FaceGeom (the produced type)
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::core::amr {

/// The device face-geometry walker: replicates AmrPoisson::forEachFaceFull + periodicNeighbor over the
/// device octree, using FvFaceEmit's geometry helpers. `g` carries the octree view + openness + h0 +
/// fineExt (built with hasOpen as the source openness has it).
template <unsigned Bits = 21u>
struct FaceGeomEmit {
  using M = typename BlockOctreeView<3, Bits>::M;
  using Coord = typename BlockOctreeView<3, Bits>::Coord;
  FvFaceEmit<3, Bits> g;

  /// Single-leaf periodic face neighbour (AmrPoisson::periodicNeighbor) — the SOU upstream probe.
  KOKKOS_INLINE_FUNCTION Index periodicNeighbor(Index i, int axis, int dir) const {
    std::array<Coord, 3> lo = M::from_code(g.ov.codes(i)).decode();
    const Coord si = Coord(Coord(1) << g.ov.levels(i));
    const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                              : static_cast<long>(lo[axis]) - 1;
    std::array<Coord, 3> p = lo;
    p[axis] = g.wrap(pc, axis);
    return g.ov.locate(M::encode(p).code());
  }

  /// Visit each (sub)face of leaf i: cb(neighbour, axis, dir, area, dist, alpha) in forEachFaceFull
  /// order. No domain-boundary skip (the collocated path is periodic), matching the host walk.
  template <class CB>
  KOKKOS_INLINE_FUNCTION void forEachFaceFull(Index i, CB& cb) const {
    std::array<Coord, 3> lo = M::from_code(g.ov.codes(i)).decode();
    const unsigned Li = g.ov.levels(i);
    const Coord si = Coord(Coord(1) << Li);
    for (int axis = 0; axis < 3; ++axis)
      for (int dir = -1; dir <= 1; dir += 2) {
        const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                  : static_cast<long>(lo[axis]) - 1;
        std::array<Coord, 3> p = lo;
        p[axis] = g.wrap(pc, axis);
        const Index j = g.ov.locate(M::encode(p).code());
        const unsigned Lj = g.ov.levels(j);
        if (Lj >= Li) {
          const Coord sj = Coord(Coord(1) << Lj);
          cb(j, axis, dir, g.areaOf(si), 0.5 * (static_cast<Real>(si) + static_cast<Real>(sj)) * g.h0,
             g.openness(i, axis, dir));
        } else {
          const Coord sj = Coord(si >> 1);
          const int nsub = 1 << 2;  // 2^(Dim-1), Dim=3
          for (int k = 0; k < nsub; ++k) {
            std::array<Coord, 3> q = lo;
            q[axis] = g.wrap(pc, axis);
            int bit = 0;
            for (int t = 0; t < 3; ++t) {
              if (t == axis) continue;
              const Coord off = ((k >> bit) & 1) ? sj : Coord(0);
              q[t] = g.wrap(static_cast<long>(lo[t]) + static_cast<long>(off), t);
              ++bit;
            }
            const Index jj = g.ov.locate(M::encode(q).code());
            cb(jj, axis, dir, g.areaOf(sj),
               0.5 * (static_cast<Real>(si) + static_cast<Real>(sj)) * g.h0,
               g.openness(jj, axis, -dir));
          }
        }
      }
  }

  KOKKOS_INLINE_FUNCTION Index count(Index i) const {
    struct CountCB {
      Index c = 0;
      KOKKOS_INLINE_FUNCTION void operator()(Index, int, int, double, double, double) { ++c; }
    } cb;
    forEachFaceFull(i, cb);
    return cb.c;
  }
};

/// Assemble the collocated FaceGeom entirely on device from a built AmrPoisson (openness set) + a
/// per-cell fluid flag + the device octree view. Equals host buildFaceGeom bit-for-bit on OpenMP.
template <unsigned Bits>
FaceGeom deviceAssembleFaceGeom(const AmrPoisson<3, Bits>& ap, const std::vector<char>& fluidHost,
                                const BlockOctreeView<3, Bits>& ov) {
  FaceGeomEmit<Bits> emit;
  emit.g.ov = ov;
  emit.g.h0 = ap.h0();
  emit.g.periodic = ap.periodic();
  emit.g.hasOpen = ap.hasOpenness();
  for (int d = 0; d < 3; ++d) emit.g.fineExt[d] = ap.fineExt()[d];
  if (ap.hasOpenness()) emit.g.alpha = toDevice(ap.opennessRaw(), "fg::alpha");

  const Index n = ov.numLeaves();
  Index nf = 0;
  View<Index> start =
      deviceScanOffsets(n, KOKKOS_LAMBDA(const Index i) { return emit.count(i); }, nf);

  FaceGeom fg;
  fg.n = n;
  fg.start = start;
  // NB: pass the label as std::string — a const char* *variable* is read by view_alloc as a
  // pointer-to-memory (only string literals are special-cased as labels).
  auto mk = [&](const char* lbl, std::size_t m) {
    return View<double>(Kokkos::view_alloc(std::string(lbl), Kokkos::WithoutInitializing), m);
  };
  auto mkI = [&](const char* lbl, std::size_t m) {
    return View<Index>(Kokkos::view_alloc(std::string(lbl), Kokkos::WithoutInitializing), m);
  };
  const std::size_t F = static_cast<std::size_t>(nf);
  fg.nbr = mkI("fg::nbr", F);
  fg.axis = View<int>(Kokkos::view_alloc("fg::axis", Kokkos::WithoutInitializing), F);
  fg.dir = View<int>(Kokkos::view_alloc("fg::dir", Kokkos::WithoutInitializing), F);
  fg.alphaArea = mk("fg::aArea", F);
  fg.rawArea = mk("fg::rArea", F);
  fg.dist = mk("fg::dist", F);
  fg.alpha = mk("fg::alpha", F);
  fg.upupI = mkI("fg::upupI", F);
  fg.upupJ = mkI("fg::upupJ", F);
  fg.invVol = mk("fg::invVol", static_cast<std::size_t>(n));
  fg.fluid = View<char>(Kokkos::view_alloc("fg::fluid", Kokkos::WithoutInitializing),
                        static_cast<std::size_t>(n));

  View<char> fluidDev = toDevice(fluidHost, "fg::fluidIn");
  FaceGeom out = fg;  // copy of the View handles for capture
  const FaceGeomEmit<Bits> e = emit;
  Kokkos::parallel_for(
      "amr::facegeom_fill", n, KOKKOS_LAMBDA(const Index i) {
        out.invVol(i) = 1.0 / e.g.cellVolume(i);
        out.fluid(i) = fluidDev(i);
        // own-slice fill cursor; cb writes one (sub)face per call (forEachFaceFull order).
        struct FillCB {
          const FaceGeomEmit<Bits>* e;
          const FaceGeom* o;  // captured-by-value FaceGeom is const; View element writes still work
          Index i;
          Index k;
          KOKKOS_INLINE_FUNCTION void operator()(Index j, int axis, int dir, double area, double dist,
                                                 double alpha) {
            o->nbr(k) = j;
            o->axis(k) = axis;
            o->dir(k) = dir;
            o->alphaArea(k) = alpha * area;
            o->rawArea(k) = area;
            o->dist(k) = dist;
            o->alpha(k) = alpha;
            o->upupI(k) = e->periodicNeighbor(i, axis, -dir);
            o->upupJ(k) = e->periodicNeighbor(j, axis, dir);
            ++k;
          }
        } cb{&e, &out, i, out.start(i)};
        e.forEachFaceFull(i, cb);
      });
  return fg;
}

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_DEVICE_FACEGEOM_ASSEMBLY_HPP

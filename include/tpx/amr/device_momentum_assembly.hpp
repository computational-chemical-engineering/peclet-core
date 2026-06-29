// transport-core — device assembly of the cut-cell MOMENTUM operator (D3), on the S1 CSR primitive.
//
// AmrCutCell::build (host) samples the SDF per cell and runs buildCutStencil to produce the ξ-overlay
// stencil (AC_/off_/cut_/rscale_); assembleOperator (host) then merges that stencil, the regular-fluid
// C/F-aware ∇², and the implicit-FOU advection into one diag + face-CSR via a std::vector<vector<pair>>.
// For a moving boundary that whole walk reruns every step on the host and round-trips the operator.
//
// This header moves the two device-portable halves onto the device:
//   * the per-cell ξ stencil rebuild — buildCutStencil is pure math (now MORTON_HD), fed the cell + 6
//     neighbour SDF samples (staged like D2 stages openness; the SDF *sampling* itself, like openness
//     sampling, stays host until a device SDF exists); and
//   * the operator merge — the three per-cell branches (solid identity / ξ-overlay / regular ∇²·μ) plus
//     the advection fold, emitted as a face-CSR through deviceBuildFaceCsr (S1) + a per-cell diagonal.
// The regular-fluid faces reuse the D2 FvFaceEmit traversal (α=1) via a scaling sink adapter, so the
// 2:1 enumeration and coeff are shared, not re-derived.
//
// Bit-exactness: emit order, branch logic and arithmetic mirror assembleOperator exactly, each cell
// fills its own CSR slice (S1, atomic-free), so on OpenMP the device MomentumOp == host assembleOperator
// bit-for-bit (test_amr_device_momentum). GPU is tolerance-not-bit-exact (FMA), per the convention.
//
// Requires a Kokkos build + the morton checkout (TPX_HAVE_MORTON).
#ifndef TPX_AMR_DEVICE_MOMENTUM_ASSEMBLY_HPP
#define TPX_AMR_DEVICE_MOMENTUM_ASSEMBLY_HPP

#ifdef TPX_HAVE_MORTON

#include "tpx/amr/block_octree_view.hpp"
#include "tpx/amr/cut_cell.hpp"
#include "tpx/amr/device_assembly.hpp"  // FvFaceEmit (the α=1 ∇² geometry traversal)
#include "tpx/amr/device_csr.hpp"
#include "tpx/amr/momentum.hpp"
#include "tpx/common/view.hpp"

namespace tpx::amr {

/// Sink adapter: forwards each face from the FvFaceEmit traversal as (j, factor·coef) to the real CSR
/// sink — turns the plain geometric coeff into the regular-fluid momentum coupling −μ·invV·(a·c).
template <class Sink>
struct MomScaleSink {
  Sink* sink;
  double factor;
  KOKKOS_INLINE_FUNCTION void operator()(Index j, double c) const { (*sink)(j, factor * c); }
};

/// Sink that just sums the geometric coeffs of a cell's faces (Σ a·c for the regular diagonal).
struct MomSumSink {
  double sum = 0.0;
  KOKKOS_INLINE_FUNCTION void operator()(Index /*j*/, double c) { sum += c; }
};

/// Device-callable reproduction of AmrCutCell::assembleOperator's per-cell row + diagonal. Holds the
/// staged stencil/advection device arrays + the shared α=1 ∇² geometry (FvFaceEmit). Trivially copyable.
template <unsigned Bits = 21u>
struct MomFaceEmit {
  FvFaceEmit<3, Bits> geom;  ///< α=1 (hasOpen=false) C/F-aware ∇² traversal for regular fluid cells
  View<const char> fluid, cut;
  View<const double> AC, off, rscale;  ///< off is n·6 (ξ-overlay), AC/rscale are n
  View<const Index> nbr6;              ///< n·6 periodic face-neighbour indices (cut-stencil order)
  View<const double> advDiag, advCoef;
  View<const Index> advStart, advNbr;
  double idiag = 0.0, mu = 1.0;
  bool hasAdv = false, scaleAdv = false;

  KOKKOS_INLINE_FUNCTION double cellVolume(Index i) const { return geom.cellVolume(i); }

  /// Emit the off-diagonal entries of row i (S1 fill order): ξ-overlay for cut cells, −μ·invV·(a·c)
  /// for regular fluid, then the folded implicit-FOU advection — exactly assembleOperator's order.
  template <class Sink>
  KOKKOS_INLINE_FUNCTION void operator()(Index i, Sink& sink) const {
    if (!fluid(i)) return;  // solid: identity row, no off-diagonals
    const std::size_t s = static_cast<std::size_t>(i);
    if (cut(i)) {
      for (int k = 0; k < 6; ++k) {
        const double a = off(s * 6 + k);
        if (a == 0.0) continue;
        const Index j = nbr6(s * 6 + k);
        if (j >= 0) sink(j, a);
      }
    } else {
      const double invV = 1.0 / geom.cellVolume(i);
      MomScaleSink<Sink> adapter{&sink, -mu * invV};
      geom(i, adapter);  // emits (j, -mu·invV·(a·c)) per C/F face (a=1)
    }
    if (hasAdv) {
      const double as = scaleAdv ? rscale(i) : 1.0;
      for (Index p = advStart(i); p < advStart(i + 1); ++p) sink(advNbr(p), as * advCoef(p));
    }
  }

  /// Diagonal of row i: 1 (solid), AC (cut), or idiag + μ·invV·Σ(a·c) (regular), plus folded advection.
  KOKKOS_INLINE_FUNCTION double diag(Index i) const {
    if (!fluid(i)) return 1.0;
    double d;
    if (cut(i)) {
      d = AC(i);
    } else {
      const double invV = 1.0 / geom.cellVolume(i);
      MomSumSink ss;
      geom(i, ss);
      d = idiag + mu * invV * ss.sum;
    }
    if (hasAdv) {
      const double as = scaleAdv ? rscale(i) : 1.0;
      d += as * advDiag(i);
    }
    return d;
  }
};

/// The per-cell ξ-overlay rebuild (build() Pass 2) on device: from the staged cell SDF + neighbour
/// indices + fluid flag, recompute AC/off/cut/rscale exactly as the host build does. The expensive part
/// of a moving-boundary re-assembly, now device-resident; bit-exact vs the host AC_/off_/cut_/rscale_.
template <unsigned Bits>
void deviceRebuildCutStencil(Index n, double beta, double AC0, const View<double>& sdfC,
                             const View<Index>& nbr6, const View<char>& fluid, View<double> AC,
                             View<double> off, View<char> cut, View<double> rscale) {
  Kokkos::parallel_for(
      "amr::cut_stencil", n, KOKKOS_LAMBDA(const Index i) {
        const std::size_t s = static_cast<std::size_t>(i);
        if (!fluid(i)) {  // solid: identity row
          AC(i) = 1.0;
          for (int k = 0; k < 6; ++k) off(s * 6 + k) = 0.0;
          cut(i) = 0;
          rscale(i) = 1.0;
          return;
        }
        double sdf_n[6];
        bool anyGhost = false;
        for (int k = 0; k < 6; ++k) {
          const Index j = nbr6(s * 6 + k);
          sdf_n[k] = (j >= 0) ? sdfC(j) : -1.0;  // missing neighbour => solid
          if (sdf_n[k] < 0.0) anyGhost = true;
        }
        cut(i) = anyGhost ? 1 : 0;
        double AClocal = AC0, offl[6];
        for (int k = 0; k < 6; ++k) offl[k] = -beta;
        double rsl = 1.0, inhoml = 0.0;
        if (anyGhost)
          AmrCutCell<Bits>::buildCutStencil(sdfC(i), sdf_n, beta, AC0, AClocal, offl, rsl, inhoml);
        AC(i) = AClocal;
        for (int k = 0; k < 6; ++k) off(s * 6 + k) = offl[k];
        rscale(i) = rsl;
      });
}

/// Assemble the MomentumOp entirely on the device from a built host AmrCutCell + a device octree view.
/// The SDF samples / neighbour indices / fluid flags / advection CSR are staged to the device (the SDF
/// sampling stays host, like D2's openness); the stencil rebuild + operator merge run on device. The
/// advection is folded into the single CSR (op.hasAdv = false), matching host assembleOperator + hostOp.
template <unsigned Bits>
MomentumOp deviceAssembleMomentum(const AmrCutCell<Bits>& ccop, const BlockOctreeView<3, Bits>& ov,
                                  bool scaleAdvByRscale = false) {
  const Index n = ov.numLeaves();
  const double beta = ccop.beta();
  const double AC0 = ccop.idiag() + 6.0 * beta;

  // Stage build inputs + rebuild the ξ stencil on device.
  View<double> sdfC = toDevice(ccop.sdfCRaw(), "mom::sdfC");
  View<Index> nbr6 = toDevice(ccop.nbRaw(), "mom::nb");
  View<char> fluid = toDevice(ccop.fluidRaw(), "mom::fluid");
  View<double> AC(Kokkos::view_alloc("mom::AC", Kokkos::WithoutInitializing), static_cast<std::size_t>(n));
  View<double> off(Kokkos::view_alloc("mom::off", Kokkos::WithoutInitializing),
                   static_cast<std::size_t>(n) * 6);
  View<char> cut(Kokkos::view_alloc("mom::cut", Kokkos::WithoutInitializing), static_cast<std::size_t>(n));
  View<double> rscale(Kokkos::view_alloc("mom::rscale", Kokkos::WithoutInitializing),
                      static_cast<std::size_t>(n));
  deviceRebuildCutStencil<Bits>(n, beta, AC0, sdfC, nbr6, fluid, AC, off, cut, rscale);

  // The α=1 ∇² geometry for regular fluid cells (reuses the D2 traversal with openness off).
  FvFaceEmit<3, Bits> geom;
  geom.ov = ov;
  geom.h0 = ccop.lap().h0();
  geom.periodic = ccop.lap().periodic();
  geom.hasOpen = false;
  for (int d = 0; d < 3; ++d) geom.fineExt[d] = ccop.lap().fineExt()[d];

  MomFaceEmit<Bits> emit;
  emit.geom = geom;
  emit.fluid = fluid;
  emit.cut = cut;
  emit.AC = AC;
  emit.off = off;
  emit.rscale = rscale;
  emit.nbr6 = nbr6;
  emit.idiag = ccop.idiag();
  emit.mu = ccop.mu();
  emit.hasAdv = ccop.hasAdv();
  emit.scaleAdv = scaleAdvByRscale;
  if (ccop.hasAdv()) {
    emit.advDiag = toDevice(ccop.advDiagRaw(), "mom::advDiag");
    emit.advCoef = toDevice(ccop.advCoefRaw(), "mom::advCoef");
    emit.advStart = toDevice(ccop.advStartRaw(), "mom::advStart");
    emit.advNbr = toDevice(ccop.advNbrRaw(), "mom::advNbr");
  }

  DeviceCsr csr = deviceBuildFaceCsr(n, emit);

  MomentumOp op;
  op.n = n;
  op.faceStart = csr.start;
  op.faceNbr = csr.nbr;
  op.faceCoef = csr.coef;
  op.hasAdv = false;  // advection folded into the single CSR (as host assembleOperator does)
  op.diag = View<double>(Kokkos::view_alloc("mom::diag", Kokkos::WithoutInitializing),
                         static_cast<std::size_t>(n));
  View<double> diagV = op.diag;
  const MomFaceEmit<Bits> e = emit;
  Kokkos::parallel_for(
      "amr::mom_diag", n, KOKKOS_LAMBDA(const Index i) { diagV(i) = e.diag(i); });
  return op;
}

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_MOMENTUM_ASSEMBLY_HPP

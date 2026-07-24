// core — directional ghost-cell closure primitives (shared flow <-> AMR).
//
// The PURE per-face pieces of the directional ghost-cell IBM projection, lifted verbatim from
// flow/src/ghost_projection.hpp (where they were validated 2nd-order on Zick & Homsy, staggered
// AND collocated — see that header for the full scheme documentation): a solid staggered/averaged
// face is closed by the momentum IBM's 1-D wall-anchored quadratic along the face's own axis,
//
//     poly_D(th) * u_ghost = 2*u_bc + poly_Nc(th)*u_near + poly_N_nb(th)*u_far
//     th = sdf_near/(sdf_near - sdf_ghost), clamped [GP_THETA_MIN, 1]
//
// with the face-state cascade COUPLED / QUAD / LIN / BC_ONLY / EXPLICIT (sliver faces use the
// extended th in (1,2)). Everything here is a pure function of 1-D sdf samples along one axis —
// grid-agnostic, so the SAME classification + weights serve flow's uniform structured grid and
// core's AMR octree (whose cut cells live in a locally uniform finest band). The grid-specific
// parts (overlay SoA layout, neighbour wrap, delta kernels) stay with each consumer.
//
// float arithmetic throughout — identical to flow's cut_cell_ibm.hpp SCHEME-0 closure polys, so
// the lifted functions are drop-in for flow (same bits on its ghost path).
//
// Host/device: functions are KOKKOS_INLINE_FUNCTION when Kokkos_Core.hpp has been included
// BEFORE this header (device consumers must include Kokkos first), plain inline otherwise (the
// host-only oracle builds).
#ifndef PECLET_CORE_SCHEME_GHOST_CLOSURE_HPP
#define PECLET_CORE_SCHEME_GHOST_CLOSURE_HPP

#include <cmath>
#include <cstdint>
#include <limits>

#ifdef KOKKOS_INLINE_FUNCTION
#define PECLET_CORE_GP_HD KOKKOS_INLINE_FUNCTION
#else
#define PECLET_CORE_GP_HD inline
#endif

namespace peclet::core::scheme {

constexpr float GP_THETA_MIN = 1e-4f;

enum GpState : int8_t {
  GP_COUPLED = 0,
  GP_QUAD = 1,
  GP_LIN = 2,
  GP_BC_ONLY = 3,
  GP_EXPLICIT = 4,
};

// The wall-anchored closure polynomials (flow cut_cell_ibm.hpp SCHEME 0, float — the same
// arithmetic the momentum IBM uses, so the ghost closure agrees with the momentum solid masks).
PECLET_CORE_GP_HD float gpPolyD(float xi) {
  return xi * (1.0f + xi);
}
PECLET_CORE_GP_HD float gpPolyNc(float xi) {
  return 2.0f * (xi * xi - 1.0f);
}
PECLET_CORE_GP_HD float gpPolyNnb(float xi) {
  return xi * (1.0f - xi);
}

// NaN sentinel + finiteness test for the optional exact-crossing thetas. The values are either
// the NaN sentinel or a finite crossing fraction (never inf), so self-comparison is the portable
// isfinite.
PECLET_CORE_GP_HD float gpNan() {
#ifdef KOKKOS_INLINE_FUNCTION
  return Kokkos::nan("");
#else
  return std::numeric_limits<float>::quiet_NaN();
#endif
}
PECLET_CORE_GP_HD bool gpFinite(float x) {
  return x == x;
}

/// Closure weights for one ghost face at the requested extrapolation order. order=2 uses the
/// wall-anchored quadratic (only available for GP_QUAD faces); order=1 (or a GP_LIN face) uses
/// the wall-anchored linear closure th*u_g = u_bc + (th-1)*u_near. D is the conditioning factor
/// feeding the row rescale.
PECLET_CORE_GP_HD void gpOrderWeights(int8_t st, float th, int order, float& wbc, float& w1,
                                      float& w2, float& D) {
  wbc = w1 = w2 = 0.0f;
  D = 1.0f;
  if (st == GP_BC_ONLY) {
    wbc = 1.0f;
    return;
  }
  if (st == GP_QUAD && order == 2) {
    D = gpPolyD(th);
    wbc = 2.0f / D;
    w1 = gpPolyNc(th) / D;
    w2 = gpPolyNnb(th) / D;
  } else {  // linear (GP_LIN, or a GP_QUAD face at order 1)
    D = th;
    wbc = 1.0f / th;
    w1 = (th - 1.0f) / th;
  }
}

struct GpFace {
  int8_t state;
  float th, wbc, w1, w2, D;
};

/// Classify + fill ONE face from its 1-D sdf samples along the face's axis.
///   sg  = ghost (this) face sdf     sn = near face sdf (the cell's other face on this axis)
///   sf  = far face sdf              sb = beyond-ghost face sdf (one further into the solid)
///   snb = neighbor center sdf       sc1/sc2 = centers needed by the near/far face gradients
///   otherSolid = the cell's other face on this axis is solid (sandwich detection)
///   exStd/exSliver = optional EXACT wall-crossing thetas (analytic-SDF capability) for the
///   standard-ghost / extended-sliver branches; NaN falls back to the linear-interp theta.
/// Pure function of the samples — shared verbatim between flow's uniform grid, the host parity
/// tests, and the AMR octree band.
PECLET_CORE_GP_HD GpFace gpClassifyFace(float sg, float sn, float sf, float sb, float snb,
                                        float sc1, float sc2, bool otherSolid,
                                        float exStd = gpNan(), float exSliver = gpNan()) {
  GpFace f{GP_COUPLED, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  if (sg >= 0.0f && snb >= 0.0f)
    return f;  // COUPLED
  if (sg < 0.0f && otherSolid) {
    f.state = GP_BC_ONLY;  // sandwich: both own faces solid, wall BCs determine the axis
    f.wbc = 1.0f;
    return f;
  }
  float th;
  if (sg < 0.0f) {  // standard ghost (near face fluid guaranteed: not sandwich)
    th = gpFinite(exStd) ? exStd : sn / (sn - sg);
    th = th < GP_THETA_MIN ? GP_THETA_MIN : (th > 1.0f ? 1.0f : th);
  } else {  // sliver: face point fluid, neighbor center solid
    if (sb >= 0.0f) {
      f.state = GP_EXPLICIT;  // no crossing on the u-line: explicit u* flux, no phi coupling
      return f;
    }
    th = gpFinite(exSliver) ? exSliver : 1.0f + sg / (sg - sb);
    const float lo = 1.0f + GP_THETA_MIN;
    th = th < lo ? lo : (th > 2.0f ? 2.0f : th);
  }
  f.th = th;
  const bool src1 = (sn >= 0.0f) && (sc1 >= 0.0f);
  const bool src2 = (sf >= 0.0f) && (sc2 >= 0.0f);
  if (!src1) {
    f.state = GP_BC_ONLY;
    f.wbc = 1.0f;
    return f;
  }
  if (src2) {
    f.state = GP_QUAD;
    f.D = gpPolyD(th);
    f.wbc = 2.0f / f.D;
    f.w1 = gpPolyNc(th) / f.D;
    f.w2 = gpPolyNnb(th) / f.D;
  } else {
    f.state = GP_LIN;
    f.D = th;
    f.wbc = 1.0f / th;
    f.w1 = (th - 1.0f) / th;
  }
  return f;
}

/// Fill one overlay row from the per-axis sample sets. F[a][m+1] = face sdf at face index i+m
/// (m = -1..2, face i+m sits between centers i+m-1 and i+m); Cq[a][q+2] = center sdf at i+q
/// (q = -2..2; Cq[a][2] = own center, fluid by construction). Face slot k = 2*axis + (0 = plus
/// side, 1 = minus side). Fills BOTH weight sets: w_* at rhsOrder (divergence RHS + diagnostic)
/// and wm_* at matrixOrder (implicit phi couplings); the row rescale comes from the matrix
/// weights (it conditions the matrix row). The overlay type OV must expose
/// cell/rescale/coupled/state/th/w_bc/w_n1/w_n2/wm_n1/wm_n2 (indexable SoA — flow's GpOverlayT
/// or the AMR overlay), with `cellId` whatever cell handle the consumer uses. Returns true if
/// any face is non-COUPLED (i.e. the row belongs in the overlay).
template <class OV, class CellId>
PECLET_CORE_GP_HD bool gpFillRow(const OV& ov, int slot, CellId cellId, const float F[3][4],
                                 const float Cq[3][5], int matrixOrder, int rhsOrder,
                                 const float* exStd = nullptr, const float* exSliver = nullptr) {
  bool any = false;
  bool anyPhi = false;
  float rho = 1.0f;
  GpFace faces[6];
  const float nanv = gpNan();
  for (int a = 0; a < 3; ++a) {
    const bool solidM = F[a][1] < 0.0f;  // own minus face (m=0)
    const bool solidP = F[a][2] < 0.0f;  // own plus face (m=1)
    // minus side (k = 2a+1): ghost m=0, near m=1, far m=2, beyond m=-1; nb center q=-1;
    // gradient cells q=+1, q=+2.
    faces[2 * a + 1] =
        gpClassifyFace(F[a][1], F[a][2], F[a][3], F[a][0], Cq[a][1], Cq[a][3], Cq[a][4], solidP,
                       exStd ? exStd[2 * a + 1] : nanv, exSliver ? exSliver[2 * a + 1] : nanv);
    // plus side (k = 2a): ghost m=1, near m=0, far m=-1, beyond m=2; nb center q=+1;
    // gradient cells q=-1, q=-2.
    faces[2 * a] =
        gpClassifyFace(F[a][2], F[a][1], F[a][0], F[a][3], Cq[a][3], Cq[a][1], Cq[a][0], solidM,
                       exStd ? exStd[2 * a] : nanv, exSliver ? exSliver[2 * a] : nanv);
  }
  float wbcM[6], w1M[6], w2M[6];  // matrix-order weights (wbc unused in the matrix)
  for (int k = 0; k < 6; ++k) {
    const GpFace& f = faces[k];
    if (f.state != GP_COUPLED)
      any = true;
    if (f.state == GP_COUPLED || f.state == GP_QUAD || f.state == GP_LIN)
      anyPhi = true;
    wbcM[k] = w1M[k] = w2M[k] = 0.0f;
    if (f.state == GP_QUAD || f.state == GP_LIN) {
      float Dm;
      gpOrderWeights(f.state, f.th, matrixOrder, wbcM[k], w1M[k], w2M[k], Dm);
      if (Dm < rho)
        rho = Dm;
    }
  }
  if (!any)
    return false;
  ov.cell(slot) = cellId;
  ov.rescale(slot) = rho;
  ov.coupled(slot) = anyPhi ? 1 : 0;
  for (int k = 0; k < 6; ++k) {
    const GpFace& f = faces[k];
    float wbc, w1, w2, D;
    gpOrderWeights(f.state, f.th, rhsOrder, wbc, w1, w2, D);
    if (f.state == GP_COUPLED || f.state == GP_EXPLICIT)
      wbc = w1 = w2 = 0.0f;
    ov.state(slot * 6 + k) = f.state;
    ov.th(slot * 6 + k) = f.th;
    ov.w_bc(slot * 6 + k) = wbc;
    ov.w_n1(slot * 6 + k) = w1;
    ov.w_n2(slot * 6 + k) = w2;
    ov.wm_n1(slot * 6 + k) = w1M[k];
    ov.wm_n2(slot * 6 + k) = w2M[k];
  }
  return true;
}

}  // namespace peclet::core::scheme

#endif  // PECLET_CORE_SCHEME_GHOST_CLOSURE_HPP

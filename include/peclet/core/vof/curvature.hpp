/// @file
/// @brief core — VoF rung V3 (WO-O): interface curvature from volume fractions, as the Popinet
/// (2009) height-function cascade with a PLIC-volumetric paraboloid-fit fallback.
///
/// **Container-free by contract**, exactly as `vof/plic.hpp`: every entry point takes scalars and
/// small local arrays only — no `Kokkos::View`, no grid indexing, no halo/topology types. These
/// kernels are scheduled for promotion to `peclet::core::vof` at the V4 freeze
/// (`suite/docs/VOF_PLAN.md` §11), where the structured solver, the AMR octree and the per-bubble
/// block container share ONE copy, and the promotion must be a plain file move. The view-level
/// driver that walks the block, owns the scratch fields and registers `"kappa"` lives next door in
/// `vof/curvature_field.hpp` — the same split `plic.hpp` / `colour_field.hpp` already uses.
///
/// ## The cascade
///
/// 1. **Standard height function** in the *preferred* direction (the Cartesian axis with the
///    largest |n_d| of the cell's own PLIC normal). Heights are column sums of C over `NH = 7`
///    cells — Han/Evrard/Desjardins (2024) eq. (3), the column length used throughout the
///    literature — over the 3x3 transverse patch, then the second-order finite differences of
///    their eqs. (4)-(5).
/// 2. **Mixed-direction HF**, in two forms, of which only the first ships on:
///    - **2a** the same construction in the two remaining directions, tried in order of
///      decreasing |n_d|. This is what `height_curvature()` in Basilisk's `curvature.h` does
///      (`foreach_dimension()` over the sorted normal), and it is free — the columns are the same
///      colour data read along another axis. **Measured: it fires on 0.00-0.06 % of a sphere's
///      interfacial cells**, because the direction with the largest |n_d| is by construction the
///      one most likely to close.
///    - **2b** Popinet's *generalised* height function / Basilisk's `height_curvature_fit`: a
///      paraboloid through the interface positions of whichever of the 27 columns DO close.
///      Implemented (`PtFit`) and **default OFF** — it is measurably worse than routing those
///      cells to tier 3. See `VofCurvature::useMixedHeightFit` for the numbers and the mechanism,
///      and the DEVIATION note at the bottom of this comment for its halo-imposed restriction.
/// 3. **PLIC-volumetric paraboloid fit** (the "PV" method of Jibben et al. 2015, as evaluated and
///    recommended by Han, Evrard & Desjardins, *IJMF* 174:104769 (2024), arXiv:2304.08643) on a
///    **5^3 stencil with a Wendland C2 radial weight of width d = 2.5 cells** — their measured
///    best cost/accuracy fallback, and the exact configuration their §5 dynamic tests use.
///
/// **Why curvature never touches a normal derivative.** Rung V0 measured that MYC's *normal* error
/// does not converge under refinement (order 0.83; ~28 % of mixed cells take the non-converging
/// Youngs branch) while its *reconstruction* error does (1.98) — `doc/vof_workorders.md`, WO-D
/// finding 2. Every branch here therefore takes its geometry from **column sums of C** (tiers 1-2)
/// or from the **PLIC planes' volumes** (tier 3). The MYC normal appears in exactly two roles,
/// both categorical rather than differentiated: it *orders the three candidate column directions*
/// (Basilisk does the same), and it *defines the local frame* of the paraboloid fit. Neither is
/// differenced, and an O(1 deg) error in either changes only which equally-valid representation of
/// the same interface is fitted.
///
/// ## Sign and units
///
/// Curvature is returned as **kappa = 2H** in units of 1/h (cell units; multiply by 1/h for
/// physical units), positive for a *convex blob of liquid* — a sphere of liquid of radius R cells
/// gives kappa = +2/R. `C` is the liquid fraction and the PLIC normal `m` points into the gas
/// (`plic.hpp` convention), so the two conventions are consistent by construction.
///
/// The height is a **signed** column sum (`hfColumnHeight`), and the sign is what makes one
/// formula serve both orientations: with liquid on the low side the sum equals `z_I + R + 1/2`,
/// with liquid on the high side it equals `R + 1/2 - z_I`, so in both cases `h` grows with the
/// amount of liquid — and Han's eq. (5), whose numerator is odd in h and whose denominator is even,
/// returns the same H either way. Only the *consistency* test cares about the orientation: all nine
/// columns of a patch must be oriented the same way, or the nine numbers do not describe one
/// single-valued surface. (`h` itself is the interface position along the column only up to that
/// sign: the position is `orient * h`, which is what tier 2b's point cloud needs.)
///
/// ## Two facts from the literature the callers must respect, not fight
///
/// - **A cascade that never falls back has a bug.** Han et al. measure the HF method failing for
///   ~51 % of random paraboloids with curvedness C > 1e-1, and — on the *2D stationary droplet*,
///   i.e. the friendliest possible case — still up to **0.9 % of interfacial cells at
///   D/dx = 102.4**. Below ~4-5 cells per diameter it fails everywhere.
/// - **Curvature error stops converging below C*dx ~ 1e-2 with advection-realistic volume
///   fractions**, for every published method (their §3 and §4.1, following Remmerswaal & Veldman's
///   Lemma 3: a first-order-in-Linf volume fraction field makes the curvature zeroth-order). A
///   plateau on fine grids is the method's physics; it is measured and reported, never chased.
///
/// ## DEVIATION recorded: which "mixed-direction HF" this is, and why 2b ships off
///
/// Popinet (2009) §3 and Basilisk's `curvature.h` offer two things that both get called "mixed":
/// (a) trying the three column directions in turn (`height_curvature`), and (b) a paraboloid fit
/// to the interface *positions* implied by every defined height, in any direction
/// (`height_curvature_fit`). Both are implemented here as tiers 2a and 2b.
///
/// 2b carries one restriction and one measured verdict.
/// - **Restriction.** Basilisk gathers heights over the 3^3 neighbourhood, which needs colour at
///   +/-1 +/- 3 = +/-4 cells, i.e. a g = 4 halo or a separate exchange of the three height fields.
///   This implementation gathers from the 3x3 patches of the three directions instead — cells at
///   offsets (+/-1, +/-1, +/-3) — so the WHOLE rung fits exactly inside the colour field's existing
///   g = 3 halo (tier 1 reaches 3x3x7, tier 3 reaches a 5^3 of MYC stencils = 7^3 of colour; both
///   are exactly +/-3). No new halo machinery is needed and none was added.
/// - **Verdict: 2b ON destroys the convergence of the MAX curvature error** (order 0.00 vs 1.86
///   over 16^3 -> 64^3 on the exact-fraction sphere), on the very cells tier 3 handles at second
///   order, for every Wendland width from 1.5 to 6.0 cells. The mechanism is structural — its data
///   set is the columns the height function could close, a *slope-selected* and therefore
///   asymmetric subset, whose lever-arm bias does not shrink with h. Full numbers and the
///   reasoning in `VofCurvature::useMixedHeightFit`; regenerated on every ctest run by gate G of
///   `tests/kokkos/test_vof_curvature.cpp`. Han et al. independently measure the PV fallback as
///   *better* than HF once volume fractions carry transport error — which is exactly the regime
///   the fallback fires in.
///
/// ## Sources
///
/// - S. Popinet, *An accurate adaptive solver for surface-tension-driven interfacial flows*, JCP
///   228:5838 (2009) — the HF cascade and the generalised height function.
/// - A. Han, F. Evrard, O. Desjardins, *Comparison of methods for curvature estimation from volume
///   fractions*, IJMF 174:104769 (2024), arXiv:2304.08643 — eqs. (3)-(5) (HF), (6), (8) (Wendland),
///   (10) (curvature of a paraboloid), (11)-(15) (the PV normal equations), and every quantitative
///   claim quoted above.
/// - Z. Jibben, N. N. Carlson, M. M. Francois, *A paraboloid fitting technique for calculating
///   curvature from piecewise-linear interface reconstructions on 3D unstructured meshes*,
///   Comput. Math. Appl. (2019), arXiv:1712.05467 — the PV method itself.
/// - H. Wendland, *Piecewise polynomial, positive definite and compactly supported radial functions
///   of minimal degree*, Adv. Comput. Math. 4:389 (1995) — the C2 kernel of eq. (8).
///
/// **Layer L1 (`VOF_PLAN.md` §11): promoted from `flow/src/vof/` to `core` (WO-W0, 2026-09-02).**
/// These are the container-free VoF math kernels — one copy shared by every container: flow's
/// structured colour field, the per-bubble block container (Part III) and the future AMR path.
/// `flow/src/vof/curvature.hpp` is now a thin include + a using-directive, so every existing
/// `peclet::flow::vof::` spelling keeps working unchanged.
#ifndef PECLET_CORE_VOF_CURVATURE_HPP
#define PECLET_CORE_VOF_CURVATURE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "peclet/core/vof/plic.hpp"

namespace peclet::core::vof {

// ---------------------------------------------------------------------------------------------
// cascade bookkeeping
// ---------------------------------------------------------------------------------------------

/// Which branch of the cascade produced a cell's curvature. Written to a companion field so the
/// branch census is a measurement and never an inference, and so a cell that got NO estimate is
/// LOUD (`kCurvNoEstimate`) instead of carrying a silently-zero curvature.
enum CurvatureBranch : int {
  kCurvNone = 0,        ///< not an interfacial cell — kappa is 0 because there is no interface
  kCurvHf = 1,          ///< tier 1: standard HF in the preferred (largest |n_d|) direction
  kCurvHfMixed = 2,     ///< tier 2a: standard HF in one of the two remaining directions
  kCurvHfFit = 3,       ///< tier 2b: paraboloid fit to the interface positions of whichever
                        ///<          columns ARE consistent, pooled over all three directions
  kCurvPv = 4,          ///< tier 3: PLIC-volumetric paraboloid fit, 5^3 Wendland-weighted
  kCurvPvReduced = 5,   ///< tier 3 with the rank-deficient 3-parameter model (a1 = a2 = a4 = 0)
  kCurvNoEstimate = 6,  ///< NO estimate: fewer than 3 usable PLIC polygons in the stencil
};

/// Column length of the height function, `NH` in Han et al. eq. (3). 7 is the value used
/// throughout the literature (Popinet 2009; Cummins 2005; Owkes 2015; Jibben 2019) and is exactly
/// what the colour field's g = 3 halo reaches.
inline constexpr int kHfColumn = 7;
/// Half-width of the PV fit stencil: S = 2*kPvHalf + 1 = 5, Han et al.'s recommended S.
inline constexpr int kPvHalf = 2;
/// Wendland support width `d` in cell units. Han et al. §5 use d = 2.5 with S = 5 and show that
/// d = 3.5 recovers first-order convergence of the spurious currents on a translating droplet
/// while d = 4.5 destroys convergence entirely (over-smoothing). Exposed as an argument so the
/// V4 rung can sweep it; 2.5 is the default and the static-test value.
inline constexpr double kPvWeightWidth = 2.5;
/// Wendland support width for the tier-2b height-position fit. Wider than the PV one because the
/// points it fits reach `sqrt(1 + 1 + 2.5^2) ~ 2.9` cells by construction and a 2.5-cell support
/// would silently discard the very columns that made the tier fire.
inline constexpr double kPtWeightWidth = 3.5;

/// Colour thresholds for "this cell is pure". A cell at 1 - 1e-12 is full for the purposes of a
/// column-consistency test; the residue it contributes to the column sum is far below the height's
/// own discretisation error.
KOKKOS_INLINE_FUNCTION bool hfIsFull(double c) {
  return c >= 1.0 - 1e-10;
}
KOKKOS_INLINE_FUNCTION bool hfIsEmpty(double c) {
  return c <= 1e-10;
}

// ---------------------------------------------------------------------------------------------
// tier 1 / 2 — height functions
// ---------------------------------------------------------------------------------------------

/// One height from one column of colour values.
///
/// @param col   `nh` colour values along the column, `col[nh/2]` being the target cell.
/// @param nh    column length (odd; `kHfColumn`).
/// @param h     [out] the height in cell units, relative to the centre cell's centre. Signed so
///              that it *increases with liquid* whichever end the liquid is at.
/// @param orient [out] +1 when the low end of the column is liquid, -1 when the high end is.
/// @param monoTol tolerance on the monotonicity test (see below).
/// @return true iff the column is CONSISTENT.
///
/// This is Popinet's **outward accumulation**, not a fixed symmetric window: starting at the centre
/// cell, walk down to the first pure cell and up to the first pure cell of the *opposite* kind, and
/// integrate over that window `[a, b]` only. The height then follows from the window's own end:
///
///     liquid low  (col[a] full,  col[b] empty):  h =  sum_{a..b} C + a - 1/2
///     liquid high (col[a] empty, col[b] full ):  h =  sum_{a..b} C - b - 1/2
///
/// (indices centre-relative). When both ends of the full column happen to be pure the two forms
/// collapse to the familiar `sum(col) - NH/2` of Han et al. eq. (3) — the window formulation is
/// strictly more permissive, because it also serves a column whose far end is *not* pure (a second
/// interface just outside, a wisp) and a column whose centre cell is itself pure (which every
/// neighbour column of a sloping patch eventually is).
///
/// A column is consistent when such a window exists AND the colour is monotone across it to within
/// `monoTol`. Monotonicity is what the work order's "where monotone columns exist" names, and it
/// is not decoration: `[1,1,0,1,0,0,0]` brackets a transition but its sum is not a height, it is a
/// height plus a sub-grid blob.
///
/// The orientation is decided by which half of the column carries more liquid, so it is a property
/// of the data rather than of an externally supplied normal — the cascade never differentiates a
/// normal (see the file header). An exactly balanced column (a symmetric double interface) is
/// rejected.
KOKKOS_INLINE_FUNCTION bool hfColumnHeight(const double* col, int nh, double& h, int& orient,
                                           double monoTol) {
  h = 0.0;
  orient = 0;
  const int R = nh / 2;
  double lo = 0.0, hi = 0.0;
  for (int k = 0; k < R; ++k) {
    lo += col[k];
    hi += col[nh - 1 - k];
  }
  if (lo > hi)
    orient = 1;
  else if (hi > lo)
    orient = -1;
  else
    return false;

  int a = -1, b = -1;
  if (orient == 1) {
    for (int k = R; k >= 0; --k)
      if (hfIsFull(col[k])) {
        a = k;
        break;
      }
    for (int k = R; k < nh; ++k)
      if (hfIsEmpty(col[k])) {
        b = k;
        break;
      }
  } else {
    for (int k = R; k >= 0; --k)
      if (hfIsEmpty(col[k])) {
        a = k;
        break;
      }
    for (int k = R; k < nh; ++k)
      if (hfIsFull(col[k])) {
        b = k;
        break;
      }
  }
  if (a < 0 || b < 0)
    return false;

  double s = 0.0;
  for (int k = a; k <= b; ++k) {
    s += col[k];
    // orient = +1 => non-increasing; orient = -1 => non-decreasing.
    if (k < b && (col[k + 1] - col[k]) * orient > monoTol)
      return false;
  }
  h = (orient == 1) ? (s + static_cast<double>(a - R) - 0.5)
                    : (s - static_cast<double>(b - R) - 0.5);
  return true;
}

/// Mean-curvature-times-two from a 3x3 patch of heights, Han et al. eqs. (4)-(5) with
/// dx = dy = 1 (cell units).
///
/// @param h  nine heights, indexed `p + 3*q` with p, q in {0,1,2} standing for the transverse
///           offsets {-1, 0, +1}. Which of the two transverse axes is p and which is q does not
///           matter: eq. (5) is symmetric under the swap.
/// @return   kappa = 2H, in 1/cell.
KOKKOS_INLINE_FUNCTION double hfPatchKappa(const double h[9]) {
  const double hx = 0.5 * (h[2 + 3 * 1] - h[0 + 3 * 1]);
  const double hy = 0.5 * (h[1 + 3 * 2] - h[1 + 3 * 0]);
  const double hxx = h[2 + 3 * 1] - 2.0 * h[1 + 3 * 1] + h[0 + 3 * 1];
  const double hyy = h[1 + 3 * 2] - 2.0 * h[1 + 3 * 1] + h[1 + 3 * 0];
  const double hxy = 0.25 * (h[2 + 3 * 2] - h[0 + 3 * 2] - h[2 + 3 * 0] + h[0 + 3 * 0]);
  const double q = 1.0 + hx * hx + hy * hy;
  const double num = hxx + hyy + hxx * hy * hy + hyy * hx * hx - 2.0 * hxy * hx * hy;
  // kappa = 2H and H = -num / (2 q^{3/2}).
  return -num / (q * Kokkos::sqrt(q));
}

// ---------------------------------------------------------------------------------------------
// small linear algebra + geometry shared by the fit
// ---------------------------------------------------------------------------------------------

/// Right-handed orthonormal frame `(t1, t2, n)` from a unit vector `n`. Deterministic (the seed
/// axis is the one with the smallest |n_a|), so the frame — and therefore every fitted coefficient
/// — is a pure function of n and reproduces bitwise on any backend and any decomposition.
KOKKOS_INLINE_FUNCTION void curvFrame(const double n[3], double t1[3], double t2[3]) {
  int a = 0;
  double m = Kokkos::fabs(n[0]);
  if (Kokkos::fabs(n[1]) < m) {
    m = Kokkos::fabs(n[1]);
    a = 1;
  }
  if (Kokkos::fabs(n[2]) < m)
    a = 2;
  double e[3] = {0.0, 0.0, 0.0};
  e[a] = 1.0;
  // t1 = normalize(e x n)
  t1[0] = e[1] * n[2] - e[2] * n[1];
  t1[1] = e[2] * n[0] - e[0] * n[2];
  t1[2] = e[0] * n[1] - e[1] * n[0];
  const double s = Kokkos::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
  const double inv = (s > 0.0) ? 1.0 / s : 0.0;
  t1[0] *= inv;
  t1[1] *= inv;
  t1[2] *= inv;
  // t2 = n x t1
  t2[0] = n[1] * t1[2] - n[2] * t1[1];
  t2[1] = n[2] * t1[0] - n[0] * t1[2];
  t2[2] = n[0] * t1[1] - n[1] * t1[0];
}

/// Wendland C2 radial basis function, Han et al. eq. (8):
/// `(1 + 4 r/d)(1 - r/d)^4` on `0 <= r <= d`, zero beyond.
KOKKOS_INLINE_FUNCTION double wendlandWeight(double r, double d) {
  if (!(d > 0.0))
    return 1.0;
  const double q = r / d;
  if (q >= 1.0)
    return 0.0;
  const double t = 1.0 - q;
  const double t2 = t * t;
  return (1.0 + 4.0 * q) * t2 * t2;
}

/// Mean curvature (times two) of the paraboloid `z = a0 + a1 x + a2 y + a3 x^2 + a4 x y + a5 y^2`
/// at the origin — Han et al. eq. (10), doubled. Sign convention: the paraboloid's z axis is the
/// interface normal pointing into the GAS, so a liquid sphere of radius R gives +2/R.
KOKKOS_INLINE_FUNCTION double paraboloidKappa(const double a[6]) {
  const double q = 1.0 + a[1] * a[1] + a[2] * a[2];
  const double num = a[5] * a[1] * a[1] - a[4] * a[1] * a[2] + a[3] * a[2] * a[2] + a[3] + a[5];
  return -2.0 * num / (q * Kokkos::sqrt(q));
}

/// Solve the symmetric positive-semidefinite `n x n` system `A x = b` (n <= 6) by LDL^T with
/// Jacobi (diagonal) pre-scaling and a relative pivot guard. Returns false — leaving `x`
/// untouched — when the system is rank deficient, which is the caller's cue to drop to a smaller
/// model rather than to emit a garbage curvature.
///
/// `A` is taken by value in the caller's local storage; this routine overwrites its own copy only.
KOKKOS_INLINE_FUNCTION bool curvSolveSym(const double Ain[6][6], const double bin[6], int n,
                                         double x[6], double pivTol) {
  double d[6];
  for (int i = 0; i < n; ++i) {
    if (!(Ain[i][i] > 0.0))
      return false;
    d[i] = 1.0 / Kokkos::sqrt(Ain[i][i]);
  }
  double A[6][6], b[6];
  for (int i = 0; i < n; ++i) {
    b[i] = bin[i] * d[i];
    for (int j = 0; j < n; ++j)
      A[i][j] = Ain[i][j] * d[i] * d[j];
  }
  // LDL^T in place (A[i][i] holds D, A[i][j] j<i holds L).
  for (int i = 0; i < n; ++i) {
    double s = A[i][i];
    for (int k = 0; k < i; ++k)
      s -= A[i][k] * A[i][k] * A[k][k];
    if (!(s > pivTol))  // NaN-safe: a NaN pivot fails the test
      return false;
    A[i][i] = s;
    for (int j = i + 1; j < n; ++j) {
      double t = A[j][i];
      for (int k = 0; k < i; ++k)
        t -= A[j][k] * A[i][k] * A[k][k];
      A[j][i] = t / s;
    }
  }
  // forward / diagonal / backward
  double y[6];
  for (int i = 0; i < n; ++i) {
    double s = b[i];
    for (int k = 0; k < i; ++k)
      s -= A[i][k] * y[k];
    y[i] = s;
  }
  for (int i = 0; i < n; ++i)
    y[i] /= A[i][i];
  for (int i = n - 1; i >= 0; --i) {
    double s = y[i];
    for (int k = i + 1; k < n; ++k)
      s -= A[k][i] * x[k];
    x[i] = s;
  }
  for (int i = 0; i < n; ++i)
    x[i] *= d[i];
  return true;
}

// ---------------------------------------------------------------------------------------------
// the PLIC polygon: plane ^ unit cube
// ---------------------------------------------------------------------------------------------

/// Vertices of the polygon `{x in [0,1]^3 : m . x = alpha}`, ordered around the polygon.
///
/// A plane cuts a cube in at most a hexagon, so `v` needs 6 slots (8 given, for the degenerate
/// corner-coincidence cases before de-duplication). Edge walk + angular sort in the plane's own
/// 2-D frame — the standard construction, and the only piece of polygon machinery this rung needs
/// (the PV normal equations are line integrals around exactly this polygon, Han et al. eq. (14)).
/// `plic.hpp` deliberately has no clipper; this one is confined to the curvature fallback.
///
/// @return the vertex count (0 when the plane misses the cube or `m` is degenerate).
KOKKOS_INLINE_FUNCTION int plicPolygon(double mx, double my, double mz, double alpha,
                                       double v[8][3]) {
  const double m[3] = {mx, my, mz};
  const double nrm2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
  if (!(nrm2 > 0.0))
    return 0;

  int nv = 0;
  for (int q = 0; q < 8 && nv < 8; ++q) {
    const double A[3] = {static_cast<double>(q & 1), static_cast<double>((q >> 1) & 1),
                         static_cast<double>((q >> 2) & 1)};
    const double fA = m[0] * A[0] + m[1] * A[1] + m[2] * A[2] - alpha;
    for (int a = 0; a < 3 && nv < 8; ++a) {
      if ((q >> a) & 1)
        continue;  // walk each edge once, from its low corner
      double B[3] = {A[0], A[1], A[2]};
      B[a] = 1.0;
      const double fB = fA + m[a];
      // Straddle test with a half-open convention so a corner exactly on the plane is counted by
      // exactly one of its edges.
      const bool cross = (fA <= 0.0 && fB > 0.0) || (fB <= 0.0 && fA > 0.0);
      if (!cross)
        continue;
      const double den = fA - fB;
      const double t = (den != 0.0) ? fA / den : 0.0;
      double p[3] = {A[0], A[1], A[2]};
      p[a] = A[a] + t * (B[a] - A[a]);
      // de-duplicate (a corner hit is found by up to three edges)
      bool dup = false;
      for (int k = 0; k < nv; ++k) {
        const double dx = p[0] - v[k][0], dy = p[1] - v[k][1], dz = p[2] - v[k][2];
        if (dx * dx + dy * dy + dz * dz < 1e-24) {
          dup = true;
          break;
        }
      }
      if (dup)
        continue;
      v[nv][0] = p[0];
      v[nv][1] = p[1];
      v[nv][2] = p[2];
      ++nv;
    }
  }
  if (nv < 3)
    return nv;

  // Angular sort about the centroid in the plane's own frame.
  const double invn = 1.0 / Kokkos::sqrt(nrm2);
  const double nh[3] = {m[0] * invn, m[1] * invn, m[2] * invn};
  double e1[3], e2[3];
  curvFrame(nh, e1, e2);
  double cx = 0.0, cy = 0.0, cz = 0.0;
  for (int k = 0; k < nv; ++k) {
    cx += v[k][0];
    cy += v[k][1];
    cz += v[k][2];
  }
  const double inv = 1.0 / static_cast<double>(nv);
  cx *= inv;
  cy *= inv;
  cz *= inv;
  double ang[8];
  for (int k = 0; k < nv; ++k) {
    const double dx = v[k][0] - cx, dy = v[k][1] - cy, dz = v[k][2] - cz;
    ang[k] =
        Kokkos::atan2(dx * e2[0] + dy * e2[1] + dz * e2[2], dx * e1[0] + dy * e1[1] + dz * e1[2]);
  }
  for (int i = 1; i < nv; ++i) {  // insertion sort, nv <= 6
    const double a0 = ang[i];
    const double p0 = v[i][0], p1 = v[i][1], p2 = v[i][2];
    int j = i - 1;
    while (j >= 0 && ang[j] > a0) {
      ang[j + 1] = ang[j];
      v[j + 1][0] = v[j][0];
      v[j + 1][1] = v[j][1];
      v[j + 1][2] = v[j][2];
      --j;
    }
    ang[j + 1] = a0;
    v[j + 1][0] = p0;
    v[j + 1][1] = p1;
    v[j + 1][2] = p2;
  }
  return nv;
}

/// Area and centroid of a planar polygon in 3-D (fan triangulation from `v[0]`).
KOKKOS_INLINE_FUNCTION void polygonAreaCentroid(const double v[8][3], int nv, double ctr[3],
                                                double& area) {
  ctr[0] = ctr[1] = ctr[2] = 0.0;
  area = 0.0;
  if (nv < 3)
    return;
  for (int k = 1; k + 1 < nv; ++k) {
    const double a1[3] = {v[k][0] - v[0][0], v[k][1] - v[0][1], v[k][2] - v[0][2]};
    const double a2[3] = {v[k + 1][0] - v[0][0], v[k + 1][1] - v[0][1], v[k + 1][2] - v[0][2]};
    const double cx = a1[1] * a2[2] - a1[2] * a2[1];
    const double cy = a1[2] * a2[0] - a1[0] * a2[2];
    const double cz = a1[0] * a2[1] - a1[1] * a2[0];
    const double t = 0.5 * Kokkos::sqrt(cx * cx + cy * cy + cz * cz);
    area += t;
    ctr[0] += t * (v[0][0] + v[k][0] + v[k + 1][0]) / 3.0;
    ctr[1] += t * (v[0][1] + v[k][1] + v[k + 1][1]) / 3.0;
    ctr[2] += t * (v[0][2] + v[k][2] + v[k + 1][2]) / 3.0;
  }
  if (area > 0.0) {
    const double inv = 1.0 / area;
    ctr[0] *= inv;
    ctr[1] *= inv;
    ctr[2] *= inv;
  }
}

/// The six monomial integrals `s = ( |G|, int x, int y, int x^2, int xy, int y^2 )` over the
/// planar polygon `xy[0..nv)`, by Green's theorem — Han et al. eqs. (14a)-(14f).
///
/// **DEVIATION recorded.** These are re-derived here rather than transcribed, using
/// `int_G x^a y^b dA = oint x^{a+1} y^b/(a+1) dy` on each edge, because the published (14f) does
/// not reproduce the elementary case: for the CCW unit square it returns `-1/3` where
/// `int y^2 dA = +1/3`, i.e. its edge factor is `(x_v - x_{v+1})` and not the printed
/// `(x_{v+1} - x_v)`. (14a)-(14e) do check out — (14d) agrees term for term with the derivation
/// after the identity `(x0+x1)(x0^2+x1^2) = x0^3 + x0^2 x1 + x0 x1^2 + x1^3`. Every form below was
/// verified against the unit square and, in the ctest, against a randomized triangulated oracle.
///
/// The whole PV assembly is quadratic in `s` (both `A_ij = sum w s_i s_j` and
/// `b_i = sum w s_i (b.s)`), so a globally flipped polygon orientation cancels exactly; the sign
/// convention below is nevertheless CCW-positive so `s[0]` is a readable area.
KOKKOS_INLINE_FUNCTION void polygonMoments2d(const double xy[8][2], int nv, double s[6]) {
  for (int i = 0; i < 6; ++i)
    s[i] = 0.0;
  if (nv < 3)
    return;
  for (int k = 0; k < nv; ++k) {
    const int k1 = (k + 1 == nv) ? 0 : k + 1;
    const double x0 = xy[k][0], y0 = xy[k][1];
    const double x1 = xy[k1][0], y1 = xy[k1][1];
    const double dx = x1 - x0, dy = y1 - y0;
    if (dy == 0.0)
      continue;
    s[0] += dy * 0.5 * (x0 + x1);
    s[1] += dy * (x0 * x0 + x0 * x1 + x1 * x1) / 6.0;
    s[2] += dy * (2.0 * x0 * y0 + x0 * y1 + x1 * y0 + 2.0 * x1 * y1) / 6.0;
    s[3] += dy * (x0 + x1) * (x0 * x0 + x1 * x1) / 12.0;
    // int_0^1 x^2 y dt, x = x0 + t dx, y = y0 + t dy
    s[4] += dy * 0.5 *
            (x0 * x0 * y0 + 0.5 * x0 * x0 * dy + x0 * dx * y0 + (2.0 / 3.0) * x0 * dx * dy +
             dx * dx * y0 / 3.0 + 0.25 * dx * dx * dy);
    // int_0^1 x y^2 dt
    s[5] += dy * (x0 * y0 * y0 + x0 * y0 * dy + x0 * dy * dy / 3.0 + 0.5 * dx * y0 * y0 +
                  (2.0 / 3.0) * dx * y0 * dy + 0.25 * dx * dy * dy);
  }
}

// ---------------------------------------------------------------------------------------------
// tier 2b — the mixed height function: a paraboloid through the interface positions that the
// consistent columns of ALL THREE directions do provide
// ---------------------------------------------------------------------------------------------

/// Accumulator for an ordinary weighted linear least-squares paraboloid through points.
///
/// This is Popinet's *generalised* height function and Basilisk's `height_curvature_fit`, with one
/// deliberate restriction: the points come only from the 3 x 3 patches of the three directions,
/// i.e. from cells at offsets `(+/-1, +/-1, +/-3)`, so the whole tier stays inside the colour
/// field's g = 3 halo. Basilisk gathers heights over the 3^3 neighbourhood instead, which needs
/// colour at +/-4 and therefore a wider halo — see the DEVIATION note in the file header.
///
/// Why it is a real tier and the direction cascade alone is not: **measured**, tier 2a fires on
/// 0.00-0.06 % of the interfacial cells of a sphere, because the direction with the largest |n_d|
/// is by construction the one whose columns are most likely to close, so when it fails the other
/// two almost always fail too. Tier 2b fails only when fewer than six of the 27 columns close,
/// which is a genuinely different and much rarer condition.
struct PtFit {
  double A[6][6];
  double b[6];
  int npt;
};

KOKKOS_INLINE_FUNCTION void ptFitInit(PtFit& f) {
  for (int i = 0; i < 6; ++i) {
    f.b[i] = 0.0;
    for (int j = 0; j < 6; ++j)
      f.A[i][j] = 0.0;
  }
  f.npt = 0;
}

/// Fold one interface point into the fit.
/// @param X    the point in target-centred cell units.
/// @param org  the fit origin (the target cell's PLIC centroid), same units.
/// @param dW   Wendland support width in cell units.
KOKKOS_INLINE_FUNCTION void ptFitAdd(PtFit& f, const double X[3], const double org[3],
                                     const double t1[3], const double t2[3], const double nn[3],
                                     double dW) {
  const double d[3] = {X[0] - org[0], X[1] - org[1], X[2] - org[2]};
  const double x = d[0] * t1[0] + d[1] * t1[1] + d[2] * t1[2];
  const double y = d[0] * t2[0] + d[1] * t2[1] + d[2] * t2[2];
  const double z = d[0] * nn[0] + d[1] * nn[1] + d[2] * nn[2];
  const double w = wendlandWeight(Kokkos::sqrt(x * x + y * y + z * z), dW);
  if (!(w > 0.0))
    return;
  const double phi[6] = {1.0, x, y, x * x, x * y, y * y};
  for (int i = 0; i < 6; ++i) {
    f.b[i] += w * phi[i] * z;
    for (int j = 0; j < 6; ++j)
      f.A[i][j] += w * phi[i] * phi[j];
  }
  ++f.npt;
}

/// Solve the point fit, with the same rank cascade as `pvFitSolve`.
KOKKOS_INLINE_FUNCTION bool ptFitSolve(const PtFit& f, double a[6], bool& red) {
  red = false;
  for (int i = 0; i < 6; ++i)
    a[i] = 0.0;
  if (f.npt >= 6 && curvSolveSym(f.A, f.b, 6, a, 1e-12))
    return true;
  if (f.npt < 3)
    return false;
  const int id[3] = {0, 3, 5};
  double Ar[6][6], br[6], ar[6];
  for (int i = 0; i < 3; ++i) {
    br[i] = f.b[id[i]];
    for (int j = 0; j < 3; ++j)
      Ar[i][j] = f.A[id[i]][id[j]];
  }
  if (!curvSolveSym(Ar, br, 3, ar, 1e-12))
    return false;
  a[0] = ar[0];
  a[3] = ar[1];
  a[5] = ar[2];
  red = true;
  return true;
}

// ---------------------------------------------------------------------------------------------
// tier 3 — the PLIC-volumetric paraboloid fit (Jibben 2019 / Han 2024 "PV")
// ---------------------------------------------------------------------------------------------

/// Accumulator for the PV normal equations, Han et al. eq. (15). Small local state (48 doubles),
/// so a device thread carries it in the fallback kernel only.
struct PvFit {
  double A[6][6];
  double b[6];
  int npoly;
};

KOKKOS_INLINE_FUNCTION void pvFitInit(PvFit& f) {
  for (int i = 0; i < 6; ++i) {
    f.b[i] = 0.0;
    for (int j = 0; j < 6; ++j)
      f.A[i][j] = 0.0;
  }
  f.npoly = 0;
}

/// Fold one stencil cell's PLIC polygon into the fit.
///
/// @param mx,my,mz,alpha  the cell's PLIC plane in ITS OWN unit-cell coordinates (`plic.hpp`).
/// @param off             the cell's integer index offset from the target cell.
/// @param org             the fit's origin (the target cell's PLIC centroid) in target-centred
///                        cell units.
/// @param t1,t2,nn        the orthonormal frame; `nn` is the target's unit PLIC normal.
/// @param dW              Wendland support width in cell units.
/// @param cosMin          reject a polygon whose own normal makes an angle with `nn` whose cosine
///                        is below this. The projection onto the (x',y') plane degenerates as that
///                        cosine goes to zero (the plane's `z' = b0 + b1 x' + b2 y'` form blows up
///                        while its projected area vanishes), and a back-facing polygon would be
///                        integrated with the wrong sign of `z'`. Jibben's area projection
///                        `A (n.n_p)` decays to zero over the same range; this is the explicit
///                        version of the same guard.
/// @return true if the polygon contributed.
KOKKOS_INLINE_FUNCTION bool pvFitAdd(PvFit& f, double mx, double my, double mz, double alpha,
                                     const double off[3], const double org[3], const double t1[3],
                                     const double t2[3], const double nn[3], double dW,
                                     double cosMin) {
  const double n2 = mx * mx + my * my + mz * mz;
  if (!(n2 > 0.0))
    return false;
  const double invn = 1.0 / Kokkos::sqrt(n2);
  const double np[3] = {(mx * t1[0] + my * t1[1] + mz * t1[2]) * invn,
                        (mx * t2[0] + my * t2[1] + mz * t2[2]) * invn,
                        (mx * nn[0] + my * nn[1] + mz * nn[2]) * invn};
  if (!(np[2] > cosMin))
    return false;

  double v[8][3];
  const int nv = plicPolygon(mx, my, mz, alpha, v);
  if (nv < 3)
    return false;

  // cell-local [0,1]^3 -> target-centred cell units -> the fit frame
  double xy[8][2];
  double zc = 0.0;
  double px = 0.0, py = 0.0;
  for (int k = 0; k < nv; ++k) {
    const double X[3] = {off[0] + v[k][0] - 0.5 - org[0], off[1] + v[k][1] - 0.5 - org[1],
                         off[2] + v[k][2] - 0.5 - org[2]};
    xy[k][0] = X[0] * t1[0] + X[1] * t1[1] + X[2] * t1[2];
    xy[k][1] = X[0] * t2[0] + X[1] * t2[1] + X[2] * t2[2];
    const double z = X[0] * nn[0] + X[1] * nn[1] + X[2] * nn[2];
    px += xy[k][0];
    py += xy[k][1];
    zc += z;
  }
  const double invv = 1.0 / static_cast<double>(nv);
  px *= invv;
  py *= invv;
  zc *= invv;

  double s[6];
  polygonMoments2d(xy, nv, s);
  if (!(Kokkos::fabs(s[0]) > 1e-14))
    return false;

  // the polygon's own plane in the fit frame: z' = b0 + b1 x' + b2 y'
  const double b1 = -np[0] / np[2], b2 = -np[1] / np[2];
  const double b0 = zc - b1 * px - b2 * py;

  const double r = Kokkos::sqrt(px * px + py * py + zc * zc);
  const double w = wendlandWeight(r, dW);
  if (!(w > 0.0))
    return false;

  const double B = b0 * s[0] + b1 * s[1] + b2 * s[2];
  for (int i = 0; i < 6; ++i) {
    f.b[i] += w * s[i] * B;
    for (int j = 0; j < 6; ++j)
      f.A[i][j] += w * s[i] * s[j];
  }
  ++f.npoly;
  return true;
}

/// Solve the accumulated PV system.
///
/// `A` is a sum of `npoly` rank-1 outer products, so it cannot have rank 6 with fewer than 6
/// polygons and it is only generically full rank with more. On a rank-deficient system the fit
/// drops to the 3-parameter sub-model `{1, x'^2, y'^2}` — the exact minimiser of the SAME cost
/// function restricted to `a1 = a2 = a4 = 0`, which is the right restriction here because the frame
/// is already aligned with the interface normal at the origin, so the linear terms are the small
/// ones. Below three polygons there is nothing to fit and the caller must say so out loud.
///
/// @param red [out] true when the reduced model was used.
/// @return false when no fit was possible at all.
KOKKOS_INLINE_FUNCTION bool pvFitSolve(const PvFit& f, double a[6], bool& red) {
  red = false;
  for (int i = 0; i < 6; ++i)
    a[i] = 0.0;
  if (f.npoly >= 6 && curvSolveSym(f.A, f.b, 6, a, 1e-12))
    return true;
  if (f.npoly < 3)
    return false;
  // reduced {1, x^2, y^2} == indices 0, 3, 5 of the full system
  const int id[3] = {0, 3, 5};
  double Ar[6][6], br[6], ar[6];
  for (int i = 0; i < 3; ++i) {
    br[i] = f.b[id[i]];
    for (int j = 0; j < 3; ++j)
      Ar[i][j] = f.A[id[i]][id[j]];
  }
  if (!curvSolveSym(Ar, br, 3, ar, 1e-12))
    return false;
  a[0] = ar[0];
  a[1] = 0.0;
  a[2] = 0.0;
  a[3] = ar[1];
  a[4] = 0.0;
  a[5] = ar[2];
  red = true;
  return true;
}

}  // namespace peclet::core::vof

#endif  // PECLET_CORE_VOF_CURVATURE_HPP

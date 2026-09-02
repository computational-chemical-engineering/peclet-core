/// @file
/// @brief core — PLIC (piecewise-linear interface construction) geometric toolbox: the analytic
/// plane<->volume relation for a unit cell, interface-normal estimators, and slab truncation.
///
/// **Container-free by contract.** Every entry point here takes scalars and small local arrays
/// only — no `Kokkos::View`, no grid indexing, no halo/topology types. That is what made the
/// promotion to `peclet::core::vof` (done at WO-W0, see `suite/docs/VOF_PLAN.md` §11) a plain file
/// move: the structured solver, the AMR octree and the per-bubble block container share ONE copy.
/// Do not add container awareness to these signatures.
///
/// ## Convention
///
/// The cell is mapped to the unit cube [0,1]^3. A PLIC plane is
///
///     m . x = alpha ,   x in [0,1]^3
///
/// and the fluid (colour C = 1) side is the half-space `m . x < alpha`, i.e. **m points into the
/// gas**, m ~ -grad(C). The canonical normalization is the L1 one, `|mx| + |my| + |mz| = 1`
/// (Scardovelli & Zaleski's choice), and every normal returned here is L1-normalized. The
/// plane<->volume routines do NOT require it: they renormalize internally, so any nonzero m with a
/// consistently scaled alpha gives the same answer (the relation V(m, alpha) is invariant under
/// (m, alpha) -> (lambda m, lambda alpha)). That is what makes `plicSlabVolume` a two-line rescale
/// rather than a polyhedron clipper.
///
/// ## Algorithm sources (followed, not approximated)
///
/// - Forward `plicVolume` — Scardovelli & Zaleski, JCP 164:228 (2000), in the branch-reduced GPU
///   form of Lehmann & Gekle, *Computation* 10:21 (2022) [arXiv:2006.12838], their eq. (11) and
///   Listing 1 (`plic_cube_inverse`): mirror to the positive octant, sort |m| ascending, fold
///   V > 1/2 onto V < 1/2, then five piecewise-cubic cases.
/// - Inverse `plicAlpha` — the analytic SZ inversion in Lehmann & Gekle's optimized L1 form,
///   their Listing 4 (`plic_cube_reduced`): closed forms for cases (5), (2), (1) and a single
///   trigonometric branch covering the two cubic cases (3) and (4).
/// - `youngsNormal` / `mycNormal` — Youngs' 27-point weighted gradient and the Mixed Youngs-Centred
///   scheme of Aulisa, Manservisi, Scardovelli & Zaleski, JCP 225:2301 (2007), transcribed from the
///   reference implementation `basilisk/src/myc.h` (authored by Scardovelli himself).
///
/// The two cubic branches were re-derived independently when this file was written (substitute
/// d = c + u into the case (3)/(4) volume polynomial, obtain the depressed cubic
/// `u^3 - 3 t^2 u + (a + 3 b c - 2 c^3) = 0` with `t = sqrt(c^2 - b)`, and solve it with
/// `u = -2 t sin(asin((c^3 - a/2 - 3 b c/2)/t^3)/3)` via `sin 3psi = 3 sin psi - 4 sin^3 psi`);
/// the derivation reproduces Listing 4 term for term, so the transcription is verified and not
/// merely copied.
///
/// **Layer L1 (`VOF_PLAN.md` §11): promoted from `flow/src/vof/` to `core` (WO-W0, 2026-09-02).**
/// These are the container-free VoF math kernels — one copy shared by every container: flow's
/// structured colour field, the per-bubble block container (Part III) and the future AMR path.
/// `flow/src/vof/plic.hpp` is now a thin include + a using-directive, so every existing
/// `peclet::flow::vof::` spelling keeps working unchanged.
#ifndef PECLET_CORE_VOF_PLIC_HPP
#define PECLET_CORE_VOF_PLIC_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace peclet::core::vof {

// ---------------------------------------------------------------------------------------------
// small local helpers
// ---------------------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION double plicSq(double x) {
  return x * x;
}
KOKKOS_INLINE_FUNCTION double plicCube(double x) {
  return x * x * x;
}

/// L1-normalize a normal in place. Returns the L1 norm before normalization (0 => untouched).
KOKKOS_INLINE_FUNCTION double plicNormalizeL1(double m[3]) {
  const double s = Kokkos::fabs(m[0]) + Kokkos::fabs(m[1]) + Kokkos::fabs(m[2]);
  if (s > 0.0) {
    const double inv = 1.0 / s;
    m[0] *= inv;
    m[1] *= inv;
    m[2] *= inv;
  }
  return s;
}

// ---------------------------------------------------------------------------------------------
// (1) forward: plane -> volume
// ---------------------------------------------------------------------------------------------

/// Reduced-symmetry forward relation. Inputs are already mirrored, sorted and L1-normalized:
/// `0 <= n1 <= n2 <= n3`, `n1 + n2 + n3 = 1`, and `0 <= w <= 1/2` is the plane offset measured
/// from the origin corner. Returns the truncated volume in [0, 1/2].
///
/// Lehmann & Gekle (2022) eq. (11) / Listing 1. Case order (5) -> (1) -> (2) -> (3|4) is the
/// paper's: it keeps the divisions by n1 (which may be zero for a 1D/2D normal) behind case
/// conditions that are unreachable when n1 = 0.
KOKKOS_INLINE_FUNCTION double plicVolumeReduced(double w, double n1, double n2, double n3) {
  const double n12 = n1 + n2;
  // case (5) — the slab regime (the plane exits through both faces normal to axis 3), no division
  // by n1/n2. NOTE the condition is `n1+n2 <= w`, NOT Lehmann & Gekle's Listing 1 shortcut
  // `min(n1+n2,n3) <= w`: eq. (11) defines case (5) as "the remaining free sector mutually excluded
  // by the other four cases", which is the interval [n1+n2, n3] and is empty unless n3 >= n1+n2.
  // Hoisting the test to the front (as the listing does, for speed) makes `min(n1+n2,n3) <= w`
  // additionally fire at the single point w == n3 when n3 < n1+n2, where the correct case is (3):
  // e.g. m = (1/3,1/3,1/3), alpha = 1/3 gives 0 instead of the exact 1/6. Using the true case-(5)
  // interval keeps the front position (and its n1 = 0 protection) without the boundary defect.
  if (n12 <= w && w <= n3)
    return (w - 0.5 * n12) / n3;
  if (w < n1)  // case (1) — corner tetrahedron (implies n1 > 0)
    return plicCube(w) / (6.0 * n1 * n2 * n3);
  if (w <= n2)  // case (2) — one corner clipped; the n1 factor cancels, so n1 = 0 is safe
    return (3.0 * w * (w - n1) + plicSq(n1)) / (6.0 * n2 * n3);
  // cases (3) and (4) — two or three corners clipped; fdim() merges the two branches. Reachable
  // only with n1 > 0 (n1 = 0 is fully covered by cases (2) and (5) above).
  const double t3 = Kokkos::fmax(w - n3, 0.0);
  return (plicCube(w) - plicCube(w - n1) - plicCube(w - n2) - plicCube(t3)) / (6.0 * n1 * n2 * n3);
}

/// Volume fraction of the unit cell [0,1]^3 lying on the fluid side `m . x < alpha`.
///
/// @param mx,my,mz  interface normal (any scale; canonical form has |mx|+|my|+|mz| = 1)
/// @param alpha     plane offset in the same scale as m
/// @return          V in [0,1]
KOKKOS_INLINE_FUNCTION double plicVolume(double mx, double my, double mz, double alpha) {
  const double ax = Kokkos::fabs(mx), ay = Kokkos::fabs(my), az = Kokkos::fabs(mz);
  const double s = ax + ay + az;
  if (!(s > 0.0))
    return alpha > 0.0 ? 1.0 : 0.0;  // degenerate normal: the half-space is trivial

  // Mirror the negative components into the positive octant. Substituting x_i -> 1 - x_i turns
  // m_i x_i into m_i - m_i x_i', so the offset picks up |m_i| for every negative component.
  double beta = alpha;
  if (mx < 0.0)
    beta += ax;
  if (my < 0.0)
    beta += ay;
  if (mz < 0.0)
    beta += az;

  const double a1 = beta / s;  // offset in the L1-normalized, positive-octant frame
  if (a1 <= 0.0)
    return 0.0;
  if (a1 >= 1.0)
    return 1.0;

  // Sort ascending. n2 from the L1 identity (never from a third min/max) so n1+n2+n3 = 1 holds.
  const double n1 = Kokkos::fmin(Kokkos::fmin(ax, ay), az) / s;
  const double n3 = Kokkos::fmax(Kokkos::fmax(ax, ay), az) / s;
  const double n2 = Kokkos::fmax(1.0 - n1 - n3, 0.0);

  // V(alpha) + V(1-alpha) = 1 (point reflection x -> 1-x), so only w = min(a1, 1-a1) is computed.
  const double v = plicVolumeReduced(Kokkos::fmin(a1, 1.0 - a1), n1, n2, n3);
  return (a1 <= 0.5) ? v : 1.0 - v;
}

// ---------------------------------------------------------------------------------------------
// (2) inverse: volume -> plane
// ---------------------------------------------------------------------------------------------

/// Reduced-symmetry analytic inverse of `plicVolumeReduced`. `0 <= n1 <= n2 <= n3`,
/// `n1 + n2 + n3 = 1`, `0 <= V <= 1/2`; returns the offset w in [0, 1/2].
///
/// Lehmann & Gekle (2022) Listing 4 — the Scardovelli-Zaleski analytic solution in L1 form,
/// branch-reduced. Case order (5) -> (2) -> (1) -> (3|4): every division by n2 or n1 sits behind
/// a condition that already excluded the corresponding degenerate normal.
KOKKOS_INLINE_FUNCTION double plicAlphaReduced(double v, double n1, double n2, double n3) {
  const double n12 = n1 + n2, n3v = n3 * v;
  if (n12 <= 2.0 * n3v)
    return n3v + 0.5 * n12;  // case (5); also catches n1 = n2 = 0 (1D normal)

  // past case (5), n2 > 0 (n2 = 0 would force n1 = 0 and hence n12 = 0 <= 2 n3 V)
  const double sqn1 = plicSq(n1), n26 = 6.0 * n2, v1 = sqn1 / n26;
  if (v1 <= n3v && n3v < v1 + 0.5 * (n2 - n1))  // case (2); also catches n1 = 0 (2D normal)
    return 0.5 * (n1 + Kokkos::sqrt(sqn1 + 8.0 * n2 * (n3v - v1)));

  const double v6 = n1 * n26 * n3v;  // = 6 n1 n2 n3 V
  if (n3v < v1)
    return Kokkos::cbrt(v6);  // case (1)

  // past case (2), n1 > 0. v3 is n3 * V at the (3)|(4) boundary w = n3 (or at the case (5)
  // boundary w = n1+n2 when the plane can never reach w = n3).
  const double v3 = (n3 < n12) ? (plicSq(n3) * (3.0 * n12 - n3) + sqn1 * (n1 - 3.0 * n3) +
                                  plicSq(n2) * (n2 - 3.0 * n3)) /
                                     (n1 * n26)
                               : 0.5 * n12;
  const double sqn12 = sqn1 + plicSq(n2), v6cb = v6 - plicCube(n1) - plicCube(n2);
  const bool case3 = (n3v < v3);
  // Both cubic cases reduce to  w^3 - 3 c w^2 + 3 b w + a = 0 :
  const double a = case3 ? v6cb : 0.5 * (v6cb - plicCube(n3));
  const double b = case3 ? sqn12 : 0.5 * (sqn12 + plicSq(n3));
  const double c = case3 ? n12 : 0.5;
  // c^2 - b > 0 is guaranteed here: 2 n1 n2 > 0 in case (3); 1/4 - (n1^2+n2^2+n3^2)/2 > 0 in case
  // (4), which is reachable only when n3 < n1+n2, i.e. n3 < 1/2.
  const double t = Kokkos::sqrt(c * c - b);
  // Casus irreducibilis: three real roots, |arg| <= 1 analytically. The clamp is a floating-point
  // guard on the case boundaries only (an out-of-range arg would otherwise return NaN).
  const double arg =
      Kokkos::fmin(Kokkos::fmax((plicCube(c) - 0.5 * a - 1.5 * b * c) / plicCube(t), -1.0), 1.0);
  return c - 2.0 * t * Kokkos::sin((1.0 / 3.0) * Kokkos::asin(arg));
}

/// Plane offset alpha such that `plicVolume(mx,my,mz,alpha) == V`, exactly (analytic inverse).
///
/// @param mx,my,mz  interface normal (any scale; alpha comes back in the same scale)
/// @param v         target volume fraction, clamped to [0,1]
KOKKOS_INLINE_FUNCTION double plicAlpha(double mx, double my, double mz, double v) {
  const double ax = Kokkos::fabs(mx), ay = Kokkos::fabs(my), az = Kokkos::fabs(mz);
  const double s = ax + ay + az;
  if (!(s > 0.0))
    return 0.0;

  const double n1 = Kokkos::fmin(Kokkos::fmin(ax, ay), az) / s;
  const double n3 = Kokkos::fmax(Kokkos::fmax(ax, ay), az) / s;
  const double n2 = Kokkos::fmax(1.0 - n1 - n3, 0.0);

  const double vc = Kokkos::fmin(Kokkos::fmax(v, 0.0), 1.0);
  const double w = plicAlphaReduced(Kokkos::fmin(vc, 1.0 - vc), n1, n2, n3);
  const double a1 = (vc <= 0.5) ? w : 1.0 - w;  // undo the V -> 1-V fold

  double alpha = s * a1;  // undo the L1 normalization
  if (mx < 0.0)
    alpha -= ax;  // undo the octant mirroring
  if (my < 0.0)
    alpha -= ay;
  if (mz < 0.0)
    alpha -= az;
  return alpha;
}

// ---------------------------------------------------------------------------------------------
// (5) slab truncation — the geometric face flux
// ---------------------------------------------------------------------------------------------

/// Fluid volume of the PLIC polyhedron inside the axis-aligned slab `a <= x_dir <= b`, expressed
/// as a fraction of the WHOLE cell (so `plicSlabVolume(..., 0, 1) == plicVolume(...)`).
///
/// No clipping code: the slab is the unit cell of a rescaled coordinate `x_dir = a + (b-a) xi`,
/// under which the plane becomes `m' . y = alpha'` with `m'_dir = m_dir (b-a)`, the other
/// components unchanged, and `alpha' = alpha - m_dir a`. The result is `(b-a) * plicVolume(m')`.
///
/// @param dir  0 = x, 1 = y, 2 = z
KOKKOS_INLINE_FUNCTION double plicSlabVolume(double mx, double my, double mz, double alpha, int dir,
                                             double a, double b) {
  const double len = b - a;
  if (!(len > 0.0))
    return 0.0;
  double m[3] = {mx, my, mz};
  const double md = m[dir];
  m[dir] = md * len;
  return len * plicVolume(m[0], m[1], m[2], alpha - md * a);
}

/// Fluid volume of the PLIC polyhedron inside the axis-aligned BOX
/// `[a0,b0] x [a1,b1] x [a2,b2]`, as a fraction of the WHOLE cell. The three-axis generalization of
/// `plicSlabVolume` and, like it, a pure coordinate rescale rather than a polyhedron clipper: under
/// `x_k = a_k + (b_k - a_k) xi_k` the plane becomes `m'.xi = alpha'` with `m'_k = m_k (b_k - a_k)`
/// and `alpha' = alpha - sum_k m_k a_k`, so the answer is `prod_k (b_k - a_k) * plicVolume(m')`.
///
/// Added at rung V2b (WO-K): the momentum control volume of component `e` is shifted half a cell in
/// `e`, so its flux slab for a sweep in a TRANSVERSE direction `d` is a box that is half a cell
/// wide in `e` and `|a|` wide in `d` — two axes at once, which `plicSlabVolume` cannot express.
/// `plicBoxVolume(..., 0,1, 0,1, 0,1) == plicVolume(...)` bitwise, and the single-axis restriction
/// reproduces `plicSlabVolume` bitwise (both gated in `tests/kokkos/test_vof_plic.cpp`).
KOKKOS_INLINE_FUNCTION double plicBoxVolume(double mx, double my, double mz, double alpha,
                                            double a0, double b0, double a1, double b1, double a2,
                                            double b2) {
  const double len[3] = {b0 - a0, b1 - a1, b2 - a2};
  if (!(len[0] > 0.0) || !(len[1] > 0.0) || !(len[2] > 0.0))
    return 0.0;
  const double m[3] = {mx, my, mz};
  const double off = alpha - (m[0] * a0 + m[1] * a1 + m[2] * a2);
  return len[0] * len[1] * len[2] * plicVolume(m[0] * len[0], m[1] * len[1], m[2] * len[2], off);
}

/// Donor-cell geometric flux volume: the fluid volume swept out of the slab `[0, f]` along `dir`,
/// as a fraction of the whole cell. `f = 1` reproduces `plicVolume` bitwise.
KOKKOS_INLINE_FUNCTION double faceFluxVolume(double mx, double my, double mz, double alpha, int dir,
                                             double f) {
  return plicSlabVolume(mx, my, mz, alpha, dir, 0.0, f);
}

// ---------------------------------------------------------------------------------------------
// (3)/(4) interface normals from a 3x3x3 colour stencil
// ---------------------------------------------------------------------------------------------

/// Stencil index: x-fastest, offsets (i-1, j-1, k-1) with i,j,k in {0,1,2}
/// (`suite/docs/CONVENTIONS.md`). `c[plicSt(1,1,1)]` is the centre cell.
KOKKOS_INLINE_FUNCTION int plicSt(int i, int j, int k) {
  return i + 3 * j + 9 * k;
}

#define PECLET_VOF_C(dx, dy, dz) c[plicSt(1 + (dx), 1 + (dy), 1 + (dz))]

/// Youngs' normal: the 27-point weighted gradient (corner 1, edge 2, face 4 on each of the two
/// opposing planes), L1-normalized, pointing into the gas (m ~ -grad C). Degenerate (zero
/// gradient) stencils return (1,0,0).
KOKKOS_INLINE_FUNCTION void youngsNormal(const double c[27], double m[3]) {
  double m1, m2;

  m1 = PECLET_VOF_C(-1, -1, -1) + PECLET_VOF_C(-1, 1, -1) + PECLET_VOF_C(-1, -1, 1) +
       PECLET_VOF_C(-1, 1, 1) +
       2.0 * (PECLET_VOF_C(-1, -1, 0) + PECLET_VOF_C(-1, 1, 0) + PECLET_VOF_C(-1, 0, -1) +
              PECLET_VOF_C(-1, 0, 1)) +
       4.0 * PECLET_VOF_C(-1, 0, 0);
  m2 = PECLET_VOF_C(1, -1, -1) + PECLET_VOF_C(1, 1, -1) + PECLET_VOF_C(1, -1, 1) +
       PECLET_VOF_C(1, 1, 1) +
       2.0 * (PECLET_VOF_C(1, -1, 0) + PECLET_VOF_C(1, 1, 0) + PECLET_VOF_C(1, 0, -1) +
              PECLET_VOF_C(1, 0, 1)) +
       4.0 * PECLET_VOF_C(1, 0, 0);
  m[0] = m1 - m2;

  m1 = PECLET_VOF_C(-1, -1, -1) + PECLET_VOF_C(-1, -1, 1) + PECLET_VOF_C(1, -1, -1) +
       PECLET_VOF_C(1, -1, 1) +
       2.0 * (PECLET_VOF_C(-1, -1, 0) + PECLET_VOF_C(1, -1, 0) + PECLET_VOF_C(0, -1, -1) +
              PECLET_VOF_C(0, -1, 1)) +
       4.0 * PECLET_VOF_C(0, -1, 0);
  m2 = PECLET_VOF_C(-1, 1, -1) + PECLET_VOF_C(-1, 1, 1) + PECLET_VOF_C(1, 1, -1) +
       PECLET_VOF_C(1, 1, 1) +
       2.0 * (PECLET_VOF_C(-1, 1, 0) + PECLET_VOF_C(1, 1, 0) + PECLET_VOF_C(0, 1, -1) +
              PECLET_VOF_C(0, 1, 1)) +
       4.0 * PECLET_VOF_C(0, 1, 0);
  m[1] = m1 - m2;

  m1 = PECLET_VOF_C(-1, -1, -1) + PECLET_VOF_C(-1, 1, -1) + PECLET_VOF_C(1, -1, -1) +
       PECLET_VOF_C(1, 1, -1) +
       2.0 * (PECLET_VOF_C(-1, 0, -1) + PECLET_VOF_C(1, 0, -1) + PECLET_VOF_C(0, -1, -1) +
              PECLET_VOF_C(0, 1, -1)) +
       4.0 * PECLET_VOF_C(0, 0, -1);
  m2 = PECLET_VOF_C(-1, -1, 1) + PECLET_VOF_C(-1, 1, 1) + PECLET_VOF_C(1, -1, 1) +
       PECLET_VOF_C(1, 1, 1) +
       2.0 * (PECLET_VOF_C(-1, 0, 1) + PECLET_VOF_C(1, 0, 1) + PECLET_VOF_C(0, -1, 1) +
              PECLET_VOF_C(0, 1, 1)) +
       4.0 * PECLET_VOF_C(0, 0, 1);
  m[2] = m1 - m2;

  if (plicNormalizeL1(m) < 1e-30) {
    m[0] = 1.0;
    m[1] = 0.0;
    m[2] = 0.0;
  }
}

/// MYC — Mixed Youngs-Centred normal (Aulisa et al. 2007). Three "centred column" candidates (one
/// per axis: the plane written as sgn(m_d) x_d = ... + alpha, with the transverse components taken
/// from central differences of the 3-cell column sums) plus the Youngs candidate. The centred
/// candidate with the largest dominant component |m[d][d]| is selected first; Youngs replaces it
/// when that dominant component exceeds Youngs' own largest component — the signature of a
/// saturated (clipped) column, where the height difference under-reports the true slope and the
/// centred estimate is artificially flat.
///
/// Transcribed from `basilisk/src/myc.h`. The returned normal is L1-normalized and points into the
/// gas. As noted there, the centre cell never enters either estimator, so an isolated droplet gives
/// a degenerate stencil; that case returns (1,0,0).
KOKKOS_INLINE_FUNCTION void mycNormal(const double c[27], double m[3]) {
  double m1, m2, mm[4][3], t0, t1, t2;
  int cn;

  // plane as sgn(mx) X = m01 Y + m02 Z + alpha
  m1 = PECLET_VOF_C(-1, 0, -1) + PECLET_VOF_C(-1, 0, 1) + PECLET_VOF_C(-1, -1, 0) +
       PECLET_VOF_C(-1, 1, 0) + PECLET_VOF_C(-1, 0, 0);
  m2 = PECLET_VOF_C(1, 0, -1) + PECLET_VOF_C(1, 0, 1) + PECLET_VOF_C(1, -1, 0) +
       PECLET_VOF_C(1, 1, 0) + PECLET_VOF_C(1, 0, 0);
  mm[0][0] = m1 > m2 ? 1.0 : -1.0;

  m1 = PECLET_VOF_C(-1, -1, 0) + PECLET_VOF_C(1, -1, 0) + PECLET_VOF_C(0, -1, 0);
  m2 = PECLET_VOF_C(-1, 1, 0) + PECLET_VOF_C(1, 1, 0) + PECLET_VOF_C(0, 1, 0);
  mm[0][1] = 0.5 * (m1 - m2);

  m1 = PECLET_VOF_C(-1, 0, -1) + PECLET_VOF_C(1, 0, -1) + PECLET_VOF_C(0, 0, -1);
  m2 = PECLET_VOF_C(-1, 0, 1) + PECLET_VOF_C(1, 0, 1) + PECLET_VOF_C(0, 0, 1);
  mm[0][2] = 0.5 * (m1 - m2);

  // plane as sgn(my) Y = m10 X + m12 Z + alpha
  m1 = PECLET_VOF_C(-1, -1, 0) + PECLET_VOF_C(-1, 1, 0) + PECLET_VOF_C(-1, 0, 0);
  m2 = PECLET_VOF_C(1, -1, 0) + PECLET_VOF_C(1, 1, 0) + PECLET_VOF_C(1, 0, 0);
  mm[1][0] = 0.5 * (m1 - m2);

  m1 = PECLET_VOF_C(0, -1, -1) + PECLET_VOF_C(0, -1, 1) + PECLET_VOF_C(1, -1, 0) +
       PECLET_VOF_C(-1, -1, 0) + PECLET_VOF_C(0, -1, 0);
  m2 = PECLET_VOF_C(0, 1, -1) + PECLET_VOF_C(0, 1, 1) + PECLET_VOF_C(1, 1, 0) +
       PECLET_VOF_C(-1, 1, 0) + PECLET_VOF_C(0, 1, 0);
  mm[1][1] = m1 > m2 ? 1.0 : -1.0;

  m1 = PECLET_VOF_C(0, -1, -1) + PECLET_VOF_C(0, 0, -1) + PECLET_VOF_C(0, 1, -1);
  m2 = PECLET_VOF_C(0, -1, 1) + PECLET_VOF_C(0, 0, 1) + PECLET_VOF_C(0, 1, 1);
  mm[1][2] = 0.5 * (m1 - m2);

  // plane as sgn(mz) Z = m20 X + m21 Y + alpha
  m1 = PECLET_VOF_C(-1, 0, -1) + PECLET_VOF_C(-1, 0, 1) + PECLET_VOF_C(-1, 0, 0);
  m2 = PECLET_VOF_C(1, 0, -1) + PECLET_VOF_C(1, 0, 1) + PECLET_VOF_C(1, 0, 0);
  mm[2][0] = 0.5 * (m1 - m2);

  m1 = PECLET_VOF_C(0, -1, -1) + PECLET_VOF_C(0, -1, 1) + PECLET_VOF_C(0, -1, 0);
  m2 = PECLET_VOF_C(0, 1, -1) + PECLET_VOF_C(0, 1, 1) + PECLET_VOF_C(0, 1, 0);
  mm[2][1] = 0.5 * (m1 - m2);

  m1 = PECLET_VOF_C(-1, 0, -1) + PECLET_VOF_C(1, 0, -1) + PECLET_VOF_C(0, -1, -1) +
       PECLET_VOF_C(0, 1, -1) + PECLET_VOF_C(0, 0, -1);
  m2 = PECLET_VOF_C(-1, 0, 1) + PECLET_VOF_C(1, 0, 1) + PECLET_VOF_C(0, -1, 1) +
       PECLET_VOF_C(0, 1, 1) + PECLET_VOF_C(0, 0, 1);
  mm[2][2] = m1 > m2 ? 1.0 : -1.0;

  plicNormalizeL1(mm[0]);
  plicNormalizeL1(mm[1]);
  plicNormalizeL1(mm[2]);

  // pick the centred candidate with the largest dominant component
  t0 = Kokkos::fabs(mm[0][0]);
  t1 = Kokkos::fabs(mm[1][1]);
  t2 = Kokkos::fabs(mm[2][2]);
  cn = 0;
  if (t1 > t0) {
    t0 = t1;
    cn = 1;
  }
  if (t2 > t0)
    cn = 2;

  youngsNormal(c, mm[3]);  // already L1-normalized; (1,0,0) on a degenerate stencil

  t0 = Kokkos::fabs(mm[3][0]);
  t1 = Kokkos::fabs(mm[3][1]);
  t2 = Kokkos::fabs(mm[3][2]);
  if (t1 > t0)
    t0 = t1;
  if (t2 > t0)
    t0 = t2;
  if (Kokkos::fabs(mm[cn][cn]) > t0)
    cn = 3;

  m[0] = mm[cn][0];
  m[1] = mm[cn][1];
  m[2] = mm[cn][2];
}

#undef PECLET_VOF_C

// ---------------------------------------------------------------------------------------------
// (6) test helpers — exact per-cell fractions for manufactured interfaces
// ---------------------------------------------------------------------------------------------

/// Exact fluid fraction of the axis-aligned cell `[x0,x0+h] x [y0,y0+h] x [z0,z0+h]` under the
/// GLOBAL plane `m . X = alphaGlobal` (fluid on the `m . X < alphaGlobal` side). This is just
/// `plicVolume` on the cell-local plane, so `initPlane` data is exact to round-off.
KOKKOS_INLINE_FUNCTION double planeCellFraction(double mx, double my, double mz, double alphaGlobal,
                                                double x0, double y0, double z0, double h) {
  const double aLoc = alphaGlobal - (mx * x0 + my * y0 + mz * z0);
  return plicVolume(mx * h, my * h, mz * h, aLoc);
}

/// Fluid fraction of a cell for a sphere (fluid = INSIDE the sphere), by recursive octree
/// subdivision with an exact tangent-plane closure at the leaves. Sub-boxes entirely inside or
/// entirely outside short-circuit on the centre-distance / half-diagonal test, so only the O(4^L)
/// boxes the surface actually crosses are refined. Iterative (explicit stack) — device-safe, no
/// recursion.
///
/// The leaf closure is second order in the leaf size delta, giving a cell-fraction error
/// ~ delta^2 / (8 h R): `levels = 4` is already ~3e-5 at R/h = 16. Test-only helper.
///
/// @param levels  subdivision depth, clamped to [0, 8]
KOKKOS_INLINE_FUNCTION double sphereCellFraction(double cx, double cy, double cz, double r,
                                                 double x0, double y0, double z0, double h,
                                                 int levels) {
  if (levels < 0)
    levels = 0;
  if (levels > 8)
    levels = 8;
  struct Box {
    double x, y, z, s;
    int lev;
  };
  Box stack[64];  // DFS over an octree of depth L keeps at most 7L+1 boxes live
  int top = 0;
  stack[top++] = Box{x0, y0, z0, h, 0};
  double vol = 0.0;
  const double kHalfDiag = 0.86602540378443865;  // sqrt(3)/2

  while (top > 0) {
    const Box b = stack[--top];
    const double half = 0.5 * b.s;
    const double dx = b.x + half - cx, dy = b.y + half - cy, dz = b.z + half - cz;
    const double d = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
    const double hd = kHalfDiag * b.s;
    if (r - d >= hd) {  // wholly inside the sphere
      vol += b.s * b.s * b.s;
      continue;
    }
    if (d - r >= hd)
      continue;  // wholly outside
    if (b.lev >= levels || !(d > 0.0)) {
      if (d > 0.0) {
        const double inv = 1.0 / d;
        const double nx = dx * inv, ny = dy * inv, nz = dz * inv;
        // fluid where (X - centre).n < r, i.e. n.X < r + n.centre
        vol += b.s * b.s * b.s *
               planeCellFraction(nx, ny, nz, r + nx * cx + ny * cy + nz * cz, b.x, b.y, b.z, b.s);
      }
      continue;
    }
    for (int q = 0; q < 8; ++q)
      stack[top++] = Box{b.x + ((q & 1) ? half : 0.0), b.y + ((q & 2) ? half : 0.0),
                         b.z + ((q & 4) ? half : 0.0), half, b.lev + 1};
  }
  return vol / (h * h * h);
}

}  // namespace peclet::core::vof

#endif  // PECLET_CORE_VOF_PLIC_HPP

/// @file
/// @brief core — VoF rung V5a (WO-Q): the cut-cell rules of the Weymouth–Yue colour transport.
///
/// **Container-free** (the `plic.hpp` rule: scalars and small arrays only, no `View`, no indexing),
/// so every rule below is a pure function that the device kernels in `advect_wy.hpp` and the solver
/// plumbing in `flow_ibm.hpp` call, and that a host oracle can call verbatim.
///
/// ## What a cut cell changes about Weymouth–Yue
///
/// `VOF_PLAN.md` §3 design rule 2: **`C` is the liquid fraction of the FLUID volume of a cell**, so
/// `C ∈ [0,1]` whatever the solid fraction is. The transported quantity is therefore `eps_i C_i`
/// (fluid volume × liquid fraction = liquid volume), and the split sweep becomes
///
///     eps_i C_i  <-  eps_i C_i + (F_{i-} - F_{i+}) + c_i ( o_{i+} a_{i+} - o_{i-} a_{i-} )
///     F_f = o_f * wyFaceFlux(a_f, ...)                                                        (1)
///
/// with `o_f` the face OPENNESS (fluid area fraction of the face, the same field `scalarBuildRhs`
/// and `divergOpen` weight with), `a_f = uf_f dt/h` the face Courant number and `c_i = H(C^n - ½)`
/// the frozen dilation indicator. Both the flux and the dilation term use the SAME `o_f a_f` per
/// face, and the projection zeroes `sum_f o_f u_f` per cell — so
///
///     d/dt sum_i eps_i C_i = (telescoping flux sum) + sum_i c_i (openness-weighted divergence) = 0
///
/// **exactly**, to the accuracy with which the projection's own `max|div(open u)|` vanishes. That
/// identity, not an accuracy statement, is what gate G2 measures.
///
/// The flux (1) uses the PLIC polyhedron reconstructed on the WHOLE unit cell (as if the cell were
/// not cut) and multiplies its slab volume by the open area. **This is an approximation** — the
/// accurate construction clips the PLIC polyhedron against the SOLID polygon as well
/// (Huang, *JCP* 2025/2026 solid-clipped flux polygons) and would put the liquid where the fluid
/// actually is inside the cell. It is *conservative* either way (one number per face, used with
/// opposite signs by the two neighbours), it is exact wherever the interface and the wall are
/// parallel or the cell is whole, and it is O(1) wrong in the *distribution* inside a cell whose
/// interface and wall cross. Recorded as the known gap; see the WO-Q findings.
///
/// ## The three rules that are NOT in the uncut scheme
///
/// 1. **Effective fluid volume.** `buildCellFraction` subsamples 4³, so its output is a multiple of
///    1/64 and a cell can read `eps == 0` while still owning an OPEN face (the face openness comes
///    from a different quadrature, `buildOpenness`). Dividing the update by such an `eps` would be
///    a division by zero on a cell that legitimately receives flux, so the scheme uses `eps_eff =
///    max(eps, kVofEpsFloor)` with `kVofEpsFloor = 1/64` — the subsampling resolution, i.e. the
///    smallest fluid volume the quadrature can resolve at all. A cell that is `eps == 0` AND has
///    six closed faces is SOLID: it is excluded from the update entirely and its colour is supplied
///    by the band fill below.
///    **Consequence for the conservation gate:** the exactly conserved functional is
///    `sum_i eps_eff_i C_i`, not `sum_i eps_i C_i` — they differ only on `eps == 0` cells with an
///    open face, where the raw sum silently drops whatever flux enters. Both are reported.
/// 2. **Boundedness.** Weymouth's proof bounds the flux volume against the CELL volume; in a cut
///    cell the volume is `eps_i` and the area is `o_f`, so the effective Courant number is
///    `o_f |a_f| / eps_i`. `vofCutCourant` is that number with a 0.1 floor on `eps`: the limiter
///    throttles for genuinely cut cells and does NOT chase slivers to dt = 0 (a sliver's own
///    boundedness is handled by the clip, which is measured).
/// 3. **The clip is a diagnostic, not a mechanism.** After each sweep C is clipped into [0,1] in
///    CUT cells only (`eps < 1`), and the liquid volume that the clip moved is accumulated. If that
///    number is not negligible the scheme is wrong and the fix is the solid-clipped flux polygon,
///    not a bigger clip.
///
/// ## The solid-band fill (the 90° / Afkhami–Bussmann limit)
///
/// The MYC 3³ stencil and the V3 height-function columns of a cell next to a wall reach INTO the
/// solid. Leaving those cells at 0 makes every wall look perfectly non-wetting; the neutral rule is
/// a **zero-slope continuation** of the colour into the wall, which is exactly the 90° contact
/// angle boundary condition of Afkhami & Bussmann (*IJNMF* 57:453, 2008) written as fractions.
/// Three passes with a SHRINKING depth budget (pass k writes solid cells at ghost depth ≤ 3−k and
/// reads only fluid cells or cells filled in an EARLIER pass) walk the value up to three cells into
/// the solid while keeping every read inside the halo the exchange has already made
/// decomposition-independent. WO-S replaces the pass-1 rule with the θ-consistent one and keeps
/// passes 2–3 and the depth budget unchanged.
///
/// **Layer L1 (`VOF_PLAN.md` §11): promoted from `flow/src/vof/` to `core` (WO-W0, 2026-09-02).**
/// These are the container-free VoF math kernels — one copy shared by every container: flow's
/// structured colour field, the per-bubble block container (Part III) and the future AMR path.
/// `flow/src/vof/cutcell.hpp` is now a thin include + a using-directive, so every existing
/// `peclet::flow::vof::` spelling keeps working unchanged.
#ifndef PECLET_CORE_VOF_CUTCELL_HPP
#define PECLET_CORE_VOF_CUTCELL_HPP

#include <Kokkos_Core.hpp>

namespace peclet::core::vof {

/// The smallest fluid volume `buildCellFraction`'s 4³ subsampling can resolve (one subcell of 64).
/// Used as the floor of the effective fluid volume the cut-cell update divides by — see rule 1.
inline constexpr double kVofEpsFloor = 1.0 / 64.0;

/// Below this the interface-local Courant number stops being amplified by 1/eps (rule 2): a cut
/// cell at eps = 0.1 already reports 10x its face Courant number, and chasing thinner slivers with
/// dt buys boundedness the clip provides for free.
inline constexpr double kVofCourantEpsFloor = 0.1;

/// Cell classification on the colour block. Kept as an unsigned char field so it can ride the
/// ordinary halo exchange as a double and be read as a flag in every kernel.
enum VofCellKind : unsigned char {
  kVofFluid = 0,  ///< carries colour; `eps_eff > 0`; at least one open face OR `eps > 0`
  kVofSolid = 1,  ///< `eps == 0` AND all six faces closed: no colour, filled for the stencils
};

/// The effective fluid volume of a cell (rule 1). Returns 0 for a solid cell, so a caller can use
/// `eps_eff > 0` as the "is advected" predicate and never divide by zero.
KOKKOS_INLINE_FUNCTION double vofEpsEff(double eps, bool solid) {
  if (solid)
    return 0.0;
  return eps > kVofEpsFloor ? eps : kVofEpsFloor;
}

/// Is this cell SOLID? `eps == 0` and every one of the six faces closed. `oLo`/`oHi` are the
/// openness of the `-d`/`+d` faces.
KOKKOS_INLINE_FUNCTION bool vofIsSolidCell(double eps, const double oLo[3], const double oHi[3]) {
  if (eps > 0.0)
    return false;
  for (int d = 0; d < 3; ++d)
    if (oLo[d] > 0.0 || oHi[d] > 0.0)
      return false;
  return true;
}

/// The openness-weighted geometric flux (1): one number per face, used with opposite signs by the
/// two neighbours, so the sum telescopes bit-exactly whatever `o` is.
KOKKOS_INLINE_FUNCTION double vofCutFlux(double openness, double rawFlux) {
  return openness * rawFlux;
}

/// Weymouth's admissible flux interval, generalized to a cut donor cell — the cut-cell twin of
/// `momentum_advect.hpp` point 3, and for the same reason.
///
/// The whole-cell PLIC slab times the open area is an APPROXIMATION of the solid-clipped flux
/// polygon, so it is not automatically bounded by what the donor actually holds: through a face of
/// open area `o` and slab thickness `|a|` the scheme sweeps a FLUID volume `o|a|`, of which at most
/// `eps_don C_don` can be liquid and at most `eps_don (1 - C_don)` can be gas. Hence
///
///     max(0, o|a| - eps_don (1 - C_don))  <=  |F|  <=  min(o|a|, eps_don C_don).
///
/// It is applied to the ONE face value both neighbours share, so conservation still telescopes
/// bit-exactly; it is inactive wherever the geometry is self-consistent; and it is applied ONLY
/// when the donor is MIXED — a pure-phase donor takes `wyFaceFlux`'s algebraic branch, whose flux
/// is `o C_don |a|` and therefore already exactly bounded, and clamping it would break the exact
/// full-cell cancellation (`o|a|` vs `min(o|a|, eps)`) that makes an interior full cell stationary
/// to the last bit.
///
/// MEASURED without it (24^3 packing, 200 kinematic steps at CFL 0.2): the [0,1] clip of rule 3
/// fires at up to 3.2e-5 liquid volume per step and the conserved functional drifts 1.3e-8 in 30
/// steps. With it: the clip stops firing and the drift is the projection floor.
KOKKOS_INLINE_FUNCTION double vofCutFluxClamp(double fluxMag, double sweep, double epsDon,
                                              double cDon) {
  double hi = epsDon * cDon;
  if (sweep < hi)
    hi = sweep;
  double lo = sweep - epsDon * (1.0 - cDon);
  if (lo < 0.0)
    lo = 0.0;
  if (lo > hi)
    lo = hi;  // a donor thinner than the swept volume: give everything it has
  return fluxMag > hi ? hi : (fluxMag < lo ? lo : fluxMag);
}

/// The boundedness-relevant Courant number of a face against a cell (rule 2).
///
/// **Two constraints, and the max of them is the answer.** WO-Q specifies `o_f |a_f| /
/// max(eps,0.1)` — the flux volume against the cell's FLUID volume — and that is the second term.
/// On its own it is WRONG, and loudly: it is *smaller* than `|a_f|` wherever `o_f < eps`, so a
/// nearly-closed face
/// (`o = 0.02`) inside an open cell would license `|a| = 10`. But `|a|` is the thickness of the
/// slab the geometric flux clips out of the DONOR CELL, and a slab thicker than the cell is not a
/// flux at all — `plicSlabVolume` is only a flux for `|a| <= 1`, and Weymouth's proof needs `|a| <=
/// 1/4` in 3D whatever the geometry is. MEASURED with the second term alone: a 24^3 sphere packing
/// at "CFL 0.2" ran at `dt = 1.85` and lost **70 % of the liquid volume in 200 steps** while the
/// flux sum still telescoped — the classic signature of an over-CFL Weymouth-Yue run (conservation
/// is algebraic and survives; boundedness does not).
///
/// So: `max( |a_f| , o_f |a_f| / max(eps_i, 0.1) )`. It reduces EXACTLY to the uncut `|a_f|` in
/// clear fluid (`o = eps = 1`), it throttles by `1/eps` in a genuinely cut cell, and the 0.1 floor
/// stops it chasing slivers to `dt = 0` (a sliver's boundedness is handled by the clip, which is a
/// measured number).
KOKKOS_INLINE_FUNCTION double vofCutCourant(double openness, double a, double eps) {
  const double e = eps > kVofCourantEpsFloor ? eps : kVofCourantEpsFloor;
  const double aa = a < 0.0 ? -a : a;
  const double cut = openness * aa / e;
  return aa > cut ? aa : cut;
}

/// The clip of rule 3, returned as the CLIPPED value; `moved` receives the liquid VOLUME the clip
/// created (positive) or destroyed (negative) in this cell, i.e. `eps_eff * (C_clipped - C_raw)`.
KOKKOS_INLINE_FUNCTION double vofClipCut(double c, double epsEff, double& moved) {
  const double cc = c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c);
  moved = epsEff * (cc - c);
  return cc;
}

/// Band-fill state, stored per cell on the colour block. `kVofFillFluid` marks a cell whose colour
/// is data (a fluid cell); `kVofFillPass0 + k` marks a solid cell filled in pass `k` (k = 1..3);
/// `kVofFillNone` is an unfilled solid cell. Pass `k` reads a neighbour iff its state is
/// `kVofFillFluid` or `kVofFillPass0 + j` with `j < k` — never a cell being written in the same
/// pass, which is what makes the three passes race-free without a second buffer.
enum VofFillState : unsigned char {
  kVofFillNone = 0,
  kVofFillFluid = 1,
  kVofFillPass0 = 1,  ///< pass k marks `kVofFillPass0 + k`, i.e. 2, 3, 4
};

/// Is `state` readable by pass `k` (1-based)? Fluid always; a solid cell only if an EARLIER pass
/// filled it.
KOKKOS_INLINE_FUNCTION bool vofFillReadable(unsigned char state, int k) {
  return state != kVofFillNone && state <= static_cast<unsigned char>(kVofFillPass0 + k - 1);
}

/// Ghost depth of an index on one axis of an extended block: 0 for an inner cell, growing outward.
KOKKOS_INLINE_FUNCTION int vofGhostDepth1(int i, int g, int n) {
  if (i < g)
    return g - i;
  if (i >= g + n)
    return i - (g + n) + 1;
  return 0;
}

/// Ghost depth of a cell = the max over the three axes.
KOKKOS_INLINE_FUNCTION int vofGhostDepth(int x, int y, int z, int g, int nx, int ny, int nz) {
  const int a = vofGhostDepth1(x, g, nx), b = vofGhostDepth1(y, g, ny),
            c = vofGhostDepth1(z, g, nz);
  const int m = a > b ? a : b;
  return m > c ? m : c;
}

}  // namespace peclet::core::vof

#endif  // PECLET_CORE_VOF_CUTCELL_HPP

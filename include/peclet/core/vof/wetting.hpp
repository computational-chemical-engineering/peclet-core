/// @file
/// @brief core — VoF rung V5b (WO-S): the theta-consistent solid-band fill (static contact angle).
///
/// **Container-free** (the `plic.hpp` rule: scalars and small arrays only, no `View`, no indexing),
/// so every rule below is a pure function the device kernels in `advect_wy.hpp` call and a host
/// oracle can call verbatim. It replaces ONLY pass 1 of WO-Q's three-pass solid-band fill; passes
/// 2-3 (the mean of already-filled neighbours) and the shrinking depth budget are unchanged.
///
/// ## What the fill has to do
///
/// The MYC 3^3 stencil and the V3 height-function columns of a cell next to a wall reach INTO the
/// solid. WO-Q's neutral rule (the mean of the fluid face neighbours) continues the colour into the
/// wall with zero slope — the 90 deg limit of the Afkhami & Bussmann height-function boundary
/// condition (*IJNMF* 57:453, 2008). To impose a contact angle theta we instead write into the band
/// the fractions of the PLANE that continues the fluid-side interface into the solid at angle
/// theta. The unmodified V3 curvature cascade then returns the curvature of an interface that meets
/// the wall at theta, and the V4 balanced force does the rest — no force is added at the wall.
///
/// ## Sign conventions (all three are load-bearing; they are checked in `test_vof_wetting.cpp`)
///
/// - `n_w = grad(sdf)/|grad(sdf)|` points from the SOLID INTO the fluid (`sdf > 0` is fluid).
/// - the PLIC normal `m` points from the LIQUID INTO the gas, liquid side `m . x < alpha`
///   (`plic.hpp`).
/// - theta is measured THROUGH THE LIQUID, and at the contact line **`m . n_w = cos(theta)`**.
///   theta = 0 (complete wetting) gives `m = n_w`: the liquid side `m . x < alpha` is then the
///   SOLID side, i.e. a liquid film wets the wall and the band fills with liquid. theta = 180 gives
///   `m = -n_w` and an empty band. theta = 90 gives `m` perpendicular to the wall — the interface
///   standing on the wall, which is the neutral case.
///
/// ## The construction
///
/// For a solid band cell `s`, walk along `n_w(s)` to the first FLUID cell `f`, then
///
///  1. take the fluid-side normal `m_f` from a **fluid-only Youngs gradient**
///     (`youngsNormalFluidOnly`) — the 27-point Youngs weights restricted to cells that are not
///     solid and renormalized per half-plane. Restricting is what breaks the circularity: the
///     ordinary MYC normal of `f` reads the very band cells this fill is writing, so using it would
///     make the fill a fixed-point iteration on its own output. (Basilisk's `contact.h` breaks the
///     same loop with a second reconstruction pass; that is the WO-S fallback if the gate fails.)
///  2. keep from `m_f` only its component IN the wall, `t = m_f - (m_f . n_w) n_w`, and build
///     **`m_theta = cos(theta) n_w + sin(theta) t_hat`** — the unique unit vector in the
///     `(n_w, t_hat)` plane making angle theta with `n_w` on the same side as `m_f`. When `|t|` is
///     below `tEps` the interface is parallel to the wall and no rotation is defined, so `m_f` is
///     kept.
///  3. anchor the plane, and
///  4. `C_s` = the fraction of cell `s`'s unit cube on the liquid side of it.
///
/// ## Only the IN-WALL half of `m_f` is used, and that is deliberate
///
/// A fluid-only estimator cannot measure the WALL-NORMAL component of the interface normal: below
/// the first fluid row there is no colour to difference against, and the one-sided substitute reads
/// a saturating profile over half the distance. Measured on EXACT plane fractions next to a flat
/// wall (gate G0e): the fluid-only normal's angle to the wall is wrong by up to **23 deg** at
/// theta = 30, against 2.3 deg for the same Youngs stencil with the solid rows present. The
/// IN-WALL components are two-sided and complete, and their azimuth is accurate to **~1 deg** over
/// the same sweep. Since the whole point of the rung is to OVERWRITE the apparent angle with the
/// prescribed one, the construction above uses the azimuth (which is measured well) and discards
/// the wall-normal component (which is not). This is what makes the rung's accuracy independent of
/// the estimator defect the fill would otherwise inherit.
///
/// ## The anchor's own column is not enough (measured)
///
/// The construction as the work order states it walks from the band cell to the first fluid cell
/// ALONG `n_w` and uses that cell's interface. Where that cell is PURE phase the rule degenerates
/// to `C_s = C_f`, and that is wrong exactly where it matters: at a wetting angle the continued
/// interface reaches into the solid BEYOND the column where the contact line crosses the first
/// fluid row, so the columns just outside the contact circle have a pure-gas anchor and receive
/// nothing, while the true continuation puts liquid there. MEASURED on an equilibrium theta = 60
/// cap (D/dx = 20, wall on a cell face): the band column one cell outside the contact column reads
/// 0 where the continued plane gives 0.63 and 0.05 in the two rows below the wall, so the height
/// function reads a wall slope closer to 90 deg than to 60 — and the equilibrium contact angle of
/// the whole run comes out **5 deg biased towards 90**.
///
/// The fix, shipped: when the anchor is pure phase, average the theta-plane fractions of the
/// anchor's MIXED fluid neighbours (its 3^3 stencil, so at most one column away) evaluated in cell
/// `s`; with no mixed neighbour the pure continuation stands. Every read stays inside the halo the
/// exchange has already made owner-consistent, so the fill is still decomposition-independent.
///
/// ## The anchor: a measured deviation from the work order
///
/// WO-S anchors the plane at the contact point `c = p_f - sdf(p_f) n_w`, i.e. the fluid cell's PLIC
/// centroid projected onto the wall ALONG `n_w`. That point is on the wall but it is **not on the
/// interface plane** unless the interface is perpendicular to the wall: projecting along `n_w`
/// changes the plane's offset by `-sdf(p_f) cos(theta)`, an O(1) *shift of the interface* (up to
/// ~1 cell at theta = 30 deg) with the wrong sign — it removes liquid from the band for a wetting
/// angle. The consequence is that the fill is **not idempotent**: an interface already meeting the
/// wall at exactly theta is not reproduced, theta is not a fixed point of the scheme, and the
/// equilibrium angle is biased. Measured (gate G0a/G0b): the band fraction is off by up to
/// **0.26** at theta = 60, against 1e-15 for the shipped rule.
///
/// The shipped default is `kVofPivotVolume`: the plane with normal `m_theta` whose liquid volume in
/// cell `f` is exactly `C_f` (`plicAlpha`). It needs no pivot point at all, it is
/// volume-consistent with the cell it is anchored on, it is **exactly idempotent** (if the true
/// interface is a plane at angle theta then `plicAlpha` returns its offset exactly, so the band
/// receives the exact fractions of the continued interface — that identity is gate G0a), and it is
/// the only variant that does not read the poorly-measured wall-normal part of `m_f` at all.
/// `kVofPivotInterface` (the plane through the PLIC centroid `p_f` reconstructed from `m_f` — the
/// Afkhami-Bussmann / Basilisk `contact.h` rule), `kVofPivotWallNormal` (WO-S as written) and
/// `kVofPivotContactLine` ship as measured ablations (`set_contact_angle_pivot`).
///
/// **Known approximation, consistent with WO-Q.** `C_f` is the liquid fraction of the cell's FLUID
/// volume, but the reconstruction is done on the WHOLE unit cube (exactly as WO-Q's flux is), so in
/// a CUT anchor cell the plane is displaced by roughly `(1 - eps_f)/2` times the wall-normal
/// component of `m_theta` — 0.14 cells at theta = 60 on a half-cut cell, zero at theta = 90 and
/// zero in an uncut cell. Removing it needs the solid-clipped volume relation (Huang, JCP
/// 2025/2026), the same refinement WO-Q's flux defers.
///
/// **Layer L1 (`VOF_PLAN.md` §11): promoted from `flow/src/vof/` to `core` (WO-W0, 2026-09-02).**
/// These are the container-free VoF math kernels — one copy shared by every container: flow's
/// structured colour field, the per-bubble block container (Part III) and the future AMR path.
/// `flow/src/vof/wetting.hpp` is now a thin include + a using-directive, so every existing
/// `peclet::flow::vof::` spelling keeps working unchanged.
#ifndef PECLET_CORE_VOF_WETTING_HPP
#define PECLET_CORE_VOF_WETTING_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "peclet/core/vof/curvature.hpp"  // plicPolygon + polygonAreaCentroid (the PLIC centroid)
#include "peclet/core/vof/plic.hpp"

namespace peclet::core::vof {

/// How the theta-plane is anchored in the fluid cell. See the file header for why the default is
/// not the work order's rule.
enum VofWettingPivot : int {
  kVofPivotVolume = 0,      ///< `plicAlpha(m_theta, C_f)`: volume-consistent. DEFAULT, idempotent.
  kVofPivotInterface = 1,   ///< through `p_f` (Afkhami-Bussmann / Basilisk). Idempotent.
  kVofPivotWallNormal = 2,  ///< WO-S as written: `c = p_f - sdf(p_f) n_w`. NOT idempotent.
  kVofPivotContactLine = 3  ///< the point of the fluid plane on the wall nearest `p_f`. Idempotent.
};

/// Which rule produced a band cell's colour, for `contact_angle_diagnostics()`.
enum VofWettingBranch : int {
  kVofWetNone = 0,       ///< not written by pass 1
  kVofWetTheta = 1,      ///< the theta-plane of this column's own anchor cell
  kVofWetNeighbour = 2,  ///< the mean of the theta-planes of the anchor's MIXED neighbours, used
                         ///< where the anchor itself is pure phase but the contact line passes
                         ///< through the column next to it — see "the anchor's own column is not
                         ///< enough" in the file header
  kVofWetPure = 3,       ///< pure-phase continuation (`C_f` is 0 or 1, no interface in reach)
  kVofWetParallel = 4,   ///< interface parallel to the wall (|t| < tEps): `m_f` kept unrotated
  kVofWetNeutral = 5,    ///< fell back to WO-Q's neutral mean (no usable `n_w`/`f`/`m_f`)
  kVofWetCount = 6
};

/// Youngs' 27-point weighted gradient **restricted to non-solid cells**.
///
/// Youngs' weights on each of the two opposing planes of axis `d` are 4 (plane centre), 2 (edge)
/// and 1 (corner), i.e. `w = (2 - |t1|)(2 - |t2|)` over the two transverse offsets, and sum to 16.
/// Here each half-plane sum is taken over the INCLUDED (fluid) cells only and divided by the
/// included weight, so the estimator is a difference of two weighted MEANS and the omission of a
/// solid cell does not read as a colour deficit. (With every cell fluid this is Youngs' own normal
/// divided by 16 — the same direction, hence the same PLIC plane.)
///
/// If one of the two planes has no fluid cell at all — the normal case for the wall-normal axis
/// right next to a wall — the difference is taken one-sided against the CENTRE plane and doubled,
/// which is the same first-order slope over half the distance. If both outer planes are empty the
/// component is 0.
///
/// @param c      27-cell colour stencil, `plicSt` order
/// @param fluid  1 where the cell is fluid (not `kVofSolid`), 0 where it is solid
/// @param m      out: L1-normalized normal pointing into the gas
/// @return       false if the gradient is degenerate (no usable direction) — the caller must then
///               fall back to the neutral rule rather than invent one.
KOKKOS_INLINE_FUNCTION bool youngsNormalFluidOnly(const double c[27], const unsigned char fluid[27],
                                                  double m[3]) {
  for (int d = 0; d < 3; ++d) {
    double s[3] = {0.0, 0.0, 0.0}, w[3] = {0.0, 0.0, 0.0};  // planes at offset -1, 0, +1 in d
    for (int q = 0; q < 27; ++q) {
      if (!fluid[q])
        continue;
      const int o[3] = {q % 3 - 1, (q / 3) % 3 - 1, q / 9 - 1};
      const int t1 = o[(d + 1) % 3], t2 = o[(d + 2) % 3];
      const double ww = (2.0 - (t1 < 0 ? -t1 : t1)) * (2.0 - (t2 < 0 ? -t2 : t2));
      const int p = o[d] + 1;
      s[p] += ww * c[q];
      w[p] += ww;
    }
    if (w[0] > 0.0 && w[2] > 0.0)
      m[d] = s[0] / w[0] - s[2] / w[2];
    else if (w[0] > 0.0 && w[1] > 0.0)
      m[d] = 2.0 * (s[0] / w[0] - s[1] / w[1]);
    else if (w[2] > 0.0 && w[1] > 0.0)
      m[d] = 2.0 * (s[1] / w[1] - s[2] / w[2]);
    else
      m[d] = 0.0;
  }
  return plicNormalizeL1(m) >= 1e-30;
}

/// The centroid of the PLIC polygon of cell `f` in the cell's own unit-cube frame, for the plane
/// `m . x = alpha`. Falls back to the plane point nearest the cell centre when the polygon is
/// degenerate (fewer than three vertices), which is the WO-S fallback written out.
KOKKOS_INLINE_FUNCTION void vofPlicCentroid(const double m[3], double alpha, double p[3]) {
  double v[8][3];
  const int nv = plicPolygon(m[0], m[1], m[2], alpha, v);
  if (nv >= 3) {
    double area = 0.0;
    polygonAreaCentroid(v, nv, p, area);
    if (area > 0.0)
      return;
  }
  const double nn = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
  const double t = nn > 0.0 ? (alpha - 0.5 * (m[0] + m[1] + m[2])) / nn : 0.0;
  p[0] = 0.5 + t * m[0];
  p[1] = 0.5 + t * m[1];
  p[2] = 0.5 + t * m[2];
}

/// Build the theta-plane in the FLUID cell's own unit-cube frame.
///
/// @param mfIn   the fluid-side normal (any scale; `youngsNormalFluidOnly`'s output). Only its
///               component IN the wall is used by the default anchor — see the file header.
/// @param cf     the fluid cell's colour (reconstructed on the WHOLE unit cell, as WO-Q's flux is)
/// @param nwIn   the wall normal, solid -> fluid (any scale)
/// @param cosT   cos(theta), @param sinT sin(theta)  (theta prescribed, through the liquid)
/// @param sdfF   the signed distance at the fluid cell's CENTRE, in cells (only read by the
///               `kVofPivotWallNormal` / `kVofPivotContactLine` ablations)
/// @param pivot  `VofWettingPivot`
/// @param tEps   below `|t| = |sin(theta_apparent)| = tEps` no rotation is defined
/// @param mth    out: the theta-plane normal, L2-normalized, pointing into the gas
/// @param alphaTh out: its offset in the fluid cell's frame (liquid side `mth . x < alphaTh`)
/// @param cosApp out: the measured apparent cosine `m_f . n_w` (the diagnostic G1 reports)
/// @return the branch (`kVofWetTheta` or `kVofWetParallel`); never fails.
KOKKOS_INLINE_FUNCTION int vofWettingPlane(const double mfIn[3], double cf, const double nwIn[3],
                                           double cosT, double sinT, double sdfF, int pivot,
                                           double tEps, double mth[3], double& alphaTh,
                                           double& cosApp) {
  // (a) L2-unit frames
  double mf[3] = {mfIn[0], mfIn[1], mfIn[2]};
  double mn = Kokkos::sqrt(mf[0] * mf[0] + mf[1] * mf[1] + mf[2] * mf[2]);
  if (!(mn > 0.0))
    mn = 1.0;
  const double mh[3] = {mf[0] / mn, mf[1] / mn, mf[2] / mn};
  double wn = Kokkos::sqrt(nwIn[0] * nwIn[0] + nwIn[1] * nwIn[1] + nwIn[2] * nwIn[2]);
  if (!(wn > 0.0))
    wn = 1.0;
  const double nw[3] = {nwIn[0] / wn, nwIn[1] / wn, nwIn[2] / wn};

  // (b) rotate to the prescribed angle about the contact line's direction: keep the AZIMUTH of the
  //     interface (its in-wall component, which a fluid-only stencil measures well) and replace the
  //     angle to the wall by the prescribed one.
  cosApp = mh[0] * nw[0] + mh[1] * nw[1] + mh[2] * nw[2];
  double t[3] = {mh[0] - cosApp * nw[0], mh[1] - cosApp * nw[1], mh[2] - cosApp * nw[2]};
  const double tn = Kokkos::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
  int branch = kVofWetTheta;
  if (tn < tEps) {
    mth[0] = mh[0];
    mth[1] = mh[1];
    mth[2] = mh[2];
    branch = kVofWetParallel;
  } else {
    for (int d = 0; d < 3; ++d)
      mth[d] = cosT * nw[d] + sinT * t[d] / tn;
  }

  // (c) anchor it
  if (pivot == kVofPivotVolume) {
    // The plane of normal m_theta whose liquid volume in the anchor cell is exactly C_f. No pivot
    // point is needed and the poorly-measured wall-normal part of m_f is never read.
    alphaTh = plicAlpha(mth[0], mth[1], mth[2], cf);
    return branch;
  }
  // The three ablations all reconstruct the FLUID cell with m_f's own normal first.
  double mfl[3] = {mf[0], mf[1], mf[2]};
  plicNormalizeL1(mfl);
  double pf[3];
  vofPlicCentroid(mfl, plicAlpha(mfl[0], mfl[1], mfl[2], cf), pf);
  double c0[3] = {pf[0], pf[1], pf[2]};
  if (pivot == kVofPivotWallNormal || pivot == kVofPivotContactLine) {
    // |grad sdf| = 1, so the signed distance at p_f is the cell-centre value plus the normal
    // displacement. (Cell centre = (0.5,0.5,0.5) in the local frame; lengths are cells.)
    const double sp = sdfF + nw[0] * (pf[0] - 0.5) + nw[1] * (pf[1] - 0.5) + nw[2] * (pf[2] - 0.5);
    if (pivot == kVofPivotWallNormal) {
      for (int d = 0; d < 3; ++d)
        c0[d] = pf[d] - sp * nw[d];
    } else if (tn >= tEps) {
      // Move within the FLUID plane along the in-plane part of n_w until the wall is reached.
      double q[3] = {nw[0] - cosApp * mh[0], nw[1] - cosApp * mh[1], nw[2] - cosApp * mh[2]};
      const double qn = Kokkos::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
      if (qn > tEps)
        for (int d = 0; d < 3; ++d)
          c0[d] = pf[d] - (sp / qn) * (q[d] / qn);
    }
  }
  alphaTh = mth[0] * c0[0] + mth[1] * c0[1] + mth[2] * c0[2];
  return branch;
}

/// The liquid fraction of the solid cell at integer offset `ds` from the fluid cell, under the
/// theta-plane returned by `vofWettingPlane`. Clamped to [0,1].
KOKKOS_INLINE_FUNCTION double vofWettingFraction(const double mth[3], double alphaTh,
                                                 const int ds[3]) {
  const double a = alphaTh - (mth[0] * ds[0] + mth[1] * ds[1] + mth[2] * ds[2]);
  const double v = plicVolume(mth[0], mth[1], mth[2], a);
  return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

}  // namespace peclet::core::vof

#endif  // PECLET_CORE_VOF_WETTING_HPP

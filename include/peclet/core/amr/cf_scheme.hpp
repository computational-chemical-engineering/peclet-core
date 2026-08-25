// core — pluggable coarse/fine (2:1) interface schemes for the collocated AMR flow solver.
//
// The graded-accuracy limiter measured in docs/amr_collocated_projection.md (addendum): the
// standard two-point C/F flux is 1st-order at level boundaries, and BOTH the aperture and the
// ghost projection inherit it identically (+9% graded drag offset on the dilute sphere,
// scheme-independent). This header makes the C/F treatment a pluggable SCHEME and provides the
// Martin–Cartwright/Martin–Colella tangential-quadratic closure (CfScheme::quadratic — the same
// coarseStar substitution AmrPoisson::applyLaplacianQuad uses) for every operator the STEADY
// solution feels:
//
//   D  — the ½/½ face-average divergence constraint      (buildCfDivDelta,  vector → scalar)
//   G  — the ABC cell gradient (−∇pⁿ predictor + cell correction)  (buildCfGradDelta, scalar → 3)
//   ∇² — the momentum (velocity) diffusion at regular fluid rows    (buildCfLapDelta,  scalar → scalar)
//
// Everything is the LINEAR substitution "coarse-side value → coarse*", where coarse* is the
// scheme's interpolation of the coarse cell at the fine cell's tangential position. So each
// operator's (quad − standard) difference is a precomputed CSR overlay over the level-boundary
// rows: built ONCE on the host (shared verbatim by the host oracle and the device — parity by
// construction), applied as one extra SpMV.
//
// STEADY-STATE PLACEMENT (the (1,2)-mixed philosophy, flow-validated): at the projection's fixed
// point φ→0, so the pressure MATRIX C/F order does not move the steady solution — the matrix (and
// the whole MG hierarchy / PCG / ghost BiCGStab) stays on the standard consistent operator, and
// the scheme enters through (a) the RHS divergence, (b) the pressure gradients, (c) the momentum
// operator (as a lagged deferred-correction RHS term, the same pattern as the implicit-FOU/SOU
// split and Multigrid::solveQuad). The per-step operator mismatch converges through the
// pseudo-transient stepping; the steady equations carry the 2nd-order C/F closure exactly.
//
// EXTENDING: add a value to CfScheme and a branch in cfAppendStencil — every operator delta picks
// it up. A scheme is fully described by the linear stencil it substitutes for the coarse-side
// value of one directed C/F sub-face.
//
// Robustness gating (per tangential axis, falling back to the raw coarse value): the tangential
// coarse neighbours must exist, be same-level, be FLUID (the caller's predicate — never lean on a
// decoupled/held solid value), and the tangential faces must be sufficiently open (openness ≥ 0.5,
// matching applyLaplacianQuad's gate). With the finest-band contract, level boundaries sit in
// smooth flow and the gates are inert there.
#ifndef PECLET_CORE_AMR_CF_SCHEME_HPP
#define PECLET_CORE_AMR_CF_SCHEME_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::amr {

/// Coarse/fine interface scheme for the collocated flow operators.
enum class CfScheme : int {
  standard = 0,   ///< raw coarse value (two-point flux; 1st-order at 2:1 faces)
  quadratic = 1,  ///< Martin–Cartwright tangential quadratic (2nd-order at 2:1 faces)
};

/// Scalar-input CSR overlay: out(i) += Σ_k coef·f(slot) over rows with C/F faces.
struct CfCsr {
  std::vector<Index> start;  ///< size n+1
  std::vector<Index> slot;
  std::vector<double> coef;
};

/// Component-tagged CSR overlay (the divergence delta): out(i) += Σ_k coef·u[comp](slot).
struct CfCompCsr {
  std::vector<Index> start;
  std::vector<Index> slot;
  std::vector<double> coef;
  std::vector<int8_t> comp;
};

namespace detail {

/// Append the scheme's (coarse* − coarse_raw) substitution stencil for ONE directed C/F sub-face
/// to `out`, scaled. `coarse` is interpolated at `fine`'s tangential position across the face
/// normal `axis`. CfScheme::standard appends nothing (no substitution). CfScheme::quadratic is
/// the Martin–Cartwright per-tangential-axis quadratic — the same arithmetic as
/// AmrPoisson::coarseStar / Multigrid::addCoarseStarStencil, plus the fluid gate.
template <unsigned Bits, class Ent, class FluidFn>
inline void cfAppendStencil(const AmrPoisson<3, Bits>& ap, const BlockOctree<3, Bits>& t,
                            std::vector<Ent>& out, Index coarse, Index fine, int axis,
                            double scale, FluidFn&& fluidOk, CfScheme scheme,
                            const Ent& proto = Ent{}) {
  if (scheme != CfScheme::quadratic)
    return;
  auto bc = t.bounds(coarse);
  auto bf = t.bounds(fine);
  const double H = ap.cellWidth(coarse);
  const double sc = static_cast<double>(Index(1) << t.level(coarse));
  const double sf = static_cast<double>(Index(1) << t.level(fine));
  auto push = [&](Index cell, double w) {
    Ent e = proto;
    e.cell = cell;
    e.w = w;
    out.push_back(e);
  };
  for (int tt = 0; tt < 3; ++tt) {
    if (tt == axis)
      continue;
    const double dt = ((static_cast<double>(bf[0][tt]) + 0.5 * sf) -
                       (static_cast<double>(bc[0][tt]) + 0.5 * sc)) *
                      ap.h0();
    // Tangential samples at the coarse cell's ± neighbours. Same-level neighbours contribute
    // directly. A FINER neighbour (an island corner/edge: the region across the tangential face
    // is refined — by 2:1 exactly one level finer) is sampled by the volume average of the 2^Dim
    // leaves covering the coarse-size region: a 2nd-order sample of the region-center value, so
    // the corner rows keep the quadratic closure instead of falling back to the raw coarse value
    // (the old P5b-style skip, which left them locally 1st-order). A COARSER neighbour (or any
    // non-fluid / irregular cover) still falls back to the skip (robust).
    struct Samp {
      Index cell[1 << 3];
      double w[1 << 3];
      int n = 0;
      bool ok = false;
    };
    auto sampleTangential = [&](int dir) {
      Samp sm;
      const Index nb = ap.periodicNeighbor(coarse, tt, dir);
      if (nb < 0)
        return sm;
      if (t.level(nb) == t.level(coarse)) {
        if (!fluidOk(nb))
          return sm;
        sm.cell[0] = nb;
        sm.w[0] = 1.0;
        sm.n = 1;
        sm.ok = true;
        return sm;
      }
      if (t.level(nb) + 1 != t.level(coarse))
        return sm;  // coarser neighbour: keep the fallback
      // Finer neighbour: enumerate the 2^Dim children covering the coarse-size region.
      auto bc2 = t.bounds(coarse);
      const auto sc2 = typename BlockOctree<3, Bits>::Coord(
          typename BlockOctree<3, Bits>::Coord(1) << t.level(coarse));
      const auto sh = typename BlockOctree<3, Bits>::Coord(sc2 >> 1);
      long ext[3];
      for (int d = 0; d < 3; ++d)
        ext[d] = static_cast<long>(t.brick()[d]) * (1L << t.lmax());
      std::array<typename BlockOctree<3, Bits>::Coord, 3> lo = bc2[0];
      const long shifted = static_cast<long>(lo[tt]) + (dir > 0 ? static_cast<long>(sc2) : -static_cast<long>(sc2));
      lo[tt] = static_cast<typename BlockOctree<3, Bits>::Coord>(((shifted % ext[tt]) + ext[tt]) %
                                                                 ext[tt]);
      for (int oct2 = 0; oct2 < (1 << 3); ++oct2) {
        std::array<typename BlockOctree<3, Bits>::Coord, 3> q = lo;
        for (int d = 0; d < 3; ++d)
          if ((oct2 >> d) & 1)
            q[d] = static_cast<typename BlockOctree<3, Bits>::Coord>(
                (static_cast<long>(q[d]) + static_cast<long>(sh)) % ext[d]);
        const Index ch = t.find(q);
        if (ch < 0 || t.level(ch) + 1 != t.level(coarse) || !fluidOk(ch))
          return Samp{};  // irregular cover or solid child: fall back
        sm.cell[sm.n] = ch;
        sm.w[sm.n] = 1.0 / static_cast<double>(1 << 3);
        ++sm.n;
      }
      sm.ok = true;
      return sm;
    };
    const Samp sp = sampleTangential(+1);
    const Samp smi = sampleTangential(-1);
    const bool okP = sp.ok && ap.faceOpenness(coarse, tt, +1) >= 0.5;
    const bool okM = smi.ok && ap.faceOpenness(coarse, tt, -1) >= 0.5;
    if (okP && okM) {
      // Two-sided (the original path, bit-identical): tangential quadratic.
      const double cUp = dt / (2.0 * H) + 0.5 * dt * dt / (H * H);
      const double cUm = -dt / (2.0 * H) + 0.5 * dt * dt / (H * H);
      const double cUc = -dt * dt / (H * H);
      push(coarse, scale * cUc);
      for (int k2 = 0; k2 < sp.n; ++k2)
        push(sp.cell[k2], scale * cUp * sp.w[k2]);
      for (int k2 = 0; k2 < smi.n; ++k2)
        push(smi.cell[k2], scale * cUm * smi.w[k2]);
      continue;
    }
    // Wall-aware one-sided fallback (mixed-level cut-band plan §6.1): when exactly one
    // tangential side survives the gates (the other is solid / closed / irregular — the
    // near-wall seam case), substitute the one-sided LINEAR interpolation dt·u' instead of
    // falling all the way back to the raw coarse value: O(H²) tangential sample error vs the
    // raw fallback's O(H) offset. Both sides gated ⇒ raw fallback (unchanged).
    if (okP != okM) {
      const Samp& s1 = okP ? sp : smi;
      const double sgn = okP ? 1.0 : -1.0;
      push(coarse, -scale * sgn * dt / H);
      for (int k2 = 0; k2 < s1.n; ++k2)
        push(s1.cell[k2], scale * sgn * (dt / H) * s1.w[k2]);
    }
  }
}

struct ScalarEnt {
  Index cell = 0;
  double w = 0.0;
};
struct CompEnt {
  Index cell = 0;
  double w = 0.0;
  int8_t comp = 0;
};

template <class Ent, class Csr>
inline void compactCsr(const std::vector<std::vector<Ent>>& per, Csr& out, Index n) {
  out.start.assign(static_cast<std::size_t>(n) + 1, 0);
  for (Index i = 0; i < n; ++i)
    out.start[static_cast<std::size_t>(i) + 1] =
        out.start[static_cast<std::size_t>(i)] +
        static_cast<Index>(per[static_cast<std::size_t>(i)].size());
  const Index nz = out.start[static_cast<std::size_t>(n)];
  out.slot.resize(static_cast<std::size_t>(nz));
  out.coef.resize(static_cast<std::size_t>(nz));
  Index k = 0;
  for (Index i = 0; i < n; ++i)
    for (const auto& e : per[static_cast<std::size_t>(i)]) {
      out.slot[static_cast<std::size_t>(k)] = e.cell;
      out.coef[static_cast<std::size_t>(k)] = e.w;
      ++k;
    }
}

}  // namespace detail

/// (∇²_scheme − ∇²_std) as a scalar CSR, ×`factor` (pass μ for the momentum deferred-correction
/// RHS, 1 for a pressure-matrix delta): row i gains scale = factor·invV·(αA/d)·(±1) per directed
/// C/F face (+ for a coarser neighbour's substituted value, − for our own value substituted at a
/// finer sub-face) — the same construction as Multigrid's quad CSR. Rows gated by `rowOk`
/// (e.g. regular fluid cells for the momentum operator).
template <unsigned Bits, class RowFn, class FluidFn>
inline CfCsr buildCfLapDelta(const AmrPoisson<3, Bits>& ap, const BlockOctree<3, Bits>& t,
                             double factor, RowFn&& rowOk, FluidFn&& fluidOk, CfScheme scheme) {
  const Index n = t.numLeaves();
  std::vector<std::vector<detail::ScalarEnt>> per(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    if (!rowOk(i))
      continue;
    const unsigned Li = t.level(i);
    const double invV = 1.0 / ap.cellVolume(i);
    ap.forEachFaceNeighbor(i, [&](Index j, Real c, int axis, double a) {
      const unsigned Lj = t.level(j);
      if (Lj == Li)
        return;
      const Index coarse = (Lj > Li) ? j : i;
      const Index fine = (Lj > Li) ? i : j;
      const double scale = factor * invV * (a * c) * ((Lj > Li) ? 1.0 : -1.0);
      detail::cfAppendStencil(ap, t, per[static_cast<std::size_t>(i)], coarse, fine, axis, scale,
                              fluidOk, scheme);
    });
  }
  CfCsr csr;
  detail::compactCsr(per, csr, n);
  return csr;
}

/// (D_scheme − D_std) for the face-average divergence div_i = invV·Σ α·A·dir·(face value).
/// The standard face value is the pinned ½(u_i+u_j) — at a 2:1 sub-face that average is offset
/// from the face in the NORMAL direction (fine center at h/2, coarse at H/2 — an O(h) value
/// error the tangential fix alone cannot remove). The scheme's face value is the
/// distance-weighted interpolation of the two O(h²)-accurate point values,
///     v = wF·u_fine + wC·u_coarse*,  wF = (H/2)/d, wC = (h/2)/d, d = (H+h)/2,
/// with u_coarse* the tangential coarse* substitution. Both incident cells use the identical
/// value (conservative telescoping). Delta emitted per C/F sub-face:
///     Δ = scale·[(wF−½)·u_F + (wC−½)·u_C + wC·(u_C* − u_C)],  scale = invV·α·A·dir.
/// Component-tagged: the substituted value is the face-normal velocity component.
template <unsigned Bits, class RowFn, class FluidFn>
inline CfCompCsr buildCfDivDelta(const AmrPoisson<3, Bits>& ap, const BlockOctree<3, Bits>& t,
                                 RowFn&& rowOk, FluidFn&& fluidOk, CfScheme scheme) {
  const Index n = t.numLeaves();
  std::vector<std::vector<detail::CompEnt>> per(static_cast<std::size_t>(n));
  if (scheme != CfScheme::standard) {
    for (Index i = 0; i < n; ++i) {
      if (!rowOk(i))
        continue;
      const unsigned Li = t.level(i);
      const double invV = 1.0 / ap.cellVolume(i);
      ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double area, double, double alpha) {
        const unsigned Lj = t.level(j);
        if (Lj == Li)
          return;
        const Index coarse = (Lj > Li) ? j : i;
        const Index fine = (Lj > Li) ? i : j;
        const double H = ap.cellWidth(coarse), h = ap.cellWidth(fine);
        const double d = 0.5 * (H + h);
        const double wF = (0.5 * H) / d, wC = (0.5 * h) / d;
        const double scale = invV * alpha * area * static_cast<double>(dir);
        detail::CompEnt proto;
        proto.comp = static_cast<int8_t>(axis);
        auto& row = per[static_cast<std::size_t>(i)];
        detail::CompEnt eF = proto, eC = proto;
        eF.cell = fine;
        eF.w = scale * (wF - 0.5);
        eC.cell = coarse;
        eC.w = scale * (wC - 0.5);
        row.push_back(eF);
        row.push_back(eC);
        detail::cfAppendStencil(ap, t, row, coarse, fine, axis, scale * wC, fluidOk, scheme,
                                proto);
      });
    }
  }
  CfCompCsr csr;
  detail::compactCsr(per, csr, n);
  csr.comp.resize(csr.slot.size());
  Index k = 0;
  for (Index i = 0; i < n; ++i)
    for (const auto& e : per[static_cast<std::size_t>(i)])
      csr.comp[static_cast<std::size_t>(k++)] = e.comp;
  return csr;
}

/// (G_scheme − G_std) for the ABC cell gradient gradOf/grad3. The standard operator averages the
/// two side face-gradients with ½/½: out[a] = ½(Σg⁺/n⁺ + Σg⁻/n⁻), g = ±(f_j−f_i)/dist over
/// α>1e-12 faces. Each face gradient samples ∇f at the MIDPOINT of its two centers (offset
/// s± = ±dist±/2 from the cell center), so at a C/F-adjacent row the ½/½ recombination misses
/// the cell center by O(h). The scheme (a) substitutes the coarse* value in each C/F face
/// gradient (tangentially 2nd-order sample) and (b) recombines the sides with the weights that
/// put the sampled gradient AT the cell center:
///     w⁺ = dist⁻/(dist⁺+dist⁻),  w⁻ = dist⁺/(dist⁺+dist⁻)   (½/½ when both sides same-level).
/// Delta per row/axis = Σ_side (w±−½)·[std side average] + Σ_side w±/n±·[C/F substitutions].
/// One scalar CSR per output axis, over rows adjacent to a level boundary.
template <unsigned Bits, class RowFn, class FluidFn>
inline std::array<CfCsr, 3> buildCfGradDelta(const AmrPoisson<3, Bits>& ap,
                                             const BlockOctree<3, Bits>& t, RowFn&& rowOk,
                                             FluidFn&& fluidOk, CfScheme scheme) {
  const Index n = t.numLeaves();
  std::array<std::vector<std::vector<detail::ScalarEnt>>, 3> per;
  for (int a = 0; a < 3; ++a)
    per[static_cast<std::size_t>(a)].resize(static_cast<std::size_t>(n));
  if (scheme != CfScheme::standard) {
    for (Index i = 0; i < n; ++i) {
      if (!rowOk(i))
        continue;
      const unsigned Li = t.level(i);
      // Per axis/side: open-face count, the side's (uniform) center-to-center distance, and
      // whether the side has a level mismatch. slot [axis][side], side 0 = +, 1 = −.
      int cnt[3][2] = {};
      double sdist[3][2] = {};
      bool cf[3][2] = {};
      ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double alpha) {
        if (alpha <= 1e-12)
          return;
        const int s = (dir > 0) ? 0 : 1;
        ++cnt[axis][s];
        sdist[axis][s] = dist;
        if (t.level(j) != Li)
          cf[axis][s] = true;
      });
      for (int a = 0; a < 3; ++a) {
        if (!cf[a][0] && !cf[a][1])
          continue;  // no level mismatch on this axis: standard ½/½ is centered
        if (cnt[a][0] == 0 || cnt[a][1] == 0)
          continue;  // one-sided (closed) axis: leave the standard treatment
        const double dp = sdist[a][0], dm = sdist[a][1];
        const double wp = dm / (dp + dm), wm = dp / (dp + dm);
        auto& row = per[static_cast<std::size_t>(a)][static_cast<std::size_t>(i)];
        ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double alpha) {
          if (axis != a || alpha <= 1e-12)
            return;
          const int s = (dir > 0) ? 0 : 1;
          const double w = (s == 0) ? wp : wm;
          const double inv = 1.0 / static_cast<double>(cnt[a][s]);
          const double gsgn = (dir > 0) ? 1.0 : -1.0;  // coefficient of f_j in g
          // (a) side reweighting of the STANDARD face gradient: (w − ½)·g/cnt.
          const double rw = (w - 0.5) * inv * gsgn / dist;
          if (rw != 0.0) {
            row.push_back({j, rw});
            row.push_back({i, -rw});
          }
          // (b) coarse* substitution inside the C/F face gradients, at the NEW side weight.
          const unsigned Lj = t.level(j);
          if (Lj == Li)
            return;
          const Index coarse = (Lj > Li) ? j : i;
          const Index fine = (Lj > Li) ? i : j;
          const double ssgn = (Lj > Li) ? gsgn : -gsgn;  // sign of the substituted value in g
          detail::cfAppendStencil(ap, t, row, coarse, fine, a, w * inv * ssgn / dist, fluidOk,
                                  scheme);
        });
      }
    }
  }
  std::array<CfCsr, 3> out;
  for (int a = 0; a < 3; ++a)
    detail::compactCsr(per[static_cast<std::size_t>(a)], out[static_cast<std::size_t>(a)], n);
  return out;
}

/// (uf_scheme − uf_std) for the div-free FACE field uf_k = ½(u_i+u_j) − (φ₊−φ₋)/d, one delta row
/// per forEachFaceFull SLOT (cell-major enumeration — the FaceGeom / oracle faceStart CSR order,
/// identical on both engines). At a 2:1 sub-face:
///   face average → the distance-weighted {u_fine, u_coarse*} interpolation (as in
///     buildCfDivDelta, so the advecting flux matches the divergence constraint):
///       Δvel = (wF−½)·u_F + (wC−½)·u_C + wC·(u_C* − u_C)      [face-normal component]
///   face gradient → coarse* substitution in the compact (φ₊−φ₋)/d (the P5b flux form):
///       Δphi = −sideSign·(φ_C* − φ_C)/d,  sideSign = +1 iff the coarse cell is on the + side.
/// Both incident slots of a shared sub-face produce the identical value (conservative). Emitted
/// only for faces whose BOTH centers are fluid (the advection gate; closed faces' uf is unused).
/// The vel/phi parts apply with the standard cfApplyComp / cfApply over uf's slot array.
struct CfUfDelta {
  CfCompCsr vel;  ///< reads the velocity components, rows = face slots
  CfCsr phi;      ///< reads the projection potential φ, rows = face slots
};

template <unsigned Bits, class FluidFn>
inline CfUfDelta buildCfUfDelta(const AmrPoisson<3, Bits>& ap, const BlockOctree<3, Bits>& t,
                                FluidFn&& fluidOk, CfScheme scheme) {
  const Index n = t.numLeaves();
  Index nSlots = 0;
  for (Index i = 0; i < n; ++i)
    ap.forEachFaceFull(i, [&](Index, int, int, double, double, double) { ++nSlots; });
  CfUfDelta d;
  d.vel.start.assign(static_cast<std::size_t>(nSlots) + 1, 0);
  d.phi.start.assign(static_cast<std::size_t>(nSlots) + 1, 0);
  if (scheme == CfScheme::standard)
    return d;
  Index slot = 0;
  for (Index i = 0; i < n; ++i) {
    const unsigned Li = t.level(i);
    ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double) {
      d.vel.start[static_cast<std::size_t>(slot) + 1] = d.vel.start[static_cast<std::size_t>(slot)];
      d.phi.start[static_cast<std::size_t>(slot) + 1] = d.phi.start[static_cast<std::size_t>(slot)];
      const unsigned Lj = t.level(j);
      if (Lj != Li && fluidOk(i) && fluidOk(j)) {
        const Index coarse = (Lj > Li) ? j : i;
        const Index fine = (Lj > Li) ? i : j;
        const double H = ap.cellWidth(coarse), h = ap.cellWidth(fine);
        const double wF = (0.5 * H) / dist, wC = (0.5 * h) / dist;
        std::vector<detail::CompEnt> ve;
        detail::CompEnt proto;
        proto.comp = static_cast<int8_t>(axis);
        detail::CompEnt eF = proto, eC = proto;
        eF.cell = fine;
        eF.w = wF - 0.5;
        eC.cell = coarse;
        eC.w = wC - 0.5;
        ve.push_back(eF);
        ve.push_back(eC);
        detail::cfAppendStencil(ap, t, ve, coarse, fine, axis, wC, fluidOk, scheme, proto);
        for (const auto& e : ve) {
          d.vel.slot.push_back(e.cell);
          d.vel.coef.push_back(e.w);
          d.vel.comp.push_back(e.comp);
          ++d.vel.start[static_cast<std::size_t>(slot) + 1];
        }
        // φ part: uf −= (φ₊−φ₋)/d; the coarse cell's φ is substituted with coarse*.
        const double sideSign = ((dir > 0) == (coarse == j)) ? 1.0 : -1.0;
        std::vector<detail::ScalarEnt> pe;
        detail::cfAppendStencil(ap, t, pe, coarse, fine, axis, -sideSign / dist, fluidOk, scheme);
        for (const auto& e : pe) {
          d.phi.slot.push_back(e.cell);
          d.phi.coef.push_back(e.w);
          ++d.phi.start[static_cast<std::size_t>(slot) + 1];
        }
      }
      ++slot;
    });
  }
  return d;
}

// ---- host applies (the oracle path; the device uses the same CSRs uploaded + SpMV kernels) ----

/// out(i) += Σ coef·f(slot) (the scalar overlay: momentum ∇² delta, gradient delta per axis).
inline void cfApplyHost(const CfCsr& c, const std::vector<double>& f, std::vector<double>& out) {
  const Index n = static_cast<Index>(c.start.size()) - 1;
  for (Index i = 0; i < n; ++i) {
    double acc = 0.0;
    for (Index k = c.start[static_cast<std::size_t>(i)]; k < c.start[static_cast<std::size_t>(i) + 1];
         ++k)
      acc += c.coef[static_cast<std::size_t>(k)] * f[static_cast<std::size_t>(c.slot[static_cast<std::size_t>(k)])];
    out[static_cast<std::size_t>(i)] += acc;
  }
}

/// out(i) += Σ coef·u[comp](slot) (the divergence overlay).
inline void cfApplyCompHost(const CfCompCsr& c, const std::array<std::vector<double>, 3>& u,
                            std::vector<double>& out) {
  const Index n = static_cast<Index>(c.start.size()) - 1;
  for (Index i = 0; i < n; ++i) {
    double acc = 0.0;
    for (Index k = c.start[static_cast<std::size_t>(i)]; k < c.start[static_cast<std::size_t>(i) + 1];
         ++k)
      acc += c.coef[static_cast<std::size_t>(k)] *
             u[static_cast<std::size_t>(c.comp[static_cast<std::size_t>(k)])]
              [static_cast<std::size_t>(c.slot[static_cast<std::size_t>(k)])];
    out[static_cast<std::size_t>(i)] += acc;
  }
}

// ---- device mirrors (Kokkos TUs only; include after a Kokkos-carrying header) ------------------
#ifdef KOKKOS_INLINE_FUNCTION

/// Device mirror of a CfCsr (empty when the scheme is standard / no C/F rows).
struct CfCsrDev {
  Index n = 0, nz = 0;
  View<Index> start, slot;
  View<double> coef;
};
struct CfCompCsrDev {
  Index n = 0, nz = 0;
  View<Index> start, slot;
  View<double> coef;
  View<int8_t> comp;
};

inline CfCsrDev uploadCfCsr(const CfCsr& h, const char* name) {
  CfCsrDev d;
  d.n = static_cast<Index>(h.start.size()) - 1;
  d.nz = static_cast<Index>(h.slot.size());
  if (d.nz == 0) {
    d.n = 0;
    return d;
  }
  d.start = toDevice(h.start, name);
  d.slot = toDevice(h.slot, name);
  d.coef = toDevice(h.coef, name);
  return d;
}
inline CfCompCsrDev uploadCfCompCsr(const CfCompCsr& h, const char* name) {
  CfCompCsrDev d;
  d.n = static_cast<Index>(h.start.size()) - 1;
  d.nz = static_cast<Index>(h.slot.size());
  if (d.nz == 0) {
    d.n = 0;
    return d;
  }
  d.start = toDevice(h.start, name);
  d.slot = toDevice(h.slot, name);
  d.coef = toDevice(h.coef, name);
  d.comp = toDevice(h.comp, name);
  return d;
}

/// out(i) += Σ coef·f(slot).
inline void cfApply(const CfCsrDev& c, View<const double> f, View<double> out) {
  if (c.n == 0)
    return;
  auto st = c.start;
  auto sl = c.slot;
  auto w = c.coef;
  Kokkos::parallel_for(
      "amr::cf_apply", c.n, KOKKOS_LAMBDA(const Index i) {
        double acc = 0.0;
        for (Index k = st(i); k < st(i + 1); ++k)
          acc += w(k) * f(sl(k));
        out(i) += acc;
      });
}

/// out(i) += Σ coef·u[comp](slot).
inline void cfApplyComp(const CfCompCsrDev& c, View<const double> u0, View<const double> u1,
                        View<const double> u2, View<double> out) {
  if (c.n == 0)
    return;
  auto st = c.start;
  auto sl = c.slot;
  auto w = c.coef;
  auto cp = c.comp;
  Kokkos::parallel_for(
      "amr::cf_apply_comp", c.n, KOKKOS_LAMBDA(const Index i) {
        double acc = 0.0;
        for (Index k = st(i); k < st(i + 1); ++k) {
          const int a = cp(k);
          const double v = (a == 0) ? u0(sl(k)) : (a == 1) ? u1(sl(k)) : u2(sl(k));
          acc += w(k) * v;
        }
        out(i) += acc;
      });
}

#endif  // KOKKOS_INLINE_FUNCTION

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_CF_SCHEME_HPP

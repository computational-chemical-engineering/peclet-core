// core — directional ghost-cell projection overlay on the AMR octree (collocated).
//
// The AMR port of flow's collocated set_ghost_projection (flow/src/ghost_projection.hpp §9 of
// flow/doc/collocated_second_order_open_problem.md): the aperture projection's two measured O(1)
// cut-cell defects (tests/study_amr_ghost_apriori.cpp — the gauge-dependent O(1/h) ABC gradient
// and the O(1) truncation of the pinned ½/½ aperture constraint) are removed by
//   * closing every solid/sliver face of the ½/½ face-AVERAGED cell field with the momentum
//     IBM's 1-D wall-anchored closure (peclet::core::scheme — the SAME pure per-face functions
//     flow uses), which after the substitution uf -= grad(phi) makes the pressure operator
//       A = L_bin + Delta:  the symmetric BINARY-openness FV Laplacian (face open iff the face
//       sample and both adjacent centers are fluid — rides the UNCHANGED openness-MG rails as
//       the preconditioner) plus a compact nonsymmetric per-row overlay (this header), both
//       row-rescaled by rho = min(1, min_f D_f);
//   * the directional ghost cell-gradient (AmrFlow::setGhostGradient, already the step-1 hybrid)
//     for the -grad(p^n) predictor and the cell correction.
//
// THE STRUCTURAL INVARIANT that makes the octree port cheap: cut cells live in a uniformly-FINEST
// band (the AmrCutCell same-level contract), so every overlay row's ±2-cell closure reach stays
// inside a locally uniform region — no closure ever crosses a 2:1 level boundary. buildGhostOverlay
// ENFORCES this (throws if a non-clean row's reach touches a different-level leaf): widen the
// refineToSdf band rather than weaken the invariant. Level boundaries then carry only the smooth
// binary operator, which the openness MG already handles conservatively.
//
// Sign conventions (differ from flow's kernels — DERIVED, do not pattern-match): the suite AMR
// operator is the NEGATIVE-definite Laplacian L solved as L phi = div(u*). The ghost system is
//   [rho·(L_bin + Delta)] phi = rho·D_g(u*),   Delta_i = invh² Σ_k sgn_k [w1(X(mn)-X(mn-1)) +
//                                                                          w2(X(mf)-X(mf-1))]
// (flow's gpApplyDelta carries the OPPOSITE sign because its binary operator is -L). The
// divergence delta keeps flow's orientation verbatim: D_g adds sgn·(w1·U(mn) + w2·U(mf)) per
// closure face (EXPLICIT: sgn·U(mg)) with U(m) = ½(u_a(cell m-1) + u_a(cell m)), scaled invh.
// Decoupled rows (no phi coupling at all) are zeroed in both matrix and RHS (phi = 0).
//
// Host-only-safe: the overlay build + host delta appliers compile without Kokkos (the oracle
// path); the device mirror + kernels are guarded on KOKKOS_INLINE_FUNCTION (include this header
// AFTER a Kokkos-carrying header in device TUs).
#ifndef PECLET_CORE_AMR_GHOST_PROJECTION_HPP
#define PECLET_CORE_AMR_GHOST_PROJECTION_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/scheme/ghost_closure.hpp"

namespace peclet::core::amr {

/// Host ghost-projection overlay: one row per non-clean fluid leaf (== cut cell: some ±1 center
/// sample solid). Face slot k = 2*axis + (0 = plus side, 1 = minus side); nbr stores the ±2
/// same-level chain per axis (slot r*15 + a*5 + (q+2), q = -2..2, q=0 the row's own leaf).
struct GhostOverlay {
  Index n = 0;
  std::vector<Index> cell;      ///< [n] leaf index
  std::vector<float> rescale;   ///< [n] rho = min(1, min_f D_f) of the MATRIX weights
  std::vector<int8_t> coupled;  ///< [n] 1 if the row has any phi coupling at all
  std::vector<int8_t> state;    ///< [n*6]
  std::vector<float> th;        ///< [n*6] (diagnostics)
  std::vector<float> w_bc, w_n1, w_n2;  ///< [n*6] RHS/diagnostic closure weights (rhsOrder)
  std::vector<float> wm_n1, wm_n2;      ///< [n*6] matrix (implicit phi) weights (matrixOrder)
  std::vector<Index> nbr;               ///< [n*15] ±2 neighbour chain per axis
  std::vector<double> invh;             ///< [n] 1/cellWidth of the row (finest band)
};

namespace detail {
// Reference adapter so scheme::gpFillRow (which writes ov.field(slot) = …) targets the host SoA.
struct GhostOverlayRef {
  GhostOverlay& g;
  Index& cell(int s) const { return g.cell[static_cast<std::size_t>(s)]; }
  float& rescale(int s) const { return g.rescale[static_cast<std::size_t>(s)]; }
  int8_t& coupled(int s) const { return g.coupled[static_cast<std::size_t>(s)]; }
  int8_t& state(int s) const { return g.state[static_cast<std::size_t>(s)]; }
  float& th(int s) const { return g.th[static_cast<std::size_t>(s)]; }
  float& w_bc(int s) const { return g.w_bc[static_cast<std::size_t>(s)]; }
  float& w_n1(int s) const { return g.w_n1[static_cast<std::size_t>(s)]; }
  float& w_n2(int s) const { return g.w_n2[static_cast<std::size_t>(s)]; }
  float& wm_n1(int s) const { return g.wm_n1[static_cast<std::size_t>(s)]; }
  float& wm_n2(int s) const { return g.wm_n2[static_cast<std::size_t>(s)]; }
};
}  // namespace detail

/// Build the overlay from the octree + the cell-centered SDF samples (AmrCutCell::sdfCRaw).
/// Classification is float, from CELL-CENTERED sdf with face values = mean of the two adjacent
/// centers — identical to flow's buildGpOverlay, so the closures agree with the momentum solid
/// masks. Throws if a non-clean row's ±2 reach touches a different-level leaf (finest-band
/// margin invariant violated — widen the refineToSdf band).
template <unsigned Bits>
inline GhostOverlay buildGhostOverlay(const BlockOctree<3, Bits>& t,
                                      const AmrPoisson<3, Bits>& pres,
                                      const std::vector<double>& sdfC, int matrixOrder,
                                      int rhsOrder) {
  GhostOverlay ov;
  const Index n = t.numLeaves();
  auto sf = [&](Index j) { return static_cast<float>(sdfC[static_cast<std::size_t>(j)]); };
  // scratch single row appended per accepted cell
  for (Index i = 0; i < n; ++i) {
    if (!(sdfC[static_cast<std::size_t>(i)] > 0.0))
      continue;  // solid-centered: decoupled row (phi = 0), not in the overlay
    Index chain[3][5];
    bool clean = true;
    for (int a = 0; a < 3; ++a) {
      chain[a][2] = i;
      chain[a][3] = pres.periodicNeighbor(i, a, +1);
      chain[a][1] = pres.periodicNeighbor(i, a, -1);
      const float c1 = sf(chain[a][1]), c3 = sf(chain[a][3]), c2 = sf(i);
      clean = clean && c1 >= 0.0f && c3 >= 0.0f && 0.5f * (c1 + c2) >= 0.0f &&
              0.5f * (c2 + c3) >= 0.0f;
    }
    if (clean)
      continue;
    // Non-clean ⇒ cut band ⇒ the finest-band contract must hold: the whole ±2 reach same-level.
    const unsigned Li = t.level(i);
    for (int a = 0; a < 3; ++a) {
      chain[a][4] = pres.periodicNeighbor(chain[a][3], a, +1);
      chain[a][0] = pres.periodicNeighbor(chain[a][1], a, -1);
      for (int q = 0; q < 5; ++q)
        if (t.level(chain[a][q]) != Li)
          throw std::runtime_error(
              "amr ghost projection: an overlay row's ±2 closure reach crosses a 2:1 level "
              "boundary — the cut band is too thin; widen the refineToSdf band margin");
    }
    float F[3][4], Cq[3][5];
    for (int a = 0; a < 3; ++a) {
      for (int q = 0; q < 5; ++q)
        Cq[a][q] = sf(chain[a][q]);
      for (int m = 0; m < 4; ++m)  // face i+m-1 = mean of centers q=m-1, q=m (slots m, m+1)
        F[a][m] = 0.5f * (Cq[a][m] + Cq[a][m + 1]);
    }
    const int slot = static_cast<int>(ov.n);
    ov.cell.resize(static_cast<std::size_t>(slot) + 1);
    ov.rescale.resize(static_cast<std::size_t>(slot) + 1);
    ov.coupled.resize(static_cast<std::size_t>(slot) + 1);
    ov.state.resize(static_cast<std::size_t>(slot + 1) * 6);
    ov.th.resize(static_cast<std::size_t>(slot + 1) * 6);
    ov.w_bc.resize(static_cast<std::size_t>(slot + 1) * 6);
    ov.w_n1.resize(static_cast<std::size_t>(slot + 1) * 6);
    ov.w_n2.resize(static_cast<std::size_t>(slot + 1) * 6);
    ov.wm_n1.resize(static_cast<std::size_t>(slot + 1) * 6);
    ov.wm_n2.resize(static_cast<std::size_t>(slot + 1) * 6);
    detail::GhostOverlayRef ref{ov};
    if (!scheme::gpFillRow(ref, slot, i, F, Cq, matrixOrder, rhsOrder)) {
      // pre-check said non-clean but every face classified COUPLED (cannot normally happen —
      // conservative float edge); drop the provisional row.
      ov.cell.resize(static_cast<std::size_t>(slot));
      ov.rescale.resize(static_cast<std::size_t>(slot));
      ov.coupled.resize(static_cast<std::size_t>(slot));
      ov.state.resize(static_cast<std::size_t>(slot) * 6);
      ov.th.resize(static_cast<std::size_t>(slot) * 6);
      ov.w_bc.resize(static_cast<std::size_t>(slot) * 6);
      ov.w_n1.resize(static_cast<std::size_t>(slot) * 6);
      ov.w_n2.resize(static_cast<std::size_t>(slot) * 6);
      ov.wm_n1.resize(static_cast<std::size_t>(slot) * 6);
      ov.wm_n2.resize(static_cast<std::size_t>(slot) * 6);
      continue;
    }
    for (int a = 0; a < 3; ++a)
      for (int q = 0; q < 5; ++q)
        ov.nbr.push_back(chain[a][q]);
    ov.invh.push_back(1.0 / pres.cellWidth(i));
    ++ov.n;
  }
  return ov;
}

/// Binary openness callable factory for the MG surrogate: a face is open iff both adjacent
/// centers (probed at ±h0/2 along the face normal — the adjacent finest-leaf centers in the cut
/// band; deep fluid elsewhere, where the probe distance is irrelevant) are fluid AND the float
/// face sample (mean of the float center samples — the overlay's classification arithmetic) is
/// fluid. Feed to Multigrid::build / AmrPoisson::buildOpenness / AmrMultigrid::setOpenness in
/// ghost mode; the openness-MG hierarchy runs on it unchanged (area-averaged coarse alpha).
template <class SdfFn>
inline auto makeBinaryOpenFn(SdfFn sdfFn, double h0) {
  return [sdfFn, h0](const Vec<3>& fc, int axis) -> double {
    Vec<3> pm = fc, pp = fc;
    pm[axis] -= 0.5 * h0;
    pp[axis] += 0.5 * h0;
    const double sm = sdfFn(pm), sp = sdfFn(pp);
    const float fm = static_cast<float>(sm), fp = static_cast<float>(sp);
    return (sm > 0.0 && sp > 0.0 && 0.5f * (fm + fp) >= 0.0f) ? 1.0 : 0.0;
  };
}

// ---- host (oracle) delta appliers --------------------------------------------------------------

/// Matrix overlay: y currently holds the BINARY L matvec; overwrite the overlay rows with
/// y = rho·(y + invh²·Delta_grid) (decoupled rows → 0). See the sign derivation in the header.
inline void ghostApplyDeltaHost(const GhostOverlay& ov, const std::vector<double>& x,
                                std::vector<double>& y) {
  for (Index r = 0; r < ov.n; ++r) {
    const std::size_t rr = static_cast<std::size_t>(r);
    const Index c = ov.cell[rr];
    if (!ov.coupled[rr]) {
      y[static_cast<std::size_t>(c)] = 0.0;
      continue;
    }
    auto X = [&](int a, int q) {
      return x[static_cast<std::size_t>(ov.nbr[rr * 15 + static_cast<std::size_t>(a) * 5 +
                                               static_cast<std::size_t>(q + 2)])];
    };
    double delta = 0.0;
    for (int k = 0; k < 6; ++k) {
      const int8_t st = ov.state[rr * 6 + static_cast<std::size_t>(k)];
      if (st != scheme::GP_QUAD && st != scheme::GP_LIN)
        continue;
      const int a = k / 2;
      const int sgn = (k & 1) ? -1 : 1;  // odd k = minus side
      const int mn = (k & 1) ? 1 : 0;    // near-face relative index
      const int mf = (k & 1) ? 2 : -1;   // far-face relative index
      const double w1 = ov.wm_n1[rr * 6 + static_cast<std::size_t>(k)];
      const double w2 = ov.wm_n2[rr * 6 + static_cast<std::size_t>(k)];
      delta += sgn * w1 * (X(a, mn) - X(a, mn - 1));  // +axis face gradient — AMR L sign
      if (st == scheme::GP_QUAD && w2 != 0.0)
        delta += sgn * w2 * (X(a, mf) - X(a, mf - 1));
    }
    const double ih = ov.invh[rr];
    y[static_cast<std::size_t>(c)] =
        ov.rescale[rr] * (y[static_cast<std::size_t>(c)] + ih * ih * delta);
  }
}

/// Divergence overlay: d currently holds the BINARY divergence (physical); overwrite overlay rows
/// with d = rho·(d + invh·delta_div) (decoupled → 0). u_bc = 0 (stationary walls).
inline void ghostDivergDeltaHost(const GhostOverlay& ov,
                                 const std::array<std::vector<double>, 3>& u,
                                 std::vector<double>& d) {
  for (Index r = 0; r < ov.n; ++r) {
    const std::size_t rr = static_cast<std::size_t>(r);
    const Index c = ov.cell[rr];
    if (!ov.coupled[rr]) {
      d[static_cast<std::size_t>(c)] = 0.0;
      continue;
    }
    auto U = [&](int a, int m) {  // face-averaged value at face index i+m along axis a
      const Index cm = ov.nbr[rr * 15 + static_cast<std::size_t>(a) * 5 +
                              static_cast<std::size_t>(m + 1)];  // q = m-1
      const Index cp = ov.nbr[rr * 15 + static_cast<std::size_t>(a) * 5 +
                              static_cast<std::size_t>(m + 2)];  // q = m
      return 0.5 * (u[static_cast<std::size_t>(a)][static_cast<std::size_t>(cm)] +
                    u[static_cast<std::size_t>(a)][static_cast<std::size_t>(cp)]);
    };
    double dd = 0.0;
    for (int k = 0; k < 6; ++k) {
      const int8_t st = ov.state[rr * 6 + static_cast<std::size_t>(k)];
      if (st == scheme::GP_COUPLED)
        continue;
      const int a = k / 2;
      const int sgn = (k & 1) ? -1 : 1;
      const int mg = (k & 1) ? 0 : 1;  // the closed face's own index
      const int mn = (k & 1) ? 1 : 0;
      const int mf = (k & 1) ? 2 : -1;
      if (st == scheme::GP_EXPLICIT) {
        dd += sgn * U(a, mg);  // sliver without crossing: explicit u* flux
        continue;
      }
      if (st == scheme::GP_BC_ONLY)
        continue;  // u_bc = 0
      double val = ov.w_n1[rr * 6 + static_cast<std::size_t>(k)] * U(a, mn);
      if (st == scheme::GP_QUAD)
        val += ov.w_n2[rr * 6 + static_cast<std::size_t>(k)] * U(a, mf);
      dd += sgn * val;
    }
    d[static_cast<std::size_t>(c)] =
        ov.rescale[rr] * (d[static_cast<std::size_t>(c)] + ov.invh[rr] * dd);
  }
}

// ---- device mirror + kernels (Kokkos TUs only; include after a Kokkos-carrying header) ---------
#ifdef KOKKOS_INLINE_FUNCTION

/// Device mirror of GhostOverlay (uploaded once per setSolid).
struct GhostOverlayDev {
  Index n = 0;
  View<Index> cell, nbr;
  View<float> rescale;
  View<int8_t> coupled, state;
  View<float> w_n1, w_n2, wm_n1, wm_n2;  // th / w_bc not needed on device (u_bc = 0)
  View<double> invh;
};

inline GhostOverlayDev uploadGhostOverlay(const GhostOverlay& h) {
  GhostOverlayDev d;
  d.n = h.n;
  if (h.n == 0)
    return d;
  d.cell = toDevice(h.cell, "gp_cell");
  d.nbr = toDevice(h.nbr, "gp_nbr");
  d.rescale = toDevice(h.rescale, "gp_rescale");
  d.coupled = toDevice(h.coupled, "gp_coupled");
  d.state = toDevice(h.state, "gp_state");
  d.w_n1 = toDevice(h.w_n1, "gp_wn1");
  d.w_n2 = toDevice(h.w_n2, "gp_wn2");
  d.wm_n1 = toDevice(h.wm_n1, "gp_wmn1");
  d.wm_n2 = toDevice(h.wm_n2, "gp_wmn2");
  d.invh = toDevice(h.invh, "gp_invh");
  return d;
}

/// Device matrix overlay (== ghostApplyDeltaHost). Distinct rows per thread: no atomics.
inline void ghostApplyDelta(const GhostOverlayDev& ov, View<const double> x, View<double> y) {
  if (ov.n == 0)
    return;
  auto cell = ov.cell;
  auto nbr = ov.nbr;
  auto resc = ov.rescale;
  auto cpl = ov.coupled;
  auto st = ov.state;
  auto wm1 = ov.wm_n1;
  auto wm2 = ov.wm_n2;
  auto invh = ov.invh;
  Kokkos::parallel_for(
      "amr::gp_apply_delta", ov.n, KOKKOS_LAMBDA(const Index r) {
        const Index c = cell(r);
        if (!cpl(r)) {
          y(c) = 0.0;
          return;
        }
        double delta = 0.0;
        for (int k = 0; k < 6; ++k) {
          const int8_t s = st(r * 6 + k);
          if (s != scheme::GP_QUAD && s != scheme::GP_LIN)
            continue;
          const int a = k / 2;
          const int sgn = (k & 1) ? -1 : 1;
          const int mn = (k & 1) ? 1 : 0;
          const int mf = (k & 1) ? 2 : -1;
          const double w1 = wm1(r * 6 + k), w2 = wm2(r * 6 + k);
          delta += sgn * w1 * (x(nbr(r * 15 + a * 5 + mn + 2)) - x(nbr(r * 15 + a * 5 + mn + 1)));
          if (s == scheme::GP_QUAD && w2 != 0.0)
            delta +=
                sgn * w2 * (x(nbr(r * 15 + a * 5 + mf + 2)) - x(nbr(r * 15 + a * 5 + mf + 1)));
        }
        const double ih = invh(r);
        y(c) = resc(r) * (y(c) + ih * ih * delta);
      });
}

/// Device divergence overlay (== ghostDivergDeltaHost).
inline void ghostDivergDelta(const GhostOverlayDev& ov, View<const double> u0,
                             View<const double> u1, View<const double> u2, View<double> d) {
  if (ov.n == 0)
    return;
  auto cell = ov.cell;
  auto nbr = ov.nbr;
  auto resc = ov.rescale;
  auto cpl = ov.coupled;
  auto st = ov.state;
  auto w1v = ov.w_n1;
  auto w2v = ov.w_n2;
  auto invh = ov.invh;
  Kokkos::parallel_for(
      "amr::gp_diverg_delta", ov.n, KOKKOS_LAMBDA(const Index r) {
        const Index c = cell(r);
        if (!cpl(r)) {
          d(c) = 0.0;
          return;
        }
        auto U = [&](int a, int m) {
          const Index cm = nbr(r * 15 + a * 5 + m + 1);
          const Index cp = nbr(r * 15 + a * 5 + m + 2);
          const double vm = (a == 0) ? u0(cm) : (a == 1) ? u1(cm) : u2(cm);
          const double vp = (a == 0) ? u0(cp) : (a == 1) ? u1(cp) : u2(cp);
          return 0.5 * (vm + vp);
        };
        double dd = 0.0;
        for (int k = 0; k < 6; ++k) {
          const int8_t s = st(r * 6 + k);
          if (s == scheme::GP_COUPLED)
            continue;
          const int a = k / 2;
          const int sgn = (k & 1) ? -1 : 1;
          const int mg = (k & 1) ? 0 : 1;
          const int mn = (k & 1) ? 1 : 0;
          const int mf = (k & 1) ? 2 : -1;
          if (s == scheme::GP_EXPLICIT) {
            dd += sgn * U(a, mg);
            continue;
          }
          if (s == scheme::GP_BC_ONLY)
            continue;
          double val = w1v(r * 6 + k) * U(a, mn);
          if (s == scheme::GP_QUAD)
            val += w2v(r * 6 + k) * U(a, mf);
          dd += sgn * val;
        }
        d(c) = resc(r) * (d(c) + invh(r) * dd);
      });
}

#endif  // KOKKOS_INLINE_FUNCTION

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_GHOST_PROJECTION_HPP

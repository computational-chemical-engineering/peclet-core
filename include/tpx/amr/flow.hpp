// transport-core — collocated incompressible Stokes step on a BlockOctree.
//
// Wires the two cut-cell IBM halves into one sdflow-style projection step:
//   * momentum  — implicit (backward-Euler) viscous solve per component with the
//                 Dirichlet ξ-polynomial cut-cell operator (AmrCutCell): no-slip
//                 u = 0 on the immersed boundary. Operator (ρ/dt)I − μ∇².
//   * pressure  — the **Almgren–Bell–Colella (ABC) approximate projection** in an
//                 **incremental rotational** form (sdflow's collocated coupling,
//                 src/mac_approx_projection.hpp). The predictor carries the old
//                 pressure gradient −∇p^n; the openness-weighted (Neumann) Poisson
//                 (AmrPoisson) solves ∇²φ = ∇·u*; the cell velocities are corrected
//                 by ½(g⁻+g⁺) of the two adjacent FACE φ-gradients (closed/solid
//                 face ⇒ zero gradient); and the pressure is updated **rotationally**
//                 p += (ρ/dt)φ − μ∇·u*. The −μ∇·u* term removes the projection's
//                 boundary-layer splitting error, so the steady drag is dt-INDEPENDENT
//                 (plain non-incremental Chorin gives an O(dt) drag error — the
//                 reason an earlier version missed Zick & Homsy; see docs/AMR.md).
//                 The openness/aperture is sdflow's gradient-normalised ccFractionCore.
//
// This collocated coupling is a deliberate choice. Do NOT replace it with a
// Rhie–Chow face-velocity interpolation: the small residual *cell* divergence is
// intrinsic to cell-centered velocity placement (the face field is exactly
// divergence-free), not a bug to be engineered away (see the amr-octree memory).
//
// Navier–Stokes (semi-implicit): implicit viscous diffusion + explicit high-order
// advection ∇·(u u) (setAdvection(true)); the high-order flux is second-order
// upwind (SOU) by default, or Koren TVD via setAdvectionScheme(1). setAdvection(false)
// ⇒ Stokes. 3D. Cut cells and the
// ±2-cell advection stencil assume same-level neighbours (resolve the boundary in a
// uniformly-finest band, so the stencils never sit on a 2:1 interface — docs/AMR.md).
// Header-only, guarded by TPX_HAVE_MORTON. Serial/host first.
#ifndef TPX_AMR_FLOW_HPP
#define TPX_AMR_FLOW_HPP

#ifdef TPX_HAVE_MORTON

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/cut_cell.hpp"
#include "tpx/amr/poisson.hpp"
#include "tpx/common/types.hpp"

namespace tpx::amr {

template <unsigned Bits = 21u>
class AmrFlow {
 public:
  using Octree = BlockOctree<3, Bits>;

  void init(const Octree& t, Real h0, Vec<3> origin = Vec<3>{}) {
    t_ = &t;
    h0_ = h0;
    origin_ = origin;
  }

  void setDensity(double rho) { rho_ = rho; }
  void setViscosity(double mu) { mu_ = mu; }
  void setDt(double dt) { dt_ = dt; }
  void setBodyForce(double fx, double fy, double fz) { f_ = {fx, fy, fz}; }
  void setAdvection(bool on) { advect_ = on; }
  /// High-order advection scheme: 0 = second-order upwind (SOU, default), 1 = Koren TVD.
  void setAdvectionScheme(int s) { advScheme_ = s; }
  /// Implicit-FOU deferred-correction advection (default ON): the first-order-upwind
  /// part is solved implicitly (in the momentum operator) and the high-order − FOU
  /// difference is explicit — unconditionally stable for advection. OFF ⇒ fully
  /// explicit high-order advection.
  void setImplicitAdvection(bool on) { implicitFou_ = on; }

  /// Conservative Koren-TVD advection term ∇·(u u_comp) at leaf i (physical units;
  /// the explicit momentum advection). Exposed for testing.
  double advectTerm(int comp, Index i) const { return advectHO(comp, i); }

  /// Build the cut-cell operators from an SDF callable sdfFn(worldPoint) (>0 fluid).
  template <class SdfFn>
  void setSolid(SdfFn&& sdfFn) {
    mom_.init(*t_, h0_, origin_);
    mom_.build(sdfFn, /*idiag=*/rho_ / dt_, /*beta=*/mu_ / (h0_ * h0_));
    pres_.init(*t_, h0_);
    pres_.setOrigin(origin_);
    pres_.buildOpenness([&](const Vec<3>& fc, int axis) { return faceFrac(sdfFn, fc, axis); });
    presMG_.build(*t_, h0_);
    presMG_.setOpenness([&](const Vec<3>& fc, int axis) { return faceFrac(sdfFn, fc, axis); });
    const Index n = t_->numLeaves();
    for (int c = 0; c < 3; ++c) u_[c].assign(static_cast<std::size_t>(n), 0.0);
    phi_.assign(static_cast<std::size_t>(n), 0.0);
    p_.assign(static_cast<std::size_t>(n), 0.0);
  }

  const std::vector<double>& velocity(int c) const { return u_[c]; }
  Index numLeaves() const { return t_->numLeaves(); }
  bool isFluid(Index i) const { return mom_.isFluid(i); }

  /// One Stokes projection step. `momSweeps` Gauss-Seidel sweeps per momentum
  /// component; `presIters`×`presSweeps` for the pressure solve.
  void step(int momSweeps = 200, int presIters = 60, int presSweeps = 4) {
    const Index n = t_->numLeaves();
    // Advection ∇·(u^n u^n_c), evaluated from u^n BEFORE the predictor mutates u_.
    // High-order (SOU/TVD) always; first-order-upwind too for the implicit-FOU
    // deferred correction (the explicit term is ρ·(HO − FOU); FOU is implicit in
    // the momentum operator, rebuilt here from the lagged u^n).
    std::array<std::vector<double>, 3> adv;  // full explicit advection term (incl. ρ)
    if (advect_) {
      if (implicitFou_) mom_.buildAdvectionFou(u_, rho_);  // implicit FOU operator (lagged u^n), C/F-conservative
      for (int c = 0; c < 3; ++c) {
        adv[c].assign(static_cast<std::size_t>(n), 0.0);
        for (Index i = 0; i < n; ++i)
          if (mom_.isFluid(i)) {
            double term = rho_ * advectHO(c, i);                       // ρ·∇·(u u_c), high order
            if (implicitFou_) term -= mom_.fouApply(i, u_[c]);         // − ρ·FOU (deferred correction)
            adv[c][static_cast<std::size_t>(i)] = term;
          }
      }
    }

    // --- momentum predictor: implicit viscous (+ implicit FOU) + body force − advection ---
    for (int c = 0; c < 3; ++c) {
      std::vector<double> src(static_cast<std::size_t>(n), 0.0);
      for (Index i = 0; i < n; ++i)
        if (mom_.isFluid(i)) {
          // incremental predictor: include the old pressure gradient −∇p^n.
          double s = (rho_ / dt_) * u_[c][static_cast<std::size_t>(i)] + f_[c] - gradOf(p_, i, c);
          if (advect_) s -= adv[c][static_cast<std::size_t>(i)];  // HO (explicit) or HO−FOU (deferred)
          src[static_cast<std::size_t>(i)] = s;
        }
      std::vector<double> b = mom_.makeRhs(src, /*u_bc=*/0.0);
      mom_.gaussSeidel(u_[c], b, momSweeps);  // u_ now holds u*
    }
    project(presIters, presSweeps);
  }

  /// Pressure projection of the current velocity in place: solve ∇²φ = ∇·u and
  /// correct u -= ∇φ. (step() calls this after the predictor; also callable alone.)
  void project(int presIters = 60, int presSweeps = 4) {
    const Index n = t_->numLeaves();
    std::vector<double> div(static_cast<std::size_t>(n), 0.0);
    for (Index i = 0; i < n; ++i)
      if (mom_.isFluid(i)) div[static_cast<std::size_t>(i)] = divergence(u_, i);

    std::fill(phi_.begin(), phi_.end(), 0.0);
    (void)presSweeps;
    // Standard (not quadratic) openness V-cycles: the operator L = div(α grad) is
    // consistent with the FV divergence / ABC gradient above (D, G, L share the
    // same face enumeration), so the collocated projection is stable across 2:1
    // interfaces. (The quadratic C/F flux is for pure-Poisson accuracy, not here.)
    for (int it = 0; it < presIters; ++it) presMG_.vcycle(0, phi_, div);

    for (int c = 0; c < 3; ++c)
      for (Index i = 0; i < n; ++i)
        if (mom_.isFluid(i)) u_[c][static_cast<std::size_t>(i)] -= gradOf(phi_, i, c);

    // Rotational incremental pressure update: p += (ρ/dt)φ − μ ∇·u*  (div is ∇·u*).
    // The −μ∇·u* rotational term removes the projection's boundary-layer splitting
    // error, making the steady solution dt-independent (vs plain Chorin).
    for (Index i = 0; i < n; ++i)
      if (mom_.isFluid(i))
        p_[static_cast<std::size_t>(i)] +=
            (rho_ / dt_) * phi_[static_cast<std::size_t>(i)] - mu_ * div[static_cast<std::size_t>(i)];
  }

  /// Openness-weighted FV divergence at leaf i, C/F-consistent: sum over (sub)faces
  /// of α·area·(outward face velocity), face velocity = ½(u_i+u_j); /V_i. Uses the
  /// same face enumeration as the pressure operator (forEachFaceFull).
  double divergence(const std::array<std::vector<double>, 3>& vel, Index i) const {
    double d = 0.0;
    pres_.forEachFaceFull(i, [&](Index j, int axis, int dir, double area, double, double alpha) {
      double ui = vel[axis][static_cast<std::size_t>(i)];
      double uj = vel[axis][static_cast<std::size_t>(j)];  // solid cells hold 0
      d += alpha * area * dir * 0.5 * (ui + uj);
    });
    return d / pres_.cellVolume(i);
  }

  double divNormL2(const std::array<std::vector<double>, 3>& vel) const {
    double s = 0.0;
    const Index n = t_->numLeaves();
    for (Index i = 0; i < n; ++i)
      if (mom_.isFluid(i)) {
        double d = divergence(vel, i);
        s += d * d;
      }
    return std::sqrt(s);
  }

  std::array<std::vector<double>, 3>& velocityRef() { return u_; }

 private:
  // ---- Koren TVD advection (faithful port of sdflow sadv::koren/tvd + cadv::advect) ----
  static double koren(double up_m1, double up, double down, double vel) {
    const double num = up - up_m1, den = down - up;
    double r = (std::fabs(den) < 1e-10) ? 0.0 : num / den;
    double psi = std::fmax(0.0, std::fmin(2.0 * r, std::fmin((1.0 + 2.0 * r) / 3.0, 2.0)));
    return vel * (up + 0.5 * psi * (down - up));
  }
  static double tvd(double LL, double L, double R, double RR, double vel) {
    return (vel > 0.0) ? koren(LL, L, R, vel) : koren(RR, R, L, vel);
  }
  // High-order face value from the two upwind cells (upup,up) and the downwind cell
  // (down): SOU = 1.5·up − 0.5·upup (default); Koren TVD via the limiter (advScheme_).
  double hoFace(double upup, double up, double down) const {
    if (advScheme_ == 0) return 1.5 * up - 0.5 * upup;  // SOU
    double den = down - up;
    double r = (std::fabs(den) < 1e-10) ? 0.0 : (up - upup) / den;
    double psi = std::fmax(0.0, std::fmin(2.0 * r, std::fmin((1.0 + 2.0 * r) / 3.0, 2.0)));
    return up + 0.5 * psi * den;  // Koren TVD
  }

  // C/F-consistent high-order advection ∇·(u u_comp) at leaf i: per (sub)face (via
  // forEachFaceFull, so a coarse cell sums its fine sub-faces — conservative), the
  // outward advective flux α-free·area·velOut·φ_face, with φ_face the SOU/TVD
  // reconstruction from the upwind cell + its upstream neighbour (point-probed, so it
  // works across 2:1 levels). No advection through wall faces. The implicit-FOU half
  // is mom_.fouApply (same faces/velocities), so the deferred correction is the
  // exact high-order − FOU difference.
  double advectHO(int comp, Index i) const {
    double out = 0.0;
    pres_.forEachFaceFull(i, [&](Index j, int axis, int dir, double area, double, double) {
      if (!mom_.isFluid(j)) return;  // no advection through the immersed boundary
      double velOut = dir * 0.5 * (u_[axis][static_cast<std::size_t>(i)] + u_[axis][static_cast<std::size_t>(j)]);
      Index up = (velOut > 0.0) ? i : j, down = (velOut > 0.0) ? j : i;
      int upDir = (up == i) ? -dir : dir;  // upstream direction from the upwind cell
      Index upup = pres_.periodicNeighbor(up, axis, upDir);
      double phiUp = u_[comp][static_cast<std::size_t>(up)];
      double phiUpUp = (upup >= 0 && mom_.isFluid(upup)) ? u_[comp][static_cast<std::size_t>(upup)] : phiUp;
      double phiFace = hoFace(phiUpUp, phiUp, u_[comp][static_cast<std::size_t>(down)]);
      out += area * velOut * phiFace;
    });
    return out / pres_.cellVolume(i);
  }

  // ABC (Almgren-Bell-Colella) cell-velocity correction gradient in direction `c`:
  // ½·(g⁻ + g⁺) of the two adjacent FACE pressure-gradients, where a CLOSED face
  // (openness 0 — solid neighbour) contributes a ZERO gradient (it does NOT read
  // the solid neighbour's φ). Verbatim form of sdflow's projectCorrectCenter
  // (src/mac_approx_projection.hpp) — the collocated approximate projection. This
  // is the chosen collocated coupling; do NOT substitute a Rhie–Chow face-velocity
  // interpolation (see docs/AMR.md and the amr-octree memory).
  double gradOf(const std::vector<double>& fld, Index i, int c) const {
    const double pi = fld[static_cast<std::size_t>(i)];
    double gp = 0, gm = 0;
    int np = 0, nm = 0;
    // Average the +axis-oriented face gradient (φ_j−φ_i)/dist over each side's
    // (sub)faces along axis c; a closed face contributes 0 (ABC). C/F-consistent.
    pres_.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double alpha) {
      if (axis != c || alpha <= 1e-12) return;
      double g = (dir > 0) ? (fld[static_cast<std::size_t>(j)] - pi) / dist
                           : (pi - fld[static_cast<std::size_t>(j)]) / dist;
      if (dir > 0) { gp += g; ++np; } else { gm += g; ++nm; }
    });
    double gpa = np ? gp / np : 0.0, gma = nm ? gm / nm : 0.0;
    return 0.5 * (gpa + gma);
  }

  // Fluid area fraction of a face, the gradient-normalised aperture (a faithful
  // port of sdflow's ccFractionCore, src/mac_cutcell.hpp): frac = 0.5 + sd/denom,
  // sd = SDF at the face centre, denom = (|n_t1| + |n_t2|)·h0 over the two
  // tangential axes (n = unit SDF gradient). This is a linear interface
  // reconstruction within the face — 2nd-order accurate, unlike indicator
  // subsampling (which is only O(1/nsub) on cut faces and made the drag 1st-order).
  template <class SdfFn>
  double faceFrac(SdfFn&& sdfFn, const Vec<3>& fc, int axis) const {
    double sd = sdfFn(fc);
    if (sd <= 0.0) return 0.0;
    Vec<3> g{};
    for (int d = 0; d < 3; ++d) {
      Vec<3> pp = fc, pm = fc;
      pp[d] += h0_;
      pm[d] -= h0_;
      g[d] = (sdfFn(pp) - sdfFn(pm)) / (2.0 * h0_);
    }
    double gmag = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    if (gmag < 1e-6) gmag = 1e-6;
    int t1 = (axis + 1) % 3, t2 = (axis + 2) % 3;
    double denom = (std::fabs(g[t1]) + std::fabs(g[t2])) / gmag * h0_;
    if (denom < 1e-9) denom = 1e-9;
    double frac = 0.5 + sd / denom;
    return frac < 0.0 ? 0.0 : (frac > 1.0 ? 1.0 : frac);
  }

  const Octree* t_ = nullptr;
  Real h0_ = 1.0;
  Vec<3> origin_{};
  double rho_ = 1.0, mu_ = 1.0, dt_ = 1e6;
  bool advect_ = false;
  bool implicitFou_ = true;  // implicit-FOU deferred-correction advection (stable)
  int advScheme_ = 0;        // 0 = SOU (default), 1 = Koren TVD
  Vec<3> f_{};
  AmrCutCell<Bits> mom_;
  AmrPoisson<3, Bits> pres_;       // openness + divergence/gradient access
  AmrMultigrid<3, Bits> presMG_;   // fast (graded-capable) pressure solve
  std::array<std::vector<double>, 3> u_;
  std::vector<double> phi_;  // pressure-increment potential (per projection)
  std::vector<double> p_;    // accumulated pressure (rotational incremental scheme)
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_FLOW_HPP

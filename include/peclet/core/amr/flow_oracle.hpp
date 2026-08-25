// core — collocated incompressible Stokes step on a BlockOctree.
//
// Wires the two cut-cell IBM halves into one flow-style projection step:
//   * momentum  — implicit (backward-Euler) viscous solve per component with the
//                 Dirichlet ξ-polynomial cut-cell operator (AmrCutCell): no-slip
//                 u = 0 on the immersed boundary. Operator (ρ/dt)I − μ∇².
//   * pressure  — the **Almgren–Bell–Colella (ABC) approximate projection** in an
//                 **incremental rotational** form (flow's collocated coupling,
//                 src/mac_approx_projection.hpp). The predictor carries the old
//                 pressure gradient −∇p^n; the openness-weighted (Neumann) Poisson
//                 (AmrPoisson) solves ∇²φ = ∇·u*; the cell velocities are corrected
//                 by ½(g⁻+g⁺) of the two adjacent FACE φ-gradients (closed/solid
//                 face ⇒ zero gradient); and the pressure is updated **rotationally**
//                 p += (ρ/dt)φ − μ∇·u*. The −μ∇·u* term removes the projection's
//                 boundary-layer splitting error, so the steady drag is dt-INDEPENDENT
//                 (plain non-incremental Chorin gives an O(dt) drag error — the
//                 reason an earlier version missed Zick & Homsy; see docs/AMR.md).
//                 The openness/aperture is flow's gradient-normalised ccFractionCore.
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
// Header-only, guarded by PECLET_CORE_HAVE_MORTON. Serial/host first.
#ifndef PECLET_CORE_AMR_FLOW_ORACLE_HPP
#define PECLET_CORE_AMR_FLOW_ORACLE_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "peclet/core/amr/adapt.hpp"         // transferField (conservative remap for finishAdapt)
#include "peclet/core/amr/advect_recon.hpp"  // shared high-order face reconstruction (host+device)
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/cf_scheme.hpp"  // pluggable 2:1 C/F interface schemes (setCfScheme)
#include "peclet/core/amr/cut_cell.hpp"
#include "peclet/core/amr/ghost_projection.hpp"  // directional ghost overlay (setGhostProjection)
#include "peclet/core/amr/ghost_projection_sampled.hpp"  // mixed-level sampled overlay (setGhostSampled)
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"

// Retired from the production path: this serial Gauss-Seidel host driver is kept ONLY as the
// development-stage oracle that the device AmrFlow (flow_device.hpp) is validated bit-for-bit
// against. It is NOT exposed by the Python bindings. Production AMR flow runs on the device path.
namespace peclet::core::amr::oracle {

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
  /// Directional ghost cell-gradient for the −∇pⁿ predictor and the projection's cell
  /// correction (the AMR analog of flow's collocated `set_face_interp(9)` hybrid): on cells with
  /// a solid-centered axis neighbour, gradOf reads the DECOUPLED p=0 of that neighbour through
  /// partially-open faces — a gauge-dependent O(1/h) gradient error at cut cells (measured in
  /// tests/study_amr_ghost_apriori.cpp). The directional gradient is central where both axis
  /// neighbours are fluid-centered and 2nd-order one-sided toward the fluid else — O(h²) and
  /// gauge-exact. The aperture projection (divergence + φ solve + face field) is UNCHANGED, so
  /// this stays throat-safe. Default OFF (bit-identical legacy path). Call before setSolid.
  void setGhostGradient(bool on) { ghostGrad_ = on; }
  /// FULL directional ghost-cell projection (the AMR port of flow's collocated
  /// set_ghost_projection): the pressure system becomes rho·(L_bin + Delta) phi = rho·D_g(u*) —
  /// the BINARY-openness FV Laplacian (preconditioned by the unchanged openness MG) plus the
  /// wall-anchored closure overlay on the finest-band rows, solved by MG-preconditioned BiCGStab;
  /// the constraint is the ghost-closed divergence of the ½/½ face-averaged field. Implies
  /// setGhostGradient (the predictor/correction gradient). (matrixOrder, rhsOrder) = closure
  /// orders for the implicit matrix vs the RHS divergence; (1, 2) is the validated flow default
  /// (2nd-order steady constraint on a 7-point near-symmetric matrix). Default OFF. Call before
  /// setSolid.
  /// QUARANTINED (2026-08-19, mirrors the device AmrFlow): DEFAULT OFF everywhere — the aperture
  /// projection is the production path for Stokes AND NS (the former NS AUTO-arm was retired with
  /// the aperture-PCG-under-advection fix; see docs/amr_aperture_advection_plan.md §RESOLVED).
  /// Engages only on an explicit setGhostProjection(true). Call before setSolid.
  void setGhostProjection(bool on, int matrixOrder = 2, int rhsOrder = 2) {
    ghostProjReq_ = on ? 1 : 0;
    gpMatrixOrder_ = matrixOrder;
    gpRhsOrder_ = rhsOrder;
  }
  /// SAMPLED ghost projection: the mixed-level cut-band prototype (the D1 machinery of
  /// docs/amr_mixed_level_cut_band_plan.md — cut cells at MULTIPLE octree levels, chain
  /// entries across 2:1 boundaries replaced by degree-2 LS virtual samples). Implies
  /// setGhostProjection (call it too, with the closure orders); replaces the finest-band
  /// contract (no band throw) and the openness rule by the level-aware canonical form
  /// (makeBinaryOpenFnMixed). Host-oracle only. Call before setSolid.
  void setGhostSampled(bool on) { ghostSampledReq_ = on ? 1 : 0; }
  /// Coarse/fine (2:1) interface scheme (cf_scheme.hpp): 0 = standard two-point flux (default,
  /// 1st-order at level boundaries, bit-identical legacy path), 1 = Martin–Cartwright tangential
  /// quadratic (2nd-order). Applied to everything the STEADY solution feels: the momentum
  /// diffusion (as a lagged deferred-correction RHS term), the RHS divergence constraint, and
  /// the pressure gradients (predictor + cell correction). The pressure MATRIX / MG hierarchy
  /// stays on the standard consistent operator (the (1,2)-mixed philosophy: at the fixed point
  /// φ→0, the matrix C/F order does not move the steady solution). Call before setSolid.
  void setCfScheme(int scheme) { cfScheme_ = static_cast<CfScheme>(scheme); }

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
    presMG_.build(*t_, h0_);
    // Resolve the projection mode (mirrors the device AmrFlow). DEFAULT = AUTO: the ghost
    // (fluid-only) projection, falling back to the aperture with a stderr notice on a
    // band-margin violation; explicit setGhostProjection(true) keeps the hard throw.
    ghostProj_ = (ghostProjReq_ != 0);
    ghostSampled_ = ghostProj_ && (ghostSampledReq_ == 1);
    if (ghostProj_ && !ghostSampled_) {
      bool viol = false;
      gpOv_ = buildGhostOverlay(*t_, pres_, mom_.sdfCRaw(), gpMatrixOrder_, gpRhsOrder_, &viol);
      if (viol) {
        if (ghostProjReq_ == 1)
          throw std::runtime_error(
              "amr ghost projection: an overlay row's ±2 closure reach crosses a 2:1 level "
              "boundary — widen the refineToSdf band margin");
        ghostProj_ = false;
        fprintf(stderr,
                "peclet::core AmrFlow(oracle): AUTO scheme fell back to the aperture projection "
                "(the finest band is too thin for the ghost overlay).\n");
      }
    }
    if (ghostSampled_)  // mixed-level cut band: sample-slot overlay, no band-margin throw
      gpOvS_ = buildGhostOverlaySampled(*t_, pres_, sdfFn, gpMatrixOrder_, gpRhsOrder_, origin_);
    if (ghostProj_) {
      // Ghost projection: the pressure geometry is the BINARY openness (a face is open iff both
      // adjacent centers + the face sample are fluid), on the UNCHANGED MG rails; the closure
      // physics lives in the overlay. World-coord sampling offset by the origin (the octree's
      // fine units start at origin_): shift the probe like cellCenter does.
      auto sdfW = [&](const Vec<3>& p) { return sdfFn(p); };
      if (ghostSampled_) {
        // Level-aware canonical openness (actual adjacent leaf centers) — the classification
        // primitive the sampled overlay's face states are forced to (no double-counted flux).
        auto binFn = makeBinaryOpenFnMixed(*t_, pres_, sdfW, h0_, origin_);
        pres_.buildOpenness(binFn);
        presMG_.setOpenness(binFn);
      } else {
        auto binFn = makeBinaryOpenFn(sdfW, h0_);
        pres_.buildOpenness(binFn);
        presMG_.setOpenness(binFn);
      }
      ghostGrad_ = true;  // the directional gradient is part of the scheme
      // Fragmentation guard (pockets decoupled) + coupled mask (1 = row in the Krylov space).
      gpPocket_ = findPocketCells(*t_, pres_, mom_.sdfCRaw());
      maskC_.assign(static_cast<std::size_t>(t_->numLeaves()), 0.0);
      for (Index i = 0; i < t_->numLeaves(); ++i)
        maskC_[static_cast<std::size_t>(i)] =
            (mom_.isFluid(i) && !(!gpPocket_.empty() && gpPocket_[static_cast<std::size_t>(i)]))
                ? 1.0
                : 0.0;
      const GhostOverlay& rows = ghostSampled_ ? gpOvS_.base : gpOv_;
      for (Index r = 0; r < rows.n; ++r)
        if (!rows.coupled[static_cast<std::size_t>(r)])
          maskC_[static_cast<std::size_t>(rows.cell[static_cast<std::size_t>(r)])] = 0.0;
    } else {
      gpPocket_.clear();
      pres_.buildOpenness([&](const Vec<3>& fc, int axis) { return faceFrac(sdfFn, fc, axis); });
      presMG_.setOpenness([&](const Vec<3>& fc, int axis) { return faceFrac(sdfFn, fc, axis); });
    }
    // C/F interface scheme overlays (cf_scheme.hpp), built from the SAME host CSR builders the
    // device uses (parity by construction). Rows: fluid; substitution stencils gated on fluid
    // tangential neighbours. The momentum delta uses the α=1 velocity geometry (mom_.lap()),
    // ×μ, on regular (non-cut) fluid rows only (cut rows are finest-band: no C/F faces).
    if (cfScheme_ != CfScheme::standard) {
      auto fluidOk = [&](Index j) { return mom_.isFluid(j); };
      auto rowFluid = [&](Index i) { return mom_.isFluid(i); };
      auto rowRegular = [&](Index i) { return mom_.isFluid(i) && !mom_.isCut(i); };
      cfMom_ = buildCfLapDelta(mom_.lap(), *t_, mu_, rowRegular, fluidOk, cfScheme_);
      cfDiv_ = buildCfDivDelta(pres_, *t_, rowFluid, fluidOk, cfScheme_);
      cfGrad_ = buildCfGradDelta(pres_, *t_, rowFluid, fluidOk, cfScheme_);
      cfUf_ = buildCfUfDelta(pres_, *t_, fluidOk, cfScheme_);
    } else {
      cfMom_ = CfCsr{};
      cfDiv_ = CfCompCsr{};
      cfGrad_ = {};
      cfUf_ = CfUfDelta{};
    }
    const Index n = t_->numLeaves();
    for (int c = 0; c < 3; ++c)
      u_[c].assign(static_cast<std::size_t>(n), 0.0);
    phi_.assign(static_cast<std::size_t>(n), 0.0);
    p_.assign(static_cast<std::size_t>(n), 0.0);
    // Face CSR offsets: count the (sub)faces forEachFaceFull emits per cell (6 interior, more for a
    // cell facing finer neighbours), prefix-sum into faceStart_.
    faceStart_.assign(static_cast<std::size_t>(n) + 1, 0);
    for (Index i = 0; i < n; ++i) {
      Index cnt = 0;
      pres_.forEachFaceFull(i, [&](Index, int, int, double, double, double) { ++cnt; });
      faceStart_[static_cast<std::size_t>(i) + 1] = faceStart_[static_cast<std::size_t>(i)] + cnt;
    }
    uf_.assign(static_cast<std::size_t>(faceStart_[static_cast<std::size_t>(n)]), 0.0);
    faceFieldBuilt_ = false;
  }

  const std::vector<double>& velocity(int c) const { return u_[c]; }
  Index numLeaves() const { return t_->numLeaves(); }
  bool isFluid(Index i) const { return mom_.isFluid(i); }

  // ---- adaptivity during a run (mirror of the device AmrFlow's beginAdapt/finishAdapt) ----------
  void beginAdapt() {
    adaptOldT_ = std::make_unique<Octree>(*t_);
    adaptU_ = u_;
    adaptP_ = p_;
  }
  template <class SdfFn>
  void finishAdapt(SdfFn&& sdfFn) {
    if (!adaptOldT_)
      throw std::runtime_error("oracle::AmrFlow::finishAdapt called without beginAdapt");
    std::array<std::vector<double>, 3> nu;
    for (int c = 0; c < 3; ++c)
      nu[static_cast<std::size_t>(c)] =
          transferField(*adaptOldT_, adaptU_[static_cast<std::size_t>(c)], *t_, /*linear=*/true);
    std::vector<double> np = transferField(*adaptOldT_, adaptP_, *t_, /*linear=*/true);
    setSolid(sdfFn);
    for (Index i = 0; i < t_->numLeaves(); ++i) {
      const bool fl = mom_.isFluid(i);
      for (int c = 0; c < 3; ++c)
        u_[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] =
            fl ? nu[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] : 0.0;
      p_[static_cast<std::size_t>(i)] = fl ? np[static_cast<std::size_t>(i)] : 0.0;
    }
    adaptOldT_.reset();
  }

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
      if (implicitFou_)
        mom_.buildAdvectionFou(u_, rho_, uf_, faceStart_,
                               faceFieldBuilt_);  // FOU advected by the div-free uf
      for (int c = 0; c < 3; ++c) {
        adv[c].assign(static_cast<std::size_t>(n), 0.0);
        for (Index i = 0; i < n; ++i)
          if (mom_.isFluid(i)) {
            double term = rho_ * advectHO(c, i);  // ρ·∇·(u u_c), high order
            if (implicitFou_)
              term -= mom_.fouApply(i, u_[c]);  // − ρ·FOU (deferred correction)
            adv[c][static_cast<std::size_t>(i)] = term;
          }
      }
    }

    // --- momentum predictor: implicit viscous (+ implicit FOU) + body force − advection ---
    for (int c = 0; c < 3; ++c) {
      // C/F-scheme deferred correction on the velocity diffusion: +μ(∇²_scheme − ∇²_std)(uⁿ_c),
      // lagged like the SOU deferred correction ⇒ the steady operator carries the 2nd-order flux.
      std::vector<double> cfm;
      if (cfScheme_ != CfScheme::standard) {
        cfm.assign(static_cast<std::size_t>(n), 0.0);
        cfApplyHost(cfMom_, u_[c], cfm);
      }
      std::vector<double> src(static_cast<std::size_t>(n), 0.0);
      for (Index i = 0; i < n; ++i)
        if (mom_.isFluid(i)) {
          // incremental predictor: include the old pressure gradient −∇p^n.
          double s = (rho_ / dt_) * u_[c][static_cast<std::size_t>(i)] + f_[c] - gradP(p_, i, c);
          if (advect_)
            s -= adv[c][static_cast<std::size_t>(i)];  // HO (explicit) or HO−FOU (deferred)
          if (!cfm.empty())
            s += cfm[static_cast<std::size_t>(i)];
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
      if (mom_.isFluid(i))
        div[static_cast<std::size_t>(i)] = divergence(u_, i);
    if (cfScheme_ != CfScheme::standard)  // 2nd-order C/F face averages in the constraint
      cfApplyCompHost(cfDiv_, u_, div);
    if (ghostProj_) {  // ghost-closed constraint: binary div (above, binary α) + closure overlay
      if (ghostSampled_)
        ghostDivergDeltaSampledHost(gpOvS_, u_, div);
      else
        ghostDivergDeltaHost(gpOv_, u_, div);
    }

    std::fill(phi_.begin(), phi_.end(), 0.0);
    (void)presSweeps;
    if (ghostProj_) {
      // Nonsymmetric ghost operator rho·(L_bin + Delta): MG-preconditioned BiCGStab on the
      // coupled subspace (decoupled rows pinned 0, coupled volume-weighted mean removed).
      solveGhostBiCGStab(phi_, div, presIters);
    } else {
      // Standard (not quadratic) openness V-cycles: the operator L = div(α grad) is
      // consistent with the FV divergence / ABC gradient above (D, G, L share the
      // same face enumeration), so the collocated projection is stable across 2:1
      // interfaces. (The quadratic C/F flux is for pure-Poisson accuracy, not here.)
      for (int it = 0; it < presIters; ++it)
        presMG_.vcycle(0, phi_, div);
    }

    // Build the divergence-free FACE field from u* (still in u_) + φ, before the cell correction.
    buildFaceField();

    for (int c = 0; c < 3; ++c)
      for (Index i = 0; i < n; ++i)
        if (mom_.isFluid(i))
          u_[c][static_cast<std::size_t>(i)] -= gradP(phi_, i, c);

    // Rotational incremental pressure update: p += (ρ/dt)φ − μ ∇·u*  (div is ∇·u*).
    // The −μ∇·u* rotational term removes the projection's boundary-layer splitting
    // error, making the steady solution dt-independent (vs plain Chorin).
    for (Index i = 0; i < n; ++i)
      if (mom_.isFluid(i))
        p_[static_cast<std::size_t>(i)] += (rho_ / dt_) * phi_[static_cast<std::size_t>(i)] -
                                           mu_ * div[static_cast<std::size_t>(i)];
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

  /// Build the ABC/Basilisk divergence-free FACE field from the current cell velocity u* and the
  /// projection potential φ: uf_f = ½(u*_i+u*_j) − (φ₊−φ₋)/d_f, the +axis face velocity. Because
  /// the pressure operator is L = D·G_face with the SAME (sub)faces, this gives D(uf)=D u*−Lφ = 0
  /// exactly (to the φ-solve residual). Call after the pressure solve, before the cell-gradient
  /// correction (so u_ still holds u*). C/F-consistent: a 2:1 sub-face uses (φ_fine−φ_coarse)/d at
  /// the fine area.
  void buildFaceField() {
    const Index n = t_->numLeaves();
    for (Index i = 0; i < n; ++i) {
      Index s = faceStart_[static_cast<std::size_t>(i)];
      pres_.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double) {
        const double uface =
            0.5 * (u_[axis][static_cast<std::size_t>(i)] + u_[axis][static_cast<std::size_t>(j)]);
        // +axis pressure gradient across the face: (φ₊−φ₋)/d, +side = dir>0 ? j : i.
        const double gphi =
            (dir > 0)
                ? (phi_[static_cast<std::size_t>(j)] - phi_[static_cast<std::size_t>(i)]) / dist
                : (phi_[static_cast<std::size_t>(i)] - phi_[static_cast<std::size_t>(j)]) / dist;
        uf_[static_cast<std::size_t>(s++)] = uface - gphi;
      });
    }
    // 2nd-order C/F face values (setCfScheme): distance-weighted average + coarse* substitution
    // on the 2:1 sub-faces — the advecting flux matches the (quad) divergence constraint.
    if (cfScheme_ != CfScheme::standard && !cfUf_.vel.start.empty()) {
      cfApplyCompHost(cfUf_.vel, u_, uf_);
      cfApplyHost(cfUf_.phi, phi_, uf_);
    }
    faceFieldBuilt_ = true;
  }

  /// L2 norm of the divergence of the FACE field uf_ (the div-free flux). After buildFaceField this
  /// is the φ-solve residual (→ machine zero with a tight solve), vs the O(h²) residual of the cell
  /// field.
  double divNormFace() const {
    double tot = 0.0;
    const Index n = t_->numLeaves();
    for (Index i = 0; i < n; ++i) {
      if (!mom_.isFluid(i))
        continue;
      Index s = faceStart_[static_cast<std::size_t>(i)];
      double d = 0.0;
      pres_.forEachFaceFull(i, [&](Index, int, int dir, double area, double, double alpha) {
        d += alpha * area * dir * uf_[static_cast<std::size_t>(s++)];
      });
      d /= pres_.cellVolume(i);
      tot += d * d;
    }
    return std::sqrt(tot);
  }

  /// The divergence-free face field (read-only): uf[faceStart()[i]+s] for cell i's s-th
  /// forEachFaceFull face.
  const std::vector<double>& faceField() const { return uf_; }
  const std::vector<Index>& faceStart() const { return faceStart_; }

 private:
  // ---- Koren TVD advection (faithful port of flow sadv::koren/tvd + cadv::advect) ----
  static double koren(double up_m1, double up, double down, double vel) {
    const double num = up - up_m1, den = down - up;
    double r = (std::fabs(den) < 1e-10) ? 0.0 : num / den;
    double psi = std::fmax(0.0, std::fmin(2.0 * r, std::fmin((1.0 + 2.0 * r) / 3.0, 2.0)));
    return vel * (up + 0.5 * psi * (down - up));
  }
  static double tvd(double LL, double L, double R, double RR, double vel) {
    return (vel > 0.0) ? koren(LL, L, R, vel) : koren(RR, R, L, vel);
  }
  // High-order face value: the SHARED reconstruction (advect_recon.hpp) the device runs too.
  double hoFace(double upup, double up, double down) const {
    return hoFaceValue(upup, up, down, advScheme_);
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
    Index s = faceFieldBuilt_ ? faceStart_[static_cast<std::size_t>(i)] : 0;
    pres_.forEachFaceFull(i, [&](Index j, int axis, int dir, double area, double, double) {
      // Advecting velocity = the divergence-free FACE field uf (built by the previous projection),
      // so the advective flux is conservative (∇·uf = 0) — Bell–Colella–Glaz / Basilisk. Falls back
      // to the simple cell average ½(u_i+u_j) before the first projection has built uf. (At steady
      // state φ→0 ⇒ uf = ½(u_i+u_j), so the steady solution is unchanged; the gain is in the
      // transient.)
      const double uface = faceFieldBuilt_ ? uf_[static_cast<std::size_t>(s)]
                                           : 0.5 * (u_[axis][static_cast<std::size_t>(i)] +
                                                    u_[axis][static_cast<std::size_t>(j)]);
      ++s;
      if (!mom_.isFluid(j))
        return;  // no advection through the immersed boundary
      double velOut = dir * uface;
      Index up = (velOut > 0.0) ? i : j, down = (velOut > 0.0) ? j : i;
      int upDir = (up == i) ? -dir : dir;  // upstream direction from the upwind cell
      Index upup = pres_.periodicNeighbor(up, axis, upDir);
      double phiUp = u_[comp][static_cast<std::size_t>(up)];
      double phiUpUp =
          (upup >= 0 && mom_.isFluid(upup)) ? u_[comp][static_cast<std::size_t>(upup)] : phiUp;
      double phiFace = hoFace(phiUpUp, phiUp, u_[comp][static_cast<std::size_t>(down)]);
      out += area * velOut * phiFace;
    });
    return out / pres_.cellVolume(i);
  }

  // Project onto the coupled subspace: pin decoupled rows (solid-centered + overlay rows with no
  // phi coupling) to 0, remove the volume-weighted mean over the coupled cells (the constant null
  // mode of the connected fluid region — computed over coupled cells ONLY, like maskSolid).
  void gpProject(std::vector<double>& v) const {
    const Index n = t_->numLeaves();
    double su = 0.0, sv = 0.0;
    for (Index i = 0; i < n; ++i) {
      const std::size_t s = static_cast<std::size_t>(i);
      v[s] *= maskC_[s];
      if (maskC_[s] > 0.0) {
        const double V = pres_.cellVolume(i);
        su += V * v[s];
        sv += V;
      }
    }
    const double m = (sv > 0.0) ? su / sv : 0.0;
    for (Index i = 0; i < n; ++i)
      v[static_cast<std::size_t>(i)] -= maskC_[static_cast<std::size_t>(i)] * m;
  }

  // MG-preconditioned BiCGStab on the nonsymmetric ghost pressure operator
  // A(x) = P[rho·(L_bin x + Delta x)] with P the coupled-subspace projection; preconditioner =
  // two binary-openness V-cycles + P. Stagnation guard: stop when the residual makes no new best
  // for 6 iterations (the ghost system has a small attainable-residual floor — flow's measured
  // compatibility gap).
  void solveGhostBiCGStab(std::vector<double>& x, const std::vector<double>& b, int maxIters,
                          double tol = 1e-10) {
    const Index n = t_->numLeaves();
    const std::size_t ns = static_cast<std::size_t>(n);
    auto applyA = [&](const std::vector<double>& v, std::vector<double>& y) {
      pres_.applyLaplacian(v, y);  // binary-openness L (ghost mode geometry)
      if (ghostSampled_)
        ghostApplyDeltaSampledHost(gpOvS_, v, y);
      else
        ghostApplyDeltaHost(gpOv_, v, y);
      gpProject(y);
    };
    auto prec = [&](const std::vector<double>& r, std::vector<double>& z) {
      std::fill(z.begin(), z.end(), 0.0);
      presMG_.vcycle(0, z, r);
      presMG_.vcycle(0, z, r);
      gpProject(z);
    };
    auto dot = [&](const std::vector<double>& a, const std::vector<double>& c) {
      double s = 0.0;
      for (std::size_t i = 0; i < ns; ++i)
        s += a[i] * c[i];
      return s;
    };
    std::vector<double> r(ns), rhat(ns), p(ns, 0.0), phat(ns), v(ns, 0.0), s(ns), shat(ns), t(ns);
    applyA(x, r);
    for (std::size_t i = 0; i < ns; ++i)
      r[i] = b[i] - r[i];
    gpProject(r);
    rhat = r;
    const double res0 = std::sqrt(dot(r, r));
    if (res0 == 0.0)
      return;
    double rho = 1, alpha = 1, omega = 1, best = res0;
    int noImprove = 0;
    for (int it = 0; it < maxIters; ++it) {
      const double rhoNew = dot(rhat, r);
      if (rhoNew == 0.0)
        break;
      const double beta = (rhoNew / rho) * (alpha / omega);
      for (std::size_t i = 0; i < ns; ++i)
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
      prec(p, phat);
      applyA(phat, v);
      const double rhatV = dot(rhat, v);
      if (rhatV == 0.0)
        break;
      alpha = rhoNew / rhatV;
      for (std::size_t i = 0; i < ns; ++i)
        s[i] = r[i] - alpha * v[i];
      double snorm = std::sqrt(dot(s, s));
      if (snorm <= tol * res0) {
        for (std::size_t i = 0; i < ns; ++i)
          x[i] += alpha * phat[i];
        break;
      }
      prec(s, shat);
      applyA(shat, t);
      const double tt = dot(t, t);
      omega = (tt != 0.0) ? dot(t, s) / tt : 0.0;
      for (std::size_t i = 0; i < ns; ++i) {
        x[i] += alpha * phat[i] + omega * shat[i];
        r[i] = s[i] - omega * t[i];
      }
      const double rnorm = std::sqrt(dot(r, r));
      if (rnorm <= tol * res0)
        break;
      if (rnorm < 0.999 * best) {
        best = rnorm;
        noImprove = 0;
      } else if (++noImprove >= 6) {
        break;  // attainable-residual floor (compatibility gap) — stagnation guard
      }
      rho = rhoNew;
      if (omega == 0.0)
        break;
    }
    gpProject(x);
  }

  // Predictor/correction cell gradient dispatch: the ABC gradOf everywhere, EXCEPT — with
  // setGhostGradient — on cut cells (fluid with a solid face neighbour), where gradOf reads the
  // decoupled solid p through partially-open faces (gauge-dependent O(1/h), measured in
  // tests/study_amr_ghost_apriori.cpp) and the directional ghost gradient below is used instead.
  double gradP(const std::vector<double>& fld, Index i, int c) const {
    if (ghostGrad_ && mom_.isCut(i)) {
      if (ghostSampled_) {
        // Mixed-level cut band: the row's sample functionals feed the gradient too (pairing —
        // constraint and gradient from the same closures). Cut cells without a row (clean per
        // the overlay classification) fall through to the level-aware gradOfDir cascade.
        const Index r = gpOvS_.rowOf[static_cast<std::size_t>(i)];
        if (r >= 0)
          return gpsDirGrad(gpOvS_, r, fld, c, 1.0 / pres_.cellWidth(i));
      }
      return gradOfDir(fld, i, c);  // cut band: no C/F faces (band contract) ⇒ no cf delta
    }
    double g = gradOf(fld, i, c);
    // 2nd-order C/F face gradients (level-boundary rows); empty unless built by setSolid.
    if (cfScheme_ != CfScheme::standard && !cfGrad_[static_cast<std::size_t>(c)].start.empty()) {
      const CfCsr& cs = cfGrad_[static_cast<std::size_t>(c)];
      for (Index k = cs.start[static_cast<std::size_t>(i)];
           k < cs.start[static_cast<std::size_t>(i) + 1]; ++k)
        g += cs.coef[static_cast<std::size_t>(k)] *
             fld[static_cast<std::size_t>(cs.slot[static_cast<std::size_t>(k)])];
    }
    return g;
  }

  // Directional ghost cell-gradient on a cut-band cell (flow's gpCenterGrad analog). Cut cells
  // have same-level face neighbours by the finest-band contract, so plain FD applies: central
  // where both axis-neighbour CENTERS are fluid; 2nd-order one-sided toward the fluid else
  // ((−3f_i+4f_{+1}−f_{+2})/2h, with a 2-point fallback when the ±2 cell is solid or not
  // same-level); 0 when sandwiched. Never reads a solid-centered (decoupled) value — O(h²) and
  // exactly gauge-invariant.
  double gradOfDir(const std::vector<double>& fld, Index i, int c) const {
    const double h = pres_.cellWidth(i);
    auto F = [&](Index j) { return fld[static_cast<std::size_t>(j)]; };
    // Pocket cells (fragmentation guard) count as solid: their φ is pinned/decoupled.
    auto ok = [&](Index j) {
      return j >= 0 && mom_.isFluid(j) && t_->level(j) == t_->level(i) &&
             !(!gpPocket_.empty() && gpPocket_[static_cast<std::size_t>(j)]);
    };
    const Index jp = pres_.periodicNeighbor(i, c, +1);
    const Index jm = pres_.periodicNeighbor(i, c, -1);
    const bool ap = ok(jp), am = ok(jm);
    if (am && ap)
      return (F(jp) - F(jm)) / (2.0 * h);
    if (ap) {
      const Index jpp = pres_.periodicNeighbor(jp, c, +1);
      return ok(jpp) ? (-3.0 * F(i) + 4.0 * F(jp) - F(jpp)) / (2.0 * h) : (F(jp) - F(i)) / h;
    }
    if (am) {
      const Index jmm = pres_.periodicNeighbor(jm, c, -1);
      return ok(jmm) ? (3.0 * F(i) - 4.0 * F(jm) + F(jmm)) / (2.0 * h) : (F(i) - F(jm)) / h;
    }
    return 0.0;
  }

  // ABC (Almgren-Bell-Colella) cell-velocity correction gradient in direction `c`:
  // ½·(g⁻ + g⁺) of the two adjacent FACE pressure-gradients, where a CLOSED face
  // (openness 0 — solid neighbour) contributes a ZERO gradient (it does NOT read
  // the solid neighbour's φ). Verbatim form of flow's projectCorrectCenter
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
      if (axis != c || alpha <= 1e-12)
        return;
      double g = (dir > 0) ? (fld[static_cast<std::size_t>(j)] - pi) / dist
                           : (pi - fld[static_cast<std::size_t>(j)]) / dist;
      if (dir > 0) {
        gp += g;
        ++np;
      } else {
        gm += g;
        ++nm;
      }
    });
    double gpa = np ? gp / np : 0.0, gma = nm ? gm / nm : 0.0;
    return 0.5 * (gpa + gma);
  }

  // Fluid area fraction of a face, the gradient-normalised aperture (a faithful
  // port of flow's ccFractionCore, src/mac_cutcell.hpp): frac = 0.5 + sd/denom,
  // sd = SDF at the face centre, denom = (|n_t1| + |n_t2|)·h0 over the two
  // tangential axes (n = unit SDF gradient). This is a linear interface
  // reconstruction within the face — 2nd-order accurate, unlike indicator
  // subsampling (which is only O(1/nsub) on cut faces and made the drag 1st-order).
  template <class SdfFn>
  double faceFrac(SdfFn&& sdfFn, const Vec<3>& fc, int axis) const {
    double sd = sdfFn(fc);
    if (sd <= 0.0)
      return 0.0;
    Vec<3> g{};
    for (int d = 0; d < 3; ++d) {
      Vec<3> pp = fc, pm = fc;
      pp[d] += h0_;
      pm[d] -= h0_;
      g[d] = (sdfFn(pp) - sdfFn(pm)) / (2.0 * h0_);
    }
    double gmag = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    if (gmag < 1e-6)
      gmag = 1e-6;
    int t1 = (axis + 1) % 3, t2 = (axis + 2) % 3;
    double denom = (std::fabs(g[t1]) + std::fabs(g[t2])) / gmag * h0_;
    if (denom < 1e-9)
      denom = 1e-9;
    double frac = 0.5 + sd / denom;
    return frac < 0.0 ? 0.0 : (frac > 1.0 ? 1.0 : frac);
  }

  const Octree* t_ = nullptr;
  Real h0_ = 1.0;
  Vec<3> origin_{};
  double rho_ = 1.0, mu_ = 1.0, dt_ = 1e6;
  bool advect_ = false;
  bool ghostGrad_ = true;   // directional ghost gradient on cut cells (setGhostGradient)
  bool ghostProj_ = false;    // RESOLVED projection mode (set by setSolid from the request)
  int8_t ghostProjReq_ = -1;  // -1 = AUTO (DEFAULT: ghost, aperture fallback on thin band),
                              // 0 = explicit aperture, 1 = explicit ghost
  int gpMatrixOrder_ = 2, gpRhsOrder_ = 2;  // closure orders (2,2 = the production pair; the
                                            // (1,2) mixed form is march-unstable at scale)
  GhostOverlay gpOv_;                       // closure overlay (finest-band rows)
  bool ghostSampled_ = false;               // RESOLVED sampled mode (set by setSolid)
  int8_t ghostSampledReq_ = 0;              // setGhostSampled request (mixed-level prototype)
  GhostOverlaySampled gpOvS_;               // sample-slot overlay (mixed-level cut band)
  std::vector<double> maskC_;               // 1 = coupled row (Krylov subspace), 0 = pinned
  std::vector<char> gpPocket_;              // fragmentation guard: 1 = decoupled pocket cell
  CfScheme cfScheme_ = CfScheme::standard;  // 2:1 C/F interface scheme (setCfScheme)
  CfCsr cfMom_;                             // +μ(∇²_scheme − ∇²_std) momentum RHS overlay
  CfCompCsr cfDiv_;                         // (D_scheme − D_std) divergence overlay
  std::array<CfCsr, 3> cfGrad_;             // (G_scheme − G_std) per gradient axis
  CfUfDelta cfUf_;                          // (uf_scheme − uf_std) face-field overlay (slots)
  bool implicitFou_ = true;  // implicit-FOU deferred-correction advection (stable)
  int advScheme_ = 0;        // 0 = SOU (default), 1 = Koren TVD
  Vec<3> f_{};
  AmrCutCell<Bits> mom_;
  AmrPoisson<3, Bits> pres_;      // openness + divergence/gradient access
  AmrMultigrid<3, Bits> presMG_;  // fast (graded-capable) pressure solve
  std::array<std::vector<double>, 3> u_;
  std::vector<double> phi_;  // pressure-increment potential (per projection)
  std::vector<double> p_;    // accumulated pressure (rotational incremental scheme)
  // Basilisk/ABC divergence-free FACE field: uf_[faceStart_[i]+s] is the +axis velocity through
  // cell i's s-th (sub)face, in forEachFaceFull order. Each internal (sub)face is stored from BOTH
  // incident cells; the orientation-based build keeps the two copies identical. At a 2:1 interface
  // a coarse cell owns 2^(Dim-1) fine sub-faces and the fine cell owns its single face — so the
  // face field is at the finest resolution touching each face and the coarse-cell divergence sums
  // its sub-faces.
  std::vector<Index> faceStart_;  // CSR offsets into uf_, size n+1
  std::vector<double> uf_;        // +axis face velocity per (cell,face) slot
  std::unique_ptr<Octree> adaptOldT_;          // beginAdapt topology snapshot
  std::array<std::vector<double>, 3> adaptU_;  // beginAdapt field snapshots
  std::vector<double> adaptP_;
  bool faceFieldBuilt_ =
      false;  // uf_ populated by a projection (else advection falls back to ½(u_i+u_j))
};

}  // namespace peclet::core::amr::oracle

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_FLOW_ORACLE_HPP

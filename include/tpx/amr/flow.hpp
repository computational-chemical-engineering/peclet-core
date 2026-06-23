// transport-core — collocated incompressible Stokes step on a BlockOctree.
//
// Wires the two cut-cell IBM halves into one sdflow-style projection step:
//   * momentum  — implicit (backward-Euler) viscous solve per component with the
//                 Dirichlet ξ-polynomial cut-cell operator (AmrCutCell): no-slip
//                 u = 0 on the immersed boundary. Operator (ρ/dt)I − μ∇².
//   * pressure  — the **Almgren–Bell–Colella (ABC) approximate projection**, the
//                 collocated coupling sdflow uses (src/mac_approx_projection.hpp):
//                 average the cell velocities onto a face (MAC) divergence, solve
//                 the openness-weighted (Neumann) pressure Poisson (AmrPoisson)
//                 ∇²φ = ∇·u*, then correct the cell velocities by ½(g⁻+g⁺) of the
//                 two adjacent FACE φ-gradients (a closed/solid face contributes a
//                 zero gradient). The *face* field is exactly divergence-free; the
//                 cell field is approximately so — hence "approximate projection".
//
// This collocated coupling is a deliberate choice. Do NOT replace it with a
// Rhie–Chow face-velocity interpolation: the small residual cell divergence is
// intrinsic to cell-centered velocity placement (see the amr-octree memory note),
// not a bug to be engineered away.
//
// Stokes only (advection — staggered/collocated Koren TVD — is a follow-up). 3D.
// Cut cells are assumed same-level (resolve the boundary in a uniformly-finest
// band, so the cut-cell stencils never sit on a 2:1 interface — see docs/AMR.md).
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

  /// Build the cut-cell operators from an SDF callable sdfFn(worldPoint) (>0 fluid).
  template <class SdfFn>
  void setSolid(SdfFn&& sdfFn) {
    mom_.init(*t_, h0_, origin_);
    mom_.build(sdfFn, /*idiag=*/rho_ / dt_, /*beta=*/mu_ / (h0_ * h0_));
    pres_.init(*t_, h0_);
    pres_.setOrigin(origin_);
    pres_.buildOpenness([&](const Vec<3>& fc, int axis) { return faceFrac(sdfFn, fc, axis); });
    const Index n = t_->numLeaves();
    for (int c = 0; c < 3; ++c) u_[c].assign(static_cast<std::size_t>(n), 0.0);
    phi_.assign(static_cast<std::size_t>(n), 0.0);
  }

  const std::vector<double>& velocity(int c) const { return u_[c]; }
  Index numLeaves() const { return t_->numLeaves(); }
  bool isFluid(Index i) const { return mom_.isFluid(i); }

  /// One Stokes projection step. `momSweeps` Gauss-Seidel sweeps per momentum
  /// component; `presIters`×`presSweeps` for the pressure solve.
  void step(int momSweeps = 200, int presIters = 60, int presSweeps = 4) {
    const Index n = t_->numLeaves();
    // --- momentum predictor (per component, implicit viscous + body force) ---
    for (int c = 0; c < 3; ++c) {
      std::vector<double> src(static_cast<std::size_t>(n), 0.0);
      for (Index i = 0; i < n; ++i)
        if (mom_.isFluid(i))
          src[static_cast<std::size_t>(i)] = (rho_ / dt_) * u_[c][static_cast<std::size_t>(i)] + f_[c];
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
    for (int it = 0; it < presIters; ++it) {
      pres_.gaussSeidel(phi_, div, presSweeps);
      pres_.removeMean(phi_);
    }

    for (int c = 0; c < 3; ++c)
      for (Index i = 0; i < n; ++i)
        if (mom_.isFluid(i)) u_[c][static_cast<std::size_t>(i)] -= gradPhi(i, c);
  }

  /// Openness-weighted (FV) divergence of velocity at leaf i (diagnostic / test).
  double divergence(const std::array<std::vector<double>, 3>& vel, Index i) const {
    double d = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      Index jp = mom_.neighborOf(i, 2 * axis);      // +axis
      Index jm = mom_.neighborOf(i, 2 * axis + 1);  // -axis
      double ui = vel[axis][static_cast<std::size_t>(i)];
      double up = (jp >= 0 && mom_.isFluid(jp)) ? vel[axis][static_cast<std::size_t>(jp)] : 0.0;
      double um = (jm >= 0 && mom_.isFluid(jm)) ? vel[axis][static_cast<std::size_t>(jm)] : 0.0;
      double ap = pres_.faceOpenness(i, axis, +1);
      double am = pres_.faceOpenness(i, axis, -1);
      d += (ap * 0.5 * (ui + up) - am * 0.5 * (ui + um)) / h0_;
    }
    return d;
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
  // ABC (Almgren-Bell-Colella) cell-velocity correction gradient in direction `c`:
  // ½·(g⁻ + g⁺) of the two adjacent FACE pressure-gradients, where a CLOSED face
  // (openness 0 — solid neighbour) contributes a ZERO gradient (it does NOT read
  // the solid neighbour's φ). Verbatim form of sdflow's projectCorrectCenter
  // (src/mac_approx_projection.hpp) — the collocated approximate projection. This
  // is the chosen collocated coupling; do NOT substitute a Rhie–Chow face-velocity
  // interpolation (see docs/AMR.md and the amr-octree memory).
  double gradPhi(Index i, int c) const {
    Index jp = mom_.neighborOf(i, 2 * c);
    Index jm = mom_.neighborOf(i, 2 * c + 1);
    double pi = phi_[static_cast<std::size_t>(i)];
    double ap = pres_.faceOpenness(i, c, +1), am = pres_.faceOpenness(i, c, -1);
    double gp = (ap > 1e-12 && jp >= 0) ? phi_[static_cast<std::size_t>(jp)] - pi : 0.0;  // g⁺
    double gm = (am > 1e-12 && jm >= 0) ? pi - phi_[static_cast<std::size_t>(jm)] : 0.0;  // g⁻
    return 0.5 * (gm + gp) / h0_;
  }

  // Fluid area fraction of a face (subsampled), for the pressure openness.
  template <class SdfFn>
  double faceFrac(SdfFn&& sdfFn, const Vec<3>& fc, int axis) const {
    const int ns = 4;
    int in = 0, tot = 0;
    for (int a = 0; a < ns; ++a)
      for (int b = 0; b < ns; ++b) {
        Vec<3> p = fc;
        int t1 = (axis + 1) % 3, t2 = (axis + 2) % 3;
        p[t1] += ((a + 0.5) / ns - 0.5) * h0_;
        p[t2] += ((b + 0.5) / ns - 0.5) * h0_;
        if (sdfFn(p) > 0.0) ++in;
        ++tot;
      }
    return static_cast<double>(in) / tot;
  }

  const Octree* t_ = nullptr;
  Real h0_ = 1.0;
  Vec<3> origin_{};
  double rho_ = 1.0, mu_ = 1.0, dt_ = 1e6;
  Vec<3> f_{};
  AmrCutCell<Bits> mom_;
  AmrPoisson<3, Bits> pres_;
  std::array<std::vector<double>, 3> u_;
  std::vector<double> phi_;
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_FLOW_HPP

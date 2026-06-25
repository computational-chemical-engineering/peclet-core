// transport-core — device (Kokkos) collocated incompressible Stokes step on a BlockOctree.
//
// The device counterpart of AmrFlow (flow.hpp): the whole cut-cell IBM projection step
// runs in Kokkos kernels instead of the host-serial gaussSeidel + host projection the drag
// study found to be the bottleneck. It reuses the host AmrCutCell / AmrPoisson to *build*
// the operators (geometry, openness, cut stencils — done once), then drives the time step
// entirely on the device:
//   * momentum predictor — DeviceMomentumOp (assembled cut-cell operator) solved with the
//     parallel BiCGStab of device_momentum.hpp;
//   * pressure projection — the openness Poisson on DeviceMultigrid / DevicePCG;
//   * divergence, ABC gradient correction, rotational pressure update — face-CSR kernels
//     (DeviceFaceGeom) that mirror AmrPoisson::forEachFaceFull (same 2:1 sub-faces +
//     openness), so D / G / L stay consistent exactly as in the host collocated coupling.
//
// Scope: Stokes (advection off) — the Zick&Homsy drag benchmark the host path validates
// against. The explicit high-order advection deferred-correction term (advectHO) is not yet
// ported; the implicit-FOU part already rides in the momentum operator via assembleOperator.
//
// Requires a Kokkos build + the morton checkout (TPX_HAVE_MORTON).
#ifndef TPX_AMR_DEVICE_FLOW_HPP
#define TPX_AMR_DEVICE_FLOW_HPP

#ifdef TPX_HAVE_MORTON

#include <array>
#include <cmath>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/cut_cell.hpp"
#include "tpx/amr/device_momentum.hpp"
#include "tpx/amr/device_multigrid.hpp"
#include "tpx/amr/device_pcg.hpp"
#include "tpx/amr/poisson.hpp"
#include "tpx/common/types.hpp"
#include "tpx/common/view.hpp"

namespace tpx::amr {

// ===========================================================================
// DeviceFaceGeom — the static (sub)face topology+geometry of the collocated
// projection as a per-cell face CSR, matching AmrPoisson::forEachFaceFull. Per
// face: neighbour, axis, dir, α·area (divergence weight), distance, α (gradient
// gate). Per cell: 1/V and a fluid flag.
// ===========================================================================
struct DeviceFaceGeom {
  View<Index> start;       ///< CSR row offsets, size n+1
  View<Index> nbr;         ///< neighbour leaf per face, size nFaces
  View<int> axis;          ///< face axis 0/1/2, size nFaces
  View<int> dir;           ///< face direction +1/-1, size nFaces
  View<double> alphaArea;  ///< α·area (physical) per face, size nFaces
  View<double> dist;       ///< face-normal distance (physical) per face, size nFaces
  View<double> alpha;      ///< openness per face (gradient gate), size nFaces
  View<double> invVol;     ///< 1/V_i per cell, size n
  View<char> fluid;        ///< per-cell fluid flag, size n
  Index n = 0;
};

/// Build DeviceFaceGeom from a built AmrPoisson (openness set) + a fluid predicate.
template <int Dim, unsigned Bits, class FluidFn>
DeviceFaceGeom buildFaceGeom(const AmrPoisson<Dim, Bits>& ap, FluidFn&& isFluid) {
  const Index n = ap.octree().numLeaves();
  std::vector<Index> start(static_cast<std::size_t>(n) + 1, 0);
  for (Index i = 0; i < n; ++i) {
    Index cnt = 0;
    ap.forEachFaceFull(i, [&](Index, int, int, double, double, double) { ++cnt; });
    start[static_cast<std::size_t>(i) + 1] = start[static_cast<std::size_t>(i)] + cnt;
  }
  const Index nf = start[static_cast<std::size_t>(n)];
  std::vector<Index> nbr(static_cast<std::size_t>(nf));
  std::vector<int> axis(static_cast<std::size_t>(nf)), dir(static_cast<std::size_t>(nf));
  std::vector<double> aArea(static_cast<std::size_t>(nf)), dist(static_cast<std::size_t>(nf)),
      alpha(static_cast<std::size_t>(nf));
  std::vector<double> invVol(static_cast<std::size_t>(n));
  std::vector<char> fluid(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    invVol[static_cast<std::size_t>(i)] = 1.0 / ap.cellVolume(i);
    fluid[static_cast<std::size_t>(i)] = isFluid(i) ? 1 : 0;
    Index k = start[static_cast<std::size_t>(i)];
    ap.forEachFaceFull(i, [&](Index j, int ax, int dr, double area, double d, double al) {
      nbr[static_cast<std::size_t>(k)] = j;
      axis[static_cast<std::size_t>(k)] = ax;
      dir[static_cast<std::size_t>(k)] = dr;
      aArea[static_cast<std::size_t>(k)] = al * area;
      dist[static_cast<std::size_t>(k)] = d;
      alpha[static_cast<std::size_t>(k)] = al;
      ++k;
    });
  }
  DeviceFaceGeom g;
  g.n = n;
  g.start = toDevice(start, "fg_start");
  g.nbr = toDevice(nbr, "fg_nbr");
  g.axis = toDevice(axis, "fg_axis");
  g.dir = toDevice(dir, "fg_dir");
  g.alphaArea = toDevice(aArea, "fg_aarea");
  g.dist = toDevice(dist, "fg_dist");
  g.alpha = toDevice(alpha, "fg_alpha");
  g.invVol = toDevice(invVol, "fg_invvol");
  g.fluid = toDevice(fluid, "fg_fluid");
  return g;
}

/// Openness-weighted FV divergence: div_i = invVol_i Σ_faces α·area·dir·½(u^axis_i+u^axis_j),
/// on fluid cells (0 elsewhere). u[0..2] are the three velocity component Views (solid cells
/// must hold 0). Mirrors AmrFlow::divergence.
inline void deviceDivergence(const DeviceFaceGeom& g, View<const double> u0, View<const double> u1,
                             View<const double> u2, View<double> div) {
  auto st = g.start;
  auto nb = g.nbr;
  auto ax = g.axis;
  auto dr = g.dir;
  auto aA = g.alphaArea;
  auto iv = g.invVol;
  auto fl = g.fluid;
  Kokkos::parallel_for(
      "amr::flow_div", g.n, KOKKOS_LAMBDA(const Index i) {
        if (!fl(i)) {
          div(i) = 0.0;
          return;
        }
        double d = 0.0;
        for (Index k = st(i); k < st(i + 1); ++k) {
          const int a = ax(k);
          const double ui = (a == 0) ? u0(i) : (a == 1) ? u1(i) : u2(i);
          const Index j = nb(k);
          const double uj = (a == 0) ? u0(j) : (a == 1) ? u1(j) : u2(j);
          d += aA(k) * dr(k) * 0.5 * (ui + uj);
        }
        div(i) = d * iv(i);
      });
}

/// ABC cell-gradient of a scalar field `f`: gx/gy/gz = ½(g⁻+g⁺) of the adjacent face
/// gradients along each axis, a closed face (α≤1e-12) contributing nothing (and not
/// counted). On fluid cells only. Mirrors AmrFlow::gradOf for all three components.
inline void deviceGrad3(const DeviceFaceGeom& g, View<const double> f, View<double> gx,
                        View<double> gy, View<double> gz) {
  auto st = g.start;
  auto nb = g.nbr;
  auto ax = g.axis;
  auto dr = g.dir;
  auto di = g.dist;
  auto al = g.alpha;
  auto fl = g.fluid;
  Kokkos::parallel_for(
      "amr::flow_grad3", g.n, KOKKOS_LAMBDA(const Index i) {
        if (!fl(i)) {
          gx(i) = gy(i) = gz(i) = 0.0;
          return;
        }
        const double fi = f(i);
        double gp[3] = {0, 0, 0}, gm[3] = {0, 0, 0};
        int np[3] = {0, 0, 0}, nm[3] = {0, 0, 0};
        for (Index k = st(i); k < st(i + 1); ++k) {
          if (al(k) <= 1e-12) continue;
          const int a = ax(k);
          const double gg =
              (dr(k) > 0) ? (f(nb(k)) - fi) / di(k) : (fi - f(nb(k))) / di(k);
          if (dr(k) > 0) {
            gp[a] += gg;
            ++np[a];
          } else {
            gm[a] += gg;
            ++nm[a];
          }
        }
        double out[3];
        for (int a = 0; a < 3; ++a) {
          const double a1 = np[a] ? gp[a] / np[a] : 0.0;
          const double a2 = nm[a] ? gm[a] / nm[a] : 0.0;
          out[a] = 0.5 * (a1 + a2);
        }
        gx(i) = out[0];
        gy(i) = out[1];
        gz(i) = out[2];
      });
}

/// Momentum RHS for one component: b_i = fluid ? (idiag·u_i + f_c − gradP_i)·rscale_i : 0
/// (== AmrCutCell::makeRhs of the AmrFlow predictor source, u_bc = 0).
inline void deviceMomRhs(View<const double> uc, View<const double> gradP, View<const double> rscale,
                         View<const char> fluid, double idiag, double fc, View<double> b, Index n) {
  Kokkos::parallel_for(
      "amr::flow_momrhs", n, KOKKOS_LAMBDA(const Index i) {
        b(i) = fluid(i) ? (idiag * uc(i) + fc - gradP(i)) * rscale(i) : 0.0;
      });
}

/// u_c -= gradPhi_c on fluid cells (the projection velocity correction).
inline void deviceCorrect(View<double> uc, View<const double> gphi, View<const char> fluid,
                          Index n) {
  Kokkos::parallel_for(
      "amr::flow_correct", n,
      KOKKOS_LAMBDA(const Index i) { if (fluid(i)) uc(i) -= gphi(i); });
}

/// Rotational incremental pressure update: p += (ρ/dt)φ − μ·div, on fluid cells.
inline void devicePresUpdate(View<double> p, View<const double> phi, View<const double> div,
                             View<const char> fluid, double rho_dt, double mu, Index n) {
  Kokkos::parallel_for(
      "amr::flow_presupd", n, KOKKOS_LAMBDA(const Index i) {
        if (fluid(i)) p(i) += rho_dt * phi(i) - mu * div(i);
      });
}

// ===========================================================================
// DeviceAmrFlow — collocated Stokes projection step, fully on device.
// ===========================================================================
template <unsigned Bits = 21u>
class DeviceAmrFlow {
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
  /// Use MG-preconditioned CG for the pressure solve (default) vs plain V-cycles.
  void setPressurePCG(bool on) { presPCG_ = on; }
  /// Opt-in (default OFF): use the openness-Helmholtz multigrid V-cycle as the momentum
  /// BiCGStab preconditioner instead of damped-Jacobi. NOTE: experimental and currently NOT
  /// a win for the cut-cell Dirichlet momentum operator at large dt — the Neumann/openness
  /// hierarchy mismatches the no-slip (Dirichlet) immersed boundary, and the shift floor that
  /// keeps solid cells stable (max(idiag, μ/L²)) over-weights the reaction vs the true tiny
  /// idiag, so it converges slower than Jacobi-BiCGStab there. A proper Dirichlet-aware
  /// cut-cell velocity-MG (ε-solid-on-coarse, volume-fraction coarsening) is the real fix.
  /// Either way the converged step is identical (preconditioner-only). Kept for moderate-dt /
  /// wall-bounded (non-immersed) problems where the Helmholtz match is good.
  void setMomentumMG(bool on) { momMGon_ = on; }

  /// Build the cut-cell operators (host) + upload all device structures. Requires the
  /// density / viscosity / dt to be set first (the momentum operator carries ρ/dt and μ).
  template <class SdfFn>
  void setSolid(SdfFn&& sdfFn) {
    const Index n = t_->numLeaves();
    // Host operator build (geometry, openness, cut stencils) — same as AmrFlow::setSolid.
    mom_.init(*t_, h0_, origin_);
    mom_.build(sdfFn, /*idiag=*/rho_ / dt_, /*beta=*/mu_ / (h0_ * h0_));
    pres_.init(*t_, h0_);
    pres_.setOrigin(origin_);
    auto openFn = [&](const Vec<3>& fc, int axis) { return faceFrac(sdfFn, fc, axis); };
    pres_.buildOpenness(openFn);
    presMG_.build(*t_, h0_, openFn, /*periodic=*/true);
    // Momentum preconditioner: the openness hierarchy turned into the Helmholtz operator
    // c0·I − μ∇², c0 = max(idiag, μ/L²). The shift floor μ/L² (L = domain size; the slowest
    // diffusion mode) is essential: at large dt idiag→0, so a *solid* (fully-closed) cell —
    // whose only diagonal contribution is c0 — would get preconditioner diagonal ≈idiag and
    // amplify its residual by 1/idiag (~1e6), diverging the BiCGStab. Flooring c0 to μ/L²
    // keeps solid cells ~O(1) (matching the operator's identity solid rows) while being
    // negligible against the fluid diffusion (~μ/h²). The shift only changes the
    // preconditioner (the BiCGStab matvec is the exact momentum operator) ⇒ same solution.
    if (momMGon_) {
      const double Ldom = h0_ * static_cast<double>(t_->brick()[0] * (Index(1) << t_->lmax()));
      const double c0 = std::max(rho_ / dt_, mu_ / (Ldom * Ldom));
      momMG_.build(*t_, h0_, openFn, /*periodic=*/true);
      momMG_.setHelmholtz(c0, -mu_);
    }

    // Device upload: momentum operator CSR, face geometry, rscale, fluid flags.
    auto A = mom_.assembleOperator();
    momOp_.n = n;
    momOp_.diag = toDevice(A.diag, "df_diag");
    momOp_.faceStart = toDevice(A.start, "df_fstart");
    momOp_.faceNbr = toDevice(A.nbr, "df_fnbr");
    momOp_.faceCoef = toDevice(A.coef, "df_fcoef");
    geom_ = buildFaceGeom(pres_, [&](Index i) { return mom_.isFluid(i); });
    std::vector<double> rs(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) rs[static_cast<std::size_t>(i)] = mom_.rhsScale(i);
    rscale_ = toDevice(rs, "df_rscale");
    fluid_ = geom_.fluid;

    // Device state.
    for (int c = 0; c < 3; ++c) {
      u_[c] = View<double>("df_u", static_cast<std::size_t>(n));
      gx_[c] = View<double>("df_g", static_cast<std::size_t>(n));
      Kokkos::deep_copy(u_[c], 0.0);
    }
    p_ = View<double>("df_p", static_cast<std::size_t>(n));
    phi_ = View<double>("df_phi", static_cast<std::size_t>(n));
    div_ = View<double>("df_div", static_cast<std::size_t>(n));
    bmom_ = View<double>("df_bmom", static_cast<std::size_t>(n));
    Kokkos::deep_copy(p_, 0.0);
    momSolver_.setJacobi(2, 0.7);
    if (momMGon_) momSolver_.setMgPreconditioner(&momMG_, 1);
    pcg_.setVcycle(2, 2, 60, 0.8);
    pcg_.setSingular(true);
    n_ = n;
  }

  /// One Stokes projection step on device. `momIters` BiCGStab iterations for each momentum
  /// component; `presIters` PCG iterations (or V-cycles) for the pressure solve.
  void step(int momIters = 100, int presIters = 60) {
    const Index n = n_;
    const double idiag = rho_ / dt_;
    lastMomIters_ = 0;
    // --- predictor: incremental BE viscous solve per component, RHS carries −∇p^n ---
    deviceGrad3(geom_, View<const double>(p_), gx_[0], gx_[1], gx_[2]);
    for (int c = 0; c < 3; ++c) {
      deviceMomRhs(View<const double>(u_[c]), View<const double>(gx_[c]),
                   View<const double>(rscale_), View<const char>(fluid_), idiag, f_[c], bmom_, n);
      // warm start from u^n (good initial guess for the time step).
      lastMomIters_ += momSolver_.solveBiCGStab(momOp_, u_[c], View<const double>(bmom_), momIters, 1e-10).iters;
    }
    project(presIters);
  }

  /// Pressure projection of the current velocity in place.
  void project(int presIters = 60) {
    const Index n = n_;
    deviceDivergence(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                     View<const double>(u_[2]), div_);
    Kokkos::deep_copy(phi_, 0.0);
    if (presPCG_) {
      lastPresIters_ = pcg_.solve(presMG_, phi_, View<const double>(div_), presIters, 1e-10).iters;
    } else {
      Kokkos::deep_copy(presMG_.b(0), div_);
      Kokkos::deep_copy(presMG_.x(0), 0.0);
      for (int it = 0; it < presIters; ++it) presMG_.vcycle(2, 2, 60, 0.8);
      Kokkos::deep_copy(phi_, presMG_.x(0));
    }
    deviceGrad3(geom_, View<const double>(phi_), gx_[0], gx_[1], gx_[2]);
    for (int c = 0; c < 3; ++c) deviceCorrect(u_[c], View<const double>(gx_[c]), View<const char>(fluid_), n);
    devicePresUpdate(p_, View<const double>(phi_), View<const double>(div_),
                     View<const char>(fluid_), rho_ / dt_, mu_, n);
  }

  /// Copy a velocity component back to host.
  std::vector<double> velocity(int c) const {
    std::vector<double> h(static_cast<std::size_t>(n_));
    auto m = Kokkos::create_mirror_view(u_[c]);
    Kokkos::deep_copy(m, u_[c]);
    for (Index i = 0; i < n_; ++i) h[static_cast<std::size_t>(i)] = m(i);
    return h;
  }
  /// L2 norm of the (openness-weighted) divergence of the current velocity.
  double divNormL2() {
    deviceDivergence(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                     View<const double>(u_[2]), div_);
    return std::sqrt(dotPlain(View<const double>(div_), View<const double>(div_), n_));
  }
  Index numLeaves() const { return n_; }
  /// Total momentum BiCGStab iterations (summed over the 3 components) of the last step.
  int lastMomIters() const { return lastMomIters_; }
  /// Pressure PCG iterations of the last step.
  int lastPresIters() const { return lastPresIters_; }

 private:
  // sdflow ccFractionCore aperture (verbatim from AmrFlow::faceFrac).
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
  Vec<3> f_{};
  bool presPCG_ = true;
  bool momMGon_ = false;  // opt-in (see setMomentumMG): Jacobi-BiCGStab is the robust default
  Index n_ = 0;
  int lastMomIters_ = 0, lastPresIters_ = 0;

  AmrCutCell<Bits> mom_;
  AmrPoisson<3, Bits> pres_;
  DeviceMultigrid<3, Bits> presMG_;
  DeviceMultigrid<3, Bits> momMG_;  // Helmholtz (idiag·I − μ∇²) momentum preconditioner
  DeviceMomentumOp momOp_;
  DeviceMomentumSolver<Bits> momSolver_;
  DevicePCG<3, Bits> pcg_;
  DeviceFaceGeom geom_;
  View<double> rscale_;
  View<char> fluid_;
  std::array<View<double>, 3> u_, gx_;
  View<double> p_, phi_, div_, bmom_;
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_DEVICE_FLOW_HPP

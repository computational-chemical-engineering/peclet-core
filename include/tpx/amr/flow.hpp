// transport-core — device (Kokkos) collocated incompressible Stokes step on a BlockOctree.
//
// The device counterpart of oracle::AmrFlow (flow.hpp): the whole cut-cell IBM projection step
// runs in Kokkos kernels instead of the host-serial gaussSeidel + host projection the drag
// study found to be the bottleneck. It reuses the host AmrCutCell / AmrPoisson to *build*
// the operators (geometry, openness, cut stencils — done once), then drives the time step
// entirely on the device:
//   * momentum predictor — MomentumOp (assembled cut-cell operator) solved with the
//     parallel BiCGStab of device_momentum.hpp;
//   * pressure projection — the openness Poisson on Multigrid / PCG;
//   * divergence, ABC gradient correction, rotational pressure update — face-CSR kernels
//     (FaceGeom) that mirror AmrPoisson::forEachFaceFull (same 2:1 sub-faces +
//     openness), so D / G / L stay consistent exactly as in the host collocated coupling.
//
// Stokes (advection off) and Navier–Stokes (setAdvection): implicit-FOU + explicit SOU/Koren
// deferred correction, advected by the divergence-free face field uf (built each projection;
// falls back to ½(u_i+u_j) until the first projection) — conservative (∇·uf = 0).
//
// Requires a Kokkos build + the morton checkout (TPX_HAVE_MORTON).
#ifndef TPX_AMR_FLOW_HPP
#define TPX_AMR_FLOW_HPP

#ifdef TPX_HAVE_MORTON

#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "tpx/amr/advect_recon.hpp"  // shared high-order face reconstruction (host+device)
#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/cut_cell.hpp"
#include "tpx/amr/face_geom.hpp"                   // FaceGeom (shared with the device assembler)
#include "tpx/amr/device_facegeom_assembly.hpp"   // deviceAssembleFaceGeom (D4/D6)
#include "tpx/amr/device_momentum_assembly.hpp"   // deviceAssembleMomentum (D3/D6)
#include "tpx/amr/momentum.hpp"
#include "tpx/amr/multigrid.hpp"
#include "tpx/amr/pcg.hpp"
#include "tpx/amr/velocity_mg.hpp"
#include "tpx/amr/poisson.hpp"
#include "tpx/common/types.hpp"
#include "tpx/common/view.hpp"

namespace tpx::amr {

// FaceGeom (the collocated projection's static face-geometry CSR) now lives in face_geom.hpp so the
// device assembler and this driver share the type without a circular include.

/// Build FaceGeom from a built AmrPoisson (openness set) + a fluid predicate.
template <int Dim, unsigned Bits, class FluidFn>
FaceGeom buildFaceGeom(const AmrPoisson<Dim, Bits>& ap, FluidFn&& isFluid) {
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
  std::vector<double> aArea(static_cast<std::size_t>(nf)), rArea(static_cast<std::size_t>(nf)),
      dist(static_cast<std::size_t>(nf)), alpha(static_cast<std::size_t>(nf));
  std::vector<Index> upupI(static_cast<std::size_t>(nf)), upupJ(static_cast<std::size_t>(nf));
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
      rArea[static_cast<std::size_t>(k)] = area;
      dist[static_cast<std::size_t>(k)] = d;
      alpha[static_cast<std::size_t>(k)] = al;
      // SOU upstream-of-upwind probes (point neighbour one cell further upstream): if i is the
      // upwind cell the upstream is across i's −dir face; if j is upwind, across j's +dir face.
      upupI[static_cast<std::size_t>(k)] = ap.periodicNeighbor(i, ax, -dr);
      upupJ[static_cast<std::size_t>(k)] = ap.periodicNeighbor(j, ax, dr);
      ++k;
    });
  }
  FaceGeom g;
  g.n = n;
  g.start = toDevice(start, "fg_start");
  g.nbr = toDevice(nbr, "fg_nbr");
  g.axis = toDevice(axis, "fg_axis");
  g.dir = toDevice(dir, "fg_dir");
  g.alphaArea = toDevice(aArea, "fg_aarea");
  g.rawArea = toDevice(rArea, "fg_rarea");
  g.dist = toDevice(dist, "fg_dist");
  g.alpha = toDevice(alpha, "fg_alpha");
  g.upupI = toDevice(upupI, "fg_upupi");
  g.upupJ = toDevice(upupJ, "fg_upupj");
  g.invVol = toDevice(invVol, "fg_invvol");
  g.fluid = toDevice(fluid, "fg_fluid");
  return g;
}

/// Openness-weighted FV divergence: div_i = invVol_i Σ_faces α·area·dir·½(u^axis_i+u^axis_j),
/// on fluid cells (0 elsewhere). u[0..2] are the three velocity component Views (solid cells
/// must hold 0). Mirrors oracle::AmrFlow::divergence.
inline void deviceDivergence(const FaceGeom& g, View<const double> u0, View<const double> u1,
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

/// Build the ABC/Basilisk divergence-free FACE field: uf(k) = ½(u^axis_i+u^axis_j) − (φ₊−φ₋)/dist for
/// face k of cell i (+axis velocity), from u* (call after the pressure solve, before the cell gradient
/// correction). Because L = D·G_face on the SAME (sub)faces, D(uf) = D u* − Lφ = 0 (to the φ residual).
/// Each (sub)face is written from both incident cells; the +axis-orientation build makes the two copies
/// identical, including across 2:1 interfaces (fine area + (φ_fine−φ_coarse)/dist on each sub-face).
inline void deviceBuildFaceField(const FaceGeom& g, View<const double> u0, View<const double> u1,
                                 View<const double> u2, View<const double> phi, View<double> uf) {
  auto st = g.start;
  auto nb = g.nbr;
  auto ax = g.axis;
  auto dr = g.dir;
  auto di = g.dist;
  Kokkos::parallel_for(
      "amr::flow_buildface", g.n, KOKKOS_LAMBDA(const Index i) {
        for (Index k = st(i); k < st(i + 1); ++k) {
          const int a = ax(k);
          const Index j = nb(k);
          const double ui = (a == 0) ? u0(i) : (a == 1) ? u1(i) : u2(i);
          const double uj = (a == 0) ? u0(j) : (a == 1) ? u1(j) : u2(j);
          const double gphi = (dr(k) > 0) ? (phi(j) - phi(i)) / di(k) : (phi(i) - phi(j)) / di(k);
          uf(k) = 0.5 * (ui + uj) - gphi;
        }
      });
}

/// L2 norm of the divergence of the FACE field uf (the div-free flux diagnostic / host-parity check).
inline double deviceDivFaceNorm(const FaceGeom& g, View<const double> uf) {
  auto st = g.start;
  auto dr = g.dir;
  auto aA = g.alphaArea;
  auto iv = g.invVol;
  auto fl = g.fluid;
  double s = 0.0;
  Kokkos::parallel_reduce(
      "amr::flow_divface", g.n,
      KOKKOS_LAMBDA(const Index i, double& acc) {
        if (!fl(i)) return;
        double d = 0.0;
        for (Index k = st(i); k < st(i + 1); ++k) d += aA(k) * dr(k) * uf(k);
        d *= iv(i);
        acc += d * d;
      },
      s);
  return std::sqrt(s);
}

/// ABC cell-gradient of a scalar field `f`: gx/gy/gz = ½(g⁻+g⁺) of the adjacent face
/// gradients along each axis, a closed face (α≤1e-12) contributing nothing (and not
/// counted). On fluid cells only. Mirrors oracle::AmrFlow::gradOf for all three components.
inline void deviceGrad3(const FaceGeom& g, View<const double> f, View<double> gx,
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

/// Momentum RHS for one component: b_i = fluid ? (idiag·u_i + f_c − gradP_i − adv_i)·rscale_i : 0
/// (== AmrCutCell::makeRhs of the oracle::AmrFlow predictor source, u_bc = 0). `adv` is the explicit
/// deferred-correction advection term ρ(SOU−FOU) (zero for Stokes / fully-implicit at steady).
inline void deviceMomRhs(View<const double> uc, View<const double> gradP, View<const double> adv,
                         View<const double> rscale, View<const char> fluid, double idiag, double fc,
                         View<double> b, Index n) {
  Kokkos::parallel_for(
      "amr::flow_momrhs", n, KOKKOS_LAMBDA(const Index i) {
        b(i) = fluid(i) ? (idiag * uc(i) + fc - gradP(i) - adv(i)) * rscale(i) : 0.0;
      });
}

/// Build the implicit-FOU advection operator from the lagged velocity `u0..2` (uⁿ) entirely on
/// device, as the per-cell outflow diagonal `advDiag` + per-face inflow coefficient `advCoef`
/// over the face-geometry CSR (`g.start/g.nbr`). velOut = dir·½(u^axis_i+u^axis_j); outflow →
/// diagonal, inflow → off-diagonal toward the upstream neighbour; faces into solid carry none.
/// Scaled by the cut-cell row scale `rscale` (= 1 on regular cells) so the operator advection is
/// consistent with the rscale-scaled RHS — equivalent to AmrCutCell::assembleOperator(scaleAdv=
/// true) added to the static Stokes operator (the AMR reference adds advection after the cut-cell
/// bake, so no K/M/X redistribution). Replaces the per-step HOST rebuild ⇒ no host round-trip.
inline void deviceBuildFou(const FaceGeom& g, View<const double> u0, View<const double> u1,
                           View<const double> u2, double rho, View<const double> rscale,
                           View<double> advDiag, View<double> advCoef, View<const double> uf,
                           bool useFace) {
  auto st = g.start;
  auto nb = g.nbr;
  auto ax = g.axis;
  auto dr = g.dir;
  auto ra = g.rawArea;
  auto iv = g.invVol;
  auto fl = g.fluid;
  Kokkos::parallel_for(
      "amr::flow_buildfou", g.n, KOKKOS_LAMBDA(const Index i) {
        if (!fl(i)) {
          advDiag(i) = 0.0;
          for (Index k = st(i); k < st(i + 1); ++k) advCoef(k) = 0.0;
          return;
        }
        const double rs = rscale(i);
        double dsum = 0.0;
        for (Index k = st(i); k < st(i + 1); ++k) {
          const Index j = nb(k);
          if (!fl(j)) {
            advCoef(k) = 0.0;
            continue;
          }
          const int a = ax(k);
          const double ui = (a == 0) ? u0(i) : (a == 1) ? u1(i) : u2(i);
          const double uj = (a == 0) ? u0(j) : (a == 1) ? u1(j) : u2(j);
          const double velOut = useFace ? dr(k) * uf(k) : dr(k) * 0.5 * (ui + uj);
          const double w = rs * rho * ra(k) * velOut * iv(i);
          if (velOut < 0.0) {
            advCoef(k) = w;  // inflow → off-diagonal toward upstream neighbour j
          } else {
            advCoef(k) = 0.0;
            dsum += w;  // outflow → diagonal
          }
        }
        advDiag(i) = dsum;
      });
}

/// Deferred-correction advection term for component `comp`: defc = ρ·SOU − ρ·FOU (UNSCALED; the
/// predictor RHS applies the cut-cell rscale once). The explicit part of the implicit-FOU/SOU
/// split, it vanishes at steady state. The advecting velocity is u0..2 (uⁿ) and — for a lagged
/// step — the advected field is the same component. SOU is the second-order-upwind
/// reconstruction 1.5·up−0.5·upup (advScheme 0) or Koren TVD (1), upstream point-probed
/// (upupI/upupJ); the FOU flux is velOut·upwind. The implicit FOU is baked into the momentum
/// operator (AmrCutCell::buildAdvectionFou + assembleOperator) so the two cancel at steady state.
inline void deviceDeferredSou(const FaceGeom& g, View<const double> u0, View<const double> u1,
                              View<const double> u2, int comp, double rho, int advScheme,
                              View<double> defc, View<const double> uf, bool useFace) {
  auto st = g.start;
  auto nb = g.nbr;
  auto ax = g.axis;
  auto dr = g.dir;
  auto ra = g.rawArea;
  auto iv = g.invVol;
  auto fl = g.fluid;
  auto uiP = g.upupI;
  auto ujP = g.upupJ;
  Kokkos::parallel_for(
      "amr::flow_defsou", g.n, KOKKOS_LAMBDA(const Index i) {
        if (!fl(i)) {
          defc(i) = 0.0;
          return;
        }
        auto fld = [&](Index c) { return (comp == 0) ? u0(c) : (comp == 1) ? u1(c) : u2(c); };
        double sou = 0.0, fou = 0.0;
        for (Index k = st(i); k < st(i + 1); ++k) {
          const Index j = nb(k);
          if (!fl(j)) continue;
          const int a = ax(k);
          const double uai = (a == 0) ? u0(i) : (a == 1) ? u1(i) : u2(i);
          const double uaj = (a == 0) ? u0(j) : (a == 1) ? u1(j) : u2(j);
          const double velOut = useFace ? dr(k) * uf(k) : dr(k) * 0.5 * (uai + uaj);
          const Index up = (velOut > 0.0) ? i : j;
          const Index down = (velOut > 0.0) ? j : i;
          const Index upup = (velOut > 0.0) ? uiP(k) : ujP(k);
          const double phiUp = fld(up);
          const double phiUpUp = (upup >= 0 && fl(upup)) ? fld(upup) : phiUp;
          const double phiDown = fld(down);
          const double phiFace = hoFaceValue(phiUpUp, phiUp, phiDown, advScheme);  // shared recon
          sou += ra(k) * velOut * phiFace;
          fou += ra(k) * velOut * fld(up);  // FOU flux = velOut · upwind value
        }
        defc(i) = rho * iv(i) * (sou - fou);  // ρ·(SOU − FOU), unscaled
      });
}

/// Fully-explicit high-order advection for component `comp`: defc = ρ·SOU (no implicit FOU; the
/// `setImplicitAdvection(false)` fallback). Same SOU/TVD reconstruction as deviceDeferredSou.
inline void deviceAdvectExplicit(const FaceGeom& g, View<const double> u0,
                                 View<const double> u1, View<const double> u2, int comp, double rho,
                                 int advScheme, View<double> defc, View<const double> uf, bool useFace) {
  auto st = g.start;
  auto nb = g.nbr;
  auto ax = g.axis;
  auto dr = g.dir;
  auto ra = g.rawArea;
  auto iv = g.invVol;
  auto fl = g.fluid;
  auto uiP = g.upupI;
  auto ujP = g.upupJ;
  Kokkos::parallel_for(
      "amr::flow_advexpl", g.n, KOKKOS_LAMBDA(const Index i) {
        if (!fl(i)) {
          defc(i) = 0.0;
          return;
        }
        auto fld = [&](Index c) { return (comp == 0) ? u0(c) : (comp == 1) ? u1(c) : u2(c); };
        double sou = 0.0;
        for (Index k = st(i); k < st(i + 1); ++k) {
          const Index j = nb(k);
          if (!fl(j)) continue;
          const int a = ax(k);
          const double uai = (a == 0) ? u0(i) : (a == 1) ? u1(i) : u2(i);
          const double uaj = (a == 0) ? u0(j) : (a == 1) ? u1(j) : u2(j);
          const double velOut = useFace ? dr(k) * uf(k) : dr(k) * 0.5 * (uai + uaj);
          const Index up = (velOut > 0.0) ? i : j;
          const Index down = (velOut > 0.0) ? j : i;
          const Index upup = (velOut > 0.0) ? uiP(k) : ujP(k);
          const double phiUp = fld(up);
          const double phiUpUp = (upup >= 0 && fl(upup)) ? fld(upup) : phiUp;
          const double phiDown = fld(down);
          const double phiFace = hoFaceValue(phiUpUp, phiUp, phiDown, advScheme);  // shared recon
          sou += ra(k) * velOut * phiFace;
        }
        defc(i) = rho * sou * iv(i);  // ρ·SOU (fully explicit)
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
// AmrFlow — collocated Stokes projection step, fully on device.
// ===========================================================================
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
  /// Use MG-preconditioned CG for the pressure solve (default) vs plain V-cycles.
  void setPressurePCG(bool on) { presPCG_ = on; }
  /// Enable momentum advection ∇·(u u) (default OFF ⇒ Stokes). The high-order flux is
  /// second-order upwind (SOU) by default; the first-order-upwind part is solved *implicitly*
  /// (folded into the momentum operator) and the (SOU−FOU) difference is the explicit deferred
  /// correction — unconditionally stable for the FOU part, exact-SOU at steady state. This is
  /// the same implicit-FOU + deferred-SOU scheme as the host oracle::AmrFlow and sdflow's collocated
  /// grid (`set_implicit_advection`).
  void setAdvection(bool on) { advect_ = on; }
  /// Implicit-FOU deferred correction (default ON). OFF ⇒ the whole high-order advection is
  /// explicit (no FOU in the operator) — only conditionally stable.
  void setImplicitAdvection(bool on) { implicitFou_ = on; }
  /// High-order advection scheme: 0 = second-order upwind (SOU, default), 1 = Koren TVD.
  void setAdvectionScheme(int s) { advScheme_ = s; }
  /// Relative tolerance for the per-step momentum BiCGStab solve (default 1e-8). The
  /// momentum predictor is one step of a pseudo-transient outer iteration to steady state,
  /// so it need not be solved to round-off — a looser tolerance bounds the per-step cost.
  /// NOTE on the cost regime: the momentum operator is the Helmholtz (ρ/dt)I − μ∇²; for a
  /// physical dt the (ρ/dt) mass term dominates and it is cheap (diagonally dominant). At the
  /// large dt used for *steady* drag, ρ/dt → 0 and it degrades to a bare elliptic Laplacian
  /// (a saddle-point Stokes problem) — as hard as the pressure Poisson and, unlike the
  /// pressure, solved here per velocity component. Bounding the tolerance (this knob) caps the
  /// over-solve; making it actually *scale* needs the velocity multigrid (setMomentumMG).
  void setMomentumTol(double tol) { momTol_ = tol; }
  /// Use the Galerkin velocity multigrid (MomentumMG) as the momentum BiCGStab
  /// preconditioner. This is the scalable momentum solver: the coarse operators are the exact
  /// assembled cut-cell operator coarsened by R·A·P, so the V-cycle is a consistent
  /// preconditioner and the momentum iteration count stays ~flat with N instead of growing
  /// like the Jacobi-preconditioned BiCGStab (the dominant cost at scale and large dt). It
  /// only changes the preconditioner (the matvec is the exact operator) ⇒ identical converged
  /// step. Call before setSolid (the hierarchy is built there). Default ON.
  void setMomentumMG(bool on) { momMGon_ = on; }

  /// Choose the momentum-MG coarse-operator strategy: false (default) = Galerkin
  /// (MomentumMG, A_c = R·A·P of the exact cut-cell operator); true = rediscretized
  /// staircase (VelocityMG, mirroring sdflow's VelocityMG). Call before setSolid. Both are
  /// device-resident BiCGStab preconditioners; this lets the two be benchmarked head-to-head.
  void setVelocityMGStaircase(bool on) { useStaircaseMG_ = on; }
  /// Pore-scale cap for the staircase velocity-MG: the coarsest level keeps ≥ this many cells, so
  /// it still resolves the immersed feature (a small object that vanishes from the coarse κ
  /// classification leaves an inconsistent operator that diverges). Feature-dependent — raise it
  /// for small immersed objects. Only affects the staircase strategy.
  void setVelocityMGMinCoarse(Index m) { mgMinCoarse_ = m; }
  /// Opt-in: use the multicolour Gauss–Seidel smoother in the momentum MG (Galerkin or staircase)
  /// instead of weighted Jacobi — ~2× better smoothing (fewer V-cycles / BiCGStab iters), and the
  /// strong fine smoother the staircase needs on the cut band. Default off. Call before setSolid.
  void setMomentumGS(bool on) { momGS_ = on; }

  /// Opt-in (P4): solve the momentum predictor with the velocity multigrid used **as the solver**
  /// — MG-preconditioned defect correction `u ← u + M⁻¹(b − A u)` iterated to tolerance — instead of
  /// BiCGStab with the same MG as a preconditioner. This is the faithful mirror of sdflow's velocity
  /// solve, which uses no Krylov method at all (RB-GS smoother / velocity-MG directly): the momentum
  /// operator is diagonally dominant and invertible, so the stationary iteration converges, and
  /// unlike BiCGStab it cannot break down on the strongly non-symmetric (D_rescale + FOU) operator
  /// (no bi-orthogonal recurrence to lose — the failure mode P5 had to symmetrise the smoother to
  /// avoid). With the GS smoother (setMomentumGS) and a velocity-MG (setMomentumMG, default on) this
  /// is the full RB-GS/MG sdflow path; with MG off it degrades to plain damped-Jacobi sweeps
  /// (sdflow's smoothComp). Default off (BiCGStab stays the validated default). Requires setMomentumMG
  /// to actually be multigrid-accelerated; otherwise the Richardson iteration converges only slowly.
  void setMomentumMGSolver(bool on) { momMGSolver_ = on; }

  /// Optional Picard outer loop over the lagged advection (mirror of sdflow's outerIters_): each
  /// outer iteration re-freezes the advecting velocity at the latest u and re-solves the predictor,
  /// stopping early when the max |Δu| between successive outer iterates falls below `tol`. `n = 1`
  /// (default) is the single lagged step used until now (no extra cost). Only meaningful with
  /// advection on; for Stokes the loop converges in one iteration. n ≥ 1.
  void setOuterIterations(int n, double tol = 1e-6) {
    outerIters_ = (n < 1) ? 1 : n;
    outerTol_ = tol;
  }

  /// Build the cut-cell operators (host) + upload all device structures. Requires the
  /// density / viscosity / dt to be set first (the momentum operator carries ρ/dt and μ).
  template <class SdfFn>
  void setSolid(SdfFn&& sdfFn) {
    const Index n = t_->numLeaves();
    // Host operator build (geometry, openness, cut stencils) — same as oracle::AmrFlow::setSolid.
    mom_.init(*t_, h0_, origin_);
    mom_.build(sdfFn, /*idiag=*/rho_ / dt_, /*beta=*/mu_ / (h0_ * h0_));
    pres_.init(*t_, h0_);
    pres_.setOrigin(origin_);
    auto openFn = [&](const Vec<3>& fc, int axis) { return faceFrac(sdfFn, fc, axis); };
    pres_.buildOpenness(openFn);
    presMG_.build(*t_, h0_, openFn, /*periodic=*/true);
    presMG_.setRemoveMean(true);  // singular periodic pressure: per-level nullspace projection

    // Device assembly (D6): the static cut-cell momentum operator CSR and the collocated face
    // geometry are assembled ON THE DEVICE (deviceAssembleMomentum / deviceAssembleFaceGeom), and the
    // pressure MG operators are device-assembled per level (Multigrid D5) — so no host CSR walk and no
    // operator round-trip. Each is bit-for-bit identical to the host assembler on OpenMP (locked in
    // test_amr_device_momentum / test_amr_device_facegeom), so the flow result is unchanged. A shared
    // device octree view backs both assemblers.
    BlockOctreeView<3, Bits> ov;
    ov.upload(*t_);
    momOp_ = deviceAssembleMomentum<Bits>(mom_, ov);  // static Stokes operator (hasAdv stays false)
    // Velocity multigrid (momentum preconditioner): the Galerkin hierarchy A_c = R·A·P built
    // directly from the exact assembled momentum CSR. Consistent with the fine cut-cell
    // operator by construction (inherits the ξ-overlay + D_rescale row scaling; a coarse cell
    // of all-solid children stays an identity row). It only changes the preconditioner (the
    // BiCGStab matvec is the exact operator) ⇒ same converged solution, but the iteration
    // count stays ~flat with N instead of growing like the Jacobi-preconditioned BiCGStab.
    // Build the chosen momentum-MG: Galerkin (MomentumMG) by default, or the rediscretized
    // staircase (VelocityMG) — both from the static Stokes operator, once.
    if (momMGon_) {
      if (useStaircaseMG_) {
        std::vector<double> kap(static_cast<std::size_t>(n));
        std::vector<char> fl(static_cast<std::size_t>(n)), cu(static_cast<std::size_t>(n));
        for (Index i = 0; i < n; ++i) {
          kap[static_cast<std::size_t>(i)] = mom_.kappa(i);
          fl[static_cast<std::size_t>(i)] = mom_.isFluid(i) ? 1 : 0;
          cu[static_cast<std::size_t>(i)] = mom_.isCut(i) ? 1 : 0;
        }
        velMG_.build(*t_, h0_, rho_ / dt_, mu_, momOp_, kap, fl, cu, mgMinCoarse_);
        velMG_.setGaussSeidel(momGS_);
      } else {
        // The Galerkin RAP hierarchy is a host triple-product over the fine CSR, so it needs the
        // operator on the host; assemble it there for the MG build only (bit-identical to momOp_).
        auto A = mom_.assembleOperator();
        momMG_.build(*t_, A.diag, A.start, A.nbr, A.coef);
        momMG_.setGaussSeidel(momGS_);
      }
    }
    std::vector<char> fluidVec(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) fluidVec[static_cast<std::size_t>(i)] = mom_.isFluid(i) ? 1 : 0;
    geom_ = deviceAssembleFaceGeom<Bits>(pres_, fluidVec, ov);
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
    uf_ = View<double>("df_uf", geom_.nbr.extent(0));  // ABC divergence-free face field (per CSR face)
    faceFieldBuilt_ = false;
    bmom_ = View<double>("df_bmom", static_cast<std::size_t>(n));
    Kokkos::deep_copy(p_, 0.0);
    // Implicit-FOU advection state. The momentum operator + its velocity-MG are rebuilt each
    // step from the *full* operator (viscous + FOU) so the MG is advection-aware (the viscous-
    // only MG diverges on the advection operator at cut cells); the FOU is baked into the CSR
    // (momOp_.hasAdv stays false). defc holds the device-computed explicit ρ(SOU−FOU)
    // deferred correction. uadvHost_ caches u^n on the host for the per-step operator rebuild.
    for (int c = 0; c < 3; ++c) {
      defc_[c] = View<double>("df_defc", static_cast<std::size_t>(n));
      Kokkos::deep_copy(defc_[c], 0.0);
      // uⁿ snapshot for the backward-Euler mass term (frozen across Picard outer iters) + the
      // previous outer iterate for the outer-loop convergence test.
      u0_[c] = View<double>("df_u0", static_cast<std::size_t>(n));
      uprev_[c] = View<double>("df_uprev", static_cast<std::size_t>(n));
    }
    // Device-resident implicit-FOU advection: the FOU operator (advDiag + per-face advCoef over the
    // face-geometry CSR) is rebuilt on device each step from uⁿ and added to the static Stokes
    // operator in the matvec (no host round-trip). The static operator + Galerkin velocity-MG are
    // built once (above); the advection is a perturbation the static MG still preconditions.
    advDiag_ = View<double>("df_advdiag", static_cast<std::size_t>(n));
    advCoef_ = View<double>("df_advcoef", geom_.nbr.extent(0));
    momOp_.advStart = geom_.start;
    momOp_.advNbr = geom_.nbr;
    momOp_.advDiag = advDiag_;
    momOp_.advCoef = advCoef_;
    momOp_.hasAdv = false;  // set per-step when advection is on
    momSolver_.setJacobi(2, 0.7);
    // Generic MG preconditioner: dispatch the chosen hierarchy's V-cycle (z = M⁻¹ r), decoupling
    // the solver from the MG type so Galerkin and staircase are interchangeable.
    if (momMGon_) {
      if (useStaircaseMG_)
        momSolver_.setPreconditioner(
            [this](View<const double> r, View<double> z) { runMgVcycle(velMG_, r, z); });
      else
        momSolver_.setPreconditioner(
            [this](View<const double> r, View<double> z) { runMgVcycle(momMG_, r, z); });
    }
    pcg_.setVcycle(2, 2, 60, 0.8);
    pcg_.setSingular(true);
    n_ = n;
  }

  /// One incompressible step on device (Stokes, or Navier–Stokes with setAdvection). `momIters`
  /// BiCGStab iterations for each momentum component; `presIters` PCG iterations for the
  /// pressure solve.
  void step(int momIters = 100, int presIters = 60) {
    const Index n = n_;
    const double idiag = rho_ / dt_;
    lastMomIters_ = 0;
    lastOuterIters_ = 1;
    // Freeze the time-level uⁿ for the backward-Euler mass term + warm start; it stays anchored
    // across the Picard outer iterations (only the advecting velocity re-lags). For outerIters_==1
    // this is just a copy of uⁿ ⇒ bit-identical to the single lagged step.
    for (int c = 0; c < 3; ++c) Kokkos::deep_copy(u0_[c], View<const double>(u_[c]));
    // −∇p^n is constant across the outer iterations (pressure is projected once, after the loop, like
    // sdflow's single per-step projection) ⇒ hoist it out.
    deviceGrad3(geom_, View<const double>(p_), gx_[0], gx_[1], gx_[2]);
    // Picard outer loop over the lagged advection only (the momentum nonlinearity); for outerIters_==1
    // this is the single lagged predictor, then one projection — bit-identical to before.
    for (int outer = 0; outer < outerIters_; ++outer) {
      // --- advection lagged to the *current* predictor iterate (uⁿ on the first pass): implicit-FOU
      // operator + explicit ρ(SOU−FOU) deferred correction. The matvec stays linear during each solve
      // (the advecting velocity is frozen in advDiag_/advCoef_). With advection OFF the operator and
      // RHS are identical every pass, so a second pass reproduces the first ⇒ instant early-stop. ---
      if (advect_) {
        momOp_.hasAdv = implicitFou_;
        // Advect with the divergence-free face field uf (from the previous projection); fall back to
        // ½(u_i+u_j) before the first projection has built it. The implicit FOU and the explicit SOU/FOU
        // deferred correction use the SAME velocity ⇒ the FOU cancels at steady state (host-parity).
        const View<const double> ufv(uf_);
        if (implicitFou_)
          deviceBuildFou(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                         View<const double>(u_[2]), rho_, View<const double>(rscale_), advDiag_,
                         advCoef_, ufv, faceFieldBuilt_);
        for (int c = 0; c < 3; ++c) {
          if (implicitFou_)
            deviceDeferredSou(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                              View<const double>(u_[2]), c, rho_, advScheme_, defc_[c], ufv, faceFieldBuilt_);
          else
            deviceAdvectExplicit(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                                 View<const double>(u_[2]), c, rho_, advScheme_, defc_[c], ufv, faceFieldBuilt_);
        }
        // The staircase MG's fine level mirrors the sharp operator; refresh it so it picks up the
        // current advection state (hasAdv). (The Galerkin MG is the static viscous operator.)
        if (momMGon_ && useStaircaseMG_) velMG_.setFineOp(momOp_);
      }
      // --- predictor: incremental BE viscous (+ implicit-FOU) solve per component, RHS carries −∇p^n
      // and −ρ(SOU−FOU); the mass term is anchored at uⁿ (u0_), the solve warm-starts at the current
      // iterate. ---
      for (int c = 0; c < 3; ++c) {
        deviceMomRhs(View<const double>(u0_[c]), View<const double>(gx_[c]),
                     View<const double>(defc_[c]), View<const double>(rscale_),
                     View<const char>(fluid_), idiag, f_[c], bmom_, n);
        // P4 (opt-in): the velocity-MG used as the *solver* — MG-preconditioned defect correction,
        // no Krylov (the sdflow RB-GS/velocity-MG mirror; cannot break down on the non-symmetric
        // operator) — vs the default MG-preconditioned BiCGStab. Both reach the same solution (the
        // matvec is the exact operator); the choice only trades robustness for convergence rate.
        lastMomIters_ +=
            (momMGSolver_ ? momSolver_.solveDefectCorrection(momOp_, u_[c],
                                                             View<const double>(bmom_), momIters, momTol_)
                          : momSolver_.solveBiCGStab(momOp_, u_[c], View<const double>(bmom_),
                                                     momIters, momTol_))
                .iters;
      }
      // Outer-loop convergence on the predictor velocity (skipped for the default outerIters_==1, so
      // that path is untouched). With advection off the second iterate equals the first ⇒ stops at 2.
      if (outerIters_ > 1) {
        lastOuterIters_ = outer + 1;
        if (outer > 0) {
          double dmax = 0.0;
          for (int c = 0; c < 3; ++c)
            dmax = std::max(dmax, maxAbsDiff(View<const double>(u_[c]), View<const double>(uprev_[c]), n));
          if (dmax < outerTol_) break;
        }
        for (int c = 0; c < 3; ++c) Kokkos::deep_copy(uprev_[c], View<const double>(u_[c]));
      }
    }
    project(presIters);  // single pressure projection per step (sdflow structure)
  }

  /// Pressure projection of the current velocity in place.
  void project(int presIters = 60) {
    const Index n = n_;
    deviceDivergence(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                     View<const double>(u_[2]), div_);
    Kokkos::deep_copy(phi_, 0.0);
    // Two selectable pressure drivers, like sdflow's CutcellMG (MG-PCG single-rank default,
    // standalone V-cycle the robust multi-rank default): MG-PCG for the near-steady, divergence-
    // compatible Stokes case (faster), and the stationary V-cycle for the large transient
    // divergence of advection, which excites near-nullspace modes of the cut-cell openness Poisson
    // (near-isolated fluid cells) that CG amplifies — the V-cycle is bounded. (Per-level mean
    // removal is on for both; making MG-PCG robust enough to also cover advection needs sdflow's
    // near-isolated-cell pinning/classification — a follow-up.)
    if (presPCG_ && !advect_) {
      lastPresIters_ = pcg_.solve(presMG_, phi_, View<const double>(div_), presIters, 1e-10).iters;
    } else {
      Kokkos::deep_copy(presMG_.b(0), div_);
      Kokkos::deep_copy(presMG_.x(0), 0.0);
      for (int it = 0; it < presIters; ++it) presMG_.vcycle(2, 2, 60, 0.8);
      Kokkos::deep_copy(phi_, presMG_.x(0));
      lastPresIters_ = presIters;
    }
    // Build the divergence-free FACE field from u* (still in u_) + φ, before the cell correction.
    deviceBuildFaceField(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                         View<const double>(u_[2]), View<const double>(phi_), uf_);
    faceFieldBuilt_ = true;
    deviceGrad3(geom_, View<const double>(phi_), gx_[0], gx_[1], gx_[2]);
    for (int c = 0; c < 3; ++c) deviceCorrect(u_[c], View<const double>(gx_[c]), View<const char>(fluid_), n);
    devicePresUpdate(p_, View<const double>(phi_), View<const double>(div_),
                     View<const char>(fluid_), rho_ / dt_, mu_, n);
  }

  /// DEBUG: the raw high-order advection ∇·(u u_comp) per cell from the current velocity
  /// (== host oracle::AmrFlow::advectTerm). Isolates the SOU kernel from the solve.
  std::vector<double> debugSou(int comp) {
    View<double> s("dbg_sou", static_cast<std::size_t>(n_));
    deviceAdvectExplicit(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                         View<const double>(u_[2]), comp, 1.0, advScheme_, s, View<const double>(uf_), false);
    std::vector<double> h(static_cast<std::size_t>(n_));
    auto m = Kokkos::create_mirror_view(s);
    Kokkos::deep_copy(m, s);
    for (Index i = 0; i < n_; ++i) h[static_cast<std::size_t>(i)] = m(i);
    return h;
  }
  /// Run one V-cycle of a momentum MG as a preconditioner: z = M⁻¹ r. Templated on the MG type
  /// (MomentumMG or VelocityMG — both expose b(0)/x(0)/vcycle), so the BiCGStab
  /// preconditioner is decoupled from the coarse-operator strategy.
  template <class MG>
  void runMgVcycle(MG& mg, View<const double> r, View<double> z) {
    Kokkos::deep_copy(mg.b(0), r);
    Kokkos::deep_copy(mg.x(0), 0.0);
    mg.vcycle(mgVcPre_, mgVcPre_, mgVcBottom_, 0.7);
    Kokkos::deep_copy(z, mg.x(0));
  }
  /// Max |a − b| over all cells (the Picard outer-loop convergence measure).
  static double maxAbsDiff(View<const double> a, View<const double> b, Index n) {
    double m = 0.0;
    Kokkos::parallel_reduce(
        "amr::flow_maxdiff", n,
        KOKKOS_LAMBDA(const Index i, double& lm) {
          double d = a(i) - b(i);
          if (d < 0.0) d = -d;
          if (d > lm) lm = d;
        },
        Kokkos::Max<double>(m));
    return m;
  }
  /// Copy a device View into a host vector (sized n_).
  void copyToHost(const View<double>& d, std::vector<double>& h) const {
    auto m = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(m, d);
    for (Index i = 0; i < n_; ++i) h[static_cast<std::size_t>(i)] = m(i);
  }
  /// Set a velocity component from host (testing / initial conditions).
  void setVelocity(int c, const std::vector<double>& h) {
    auto m = Kokkos::create_mirror_view(u_[c]);
    for (Index i = 0; i < n_; ++i) m(i) = h[static_cast<std::size_t>(i)];
    Kokkos::deep_copy(u_[c], m);
  }
  /// Copy a velocity component back to host (single D2H, no host loop — S2a).
  std::vector<double> velocity(int c) const { return tpx::toVector(u_[c]); }

  /// All three velocity components interleaved as a flat (n,3) row-major host buffer
  /// (out[i*3+c]) with a single device→host transfer (G6): the three independent component
  /// Views are packed on-device first (cheap), so the boundary is crossed once rather than
  /// three times as repeated velocity(c) calls would.
  std::vector<double> velocities() const {
    View<double> packed(Kokkos::view_alloc("amr::vel_packed", Kokkos::WithoutInitializing),
                        static_cast<std::size_t>(n_) * 3);
    for (int c = 0; c < 3; ++c) {
      auto uc = u_[c];
      auto p = packed;
      const int cc = c;
      Kokkos::parallel_for(
          "amr::pack_vel", n_, KOKKOS_LAMBDA(const Index i) { p(i * 3 + cc) = uc(i); });
    }
    return tpx::toVector(packed);
  }
  /// L2 norm of the (openness-weighted) divergence of the current velocity.
  double divNormL2() {
    deviceDivergence(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                     View<const double>(u_[2]), div_);
    return std::sqrt(dotPlain(View<const double>(div_), View<const double>(div_), n_));
  }
  /// L2 norm of the divergence of the ABC face field uf_ (built each project()): the φ-solve residual,
  /// far below the cell field's O(h²) divNormL2 — including across 2:1 interfaces.
  double divNormFace() { return deviceDivFaceNorm(geom_, View<const double>(uf_)); }
  /// Copy the divergence-free face field to host (one value per CSR (sub)face, forEachFaceFull order).
  std::vector<double> faceField() const { return tpx::toVector(uf_); }
  Index numLeaves() const { return n_; }
  /// Per-leaf fluid mask (false inside the solid) — for host-side post-processing / bindings.
  bool isFluid(Index i) const { return mom_.isFluid(i); }
  /// Total momentum BiCGStab iterations (summed over the 3 components) of the last step.
  int lastMomIters() const { return lastMomIters_; }
  /// Pressure PCG iterations of the last step.
  int lastPresIters() const { return lastPresIters_; }
  /// Picard outer iterations actually run in the last step (1 unless setOuterIterations(>1)).
  int lastOuterIters() const { return lastOuterIters_; }

 private:
  // sdflow ccFractionCore aperture (verbatim from oracle::AmrFlow::faceFrac).
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
  bool momMGon_ = true;        // velocity-MG momentum preconditioner (scalable; see setMomentumMG)
  bool useStaircaseMG_ = false;  // false = Galerkin (MomentumMG), true = staircase (VelocityMG)
  int mgVcPre_ = 2, mgVcBottom_ = 30;  // momentum-MG V-cycle pre/post sweeps + bottom sweeps
  Index mgMinCoarse_ = 256;            // staircase velocity-MG pore-scale cap (coarsest cell count)
  bool momGS_ = false;                 // opt-in: multicolour Gauss–Seidel smoother in the momentum MG
  bool momMGSolver_ = false;           // opt-in (P4): velocity-MG as the solver (defect correction), not BiCGStab
  int outerIters_ = 1;                 // Picard outer iterations over the lagged advection (default 1)
  double outerTol_ = 1e-6;             // outer-loop early-stop tolerance on max|Δu|
  double momTol_ = 1e-8;  // per-step momentum BiCGStab relative tolerance (Phase-0 knob)
  bool advect_ = false;       // momentum advection ∇·(u u) (off ⇒ Stokes)
  bool implicitFou_ = true;   // implicit-FOU deferred-correction (stable) vs fully-explicit
  int advScheme_ = 0;         // high-order flux: 0 = SOU (default), 1 = Koren TVD
  Index n_ = 0;
  int lastMomIters_ = 0, lastPresIters_ = 0, lastOuterIters_ = 1;

  AmrCutCell<Bits> mom_;
  AmrPoisson<3, Bits> pres_;
  Multigrid<3, Bits> presMG_;
  MomentumMG<Bits> momMG_;  // Galerkin velocity multigrid (momentum preconditioner)
  VelocityMG<Bits> velMG_;  // rediscretized staircase velocity multigrid (alternative)
  MomentumOp momOp_;
  MomentumSolver<Bits> momSolver_;
  PCG<3, Bits> pcg_;
  std::array<View<double>, 3> defc_;  // explicit ρ(SOU−FOU) deferred correction per component
  View<double> advDiag_, advCoef_;    // device-resident implicit-FOU operator (rebuilt each step)
  FaceGeom geom_;
  View<double> rscale_;
  View<char> fluid_;
  std::array<View<double>, 3> u_, gx_;
  std::array<View<double>, 3> u0_, uprev_;  // frozen uⁿ (BE mass term) + previous Picard outer iterate
  View<double> p_, phi_, div_, bmom_;
  View<double> uf_;  // ABC/Basilisk divergence-free face field (one per CSR (sub)face)
  bool faceFieldBuilt_ = false;  // uf_ populated by a projection (else advection falls back to ½(u_i+u_j))
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_FLOW_HPP

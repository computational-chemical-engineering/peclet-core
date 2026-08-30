// core — device (Kokkos) collocated incompressible Stokes step on a BlockOctree.
//
// The device counterpart of oracle::AmrFlow (flow.hpp): the whole cut-cell IBM projection step
// runs in Kokkos kernels instead of the host-serial gaussSeidel + host projection the drag
// study found to be the bottleneck. It reuses the host AmrCutCell / AmrPoisson to *build*
// the operators (geometry, openness, cut stencils — done once), then drives the time step
// entirely on the device:
//   * momentum predictor — MomentumOp (assembled cut-cell operator) solved with the
//     parallel BiCGStab of momentum.hpp;
//   * pressure projection — the openness Poisson on Multigrid / PCG;
//   * divergence, ABC gradient correction, rotational pressure update — face-CSR kernels
//     (FaceGeom) that mirror AmrPoisson::forEachFaceFull (same 2:1 sub-faces +
//     openness), so D / G / L stay consistent exactly as in the host collocated coupling.
//
// Stokes (advection off) and Navier–Stokes (setAdvection): implicit-FOU + explicit SOU/Koren
// deferred correction, advected by the divergence-free face field uf (built each projection;
// falls back to ½(u_i+u_j) until the first projection) — conservative (∇·uf = 0).
//
// Requires a Kokkos build + the morton checkout (PECLET_CORE_HAVE_MORTON).
#ifndef PECLET_CORE_AMR_FLOW_HPP
#define PECLET_CORE_AMR_FLOW_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "peclet/core/amr/adapt.hpp"  // transferField (conservative remap for finishAdapt)

#include "peclet/core/amr/advect_recon.hpp"  // shared high-order face reconstruction (host+device)
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/cut_cell.hpp"
#include "peclet/core/amr/face_geom.hpp"          // FaceGeom (shared with the device assembler)
#include "peclet/core/amr/cf_scheme.hpp"          // pluggable 2:1 C/F schemes (setCfScheme)
#include "peclet/core/amr/facegeom_assembly.hpp"  // assembleFaceGeom (D4/D6)
#include "peclet/core/amr/ghost_projection.hpp"   // directional ghost overlay (setGhostProjection)
#include "peclet/core/amr/ghost_projection_sampled.hpp"  // mixed-level sampled overlay (setGhostSampled)
#include "peclet/core/amr/momentum.hpp"
#include "peclet/core/amr/momentum_assembly.hpp"  // assembleMomentum (D3/D6)
#include "peclet/core/amr/multigrid.hpp"
#include "peclet/core/amr/pcg.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/amr/velocity_mg.hpp"
#include "peclet/core/common/host_parallel.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"

#include "peclet/core/amr/distributed_adapt.hpp"    // transferGradients (distributed finishAdapt)
#include "peclet/core/amr/distributed_flow_mg.hpp"  // distributed pressure MG (initMpi mode)
#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/leaf_halo.hpp"

namespace peclet::core::amr {

/// Truthy environment flag (unset / "" / "0" ⇒ false). Characterisation knobs only — never a
/// production configuration channel (those are the set* methods).
inline bool amrEnvFlag(const char* name) {
  const char* v = std::getenv(name);
  return v && v[0] && !(v[0] == '0' && v[1] == '\0');
}

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
inline void divergence(const FaceGeom& g, View<const double> u0, View<const double> u1,
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

/// Build the ABC/Basilisk divergence-free FACE field: uf(k) = ½(u^axis_i+u^axis_j) − (φ₊−φ₋)/dist
/// for face k of cell i (+axis velocity), from u* (call after the pressure solve, before the cell
/// gradient correction). Because L = D·G_face on the SAME (sub)faces, D(uf) = D u* − Lφ = 0 (to the
/// φ residual). Each (sub)face is written from both incident cells; the +axis-orientation build
/// makes the two copies identical, including across 2:1 interfaces (fine area +
/// (φ_fine−φ_coarse)/dist on each sub-face).
inline void buildFaceField(const FaceGeom& g, View<const double> u0, View<const double> u1,
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

/// L2 norm of the divergence of the FACE field uf (the div-free flux diagnostic / host-parity
/// check).
inline double divFaceNorm(const FaceGeom& g, View<const double> uf) {
  auto st = g.start;
  auto dr = g.dir;
  auto aA = g.alphaArea;
  auto iv = g.invVol;
  auto fl = g.fluid;
  double s = 0.0;
  Kokkos::parallel_reduce(
      "amr::flow_divface", g.n,
      KOKKOS_LAMBDA(const Index i, double& acc) {
        if (!fl(i))
          return;
        double d = 0.0;
        for (Index k = st(i); k < st(i + 1); ++k)
          d += aA(k) * dr(k) * uf(k);
        d *= iv(i);
        acc += d * d;
      },
      s);
  return std::sqrt(s);
}

/// ABC cell-gradient of a scalar field `f`: gx/gy/gz = ½(g⁻+g⁺) of the adjacent face
/// gradients along each axis, a closed face (α≤1e-12) contributing nothing (and not
/// counted). On fluid cells only. Mirrors oracle::AmrFlow::gradOf for all three components.
inline void grad3(const FaceGeom& g, View<const double> f, View<double> gx, View<double> gy,
                  View<double> gz) {
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
          if (al(k) <= 1e-12)
            continue;
          const int a = ax(k);
          const double gg = (dr(k) > 0) ? (f(nb(k)) - fi) / di(k) : (fi - f(nb(k))) / di(k);
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

/// Ghost-gradient overlay (setGhostGradient): per cut cell, a precomputed 3-point directional
/// FD stencil per axis replacing the ABC grad3 value — central where both axis-neighbour centers
/// are fluid, 2nd-order one-sided toward the fluid else (never reads a decoupled solid-centered
/// value; gradOf's read of the pinned p=0 through partially-open faces is a gauge-dependent
/// O(1/h) error at cut cells — measured in tests/study_amr_ghost_apriori.cpp). Weights include
/// the 1/h factors; unused entries carry w=0.
struct GhostGradOverlay {
  Index n = 0;                 ///< number of overlay (cut) cells
  View<Index> cell;            ///< [n] leaf index
  View<Index> idx;             ///< [n*9] stencil cell, slot s*9 + axis*3 + k
  View<double> w;              ///< [n*9] stencil weight (0 = unused)
};

/// Overwrite gx/gy/gz on the overlay cells with the directional stencil applied to `f`.
inline void applyGhostGrad(const GhostGradOverlay& ov, View<const double> f, View<double> gx,
                           View<double> gy, View<double> gz) {
  if (ov.n == 0)
    return;
  auto cell = ov.cell;
  auto idx = ov.idx;
  auto w = ov.w;
  Kokkos::parallel_for(
      "amr::flow_ghostgrad", ov.n, KOKKOS_LAMBDA(const Index s) {
        const Index i = cell(s);
        double g[3];
        for (int a = 0; a < 3; ++a) {
          double acc = 0.0;
          for (int k = 0; k < 3; ++k)
            acc += w(s * 9 + a * 3 + k) * f(idx(s * 9 + a * 3 + k));
          g[a] = acc;
        }
        gx(i) = g[0];
        gy(i) = g[1];
        gz(i) = g[2];
      });
}

/// Momentum RHS for one component: b_i = fluid ? (idiag·u_i + f_c − gradP_i − adv_i)·rscale_i : 0
/// (== AmrCutCell::makeRhs of the oracle::AmrFlow predictor source, u_bc = 0). `adv` is the
/// explicit deferred-correction advection term ρ(SOU−FOU) (zero for Stokes / fully-implicit at
/// steady).
inline void momRhs(View<const double> uc, View<const double> gradP, View<const double> adv,
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
inline void buildFou(const FaceGeom& g, View<const double> u0, View<const double> u1,
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
          for (Index k = st(i); k < st(i + 1); ++k)
            advCoef(k) = 0.0;
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
inline void deferredSou(const FaceGeom& g, View<const double> u0, View<const double> u1,
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
          if (!fl(j))
            continue;
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
/// `setImplicitAdvection(false)` fallback). Same SOU/TVD reconstruction as deferredSou.
inline void advectExplicit(const FaceGeom& g, View<const double> u0, View<const double> u1,
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
      "amr::flow_advexpl", g.n, KOKKOS_LAMBDA(const Index i) {
        if (!fl(i)) {
          defc(i) = 0.0;
          return;
        }
        auto fld = [&](Index c) { return (comp == 0) ? u0(c) : (comp == 1) ? u1(c) : u2(c); };
        double sou = 0.0;
        for (Index k = st(i); k < st(i + 1); ++k) {
          const Index j = nb(k);
          if (!fl(j))
            continue;
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
inline void correct(View<double> uc, View<const double> gphi, View<const char> fluid, Index n) {
  Kokkos::parallel_for(
      "amr::flow_correct", n, KOKKOS_LAMBDA(const Index i) {
        if (fluid(i))
          uc(i) -= gphi(i);
      });
}

/// Rotational incremental pressure update: p += (ρ/dt)φ − μ·div, on fluid cells.
inline void presUpdate(View<double> p, View<const double> phi, View<const double> div,
                       View<const char> fluid, double rho_dt, double mu, Index n) {
  Kokkos::parallel_for(
      "amr::flow_presupd", n, KOKKOS_LAMBDA(const Index i) {
        if (fluid(i))
          p(i) += rho_dt * phi(i) - mu * div(i);
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
    dist_ = nullptr;  // single-rank unless initMpi is used
  }

  /// Distributed mode (docs/amr_distributed_flow.md, rung 4): run this solver on one ORB block
  /// of a DistributedOctree — the whole step (momentum, pressure, overlays) then executes
  /// multi-rank through the LeafHalo ±2 ghost registry established in setSolid. The octree
  /// must be graded + cross-block 2:1 balanced; the origin is the GLOBAL one (every rank
  /// samples the geometry at bit-identical world points). Periodic domains only. All the
  /// set* configuration calls apply as usual (before setSolid); setSolid / step / project are
  /// COLLECTIVE over the octree's communicator. np=1 is bit-identical to init() + the
  /// single-rank path on the same octree by construction.
  void initMpi(DistributedOctree<3, Bits>& d) {
    init(d.local(), d.h0(), d.globalGeometry().origin);
    dist_ = &d;
  }
  void setDensity(double rho) { rho_ = rho; }
  void setViscosity(double mu) { mu_ = mu; }
  void setDt(double dt) { dt_ = dt; }
  void setBodyForce(double fx, double fy, double fz) { f_ = {fx, fy, fz}; }
  /// Use MG-preconditioned CG for the pressure solve (default) vs plain V-cycles.
  void setPressurePCG(bool on) { presPCG_ = on; }
  /// Directional ghost cell-gradient for the −∇pⁿ predictor and the projection's cell
  /// correction (the AMR analog of flow's collocated `set_face_interp(9)` hybrid): on cut cells
  /// the ABC grad3 reads the DECOUPLED p=0 of solid-centered neighbours through partially-open
  /// faces — a gauge-dependent O(1/h) gradient error (measured in
  /// tests/study_amr_ghost_apriori.cpp). The directional gradient is central where both axis
  /// neighbours are fluid-centered and 2nd-order one-sided toward the fluid else — O(h²),
  /// gauge-exact. The aperture projection (divergence + φ solve + uf) is UNCHANGED (throat-safe).
  ///
  /// DEFAULT ON since 2026-08-18, mirroring flow's collocated default
  /// `set_collocated_scheme("gauge-exact")`. Measured there on two periodic sphere beds
  /// (peclet-examples benchmarks/porous-scaling): second order at phi=0.50 AND at a contact-tight
  /// phi=0.60, against first order for the plain gradient — which additionally failed to reach
  /// steady state within 800 steps on three of five rungs of the dense bed. `setGhostGradient(false)`
  /// restores the legacy path. Call before setSolid.
  void setGhostGradient(bool on) { ghostGrad_ = on; }
  /// FULL directional ghost-cell projection (the AMR port of flow's collocated
  /// set_ghost_projection): the pressure system becomes rho·(L_bin + Delta) phi = rho·D_g(u*) —
  /// the BINARY-openness FV Laplacian (a face is open iff its sample + both adjacent centers are
  /// fluid; preconditioned by the UNCHANGED openness-MG hierarchy built on that openness) plus
  /// the wall-anchored closure overlay on the finest-band rows (peclet::core::scheme, shared with
  /// flow), solved by MG-preconditioned BiCGStab on the coupled subspace; the constraint is the
  /// ghost-closed divergence of the ½/½ face-averaged field. Implies setGhostGradient.
  /// (matrixOrder, rhsOrder) = closure orders for the implicit matrix vs the RHS divergence;
  /// PRODUCTION CANDIDATE (re-framed 2026-08-24 after flow's attractor campaign,
  /// flow/doc/collocated_invisible_subspace.md + fluid_only_constraint_plan.md): the ghost
  /// projection is the FLUID-ONLY constraint scheme — the only measured collocated architecture
  /// that is stable, unique (no attractor family: the aperture scheme's solid-centered pressure
  /// DOFs are constraint multipliers its gauge-exact gradient never reads, giving an affine
  /// family of steady states selected by march protocol) AND converging (flow clean ladders
  /// both beds; Z&H −0.018% at N=128). Default remains OFF (aperture, matching flow's
  /// gauge-exact default) until the suite-wide flip decision; select with
  /// setGhostProjection(true) == flow's set_collocated_scheme("ghost").
  ///
  /// Closure orders: (2, 2) is the DEFAULT and the only order pair cleared for production —
  /// flow's hardening Phase A measured the (1, 2) mixed form march-UNSTABLE above ~2000 spheres
  /// (an operator/RHS mismatch re-injected each step through the rotational pressure
  /// accumulation; flow/doc/ghost_hardening_findings_A.md). (1, 2) stays callable for the
  /// existing parity records only. Throws in setSolid if an overlay row's ±2 reach crosses a
  /// 2:1 level boundary (finest-band margin). The 2026-08-19 advection-economics retirement
  /// (aperture MG-PCG cheaper per step, docs/amr_aperture_advection_plan.md §RESOLVED) still
  /// holds as a COST statement; the scheme choice is now a robustness/uniqueness decision.
  /// Call before setSolid.
  /// Aperture estimator order for the (fallback) aperture projection: 2 = analytic
  /// marching-squares (DEFAULT since 2026-08-26), 1 = legacy one-sample model. Before setSolid.
  void setApertureOrder(int order) { apertureOrder_ = order; }
  void setGhostProjection(bool on, int matrixOrder = 2, int rhsOrder = 2) {
    ghostProjReq_ = on ? 1 : 0;  // explicit selection disables the AUTO default
    gpMatrixOrder_ = matrixOrder;
    gpRhsOrder_ = rhsOrder;
  }
  /// SAMPLED ghost projection: the mixed-level cut-band mode (docs/amr_mixed_level_cut_band_plan.md
  /// — cut cells at MULTIPLE octree levels; chain entries across 2:1 boundaries become degree-2
  /// LS virtual samples; no finest-band contract / band throw; level-aware canonical openness;
  /// momentum ξ-row seam correction rides along). Implies the ghost projection (engages when the
  /// resolved scheme is ghost — the AUTO default or an explicit setGhostProjection(true)).
  /// Single-rank only (the distributed sample halo is a later rung). Call before setSolid.
  void setGhostSampled(bool on) { ghostSampledReq_ = on ? 1 : 0; }
  /// Coarse/fine (2:1) interface scheme (cf_scheme.hpp): 0 = standard two-point flux (default,
  /// 1st-order at level boundaries, bit-identical legacy path), 1 = Martin–Cartwright tangential
  /// quadratic (2nd-order — measured at C/F rows: divergence 1.95, cell gradient 1.95, momentum
  /// SOLUTION 2.0 in test_amr_cf_vector). Applied to everything the STEADY solution feels: the
  /// momentum diffusion (lagged deferred-correction RHS term), the RHS divergence constraint,
  /// and the pressure gradients (predictor + cell correction). The pressure MATRIX / MG rails /
  /// PCG / ghost BiCGStab stay on the standard consistent operator (at the fixed point φ→0, the
  /// matrix C/F order does not move the steady solution — the (1,2)-mixed philosophy). Works in
  /// both aperture and ghost-projection modes. Call before setSolid.
  void setCfScheme(int scheme) { cfScheme_ = static_cast<CfScheme>(scheme); }
  /// Enable momentum advection ∇·(u u) (default OFF ⇒ Stokes). The high-order flux is
  /// second-order upwind (SOU) by default; the first-order-upwind part is solved *implicitly*
  /// (folded into the momentum operator) and the (SOU−FOU) difference is the explicit deferred
  /// correction — unconditionally stable for the FOU part, exact-SOU at steady state. This is
  /// the same implicit-FOU + deferred-SOU scheme as the host oracle::AmrFlow and flow's collocated
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
  /// staircase (VelocityMG, mirroring flow's VelocityMG). Call before setSolid. Both are
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
  /// — MG-preconditioned defect correction `u ← u + M⁻¹(b − A u)` iterated to tolerance — instead
  /// of BiCGStab with the same MG as a preconditioner. This is the faithful mirror of flow's
  /// velocity solve, which uses no Krylov method at all (RB-GS smoother / velocity-MG directly):
  /// the momentum operator is diagonally dominant and invertible, so the stationary iteration
  /// converges, and unlike BiCGStab it cannot break down on the strongly non-symmetric (D_rescale +
  /// FOU) operator (no bi-orthogonal recurrence to lose — the failure mode P5 had to symmetrise the
  /// smoother to avoid). With the GS smoother (setMomentumGS) and a velocity-MG (setMomentumMG,
  /// default on) this is the full RB-GS/MG flow path; with MG off it degrades to plain
  /// damped-Jacobi sweeps (flow's smoothComp). Default off (BiCGStab stays the validated default).
  /// Requires setMomentumMG to actually be multigrid-accelerated; otherwise the Richardson
  /// iteration converges only slowly.
  void setMomentumMGSolver(bool on) { momMGSolver_ = on; }

  /// Optional Picard outer loop over the lagged advection (mirror of flow's outerIters_): each
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
    if (dist_ && cfScheme_ != CfScheme::standard)
      throw std::runtime_error(
          "amr::AmrFlow: the C/F quadratic scheme is not distributed yet (rung-4 follow-up)");
    // Phase profiler (PECLET_CORE_PROFILE_SETUP=1): setSolid is the whole setup cost at bed
    // scale (measured 127 us/leaf single-threaded, [[performance-sota-yardstick]]) — the
    // per-phase breakdown is what any optimization must start from.
    const bool prof_ = [] {
      const char* e = std::getenv("PECLET_CORE_PROFILE_SETUP");
      return e && e[0] == '1';
    }();
    auto profT0_ = std::chrono::steady_clock::now();
    auto profPhase = [&](const char* name) {
      if (!prof_)
        return;
      Kokkos::fence();
      auto now = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(now - profT0_).count();
      std::fprintf(stderr, "[setSolid] %-28s %9.1f ms  (%6.1f us/leaf)\n", name, ms,
                   1e3 * ms / static_cast<double>(n));
      profT0_ = std::chrono::steady_clock::now();
    };
    // Host operator build (geometry, openness, cut stencils) — same as oracle::AmrFlow::setSolid.
    mom_.init(*t_, h0_, origin_);
    pres_.init(*t_, h0_);
    pres_.setOrigin(origin_);
    profPhase("mom.init + pres.init");
    // Resolve the projection mode. DEFAULT SWITCH (2026-08-25, user decision): AUTO = the
    // GHOST (fluid-only) projection — family-free/stable/unique (see the setter doc) — falling
    // back to the aperture projection with a stderr notice when the finest band is too thin for
    // the overlay (probed below; the July AUTO arm restored, now unconditional). Explicit
    // setGhostProjection(true/false) pins the scheme (band violations then THROW as before).
    // The overlay is built below (it needs only mom_'s sdf samples + pres_'s topology walk,
    // no openness).
    const bool wantGhost = (ghostProjReq_ != 0);  // explicit on (1) or AUTO (-1)
    ghostProj_ = wantGhost;                       // provisional; AUTO may fall back below
    ghostSampled_ = (ghostSampledReq_ == 1) && wantGhost;
    if (ghostSampled_ && dist_) {
      // D1 (docs/amr_march_perf_and_distributed_plan.md): the sampled builders now run through the
      // distributed seam — their probes join prepareDistributed's miss-collect fixpoint below and
      // the builder works in the global frame. np=1 is enabled and gated bit-identical to the
      // single-rank path (tests/test_amr_distributed_seam_mpi.cpp); np>1 additionally needs the
      // clouds to READ ghost slots, which is rung D2, so it still refuses.
      int sz = 1;
      MPI_Comm_size(dist_->comm(), &sz);
      if (sz > 1)
        throw std::runtime_error(
            "amr::AmrFlow: setGhostSampled is np=1 only so far (the LS clouds do not yet read "
            "ghost slots — rung D2)");
    }
    if (dist_) {
      // Distributed: install the resolver seams and run every prober to the miss-collect
      // fixpoint (docs/amr_distributed_flow.md). Freezes the ±2 halo, leaves mom_ FULLY built
      // (its final round ran with every ghost resolved) and the solver hooks installed.
      prepareDistributed(sdfFn);
    } else {
      nExt_ = n;
      allred_ = {};
      momSolver_.setDistributed({}, {}, 0);
      pcg_.setDistributed({}, {}, 0);
      mom_.build(sdfFn, /*idiag=*/rho_ / dt_, /*beta=*/mu_ / (h0_ * h0_));
    }
    profPhase("mom.build (SDF+cut stencils)");
    GhostOverlay hov;
    GhostOverlaySampled hovS;
    if (ghostSampled_) {  // mixed-level cut band: sample-slot overlay, no band-margin probe
      const auto gfine = globalFineExtent();
      hovS = buildGhostOverlaySampled(*t_, pres_, sdfFn, gpMatrixOrder_, gpRhsOrder_, origin_,
                                      &gfine, shiftD_);
      profPhase("buildGhostOverlaySampled");
    } else if (ghostProj_) {  // probe the band margin; explicit request throws on violation, AUTO falls
      bool viol = false;
      hov = buildGhostOverlay(*t_, pres_, mom_.sdfCRaw(), gpMatrixOrder_, gpRhsOrder_, &viol);
      if (dist_) {
        // COLLECTIVE band-margin decision: the flag is agreed across ranks before anyone
        // commits (one rank throwing/falling back alone would deadlock the MG collectives).
        int lv = viol ? 1 : 0, gv = 0;
        MPI_Allreduce(&lv, &gv, 1, MPI_INT, MPI_LOR, dist_->comm());
        viol = gv != 0;
      }
      if (viol) {
        if (ghostProjReq_ == 1)
          throw std::runtime_error(
              "amr ghost projection: an overlay row's ±2 closure reach crosses a 2:1 level "
              "boundary — widen the refineToSdf band margin");
        ghostProj_ = false;  // AUTO: fall back to the aperture projection
        fprintf(stderr,
                "peclet::core AmrFlow: AUTO scheme fell back to the aperture projection (the "
                "finest band is too thin for the ghost overlay). Select explicitly with "
                "setGhostProjection to silence this notice.\n");
      }
    }
    if (ghostProj_) {
      // Ghost projection: the pressure geometry is the BINARY openness on the unchanged MG
      // rails; the closure physics lives in the overlay (built above). Sampled mode: the
      // level-aware canonical rule (the sampled overlay's face states are FORCED to it — the
      // overlay-closed <=> binary-closed invariant).
      if (ghostSampled_) {
        auto binFn = makeBinaryOpenFnMixed(
            *t_, pres_, [&sdfFn](const Vec<3>& p) { return sdfFn(p); }, h0_, origin_, shiftD_);
        pres_.buildOpenness(binFn);
        profPhase("buildOpenness (mixed)");
        // D1: the pressure hierarchy has to be the DISTRIBUTED one when a seam is installed —
        // gpOp0() reads presMGD_ whenever dist_ is set, so building presMG_ here left the ghost
        // solver pointing at an empty operator.
        if (dist_)
          presMGD_.build(*dist_, h0_, binFn, &dhalo_);
        else
          presMG_.build(*t_, h0_, binFn, /*periodic=*/true);
        profPhase("presMG.build");
      } else {
        auto binFn = makeBinaryOpenFn([&sdfFn](const Vec<3>& p) { return sdfFn(p); }, h0_);
        pres_.buildOpenness(binFn);
        if (dist_)
          presMGD_.build(*dist_, h0_, binFn, &dhalo_);
        else
          presMG_.build(*t_, h0_, binFn, /*periodic=*/true);
      }
      ghostGrad_ = true;  // the directional gradient is part of the scheme
      // Fragmentation guard: pockets outside the main binary component are decoupled (see
      // findPocketCells) — folded into maskC_ below and hidden from the directional gradients.
      // Single-rank host BFS only: the DISTRIBUTED label-propagation guard is a rung-6 item
      // (a rank-local BFS would mislabel components that span ranks), so multi-rank runs are
      // unprotected on fragmenting geometries until it lands.
      gpPocket_ = dist_ ? std::vector<char>{} : findPocketCells(*t_, pres_, mom_.sdfCRaw());
      profPhase("findPocketCells");
    } else {
      gpPocket_.clear();
      auto openFn = [&](const Vec<3>& fc, int axis) { return faceFrac(sdfFn, fc, axis); };
      pres_.buildOpenness(openFn);
      if (dist_)
        presMGD_.build(*dist_, h0_, openFn, &dhalo_);
      else
        presMG_.build(*t_, h0_, openFn, /*periodic=*/true);
    }
    // Singular periodic pressure: per-level nullspace projection.
    if (dist_) {
      presMGD_.setRemoveMean(true);
      pcg_.setDistributed([this](View<double> v) { presMGD_.sync(0, v); }, allred_,
                          presMGD_.extendedSize(0));
    } else {
      presMG_.setRemoveMean(true);
    }
    profPhase("nullspace setup");

    // Device assembly (D6): the static cut-cell momentum operator CSR and the collocated face
    // geometry are assembled ON THE DEVICE (assembleMomentum / assembleFaceGeom), and the
    // pressure MG operators are device-assembled per level (Multigrid D5) — so no host CSR walk and
    // no operator round-trip. Each is bit-for-bit identical to the host assembler on OpenMP (locked
    // in test_amr_momentum / test_amr_facegeom), so the flow result is unchanged. A shared device
    // octree view backs both assemblers.
    // DISTRIBUTED: assemble on the HOST through the resolver seam instead (the device walkers
    // cannot resolve cross-block probes) and upload — same parity-locked builders, ghost
    // columns included.
    BlockOctreeView<3, Bits> ov;
    if (!dist_)
      ov.upload(*t_);
    if (dist_) {
      auto A = mom_.assembleOperator();
      momOp_ = MomentumOp{};
      momOp_.n = n;
      momOp_.diag = toDevice(A.diag, "mo_diag");
      momOp_.faceStart = toDevice(A.start, "mo_start");
      momOp_.faceNbr = toDevice(A.nbr, "mo_nbr");
      momOp_.faceCoef = toDevice(A.coef, "mo_coef");
    } else {
      momOp_ = assembleMomentum<Bits>(mom_, ov);  // static Stokes operator (hasAdv stays false)
      profPhase("assembleMomentum (device)");
    }
    // Velocity multigrid (momentum preconditioner): the Galerkin hierarchy A_c = R·A·P built
    // directly from the exact assembled momentum CSR. Consistent with the fine cut-cell
    // operator by construction (inherits the ξ-overlay + D_rescale row scaling; a coarse cell
    // of all-solid children stays an identity row). It only changes the preconditioner (the
    // BiCGStab matvec is the exact operator) ⇒ same converged solution, but the iteration
    // count stays ~flat with N instead of growing like the Jacobi-preconditioned BiCGStab.
    // Build the chosen momentum-MG: Galerkin (MomentumMG) by default, or the rediscretized
    // staircase (VelocityMG) — both from the static Stokes operator, once.
    // DISTRIBUTED: the Galerkin MG becomes RANK-LOCAL (additive-Schwarz preconditioner): the
    // local rows of the exact operator with the ghost COLUMNS dropped. At np=1 nothing is
    // dropped, so the preconditioner — and hence the whole step — stays bit-identical to the
    // single-rank path; at np>1 it is a block preconditioner (iterations may grow with np,
    // the converged step is unchanged — the preconditioner never moves the solution). The
    // EXACT cross-rank Galerkin RAP is the documented follow-up
    // (docs/amr_distributed_flow.md §4c); the staircase variant is single-rank-only for now
    // (distributed falls back to Jacobi-with-halo preconditioning).
    if (momMGon_ && dist_ && !useStaircaseMG_) {
      auto A = mom_.assembleOperator();
      std::vector<Index> lstart(1, 0);
      std::vector<Index> lnbr;
      std::vector<double> lcoef;
      for (Index i = 0; i < n; ++i) {
        for (Index k = A.start[static_cast<std::size_t>(i)];
             k < A.start[static_cast<std::size_t>(i) + 1]; ++k)
          if (A.nbr[static_cast<std::size_t>(k)] < n) {  // drop ghost columns (Schwarz cut)
            lnbr.push_back(A.nbr[static_cast<std::size_t>(k)]);
            lcoef.push_back(A.coef[static_cast<std::size_t>(k)]);
          }
        lstart.push_back(static_cast<Index>(lnbr.size()));
      }
      momMG_.build(*t_, A.diag, lstart, lnbr, lcoef);
      momMG_.setGaussSeidel(momGS_);
    } else if (momMGon_ && !dist_) {
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
    profPhase("velocity MG build");
    std::vector<char> fluidVec(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i)
      fluidVec[static_cast<std::size_t>(i)] = mom_.isFluid(i) ? 1 : 0;
    if (dist_) {
      // Host walker through the resolver seam (nbr/upup may be ghost slots), then EXTEND the
      // fluid flags over the ghost tail — the advection kernels read fl(j)/fl(upup) at ghost
      // slots (mom_.fluidRaw() carries the ghost metadata filled from the world SdfFn).
      geom_ = buildFaceGeom(pres_, [&](Index i) { return mom_.isFluid(i); });
      geom_.fluid = toDevice(mom_.fluidRaw(), "fg_fluid_ext");
    } else {
      geom_ = assembleFaceGeom<Bits>(pres_, fluidVec, ov);
    }
    profPhase("assembleFaceGeom");
    std::vector<double> rs(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i)
      rs[static_cast<std::size_t>(i)] = mom_.rhsScale(i);
    rscale_ = toDevice(rs, "df_rscale");
    fluid_ = geom_.fluid;
    if (ghostGrad_ && !ghostSampled_)
      buildGhostGradOverlay();
    else
      gc_ = GhostGradOverlay{};  // sampled mode: the CSR overlay gcS_ owns all cut cells
    // C/F interface scheme overlays (cf_scheme.hpp): the same host builders the oracle uses
    // (parity by construction), uploaded once. Momentum delta = ×μ on the α=1 velocity geometry.
    if (cfScheme_ != CfScheme::standard) {
      auto fluidOk = [&](Index i) { return mom_.isFluid(i); };
      auto rowRegular = [&](Index i) { return mom_.isFluid(i) && !mom_.isCut(i); };
      cfMom_ = uploadCfCsr(buildCfLapDelta(mom_.lap(), *t_, mu_, rowRegular, fluidOk, cfScheme_),
                           "cf_mom");
      // cfDiv/cfGrad rows: REGULAR fluid only — cut rows belong to the ghost closure family
      // (the overlay owns their divergence and gradients; adding the smooth-field C/F
      // substitution on top gives the constraint velocity reads the row's gradient never sees —
      // a support-consistency violation). Measured (two-sphere throat, P3a follow-up
      // 2026-08-27): with cut rows included, 2 of 12 throat-graded meshes march to k~1e12 by
      // step ~100; the cfDiv delta alone carries it (momentum/gradient/uf deltas exonerated by
      // bisection), and this row gate alone restores stability. On finest-band meshes cut rows
      // have no C/F faces, so the gate is inert there — bit-identity by geometry.
      cfDiv_ = uploadCfCompCsr(buildCfDivDelta(pres_, *t_, rowRegular, fluidOk, cfScheme_),
                               "cf_div");
      auto gd = buildCfGradDelta(pres_, *t_, rowRegular, fluidOk, cfScheme_);
      for (int a = 0; a < 3; ++a)
        cfGrad_[static_cast<std::size_t>(a)] =
            uploadCfCsr(gd[static_cast<std::size_t>(a)], "cf_grad");
      auto ufd = buildCfUfDelta(pres_, *t_, fluidOk, cfScheme_);
      cfUfVel_ = uploadCfCompCsr(ufd.vel, "cf_ufvel");
      cfUfPhi_ = uploadCfCsr(ufd.phi, "cf_ufphi");
    } else {
      cfMom_ = CfCsrDev{};
      cfDiv_ = CfCompCsrDev{};
      for (int a = 0; a < 3; ++a)
        cfGrad_[static_cast<std::size_t>(a)] = CfCsrDev{};
      cfUfVel_ = CfCompCsrDev{};
      cfUfPhi_ = CfCsrDev{};
    }
    profPhase("cf overlays");
    if (ghostProj_) {
      // Closure overlay (pre-built in the mode-resolve step above) + the coupled-subspace mask
      // for the BiCGStab projection. Sampled mode uploads the sample-slot overlay instead (the
      // classic gpOv_ stays empty — every delta site calls both, empties no-op).
      gpOv_ = ghostSampled_ ? GhostOverlayDev{} : uploadGhostOverlay(hov);
      gpOvS_ = ghostSampled_ ? uploadGhostOverlaySampled(hovS) : GhostOverlaySampledDev{};
      const GhostOverlay& rows = ghostSampled_ ? hovS.base : hov;
      std::vector<double> mc(static_cast<std::size_t>(n), 0.0);
      for (Index i = 0; i < n; ++i)
        mc[static_cast<std::size_t>(i)] =
            (mom_.isFluid(i) && !(!gpPocket_.empty() && gpPocket_[static_cast<std::size_t>(i)]))
                ? 1.0
                : 0.0;
      for (Index r = 0; r < rows.n; ++r)
        if (!rows.coupled[static_cast<std::size_t>(r)])
          mc[static_cast<std::size_t>(rows.cell[static_cast<std::size_t>(r)])] = 0.0;
      maskC_ = toDevice(mc, "gp_maskc");
      // Sampled mode: momentum ξ-row seam correction (host CSR folds 1/rscale for the oracle's
      // pre-rscale source; the device adds POST-rscale via cfApply, so re-fold rscale per row)
      // + the CSR directional-gradient overlay (cascade over sample functionals; classic
      // same-level fallback on cut cells without a row — the oracle's gradP split).
      if (ghostSampled_) {
        CfCsr msd = buildMomSeamDelta(hovS, *t_, pres_, mom_, sdfFn, origin_, rho_ / dt_, mu_);
        for (Index i = 0; i < n; ++i) {
          const double rs = mom_.rhsScale(i);
          for (Index k = msd.start[static_cast<std::size_t>(i)];
               k < msd.start[static_cast<std::size_t>(i) + 1]; ++k)
            msd.coef[static_cast<std::size_t>(k)] *= rs;
        }
        gpsMomDelta_ = uploadCfCsr(msd, "gps_momdelta");
        auto classicOk = [&](Index j, Index i) {
          return j >= 0 && mom_.isFluid(j) && pres_.levelOf(j) == pres_.levelOf(i) &&
                 !(!gpPocket_.empty() && j < n && gpPocket_[static_cast<std::size_t>(j)]);
        };
        gcS_ = uploadGhostGradCsr(buildSampledGradCsr(
            hovS, *t_, pres_, [&](Index i) { return mom_.isCut(i); }, classicOk));
      } else {
        gpsMomDelta_ = CfCsrDev{};
        gcS_ = GhostGradCsrDev{};
      }
      // Krylov scratch carries the ghost tail (matvec inputs) in distributed mode.
      auto mk = [&](const char* l) { return View<double>(l, static_cast<std::size_t>(nExt_)); };
      gpr_ = mk("gp_r");
      gprh_ = mk("gp_rhat");
      gpp_ = mk("gp_p");
      gpph_ = mk("gp_phat");
      gpv_ = mk("gp_v");
      gps_ = mk("gp_s");
      gpsh_ = mk("gp_shat");
      gpt_ = mk("gp_t");
    }

    // Device state. Fields whose ghost entries are READ through the CSRs / overlays (u, p, φ)
    // — plus div (its View is deep_copied into the nExt-sized distributed MG rhs) and the u
    // snapshots (full-extent deep_copies of u) — carry the ghost tail [n, nExt); nExt == n
    // single-rank, so those allocations are unchanged there.
    for (int c = 0; c < 3; ++c) {
      u_[c] = View<double>("df_u", static_cast<std::size_t>(nExt_));
      gx_[c] = View<double>("df_g", static_cast<std::size_t>(n));
      Kokkos::deep_copy(u_[c], 0.0);
    }
    p_ = View<double>("df_p", static_cast<std::size_t>(nExt_));
    phi_ = View<double>("df_phi", static_cast<std::size_t>(nExt_));
    div_ = View<double>("df_div", static_cast<std::size_t>(nExt_));
    uf_ = View<double>("df_uf",
                       geom_.nbr.extent(0));  // ABC divergence-free face field (per CSR face)
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
      u0_[c] = View<double>("df_u0", static_cast<std::size_t>(nExt_));
      uprev_[c] = View<double>("df_uprev", static_cast<std::size_t>(nExt_));
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
    // the solver from the MG type so Galerkin and staircase are interchangeable. Distributed:
    // the rank-local Galerkin hierarchy built above (staircase unsupported there ⇒ Jacobi).
    if (momMGon_ && !(dist_ && useStaircaseMG_)) {
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
    profPhase("overlays upload + state alloc");
  }

  /// One incompressible step on device (Stokes, or Navier–Stokes with setAdvection). `momIters`
  /// BiCGStab iterations for each momentum component; `presIters` PCG iterations for the
  /// pressure solve.
  // ---- M0 (docs/amr_march_perf_and_distributed_plan.md): the per-phase STEP profiler -----------
  // `PECLET_CORE_PROFILE_STEP=1` accumulates FENCED per-phase timings over a window of steps
  // (`PECLET_CORE_PROFILE_STEP_WINDOW`, default 50) and prints ms/step and µs/leaf per phase next
  // to the iteration counters — the instrument Phase M's attribution matrix reads. The four
  // pressure-solve rows are NESTED inside `pressure solve` (the residual is Krylov vector work and
  // the reductions), so they are printed indented and must not be added to the total.
  //
  // OFF (the default) every hook is one bool test: no fence, no clock, no change to the kernel
  // sequence — so a profiler-off march is bitwise what it was before the profiler existed.
  enum StepPhase {
    SP_GLUE,
    SP_GRAD,
    SP_CF,
    SP_OVERLAY,
    SP_ADV,
    SP_MOMRHS,
    SP_MOMSOLVE,
    SP_DIV,
    SP_PRES,
    SP_PRES_MV,
    SP_PRES_MVOV,
    SP_PRES_PROJ,
    SP_PRES_PC,
    SP_FINISH,
    SP_N
  };
  using SpClock = std::chrono::steady_clock;
  SpClock::time_point spMark() const {
    if (!stepProf_)
      return SpClock::time_point{};
    Kokkos::fence();
    return SpClock::now();
  }
  void spAdd(int ph, SpClock::time_point t0) {
    if (!stepProf_)
      return;
    Kokkos::fence();
    spAcc_[ph] += std::chrono::duration<double, std::milli>(SpClock::now() - t0).count();
  }
  /// Close a profiled step; print + reset when the window fills.
  void spEndStep() {
    if (!stepProf_)
      return;
    ++spSteps_;
    spMom_ += lastMomIters_;
    spPres_ += lastPresIters_;
    spOuter_ += lastOuterIters_;
    if (!spHeader_) {  // the static hierarchy shape, on the first profiled step (H-mg's
                       // denominator, and the one line a contention-exposed box can still trust)
      const std::size_t nl = dist_ ? presMGD_.numLevels() : presMG_.numLevels();
      std::fprintf(stderr, "[step-prof] leaves %lld | pressure MG levels %zu:", (long long)n_, nl);
      for (std::size_t L = 0; L < nl; ++L)
        std::fprintf(stderr, " %lld",
                     (long long)(dist_ ? presMGD_.numLeaves(L) : presMG_.numLeaves(L)));
      std::fprintf(stderr, "\n");
      spHeader_ = true;
    }
    if (spSteps_ < spWindow_)
      return;
    static const char* kName[SP_N] = {
        "glue (copies + syncs)", "grad3 (binary)",     "cf delta kernels",
        "overlay delta kernels", "advection build",    "momentum rhs",
        "momentum solve",        "divergence",         "pressure solve",
        "  . binary matvec",     "  . overlay matvec", "  . projection",
        "  . MG preconditioner", "finish projection"};
    const double w = static_cast<double>(spSteps_);
    const double leaves = static_cast<double>(n_);
    double tot = 0.0;
    for (int k = 0; k < SP_N; ++k)
      if (k < SP_PRES_MV || k > SP_PRES_PC)
        tot += spAcc_[k];
    std::fprintf(stderr, "[step-prof] %d steps | %.3f ms/step (%.3f us/leaf) | mom %.1f it, "
                         "pres %.1f it, outer %.2f\n",
                 spSteps_, tot / w, 1e3 * tot / w / leaves, spMom_ / w, spPres_ / w, spOuter_ / w);
    for (int k = 0; k < SP_N; ++k) {
      const bool nested = (k >= SP_PRES_MV && k <= SP_PRES_PC);
      std::fprintf(stderr, "[step-prof]   %-22s %8.3f ms/step  %8.4f us/leaf  %5.1f%%%s\n",
                   kName[k], spAcc_[k] / w, 1e3 * spAcc_[k] / w / leaves,
                   100.0 * spAcc_[k] / (tot > 0 ? tot : 1.0), nested ? "  (nested)" : "");
    }
    for (int k = 0; k < SP_N; ++k)
      spAcc_[k] = 0.0;
    spSteps_ = 0;
    spMom_ = spPres_ = spOuter_ = 0.0;
  }

  void step(int momIters = 100, int presIters = 60) {
    const Index n = n_;
    const double idiag = rho_ / dt_;
    lastMomIters_ = 0;
    lastOuterIters_ = 1;
    const auto spT0 = spMark();
    // Freeze the time-level uⁿ for the backward-Euler mass term + warm start; it stays anchored
    // across the Picard outer iterations (only the advecting velocity re-lags). For outerIters_==1
    // this is just a copy of uⁿ ⇒ bit-identical to the single lagged step.
    for (int c = 0; c < 3; ++c)
      Kokkos::deep_copy(u0_[c], View<const double>(u_[c]));
    // −∇p^n is constant across the outer iterations (pressure is projected once, after the loop,
    // like flow's single per-step projection) ⇒ hoist it out.
    syncScalar(p_);  // ghost tail of pⁿ before the gradient reads (no-op single-rank)
    spAdd(SP_GLUE, spT0);
    auto spT = spMark();
    grad3(geom_, View<const double>(p_), gx_[0], gx_[1], gx_[2]);
    spAdd(SP_GRAD, spT);
    spT = spMark();
    for (int a = 0; a < 3; ++a)  // 2nd-order C/F face gradients (level-boundary rows)
      cfApply(cfGrad_[static_cast<std::size_t>(a)], View<const double>(p_), gx_[a]);
    spAdd(SP_CF, spT);
    spT = spMark();
    applyGhostGrad(gc_, View<const double>(p_), gx_[0], gx_[1], gx_[2]);
    applyGhostGradCsr(gcS_, View<const double>(p_), gx_[0], gx_[1], gx_[2]);
    spAdd(SP_OVERLAY, spT);
    // Picard outer loop over the lagged advection only (the momentum nonlinearity); for
    // outerIters_==1 this is the single lagged predictor, then one projection — bit-identical to
    // before.
    for (int outer = 0; outer < outerIters_; ++outer) {
      // --- advection lagged to the *current* predictor iterate (uⁿ on the first pass):
      // implicit-FOU operator + explicit ρ(SOU−FOU) deferred correction. The matvec stays linear
      // during each solve (the advecting velocity is frozen in advDiag_/advCoef_). With advection
      // OFF the operator and RHS are identical every pass, so a second pass reproduces the first ⇒
      // instant early-stop. ---
      spT = spMark();
      if (advect_) {
        momOp_.hasAdv = implicitFou_;
        syncVel();  // ghost tails of the (lagged) advecting velocity for buildFou/deferredSou
        // Advect with the divergence-free face field uf (from the previous projection); fall back
        // to ½(u_i+u_j) before the first projection has built it. The implicit FOU and the explicit
        // SOU/FOU deferred correction use the SAME velocity ⇒ the FOU cancels at steady state
        // (host-parity).
        const View<const double> ufv(uf_);
        if (implicitFou_)
          buildFou(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                   View<const double>(u_[2]), rho_, View<const double>(rscale_), advDiag_, advCoef_,
                   ufv, faceFieldBuilt_);
        for (int c = 0; c < 3; ++c) {
          if (implicitFou_)
            deferredSou(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                        View<const double>(u_[2]), c, rho_, advScheme_, defc_[c], ufv,
                        faceFieldBuilt_);
          else
            advectExplicit(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                           View<const double>(u_[2]), c, rho_, advScheme_, defc_[c], ufv,
                           faceFieldBuilt_);
        }
        // The staircase MG's fine level mirrors the sharp operator; refresh it so it picks up the
        // current advection state (hasAdv). (The Galerkin MG is the static viscous operator.)
        if (momMGon_ && useStaircaseMG_)
          velMG_.setFineOp(momOp_);
      }
      spAdd(SP_ADV, spT);
      // --- predictor: incremental BE viscous (+ implicit-FOU) solve per component, RHS carries
      // −∇p^n and −ρ(SOU−FOU); the mass term is anchored at uⁿ (u0_), the solve warm-starts at the
      // current iterate. ---
      for (int c = 0; c < 3; ++c) {
        spT = spMark();
        momRhs(View<const double>(u0_[c]), View<const double>(gx_[c]), View<const double>(defc_[c]),
               View<const double>(rscale_), View<const char>(fluid_), idiag, f_[c], bmom_, n);
        spAdd(SP_MOMRHS, spT);
        spT = spMark();
        // C/F-scheme deferred correction on the velocity diffusion: b += μ(∇²_scheme − ∇²_std)
        // of the lagged component (regular rows only, rscale = 1 there).
        cfApply(cfMom_, View<const double>(u_[c]), bmom_);
        // Momentum ξ-row seam correction (sampled mode; rscale re-folded at upload).
        cfApply(gpsMomDelta_, View<const double>(u_[c]), bmom_);
        spAdd(SP_CF, spT);
        spT = spMark();
        // P4 (opt-in): the velocity-MG used as the *solver* — MG-preconditioned defect correction,
        // no Krylov (the flow RB-GS/velocity-MG mirror; cannot break down on the non-symmetric
        // operator) — vs the default MG-preconditioned BiCGStab. Both reach the same solution (the
        // matvec is the exact operator); the choice only trades robustness for convergence rate.
        lastMomIters_ +=
            (momMGSolver_ ? momSolver_.solveDefectCorrection(
                                momOp_, u_[c], View<const double>(bmom_), momIters, momTol_)
                          : momSolver_.solveBiCGStab(momOp_, u_[c], View<const double>(bmom_),
                                                     momIters, momTol_))
                .iters;
        spAdd(SP_MOMSOLVE, spT);
      }
      // Outer-loop convergence on the predictor velocity (skipped for the default outerIters_==1,
      // so that path is untouched). With advection off the second iterate equals the first ⇒ stops
      // at 2.
      if (outerIters_ > 1) {
        lastOuterIters_ = outer + 1;
        if (outer > 0) {
          double dmax = 0.0;
          for (int c = 0; c < 3; ++c)
            dmax = std::max(
                dmax, maxAbsDiff(View<const double>(u_[c]), View<const double>(uprev_[c]), n));
          if (dmax < outerTol_)
            break;
        }
        for (int c = 0; c < 3; ++c)
          Kokkos::deep_copy(uprev_[c], View<const double>(u_[c]));
      }
    }
    project(presIters);  // single pressure projection per step (flow structure)
    spEndStep();
  }

  /// Pressure projection of the current velocity in place.
  void project(int presIters = 60) {
    const Index n = n_;
    auto spT = spMark();
    syncVel();  // ghost tails of u* for the divergence (+ the ghost-closed overlay delta)
    divergence(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
               View<const double>(u_[2]), div_);
    spAdd(SP_DIV, spT);
    spT = spMark();
    cfApplyComp(cfDiv_, View<const double>(u_[0]), View<const double>(u_[1]),
                View<const double>(u_[2]), div_);  // 2nd-order C/F face averages (setCfScheme)
    spAdd(SP_CF, spT);
    spT = spMark();
    if (ghostProj_) {  // ghost-closed constraint: binary div (geom_ carries binary α) + overlay
      ghostDivergDelta(gpOv_, View<const double>(u_[0]), View<const double>(u_[1]),
                       View<const double>(u_[2]), div_);
      ghostDivergDeltaSampled(gpOvS_, View<const double>(u_[0]), View<const double>(u_[1]),
                              View<const double>(u_[2]), div_);
    }
    spAdd(SP_OVERLAY, spT);
    Kokkos::deep_copy(phi_, 0.0);
    if (ghostProj_) {
      spT = spMark();
      lastPresIters_ = solveGhostBiCGStab(phi_, View<const double>(div_), presIters);
      spAdd(SP_PRES, spT);
      spT = spMark();
      finishProjection(n);
      spAdd(SP_FINISH, spT);
      return;
    }
    // Two selectable pressure drivers, like flow's CutcellMG: MG-PCG (default, presPCG_) and the
    // bounded stationary V-cycle (setPressurePCG(false)). MG-PCG covers ADVECTION too — the
    // historic exclusion ("transient near-nullspace issue") was characterised 2026-08-19 and was
    // NOT a property of the operator (which is geometry-only, advection-independent, SPD in the
    // volume-weighted inner product): the aperture RHS is INCOMPATIBLE by a fluid-mean component
    // (div_ is zeroed at solid-centered cells whose faces are partially open, breaking the
    // telescoping that would make Σ V·div = 0 over the operator's DOF set; the defect grows with
    // the developed flow, ~3e-3 relative at steady state on the Z&H sphere). The un-deflated
    // V-cycle's residual therefore STALLS at exactly |mean|·sqrt(V_fluid) — the old "60 bounded
    // cycles" was a stagnation cap, not a convergence count. The PCG's per-iteration fluid-range
    // projection (maskSolid + volume-weighted fluid-mean removal) deflates exactly that component,
    // so CG is valid and healthy: measured flat 15–17 iters/step (tol 1e-10) across the whole
    // impulsive N=32 transient, steady K identical to the V-cycle path to 4+ digits.
    // Debug knob (env, default-off): PECLET_CORE_AMR_PRES_DEBUG=1 — per-cycle/-solve residual +
    // RHS-compatibility trace to stderr (the characterisation instrumentation, kept).
    const bool dbg = amrEnvFlag("PECLET_CORE_AMR_PRES_DEBUG");
    if (dbg && !dist_) {
      // RHS compatibility: the operator's left null vector is the constant over fluid cells in the
      // volume-weighted inner product, so a solvable RHS needs Σ V_i·div_i ≈ 0 over fluid cells.
      const FvOp& op0 = presMG_.op(0);
      auto st = op0.faceStart;
      auto w = op0.faceW;
      auto bcD = op0.bcDiag;
      auto iv = op0.invVol;
      auto dv = div_;
      double su = 0.0, sv = 0.0, sn = 0.0;
      Kokkos::parallel_reduce(
          "amr::dbg_compat", n,
          KOKKOS_LAMBDA(const Index i, double& a, double& b, double& c) {
            double d = bcD(i);
            for (Index k = st(i); k < st(i + 1); ++k)
              d += w(k);
            const double m = (d > 1e-30) ? 1.0 : 0.0;
            a += m * dv(i) / iv(i);
            b += m / iv(i);
            c += m * dv(i) * dv(i) / iv(i);
          },
          su, sv, sn);
      std::fprintf(stderr, "[amr pres] rhs fluid-mean=%.3e |rhs|_D=%.3e (rel mean %.3e)\n",
                   su / sv, std::sqrt(sn), (su / sv) / (std::sqrt(sn) + 1e-300));
      if (!presDbgSpdDone_) {
        // One-shot direct SPD probe of the assembled pressure operator (advection cannot enter
        // the assembly — this measures it): symmetry <y,Lx>_D vs <x,Ly>_D on deterministic
        // pseudo-random vectors, and the Rayleigh quotient sign (L negative-semidefinite).
        presDbgSpdDone_ = true;
        View<double> xr("dbg_x", static_cast<std::size_t>(n)),
            yr("dbg_y", static_cast<std::size_t>(n)), Ax("dbg_Ax", static_cast<std::size_t>(n)),
            Ay("dbg_Ay", static_cast<std::size_t>(n));
        Kokkos::parallel_for(
            "amr::dbg_fill", n, KOKKOS_LAMBDA(const Index i) {
              xr(i) = std::sin(0.7 * static_cast<double>(i) + 0.3);
              yr(i) = std::cos(1.3 * static_cast<double>(i) + 1.1);
            });
        applyFv(op0, View<const double>(xr), Ax);
        applyFv(op0, View<const double>(yr), Ay);
        auto dotD = [&](View<const double> a, View<const double> b) {
          double s = 0.0;
          Kokkos::parallel_reduce(
              "amr::dbg_dot", n,
              KOKKOS_LAMBDA(const Index i, double& acc) { acc += a(i) * b(i) / iv(i); }, s);
          return s;
        };
        const double yLx = dotD(View<const double>(yr), View<const double>(Ax));
        const double xLy = dotD(View<const double>(xr), View<const double>(Ay));
        const double xLx = dotD(View<const double>(xr), View<const double>(Ax));
        const double yLy = dotD(View<const double>(yr), View<const double>(Ay));
        std::fprintf(stderr,
                     "[amr pres] SPD probe: <y,Lx>_D=%.15e <x,Ly>_D=%.15e rel asym=%.2e; "
                     "Rayleigh <x,Lx>_D=%.3e <y,Ly>_D=%.3e (must be <=0)\n",
                     yLx, xLy, std::fabs(yLx - xLy) / (std::fabs(yLx) + 1e-300), xLx, yLy);
      }
    }
    spT = spMark();
    if (presPCG_) {
      const auto R =
          dist_ ? pcg_.solve(presMGD_, phi_, View<const double>(div_), presIters, 1e-10)
                : pcg_.solve(presMG_, phi_, View<const double>(div_), presIters, 1e-10);
      lastPresIters_ = R.iters;
      if (dbg)
        std::fprintf(stderr, "[amr pres] pcg iters=%d res0=%.3e res=%.3e rel=%.3e\n", R.iters,
                     R.res0, R.res, R.res0 > 0 ? R.res / R.res0 : 0.0);
    } else if (dist_) {
      Kokkos::deep_copy(presMGD_.b(0), div_);
      Kokkos::deep_copy(presMGD_.x(0), 0.0);
      for (int it = 0; it < presIters; ++it)
        presMGD_.vcycle(2, 2, 60, 0.8);
      Kokkos::deep_copy(phi_, presMGD_.x(0));
      lastPresIters_ = presIters;
    } else {
      Kokkos::deep_copy(presMG_.b(0), div_);
      Kokkos::deep_copy(presMG_.x(0), 0.0);
      View<double> rdbg;
      if (dbg)
        rdbg = View<double>("pres_dbg_res", static_cast<std::size_t>(n));
      for (int it = 0; it < presIters; ++it) {
        presMG_.vcycle(2, 2, 60, 0.8);
        if (dbg) {
          residualFv(presMG_.op(0), View<const double>(presMG_.x(0)),
                     View<const double>(presMG_.b(0)), rdbg);
          const double rn =
              std::sqrt(dotPlain(View<const double>(rdbg), View<const double>(rdbg), n));
          std::fprintf(stderr, "[amr pres] vcycle %2d |r|=%.6e\n", it + 1, rn);
        }
      }
      Kokkos::deep_copy(phi_, presMG_.x(0));
      lastPresIters_ = presIters;
    }
    spAdd(SP_PRES, spT);
    spT = spMark();
    finishProjection(n);
    spAdd(SP_FINISH, spT);
  }

  /// Shared projection tail: build the div-free face field from u* + φ, correct the cell
  /// velocities (ABC / ghost gradient), rotational pressure update. div_ holds the projection's
  /// RHS divergence (aperture or ghost-closed).
  void finishProjection(Index n) {
    // u*'s ghost tail is current (project() synced it and u is untouched since); φ's is not.
    syncScalar(phi_);
    buildFaceField(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                   View<const double>(u_[2]), View<const double>(phi_), uf_);
    // 2nd-order C/F face values (setCfScheme): distance-weighted average + coarse* substitution
    // on the 2:1 sub-faces — the advecting flux matches the (quad) divergence constraint.
    cfApplyComp(cfUfVel_, View<const double>(u_[0]), View<const double>(u_[1]),
                View<const double>(u_[2]), uf_);
    cfApply(cfUfPhi_, View<const double>(phi_), uf_);
    faceFieldBuilt_ = true;
    grad3(geom_, View<const double>(phi_), gx_[0], gx_[1], gx_[2]);
    for (int a = 0; a < 3; ++a)  // 2nd-order C/F face gradients (level-boundary rows)
      cfApply(cfGrad_[static_cast<std::size_t>(a)], View<const double>(phi_), gx_[a]);
    applyGhostGrad(gc_, View<const double>(phi_), gx_[0], gx_[1], gx_[2]);
    applyGhostGradCsr(gcS_, View<const double>(phi_), gx_[0], gx_[1], gx_[2]);
    for (int c = 0; c < 3; ++c)
      correct(u_[c], View<const double>(gx_[c]), View<const char>(fluid_), n);
    presUpdate(p_, View<const double>(phi_), View<const double>(div_), View<const char>(fluid_),
               rho_ / dt_, mu_, n);
  }

  // ---- adaptivity during a run (ladder step 5) -------------------------------------------------
  // The octree is borrowed by pointer and mutated EXTERNALLY (refine/coarsen/balance/adapt on the
  // same object). beginAdapt snapshots the current topology + fields; after the mutation,
  // finishAdapt conservatively remaps u and the accumulated rotational p onto the new mesh
  // (minmod-limited linear transferField) and rebuilds every solver structure via setSolid. The
  // caller must keep the cut band at the finest level on the new mesh (refineToSdf the geometry
  // band again after a solution-driven adapt): the ghost overlay build throws / auto-falls-back
  // exactly as in setSolid. uf restarts from the ½-average fallback for one step.

  /// Snapshot the octree topology + (u, p) ahead of an external mesh mutation. Distributed:
  /// the mutation is distributedAdapt (+ the driver's geometry-band re-refinement + balance),
  /// which KEEPS the ORB ownership — so the snapshot, the conservative transferField in
  /// finishAdapt and the field restore are all block-local per rank, and finishAdapt's
  /// setSolid rebuilds the ±2 halo/operators on the new local mesh (collective). For
  /// OWNERSHIP changes use rebalanceMpi instead.
  void beginAdapt() {
    adaptOldT_ = std::make_unique<Octree>(*t_);
    for (int c = 0; c < 3; ++c)
      adaptU_[static_cast<std::size_t>(c)] = velocity(c);
    adaptP_ = pressure();
    if (dist_) {
      // Halo-completed prolongation gradients on the OLD mesh (while dist_ still holds it):
      // the block-local transfer stencil zeroes gradients at interior block boundaries,
      // which would make the remap np-dependent there (measured ~5% field divergence).
      for (int c = 0; c < 3; ++c)
        adaptGradU_[static_cast<std::size_t>(c)] =
            transferGradients(*dist_, adaptU_[static_cast<std::size_t>(c)]);
      adaptGradP_ = transferGradients(*dist_, adaptP_);
    }
  }

  /// Rebuild on the mutated octree and transfer the snapshotted fields onto it.
  template <class SdfFn>
  void finishAdapt(SdfFn&& sdfFn) {
    if (!adaptOldT_)
      throw std::runtime_error("amr::AmrFlow::finishAdapt called without beginAdapt");
    std::array<std::vector<double>, 3> nu;
    for (int c = 0; c < 3; ++c)
      nu[static_cast<std::size_t>(c)] =
          transferField(*adaptOldT_, adaptU_[static_cast<std::size_t>(c)], *t_, /*linear=*/true,
                        dist_ ? &adaptGradU_[static_cast<std::size_t>(c)] : nullptr);
    std::vector<double> np = transferField(*adaptOldT_, adaptP_, *t_, /*linear=*/true,
                                           dist_ ? &adaptGradP_ : nullptr);
    setSolid(sdfFn);  // full operator/overlay rebuild on the new topology (zeroes the fields)
    for (int c = 0; c < 3; ++c) {
      setVelocity(c, nu[static_cast<std::size_t>(c)]);
      zeroSolid(u_[c]);  // cells that became solid on the new mesh hold 0 (no-slip state)
    }
    setPressure(np);
    zeroSolid(p_);  // solid p is pinned/decoupled
    adaptOldT_.reset();
    for (int c = 0; c < 3; ++c)
      adaptU_[static_cast<std::size_t>(c)].clear();
    adaptP_.clear();
  }

  /// Distributed load rebalance (docs/amr_distributed_flow.md, rung 6): re-decompose the
  /// octree by leaf count (weighted ORB) and migrate the leaves WITH the state (u, p) to the
  /// new owners, then rebuild every solver structure on the new block (full distributed
  /// setSolid: new ±2 registry, halos, operators, pressure hierarchy). Pure redistribution —
  /// every cell's u/p value is preserved bit-for-bit; uf restarts from the ½-average fallback
  /// for one step (as after finishAdapt). Collective; distributed mode only.
  template <class SdfFn>
  void rebalanceMpi(SdfFn&& sdfFn) {
    if (!dist_)
      throw std::runtime_error("amr::AmrFlow::rebalanceMpi requires initMpi");
    std::vector<std::vector<double>> cols(4);
    for (int c = 0; c < 3; ++c)
      cols[static_cast<std::size_t>(c)] = velocity(c);
    cols[3] = pressure();
    dist_->rebalance(cols);  // t_ still points at dist_->local(), now the new block's octree
    setSolid(sdfFn);         // full rebuild (registry, halos, operators) on the new block
    for (int c = 0; c < 3; ++c) {
      setVelocity(c, cols[static_cast<std::size_t>(c)]);
      zeroSolid(u_[c]);
    }
    setPressure(cols[3]);
    zeroSolid(p_);
  }

  /// Write the accumulated rotational pressure from host (restart / finishAdapt).
  void setPressure(const std::vector<double>& h) {
    auto m = Kokkos::create_mirror_view(p_);
    for (Index i = 0; i < n_; ++i)
      m(i) = h[static_cast<std::size_t>(i)];
    Kokkos::deep_copy(p_, m);
  }

  /// Zero a per-leaf field on non-fluid cells (the transferred fields' solid cleanup).
  void zeroSolid(View<double> v) {
    auto fl = fluid_;
    Kokkos::parallel_for(
        "amr::flow_zerosolid", n_, KOKKOS_LAMBDA(const Index i) {
          if (!fl(i))
            v(i) = 0.0;
        });
  }

  /// DEBUG: the raw high-order advection ∇·(u u_comp) per cell from the current velocity
  /// (== host oracle::AmrFlow::advectTerm). Isolates the SOU kernel from the solve.
  std::vector<double> debugSou(int comp) {
    syncVel();
    View<double> s("dbg_sou", static_cast<std::size_t>(n_));
    advectExplicit(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
                   View<const double>(u_[2]), comp, 1.0, advScheme_, s, View<const double>(uf_),
                   false);
    std::vector<double> h(static_cast<std::size_t>(n_));
    auto m = Kokkos::create_mirror_view(s);
    Kokkos::deep_copy(m, s);
    for (Index i = 0; i < n_; ++i)
      h[static_cast<std::size_t>(i)] = m(i);
    return h;
  }
  /// Run one V-cycle of a momentum MG as a preconditioner: z = M⁻¹ r. Templated on the MG type
  /// (MomentumMG or VelocityMG — both expose b(0)/x(0)/vcycle), so the BiCGStab
  /// preconditioner is decoupled from the coarse-operator strategy.
  template <class MG>
  void runMgVcycle(MG& mg, View<const double> r, View<double> z) {
    // Element copies over the LOCAL rows (not deep_copy): in distributed mode the Krylov
    // scratch carries a ghost tail while the (rank-local) MG levels are local-sized; the
    // solver refreshes z's ghost tail before the next matvec. Same values single-rank.
    const Index nl = mg.numLeaves(0);
    {
      auto b0 = mg.b(0);
      Kokkos::parallel_for(
          "amr::mgv_b", nl, KOKKOS_LAMBDA(const Index i) { b0(i) = r(i); });
    }
    Kokkos::deep_copy(mg.x(0), 0.0);
    mg.vcycle(mgVcPre_, mgVcPre_, mgVcBottom_, 0.7);
    {
      auto x0 = mg.x(0);
      Kokkos::parallel_for(
          "amr::mgv_z", nl, KOKKOS_LAMBDA(const Index i) { z(i) = x0(i); });
    }
  }
  /// Max |a − b| over all cells (the Picard outer-loop convergence measure).
  static double maxAbsDiff(View<const double> a, View<const double> b, Index n) {
    double m = 0.0;
    Kokkos::parallel_reduce(
        "amr::flow_maxdiff", n,
        KOKKOS_LAMBDA(const Index i, double& lm) {
          double d = a(i) - b(i);
          if (d < 0.0)
            d = -d;
          if (d > lm)
            lm = d;
        },
        Kokkos::Max<double>(m));
    return m;
  }
  /// Copy a device View into a host vector (sized n_).
  void copyToHost(const View<double>& d, std::vector<double>& h) const {
    auto m = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(m, d);
    for (Index i = 0; i < n_; ++i)
      h[static_cast<std::size_t>(i)] = m(i);
  }
  /// Set a velocity component from host (testing / initial conditions).
  void setVelocity(int c, const std::vector<double>& h) {
    auto m = Kokkos::create_mirror_view(u_[c]);
    for (Index i = 0; i < n_; ++i)
      m(i) = h[static_cast<std::size_t>(i)];
    Kokkos::deep_copy(u_[c], m);
  }
  /// Copy a velocity component back to host (single D2H, no host loop — S2a). Local rows only
  /// (the distributed ghost tail is an implementation detail).
  std::vector<double> velocity(int c) const { return localVector(u_[c]); }
  /// Copy the pressure field back to host (single D2H), (num_leaves,) — the incremental-rotational
  /// p. Local rows only.
  std::vector<double> pressure() const { return localVector(p_); }

  /// Host copy of the LOCAL rows of a (possibly ghost-extended) per-cell field.
  std::vector<double> localVector(const View<double>& v) const {
    if (v.extent(0) == static_cast<std::size_t>(n_))
      return peclet::core::toVector(v);
    View<double> packed(Kokkos::view_alloc("amr::local_packed", Kokkos::WithoutInitializing),
                        static_cast<std::size_t>(n_));
    Kokkos::parallel_for(
        "amr::pack_local", n_, KOKKOS_LAMBDA(const Index i) { packed(i) = v(i); });
    return peclet::core::toVector(packed);
  }

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
    return peclet::core::toVector(packed);
  }
  /// L2 norm of the (openness-weighted) divergence of the current velocity — the ghost-closed
  /// divergence when the ghost projection is on (its constraint IS the residual diagnostic).
  double divNormL2() {
    syncVel();
    divergence(geom_, View<const double>(u_[0]), View<const double>(u_[1]),
               View<const double>(u_[2]), div_);
    cfApplyComp(cfDiv_, View<const double>(u_[0]), View<const double>(u_[1]),
                View<const double>(u_[2]), div_);
    if (ghostProj_) {
      ghostDivergDelta(gpOv_, View<const double>(u_[0]), View<const double>(u_[1]),
                       View<const double>(u_[2]), div_);
      ghostDivergDeltaSampled(gpOvS_, View<const double>(u_[0]), View<const double>(u_[1]),
                              View<const double>(u_[2]), div_);
    }
    return std::sqrt(allSum(dotPlain(View<const double>(div_), View<const double>(div_), n_)));
  }
  /// L2 norm of the divergence of the ABC face field uf_ (built each project()): the φ-solve
  /// residual, far below the cell field's O(h²) divNormL2 — including across 2:1 interfaces.
  double divNormFace() {
    const double l = divFaceNorm(geom_, View<const double>(uf_));
    return std::sqrt(allSum(l * l));
  }
  /// Copy the divergence-free face field to host (one value per CSR (sub)face, forEachFaceFull
  /// order).
  std::vector<double> faceField() const { return peclet::core::toVector(uf_); }
  Index numLeaves() const { return n_; }
  /// Distributed: number of ghost slots in the ±2 registry (0 single-rank).
  Index numGhostCells() const { return nExt_ - n_; }
  /// Per-leaf fluid mask (false inside the solid) — for host-side post-processing / bindings.
  bool isFluid(Index i) const { return mom_.isFluid(i); }
  /// Total momentum BiCGStab iterations (summed over the 3 components) of the last step.
  int lastMomIters() const { return lastMomIters_; }
  /// Pressure PCG iterations of the last step.
  int lastPresIters() const { return lastPresIters_; }
  /// Picard outer iterations actually run in the last step (1 unless setOuterIterations(>1)).
  int lastOuterIters() const { return lastOuterIters_; }

  // ---- distributed plumbing (docs/amr_distributed_flow.md, rung 4) ----------------------------

  /// Refresh the ghost tails of the three velocity components (one batched message round).
  void syncVel() {
    if (dist_)
      dhex_.exchange3(u_[0], u_[1], u_[2]);
  }
  /// Refresh the ghost tail of one cell scalar (p, φ, Krylov scratch).
  void syncScalar(View<double> v) {
    if (dist_)
      dhex_.exchange(v);
  }
  /// Global sum (identity single-rank).
  double allSum(double s) const {
    if (!dist_)
      return s;
    double g = 0.0;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, dist_->comm());
    return g;
  }
  /// The DOMAIN's fine extent (D1): the block's own single-rank, where block == domain. The
  /// sampled overlay's minimum-image period must be this and never the block's, or the cloud
  /// metric would differ per rank (finding F2 in decomposition-dependent form).
  std::array<long, 3> globalFineExtent() const {
    std::array<long, 3> g{};
    for (int a = 0; a < 3; ++a)
      g[a] = dist_ ? static_cast<long>(dist_->globalFineSize()[a])
                   : static_cast<long>(pres_.fineExt()[a]);
    return g;
  }

  /// The pressure operator the ghost solver runs on (distributed MG level 0 or presMG_'s).
  const FvOp& gpOp0() { return dist_ ? presMGD_.op(0) : presMG_.op(0); }

  /// Install the resolver seams, run every prober to the miss-collect fixpoint, freeze the ±2
  /// halo, hook the solvers. Leaves mom_ FULLY built (its final round ran with all ghosts
  /// resolved). Collective.
  template <class SdfFn>
  void prepareDistributed(SdfFn&& sdfFn) {
    dhalo_.init(*dist_);
    for (int a = 0; a < 3; ++a)
      shiftD_[a] = dist_->blockFineOrigin()[a];
    auto resv = [hp = &dhalo_, sh = shiftD_](const std::array<long, 3>& p) -> Index {
      std::array<long, 3> g = p;
      for (int a = 0; a < 3; ++a)
        g[a] += sh[a];
      return hp->resolveGlobal(g);
    };
    mom_.setFrameShift(shiftD_);
    mom_.setResolver(resv);
    pres_.setFrameShift(shiftD_);
    pres_.setResolver(resv);
    const Index n = t_->numLeaves();
    const double beta = mu_ / (h0_ * h0_);
    for (;;) {
      installGhostMeta();  // metadata for every ghost known so far (same-round hits read it)
      mom_.build(sdfFn, rho_ / dt_, beta);  // ±1 probes (also fills the extended sdfC/fluid)
      // FaceGeom probes: the full face enumeration + the SOU upstream-of-upwind ±2 reach.
      // Host-parallel since the F1 resolution (miss registration is mutex-guarded and the miss
      // SET is order-canonical — see LeafHalo::resolve); pure discovery, results discarded.
      hostParFor(n, [&](Index i) {
        pres_.forEachFaceFull(i, [&](Index j, int ax, int dr, double, double, double) {
          (void)pres_.periodicNeighbor(i, ax, -dr);
          if (j >= 0)
            (void)pres_.periodicNeighbor(j, ax, dr);
        });
      });
      // Overlay / directional-gradient ±2 chains (every cut cell is a non-clean overlay row,
      // so this covers buildGhostGradOverlay's probes too). Discovery only — result discarded.
      if (ghostSampled_) {
        // D1 (DD2): the SAMPLED builders are probers too, and they reach further than the classic
        // ±2 chain — an LS cloud descends a whole box of radius 2.2·max(h,H) around each virtual
        // position, and the mixed openness rule probes ±¼h0 off every face centroid. Both run here
        // exactly as they will run for real, tolerating kPending (results discarded until the last
        // round) the way mom_.build does. Note the descent deliberately does NOT descend into a
        // pending region: it learns one octree level per round, so this fixpoint takes a few more
        // rounds than the classic ±2 one — which is why it is a fixpoint and not a fixed count.
        const auto gfine = globalFineExtent();
        (void)buildGhostOverlaySampled(*t_, pres_, sdfFn, gpMatrixOrder_, gpRhsOrder_, origin_,
                                       &gfine, shiftD_);
        auto binFn = makeBinaryOpenFnMixed(
            *t_, pres_, [&sdfFn](const Vec<3>& p) { return sdfFn(p); }, h0_, origin_, shiftD_);
        pres_.buildOpenness(binFn);  // discovery only; setSolid rebuilds it after the fixpoint
      } else if (ghostProj_ || ghostGrad_) {
        bool viol = false;
        (void)buildGhostOverlay(*t_, pres_, mom_.sdfCRaw(), gpMatrixOrder_, gpRhsOrder_, &viol);
      }
      if (dhalo_.resolveMisses() == 0)
        break;
    }
    installGhostMeta();
    dhalo_.finalize();
    nExt_ = dhalo_.extendedSize();
    dhex_.init(dhalo_);
    allred_ = [this](double s) { return allSum(s); };
    momSolver_.setDistributed([this](View<double> v) { dhex_.exchange(v); }, allred_, nExt_);
  }

  /// Mirror the halo registry's ghost metadata (block-local lo + level) into mom_ and pres_.
  void installGhostMeta() {
    const Index ng = dhalo_.numGhosts();
    std::vector<std::array<long, 3>> glo(static_cast<std::size_t>(ng));
    std::vector<unsigned> glv(static_cast<std::size_t>(ng));
    for (Index g = 0; g < ng; ++g) {
      for (int a = 0; a < 3; ++a)
        glo[static_cast<std::size_t>(g)][a] =
            static_cast<long>(dhalo_.ghostCoord(g)[a]) - shiftD_[a];
      glv[static_cast<std::size_t>(g)] =
          static_cast<unsigned>(dhalo_.level(dhalo_.numLocal() + g));
    }
    mom_.setGhosts(glo, glv);  // copies — pres_ takes the originals
    pres_.setGhosts(std::move(glo), std::move(glv));
  }

  // (public like runMgVcycle: nvcc rejects extended device lambdas in private member functions)
  // Project onto the coupled subspace: pin decoupled rows (solid-centered + no-phi-coupling
  // overlay rows) to 0 and remove the volume-weighted mean over the coupled cells (the constant
  // null mode of the connected fluid region) — removeMeanVol with the coupled mask (the mean
  // reduced globally in distributed mode; allred_ is empty single-rank ⇒ bit-identical).
  void gpProject(View<double> v) {
    removeMeanVolReduced(v, gpOp0().invVol, maskC_, n_, allred_);
  }

  // Nonsymmetric ghost pressure matvec: y = P[rho·(L_bin x + Delta x)]. The caller keeps x's
  // ghost tail current (syncScalar before every call in distributed mode).
  void ghostMatvec(View<const double> x, View<double> y) {
    auto spT = spMark();
    applyFv(gpOp0(), x, y);
    spAdd(SP_PRES_MV, spT);
    spT = spMark();
    ghostApplyDelta(gpOv_, x, y);
    ghostApplyDeltaSampled(gpOvS_, x, y);
    spAdd(SP_PRES_MVOV, spT);
    spT = spMark();
    gpProject(y);
    spAdd(SP_PRES_PROJ, spT);
  }

  // Preconditioner: two binary-openness V-cycles (the unchanged MG hierarchy) + projection.
  void ghostPrec(View<const double> r, View<double> z) {
    const auto spT = spMark();
    if (dist_) {
      Kokkos::deep_copy(presMGD_.b(0), r);
      Kokkos::deep_copy(presMGD_.x(0), 0.0);
      presMGD_.vcycle(2, 2, 60, 0.8);
      presMGD_.vcycle(2, 2, 60, 0.8);
      Kokkos::deep_copy(z, presMGD_.x(0));
    } else {
      Kokkos::deep_copy(presMG_.b(0), r);
      Kokkos::deep_copy(presMG_.x(0), 0.0);
      presMG_.vcycle(2, 2, 60, 0.8);
      presMG_.vcycle(2, 2, 60, 0.8);
      Kokkos::deep_copy(z, presMG_.x(0));
    }
    gpProject(z);
    spAdd(SP_PRES_PC, spT);
  }

  // MG-preconditioned BiCGStab on the ghost pressure operator (device mirror of the oracle's
  // solveGhostBiCGStab: same projection, same stagnation guard against the small
  // attainable-residual floor of the slightly incompatible ghost system). Returns iterations.
  int solveGhostBiCGStab(View<double> x, View<const double> b, int maxIters, double tol = 1e-10) {
    const Index n = n_;
    // Dots are locally summed then globally reduced (allSum = identity single-rank); every
    // matvec input's ghost tail is refreshed first. Stagnation/early-break branches depend on
    // reduced scalars only ⇒ every rank takes the same branch.
    syncScalar(x);
    ghostMatvec(View<const double>(x), gpr_);
    {
      auto r = gpr_;
      auto bb = b;
      Kokkos::parallel_for(
          "amr::gp_r0", n, KOKKOS_LAMBDA(const Index i) { r(i) = bb(i) - r(i); });
    }
    gpProject(gpr_);
    Kokkos::deep_copy(gprh_, gpr_);
    const double res0 =
        std::sqrt(allSum(dotPlain(View<const double>(gpr_), View<const double>(gpr_), n)));
    if (res0 == 0.0)
      return 0;
    double rho = 1, alpha = 1, omega = 1, best = res0;
    int noImprove = 0;
    Kokkos::deep_copy(gpv_, 0.0);
    Kokkos::deep_copy(gpp_, 0.0);
    int it = 0;
    for (; it < maxIters; ++it) {
      const double rhoNew =
          allSum(dotPlain(View<const double>(gprh_), View<const double>(gpr_), n));
      if (rhoNew == 0.0)
        break;
      const double beta = (rhoNew / rho) * (alpha / omega);
      bicgPUpdate(gpp_, View<const double>(gpr_), View<const double>(gpv_), beta, omega, n);
      ghostPrec(View<const double>(gpp_), gpph_);
      syncScalar(gpph_);
      ghostMatvec(View<const double>(gpph_), gpv_);
      const double rhatV =
          allSum(dotPlain(View<const double>(gprh_), View<const double>(gpv_), n));
      if (rhatV == 0.0)
        break;
      alpha = rhoNew / rhatV;
      Kokkos::deep_copy(gps_, gpr_);
      axpy(gps_, -alpha, View<const double>(gpv_), n);
      const double snorm =
          std::sqrt(allSum(dotPlain(View<const double>(gps_), View<const double>(gps_), n)));
      if (snorm <= tol * res0) {
        axpy(x, alpha, View<const double>(gpph_), n);
        ++it;
        break;
      }
      ghostPrec(View<const double>(gps_), gpsh_);
      syncScalar(gpsh_);
      ghostMatvec(View<const double>(gpsh_), gpt_);
      const double tt = allSum(dotPlain(View<const double>(gpt_), View<const double>(gpt_), n));
      omega = (tt != 0.0)
                  ? allSum(dotPlain(View<const double>(gpt_), View<const double>(gps_), n)) / tt
                  : 0.0;
      axpy(x, alpha, View<const double>(gpph_), n);
      axpy(x, omega, View<const double>(gpsh_), n);
      Kokkos::deep_copy(gpr_, gps_);
      axpy(gpr_, -omega, View<const double>(gpt_), n);
      const double rnorm =
          std::sqrt(allSum(dotPlain(View<const double>(gpr_), View<const double>(gpr_), n)));
      if (rnorm <= tol * res0) {
        ++it;
        break;
      }
      if (rnorm < 0.999 * best) {
        best = rnorm;
        noImprove = 0;
      } else if (++noImprove >= 6) {
        ++it;
        break;  // attainable-residual floor (compatibility gap) — stagnation guard
      }
      rho = rhoNew;
      if (omega == 0.0)
        break;
    }
    gpProject(x);
    return it;
  }

 private:
  // Host build of the ghost-gradient overlay (setGhostGradient): one row per cut cell (fluid
  // with a solid face neighbour — the cells where the ABC grad3 is gauge-dependent O(1/h)),
  // holding a 3-point directional FD stencil per axis. Mirrors oracle::AmrFlow::gradOfDir: cut
  // cells have same-level face neighbours by the finest-band contract; the ±2 probe falls back
  // to the 2-point one-sided closure when that cell is solid or not same-level.
  void buildGhostGradOverlay() {
    const Index n = t_->numLeaves();
    std::vector<Index> cells;
    for (Index i = 0; i < n; ++i)
      if (mom_.isCut(i))
        cells.push_back(i);
    const Index m = static_cast<Index>(cells.size());
    std::vector<Index> idx(static_cast<std::size_t>(m) * 9, 0);
    std::vector<double> w(static_cast<std::size_t>(m) * 9, 0.0);
    // Pocket cells (fragmentation guard) count as solid for the directional gradient: their φ is
    // pinned/decoupled, and reading the pinned 0 is the gauge-dependent defect the ghost
    // gradient exists to avoid. Levels via pres_.levelOf: ghost-slot-safe in distributed
    // builds (identical to t_->level for local leaves; the pocket guard is single-rank-only —
    // gpPocket_ stays empty distributed until the label-propagation guard lands).
    auto ok = [&](Index j, Index i) {
      return j >= 0 && mom_.isFluid(j) && pres_.levelOf(j) == pres_.levelOf(i) &&
             !(!gpPocket_.empty() && j < n && gpPocket_[static_cast<std::size_t>(j)]);
    };
    for (Index s = 0; s < m; ++s) {
      const Index i = cells[static_cast<std::size_t>(s)];
      const double h = pres_.cellWidth(i);
      for (int a = 0; a < 3; ++a) {
        const std::size_t o = static_cast<std::size_t>(s) * 9 + static_cast<std::size_t>(a) * 3;
        for (int k = 0; k < 3; ++k)
          idx[o + static_cast<std::size_t>(k)] = i;  // safe defaults (w = 0)
        const Index jp = pres_.periodicNeighbor(i, a, +1);
        const Index jm = pres_.periodicNeighbor(i, a, -1);
        const bool ap = ok(jp, i), am = ok(jm, i);
        if (am && ap) {
          idx[o] = jp;
          w[o] = 0.5 / h;
          idx[o + 1] = jm;
          w[o + 1] = -0.5 / h;
        } else if (ap) {
          const Index jpp = pres_.periodicNeighbor(jp, a, +1);
          if (ok(jpp, i)) {
            idx[o] = i;
            w[o] = -1.5 / h;
            idx[o + 1] = jp;
            w[o + 1] = 2.0 / h;
            idx[o + 2] = jpp;
            w[o + 2] = -0.5 / h;
          } else {
            idx[o] = jp;
            w[o] = 1.0 / h;
            idx[o + 1] = i;
            w[o + 1] = -1.0 / h;
          }
        } else if (am) {
          const Index jmm = pres_.periodicNeighbor(jm, a, -1);
          if (ok(jmm, i)) {
            idx[o] = i;
            w[o] = 1.5 / h;
            idx[o + 1] = jm;
            w[o + 1] = -2.0 / h;
            idx[o + 2] = jmm;
            w[o + 2] = 0.5 / h;
          } else {
            idx[o] = i;
            w[o] = 1.0 / h;
            idx[o + 1] = jm;
            w[o + 1] = -1.0 / h;
          }
        }  // sandwiched: all weights stay 0
      }
    }
    gc_.n = m;
    gc_.cell = toDevice(cells, "gc_cell");
    gc_.idx = toDevice(idx, "gc_idx");
    gc_.w = toDevice(w, "gc_w");
  }

  // Fluid area fraction of a face. DEFAULT (order 2, 2026-08-26 user decision): triangle-fan
  // marching squares on FIVE ANALYTIC samples (4 corners + centre) -- exact linear fraction per
  // triangle, O(h^2), no saddle ambiguity, and the sub-resolution floor 1e-6 (alpha ~ 1e-12 rows
  // destroy the pressure conditioning -- flow tracker rows 51/52). Because the AMR solver holds
  // the analytic sdfFn, this samples the TRUE geometry (no trilinear ceiling). order 1 = the
  // legacy one-sample gradient-normalised linear model (kept for A/B), which carries a SIGNED
  // convexity bias measured at +0.59%/+0.27% bed permeability at R=8/12 (decay ~h^2).
  static double triFrac(double a, double b, double c) {
    const bool pa = a >= 0.0, pb = b >= 0.0, pc = c >= 0.0;
    const int np = (pa ? 1 : 0) + (pb ? 1 : 0) + (pc ? 1 : 0);
    if (np == 3)
      return 1.0;
    if (np == 0)
      return 0.0;
    double x, y, z;
    if (np == 1) {
      if (pa) { x = a; y = b; z = c; } else if (pb) { x = b; y = c; z = a; } else { x = c; y = a; z = b; }
      const double den = (x - y) * (x - z);
      return den > 1e-300 ? (x * x) / den : 1.0;
    }
    if (!pa) { x = a; y = b; z = c; } else if (!pb) { x = b; y = c; z = a; } else { x = c; y = a; z = b; }
    const double den = (x - y) * (x - z);
    return 1.0 - (den > 1e-300 ? (x * x) / den : 1.0);
  }
  template <class SdfFn>
  double faceFrac(SdfFn&& sdfFn, const Vec<3>& fc, int axis) const {
    if (apertureOrder_ >= 2) {
      if (sdfFn(fc) <= 0.0)
        return 0.0;  // center gate (kept from order 1; see flow ccFaceOpenMS)
      const int t1 = (axis + 1) % 3, t2 = (axis + 2) % 3;
      const double e = 0.5 * h0_;
      auto at = [&](double d1, double d2) {
        Vec<3> p = fc;
        p[t1] += d1;
        p[t2] += d2;
        return sdfFn(p);
      };
      const double c00 = at(-e, -e), c10 = at(e, -e), c11 = at(e, e), c01 = at(-e, e);
      const double cc = sdfFn(fc);
      const double frac = 0.25 * (triFrac(c00, c10, cc) + triFrac(c10, c11, cc) +
                                  triFrac(c11, c01, cc) + triFrac(c01, c00, cc));
      return frac < 1e-3 ? 0.0 : (frac > 1.0 - 1e-12 ? 1.0 : frac);  // floor: see flow ccFaceOpenMS
    }
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
  Vec<3> f_{};
  bool presPCG_ = true;
  bool presDbgSpdDone_ = false;  // one-shot debug SPD probe (PECLET_CORE_AMR_PRES_DEBUG)
  bool momMGon_ = true;  // velocity-MG momentum preconditioner (scalable; see setMomentumMG)
  bool useStaircaseMG_ = false;  // false = Galerkin (MomentumMG), true = staircase (VelocityMG)
  int mgVcPre_ = 2, mgVcBottom_ = 30;  // momentum-MG V-cycle pre/post sweeps + bottom sweeps
  Index mgMinCoarse_ = 256;            // staircase velocity-MG pore-scale cap (coarsest cell count)
  bool momGS_ = false;  // opt-in: multicolour Gauss–Seidel smoother in the momentum MG
  bool momMGSolver_ =
      false;            // opt-in (P4): velocity-MG as the solver (defect correction), not BiCGStab
  bool ghostGrad_ = true;   // gauge-exact directional cell gradient (setGhostGradient) — the
                            // DEFAULT since 2026-08-18, mirroring flow's collocated
                            // set_collocated_scheme("gauge-exact")
  bool ghostProj_ = false;    // RESOLVED projection mode (set by setSolid from the request)
  int8_t ghostProjReq_ = -1;  // -1 = AUTO (DEFAULT since 2026-08-25: ghost, aperture fallback
                              // on thin band), 0 = explicit aperture, 1 = explicit ghost
  int apertureOrder_ = 2;  // aperture estimator order (setApertureOrder; default 2)
  int gpMatrixOrder_ = 2, gpRhsOrder_ = 2;  // closure orders (2,2 = the production pair; the
                                            // (1,2) mixed form is march-unstable at scale)
  CfScheme cfScheme_ = CfScheme::standard;  // 2:1 C/F interface scheme (setCfScheme)
  int outerIters_ = 1;  // Picard outer iterations over the lagged advection (default 1)
  double outerTol_ = 1e-6;   // outer-loop early-stop tolerance on max|Δu|
  double momTol_ = 1e-8;     // per-step momentum BiCGStab relative tolerance (Phase-0 knob)
  bool advect_ = false;      // momentum advection ∇·(u u) (off ⇒ Stokes)
  bool implicitFou_ = true;  // implicit-FOU deferred-correction (stable) vs fully-explicit
  int advScheme_ = 0;        // high-order flux: 0 = SOU (default), 1 = Koren TVD
  Index n_ = 0;
  int lastMomIters_ = 0, lastPresIters_ = 0, lastOuterIters_ = 1;
  // M0 step profiler (PECLET_CORE_PROFILE_STEP): all inert unless stepProf_.
  bool stepProf_ = amrEnvFlag("PECLET_CORE_PROFILE_STEP");
  bool spHeader_ = false;
  int spWindow_ = [] {
    const char* e = std::getenv("PECLET_CORE_PROFILE_STEP_WINDOW");
    const int v = e ? std::atoi(e) : 0;
    return v > 0 ? v : 50;
  }();
  int spSteps_ = 0;
  double spAcc_[SP_N] = {};
  double spMom_ = 0.0, spPres_ = 0.0, spOuter_ = 0.0;

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
  GhostGradOverlay gc_;  // directional ghost-gradient overlay (empty unless setGhostGradient)
  GhostOverlayDev gpOv_;  // closure overlay (empty unless setGhostProjection)
  bool ghostSampled_ = false;          // RESOLVED sampled mode (set by setSolid)
  int8_t ghostSampledReq_ = 0;         // setGhostSampled request (mixed-level cut band)
  GhostOverlaySampledDev gpOvS_;       // sample-slot overlay (empty unless sampled)
  GhostGradCsrDev gcS_;                // sampled CSR directional-gradient overlay
  CfCsrDev gpsMomDelta_;               // momentum ξ-row seam correction (rscale-folded)
  std::vector<char> gpPocket_;  // fragmentation guard: 1 = decoupled pocket cell (ghost mode)
  CfCsrDev cfMom_;                  // +μ(∇²_scheme − ∇²_std) momentum RHS overlay
  CfCompCsrDev cfDiv_;              // (D_scheme − D_std) divergence overlay
  std::array<CfCsrDev, 3> cfGrad_;  // (G_scheme − G_std) per gradient axis
  CfCompCsrDev cfUfVel_;            // (uf_scheme − uf_std) face-field overlay: velocity part
  CfCsrDev cfUfPhi_;                //                                          φ part
  View<double> maskC_;    // 1 = coupled row (Krylov subspace), 0 = pinned
  View<double> gpr_, gprh_, gpp_, gpph_, gpv_, gps_, gpsh_, gpt_;  // ghost BiCGStab scratch
  View<double> rscale_;
  View<char> fluid_;
  std::array<View<double>, 3> u_, gx_;
  std::array<View<double>, 3> u0_,
      uprev_;  // frozen uⁿ (BE mass term) + previous Picard outer iterate
  View<double> p_, phi_, div_, bmom_;
  View<double> uf_;  // ABC/Basilisk divergence-free face field (one per CSR (sub)face)
  bool faceFieldBuilt_ =
      false;  // uf_ populated by a projection (else advection falls back to ½(u_i+u_j))
  std::unique_ptr<Octree> adaptOldT_;      // beginAdapt topology snapshot
  std::array<std::vector<double>, 3> adaptU_;  // beginAdapt field snapshots
  std::vector<double> adaptP_;
  std::array<std::vector<std::array<double, 3>>, 3> adaptGradU_;  // distributed transfer grads
  std::vector<std::array<double, 3>> adaptGradP_;

  // ---- distributed context (initMpi; all null/empty single-rank) -----------------------------
  DistributedOctree<3, Bits>* dist_ = nullptr;  // the ORB block + communicator
  LeafHalo<3, Bits> dhalo_;                     // the flow's ±2 ghost registry (frozen in setSolid)
  LeafHaloExchange dhex_;                       // device value refresh over dhalo_
  DistributedFlowMultigrid<3, Bits> presMGD_;   // distributed pressure MG (level 0 on dhalo_)
  std::array<long, 3> shiftD_{};                // block global fine origin
  Index nExt_ = 0;                              // n_ + ghosts (== n_ single-rank)
  std::function<double(double)> allred_;        // Allreduce hook (empty single-rank)
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_FLOW_HPP

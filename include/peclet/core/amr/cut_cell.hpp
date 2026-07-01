// transport-core — Robust-Scaled cut-cell Dirichlet operator on a BlockOctree.
//
// A faithful port of sdflow's ξ-polynomial sub-cell boundary scheme
// (sdflow/src/cut_cell_ibm.hpp: poly_*, ibmFillEntry, ibmModifyStencil) onto the
// cell-centered octree. Where the openness/aperture scheme (poisson.hpp) imposes a
// *Neumann* wall (no flux through solid faces), this imposes a *Dirichlet* value
// u = u_bc on the immersed boundary located at the true sub-cell distance ξ·h
// (Shortley–Weller): for a cut cell whose neighbour in some direction is solid,
// the 7-point stencil is modified by the boundary-distance polynomials, with
// D_rescale row-scaling for the small-cell problem. This is the velocity-diffusion
// / scalar-Dirichlet half of the cut-cell IBM (the user's "ξ-polynomial sub-cell
// BC"); the cell fluid-volume fraction κ classifies solid/fluid/cut cells.
//
// Cut cells are assumed to have same-level face neighbours (the suite contract:
// resolve the immersed boundary in a uniformly-finest band, so cut cells never sit
// on a 2:1 interface — see docs/AMR.md). 3D (sdflow's 6-direction scheme).
// Header-only, guarded by PECLET_CORE_HAVE_MORTON. Serial/host first.
#ifndef PECLET_CORE_AMR_CUT_CELL_HPP
#define PECLET_CORE_AMR_CUT_CELL_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/face_csr.hpp"  // shared host+device assembled-operator row kernels
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::amr {

// ---- boundary-distance polynomials (port of sdflow cut_cell_ibm.hpp, SCHEME 0,
//      double precision) ----
namespace cc {
// MORTON_HD (from face_csr.hpp): KOKKOS_FUNCTION on a Kokkos build, empty otherwise — so buildCutStencil
// and these polynomials are device-callable under the AMR device assembler (assembly.hpp) yet
// compile unchanged in the pure-C++ host oracle build. poly_abs replaces std::fabs (host-only under a
// CUDA device pass); it is bit-identical to std::fabs for every value buildCutStencil feeds it (the
// only difference, fabs(-0.0)=+0.0 vs −0.0, never occurs and would not change the |·|< comparisons).
MORTON_HD inline double poly_abs(double x) { return x < 0.0 ? -x : x; }
MORTON_HD inline double poly_D(double xi) { return xi * (1.0 + xi); }
MORTON_HD inline double poly_N_nb(double xi) { return xi * (1.0 - xi); }
MORTON_HD inline double poly_Nc(double xi) { return 2.0 * (xi * xi - 1.0); }
MORTON_HD inline double poly_Nbc(double) { return 2.0; }
MORTON_HD inline double poly_D_sandwich(double xm, double xp) { return xm * xp; }
MORTON_HD inline double poly_N_c_sandwich(double xm, double xp) { return (xm + 1.0) * (xp - 1.0); }
MORTON_HD inline double poly_Nbc_pp_sw(double xm, double xp) { return (xm / (xm + xp)) * (1.0 + xm); }
MORTON_HD inline double poly_Nbc_mp_sw(double xm, double xp) { return (xp / (xm + xp)) * (1.0 - xp); }
}  // namespace cc

template <unsigned Bits = 21u>
class AmrCutCell {
 public:
  static constexpr int Dim = 3;
  using Octree = BlockOctree<3, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;
  using Coord = typename Octree::Coord;

  // direction k: 0=+x,1=-x,2=+y,3=-y,4=+z,5=-z (sdflow order); OPP swaps sides.
  static constexpr int OPP[6] = {1, 0, 3, 2, 5, 4};

  void init(const Octree& t, Real h0, Vec<3> origin = Vec<3>{}) {
    t_ = &t;
    h0_ = h0;
    origin_ = origin;
    for (int d = 0; d < 3; ++d) fineExt_[d] = static_cast<Coord>(t.brick()[d] * (Index(1) << t.lmax()));
  }

  Index numLeaves() const { return t_->numLeaves(); }
  bool isFluid(Index i) const { return fluid_[static_cast<std::size_t>(i)]; }
  /// True for a cut cell: a fluid cell with at least one solid face neighbour (the row-scaled
  /// ξ-overlay band). Its residual is D_rescale-scaled / inconsistent, so it is excluded from the
  /// velocity-MG coarse defect (the clean-fluid mask — see VelocityMG).
  bool isCut(Index i) const { return cut_[static_cast<std::size_t>(i)] != 0; }
  /// Periodic face neighbour of leaf i in direction k (0=+x,1=-x,2=+y,3=-y,4=+z,5=-z).
  Index neighborOf(Index i, int k) const { return nb_[static_cast<std::size_t>(i) * 6 + k]; }
  double kappa(Index i) const { return kappa_[static_cast<std::size_t>(i)]; }
  double rhsScale(Index i) const { return rscale_[static_cast<std::size_t>(i)]; }

  // ---- read-only views of the built geometry, for the device assembler (device_momentum_assembly.hpp)
  // to stage to the device and reproduce build()/assembleOperator there. Mirrors how AmrPoisson exposes
  // its openness to the device FV assembler.
  const AmrPoisson<3, Bits>& lap() const { return lap_; }        ///< α=1 C/F ∇² geometry for regular cells
  const std::vector<double>& sdfCRaw() const { return sdfC_; }   ///< per-cell SDF sample (build Pass 1)
  const std::vector<Index>& nbRaw() const { return nb_; }        ///< n·6 periodic face-neighbour indices
  const std::vector<char>& fluidRaw() const { return fluid_; }
  const std::vector<char>& cutRaw() const { return cut_; }
  const std::vector<double>& acRaw() const { return AC_; }
  const std::vector<double>& offRaw() const { return off_; }     ///< n·6 ξ-overlay off-diagonals
  const std::vector<double>& rscaleRaw() const { return rscale_; }
  double idiag() const { return idiag_; }
  double mu() const { return mu_; }
  double beta() const { return mu_ / (h0_ * h0_); }              ///< buildCutStencil's β (= mu_/h0²)
  bool hasAdv() const { return hasAdv_; }
  const std::vector<double>& advDiagRaw() const { return advDiag_; }
  const std::vector<double>& advCoefRaw() const { return advCoef_; }
  const std::vector<Index>& advStartRaw() const { return advStart_; }
  const std::vector<Index>& advNbrRaw() const { return advNbr_; }

  /// Build the cut-cell stencils from an SDF callable sdfFn(worldPoint) (>0 fluid,
  /// <0 solid). Operator A = idiag*I - beta*Laplacian (grid units, dx=1). `nsub`
  /// is the per-axis subsampling for the volume fraction κ.
  template <class SdfFn>
  void build(SdfFn&& sdfFn, double idiag = 0.0, double beta = 1.0, int nsub = 4) {
    const Index n = numLeaves();
    // C/F-aware Laplacian provider for regular (non-cut) fluid cells: an AmrPoisson
    // with NO openness (α=1) -> the plain ∇² with 2:1-interface coeff(si,sj). The
    // cut cells (finest, same-level) keep the ξ overlay below.
    lap_.init(*t_, h0_);
    idiag_ = idiag;
    mu_ = beta * h0_ * h0_;  // physical μ (operator A = idiag·I − μ∇²)
    sdfC_.assign(static_cast<std::size_t>(n), 0.0);
    kappa_.assign(static_cast<std::size_t>(n), 0.0);
    fluid_.assign(static_cast<std::size_t>(n), false);
    cut_.assign(static_cast<std::size_t>(n), 0);
    hasAdv_ = false;  // advection FOU is rebuilt per step via buildAdvectionFou
    AC_.assign(static_cast<std::size_t>(n), 1.0);
    rscale_.assign(static_cast<std::size_t>(n), 1.0);
    inhom_.assign(static_cast<std::size_t>(n), 0.0);
    off_.assign(static_cast<std::size_t>(n) * 6, 0.0);
    nb_.assign(static_cast<std::size_t>(n) * 6, -1);

    // Pass 1: cell-centre SDF, κ (subsampled), fluid flag, neighbour indices.
    for (Index i = 0; i < n; ++i) {
      Vec<3> c = cellCenter(i);
      double sc = sdfFn(c);
      sdfC_[static_cast<std::size_t>(i)] = sc;
      fluid_[static_cast<std::size_t>(i)] = sc > 0.0;
      kappa_[static_cast<std::size_t>(i)] = volumeFraction(i, sdfFn, nsub);
      for (int k = 0; k < 6; ++k) nb_[static_cast<std::size_t>(i) * 6 + k] = neighbor(i, k);
    }

    // Pass 2: build per-leaf stencil.
    const double AC0 = idiag + 6.0 * beta;
    for (Index i = 0; i < n; ++i) {
      if (!fluid_[static_cast<std::size_t>(i)]) {  // solid: identity row u=0
        AC_[static_cast<std::size_t>(i)] = 1.0;
        for (int k = 0; k < 6; ++k) off_[static_cast<std::size_t>(i) * 6 + k] = 0.0;
        continue;
      }
      double sdf_n[6];
      bool anyGhost = false;
      for (int k = 0; k < 6; ++k) {
        Index j = nb_[static_cast<std::size_t>(i) * 6 + k];
        sdf_n[k] = (j >= 0) ? sdfC_[static_cast<std::size_t>(j)] : -1.0;  // missing => solid
        if (sdf_n[k] < 0.0) anyGhost = true;
      }
      cut_[static_cast<std::size_t>(i)] = anyGhost ? 1 : 0;
      double AC = AC0, off[6];
      for (int k = 0; k < 6; ++k) off[k] = -beta;
      double rscale = 1.0, inhomCoef = 0.0;
      if (anyGhost)
        buildCutStencil(sdfC_[static_cast<std::size_t>(i)], sdf_n, beta, AC0, AC, off, rscale, inhomCoef);
      AC_[static_cast<std::size_t>(i)] = AC;
      for (int k = 0; k < 6; ++k) off_[static_cast<std::size_t>(i) * 6 + k] = off[k];
      rscale_[static_cast<std::size_t>(i)] = rscale;
      inhom_[static_cast<std::size_t>(i)] = inhomCoef;
    }
  }

  /// The assembled linear operator A as a per-cell diagonal + face CSR:
  ///   (A u)_i = diag[i]·u_i + Σ_{k∈[start[i],start[i+1])} coef[k]·u[nbr[k]].
  /// Reproduces applyOp exactly (same coefficients): identity on solid rows; the C/F-aware
  /// idiag·I − μ∇² on regular fluid cells; the ξ-overlay 6-stencil on cut cells; plus the
  /// implicit-FOU advection coupling when buildAdvectionFou has been called. This is the
  /// device-portable form of the host operator — upload it once and run a parallel
  /// smoother / Krylov over it (momentum.hpp), instead of the serial gaussSeidel.
  struct Assembled {
    std::vector<double> diag;   ///< size n
    std::vector<Index> start;   ///< CSR row offsets, size n+1
    std::vector<Index> nbr;     ///< neighbour leaf per off-diagonal, size nnz
    std::vector<double> coef;   ///< off-diagonal coefficient, size nnz
  };
  /// `scaleAdvByRscale` (default false ⇒ reproduces applyOp/gaussSeidel exactly, for the matvec
  /// test): when true, the implicit-FOU advection is multiplied by the cut-cell D_rescale row
  /// scale, so the *entire* cut-cell row (base + advection) is scaled consistently. applyOp
  /// scales the base + RHS but not the advection — fine for the host serial GS (which gives a
  /// bounded approximation) but the inconsistency leaves a (1−rscale)/rscale·FOU amplification in
  /// the *exact* solution that blows up an accurate device solve at thin cut cells. With the
  /// advection scaled, the row is the unscaled equation × rscale and the implicit FOU cancels the
  /// (rscale-scaled) explicit deferred-correction FOU at steady state. The collocated device flow
  /// uses this; regular fluid cells (rscale=1) are unchanged.
  Assembled assembleOperator(bool scaleAdvByRscale = false) const {
    const Index n = numLeaves();
    Assembled A;
    A.diag.assign(static_cast<std::size_t>(n), 0.0);
    A.start.assign(static_cast<std::size_t>(n) + 1, 0);
    std::vector<std::vector<std::pair<Index, double>>> rows(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
      const std::size_t s = static_cast<std::size_t>(i);
      if (!fluid_[s]) {  // solid: identity row (u = u_bc)
        A.diag[s] = 1.0;
        continue;
      }
      if (cut_[s]) {  // ξ-overlay stencil (same-level neighbours)
        A.diag[s] = AC_[s];
        for (int k = 0; k < 6; ++k) {
          double a = off_[s * 6 + k];
          if (a == 0.0) continue;
          Index j = nb_[s * 6 + k];
          if (j >= 0) rows[s].emplace_back(j, a);
        }
      } else {  // regular fluid: idiag·I − μ∇² with C/F-aware face coupling
        const double invV = 1.0 / lap_.cellVolume(i);
        double dsum = 0.0;
        lap_.forEachFaceNeighbor(i, [&](Index j, Real c, int, double a) {
          rows[s].emplace_back(j, -mu_ * invV * (a * c));
          dsum += a * c;
        });
        A.diag[s] = idiag_ + mu_ * invV * dsum;
      }
      if (hasAdv_) {  // implicit-FOU advection: diagonal (outflow) + CSR (inflow)
        const double as = scaleAdvByRscale ? rscale_[s] : 1.0;
        A.diag[s] += as * advDiag_[s];
        for (std::size_t p = static_cast<std::size_t>(advStart_[s]);
             p < static_cast<std::size_t>(advStart_[static_cast<std::size_t>(i) + 1]); ++p)
          rows[s].emplace_back(advNbr_[p], as * advCoef_[p]);
      }
    }
    for (Index i = 0; i < n; ++i)
      A.start[static_cast<std::size_t>(i) + 1] =
          A.start[static_cast<std::size_t>(i)] + static_cast<Index>(rows[static_cast<std::size_t>(i)].size());
    const Index nnz = A.start[static_cast<std::size_t>(n)];
    A.nbr.resize(static_cast<std::size_t>(nnz));
    A.coef.resize(static_cast<std::size_t>(nnz));
    for (Index i = 0; i < n; ++i) {
      Index k = A.start[static_cast<std::size_t>(i)];
      for (auto& e : rows[static_cast<std::size_t>(i)]) {
        A.nbr[static_cast<std::size_t>(k)] = e.first;
        A.coef[static_cast<std::size_t>(k)] = e.second;
        ++k;
      }
    }
    return A;
  }

  // ---- Runtime operator: the assembled CSR applied with the SHARED face_csr.hpp row kernels — the
  // exact same arithmetic the device runs (momentum.hpp), executed serially here. assembleOperator
  // folds the implicit-FOU advection into the single CSR, so the FaceCsrOpT view sets hasAdv=false. The
  // *Geometric variants above are the independent reference the oracle tests check this against.

  /// View a host Assembled as a backend-agnostic FaceCsrOpT for the shared row kernels.
  FaceCsrOpT<HostArr<double>, HostArr<Index>> hostOp(const Assembled& A) const {
    FaceCsrOpT<HostArr<double>, HostArr<Index>> v;
    v.n = numLeaves();
    v.diag = HostArr<double>(A.diag.data());
    v.coef = HostArr<double>(A.coef.data());
    v.start = HostArr<Index>(A.start.data());
    v.nbr = HostArr<Index>(A.nbr.data());
    v.hasAdv = false;  // advection already folded into the single CSR by assembleOperator
    return v;
  }

  /// out = A u, via the shared kernel over the assembled CSR (== device applyMom arithmetic).
  void applyOp(const std::vector<double>& u, std::vector<double>& out) const {
    const Assembled A = assembleOperator();
    const auto op = hostOp(A);
    const Index n = numLeaves();
    out.assign(static_cast<std::size_t>(n), 0.0);
    const HostArr<double> uacc(u.data());
    for (Index i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = faceCsrApplyRow(op, i, uacc);
  }

  double residual(const std::vector<double>& u, const std::vector<double>& b,
                  std::vector<double>& res) const {
    const Assembled A = assembleOperator();
    const auto op = hostOp(A);
    const Index n = numLeaves();
    res.assign(static_cast<std::size_t>(n), 0.0);
    const HostArr<double> uacc(u.data());
    double s = 0.0;
    for (Index i = 0; i < n; ++i) {
      double r = b[static_cast<std::size_t>(i)] - faceCsrApplyRow(op, i, uacc);
      res[static_cast<std::size_t>(i)] = r;
      if (fluid_[static_cast<std::size_t>(i)]) s += r * r;
    }
    return std::sqrt(s);
  }

  /// `sweeps` true serial Gauss–Seidel sweeps (ω=1, in place) over the assembled CSR using the shared
  /// point-update kernel — the host counterpart of the device multicolour GS, same per-cell formula.
  void gaussSeidel(std::vector<double>& u, const std::vector<double>& b, int sweeps) const {
    const Assembled A = assembleOperator();
    const auto op = hostOp(A);
    const Index n = numLeaves();
    const HostArr<double> uacc(u.data());
    for (int s = 0; s < sweeps; ++s)
      for (Index i = 0; i < n; ++i) {
        double off, d;
        faceCsrOffDiag(op, i, uacc, off, d);
        u[static_cast<std::size_t>(i)] =
            faceCsrPointUpdate(b[static_cast<std::size_t>(i)], off, d, u[static_cast<std::size_t>(i)], 1.0);
      }
  }

  /// Geometric operator apply (walks the octree live) — the INDEPENDENT reference encoding, kept as
  /// the test oracle for the assembled CSR (and hence the device kernels). The runtime `applyOp`
  /// below routes through the shared face_csr.hpp kernels instead, so host and device run identical
  /// arithmetic; `test_amr_cut_cell` asserts the two agree.
  /// out = A u. A = idiag·I − μ∇² (C/F-aware) on regular fluid cells; the ξ-overlay
  /// stencil on cut cells (finest, same-level); identity on solid cells (held at u_bc).
  void applyOpGeometric(const std::vector<double>& u, std::vector<double>& out) const {
    const Index n = numLeaves();
    out.assign(static_cast<std::size_t>(n), 0.0);
    std::vector<double> Lu;
    lap_.applyLaplacian(u, Lu);  // C/F-aware ∇² (α=1); used only for regular fluid cells
    for (Index i = 0; i < n; ++i) {
      const std::size_t s = static_cast<std::size_t>(i);
      if (!fluid_[s]) {
        out[s] = u[s];
      } else if (cut_[s]) {
        double acc = AC_[s] * u[s];
        for (int k = 0; k < 6; ++k) {
          double a = off_[s * 6 + k];
          if (a == 0.0) continue;
          Index j = nb_[s * 6 + k];
          if (j >= 0) acc += a * u[static_cast<std::size_t>(j)];
        }
        out[s] = acc;
      } else {
        out[s] = idiag_ * u[s] - mu_ * Lu[s];  // regular fluid (C/F-consistent)
      }
      if (hasAdv_ && fluid_[s]) out[s] += advApply(i, u);  // implicit FOU advection
    }
  }

  /// Build the implicit first-order-upwind advection operator from a (lagged)
  /// advecting velocity field `uadv` (3 components, cell-centred), scaled by `rho`.
  /// **C/F-conservative**: each face (incl. the 2^(Dim-1) fine sub-faces of a coarse
  /// cell, via forEachFaceFull) contributes `(1/V)·A·velOut·ρ` to the diagonal
  /// (outflow) or to a CSR off-diagonal toward the upstream neighbour (inflow); the
  /// advecting velocity at a wall (solid neighbour) face is zero (no flow through the
  /// immersed boundary). Rebuilt per step. Stable base of the deferred correction.
  /// `uf`/`faceStart` (+axis face velocity per forEachFaceFull (sub)face, when `useFace`) is the
  /// divergence-free advecting velocity; otherwise the cell average ½(uadv_i+uadv_j) is used (before
  /// the first projection has built uf). The implicit FOU and the explicit deferred correction must use
  /// the SAME velocity, hence it lives here too — not only in the high-order term.
  void buildAdvectionFou(const std::array<std::vector<double>, 3>& uadv, double rho,
                         const std::vector<double>& uf, const std::vector<Index>& faceStart,
                         bool useFace) {
    const Index n = numLeaves();
    advDiag_.assign(static_cast<std::size_t>(n), 0.0);
    advStart_.assign(static_cast<std::size_t>(n) + 1, 0);
    hasAdv_ = true;
    auto velOutOf = [&](Index i, Index j, int axis, int dir, Index slot) {
      return useFace ? dir * uf[static_cast<std::size_t>(slot)]
                     : dir * 0.5 * (uadv[axis][static_cast<std::size_t>(i)] + uadv[axis][static_cast<std::size_t>(j)]);
    };
    // pass 1: count inflow (off-diagonal) fluid faces per cell.
    for (Index i = 0; i < n; ++i) {
      if (!fluid_[static_cast<std::size_t>(i)]) continue;
      int cnt = 0;
      Index s = useFace ? faceStart[static_cast<std::size_t>(i)] : 0;
      lap_.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double, double) {
        if (fluid_[static_cast<std::size_t>(j)] && velOutOf(i, j, axis, dir, s) < 0.0) ++cnt;
        ++s;
      });
      advStart_[static_cast<std::size_t>(i) + 1] = cnt;
    }
    for (Index i = 0; i < n; ++i) advStart_[static_cast<std::size_t>(i) + 1] += advStart_[static_cast<std::size_t>(i)];
    advNbr_.assign(advStart_[static_cast<std::size_t>(n)], -1);
    advCoef_.assign(advStart_[static_cast<std::size_t>(n)], 0.0);
    // pass 2: fill diagonal (outflow) + CSR off-diagonals (inflow).
    for (Index i = 0; i < n; ++i) {
      if (!fluid_[static_cast<std::size_t>(i)]) continue;
      const double Vi = lap_.cellVolume(i);
      std::size_t pos = static_cast<std::size_t>(advStart_[static_cast<std::size_t>(i)]);
      Index s = useFace ? faceStart[static_cast<std::size_t>(i)] : 0;
      lap_.forEachFaceFull(i, [&](Index j, int axis, int dir, double area, double, double) {
        const Index slot = s++;
        if (!fluid_[static_cast<std::size_t>(j)]) return;
        double velOut = velOutOf(i, j, axis, dir, slot);
        double w = rho * area * velOut / Vi;
        if (velOut < 0.0) {       // inflow → couple to upstream neighbour j (matches pass 1)
          advNbr_[pos] = j;
          advCoef_[pos] = w;
          ++pos;
        } else {
          advDiag_[static_cast<std::size_t>(i)] += w;  // outflow → diagonal
        }
      });
    }
  }

  /// Effective RHS for source `src` (≈ -h^2 f at cell centres) and wall value u_bc:
  /// row-scaled by D_rescale and shifted by the inhomogeneous boundary term.
  std::vector<double> makeRhs(const std::vector<double>& src, double u_bc) const {
    const Index n = numLeaves();
    std::vector<double> b(static_cast<std::size_t>(n), 0.0);
    for (Index i = 0; i < n; ++i) {
      if (!fluid_[static_cast<std::size_t>(i)]) {
        b[static_cast<std::size_t>(i)] = u_bc;  // solid held at u_bc
        continue;
      }
      b[static_cast<std::size_t>(i)] = src[static_cast<std::size_t>(i)] * rscale_[static_cast<std::size_t>(i)] +
                                       inhom_[static_cast<std::size_t>(i)] * u_bc;
    }
    return b;
  }

  double residualGeometric(const std::vector<double>& u, const std::vector<double>& b,
                           std::vector<double>& res) const {
    applyOpGeometric(u, res);
    double s = 0.0;
    const Index n = numLeaves();
    for (Index i = 0; i < n; ++i) {
      double r = b[static_cast<std::size_t>(i)] - res[static_cast<std::size_t>(i)];
      res[static_cast<std::size_t>(i)] = r;
      if (fluid_[static_cast<std::size_t>(i)]) s += r * r;
    }
    return std::sqrt(s);
  }

  void gaussSeidelGeometric(std::vector<double>& u, const std::vector<double>& b, int sweeps) const {
    const Index n = numLeaves();
    for (int s = 0; s < sweeps; ++s)
      for (Index i = 0; i < n; ++i) {
        const std::size_t si = static_cast<std::size_t>(i);
        if (!fluid_[si]) {
          u[si] = b[si];
        } else if (cut_[si]) {  // ξ-overlay stencil (same-level neighbours)
          double sum = b[si];
          for (int k = 0; k < 6; ++k) {
            double a = off_[si * 6 + k];
            if (a == 0.0) continue;
            Index j = nb_[si * 6 + k];
            if (j >= 0) sum -= a * u[static_cast<std::size_t>(j)];
          }
          double d = AC_[si];
          if (hasAdv_) { sum -= advOffSum(i, u); d += advDiag_[si]; }
          if (d != 0.0) u[si] = sum / d;
        } else {  // regular fluid: idiag·u − μ∇² with C/F-aware face coupling
          double offsum = 0.0, dsum = 0.0;
          lap_.forEachFaceNeighbor(i, [&](Index j, Real c, int, double) {
            offsum += c * u[static_cast<std::size_t>(j)];
            dsum += c;
          });
          double Vi = lap_.cellVolume(i);
          double diagA = idiag_ + mu_ * dsum / Vi;
          double sum = b[si] + mu_ * offsum / Vi;
          if (hasAdv_) { sum -= advOffSum(i, u); diagA += advDiag_[si]; }
          u[si] = sum / diagA;
        }
      }
  }

 public:
  /// The implicit-FOU advection operator applied to `field` at leaf i:
  /// advDiag·field_i + Σ_csr coef·field_nbr (= ρ·∇·(u_adv field) FOU). Used by the
  /// flow solver for the explicit deferred-correction term (high-order − this).
  double fouApply(Index i, const std::vector<double>& field) const {
    return advDiag_[static_cast<std::size_t>(i)] * field[static_cast<std::size_t>(i)] +
           advOffSum(i, field);
  }
  bool hasAdvection() const { return hasAdv_; }

 private:
  // Σ_csr coef·field[nbr] — the implicit-FOU advection off-diagonal coupling.
  double advOffSum(Index i, const std::vector<double>& field) const {
    double s = 0.0;
    for (std::size_t p = static_cast<std::size_t>(advStart_[static_cast<std::size_t>(i)]);
         p < static_cast<std::size_t>(advStart_[static_cast<std::size_t>(i) + 1]); ++p)
      s += advCoef_[p] * field[static_cast<std::size_t>(advNbr_[p])];
    return s;
  }
  double advApply(Index i, const std::vector<double>& u) const { return fouApply(i, u); }

 public:
  // Port of ibmFillEntry<0> + ibmModifyStencil for one cut cell (Dirichlet). Public + MORTON_HD so the
  // device assembler (device_momentum_assembly.hpp) runs the SAME per-cell stencil build on device.
  MORTON_HD static void buildCutStencil(double sdf_c, const double sdf_n[6], double beta, double AC0,
                                        double& ACout, double off[6], double& rscaleOut,
                                        double& inhomOut) {
    bool ghost[6];
    double xi[6], D[6];
    for (int k = 0; k < 6; ++k) {
      if (sdf_n[k] < 0.0) {
        ghost[k] = true;
        double th = sdf_c / (sdf_c - sdf_n[k]);
        th = th < 1e-4 ? 1e-4 : (th > 1.0 ? 1.0 : th);
        xi[k] = th;
        D[k] = cc::poly_D(th);
      } else {
        ghost[k] = false;
        xi[k] = 1.0;
        D[k] = 1e9;
      }
    }
    bool sand[3] = {ghost[0] && ghost[1], ghost[2] && ghost[3], ghost[4] && ghost[5]};
    double Dsand[3] = {0, 0, 0};
    for (int a = 0; a < 3; ++a)
      if (sand[a]) Dsand[a] = cc::poly_D_sandwich(xi[2 * a + 1], xi[2 * a]);
    double minAbs = 1e30, descale = 1.0;
    auto upd = [&](double v) { if (cc::poly_abs(v) < minAbs) { minAbs = cc::poly_abs(v); descale = v; } };
    for (int a = 0; a < 3; ++a) {
      if (sand[a])
        upd(Dsand[a]);
      else {
        if (ghost[2 * a]) upd(D[2 * a]);
        if (ghost[2 * a + 1]) upd(D[2 * a + 1]);
      }
    }
    double K[6] = {0}, Mf[6] = {1, 1, 1, 1, 1, 1}, X[6] = {0}, Nbc[6] = {0}, R[6] = {1, 1, 1, 1, 1, 1};
    for (int a = 0; a < 3; ++a) {
      int km = 2 * a + 1, kp = 2 * a;
      double Daxis = sand[a] ? Dsand[a] : (ghost[kp] ? D[kp] : (ghost[km] ? D[km] : descale));
      double r = descale / Daxis;
      if (cc::poly_abs(Daxis) < 1e-9) r = 1.0;
      R[kp] = R[km] = r;
      if (sand[a]) {
        K[kp] = cc::poly_N_c_sandwich(xi[km], xi[kp]) * r;
        K[km] = cc::poly_N_c_sandwich(xi[kp], xi[km]) * r;
        Nbc[kp] = (cc::poly_Nbc_pp_sw(xi[km], xi[kp]) + cc::poly_Nbc_mp_sw(xi[km], xi[kp])) * r;
        Nbc[km] = (cc::poly_Nbc_pp_sw(xi[kp], xi[km]) + cc::poly_Nbc_mp_sw(xi[kp], xi[km])) * r;
        Mf[kp] = Mf[km] = 0.0;
      } else {
        for (int side = 0; side < 2; ++side) {
          int kk = side == 0 ? kp : km;
          if (ghost[kk]) {
            K[kk] = cc::poly_Nc(xi[kk]) * r;
            X[kk] = cc::poly_N_nb(xi[kk]) * r;
            Nbc[kk] = cc::poly_Nbc(xi[kk]) * r;
            Mf[kk] = 0.0;
          } else {
            K[kk] = 0.0; Mf[kk] = 1.0; X[kk] = 0.0; Nbc[kk] = 0.0;
          }
        }
      }
    }
    // ibmModifyStencil (orig off-diagonal = -beta for every direction). OPP is a function-local
    // constexpr (not the static member) so the runtime index mod[OPP[k]] is device-safe under CUDA.
    constexpr int OPP_[6] = {1, 0, 3, 2, 5, 4};
    double aC = AC0 * descale, mod[6] = {0, 0, 0, 0, 0, 0}, inhom = 0.0;
    for (int k = 0; k < 6; ++k) {
      double vnb = -beta;
      aC += vnb * K[k];
      inhom += Nbc[k] * vnb;
      mod[k] += vnb * (descale * Mf[k] - 1.0);
      mod[OPP_[k]] += vnb * X[k];
    }
    ACout = aC;
    for (int k = 0; k < 6; ++k) off[k] = -beta + mod[k];
    rscaleOut = descale;
    inhomOut = inhom;
  }

 private:
  Vec<3> cellCenter(Index i) const {
    auto b = t_->bounds(i);
    double s = static_cast<double>(Index(1) << t_->level(i));
    Vec<3> c{};
    for (int d = 0; d < 3; ++d) c[d] = origin_[d] + (static_cast<double>(b[0][d]) + 0.5 * s) * h0_;
    return c;
  }

  template <class SdfFn>
  double volumeFraction(Index i, SdfFn&& sdfFn, int nsub) const {
    auto b = t_->bounds(i);
    double s = static_cast<double>(Index(1) << t_->level(i));
    double w = s * h0_;
    int inside = 0, total = nsub * nsub * nsub;
    for (int a = 0; a < nsub; ++a)
      for (int bb = 0; bb < nsub; ++bb)
        for (int cc2 = 0; cc2 < nsub; ++cc2) {
          Vec<3> p{origin_[0] + static_cast<double>(b[0][0]) * h0_ + (a + 0.5) / nsub * w,
                   origin_[1] + static_cast<double>(b[0][1]) * h0_ + (bb + 0.5) / nsub * w,
                   origin_[2] + static_cast<double>(b[0][2]) * h0_ + (cc2 + 0.5) / nsub * w};
          if (sdfFn(p) > 0.0) ++inside;
        }
    return static_cast<double>(inside) / total;
  }

  Index neighbor(Index i, int k) const {
    int axis = k / 2, dir = (k % 2 == 0) ? +1 : -1;
    auto b = t_->bounds(i);
    const auto& lo = b[0];
    Coord si = Coord(Coord(1) << t_->level(i));
    long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si) : static_cast<long>(lo[axis]) - 1;
    long e = static_cast<long>(fineExt_[axis]);
    std::array<Coord, 3> p = lo;
    p[axis] = static_cast<Coord>(((pc % e) + e) % e);
    return t_->find(M::encode(p).code());
  }

  const Octree* t_ = nullptr;
  Real h0_ = 1.0;
  Vec<3> origin_{};
  std::array<Coord, 3> fineExt_{};
  std::vector<double> sdfC_, kappa_, AC_, rscale_, inhom_, off_;
  std::vector<Index> nb_;
  std::vector<char> fluid_, cut_;
  AmrPoisson<3, Bits> lap_;   // C/F-aware ∇² provider for regular fluid cells (α=1)
  double idiag_ = 0.0, mu_ = 1.0;
  std::vector<double> advDiag_, advCoef_;     // implicit-FOU advection (rebuilt per step)
  std::vector<Index> advStart_, advNbr_;      // CSR off-diagonals (C/F-conservative)
  bool hasAdv_ = false;
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_CUT_CELL_HPP

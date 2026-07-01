// transport-core — device (Kokkos) momentum operator + smoother/solver for the AMR
// collocated flow step.
//
// The cut-cell momentum operator A = (ρ/dt)I − μ∇² (+ implicit-FOU advection + the
// ξ-polynomial Dirichlet overlay on cut cells) assembled by AmrCutCell::assembleOperator
// as a per-cell diagonal + general face CSR, uploaded once and applied / solved entirely
// in device kernels. This replaces the host-serial AmrCutCell::gaussSeidel that the drag
// study found to be the bottleneck (the pressure Poisson was already on the device path).
//
// A is generally NON-symmetric (the cut-cell D_rescale row scaling and the upwind
// advection break symmetry), so the smoother is weighted Jacobi (parallel, deterministic:
// reads only the previous iterate) and the Krylov accelerator is BiCGStab (handles
// non-symmetric A) rather than CG. For moderate dt the reaction diagonal (ρ/dt) makes A
// strongly diagonally dominant ⇒ Jacobi alone converges fast; BiCGStab covers the
// large-dt / steady regime where the operator approaches the (ill-conditioned) viscous
// Laplacian.
//
// Validation is by convergence + agreement with the host operator to tolerance (the GPU
// matvec differs from host in the last bit by FMA contraction). Requires a Kokkos build +
// the morton checkout (PECLET_CORE_HAVE_MORTON).
#ifndef PECLET_CORE_AMR_MOMENTUM_HPP
#define PECLET_CORE_AMR_MOMENTUM_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <cmath>
#include <functional>
#include <map>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/multigrid.hpp"  // restrictField / prolongAdd transfer kernels
#include "peclet/core/amr/pcg.hpp"        // dotPlain-style primitives: axpy, zpby, negate
#include "peclet/core/amr/face_csr.hpp"          // shared host+device assembled-operator row kernels
#include "peclet/core/common/view.hpp"

namespace peclet::core::amr {

/// Assembled momentum operator on the device: (A u)_i = diag_i u_i + Σ coef·u[nbr], with an
/// optional implicit-FOU advection part (rebuilt each step from the lagged velocity): a
/// per-cell outflow diagonal `advDiag` + per-face inflow coefficients over a second
/// (face-geometry) CSR. hasAdv=false ⇒ the pure cut-cell operator, bit-exact unchanged.
struct MomentumOp {
  View<double> diag;      ///< size n
  View<Index> faceStart;  ///< CSR row offsets, size n+1
  View<Index> faceNbr;    ///< neighbour leaf per off-diagonal, size nnz
  View<double> faceCoef;  ///< off-diagonal coefficient, size nnz
  Index n = 0;
  // Optional implicit-FOU advection (over the face-geometry CSR):
  bool hasAdv = false;
  View<double> advDiag;   ///< per-cell outflow (diagonal) advection weight, size n
  View<Index> advStart;   ///< face-geom CSR row offsets, size n+1
  View<Index> advNbr;     ///< face-geom neighbour per face, size nFaces
  View<double> advCoef;   ///< per-face inflow advection coefficient (0 on outflow/solid faces)
};

/// View the assembled momentum operator through the shared, backend-agnostic FaceCsrOpT, so the
/// device kernels and the host serial solver (cut_cell.hpp) run the *same* row arithmetic
/// (face_csr.hpp) and cannot drift. Non-const Views convert to their const accessor form implicitly;
/// the advection arrays are empty (and untouched) when hasAdv is false.
inline FaceCsrOpT<View<const double>, View<const Index>> momView(const MomentumOp& op) {
  FaceCsrOpT<View<const double>, View<const Index>> v;
  v.n = op.n;
  v.diag = op.diag;
  v.coef = op.faceCoef;
  v.start = op.faceStart;
  v.nbr = op.faceNbr;
  v.hasAdv = op.hasAdv;
  v.advDiag = op.advDiag;
  v.advCoef = op.advCoef;
  v.advStart = op.advStart;
  v.advNbr = op.advNbr;
  return v;
}

/// Au = A u (cut-cell operator + optional implicit-FOU advection).
inline void applyMom(const MomentumOp& op, View<const double> u, View<double> Au) {
  const auto A = momView(op);
  Kokkos::parallel_for(
      "amr::mom_apply", op.n,
      KOKKOS_LAMBDA(const Index i) { Au(i) = faceCsrApplyRow(A, i, u); });
}

/// res = b − A u.
inline void residualMom(const MomentumOp& op, View<const double> u,
                              View<const double> b, View<double> res) {
  const auto A = momView(op);
  Kokkos::parallel_for(
      "amr::mom_residual", op.n,
      KOKKOS_LAMBDA(const Index i) { res(i) = b(i) - faceCsrApplyRow(A, i, u); });
}

/// One weighted-Jacobi sweep of A u = b (in place). `tmp` is scratch (size n). Pass 1
/// reads only the previous iterate, pass 2 updates ⇒ order-independent / deterministic.
inline void jacobiMom(const MomentumOp& op, View<double> u, View<const double> b,
                            View<double> tmp, double omega) {
  const auto A = momView(op);
  Kokkos::parallel_for(
      "amr::mom_jacobi_compute", op.n, KOKKOS_LAMBDA(const Index i) {
        double off, d;
        faceCsrOffDiag(A, i, u, off, d);
        tmp(i) = (d != 0.0) ? (b(i) - off) / d : u(i);
      });
  Kokkos::parallel_for(
      "amr::mom_jacobi_update", op.n,
      KOKKOS_LAMBDA(const Index i) { u(i) = (1.0 - omega) * u(i) + omega * tmp(i); });
}

/// Plain (unweighted) dot product.
inline double dotPlain(View<const double> a, View<const double> b, Index n) {
  double s = 0.0;
  Kokkos::parallel_reduce(
      "amr::mom_dot", n, KOKKOS_LAMBDA(const Index i, double& acc) { acc += a(i) * b(i); }, s);
  return s;
}

/// BiCGStab direction update: p = r + β(p − ω v). (Free function — an extended
/// __host__ __device__ lambda may not live in a private/protected member function.)
inline void bicgPUpdate(View<double> p, View<const double> r, View<const double> v, double beta,
                        double omega, Index n) {
  Kokkos::parallel_for(
      "amr::mom_pupdate", n,
      KOKKOS_LAMBDA(const Index i) { p(i) = r(i) + beta * (p(i) - omega * v(i)); });
}

// ===========================================================================
// Multicolour Gauss–Seidel smoother + graph colouring (the RB-GS mirror of sdflow's
// ibmRbgsStencilColor on the AMR). On a 2:1-graded octree (or a Voronoi-cell mesh) the
// face-adjacency graph needs a general greedy colouring (~6–8 colours); a colour is a set of
// cells with no shared face, so all of one colour update in parallel reading the already-updated
// other colours — a true GS sweep, deterministic (fixed cell order ⇒ fixed colouring). GS smooths
// ~2× better than damped Jacobi (and is the strong fine smoother that "owns the cut band" the
// rediscretized staircase velocity-MG excludes from its coarse grid). Mesh-agnostic: operates on a
// face CSR + the assembled operator, no octree types.
// ===========================================================================

/// A graph colouring of a face CSR: cells grouped by colour. `hStart` (host, size nColors+1) slices
/// `idx` (device, cells in colour order). Rebuilt when the connectivity changes (adapt / re-tess).
struct Coloring {
  std::vector<Index> hStart;
  View<Index> idx;
  int nColors = 1;
};

/// Greedy colouring of the face CSR (`start`/`nbr`, host): each cell gets the smallest colour not
/// used by any face neighbour, so cells of one colour share no edge (race-free parallel GS sweep).
/// Deterministic from the natural cell order.
///
/// The adjacency is **symmetrised** first (undirected: i conflicts with j if i∈nbr(j) OR j∈nbr(i)).
/// The assembled cut-cell operator's CSR can be structurally *asymmetric* — the ξ-polynomial Dirichlet
/// overlay adds extrapolation entries a cut cell references but its target doesn't reference back — and
/// colouring only the outgoing edges would then leave two mutually-adjacent cells the same colour, a
/// data race that makes the GS sweep a non-deterministic (inconsistent) operator and silently breaks
/// the BiCGStab it preconditions (false convergence to NaN at scale). Symmetrising is the correctness
/// guard; it costs one O(nnz) host pass at build/adapt time.
inline Coloring greedyColoring(const std::vector<Index>& start, const std::vector<Index>& nbr,
                               Index n) {
  // Build the symmetric (undirected) adjacency in CSR form.
  std::vector<Index> deg(static_cast<std::size_t>(n) + 1, 0);
  for (Index i = 0; i < n; ++i)
    for (Index k = start[static_cast<std::size_t>(i)]; k < start[static_cast<std::size_t>(i) + 1]; ++k) {
      ++deg[static_cast<std::size_t>(i) + 1];
      ++deg[static_cast<std::size_t>(nbr[static_cast<std::size_t>(k)]) + 1];
    }
  for (Index i = 0; i < n; ++i) deg[static_cast<std::size_t>(i) + 1] += deg[static_cast<std::size_t>(i)];
  std::vector<Index> aStart(deg);  // copy of the offsets
  std::vector<Index> aNbr(static_cast<std::size_t>(deg[static_cast<std::size_t>(n)]));
  std::vector<Index> acur(deg.begin(), deg.end() - 1);
  for (Index i = 0; i < n; ++i)
    for (Index k = start[static_cast<std::size_t>(i)]; k < start[static_cast<std::size_t>(i) + 1]; ++k) {
      const Index j = nbr[static_cast<std::size_t>(k)];
      aNbr[static_cast<std::size_t>(acur[static_cast<std::size_t>(i)]++)] = j;
      aNbr[static_cast<std::size_t>(acur[static_cast<std::size_t>(j)]++)] = i;
    }
  std::vector<int> color(static_cast<std::size_t>(n), -1);
  std::vector<int> stamp;  // stamp[c]==i ⇒ colour c forbidden for cell i (avoids per-cell clears)
  int nColors = 1;
  for (Index i = 0; i < n; ++i) {
    for (Index k = aStart[static_cast<std::size_t>(i)]; k < aStart[static_cast<std::size_t>(i) + 1]; ++k) {
      const int nc = color[static_cast<std::size_t>(aNbr[static_cast<std::size_t>(k)])];
      if (nc >= 0) {
        if (static_cast<std::size_t>(nc) >= stamp.size()) stamp.resize(static_cast<std::size_t>(nc) + 1, -1);
        stamp[static_cast<std::size_t>(nc)] = static_cast<int>(i);
      }
    }
    int c = 0;
    while (c < static_cast<int>(stamp.size()) && stamp[static_cast<std::size_t>(c)] == static_cast<int>(i)) ++c;
    color[static_cast<std::size_t>(i)] = c;
    if (c + 1 > nColors) nColors = c + 1;
  }
  Coloring col;
  col.nColors = nColors;
  col.hStart.assign(static_cast<std::size_t>(nColors) + 1, 0);
  for (Index i = 0; i < n; ++i) ++col.hStart[static_cast<std::size_t>(color[static_cast<std::size_t>(i)]) + 1];
  for (int c = 0; c < nColors; ++c)
    col.hStart[static_cast<std::size_t>(c) + 1] += col.hStart[static_cast<std::size_t>(c)];
  std::vector<Index> idx(static_cast<std::size_t>(n));
  std::vector<Index> cur(col.hStart.begin(), col.hStart.end() - 1);
  for (Index i = 0; i < n; ++i) {
    const int c = color[static_cast<std::size_t>(i)];
    idx[static_cast<std::size_t>(cur[static_cast<std::size_t>(c)]++)] = i;
  }
  col.idx = toDevice(idx, "gs_coloring");
  return col;
}

/// One **symmetric** multicolour Gauss–Seidel sweep of A u = b in place (momentum operator: diag +
/// face CSR + optional implicit-FOU advection): a forward pass over colours 0…C-1 followed by a
/// reverse pass C-1…0. Each colour is a parallel_for over its cells doing the GS point update reading
/// the current (already-updated) neighbours; cells of one colour share no edge ⇒ race-free.
///
/// The forward+reverse pairing makes the smoother **symmetric**, which matters when the MG V-cycle is
/// used as a *preconditioner* for BiCGStab (the momentum path): a forward-only GS V-cycle is a
/// non-symmetric, non-normal operator that breaks BiCGStab's bi-orthogonal recurrence on the larger
/// non-symmetric 64³ system (false convergence to NaN), whereas the symmetric (SGS) V-cycle keeps it
/// robust — the textbook remedy, and the behaviour sdflow gets from its RB-GS / MG-as-solver path.
inline void multicolorGSMom(const MomentumOp& op, View<double> u, View<const double> b,
                                  const Coloring& col, double omega) {
  const auto A = momView(op);
  auto idx = col.idx;
  auto colorPass = [&](int c) {
    const Index a0 = col.hStart[static_cast<std::size_t>(c)];
    const Index a1 = col.hStart[static_cast<std::size_t>(c) + 1];
    Kokkos::parallel_for(
        "amr::gs_mom", Kokkos::RangePolicy<ExecSpace>(a0, a1), KOKKOS_LAMBDA(const Index k) {
          const Index i = idx(k);
          double off, d;
          faceCsrOffDiag(A, i, u, off, d);
          u(i) = faceCsrPointUpdate(b(i), off, d, u(i), omega);
        });
  };
  for (int c = 0; c < col.nColors; ++c) colorPass(c);             // forward
  for (int c = col.nColors - 2; c >= 0; --c) colorPass(c);         // reverse (last colour not repeated)
}

// ===========================================================================
// MomentumMG — Galerkin geometric multigrid for the momentum operator.
//
// The cut-cell momentum operator carries the ξ-polynomial Dirichlet overlay and its
// D_rescale row scaling, so a *rediscretised* coarse operator (the openness-Helmholtz
// attempt) mismatches it and makes a poor preconditioner. Instead the coarse operators are
// built by **Galerkin coarsening** A_c = R·A·P of the exact assembled fine CSR: R = volume
// average over a coarse cell's children, P = piecewise-constant injection (the same transfer
// pair the pressure MG uses). This is consistent with the fine operator by construction — it
// inherits the cut-cell stencil and row scaling, and a coarse cell whose children are all
// solid (identity rows) stays an identity row (ε-solid-on-coarse emerges for free). The
// hierarchy is the uniformly-coarsened octree; the smoother is jacobiMom, the residual
// restriction / correction prolongation are the shared restrictField / prolongAdd.
// Used as the momentum BiCGStab preconditioner ⇒ the iteration count stays ~flat with N.
// ===========================================================================
template <unsigned Bits = 21u>
class MomentumMG {
 public:
  using Octree = BlockOctree<3, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;

  /// Build the Galerkin hierarchy from the finest octree + the assembled fine operator CSR
  /// (diag + face CSR, as produced by AmrCutCell::assembleOperator).
  void build(const Octree& finest, const std::vector<double>& diag0,
             const std::vector<Index>& start0, const std::vector<Index>& nbr0,
             const std::vector<double>& coef0) {
    octs_.clear();
    octs_.push_back(finest);
    for (;;) {
      Octree c = octs_.back();
      Index merged = c.coarsenIf([](Code, unsigned) { return true; });
      if (merged == 0 || c.numLeaves() == octs_.back().numLeaves()) break;
      octs_.push_back(c);
      if (c.numLeaves() == 1) break;
    }
    const std::size_t nl = octs_.size();
    levels_.clear();
    levels_.resize(nl);

    std::vector<double> hdiag = diag0, hcoef = coef0;
    std::vector<Index> hstart = start0, hnbr = nbr0;
    uploadLevel(0, hdiag, hstart, hnbr, hcoef);

    for (std::size_t L = 0; L + 1 < nl; ++L) {
      const Octree& f = octs_[L];
      const Octree& c = octs_[L + 1];
      const Index nf = f.numLeaves(), nc = c.numLeaves();
      std::vector<Index> c2p(static_cast<std::size_t>(nf));
      std::vector<Index> cnt(static_cast<std::size_t>(nc), 0);
      for (Index i = 0; i < nf; ++i) {
        Code par = M::from_code(f.code(i)).ancestor(f.level(i) + 1).code();
        Index p = c.find(par);
        c2p[static_cast<std::size_t>(i)] = p;
        if (p >= 0) ++cnt[static_cast<std::size_t>(p)];
      }
      std::vector<Index> cstart(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p)
        cstart[static_cast<std::size_t>(p) + 1] = cstart[static_cast<std::size_t>(p)] + cnt[static_cast<std::size_t>(p)];
      std::vector<Index> cidx(static_cast<std::size_t>(nf));
      std::vector<Index> cur(cstart.begin(), cstart.end() - 1);
      for (Index i = 0; i < nf; ++i) {
        Index p = c2p[static_cast<std::size_t>(i)];
        if (p >= 0) cidx[static_cast<std::size_t>(cur[static_cast<std::size_t>(p)]++)] = i;
      }
      levels_[L].c2p = toDevice(c2p, "mmg_c2p");
      levels_[L].childStart = toDevice(cstart, "mmg_cstart");
      levels_[L].childIdx = toDevice(cidx, "mmg_cidx");

      // Galerkin A_c[p][q] = (1/n_ch[p]) Σ_{i child of p} ( A[i] entries mapped to parents ).
      std::vector<std::map<Index, double>> acc(static_cast<std::size_t>(nc));
      for (Index i = 0; i < nf; ++i) {
        Index p = c2p[static_cast<std::size_t>(i)];
        if (p < 0) continue;
        const double w = 1.0 / static_cast<double>(cnt[static_cast<std::size_t>(p)]);
        acc[static_cast<std::size_t>(p)][p] += w * hdiag[static_cast<std::size_t>(i)];
        for (Index k = hstart[static_cast<std::size_t>(i)]; k < hstart[static_cast<std::size_t>(i) + 1]; ++k) {
          Index q = c2p[static_cast<std::size_t>(hnbr[static_cast<std::size_t>(k)])];
          if (q < 0) continue;
          acc[static_cast<std::size_t>(p)][q] += w * hcoef[static_cast<std::size_t>(k)];
        }
      }
      std::vector<double> cdiag(static_cast<std::size_t>(nc), 0.0);
      std::vector<Index> cs(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p) {
        int off = 0;
        for (auto& e : acc[static_cast<std::size_t>(p)]) {
          if (e.first == p)
            cdiag[static_cast<std::size_t>(p)] = e.second;
          else
            ++off;
        }
        cs[static_cast<std::size_t>(p) + 1] = cs[static_cast<std::size_t>(p)] + off;
      }
      std::vector<Index> cn(static_cast<std::size_t>(cs[static_cast<std::size_t>(nc)]));
      std::vector<double> ccoef(static_cast<std::size_t>(cs[static_cast<std::size_t>(nc)]));
      for (Index p = 0; p < nc; ++p) {
        Index k = cs[static_cast<std::size_t>(p)];
        for (auto& e : acc[static_cast<std::size_t>(p)])
          if (e.first != p) {
            cn[static_cast<std::size_t>(k)] = e.first;
            ccoef[static_cast<std::size_t>(k)] = e.second;
            ++k;
          }
      }
      uploadLevel(L + 1, cdiag, cs, cn, ccoef);
      hdiag = cdiag;
      hstart = cs;
      hnbr = cn;
      hcoef = ccoef;
    }
    for (auto& lv : levels_) {
      lv.x = View<double>("mmg_x", static_cast<std::size_t>(lv.op.n));
      lv.b = View<double>("mmg_b", static_cast<std::size_t>(lv.op.n));
      lv.res = View<double>("mmg_res", static_cast<std::size_t>(lv.op.n));
      lv.tmp = View<double>("mmg_tmp", static_cast<std::size_t>(lv.op.n));
    }
  }

  /// Opt-in: use multicolour Gauss–Seidel as the smoother (per-level colouring built at build)
  /// instead of weighted Jacobi — ~2× better smoothing, fewer V-cycles. Default off (Jacobi).
  void setGaussSeidel(bool on) { useGS_ = on; }

  /// One V-cycle on level L solving A u = b (correction scheme).
  void vcycle(int pre = 2, int post = 2, int bottom = 30, double omega = 0.7, std::size_t L = 0) {
    Level& lv = levels_[L];
    View<const double> bc(lv.b);
    if (L + 1 == levels_.size()) {
      smooth(lv, bottom, omega);
      return;
    }
    smooth(lv, pre, omega);
    residualMom(lv.op, View<const double>(lv.x), bc, lv.res);
    Level& cl = levels_[L + 1];
    restrictField(lv.childStart, lv.childIdx, View<const double>(lv.res), cl.b, cl.op.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(pre, post, bottom, omega, L + 1);
    prolongAdd(lv.c2p, View<const double>(cl.x), lv.x, lv.op.n);
    smooth(lv, post, omega);
  }

  std::size_t numLevels() const { return levels_.size(); }
  Index numLeaves(std::size_t L = 0) const { return levels_[L].op.n; }
  View<double> x(std::size_t L = 0) { return levels_[L].x; }
  View<double> b(std::size_t L = 0) { return levels_[L].b; }
  const MomentumOp& op(std::size_t L = 0) const { return levels_[L].op; }

 private:
  struct Level {
    MomentumOp op;
    View<double> x, b, res, tmp;
    View<Index> c2p, childStart, childIdx;
    Coloring col;
  };
  void smooth(Level& lv, int sweeps, double omega) {
    View<const double> bc(lv.b);
    if (useGS_)
      // Multicolour GS is stable undamped on this diagonally-dominant operator (ρ/dt + 6μ + FOU > 0);
      // the passed omega is Jacobi's damping limit (~0.7) and would needlessly weaken GS, so use 1.0.
      for (int s = 0; s < sweeps; ++s) multicolorGSMom(lv.op, lv.x, bc, lv.col, 1.0);
    else
      for (int s = 0; s < sweeps; ++s) jacobiMom(lv.op, lv.x, bc, lv.tmp, omega);
  }
  void uploadLevel(std::size_t L, const std::vector<double>& diag, const std::vector<Index>& start,
                   const std::vector<Index>& nbr, const std::vector<double>& coef) {
    Level& lv = levels_[L];
    lv.op.n = static_cast<Index>(diag.size());
    lv.op.diag = toDevice(diag, "mmg_diag");
    lv.op.faceStart = toDevice(start, "mmg_start");
    lv.op.faceNbr = toDevice(nbr, "mmg_nbr");
    lv.op.faceCoef = toDevice(coef, "mmg_coef");
    lv.col = greedyColoring(start, nbr, lv.op.n);
  }
  std::vector<Octree> octs_;
  std::vector<Level> levels_;
  bool useGS_ = false;  // multicolour Gauss–Seidel smoother (opt-in; default weighted Jacobi)
};

// ---------------------------------------------------------------------------
// Jacobi-preconditioned BiCGStab for the (non-symmetric) momentum operator. Reuses the
// device matvec + Kokkos reductions; the preconditioner is `jacPre` damped-Jacobi sweeps
// of A (diagonal-dominant ⇒ a cheap, effective smoother-preconditioner). Robust where
// plain Jacobi stalls (large dt / weak reaction term).
// ---------------------------------------------------------------------------
template <unsigned Bits = 21u>
class MomentumSolver {
 public:
  void setJacobi(int preSweeps, double omega) {
    jacPre_ = preSweeps;
    omega_ = omega;
  }

  /// Set a generic preconditioner `z = M⁻¹ r` (a host callable that launches device kernels) — the
  /// multigrid V-cycle gives the smooth-mode coverage Jacobi lacks, so the momentum iteration count
  /// stops growing with N. Decoupled from the MG type (Galerkin MomentumMG or rediscretized
  /// VelocityMG) via std::function, so the two coarse-operator strategies are interchangeable
  /// (and the solver carries no MG type). Pass an empty function to revert to damped-Jacobi. The
  /// preconditioner never changes the converged solution (the matvec is the exact operator).
  void setPreconditioner(std::function<void(View<const double>, View<double>)> fn) {
    precFn_ = std::move(fn);
  }

  /// Plain weighted-Jacobi solve (the simple parallel mirror of the host GS smoother):
  /// `sweeps` damped-Jacobi sweeps of A u = b in place. Returns the final residual L2.
  double solveJacobi(const MomentumOp& op, View<double> u, View<const double> b,
                     int sweeps) {
    ensure(op.n);
    for (int s = 0; s < sweeps; ++s) jacobiMom(op, u, b, tmp_, omega_);
    residualMom(op, View<const double>(u), b, r_);
    return std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), op.n));
  }

  struct Result {
    int iters = 0;
    double res0 = 0.0;
    double res = 0.0;
  };

  /// MG-preconditioned defect-correction (Richardson) solve of A u = b in place:
  /// u ← u + M⁻¹(b − A u), M = the preconditioner (velocity-MG if set, else Jacobi sweeps).
  /// Unlike BiCGStab it cannot break down — robust for the strongly non-symmetric momentum
  /// operator with implicit-FOU advection, where the velocity-MG (built from the viscous base)
  /// is only an approximate inverse. Converges when the advection is a perturbation of the
  /// viscous+reaction operator (low–moderate cell Reynolds number). `maxIters` caps the
  /// iterations; `tol` is relative to ||b−Au₀||.
  Result solveDefectCorrection(const MomentumOp& op, View<double> u, View<const double> b,
                               int maxIters = 200, double tol = 1e-8) {
    const Index n = op.n;
    ensure(n);
    Result R;
    residualMom(op, View<const double>(u), b, r_);
    R.res0 = std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), n));
    if (R.res0 == 0.0) return R;
    double rnorm = R.res0;
    int it = 0;
    for (; it < maxIters; ++it) {
      applyPrec(op, r_, phat_);                       // phat = M⁻¹ r
      axpy(u, 1.0, View<const double>(phat_), n);     // u += phat
      residualMom(op, View<const double>(u), b, r_);
      rnorm = std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), n));
      if (rnorm <= tol * R.res0) {
        ++it;
        break;
      }
    }
    R.iters = it;
    R.res = rnorm;
    return R;
  }

  /// Jacobi-preconditioned BiCGStab solve of A u = b in place. `maxIters` caps the outer
  /// iterations; `tol` is relative to ||b−Au0||. Returns {iters, final residual L2}.
  Result solveBiCGStab(const MomentumOp& op, View<double> u, View<const double> b,
                       int maxIters = 500, double tol = 1e-10) {
    const Index n = op.n;
    ensure(n);
    Result R;
    // r = b − A u
    residualMom(op, View<const double>(u), b, r_);
    Kokkos::deep_copy(rhat_, r_);  // shadow residual
    R.res0 = std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), n));
    if (R.res0 == 0.0) return R;
    double rho = 1, alpha = 1, omega = 1;
    Kokkos::deep_copy(v_, 0.0);
    Kokkos::deep_copy(p_, 0.0);
    double rnorm = R.res0;
    int it = 0;
    for (; it < maxIters; ++it) {
      double rhoNew = dotPlain(View<const double>(rhat_), View<const double>(r_), n);
      if (rhoNew == 0.0) break;
      double beta = (rhoNew / rho) * (alpha / omega);
      // p = r + beta (p − omega v)
      bicgPUpdate(p_, View<const double>(r_), View<const double>(v_), beta, omega, n);
      applyPrec(op, p_, phat_);  // phat = M^{-1} p
      applyMom(op, View<const double>(phat_), v_);
      double rhatV = dotPlain(View<const double>(rhat_), View<const double>(v_), n);
      alpha = rhoNew / rhatV;
      // s = r − alpha v
      Kokkos::deep_copy(s_, r_);
      axpy(s_, -alpha, View<const double>(v_), n);
      double snorm = std::sqrt(dotPlain(View<const double>(s_), View<const double>(s_), n));
      if (snorm <= tol * R.res0) {
        axpy(u, alpha, View<const double>(phat_), n);  // u += alpha phat
        rnorm = snorm;
        ++it;
        break;
      }
      applyPrec(op, s_, shat_);  // shat = M^{-1} s
      applyMom(op, View<const double>(shat_), t_);
      double tt = dotPlain(View<const double>(t_), View<const double>(t_), n);
      omega = (tt != 0.0) ? dotPlain(View<const double>(t_), View<const double>(s_), n) / tt : 0.0;
      // u += alpha phat + omega shat
      axpy(u, alpha, View<const double>(phat_), n);
      axpy(u, omega, View<const double>(shat_), n);
      // r = s − omega t
      Kokkos::deep_copy(r_, s_);
      axpy(r_, -omega, View<const double>(t_), n);
      rnorm = std::sqrt(dotPlain(View<const double>(r_), View<const double>(r_), n));
      if (rnorm <= tol * R.res0) {
        ++it;
        break;
      }
      rho = rhoNew;
      if (omega == 0.0) break;
    }
    R.iters = it;
    R.res = rnorm;
    return R;
  }

 private:
  // z = M^{-1} v : the generic MG preconditioner if set, else `jacPre_` damped-Jacobi sweeps of
  // A z = v starting from z = 0.
  void applyPrec(const MomentumOp& op, View<double> v, View<double> z) {
    if (precFn_) {
      precFn_(View<const double>(v), z);
      return;
    }
    Kokkos::deep_copy(z, 0.0);
    if (jacPre_ <= 0) {  // no preconditioner ⇒ identity
      Kokkos::deep_copy(z, v);
      return;
    }
    for (int s = 0; s < jacPre_; ++s) jacobiMom(op, z, View<const double>(v), tmp_, omega_);
  }
  void ensure(Index n) {
    if (r_.extent(0) == static_cast<std::size_t>(n)) return;
    auto mk = [&](const char* l) { return View<double>(l, static_cast<std::size_t>(n)); };
    r_ = mk("mom_r");
    rhat_ = mk("mom_rhat");
    p_ = mk("mom_p");
    phat_ = mk("mom_phat");
    v_ = mk("mom_v");
    s_ = mk("mom_s");
    shat_ = mk("mom_shat");
    t_ = mk("mom_t");
    tmp_ = mk("mom_tmp");
  }

  View<double> r_, rhat_, p_, phat_, v_, s_, shat_, t_, tmp_;
  int jacPre_ = 2;
  double omega_ = 0.7;
  std::function<void(View<const double>, View<double>)> precFn_;  // generic z = M^{-1} r
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_MOMENTUM_HPP

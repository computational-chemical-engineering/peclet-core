// core — smoothed-aggregation algebraic multigrid (SA-AMG) for a general assembled sparse SPD
// operator on an irregular graph. Mesh-agnostic: it names no octree / momentum / AMR types and
// assumes nothing beyond a CSR (diagonal + off-diagonal). The motivating consumer is the voro
// energy-minimisation mesh optimiser, whose Gauss–Newton Hessian H = 2γ JᵀJ lives on the Voronoi
// cell-adjacency graph (a "Laplacian-squared", SPD, no geometric grid hierarchy) — so a geometric
// multigrid does not apply and only ALGEBRAIC coarsening gives mesh-independent iteration counts,
// i.e. an O(N) solve. But nothing here is specific to that operator.
//
// This is the HOST ORACLE per docs/DEVICE_RESIDENCY_PLAN.md: plain std::vector, no Kokkos, so it
// builds and runs in the dependency-free core test suite and pins the mesh-independence contract
// (CG iteration count flat as N grows) that the later device + MPI ports must reproduce.
//
// Why a NEW class and not core::amr::MomentumMG: that one is hard-wired to BlockOctree (its
// hierarchy is octree coarsening via Morton ancestry) and its Galerkin triple-product is
// specialised to a piecewise-constant *injection* prolongator. Smoothed aggregation's prolongator
// P = (I − ω D⁻¹A) P₀ is a general sparse matrix, so the coarse operator A_c = PᵀAP is a general
// sparse triple product — a different computation on a different (graph) hierarchy.
//
// Pipeline (smoothed aggregation, Vaněk et al. 1996):
//   1. strength-of-connection on the NODAL graph (block size s = ndofPerNode: 3 for Voronoi seed
//      positions, 4 with power weights) — aggregation is nodal so the tentative prolongator keeps
//      the s near-nullspace modes (the uniform per-component "translation" modes of H) per node;
//   2. greedy aggregation (root pass + assign-leftovers pass + new-aggregate pass);
//   3. tentative prolongator P₀ — piecewise constant over aggregates, one coarse block per
//      aggregate, columns normalised to unit 2-norm;
//   4. prolongator smoothing P = (I − (ω/λ_max) D⁻¹A) P₀ (λ_max = spectral radius of D⁻¹A);
//   5. Galerkin coarse operator A_c = PᵀAP (general sparse RAP);
//   6. V-cycle with a Chebyshev polynomial smoother (GPU-parallel, near-Gauss–Seidel, no
//      colouring / no extra MPI comms — the decision recorded against colored-GS) or damped
//      Jacobi, used as a CG preconditioner.
#ifndef PECLET_CORE_SOLVER_GRAPH_AMG_HPP
#define PECLET_CORE_SOLVER_GRAPH_AMG_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

#include "peclet/core/common/types.hpp"

namespace peclet::core::solver {

using peclet::core::Index;

/// A general assembled sparse operator in CSR form: the diagonal is stored separately and the CSR
/// (`start`/`nbr`/`coef`) holds only the OFF-diagonal entries. SPD is assumed for the SA-AMG
/// hierarchy. This is the mesh-agnostic operator the device path will mirror as a Kokkos-View CSR
/// (the same layout `core::amr::FaceCsrOpT` exposes), so the host oracle and the device operator
/// share one arithmetic.
struct HostCsrOp {
  Index n = 0;
  std::vector<double> diag;   ///< size n
  std::vector<Index> start;   ///< size n+1, off-diagonal row offsets
  std::vector<Index> nbr;     ///< size nnz (off-diagonal column indices)
  std::vector<double> coef;   ///< size nnz (off-diagonal values)

  /// y = A x.
  void apply(const std::vector<double>& x, std::vector<double>& y) const {
    for (Index i = 0; i < n; ++i) {
      double s = diag[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
      for (Index k = start[static_cast<std::size_t>(i)]; k < start[static_cast<std::size_t>(i) + 1];
           ++k)
        s += coef[static_cast<std::size_t>(k)] *
             x[static_cast<std::size_t>(nbr[static_cast<std::size_t>(k)])];
      y[static_cast<std::size_t>(i)] = s;
    }
  }
};

struct AmgParams {
  int ndofPerNode = 1;      ///< block size for nodal aggregation (Voronoi positions ⇒ 3, +weight ⇒ 4)
  double theta = 0.05;      ///< strength-of-connection threshold (fraction of √(A_ii·A_jj))
  double smoothOmega = 4.0 / 3.0;  ///< prolongator-smoothing damping (× 1/λ_max)
  int maxLevels = 25;       ///< hierarchy depth cap
  Index coarsest = 40;      ///< stop coarsening once a level has ≤ this many DOFs
  int pre = 1, post = 1;    ///< smoother sweeps per level (pre == post keeps the V-cycle symmetric)
  int chebDegree = 2;       ///< Chebyshev smoother polynomial degree; 0 ⇒ damped-Jacobi smoother
  double jacobiOmega = 0.6; ///< damped-Jacobi smoother relaxation (used when chebDegree == 0)
  int eigIters = 15;        ///< power-iteration steps for the per-level λ_max(D⁻¹A) estimate
};

/// Smoothed-aggregation AMG hierarchy usable as `z = M⁻¹ r` (one symmetric V-cycle from a zero
/// initial guess) — the CG preconditioner that makes the iteration count mesh-independent.
class GraphAMG {
 public:
  void build(const HostCsrOp& A, const AmgParams& prm = {}) {
    prm_ = prm;
    levels_.clear();
    levels_.emplace_back();
    levels_.back().A = A;
    for (int L = 0; L + 1 < prm_.maxLevels; ++L) {
      finalizeSmoother(levels_[static_cast<std::size_t>(L)]);
      if (levels_[static_cast<std::size_t>(L)].A.n <= prm_.coarsest)
        break;
      Level next;
      if (!coarsen(levels_[static_cast<std::size_t>(L)], next))
        break;  // no further coarsening possible (already minimal)
      levels_.push_back(std::move(next));
    }
    finalizeSmoother(levels_.back());
    for (auto& lv : levels_)
      allocScratch(lv);
  }

  /// z = M⁻¹ r: one V-cycle (correction scheme) from a zero initial guess.
  void apply(const std::vector<double>& r, std::vector<double>& z) const {
    Level& l0 = levels_[0];
    l0.b = r;
    std::fill(l0.x.begin(), l0.x.end(), 0.0);
    vcycle(0);
    z = l0.x;
  }

  int numLevels() const { return static_cast<int>(levels_.size()); }
  Index size(int L = 0) const { return levels_[static_cast<std::size_t>(L)].A.n; }

  /// Total nonzeros / finest nonzeros — the grid+operator complexity (should stay ~<2 for a healthy
  /// hierarchy; a useful health check when tuning theta).
  double operatorComplexity() const {
    double nnz0 = static_cast<double>(levels_[0].A.coef.size() + levels_[0].A.diag.size());
    double tot = 0;
    for (auto& lv : levels_)
      tot += static_cast<double>(lv.A.coef.size() + lv.A.diag.size());
    return (nnz0 > 0) ? tot / nnz0 : 0.0;
  }

 private:
  struct Level {
    HostCsrOp A;
    // Prolongation to the next-coarser level (this level → L+1), row-CSR: fine row i has entries
    // (Pcol, Pval) over coarse DOFs. Empty on the coarsest level.
    std::vector<Index> Pstart, Pcol;
    std::vector<double> Pval;
    Index nc = 0;  // number of coarse DOFs (columns of P)
    // Smoother data:
    std::vector<double> invDiag;  // 1/diag (0 where diag == 0)
    double lmax = 1.0;            // spectral radius estimate of D⁻¹A on this level
    // Scratch (mutable — apply() is logically const):
    mutable std::vector<double> x, b, res, t0, t1;
  };

  AmgParams prm_;
  mutable std::vector<Level> levels_;

  static void allocScratch(Level& lv) {
    const std::size_t n = static_cast<std::size_t>(lv.A.n);
    lv.x.assign(n, 0.0);
    lv.b.assign(n, 0.0);
    lv.res.assign(n, 0.0);
    lv.t0.assign(n, 0.0);
    lv.t1.assign(n, 0.0);
  }

  // 1/diag + power-iteration estimate of λ_max(D⁻¹A) for the Chebyshev / Jacobi smoother bounds.
  void finalizeSmoother(Level& lv) {
    const std::size_t n = static_cast<std::size_t>(lv.A.n);
    lv.invDiag.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double d = lv.A.diag[i];
      lv.invDiag[i] = (std::fabs(d) > 1e-300) ? 1.0 / d : 0.0;
    }
    // v ← D⁻¹A v (Rayleigh quotient in the standard inner product approximates ρ(D⁻¹A)). Seed with
    // a high-frequency ±1 hash pattern, NOT a smooth vector: the dominant eigenvector of D⁻¹A is the
    // highest-frequency mode, so a smooth seed overlaps it weakly and power iteration underestimates
    // λ_max — which would let a higher-degree Chebyshev amplify the modes above the (too-low) bound.
    std::vector<double> v(n, 0.0), w(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      // splitmix64 bit-mixing → a well-decorrelated ±1 vector (O(1) overlap with the top mode, so
      // power iteration converges in a handful of steps). A poorly-mixed hash — e.g. i·odd & 1,
      // which only preserves i's low bit — gives a degenerate, non-dominant pattern that never
      // converges.
      std::uint64_t z = static_cast<std::uint64_t>(i) + 0x9E3779B97F4A7C15ULL;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z = z ^ (z >> 31);
      v[i] = (z & 1ULL) ? 1.0 : -1.0;
    }
    double nrm = std::sqrt(dot(v, v));
    if (nrm == 0.0) {
      lv.lmax = 1.0;
      return;
    }
    for (auto& e : v)
      e /= nrm;
    double lam = 1.0;
    for (int k = 0; k < prm_.eigIters; ++k) {
      lv.A.apply(v, w);
      for (std::size_t i = 0; i < n; ++i)
        w[i] *= lv.invDiag[i];
      lam = dot(v, w);  // v is unit-norm ⇒ Rayleigh quotient
      const double wn = std::sqrt(dot(w, w));
      if (wn == 0.0)
        break;
      for (std::size_t i = 0; i < n; ++i)
        v[i] = w[i] / wn;
    }
    lv.lmax = (lam > 0.0) ? lam : 1.0;
  }

  // Build the coarse level (aggregation → tentative P₀ → smoothed P → Galerkin A_c). Returns false
  // if the graph cannot be coarsened further (one aggregate).
  bool coarsen(Level& lv, Level& next) {
    const Index n = lv.A.n;
    const int s = prm_.ndofPerNode;
    const Index nNodes = n / s;

    // --- 1. nodal strength-of-connection (Frobenius norm of each s×s block) ---
    // blockFro[I] : map J -> ‖A_block(I,J)‖_F² (accumulated), plus the diagonal block norm.
    std::vector<std::map<Index, double>> blk(static_cast<std::size_t>(nNodes));
    std::vector<double> ddiag(static_cast<std::size_t>(nNodes), 0.0);  // ‖diagonal block‖_F²
    for (Index i = 0; i < n; ++i) {
      const Index I = i / s;
      ddiag[static_cast<std::size_t>(I)] += lv.A.diag[static_cast<std::size_t>(i)] *
                                            lv.A.diag[static_cast<std::size_t>(i)];
      for (Index k = lv.A.start[static_cast<std::size_t>(i)];
           k < lv.A.start[static_cast<std::size_t>(i) + 1]; ++k) {
        const Index J = lv.A.nbr[static_cast<std::size_t>(k)] / s;
        const double c = lv.A.coef[static_cast<std::size_t>(k)];
        if (J == I)
          ddiag[static_cast<std::size_t>(I)] += c * c;
        else
          blk[static_cast<std::size_t>(I)][J] += c * c;
      }
    }
    // Strong set per node: J strong to I iff ‖A(I,J)‖_F ≥ θ·√(‖A(I,I)‖_F·‖A(J,J)‖_F).
    std::vector<std::vector<Index>> strong(static_cast<std::size_t>(nNodes));
    const double th2 = prm_.theta * prm_.theta;
    // (blk/ddiag hold SQUARED Frobenius norms, so square the criterion: ‖S‖_F² ≥ θ²·‖D_I‖_F·‖D_J‖_F
    //  = θ²·√(ddiag_I·ddiag_J).)
    for (Index I = 0; I < nNodes; ++I)
      for (auto& [J, fro2] : blk[static_cast<std::size_t>(I)])
        if (fro2 >= th2 * std::sqrt(ddiag[static_cast<std::size_t>(I)] *
                                    ddiag[static_cast<std::size_t>(J)]))
          strong[static_cast<std::size_t>(I)].push_back(J);  // (blk iterates J in sorted order)
    auto isStrong = [&](Index I, Index J) {
      const auto& s = strong[static_cast<std::size_t>(I)];
      return std::binary_search(s.begin(), s.end(), J);
    };

    // --- 2. greedy aggregation ---
    std::vector<Index> agg(static_cast<std::size_t>(nNodes), -1);
    Index nAgg = 0;
    // Pass 1 — roots: a fully-unaggregated node with a fully-unaggregated strong neighbourhood
    // seeds a new aggregate consisting of itself + those neighbours.
    for (Index I = 0; I < nNodes; ++I) {
      if (agg[static_cast<std::size_t>(I)] != -1)
        continue;
      bool ok = true;
      for (Index J : strong[static_cast<std::size_t>(I)])
        if (agg[static_cast<std::size_t>(J)] != -1) {
          ok = false;
          break;
        }
      if (!ok)
        continue;
      agg[static_cast<std::size_t>(I)] = nAgg;
      for (Index J : strong[static_cast<std::size_t>(I)])
        agg[static_cast<std::size_t>(J)] = nAgg;
      ++nAgg;
    }
    // Pass 2 — attach each leftover to the adjacent aggregate it is most strongly tied to.
    for (Index I = 0; I < nNodes; ++I) {
      if (agg[static_cast<std::size_t>(I)] != -1)
        continue;
      Index best = -1;
      double bestw = -1.0;
      for (Index J : strong[static_cast<std::size_t>(I)]) {
        const Index a = agg[static_cast<std::size_t>(J)];
        if (a < 0)
          continue;
        const double w = blk[static_cast<std::size_t>(I)][J];
        if (w > bestw) {
          bestw = w;
          best = a;
        }
      }
      if (best >= 0)
        agg[static_cast<std::size_t>(I)] = best;
    }
    // Pass 3 — remaining nodes form fresh aggregates from their still-unaggregated strong clusters
    // (a fully isolated node becomes a singleton aggregate).
    for (Index I = 0; I < nNodes; ++I) {
      if (agg[static_cast<std::size_t>(I)] != -1)
        continue;
      agg[static_cast<std::size_t>(I)] = nAgg;
      for (Index J : strong[static_cast<std::size_t>(I)])
        if (agg[static_cast<std::size_t>(J)] == -1)
          agg[static_cast<std::size_t>(J)] = nAgg;
      ++nAgg;
    }
    if (nAgg <= 1 || nAgg == nNodes)
      return false;  // no useful coarsening

    // --- 3. tentative prolongator P₀ (block piecewise-constant, unit-2-norm columns) ---
    // Each aggregate → s coarse DOFs; fine DOF (I,d) maps to coarse DOF (agg(I)·s + d) with weight
    // 1/√|agg(I)| (the QR normalisation of the constant near-nullspace mode over the aggregate).
    std::vector<Index> cnt(static_cast<std::size_t>(nAgg), 0);
    for (Index I = 0; I < nNodes; ++I)
      ++cnt[static_cast<std::size_t>(agg[static_cast<std::size_t>(I)])];
    std::vector<double> invSqrtCnt(static_cast<std::size_t>(nAgg));
    for (Index a = 0; a < nAgg; ++a)
      invSqrtCnt[static_cast<std::size_t>(a)] =
          1.0 / std::sqrt(static_cast<double>(cnt[static_cast<std::size_t>(a)]));
    const Index ncoarse = nAgg * s;

    // --- 4. FILTERED prolongator smoothing P = (I − (ω/λ_max) D_F⁻¹ A^F) P₀ ---
    // A^F is the filtered operator: off-diagonal entries between nodes that are NOT strongly
    // connected are dropped and LUMPED into the diagonal (row sums, hence the constant nullspace,
    // preserved). Filtering is the standard SA fill-control: smoothing along only the strong graph
    // keeps P — and therefore the Galerkin coarse operator — sparse, so operator complexity stays
    // bounded (unfiltered smoothing densifies the coarse levels and, through the lmax-dependent
    // step, makes the whole hierarchy hypersensitive / erratic). P₀ row for fine DOF i=(I,d) is a
    // single entry (agg(I)·s + d, invSqrtCnt).
    const double sc = prm_.smoothOmega / lv.lmax;
    auto p0col = [&](Index i) -> Index {
      const Index I = i / s, d = i % s;
      return agg[static_cast<std::size_t>(I)] * s + d;
    };
    auto p0val = [&](Index i) -> double {
      return invSqrtCnt[static_cast<std::size_t>(agg[static_cast<std::size_t>(i / s)])];
    };
    std::vector<Index> Pstart(static_cast<std::size_t>(n) + 1, 0), Pcol;
    std::vector<double> Pval;
    std::map<Index, double> row;
    for (Index i = 0; i < n; ++i) {
      const Index I = i / s;
      // Filtered diagonal: original diagonal + lumped weak off-diagonals.
      double dF = lv.A.diag[static_cast<std::size_t>(i)];
      for (Index k = lv.A.start[static_cast<std::size_t>(i)];
           k < lv.A.start[static_cast<std::size_t>(i) + 1]; ++k) {
        const Index J = lv.A.nbr[static_cast<std::size_t>(k)] / s;
        if (J != I && !isStrong(I, J))
          dF += lv.A.coef[static_cast<std::size_t>(k)];  // lump weak connection into the diagonal
      }
      row.clear();
      row[p0col(i)] += p0val(i);  // P₀ contribution
      // − sc·(1/dF)·(A^F P₀)_i, with A^F_ii = dF and A^F_ij kept only for strong (or intra-node) j.
      const double f = (std::fabs(dF) > 1e-300) ? (-sc / dF) : 0.0;
      row[p0col(i)] += f * dF * p0val(i);  // diagonal term ⇒ P₀_i·(1 − sc)
      for (Index k = lv.A.start[static_cast<std::size_t>(i)];
           k < lv.A.start[static_cast<std::size_t>(i) + 1]; ++k) {
        const Index j = lv.A.nbr[static_cast<std::size_t>(k)];
        const Index J = j / s;
        if (J != I && !isStrong(I, J))
          continue;  // weak: already lumped into dF
        row[p0col(j)] += f * lv.A.coef[static_cast<std::size_t>(k)] * p0val(j);
      }
      for (auto& [c, val] : row)
        if (val != 0.0) {
          Pcol.push_back(c);
          Pval.push_back(val);
        }
      Pstart[static_cast<std::size_t>(i) + 1] = static_cast<Index>(Pcol.size());
    }

    // --- 5. Galerkin coarse operator A_c = PᵀAP (general sparse RAP) ---
    // AP_i,: = diag_i·P_i,: + Σ_k coef_k·P_{nbr_k,:}   (fine row → coarse columns)
    // A_c[c1][c2] += Σ_i P[i][c1]·AP[i][c2]
    std::vector<std::map<Index, double>> Ac(static_cast<std::size_t>(ncoarse));
    std::map<Index, double> apRow;
    for (Index i = 0; i < n; ++i) {
      apRow.clear();
      const double di = lv.A.diag[static_cast<std::size_t>(i)];
      for (Index k = Pstart[static_cast<std::size_t>(i)];
           k < Pstart[static_cast<std::size_t>(i) + 1]; ++k)
        apRow[Pcol[static_cast<std::size_t>(k)]] += di * Pval[static_cast<std::size_t>(k)];
      for (Index k = lv.A.start[static_cast<std::size_t>(i)];
           k < lv.A.start[static_cast<std::size_t>(i) + 1]; ++k) {
        const Index j = lv.A.nbr[static_cast<std::size_t>(k)];
        const double c = lv.A.coef[static_cast<std::size_t>(k)];
        for (Index kk = Pstart[static_cast<std::size_t>(j)];
             kk < Pstart[static_cast<std::size_t>(j) + 1]; ++kk)
          apRow[Pcol[static_cast<std::size_t>(kk)]] += c * Pval[static_cast<std::size_t>(kk)];
      }
      for (Index k = Pstart[static_cast<std::size_t>(i)];
           k < Pstart[static_cast<std::size_t>(i) + 1]; ++k) {
        const Index c1 = Pcol[static_cast<std::size_t>(k)];
        const double p1 = Pval[static_cast<std::size_t>(k)];
        auto& r1 = Ac[static_cast<std::size_t>(c1)];
        for (auto& [c2, apv] : apRow)
          r1[c2] += p1 * apv;
      }
    }

    // Flatten A_c into the coarse HostCsrOp (drop structural zeros).
    next.A.n = ncoarse;
    next.A.diag.assign(static_cast<std::size_t>(ncoarse), 0.0);
    next.A.start.assign(static_cast<std::size_t>(ncoarse) + 1, 0);
    next.A.nbr.clear();
    next.A.coef.clear();
    for (Index c = 0; c < ncoarse; ++c) {
      for (auto& [c2, val] : Ac[static_cast<std::size_t>(c)]) {
        if (c2 == c) {
          next.A.diag[static_cast<std::size_t>(c)] = val;
        } else if (val != 0.0) {
          next.A.nbr.push_back(c2);
          next.A.coef.push_back(val);
        }
      }
      next.A.start[static_cast<std::size_t>(c) + 1] = static_cast<Index>(next.A.nbr.size());
    }

    // Stash the transfer on the fine level.
    lv.Pstart = std::move(Pstart);
    lv.Pcol = std::move(Pcol);
    lv.Pval = std::move(Pval);
    lv.nc = ncoarse;
    return true;
  }

  // ---- V-cycle ----
  void vcycle(int L) const {
    Level& lv = levels_[static_cast<std::size_t>(L)];
    if (L + 1 == static_cast<int>(levels_.size())) {
      coarseSolve(lv);
      return;
    }
    smooth(lv, prm_.pre);
    // res = b − A x
    lv.A.apply(lv.x, lv.res);
    for (std::size_t i = 0; i < lv.res.size(); ++i)
      lv.res[i] = lv.b[i] - lv.res[i];
    // restrict residual to coarse RHS (Pᵀ res), start coarse x at 0.
    Level& cl = levels_[static_cast<std::size_t>(L + 1)];
    std::fill(cl.b.begin(), cl.b.end(), 0.0);
    for (Index i = 0; i < lv.A.n; ++i)
      for (Index k = lv.Pstart[static_cast<std::size_t>(i)];
           k < lv.Pstart[static_cast<std::size_t>(i) + 1]; ++k)
        cl.b[static_cast<std::size_t>(lv.Pcol[static_cast<std::size_t>(k)])] +=
            lv.Pval[static_cast<std::size_t>(k)] * lv.res[static_cast<std::size_t>(i)];
    std::fill(cl.x.begin(), cl.x.end(), 0.0);
    vcycle(L + 1);
    // prolong + correct: x += P x_c
    for (Index i = 0; i < lv.A.n; ++i) {
      double s = 0.0;
      for (Index k = lv.Pstart[static_cast<std::size_t>(i)];
           k < lv.Pstart[static_cast<std::size_t>(i) + 1]; ++k)
        s += lv.Pval[static_cast<std::size_t>(k)] *
             cl.x[static_cast<std::size_t>(lv.Pcol[static_cast<std::size_t>(k)])];
      lv.x[static_cast<std::size_t>(i)] += s;
    }
    smooth(lv, prm_.post);
  }

  // Smoother: 4th-kind Chebyshev of degree prm_.chebDegree (default), or damped Jacobi when
  // chebDegree == 0. Both read only the previous iterate per matvec and are symmetric polynomials
  // in D⁻¹A ⇒ a valid SPD CG preconditioner with pre == post.
  void smooth(const Level& lv, int sweeps) const {
    if (sweeps <= 0)
      return;
    if (prm_.chebDegree <= 0) {
      for (int s = 0; s < sweeps; ++s)
        jacobiSweep(lv);
      return;
    }
    for (int s = 0; s < sweeps; ++s)
      chebSweep(lv);
  }

  void jacobiSweep(const Level& lv) const {
    lv.A.apply(lv.x, lv.res);  // res = A x
    for (std::size_t i = 0; i < lv.x.size(); ++i)
      lv.x[i] += prm_.jacobiOmega * lv.invDiag[i] * (lv.b[i] - lv.res[i]);
  }

  // One 4th-kind Chebyshev smoothing pass of degree k on A x = b (in place), Jacobi (D) prec.
  // (Lottes 2022 / Phillips & Fischer 2022.) Unlike the 1st-kind form it needs ONLY an upper bound
  // λ on ρ(D⁻¹A) — no lower-bound guess to get wrong on the Galerkin coarse levels — and its first
  // step is exactly the optimal 4/(3λ) damped-Jacobi step. Guaranteed a good, symmetric smoother.
  void chebSweep(const Level& lv) const {
    const int k = prm_.chebDegree;
    const double lam = 1.1 * lv.lmax;  // safety over-estimate: λ must bound the true spectral radius
    auto& r = lv.res;
    auto& d = lv.t0;
    auto& Ad = lv.t1;
    lv.A.apply(lv.x, r);  // r = b − A x
    for (std::size_t i = 0; i < r.size(); ++i)
      r[i] = lv.b[i] - r[i];
    std::fill(d.begin(), d.end(), 0.0);
    for (int i = 1; i <= k; ++i) {
      const double c1 = (2.0 * i - 3.0) / (2.0 * i + 1.0);
      const double c2 = (8.0 * i - 4.0) / ((2.0 * i + 1.0) * lam);
      for (std::size_t j = 0; j < d.size(); ++j)
        d[j] = c1 * d[j] + c2 * lv.invDiag[j] * r[j];
      for (std::size_t j = 0; j < lv.x.size(); ++j)
        lv.x[j] += d[j];
      if (i < k) {
        lv.A.apply(d, Ad);  // r −= A d
        for (std::size_t j = 0; j < r.size(); ++j)
          r[j] -= Ad[j];
      }
    }
  }

  // Coarsest level: a short unpreconditioned CG (the level is tiny, ≤ coarsest DOFs) — robust even
  // if A_c is only semidefinite (a consistent RHS still converges on the range).
  void coarseSolve(Level& lv) const {
    const Index n = lv.A.n;
    auto& x = lv.x;
    auto& r = lv.res;
    auto& p = lv.t0;
    auto& Ap = lv.t1;
    lv.A.apply(x, r);
    for (Index i = 0; i < n; ++i)
      r[static_cast<std::size_t>(i)] =
          lv.b[static_cast<std::size_t>(i)] - r[static_cast<std::size_t>(i)];
    p = r;
    double rr = dot(r, r);
    const double rr0 = rr;
    const int maxit = static_cast<int>(std::min<Index>(n, 200));
    for (int k = 0; k < maxit && rr > 1e-24 * rr0; ++k) {
      lv.A.apply(p, Ap);
      const double pAp = dot(p, Ap);
      if (pAp <= 0.0)
        break;
      const double alpha = rr / pAp;
      for (Index i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] += alpha * p[static_cast<std::size_t>(i)];
        r[static_cast<std::size_t>(i)] -= alpha * Ap[static_cast<std::size_t>(i)];
      }
      const double rrn = dot(r, r);
      const double beta = rrn / rr;
      for (Index i = 0; i < n; ++i)
        p[static_cast<std::size_t>(i)] =
            r[static_cast<std::size_t>(i)] + beta * p[static_cast<std::size_t>(i)];
      rr = rrn;
    }
  }

  static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
      s += a[i] * b[i];
    return s;
  }
};

/// PCG on A x = b preconditioned by an already-built GraphAMG (one V-cycle per iteration). Returns
/// {iterations, initial residual L2, final residual L2}. `tol` is relative to ‖b − A x₀‖.
struct CgResult {
  int iters = 0;
  double res0 = 0.0;
  double res = 0.0;
};

inline CgResult amgPcg(const HostCsrOp& A, std::vector<double>& x, const std::vector<double>& b,
                       const GraphAMG& M, int maxIters = 500, double tol = 1e-8) {
  const Index n = A.n;
  const std::size_t nn = static_cast<std::size_t>(n);
  std::vector<double> r(nn), z(nn), p(nn), Ap(nn);
  auto dot = [&](const std::vector<double>& u, const std::vector<double>& v) {
    double s = 0.0;
    for (std::size_t i = 0; i < nn; ++i)
      s += u[i] * v[i];
    return s;
  };
  A.apply(x, Ap);
  for (std::size_t i = 0; i < nn; ++i)
    r[i] = b[i] - Ap[i];
  CgResult R;
  R.res0 = std::sqrt(dot(r, r));
  if (R.res0 == 0.0)
    return R;
  M.apply(r, z);
  p = z;
  double rz = dot(r, z);
  double rnorm = R.res0;
  int it = 0;
  for (; it < maxIters; ++it) {
    A.apply(p, Ap);
    const double pAp = dot(p, Ap);
    if (pAp <= 0.0)
      break;
    const double alpha = rz / pAp;
    for (std::size_t i = 0; i < nn; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    rnorm = std::sqrt(dot(r, r));
    if (rnorm <= tol * R.res0) {
      ++it;
      break;
    }
    M.apply(r, z);
    const double rzNew = dot(r, z);
    const double beta = rzNew / rz;
    for (std::size_t i = 0; i < nn; ++i)
      p[i] = z[i] + beta * p[i];
    rz = rzNew;
  }
  R.iters = it;
  R.res = rnorm;
  return R;
}

}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_GRAPH_AMG_HPP

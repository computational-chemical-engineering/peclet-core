// core — device (Kokkos) APPLY for the smoothed-aggregation GraphAMG hierarchy.
//
// Setup (strength graph, greedy aggregation, smoothed prolongator, Galerkin RAP) stays on the
// HOST oracle (graph_amg.hpp) — it is sequential-greedy and runs once per operator rebuild. This
// class mirrors the finished hierarchy into device CSR Views and runs the HOT path — the V-cycle
// apply (per-level SpMV, 4th-kind-Chebyshev / damped-Jacobi smoothing, transfers, tiny coarse CG)
// — entirely on the device, so a device consumer (voro's device mesh optimiser, flow's device
// PCG) never round-trips the iterate to the host.
//
// Determinism vs the host oracle: SpMV sums each row sequentially in the same order, transfers are
// deterministic gathers (restriction uses a host-built transpose R = Pᵀ whose per-coarse-row
// entries keep increasing-fine-index order — the exact order of the host scatter), and the vector
// updates are elementwise ⇒ the V-cycle matches the host bit-for-bit EXCEPT the coarsest-level CG
// dot products (parallel_reduce reorders the sums), which perturb only the tightly-converged
// coarse correction (validated ~1e-12 relative in tests/test_graph_amg_device.cpp).
#ifndef PECLET_CORE_SOLVER_GRAPH_AMG_DEVICE_HPP
#define PECLET_CORE_SOLVER_GRAPH_AMG_DEVICE_HPP

#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/graph_amg.hpp"

namespace peclet::core::solver {

class GraphAMGDevice {
 public:
  using DView = View<double>;
  using IView = View<Index>;

  /// Host setup + one-time device mirror. Rebuild whenever the operator changes.
  void build(const HostCsrOp& A, const AmgParams& prm = {}) {
    host_.build(A, prm);
    prm_ = prm;
    mirror();
  }

  /// z = M⁻¹ r: one V-cycle (correction scheme) from a zero initial guess. Device views, size n.
  void apply(const DView& r, const DView& z) const {
    const DLevel& l0 = lv_[0];
    Kokkos::deep_copy(l0.b, r);
    Kokkos::deep_copy(l0.x, 0.0);
    vcycle(0);
    Kokkos::deep_copy(z, l0.x);
  }

  int numLevels() const { return (int)lv_.size(); }
  Index size(int L = 0) const { return lv_[(std::size_t)L].n; }
  const GraphAMG& hostHierarchy() const { return host_; }

  // DLevel is public for the same nvcc extended-lambda stub-generation reason as the methods.
  struct DLevel {
    Index n = 0, nc = 0;
    IView start, nbr;          // off-diagonal CSR
    DView coef, diag, invDiag;
    IView Pstart, Pcol;        // prolongation (fine rows)
    DView Pval;
    IView Rstart, Rcol;        // restriction = Pᵀ (coarse rows, entries in fine-index order)
    DView Rval;
    double lmax = 1.0;
    DView x, b, res, t0, t1;   // scratch
  };

  template <class T>
  static View<T> toDev(const std::vector<T>& v, const char* name) {
    View<T> d(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < v.size(); ++i)
      h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  }

  void mirror() {
    lv_.clear();
    for (const auto& hl : host_.levels()) {
      DLevel d;
      d.n = hl.A.n;
      d.nc = hl.nc;
      d.lmax = hl.lmax;
      d.start = toDev(hl.A.start, "amgd_start");
      d.nbr = toDev(hl.A.nbr, "amgd_nbr");
      d.coef = toDev(hl.A.coef, "amgd_coef");
      d.diag = toDev(hl.A.diag, "amgd_diag");
      d.invDiag = toDev(hl.invDiag, "amgd_invDiag");
      if (!hl.Pstart.empty()) {
        d.Pstart = toDev(hl.Pstart, "amgd_Pstart");
        d.Pcol = toDev(hl.Pcol, "amgd_Pcol");
        d.Pval = toDev(hl.Pval, "amgd_Pval");
        // Host-built transpose R = Pᵀ. Two-pass CSR transpose scanning fine rows in increasing
        // order, so each coarse row's entries keep increasing-fine-index order — the exact
        // accumulation order of the host oracle's restriction scatter (bit-compatible sums).
        std::vector<Index> Rstart((std::size_t)hl.nc + 1, 0), Rcol(hl.Pcol.size());
        std::vector<double> Rval(hl.Pval.size());
        for (Index c : hl.Pcol)
          ++Rstart[(std::size_t)c + 1];
        for (std::size_t c = 0; c < (std::size_t)hl.nc; ++c)
          Rstart[c + 1] += Rstart[c];
        std::vector<Index> cur(Rstart.begin(), Rstart.end() - 1);
        for (Index i = 0; i < hl.A.n; ++i)
          for (Index k = hl.Pstart[(std::size_t)i]; k < hl.Pstart[(std::size_t)i + 1]; ++k) {
            const Index c = hl.Pcol[(std::size_t)k];
            const Index at = cur[(std::size_t)c]++;
            Rcol[(std::size_t)at] = i;
            Rval[(std::size_t)at] = hl.Pval[(std::size_t)k];
          }
        d.Rstart = toDev(Rstart, "amgd_Rstart");
        d.Rcol = toDev(Rcol, "amgd_Rcol");
        d.Rval = toDev(Rval, "amgd_Rval");
      }
      auto scratch = [n = (std::size_t)d.n](const char* nm) {
        return DView(Kokkos::view_alloc(std::string(nm), Kokkos::WithoutInitializing), n);
      };
      d.x = scratch("amgd_x");
      d.b = scratch("amgd_b");
      d.res = scratch("amgd_res");
      d.t0 = scratch("amgd_t0");
      d.t1 = scratch("amgd_t1");
      lv_.push_back(std::move(d));
    }
  }

  // nvcc requires member functions containing extended (device) lambdas to be PUBLIC — the
  // OpenMP/host build accepts them private, so the breakage only shows on the CUDA backend.
 public:
  // y = A x (diag + off-diagonal CSR; each row summed sequentially — the host order).
  static void spmv(const DLevel& lv, const DView& x, const DView& y) {
    IView start = lv.start, nbr = lv.nbr;
    DView coef = lv.coef, diag = lv.diag;
    Kokkos::parallel_for(
        "peclet::core::amgd_spmv", Kokkos::RangePolicy<ExecSpace>(0, lv.n),
        KOKKOS_LAMBDA(const Index i) {
          double s = diag(i) * x(i);
          for (Index k = start(i); k < start(i + 1); ++k)
            s += coef(k) * x(nbr(k));
          y(i) = s;
        });
  }

  void vcycle(int L) const {
    const DLevel& lv = lv_[(std::size_t)L];
    if (L + 1 == (int)lv_.size()) {
      coarseSolve(lv);
      return;
    }
    smooth(lv, prm_.pre);
    spmv(lv, lv.x, lv.res);  // res = b − A x
    {
      DView res = lv.res, b = lv.b;
      Kokkos::parallel_for(
          "peclet::core::amgd_resid", Kokkos::RangePolicy<ExecSpace>(0, lv.n),
          KOKKOS_LAMBDA(const Index i) { res(i) = b(i) - res(i); });
    }
    const DLevel& cl = lv_[(std::size_t)L + 1];
    {  // coarse b = Pᵀ res via the transpose gather (fine-index order per coarse row)
      IView Rstart = lv.Rstart, Rcol = lv.Rcol;
      DView Rval = lv.Rval, res = lv.res, cb = cl.b;
      Kokkos::parallel_for(
          "peclet::core::amgd_restrict", Kokkos::RangePolicy<ExecSpace>(0, cl.n),
          KOKKOS_LAMBDA(const Index c) {
            double s = 0.0;
            for (Index k = Rstart(c); k < Rstart(c + 1); ++k)
              s += Rval(k) * res(Rcol(k));
            cb(c) = s;
          });
    }
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(L + 1);
    {  // x += P x_c
      IView Pstart = lv.Pstart, Pcol = lv.Pcol;
      DView Pval = lv.Pval, x = lv.x, cx = cl.x;
      Kokkos::parallel_for(
          "peclet::core::amgd_prolong", Kokkos::RangePolicy<ExecSpace>(0, lv.n),
          KOKKOS_LAMBDA(const Index i) {
            double s = 0.0;
            for (Index k = Pstart(i); k < Pstart(i + 1); ++k)
              s += Pval(k) * cx(Pcol(k));
            x(i) += s;
          });
    }
    smooth(lv, prm_.post);
  }

  void smooth(const DLevel& lv, int sweeps) const {
    if (sweeps <= 0)
      return;
    for (int s = 0; s < sweeps; ++s)
      if (prm_.chebDegree <= 0)
        jacobiSweep(lv);
      else
        chebSweep(lv);
  }

  void jacobiSweep(const DLevel& lv) const {
    const double step = prm_.jacobiOmega / lv.lmax;
    spmv(lv, lv.x, lv.res);
    DView x = lv.x, b = lv.b, res = lv.res, invD = lv.invDiag;
    Kokkos::parallel_for(
        "peclet::core::amgd_jacobi", Kokkos::RangePolicy<ExecSpace>(0, lv.n),
        KOKKOS_LAMBDA(const Index i) { x(i) += step * invD(i) * (b(i) - res(i)); });
  }

  void chebSweep(const DLevel& lv) const {
    const int k = prm_.chebDegree;
    const double lam = 1.1 * lv.lmax;
    DView r = lv.res, d = lv.t0, Ad = lv.t1, x = lv.x, b = lv.b, invD = lv.invDiag;
    spmv(lv, lv.x, r);  // r = b − A x
    Kokkos::parallel_for(
        "peclet::core::amgd_cheb_r0", Kokkos::RangePolicy<ExecSpace>(0, lv.n),
        KOKKOS_LAMBDA(const Index i) { r(i) = b(i) - r(i); });
    Kokkos::deep_copy(d, 0.0);
    for (int i = 1; i <= k; ++i) {
      const double c1 = (2.0 * i - 3.0) / (2.0 * i + 1.0);
      const double c2 = (8.0 * i - 4.0) / ((2.0 * i + 1.0) * lam);
      Kokkos::parallel_for(
          "peclet::core::amgd_cheb_d", Kokkos::RangePolicy<ExecSpace>(0, lv.n),
          KOKKOS_LAMBDA(const Index j) {
            d(j) = c1 * d(j) + c2 * invD(j) * r(j);
            x(j) += d(j);
          });
      if (i < k) {
        spmv(lv, d, Ad);
        Kokkos::parallel_for(
            "peclet::core::amgd_cheb_rup", Kokkos::RangePolicy<ExecSpace>(0, lv.n),
            KOKKOS_LAMBDA(const Index j) { r(j) -= Ad(j); });
      }
    }
  }

  static double dot(const DView& a, const DView& b, Index n) {
    double s = 0.0;
    Kokkos::parallel_reduce(
        "peclet::core::amgd_dot", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const Index i, double& acc) { acc += a(i) * b(i); }, s);
    return s;
  }

  // Coarsest level: short unpreconditioned CG (tiny n). The dot products are the ONE place the
  // device apply reorders sums vs the host oracle.
  void coarseSolve(const DLevel& lv) const {
    if (prm_.coarseSweeps > 0) {
      smooth(lv, prm_.coarseSweeps);
      return;
    }
    const Index n = lv.n;
    DView x = lv.x, r = lv.res, p = lv.t0, Ap = lv.t1, b = lv.b;
    spmv(lv, x, r);
    Kokkos::parallel_for(
        "peclet::core::amgd_cg_r0", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const Index i) {
          r(i) = b(i) - r(i);
          p(i) = r(i);
        });
    double rr = dot(r, r, n);
    const double rr0 = rr;
    const int maxit = (int)std::min<Index>(n, 200);
    for (int it = 0; it < maxit && rr > 1e-24 * rr0; ++it) {
      spmv(lv, p, Ap);
      const double pAp = dot(p, Ap, n);
      if (pAp <= 0.0)
        break;
      const double alpha = rr / pAp;
      Kokkos::parallel_for(
          "peclet::core::amgd_cg_up", Kokkos::RangePolicy<ExecSpace>(0, n),
          KOKKOS_LAMBDA(const Index i) {
            x(i) += alpha * p(i);
            r(i) -= alpha * Ap(i);
          });
      const double rrn = dot(r, r, n);
      const double beta = rrn / rr;
      Kokkos::parallel_for(
          "peclet::core::amgd_cg_p", Kokkos::RangePolicy<ExecSpace>(0, n),
          KOKKOS_LAMBDA(const Index i) { p(i) = r(i) + beta * p(i); });
      rr = rrn;
    }
  }

 private:
  GraphAMG host_;
  AmgParams prm_;
  std::vector<DLevel> lv_;
};

}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_GRAPH_AMG_DEVICE_HPP

// core — Anderson acceleration of a steady march: the grid-agnostic data path.
//
// Design: flow doc/steady_acceleration.md (D13: AndersonCore lives in core from the start; §3 the
// state and the metric; §4 the algorithm; §6 MPI and GPU). The caller owns a fixed-point map
// x ↦ g(x) that reads and writes a set of padded device buffers in place (flow: one
// `Solver::step()`); this header owns the history, the reductions, the least squares, the
// safeguards and the lazy mix of those buffers.
//
// MPI-free: a distributed run passes its collectives as two callables in `AndersonComm` (the
// pattern of csr_bicgstab.hpp's setDistributed), built from an MPI_Comm by
// solver/anderson_mpi.hpp, which is the MPI side. An empty `AndersonComm` is one rank, and an
// MPI_Allreduce / MPI_Bcast on one rank is the identity, so np = 1 is bit-identical to serial.
#ifndef PECLET_CORE_SOLVER_ANDERSON_HPP
#define PECLET_CORE_SOLVER_ANDERSON_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::core::solver {

/// What a state field is to the metric (design §3.1–3.2, rev 1: the metric is velocity only).
/// The integer values are those of revision 0, whose Pressure role (= 1) was deleted (WO-3b).
enum class AndersonRole : int {
  Velocity = 0,  ///< measured at unit weight, every inner entry
  Carried = 2,   ///< mixed, stored and differenced like the others, but not measured (e.g. P)
};

/// The collectives of a distributed run. Both callables empty = a single rank (serial, or np = 1
/// without them). `sumAll(data, n)` replaces data[0..n) by its global sum on every rank
/// (MPI_Allreduce SUM); `broadcast(data, bytes)` copies rank 0's bytes to every rank (MPI_Bcast).
/// solver/anderson_mpi.hpp builds one from an MPI_Comm.
struct AndersonComm {
  std::function<void(double* data, int count)> sumAll;
  std::function<void(void* data, std::size_t bytes)> broadcast;
  int rank = 0;  ///< this rank in the communicator; rank 0's decisions win (design §6.1)
};

/// The state descriptor (design §5.1): exactly what the caller's step reads across steps, as full
/// padded x-fastest buffers over one box, with their roles. Grid-agnostic; the caller's
/// parameter signature (flow: dt, ρ, μ, F) is NOT here — the adapter checks it.
struct AndersonState {
  /// The state buffers, each over the full padded box (extent(0) == extent[0]·extent[1]·extent[2]),
  /// read and written in place by the caller's step. The mix writes them too.
  std::vector<View<double>> fields;
  /// One role per field.
  std::vector<AndersonRole> roles;
  /// Padded extents e (inner + 2·ghost per axis); inner entries are ghost ≤ i < e − ghost.
  IVec<3> extent{};
  /// Ghost width G.
  int ghost = 2;
  /// Collectives; empty = a single rank.
  AndersonComm comm;
};

// ---------------------------------------------------------------------------------------------
// Device kernels and host linear algebra. Free functions, because an extended __host__ __device__
// lambda may not live in a private member function.
// ---------------------------------------------------------------------------------------------
namespace detail::anderson {

inline constexpr int kMaxWindow = 8;

/// Inner-region policy, x (the first index) fastest in tiles and within a tile on every backend
/// (suite CONVENTIONS §1; the same policy as flow's MDRange3).
using InnerPolicy =
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::IndexType<Index>,
                          Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;

inline InnerPolicy innerPolicy(const IVec<3>& e, int g) {
  return InnerPolicy({Index(g), Index(g), Index(g)}, {e[0] - g, e[1] - g, e[2] - g});
}

/// N independent double sums in one reduction.
template <int N>
struct SumN {
  double v[N];
  KOKKOS_INLINE_FUNCTION SumN() {
    for (int i = 0; i < N; ++i)
      v[i] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION SumN& operator+=(const SumN& o) {
    for (int i = 0; i < N; ++i)
      v[i] += o.v[i];
    return *this;
  }
};

}  // namespace detail::anderson
}  // namespace peclet::core::solver

namespace Kokkos {
template <int N>
struct reduction_identity<peclet::core::solver::detail::anderson::SumN<N>> {
  KOKKOS_FORCEINLINE_FUNCTION static peclet::core::solver::detail::anderson::SumN<N> sum() {
    return peclet::core::solver::detail::anderson::SumN<N>();
  }
};
}  // namespace Kokkos

namespace peclet::core::solver {
namespace detail::anderson {

/// A reduction result held on the device: a reduction into it does not synchronise the host, so
/// a phase's reductions queue back to back and are read back behind one fence (design §6.1).
template <class T>
using DeviceScalar = Kokkos::View<T, MemSpace>;

/// COUNT: inner entries where buf != prev, into `out` (no host synchronisation).
inline void countChanged(View<const double> buf, View<const double> prev, const IVec<3>& e, int g,
                         const DeviceScalar<Index>& out) {
  const Index ex = e[0], exy = e[0] * e[1];
  Kokkos::parallel_reduce(
      "anderson::count", innerPolicy(e, g),
      KOKKOS_LAMBDA(const Index x, const Index y, const Index z, Index& acc) {
        const Index i = x + y * ex + z * exy;
        if (buf(i) != prev(i))
          ++acc;
      },
      Kokkos::Sum<Index, MemSpace>(out));
}

/// The window columns the mix reads, oldest first.
struct MixColumns {
  View<const double> dG[kMaxWindow];
  View<const double> dR[kMaxWindow];
  double gamma[kMaxWindow] = {};
  int count = 0;
};

/// MIX (design §4.3 step 1, §4.5): x = buf − ΔG γ − (1−β)(Rprev − ΔR γ); buf = X = x. Padded.
inline void mix(View<double> buf, View<double> xOut, View<const double> rPrev,
                const MixColumns& cols, double beta) {
  const double damp = 1.0 - beta;
  Kokkos::parallel_for(
      "anderson::mix", Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<Index>>(0, buf.extent(0)),
      KOKKOS_LAMBDA(const Index i) {
        double c = 0.0, d = 0.0;
        for (int j = 0; j < cols.count; ++j) {
          c += cols.gamma[j] * cols.dG[j](i);
          d += cols.gamma[j] * cols.dR[j](i);
        }
        const double x = buf(i) - c - damp * (rPrev(i) - d);
        buf(i) = x;
        xOut(i) = x;
      });
}

/// DIFF (step 3): r = buf − X; dR_s = r − Rprev; dG_s = buf − Gprev; X = r. Padded.
inline void diff(View<const double> buf, View<double> x, View<const double> rPrev,
                 View<const double> gPrev, View<double> dRs, View<double> dGs) {
  Kokkos::parallel_for(
      "anderson::diff", Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<Index>>(0, buf.extent(0)),
      KOKKOS_LAMBDA(const Index i) {
        const double r = buf(i) - x(i);
        dRs(i) = r - rPrev(i);
        dGs(i) = buf(i) - gPrev(i);
        x(i) = r;
      });
}

/// R (step 3, first evaluation): X = buf − X. Padded.
inline void residual(View<const double> buf, View<double> x) {
  Kokkos::parallel_for(
      "anderson::residual",
      Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<Index>>(0, buf.extent(0)),
      KOKKOS_LAMBDA(const Index i) { x(i) = buf(i) - x(i); });
}

/// Pass 2, residual part: {Σ X², Σ buf²} over the inner entries of one Velocity field, into `out`
/// (no host synchronisation).
inline void selfSums(View<const double> x, View<const double> buf, const IVec<3>& e, int g,
                     const DeviceScalar<SumN<2>>& out) {
  const Index ex = e[0], exy = e[0] * e[1];
  Kokkos::parallel_reduce(
      "anderson::self_sums", innerPolicy(e, g),
      KOKKOS_LAMBDA(const Index xi, const Index y, const Index z, SumN<2>& acc) {
        const Index i = xi + y * ex + z * exy;
        acc.v[0] += x(i) * x(i);
        acc.v[1] += buf(i) * buf(i);
      },
      Kokkos::Sum<SumN<2>, MemSpace>(out));
}

/// Pass 2, one window column j against the new column s, over the inner entries of one Velocity
/// field: {⟨dR_s,dR_j⟩, ⟨dR_j,X⟩}, into `out` (no host synchronisation). Reads dR_s, dR_j and X
/// only (design §6.1, rev 2: no reduction reads a dG column; dG is read by MIX alone).
inline void columnSums(View<const double> dRs, View<const double> dRj, View<const double> x,
                       const IVec<3>& e, int g, const DeviceScalar<SumN<2>>& out) {
  const Index ex = e[0], exy = e[0] * e[1];
  Kokkos::parallel_reduce(
      "anderson::column_sums", innerPolicy(e, g),
      KOKKOS_LAMBDA(const Index xi, const Index y, const Index z, SumN<2>& acc) {
        const Index i = xi + y * ex + z * exy;
        const double rs = dRs(i), rj = dRj(i);
        acc.v[0] += rs * rj;
        acc.v[1] += rj * x(i);
      },
      Kokkos::Sum<SumN<2>, MemSpace>(out));
}

// ---- host linear algebra (deterministic, n ≤ kMaxWindow) ----------------------------------------

using Mat = double[kMaxWindow][kMaxWindow];

/// Cyclic Jacobi eigendecomposition of the symmetric n×n `a` (overwritten), fixed sweep order
/// p < q row-major; stops when the off-diagonal Frobenius norm is ≤ 1e-15·‖A‖_F or after 50 sweeps
/// (design §4.4 step 3). Eigenvalue i is lam[i], its eigenvector the column v[·][i].
inline void jacobiEigen(int n, Mat& a, Mat& v, double (&lam)[kMaxWindow]) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      v[i][j] = (i == j) ? 1.0 : 0.0;
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0.0, all = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        all += a[i][j] * a[i][j];
        if (i != j)
          off += a[i][j] * a[i][j];
      }
    if (std::sqrt(off) <= 1e-15 * std::sqrt(all))
      break;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        if (a[p][q] == 0.0)
          continue;
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t =
            (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < n; ++k) {  // A ← A·J
          const double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {  // A ← Jᵀ·A
          const double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {  // V ← V·J
          const double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
  }
  for (int i = 0; i < n; ++i)
    lam[i] = a[i][i];
}

/// The Jacobi-scaled eigendecomposition of a Gram block G: A = D⁻¹ G D⁻¹, D = diag(G)^{1/2}
/// (design §4.4 steps 1–3). A zero D_j gets D⁻¹_j = 0 (its row and column of A vanish, so it is
/// a zero eigenvalue the truncation discards).
struct ScaledEig {
  int n = 0;
  double dinv[kMaxWindow] = {};
  double lam[kMaxWindow] = {};
  Mat v = {};
  double lamMin = 0.0, lamMax = 0.0;
};

inline ScaledEig scaledEigen(int n, const Mat& gram) {
  ScaledEig r;
  r.n = n;
  for (int j = 0; j < n; ++j) {
    const double d = std::sqrt(gram[j][j]);
    r.dinv[j] = d > 0.0 ? 1.0 / d : 0.0;
  }
  Mat a = {};
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      a[i][j] = r.dinv[i] * gram[i][j] * r.dinv[j];
  jacobiEigen(n, a, r.v, r.lam);
  r.lamMin = r.lamMax = n > 0 ? r.lam[0] : 0.0;
  for (int i = 1; i < n; ++i) {
    r.lamMin = std::min(r.lamMin, r.lam[i]);
    r.lamMax = std::max(r.lamMax, r.lam[i]);
  }
  return r;
}

/// x = D⁻¹ Σ_i v_i (v_iᵀ D⁻¹ b)/λ_i over the eigenvalues λ_i ≥ condMin·λ_max (and > 0): the
/// truncated solve of design §4.4 step 5 (after the caller's column drops, nothing is truncated).
inline void truncatedSolve(const ScaledEig& se, const double* b, double condMin, double* x) {
  const int n = se.n;
  double bt[kMaxWindow], y[kMaxWindow] = {};
  for (int j = 0; j < n; ++j)
    bt[j] = se.dinv[j] * b[j];
  for (int i = 0; i < n; ++i) {
    if (!(se.lam[i] > 0.0) || se.lam[i] < condMin * se.lamMax)
      continue;
    double vb = 0.0;
    for (int l = 0; l < n; ++l)
      vb += se.v[l][i] * bt[l];
    const double w = vb / se.lam[i];
    for (int k = 0; k < n; ++k)
      y[k] += se.v[k][i] * w;
  }
  for (int k = 0; k < n; ++k)
    x[k] = se.dinv[k] * y[k];
}

}  // namespace detail::anderson

// ---------------------------------------------------------------------------------------------
// AndersonCore — type-II Anderson acceleration of the caller's map (design §4).
//
// One map evaluation is
//     core.prepare(accelerate);   // §4.3 steps 0–1: state check, lazy mix
//     <the caller's step>          // §4.3 step 2: reads and writes state.fields in place
//     core.complete(failed);       // §4.3 steps 3–7: column, reductions, decisions, coefficients
// and when the caller's step throws, `if (!core.stepFailed(e.what())) throw;` restores the last
// map output and disables acceleration if the failing input was a mixed one (else the failure is
// the plain march's own and the caller rethrows). A caller-side parameter change (flow: Δt, ρ, μ,
// F) is reported with invalidate() before prepare().
// ---------------------------------------------------------------------------------------------
class AndersonCore {
 public:
  // ---- fixed constants (design §4.1) ----
  static constexpr int kMaxWindow = detail::anderson::kMaxWindow;
  static constexpr int kEngageDecreases = 2;
  static constexpr double kRestartGrowth = 4.0;
  static constexpr int kMaxRestarts = 5;
  static constexpr double kNoiseFloor = 1e-10;
  static constexpr double kCondMin = 1e-12;
  // Rev 2: no instability guard — no Ritz estimate, no status "unstable". A Ritz value of the
  // caller's non-normal map in the velocity-only metric is no stability test; evidence that the
  // plain map is unstable comes from the caller's plain steps (design "Revision 2", D5).

  enum class Status : int { Active = 0, Disabled = 1 };

  /// Allocates the history (design §6.3: (2m+3)·n_s·8·n_pad bytes). Collective over state.comm
  /// (one 2-double sum {invalid descriptor, failed allocation}, so that every rank throws when one
  /// rank's descriptor is inconsistent or one rank's allocation fails). Throws
  /// std::invalid_argument (on every rank) for an inconsistent descriptor and std::runtime_error
  /// (on every rank) when the history does not fit.
  explicit AndersonCore(AndersonState state, int window = 5, double mixing = 1.0)
      : st_(std::move(state)), m_(window), beta_(mixing) {
    double failed[2] = {0.0, 0.0};  // {invalid descriptor, failed allocation}, summed over ranks
    std::string invalid;
    try {
      validate();
    } catch (const std::invalid_argument& e) {
      invalid = e.what();
      failed[0] = 1.0;
    }
    const std::size_t ns = st_.fields.size();
    const Index nPad = invalid.empty() ? numPadded() : 0;
    if (invalid.empty()) {
      try {
        auto alloc = [&](std::vector<View<double>>& set, const char* label) {
          set.resize(ns);
          for (std::size_t f = 0; f < ns; ++f)
            set[f] = View<double>(
                Kokkos::view_alloc(std::string("anderson::") + label, Kokkos::WithoutInitializing),
                nPad);
        };
        alloc(x_, "X");
        alloc(rPrev_, "Rprev");
        alloc(gPrev_, "Gprev");
        for (int s = 0; s < m_; ++s) {
          alloc(dR_[s], "dR");
          alloc(dG_[s], "dG");
        }
        allocReductionScratch();
      } catch (const std::exception&) {
        failed[1] = 1.0;
      }
    }
    sumAll(failed, 2);
    if (failed[0] > 0.0)
      throw std::invalid_argument(invalid.empty()
                                      ? std::string("AndersonCore: the state descriptor is "
                                                    "inconsistent on another rank")
                                      : invalid);
    if (failed[1] > 0.0) {
      char msg[512];
      std::snprintf(msg, sizeof msg,
                    "AndersonCore: window %d needs %zu bytes ((2m+3)*n_s*8*n_pad, n_s = %zu, "
                    "n_pad = %lld per rank); window 2 needs %zu; run on more ranks or without "
                    "acceleration",
                    m_, memoryBytesFor(m_, ns, nPad), ns, static_cast<long long>(nPad),
                    memoryBytesFor(2, ns, nPad));
      throw std::runtime_error(msg);
    }
  }

  AndersonCore(const AndersonCore&) = delete;
  AndersonCore& operator=(const AndersonCore&) = delete;
  AndersonCore(AndersonCore&&) = default;
  AndersonCore& operator=(AndersonCore&&) = default;

  /// The §6.3 memory formula: (2m + 3)·n_s·8·n_pad bytes.
  static std::size_t memoryBytesFor(int window, std::size_t numFields, Index nPad) {
    return static_cast<std::size_t>(2 * window + 3) * numFields * sizeof(double) *
           static_cast<std::size_t>(nPad);
  }

  // ---- one map evaluation -------------------------------------------------------------------

  /// §4.3 steps 0–1. Detects an external write of the state (inner entries only), then either
  /// applies the pending mix (accelerate, pending, active) or records the input. Returns whether
  /// the state now holds a mixed iterate. Collective (one 1-double sum) while active; one host
  /// synchronisation (the COUNT read-back).
  bool prepare(bool accelerate) {
    if (prepared_)
      throw std::logic_error("AndersonCore::prepare: called twice without complete()");
    prepared_ = true;
    mixed_ = false;
    if (status_ != Status::Active)
      return false;
    if (havePrev_) {
      const std::size_t ns = st_.fields.size();
      for (std::size_t f = 0; f < ns; ++f)
        detail::anderson::countChanged(st_.fields[f], gPrev_[f], st_.extent, st_.ghost,
                                       Kokkos::subview(countDev_, f));
      Kokkos::deep_copy(ExecSpace(), countHost_, countDev_);
      ExecSpace().fence("anderson::count");  // the one host synchronisation of prepare()
      double changed = 0.0;
      for (std::size_t f = 0; f < ns; ++f)
        changed += static_cast<double>(countHost_(f));
      sumAll(&changed, 1);
      if (changed > 0.0)
        invalidate();
    }
    mixed_ = accelerate && pending_ && status_ == Status::Active;
    if (mixed_) {
      for (std::size_t f = 0; f < st_.fields.size(); ++f) {
        detail::anderson::MixColumns cols;
        cols.count = mk_;
        for (int k = 0; k < mk_; ++k) {
          cols.dG[k] = dG_[order_[k]][f];
          cols.dR[k] = dR_[order_[k]][f];
          cols.gamma[k] = gamma_[k];
        }
        detail::anderson::mix(st_.fields[f], x_[f], rPrev_[f], cols, beta_);
      }
    } else {
      for (std::size_t f = 0; f < st_.fields.size(); ++f)
        Kokkos::deep_copy(ExecSpace(), x_[f], st_.fields[f]);  // stream-ordered, no fence
    }
    pending_ = false;
    return mixed_;
  }

  /// §4.3 step 2, the caller's step threw. Returns true when the failure was absorbed: the input
  /// was a mixed iterate, so the state is restored to the last map output, acceleration is
  /// disabled and the caller continues (plain). Returns false when the caller must rethrow.
  bool stepFailed(const std::string& what) {
    prepared_ = false;
    if (!mixed_ || status_ != Status::Active)
      return false;
    restoreLastOutput();
    status_ = Status::Disabled;
    reasonCode_ = Reason::StepFailed;
    reason_ = "step failed at an accelerated iterate: " + what;
    reset();
    if (st_.comm.rank == 0 && !noticed_) {
      std::fprintf(stderr, "AndersonCore: %s; acceleration disabled\n", reason_.c_str());
      noticed_ = true;
    }
    return true;
  }

  /// §4.3 steps 3–7, after the caller's step. `pressureSolveFailed` must be rank-consistent.
  /// Collective while active: a (2·columns + 2)-double sum and a broadcast of rank 0's decisions;
  /// one host synchronisation (the pass-2 read-back). The device copies are stream-ordered.
  void complete(bool pressureSolveFailed = false) {
    if (!prepared_)
      throw std::logic_error("AndersonCore::complete: called without prepare()");
    prepared_ = false;
    if (status_ != Status::Active)
      return;
    namespace da = detail::anderson;
    const std::size_t ns = st_.fields.size();
    const IVec<3>& e = st_.extent;
    const int g = st_.ghost;

    // --- 3. provisional new column (Rprev / Gprev untouched) ---
    int s = -1;
    int cols[kMaxWindow];  // the window after acceptance, oldest first, s last
    int nCols = 0;
    if (havePrev_) {
      if (mk_ < m_) {
        for (int c = 0; c < m_ && s < 0; ++c)
          if (std::find(order_, order_ + mk_, c) == order_ + mk_)
            s = c;
        for (int k = 0; k < mk_; ++k)
          cols[nCols++] = order_[k];
      } else {
        s = order_[0];
        for (int k = 1; k < mk_; ++k)
          cols[nCols++] = order_[k];
      }
      cols[nCols++] = s;
      for (std::size_t f = 0; f < ns; ++f)
        da::diff(st_.fields[f], x_[f], rPrev_[f], gPrev_[f], dR_[s][f], dG_[s][f]);
    } else {
      for (std::size_t f = 0; f < ns; ++f)
        da::residual(st_.fields[f], x_[f]);
    }

    // --- 4. reductions (§6.1; rev 1: Velocity fields only, no pass 1; rev 2: RR and b only) ---
    // Every reduction lands in its own device slot (velocity field v, window position k), so the
    // (nCols + 1)·n_v reductions queue without a host synchronisation and are read back behind ONE
    // fence. The per-field partials are added on the host in field order.
    for (std::size_t f = 0, v = 0; f < ns; ++f) {
      if (st_.roles[f] != AndersonRole::Velocity)
        continue;
      da::selfSums(x_[f], st_.fields[f], e, g, Kokkos::subview(selfDev_, v));
      for (int k = 0; k < nCols; ++k) {
        const int j = cols[k];
        da::columnSums(dR_[s][f], dR_[j][f], x_[f], e, g,
                       Kokkos::subview(colDev_, v * kMaxWindow + k));
      }
      ++v;
    }
    Kokkos::deep_copy(ExecSpace(), selfHost_, selfDev_);
    if (nCols > 0)
      Kokkos::deep_copy(ExecSpace(), colHost_, colDev_);
    ExecSpace().fence("anderson::pass2");  // the one host synchronisation of complete()
    std::vector<double> packet(static_cast<std::size_t>(2 * nCols + 2), 0.0);
    for (std::size_t v = 0; v < selfHost_.extent(0); ++v) {
      const da::SumN<2>& ss = selfHost_(v);
      packet[0] += ss.v[0];
      packet[1] += ss.v[1];
      for (int k = 0; k < nCols; ++k) {
        const da::SumN<2>& cs = colHost_(v * kMaxWindow + k);
        for (int q = 0; q < 2; ++q)
          packet[static_cast<std::size_t>(2 + 2 * k + q)] += cs.v[q];
      }
    }
    sumAll(packet.data(), static_cast<int>(packet.size()));

    // --- 5. host decisions (every rank; rank 0's are broadcast and win) ---
    const double xx = packet[0], uu = packet[1];
    const double rho = std::sqrt(xx);
    if (xx == 0.0)
      residual_ = 0.0;
    else if (uu == 0.0)
      residual_ = std::numeric_limits<double>::infinity();
    else
      residual_ = std::sqrt(xx / uu);
    if (!std::isfinite(rho) || (mixed_ && pressureSolveFailed)) {
      if (mixed_) {
        restoreLastOutput();
        reasonCode_ = Reason::NonFiniteMixed;
      } else {
        reasonCode_ = Reason::NonFinitePlain;
      }
      reason_ = reasonText(reasonCode_);
      status_ = Status::Disabled;
      reset();
      havePrev_ = false;
      return;
    }
    if (havePrev_) {
      for (int k = 0; k < nCols; ++k) {
        const int j = cols[k];
        const double* v = &packet[static_cast<std::size_t>(2 + 2 * k)];
        rr_[s][j] = rr_[j][s] = v[0];
        b_[j] = v[1];
      }
      for (int k = 0; k < nCols; ++k)
        order_[k] = cols[k];
      mk_ = nCols;
    }
    if (mixed_ && residual_ >= kNoiseFloor && rho > kRestartGrowth * rhoMin_) {
      reset();
      ++numRestarts_;
      rhoMin_ = rho;
      if (numRestarts_ >= kMaxRestarts) {
        status_ = Status::Disabled;
        reasonCode_ = Reason::TooManyRestarts;
        reason_ = reasonText(reasonCode_);
      }
    }
    decCount_ = (rho < rhoPrev_) ? decCount_ + 1 : 0;
    if (decCount_ >= kEngageDecreases)
      engaged_ = true;
    rhoPrev_ = rho;
    rhoMin_ = std::min(rhoMin_, rho);

    // --- 6. commit on the device ---
    std::swap(rPrev_, x_);
    for (std::size_t f = 0; f < ns; ++f)
      Kokkos::deep_copy(ExecSpace(), gPrev_[f], st_.fields[f]);  // stream-ordered, no fence
    havePrev_ = true;

    // --- 7. next coefficients (rev 2: no Ritz block) ---
    if (status_ == Status::Active && engaged_ && mk_ >= 1) {
      // A column with a zero norm (D_j = 0, §4.4 step 1) carries nothing: drop it first. Under the
      // velocity metric that is a column in which only Carried fields moved.
      for (int k = 0; k < mk_;) {
        if (rr_[order_[k]][order_[k]] > 0.0) {
          ++k;
          continue;
        }
        dropColumn(k);
      }
      while (mk_ > 1 && condScaled() < kCondMin)
        dropColumn(0);
      if (mk_ >= 1) {
        solveTruncated();
        pending_ = true;
      }
    }
    broadcastDecisions();
  }

  // ---- control ------------------------------------------------------------------------------

  /// Clears the history and the counters (design §4.2); keeps the previous evaluation, so the next
  /// call forms a column at once.
  void reset() {
    mk_ = 0;
    pending_ = engaged_ = false;
    decCount_ = 0;
    rhoMin_ = rhoPrev_ = std::numeric_limits<double>::infinity();
  }

  /// The caller's map changed (flow: its parameter signature) or the state was written: reset()
  /// and forget the previous evaluation; counts in numResets().
  void invalidate() {
    reset();
    havePrev_ = false;
    ++numResets_;
  }

  /// Stops acceleration for good: prepare/complete become no-ops around a plain step.
  void disable() {
    if (status_ == Status::Active) {
      status_ = Status::Disabled;
      reasonCode_ = Reason::DisabledByCaller;
      reason_ = reasonText(reasonCode_);
    }
    pending_ = false;
  }

  // ---- status ---------------------------------------------------------------------------------

  Status status() const { return status_; }
  /// "active" | "disabled".
  const char* statusName() const { return status_ == Status::Active ? "active" : "disabled"; }
  const std::string& reason() const { return reason_; }
  /// Relative velocity residual of the last evaluation (design §3.2); +inf before the first.
  double residual() const { return residual_; }
  int numRestarts() const { return numRestarts_; }
  int numResets() const { return numResets_; }
  /// Window columns in use (mk).
  int numColumns() const { return mk_; }
  int window() const { return m_; }
  double mixing() const { return beta_; }
  bool engaged() const { return engaged_; }
  /// A mix is pending for the next accelerated call.
  bool pending() const { return pending_; }
  /// The pending coefficients γ, oldest column first (numColumns() of them).
  std::vector<double> gamma() const { return std::vector<double>(gamma_, gamma_ + mk_); }
  /// Bytes held by the history, X, Rprev and Gprev: equals memoryBytesFor(window(), n_s, n_pad).
  std::size_t memoryBytes() const {
    std::size_t b = 0;
    auto add = [&b](const std::vector<View<double>>& set) {
      for (const auto& v : set)
        b += v.span() * sizeof(double);
    };
    add(x_);
    add(rPrev_);
    add(gPrev_);
    for (int s = 0; s < m_; ++s) {
      add(dR_[s]);
      add(dG_[s]);
    }
    return b;
  }
  const AndersonState& state() const { return st_; }

 private:
  enum class Reason : int {
    None = 0,
    StepFailed,
    NonFiniteMixed,
    NonFinitePlain,
    TooManyRestarts,
    DisabledByCaller,
  };

  /// Rank 0's decisions (design §4.3 step 7, §6.1): identical γ, status, mk and slot order on every
  /// rank are a correctness requirement — each rank mixes its own ghosts.
  struct DecisionPacket {
    int status;
    int reason;
    int mk;
    int pending;
    int numRestarts;
    int order[kMaxWindow];
    double gamma[kMaxWindow];
    double residual;
  };

  static std::string reasonText(Reason r) {
    switch (r) {
      case Reason::None:
        return "";
      case Reason::StepFailed:
        return "step failed at an accelerated iterate";
      case Reason::NonFiniteMixed:
        return "non-finite residual / failed pressure solve at an accelerated iterate";
      case Reason::NonFinitePlain:
        return "non-finite residual on a plain step";
      case Reason::TooManyRestarts:
        return "too many restarts";
      case Reason::DisabledByCaller:
        return "disabled by the caller";
    }
    return "";
  }

  void validate() {
    const std::size_t ns = st_.fields.size();
    if (ns == 0)
      throw std::invalid_argument("AndersonCore: the state has no fields");
    if (st_.roles.size() != ns)
      throw std::invalid_argument("AndersonCore: one role per state field is required");
    if (m_ < 1 || m_ > kMaxWindow)
      throw std::invalid_argument("AndersonCore: window must be in [1, 8]");
    if (!(beta_ > 0.0 && beta_ <= 1.0))
      throw std::invalid_argument("AndersonCore: mixing must be in (0, 1]");
    if (st_.ghost < 0 || st_.extent[0] <= 2 * st_.ghost || st_.extent[1] <= 2 * st_.ghost ||
        st_.extent[2] <= 2 * st_.ghost)
      throw std::invalid_argument("AndersonCore: the padded extent has no inner entries");
    const Index nPad = numPadded();
    for (std::size_t f = 0; f < ns; ++f) {
      if (static_cast<Index>(st_.fields[f].extent(0)) != nPad)
        throw std::invalid_argument("AndersonCore: a state field is not the full padded box");
      if (st_.roles[f] != AndersonRole::Velocity && st_.roles[f] != AndersonRole::Carried)
        throw std::invalid_argument("AndersonCore: a role must be Velocity or Carried");
    }
  }

  Index numPadded() const { return st_.extent[0] * st_.extent[1] * st_.extent[2]; }

  void sumAll(double* data, int count) const {
    if (st_.comm.sumAll)
      st_.comm.sumAll(data, count);
  }

  void restoreLastOutput() {
    for (std::size_t f = 0; f < st_.fields.size(); ++f)
      Kokkos::deep_copy(ExecSpace(), st_.fields[f], gPrev_[f]);  // stream-ordered, no fence
  }

  /// The device slots of the per-step reductions and their host mirrors (a few hundred bytes; not
  /// part of the §6.3 history and not in memoryBytes()).
  void allocReductionScratch() {
    namespace da = detail::anderson;
    std::size_t nv = 0;
    for (const AndersonRole r : st_.roles)
      nv += r == AndersonRole::Velocity ? 1 : 0;
    countDev_ = Kokkos::View<Index*, MemSpace>("anderson::count", st_.fields.size());
    selfDev_ = Kokkos::View<da::SumN<2>*, MemSpace>("anderson::self_sums", nv);
    colDev_ = Kokkos::View<da::SumN<2>*, MemSpace>("anderson::column_sums", nv * kMaxWindow);
    countHost_ = Kokkos::create_mirror_view(countDev_);
    selfHost_ = Kokkos::create_mirror_view(selfDev_);
    colHost_ = Kokkos::create_mirror_view(colDev_);
  }

  void dropColumn(int k) {
    for (int q = k; q + 1 < mk_; ++q)
      order_[q] = order_[q + 1];
    --mk_;
  }

  void windowBlock(const double (&full)[kMaxWindow][kMaxWindow], detail::anderson::Mat& out) const {
    for (int a = 0; a < mk_; ++a)
      for (int c = 0; c < mk_; ++c)
        out[a][c] = full[order_[a]][order_[c]];
  }

  /// λ_min/λ_max of the Jacobi-scaled window Gram (design §4.4 step 4).
  double condScaled() const {
    detail::anderson::Mat a = {};
    windowBlock(rr_, a);
    const auto se = detail::anderson::scaledEigen(mk_, a);
    return se.lamMin / se.lamMax;
  }

  /// γ = argmin ‖r_k − ΔR γ‖ by the scaled normal equations (design §4.4).
  void solveTruncated() {
    detail::anderson::Mat a = {};
    windowBlock(rr_, a);
    double b[kMaxWindow];
    for (int k = 0; k < mk_; ++k)
      b[k] = b_[order_[k]];
    const auto se = detail::anderson::scaledEigen(mk_, a);
    detail::anderson::truncatedSolve(se, b, kCondMin, gamma_);
  }

  void broadcastDecisions() {
    if (!st_.comm.broadcast)
      return;
    DecisionPacket p{};
    p.status = static_cast<int>(status_);
    p.reason = static_cast<int>(reasonCode_);
    p.mk = mk_;
    p.pending = pending_ ? 1 : 0;
    p.numRestarts = numRestarts_;
    for (int k = 0; k < kMaxWindow; ++k) {
      p.order[k] = k < mk_ ? order_[k] : -1;
      p.gamma[k] = k < mk_ ? gamma_[k] : 0.0;
    }
    p.residual = residual_;
    st_.comm.broadcast(&p, sizeof p);
    if (st_.comm.rank == 0)
      return;
    const Reason oldReason = reasonCode_;
    status_ = static_cast<Status>(p.status);
    reasonCode_ = static_cast<Reason>(p.reason);
    mk_ = p.mk;
    pending_ = p.pending != 0;
    numRestarts_ = p.numRestarts;
    for (int k = 0; k < kMaxWindow; ++k) {
      order_[k] = p.order[k] >= 0 ? p.order[k] : 0;
      gamma_[k] = p.gamma[k];
    }
    residual_ = p.residual;
    if (reasonCode_ != oldReason)
      reason_ = reasonText(reasonCode_);
  }

  AndersonState st_;
  int m_;
  double beta_;

  // device (design §4.2): one view per state field per vector
  std::vector<View<double>> x_, rPrev_, gPrev_;
  std::vector<View<double>> dR_[kMaxWindow], dG_[kMaxWindow];
  // per-step reduction slots: COUNT by field; pass 2 by velocity field (· kMaxWindow + position)
  Kokkos::View<Index*, MemSpace> countDev_;
  Kokkos::View<detail::anderson::SumN<2>*, MemSpace> selfDev_;
  Kokkos::View<detail::anderson::SumN<2>*, MemSpace> colDev_;
  typename Kokkos::View<Index*, MemSpace>::host_mirror_type countHost_;
  typename Kokkos::View<detail::anderson::SumN<2>*, MemSpace>::host_mirror_type selfHost_;
  typename Kokkos::View<detail::anderson::SumN<2>*, MemSpace>::host_mirror_type colHost_;

  // host
  int mk_ = 0;
  int order_[kMaxWindow] = {};              // slot indices, oldest → newest
  double rr_[kMaxWindow][kMaxWindow] = {};  // ⟨Δr_i, Δr_j⟩ by slot
  double b_[kMaxWindow] = {};               // ⟨Δr_j, r_k⟩ by slot
  double gamma_[kMaxWindow] = {};           // by window position, oldest first
  bool havePrev_ = false, pending_ = false, engaged_ = false;
  bool prepared_ = false, mixed_ = false, noticed_ = false;
  int decCount_ = 0, numRestarts_ = 0, numResets_ = 0;
  double rhoPrev_ = std::numeric_limits<double>::infinity();
  double rhoMin_ = std::numeric_limits<double>::infinity();
  Status status_ = Status::Active;
  Reason reasonCode_ = Reason::None;
  std::string reason_;
  double residual_ = std::numeric_limits<double>::infinity();
};

}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_ANDERSON_HPP

// core — PARALLEL, deterministic setup for the smoothed-aggregation GraphAMG (opt-in:
// AmgParams::aggregation = Aggregation::Mis2). Kokkos, host-OpenMP today, device-capable code.
//
// Design: peclet-papers work/M7/DESIGN_NOTE.md §14.5 (C1); results and the decisions taken on it in
// core/docs/graph_amg_parallel_setup.md. The host greedy setup (graph_amg.hpp) stays the default
// and the oracle; nothing here is reached unless a caller opts in.
//
// Determinism contract: every value written here is either integer logic, an order-independent
// max reduction, or a floating-point sum accumulated SEQUENTIALLY by one thread in a fixed order
// (the host oracle's order wherever the host has one). Aggregates and hierarchy values are
// therefore identical for every thread count on one backend.
//
// Pipeline per level (§14.5 C1):
//   1. nodal strength, the host test verbatim (‖A_IJ‖_F² ≥ θ²·√(‖A_II‖_F²·‖A_JJ‖_F²));
//   2. MIS-2 on the strong graph (hashed priorities, synchronous double-buffered rounds, ≤ 64);
//   3. aggregates: root + strong 1-hop; leftovers join the strong-adjacent aggregate with the
//      largest Σ‖A_IJ‖_F (ties → smaller root gid); still-unassigned nodes get MIS-2 on their
//      induced subgraph, then singletons; ids by an exclusive scan over the roots in node order;
//   4. P₀ as the host builds it; λ(D_F⁻¹A_F) by ∞-norm power iteration (no safety factor);
//   5. smoothed P, one thread per fine row (≤ 64 coarse columns, sorted);
//   6. R = Pᵀ (atomic counts, scan, atomic fill, per-row sort by fine index);
//   7. A_c = R(AP), two-phase Gustavson, one thread per output row, sequential accumulation in
//      increasing input index, sorted columns.
#ifndef PECLET_CORE_SOLVER_GRAPH_AMG_SETUP_HPP
#define PECLET_CORE_SOLVER_GRAPH_AMG_SETUP_HPP

#include <chrono>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/graph_amg.hpp"

namespace peclet::core::solver {

/// Device-resident operator in the split CSR layout of HostCsrOp: the diagonal separately, the
/// CSR (start / nbr / coef) holding only the OFF-diagonal entries.
struct DeviceCsr {
  Index n = 0;
  View<double> diag;  ///< size n
  View<Index> start;  ///< size n+1
  View<Index> nbr;    ///< off-diagonal columns
  View<double> coef;  ///< off-diagonal values
};

namespace amg_setup {

using DView = View<double>;
using IView = View<Index>;
using BView = View<std::uint8_t>;
using Policy = Kokkos::RangePolicy<ExecSpace>;
using Token =
    Kokkos::Experimental::UniqueToken<ExecSpace, Kokkos::Experimental::UniqueTokenScope::Global>;

#ifdef PECLET_CORE_AMGS_PROFILE
inline double gProf[8] = {};  // strength, mis2+agg, filter+lambdaF, P, R, AP, RAP, smootherLambda
inline double profNow() {
  Kokkos::fence();
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define AMGS_T(k, t0)             \
  do {                            \
    const double t1_ = profNow(); \
    gProf[k] += t1_ - t0;         \
    t0 = t1_;                     \
  } while (0)
#else
#define AMGS_T(k, t0) (void)0
#endif

inline constexpr int kMaxPCols = 64;  ///< coarse columns per smoothed-P row (§14.5 step 5)
inline constexpr int kMaxMisRounds = 64;
inline constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

/// splitmix64 of x (the host oracle's λ-seed hash: z = x + golden, then the mix).
KOKKOS_INLINE_FUNCTION std::uint64_t splitmix64(std::uint64_t x) {
  std::uint64_t z = x + kGolden;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/// MIS-2 priority p(I) = splitmix64(gid ⊕ golden·(level+1)); a bijection of gid per level.
KOKKOS_INLINE_FUNCTION std::uint64_t misPriority(Index gid, int level) {
  return splitmix64(static_cast<std::uint64_t>(gid) ^
                    (kGolden * static_cast<std::uint64_t>(level + 1)));
}

/// In-place sort of (key, val) pairs by key (keys unique or ties irrelevant): insertion sort for
/// short runs, Shell sort (Ciura gaps) above. Deterministic; runs inside one thread.
template <class K, class V>
KOKKOS_INLINE_FUNCTION void sortPairs(K* key, V* val, Index len) {
  constexpr Index gaps[8] = {701, 301, 132, 57, 23, 10, 4, 1};
  for (int g = (len > 24 ? 0 : 7); g < 8; ++g) {
    const Index gap = gaps[g];
    for (Index i = gap; i < len; ++i) {
      const K k = key[i];
      const V v = val[i];
      Index j = i;
      while (j >= gap && key[j - gap] > k) {
        key[j] = key[j - gap];
        val[j] = val[j - gap];
        j -= gap;
      }
      key[j] = k;
      val[j] = v;
    }
  }
}

/// offs(i) = Σ_{j<i} counts(j) for i ≤ n; returns the total.
inline Index scanCounts(const IView& counts, const IView& offs, Index n) {
  Index total = 0;
  Kokkos::parallel_scan(
      "peclet::core::amgs_scan", Policy(0, n),
      KOKKOS_LAMBDA(const Index i, Index& upd, const bool fin) {
        const Index c = counts(i);
        if (fin)
          offs(i) = upd;
        upd += c;
      },
      total);
  Kokkos::deep_copy(Kokkos::subview(offs, n), total);
  return total;
}

/// A per-thread dense marker/accumulator pool (Gustavson): one row of `width` per concurrent
/// thread, markers −1 between uses. Memory = concurrency × width; fine on host backends (on CUDA
/// a hashed accumulator would replace it — results note, open item).
struct AccumPool {
  Kokkos::View<Index**, MemSpace> mark;
  Kokkos::View<double**, MemSpace> acc;
  Token token;
  AccumPool(Index width, bool withAcc) : token(ExecSpace()) {
    const auto conc = static_cast<std::size_t>(token.size());
    mark = Kokkos::View<Index**, MemSpace>(
        Kokkos::view_alloc(std::string("amgs_mark"), Kokkos::WithoutInitializing), conc,
        static_cast<std::size_t>(width));
    Kokkos::deep_copy(mark, Index(-1));
    if (withAcc)
      acc = Kokkos::View<double**, MemSpace>("amgs_acc", conc, static_cast<std::size_t>(width));
  }
};

/// ∞-norm power iteration for ρ(D̃⁻¹Ã), Ã = (dd on the diagonal, the off-diagonals with keep(k)
/// set — all of them when keep is empty). Seed ±1 from splitmix64(gid of the DOF); v is
/// normalised by its ∞-norm (a max reduction ⇒ order-independent) and the estimate is ‖D̃⁻¹Ãv‖_∞
/// with ‖v‖_∞ = 1. No safety factor here (the Chebyshev smoother applies its own 1.1).
inline double powerLambdaInf(const DeviceCsr& A, const DView& dd, const DView& invdd,
                             const BView& keep, int iters) {
  const Index n = A.n;
  if (n == 0)
    return 1.0;
  DView v(Kokkos::view_alloc(std::string("amgs_pv"), Kokkos::WithoutInitializing), n);
  DView w(Kokkos::view_alloc(std::string("amgs_pw"), Kokkos::WithoutInitializing), n);
  Kokkos::parallel_for(
      "peclet::core::amgs_pseed", Policy(0, n), KOKKOS_LAMBDA(const Index i) {
        v(i) = (splitmix64(static_cast<std::uint64_t>(i)) & 1ULL) ? 1.0 : -1.0;
      });
  const bool useKeep = keep.extent(0) > 0;
  IView start = A.start, nbr = A.nbr;
  DView coef = A.coef;
  double lam = 1.0;
  for (int it = 0; it < iters; ++it) {
    double mx = 0.0;
    Kokkos::parallel_reduce(
        "peclet::core::amgs_papply", Policy(0, n),
        KOKKOS_LAMBDA(const Index i, double& m) {
          double s = dd(i) * v(i);
          for (Index k = start(i); k < start(i + 1); ++k)
            if (!useKeep || keep(k))
              s += coef(k) * v(nbr(k));
          const double y = invdd(i) * s;
          w(i) = y;
          const double a = Kokkos::fabs(y);
          if (a > m)
            m = a;
        },
        Kokkos::Max<double>(mx));
    if (!(mx > 0.0))
      break;
    lam = mx;
    Kokkos::parallel_for(
        "peclet::core::amgs_pnorm", Policy(0, n),
        KOKKOS_LAMBDA(const Index i) { v(i) = w(i) / mx; });
  }
  return (lam > 0.0) ? lam : 1.0;
}

/// The two estimates of one level in ONE pass over the matrix: ρ(D⁻¹A) (lamS: dd = diag, every
/// entry) and ρ(D_F⁻¹A_F) (lamF: dd = dF, the kept entries). Each is the arithmetic of
/// powerLambdaInf exactly (same seed, same per-element operations, same stopping rule), so the
/// values are bit-identical to two separate calls; only the matrix is read once instead of twice.
inline void powerLambdaInf2(const DeviceCsr& A, const DView& dS, const DView& iS, const DView& dF,
                            const DView& iF, const BView& keep, int iters, double& lamS,
                            double& lamF) {
  const Index n = A.n;
  lamS = lamF = 1.0;
  if (n == 0)
    return;
  auto alloc = [n](const char* nm) {
    return DView(Kokkos::view_alloc(std::string(nm), Kokkos::WithoutInitializing), n);
  };
  DView vS = alloc("amgs_pvS"), wS = alloc("amgs_pwS"), vF = alloc("amgs_pvF"),
        wF = alloc("amgs_pwF");
  Kokkos::parallel_for(
      "peclet::core::amgs_pseed2", Policy(0, n), KOKKOS_LAMBDA(const Index i) {
        const double sd = (splitmix64(static_cast<std::uint64_t>(i)) & 1ULL) ? 1.0 : -1.0;
        vS(i) = sd;
        vF(i) = sd;
      });
  IView start = A.start, nbr = A.nbr;
  DView coef = A.coef;
  bool runS = true, runF = true;
  for (int it = 0; it < iters && (runS || runF); ++it) {
    double mS = 0.0, mF = 0.0;
    const bool rS = runS, rF = runF;
    Kokkos::parallel_reduce(
        "peclet::core::amgs_papply2", Policy(0, n),
        KOKKOS_LAMBDA(const Index i, double& aS, double& aF) {
          double sS = dS(i) * vS(i), sF = dF(i) * vF(i);
          for (Index k = start(i); k < start(i + 1); ++k) {
            const Index j = nbr(k);
            sS += coef(k) * vS(j);
            if (keep(k))
              sF += coef(k) * vF(j);
          }
          if (rS) {
            const double y = iS(i) * sS;
            wS(i) = y;
            if (Kokkos::fabs(y) > aS)
              aS = Kokkos::fabs(y);
          }
          if (rF) {
            const double y = iF(i) * sF;
            wF(i) = y;
            if (Kokkos::fabs(y) > aF)
              aF = Kokkos::fabs(y);
          }
        },
        Kokkos::Max<double>(mS), Kokkos::Max<double>(mF));
    if (rS && !(mS > 0.0))
      runS = false;
    if (rF && !(mF > 0.0))
      runF = false;
    if (runS)
      lamS = mS;
    if (runF)
      lamF = mF;
    const bool uS = runS, uF = runF;
    Kokkos::parallel_for(
        "peclet::core::amgs_pnorm2", Policy(0, n), KOKKOS_LAMBDA(const Index i) {
          if (uS)
            vS(i) = wS(i) / mS;
          if (uF)
            vF(i) = wF(i) / mF;
        });
  }
  lamS = (lamS > 0.0) ? lamS : 1.0;
  lamF = (lamF > 0.0) ? lamF : 1.0;
}

/// 1/diag (0 where |diag| ≤ 1e-300), the host oracle's formula.
inline DView invDiagOf(const DView& d, Index n) {
  DView inv(Kokkos::view_alloc(std::string("amgs_invD"), Kokkos::WithoutInitializing), n);
  Kokkos::parallel_for(
      "peclet::core::amgs_invd", Policy(0, n),
      KOKKOS_LAMBDA(const Index i) { inv(i) = (Kokkos::fabs(d(i)) > 1e-300) ? 1.0 / d(i) : 0.0; });
  return inv;
}

/// Nodal strong graph (CSR over nodes, columns sorted) with the Frobenius norm ‖A_IJ‖_F per edge.
struct StrongGraph {
  Index nNodes = 0;
  IView start, col;
  DView fro;  ///< ‖A_IJ‖_F (not squared) per strong edge
};

KOKKOS_INLINE_FUNCTION bool isStrongEdge(const IView& sStart, const IView& sCol, Index I, Index J) {
  Index lo = sStart(I), hi = sStart(I + 1);
  while (lo < hi) {
    const Index mid = lo + (hi - lo) / 2;
    const Index c = sCol(mid);
    if (c == J)
      return true;
    if (c < J)
      lo = mid + 1;
    else
      hi = mid;
  }
  return false;
}

/// Step 1: nodal strength, the host test verbatim. Per-node block norms are accumulated in the
/// host's (row, entry) order through a dense marker, then sorted by J.
inline StrongGraph buildStrength(const DeviceCsr& A, int s, double theta) {
  const Index nNodes = A.n / s;
  const Index nnz = static_cast<Index>(A.nbr.extent(0));
  IView start = A.start, nbr = A.nbr;
  DView coef = A.coef, diag = A.diag;
  IView segJ(Kokkos::view_alloc(std::string("amgs_segJ"), Kokkos::WithoutInitializing), nnz);
  DView segV(Kokkos::view_alloc(std::string("amgs_segV"), Kokkos::WithoutInitializing), nnz);
  IView ucnt("amgs_ucnt", nNodes);
  DView ddiag("amgs_ddiag", nNodes);
  AccumPool pool(nNodes, false);
  auto mark = pool.mark;
  auto token = pool.token;
  Kokkos::parallel_for(
      "peclet::core::amgs_blocks", Policy(0, nNodes), KOKKOS_LAMBDA(const Index I) {
        const auto t = token.acquire();
        const Index seg = start(I * s);
        Index u = 0;
        double dd = 0.0;
        for (Index i = I * s; i < (I + 1) * s; ++i) {
          dd += diag(i) * diag(i);
          for (Index k = start(i); k < start(i + 1); ++k) {
            const Index J = nbr(k) / s;
            const double c = coef(k);
            if (J == I) {
              dd += c * c;
            } else {
              const Index m = mark(t, J);
              if (m < 0) {
                mark(t, J) = u;
                segJ(seg + u) = J;
                segV(seg + u) = 0.0 + c * c;
                ++u;
              } else {
                segV(seg + m) += c * c;
              }
            }
          }
        }
        for (Index m = 0; m < u; ++m)
          mark(t, segJ(seg + m)) = -1;
        token.release(t);
        sortPairs(&segJ(seg), &segV(seg), u);
        ucnt(I) = u;
        ddiag(I) = dd;
      });
  const double th2 = theta * theta;
  IView scnt("amgs_scnt", nNodes);
  Kokkos::parallel_for(
      "peclet::core::amgs_strong_cnt", Policy(0, nNodes), KOKKOS_LAMBDA(const Index I) {
        const Index seg = start(I * s);
        Index c = 0;
        for (Index m = 0; m < ucnt(I); ++m) {
          const Index J = segJ(seg + m);
          if (segV(seg + m) >= th2 * Kokkos::sqrt(ddiag(I) * ddiag(J)))
            ++c;
          else
            segJ(seg + m) = -1 - J;  // weak: marked for the fill pass
        }
        scnt(I) = c;
      });
  StrongGraph G;
  G.nNodes = nNodes;
  G.start = IView("amgs_sstart", nNodes + 1);
  const Index ns = scanCounts(scnt, G.start, nNodes);
  G.col = IView(Kokkos::view_alloc(std::string("amgs_scol"), Kokkos::WithoutInitializing), ns);
  G.fro = DView(Kokkos::view_alloc(std::string("amgs_sfro"), Kokkos::WithoutInitializing), ns);
  IView sStart = G.start, sCol = G.col;
  DView sFro = G.fro;
  Kokkos::parallel_for(
      "peclet::core::amgs_strong_fill", Policy(0, nNodes), KOKKOS_LAMBDA(const Index I) {
        const Index seg = start(I * s);
        Index at = sStart(I);
        for (Index m = 0; m < ucnt(I); ++m) {
          const Index J = segJ(seg + m);
          if (J >= 0) {
            sCol(at) = J;
            sFro(at) = Kokkos::sqrt(segV(seg + m));
            ++at;
          }
        }
      });
  return G;
}

// MIS-2 node states.
inline constexpr std::uint8_t kUndecided = 0, kRoot = 1, kExcluded = 2;

/// Step 2: MIS-2 over the strong graph restricted to `active` nodes (all when empty). Writes
/// state(I) ∈ {kRoot, kExcluded} for every active node (inactive nodes are left untouched).
/// Synchronous, double-buffered rounds: two passes for the 2-hop maximum of (undecided, p), roots
/// where it equals the node's own priority, two passes excluding everything within 2 hops of a new
/// root. After kMaxMisRounds, undecided nodes become roots.
inline void mis2(const StrongGraph& G, const BView& active, int level, const BView& state) {
  const Index nN = G.nNodes;
  const bool all = active.extent(0) == 0;
  IView sStart = G.start, sCol = G.col;
  Kokkos::View<std::uint64_t*, MemSpace> m1(
      Kokkos::view_alloc(std::string("amgs_m1"), Kokkos::WithoutInitializing), nN),
      m2(Kokkos::view_alloc(std::string("amgs_m2"), Kokkos::WithoutInitializing), nN);
  BView h1(Kokkos::view_alloc(std::string("amgs_h1"), Kokkos::WithoutInitializing), nN),
      h2(Kokkos::view_alloc(std::string("amgs_h2"), Kokkos::WithoutInitializing), nN),
      f1(Kokkos::view_alloc(std::string("amgs_f1"), Kokkos::WithoutInitializing), nN);
  Kokkos::parallel_for(
      "peclet::core::amgs_mis_init", Policy(0, nN), KOKKOS_LAMBDA(const Index I) {
        if (all || active(I))
          state(I) = kUndecided;
      });
  for (int round = 0; round < kMaxMisRounds; ++round) {
    // pass 1: 1-hop maximum of the undecided priorities
    Kokkos::parallel_for(
        "peclet::core::amgs_mis_max1", Policy(0, nN), KOKKOS_LAMBDA(const Index I) {
          if (!(all || active(I))) {
            h1(I) = 0;
            return;
          }
          bool has = state(I) == kUndecided;
          std::uint64_t best = has ? misPriority(I, level) : 0;
          for (Index k = sStart(I); k < sStart(I + 1); ++k) {
            const Index J = sCol(k);
            if (!(all || active(J)) || state(J) != kUndecided)
              continue;
            const std::uint64_t p = misPriority(J, level);
            if (!has || p > best) {
              best = p;
              has = true;
            }
          }
          h1(I) = has ? 1 : 0;
          m1(I) = best;
        });
    // pass 2: 2-hop maximum
    Kokkos::parallel_for(
        "peclet::core::amgs_mis_max2", Policy(0, nN), KOKKOS_LAMBDA(const Index I) {
          if (!(all || active(I))) {
            h2(I) = 0;
            return;
          }
          bool has = h1(I) != 0;
          std::uint64_t best = has ? m1(I) : 0;
          for (Index k = sStart(I); k < sStart(I + 1); ++k) {
            const Index J = sCol(k);
            if (!(all || active(J)) || !h1(J))
              continue;
            if (!has || m1(J) > best) {
              best = m1(J);
              has = true;
            }
          }
          h2(I) = has ? 1 : 0;
          m2(I) = best;
        });
    // new roots (h2 reused as the new-root flag) and pass 3: within 1 hop of a new root
    Kokkos::parallel_for(
        "peclet::core::amgs_mis_root", Policy(0, nN), KOKKOS_LAMBDA(const Index I) {
          const bool act = all || active(I);
          h1(I) =
              (act && state(I) == kUndecided && h2(I) && m2(I) == misPriority(I, level)) ? 1 : 0;
        });
    Kokkos::parallel_for(
        "peclet::core::amgs_mis_ex1", Policy(0, nN), KOKKOS_LAMBDA(const Index I) {
          if (!(all || active(I))) {
            f1(I) = 0;
            return;
          }
          std::uint8_t f = h1(I);
          for (Index k = sStart(I); k < sStart(I + 1) && !f; ++k) {
            const Index J = sCol(k);
            if ((all || active(J)) && h1(J))
              f = 1;
          }
          f1(I) = f;
        });
    // pass 4: within 2 hops; update the state (reads only f1 / h1 of others)
    Index undecided = 0;
    Kokkos::parallel_reduce(
        "peclet::core::amgs_mis_ex2", Policy(0, nN),
        KOKKOS_LAMBDA(const Index I, Index& cnt) {
          if (!(all || active(I)) || state(I) != kUndecided)
            return;
          if (h1(I)) {
            state(I) = kRoot;
            return;
          }
          bool f = f1(I) != 0;
          for (Index k = sStart(I); k < sStart(I + 1) && !f; ++k) {
            const Index J = sCol(k);
            if ((all || active(J)) && f1(J))
              f = true;
          }
          if (f)
            state(I) = kExcluded;
          else
            ++cnt;
        },
        undecided);
    if (undecided == 0)
      return;
  }
  Kokkos::parallel_for(
      "peclet::core::amgs_mis_force", Policy(0, nN), KOKKOS_LAMBDA(const Index I) {
        if ((all || active(I)) && state(I) == kUndecided)
          state(I) = kRoot;
      });
}

/// Step 3: aggregates. Returns the number of aggregates; agg(I) = aggregate id.
inline Index aggregate(const StrongGraph& G, int level, const IView& agg) {
  const Index nN = G.nNodes;
  IView sStart = G.start, sCol = G.col;
  DView sFro = G.fro;
  BView state("amgs_state", nN);
  mis2(G, BView(), level, state);
  IView root(Kokkos::view_alloc(std::string("amgs_root"), Kokkos::WithoutInitializing), nN);
  // roots + their strong 1-hop neighbours (pull: a non-root joins its strong root neighbour; a
  // node seeing several — only after forced roots — takes the largest ‖A_IJ‖_F, ties smaller gid)
  Kokkos::parallel_for(
      "peclet::core::amgs_agg_hop1", Policy(0, nN), KOKKOS_LAMBDA(const Index I) {
        if (state(I) == kRoot) {
          root(I) = I;
          return;
        }
        Index best = -1;
        double bw = -1.0;
        for (Index k = sStart(I); k < sStart(I + 1); ++k) {
          const Index J = sCol(k);
          if (state(J) != kRoot)
            continue;
          if (sFro(k) > bw || (sFro(k) == bw && J < best)) {
            bw = sFro(k);
            best = J;
          }
        }
        root(I) = best;
      });
  // leftovers: join the strong-adjacent aggregate with the largest Σ‖A_IJ‖_F (ties → smaller
  // root gid); reads the snapshot `root`, writes `root2`.
  IView root2(Kokkos::view_alloc(std::string("amgs_root2"), Kokkos::WithoutInitializing), nN);
  Index unassigned = 0;
  Kokkos::parallel_reduce(
      "peclet::core::amgs_agg_left", Policy(0, nN),
      KOKKOS_LAMBDA(const Index I, Index& cnt) {
        if (root(I) >= 0) {
          root2(I) = root(I);
          return;
        }
        Index best = -1;
        double bw = -1.0;
        for (Index k = sStart(I); k < sStart(I + 1); ++k) {
          const Index r = root(sCol(k));
          if (r < 0)
            continue;
          bool seen = false;  // evaluate each distinct aggregate once, at its first neighbour
          for (Index q = sStart(I); q < k && !seen; ++q)
            seen = root(sCol(q)) == r;
          if (seen)
            continue;
          double w = 0.0;
          for (Index q = k; q < sStart(I + 1); ++q)
            if (root(sCol(q)) == r)
              w += sFro(q);
          if (w > bw || (w == bw && r < best)) {
            bw = w;
            best = r;
          }
        }
        root2(I) = best;
        if (best < 0)
          ++cnt;
      },
      unassigned);
  if (unassigned > 0) {
    // still unassigned: MIS-2 on their induced strong subgraph, root + 1-hop there, singletons.
    BView active("amgs_active", nN), st2("amgs_state2", nN);
    Kokkos::parallel_for(
        "peclet::core::amgs_agg_act", Policy(0, nN),
        KOKKOS_LAMBDA(const Index I) { active(I) = root2(I) < 0 ? 1 : 0; });
    mis2(G, active, level, st2);
    Kokkos::parallel_for(
        "peclet::core::amgs_agg_sub", Policy(0, nN), KOKKOS_LAMBDA(const Index I) {
          if (!active(I))
            return;
          if (st2(I) == kRoot) {
            root2(I) = I;
            return;
          }
          Index best = -1;
          double bw = -1.0;
          for (Index k = sStart(I); k < sStart(I + 1); ++k) {
            const Index J = sCol(k);
            if (!active(J) || st2(J) != kRoot)
              continue;
            if (sFro(k) > bw || (sFro(k) == bw && J < best)) {
              bw = sFro(k);
              best = J;
            }
          }
          root2(I) = best >= 0 ? best : I;  // singleton
        });
  }
  // aggregate ids: exclusive scan over the roots in node order
  IView isRoot("amgs_isroot", nN), rid("amgs_rid", nN + 1);
  Kokkos::parallel_for(
      "peclet::core::amgs_agg_isroot", Policy(0, nN),
      KOKKOS_LAMBDA(const Index I) { isRoot(I) = root2(I) == I ? 1 : 0; });
  const Index nAgg = scanCounts(isRoot, rid, nN);
  Kokkos::parallel_for(
      "peclet::core::amgs_agg_id", Policy(0, nN),
      KOKKOS_LAMBDA(const Index I) { agg(I) = rid(root2(I)); });
  return nAgg;
}

/// One smoothed-P row (§14.5 step 5), the host oracle's accumulation order: P₀ term, the diagonal
/// term f·dF·P₀, then the kept off-diagonals in entry order. Returns the count after dropping
/// exact zeros (columns sorted), or −1 on more than kMaxPCols columns.
struct PRowBuilder {
  IView start, nbr, agg;
  DView coef, dF, invSqrtCnt;
  BView keep;
  int s;
  double sc;
  KOKKOS_INLINE_FUNCTION Index p0col(Index i) const { return agg(i / s) * s + i % s; }
  KOKKOS_INLINE_FUNCTION double p0val(Index i) const { return invSqrtCnt(agg(i / s)); }
  KOKKOS_INLINE_FUNCTION int build(Index i, Index* cols, double* vals) const {
    int m = 0;
    auto add = [&](Index c, double v) -> bool {
      for (int q = 0; q < m; ++q)
        if (cols[q] == c) {
          vals[q] += v;
          return true;
        }
      if (m == kMaxPCols)
        return false;
      cols[m] = c;
      vals[m] = 0.0 + v;
      ++m;
      return true;
    };
    const double d = dF(i);
    const double f = (Kokkos::fabs(d) > 1e-300) ? (-sc / d) : 0.0;
    bool ok = add(p0col(i), p0val(i));
    ok = ok && add(p0col(i), f * d * p0val(i));
    for (Index k = start(i); ok && k < start(i + 1); ++k) {
      if (!keep(k))
        continue;
      const Index j = nbr(k);
      ok = add(p0col(j), f * coef(k) * p0val(j));
    }
    if (!ok)
      return -1;
    sortPairs(cols, vals, static_cast<Index>(m));
    int out = 0;
    for (int q = 0; q < m; ++q)
      if (vals[q] != 0.0) {
        cols[out] = cols[q];
        vals[out] = vals[q];
        ++out;
      }
    return out;
  }
};

/// One coarsening step's output.
struct CoarseLevel {
  bool ok = false;
  Index nAgg = 0, nc = 0;
  double lmaxF = 1.0;  ///< λ(D_F⁻¹A_F) used for prolongator smoothing
  double lmaxS = 1.0;  ///< λ(D⁻¹A) of this (fine) level, the smoother bound (valid when ok)
  IView agg;           ///< node → aggregate id
  IView Pstart, Pcol;
  DView Pval;
  IView Rstart, Rcol;
  DView Rval;
  DeviceCsr Ac;
};

/// Steps 1–7 for one level. `level` salts the MIS-2 priorities.
inline CoarseLevel coarsen(const DeviceCsr& A, const DView& invDiag, const AmgParams& prm,
                           int level) {
  CoarseLevel out;
  const int s = prm.ndofPerNode;
  const Index n = A.n;
  const Index nN = n / s;
  const Index nnz = static_cast<Index>(A.nbr.extent(0));
  IView start = A.start, nbr = A.nbr;
  DView coef = A.coef, diag = A.diag;

  // 1–3: strength, MIS-2, aggregates
#ifdef PECLET_CORE_AMGS_PROFILE
  double tp = profNow();
#endif
  const StrongGraph G = buildStrength(A, s, prm.theta);
  AMGS_T(0, tp);
  out.agg = IView("amgs_agg", nN);
  const Index nAgg = aggregate(G, level, out.agg);
  out.nAgg = nAgg;
  AMGS_T(1, tp);
  if (nAgg <= 1 || nAgg == nN)
    return out;  // no useful coarsening (the host oracle's rule)
  IView agg = out.agg;

  // 4: P₀ normalisation; filtered diagonal + keep mask; λ(D_F⁻¹A_F)
  IView cnt("amgs_cnt", nAgg);
  Kokkos::parallel_for(
      "peclet::core::amgs_cnt", Policy(0, nN),
      KOKKOS_LAMBDA(const Index I) { Kokkos::atomic_add(&cnt(agg(I)), Index(1)); });
  DView invSqrtCnt(Kokkos::view_alloc(std::string("amgs_isc"), Kokkos::WithoutInitializing), nAgg);
  Kokkos::parallel_for(
      "peclet::core::amgs_isc", Policy(0, nAgg), KOKKOS_LAMBDA(const Index a) {
        invSqrtCnt(a) = 1.0 / Kokkos::sqrt(static_cast<double>(cnt(a)));
      });
  DView dF(Kokkos::view_alloc(std::string("amgs_dF"), Kokkos::WithoutInitializing), n);
  BView keep(Kokkos::view_alloc(std::string("amgs_keep"), Kokkos::WithoutInitializing), nnz);
  IView sStart = G.start, sCol = G.col;
  Kokkos::parallel_for(
      "peclet::core::amgs_filter", Policy(0, n), KOKKOS_LAMBDA(const Index i) {
        const Index I = i / s;
        double d = diag(i);
        for (Index k = start(i); k < start(i + 1); ++k) {
          const Index J = nbr(k) / s;
          const bool kp = (J == I) || isStrongEdge(sStart, sCol, I, J);
          keep(k) = kp ? 1 : 0;
          if (!kp)
            d += coef(k);  // lump the weak connection into the diagonal
        }
        dF(i) = d;
      });
  const DView invdF = invDiagOf(dF, n);
  powerLambdaInf2(A, diag, invDiag, dF, invdF, keep, prm.eigIters, out.lmaxS, out.lmaxF);
  const Index nc = nAgg * s;
  out.nc = nc;
  AMGS_T(2, tp);

  // 5: smoothed P — one build per row into a bounded scratch (≤ min(64, 1 + kept entries)
  // columns), then a scan and a compaction copy.
  PRowBuilder pb{start, nbr, agg, coef, dF, invSqrtCnt, keep, s, prm.smoothOmega / out.lmaxF};
  IView pb_bound("amgs_pbound", n), pb_off("amgs_pboff", n + 1);
  Kokkos::parallel_for(
      "peclet::core::amgs_p_bound", Policy(0, n), KOKKOS_LAMBDA(const Index i) {
        Index c = 1;
        for (Index k = start(i); k < start(i + 1); ++k)
          c += keep(k);
        pb_bound(i) = c < kMaxPCols ? c : kMaxPCols;
      });
  const Index pbTot = scanCounts(pb_bound, pb_off, n);
  IView tCol(Kokkos::view_alloc(std::string("amgs_pTc"), Kokkos::WithoutInitializing), pbTot);
  DView tVal(Kokkos::view_alloc(std::string("amgs_pTv"), Kokkos::WithoutInitializing), pbTot);
  IView pc("amgs_pc", n);
  Index overflow = 0;
  Kokkos::parallel_reduce(
      "peclet::core::amgs_p_build", Policy(0, n),
      KOKKOS_LAMBDA(const Index i, Index& bad) {
        Index cols[kMaxPCols];
        double vals[kMaxPCols];
        const int m = pb.build(i, cols, vals);
        if (m < 0) {
          ++bad;
          pc(i) = 0;
          return;
        }
        const Index o = pb_off(i);
        for (int q = 0; q < m; ++q) {
          tCol(o + q) = cols[q];
          tVal(o + q) = vals[q];
        }
        pc(i) = m;
      },
      overflow);
  if (overflow > 0)
    throw std::runtime_error("GraphAMG mis2 setup: a smoothed-P row exceeds " +
                             std::to_string(kMaxPCols) + " coarse columns");
  out.Pstart = IView("amgs_Pstart", n + 1);
  const Index pnnz = scanCounts(pc, out.Pstart, n);
  out.Pcol = IView(Kokkos::view_alloc(std::string("amgs_Pcol"), Kokkos::WithoutInitializing), pnnz);
  out.Pval = DView(Kokkos::view_alloc(std::string("amgs_Pval"), Kokkos::WithoutInitializing), pnnz);
  IView Pstart = out.Pstart, Pcol = out.Pcol;
  DView Pval = out.Pval;
  Kokkos::parallel_for(
      "peclet::core::amgs_p_compact", Policy(0, n), KOKKOS_LAMBDA(const Index i) {
        const Index o = Pstart(i), t = pb_off(i);
        for (Index q = 0; q < pc(i); ++q) {
          Pcol(o + q) = tCol(t + q);
          Pval(o + q) = tVal(t + q);
        }
      });

  AMGS_T(3, tp);
  // 6: R = Pᵀ — atomic counts, scan, atomic fill, per-row sort by fine index
  IView rc("amgs_rc", nc);
  Kokkos::parallel_for(
      "peclet::core::amgs_r_cnt", Policy(0, pnnz),
      KOKKOS_LAMBDA(const Index k) { Kokkos::atomic_add(&rc(Pcol(k)), Index(1)); });
  out.Rstart = IView("amgs_Rstart", nc + 1);
  scanCounts(rc, out.Rstart, nc);
  out.Rcol = IView(Kokkos::view_alloc(std::string("amgs_Rcol"), Kokkos::WithoutInitializing), pnnz);
  out.Rval = DView(Kokkos::view_alloc(std::string("amgs_Rval"), Kokkos::WithoutInitializing), pnnz);
  IView Rstart = out.Rstart, Rcol = out.Rcol;
  DView Rval = out.Rval;
  IView cur(Kokkos::view_alloc(std::string("amgs_cur"), Kokkos::WithoutInitializing), nc);
  Kokkos::deep_copy(cur, Kokkos::subview(Rstart, Kokkos::make_pair(Index(0), nc)));
  Kokkos::parallel_for(
      "peclet::core::amgs_r_fill", Policy(0, n), KOKKOS_LAMBDA(const Index i) {
        for (Index k = Pstart(i); k < Pstart(i + 1); ++k) {
          const Index at = Kokkos::atomic_fetch_add(&cur(Pcol(k)), Index(1));
          Rcol(at) = i;
          Rval(at) = Pval(k);
        }
      });
  Kokkos::parallel_for(
      "peclet::core::amgs_r_sort", Policy(0, nc), KOKKOS_LAMBDA(const Index c) {
        sortPairs(&Rcol(Rstart(c)), &Rval(Rstart(c)), Rstart(c + 1) - Rstart(c));
      });

  AMGS_T(4, tp);
  // 7a: AP = A·P, Gustavson with the host's order (diag·P_i first, then entries in order)
  AccumPool pool(nc, true);
  auto mark = pool.mark;
  auto acc = pool.acc;
  auto token = pool.token;
  IView apc("amgs_apc", n);
  Kokkos::parallel_for(
      "peclet::core::amgs_ap_cnt", Policy(0, n), KOKKOS_LAMBDA(const Index i) {
        const auto t = token.acquire();
        Index u = 0;
        auto touch = [&](Index c) {
          if (mark(t, c) < 0)
            mark(t, c) = u++;
        };
        for (Index k = Pstart(i); k < Pstart(i + 1); ++k)
          touch(Pcol(k));
        for (Index k = start(i); k < start(i + 1); ++k) {
          const Index j = nbr(k);
          for (Index kk = Pstart(j); kk < Pstart(j + 1); ++kk)
            touch(Pcol(kk));
        }
        // reset
        for (Index k = Pstart(i); k < Pstart(i + 1); ++k)
          mark(t, Pcol(k)) = -1;
        for (Index k = start(i); k < start(i + 1); ++k) {
          const Index j = nbr(k);
          for (Index kk = Pstart(j); kk < Pstart(j + 1); ++kk)
            mark(t, Pcol(kk)) = -1;
        }
        token.release(t);
        apc(i) = u;
      });
  IView APstart("amgs_APstart", n + 1);
  const Index apnnz = scanCounts(apc, APstart, n);
  IView APcol(Kokkos::view_alloc(std::string("amgs_APcol"), Kokkos::WithoutInitializing), apnnz);
  DView APval(Kokkos::view_alloc(std::string("amgs_APval"), Kokkos::WithoutInitializing), apnnz);
  Kokkos::parallel_for(
      "peclet::core::amgs_ap_fill", Policy(0, n), KOKKOS_LAMBDA(const Index i) {
        const auto t = token.acquire();
        const Index o = APstart(i);
        Index u = 0;
        auto addv = [&](Index c, double v) {
          Index m = mark(t, c);
          if (m < 0) {
            m = u++;
            mark(t, c) = m;
            APcol(o + m) = c;
            APval(o + m) = 0.0;
          }
          APval(o + m) += v;
        };
        const double di = diag(i);
        for (Index k = Pstart(i); k < Pstart(i + 1); ++k)
          addv(Pcol(k), di * Pval(k));
        for (Index k = start(i); k < start(i + 1); ++k) {
          const Index j = nbr(k);
          const double c = coef(k);
          for (Index kk = Pstart(j); kk < Pstart(j + 1); ++kk)
            addv(Pcol(kk), c * Pval(kk));
        }
        for (Index m = 0; m < u; ++m)
          mark(t, APcol(o + m)) = -1;
        token.release(t);
        sortPairs(&APcol(o), &APval(o), u);
      });

  AMGS_T(5, tp);
  // 7b: A_c = R·AP — one thread per coarse row, accumulated in increasing fine index (R rows are
  // sorted), structural count, fill, then exact zeros dropped from the off-diagonal (host rule).
  IView acc_cnt("amgs_accnt", nc);
  Kokkos::parallel_for(
      "peclet::core::amgs_rap_cnt", Policy(0, nc), KOKKOS_LAMBDA(const Index c1) {
        const auto t = token.acquire();
        Index u = 0;
        for (Index k = Rstart(c1); k < Rstart(c1 + 1); ++k) {
          const Index i = Rcol(k);
          for (Index e = APstart(i); e < APstart(i + 1); ++e) {
            const Index c2 = APcol(e);
            if (c2 != c1 && mark(t, c2) < 0)
              mark(t, c2) = u++;
          }
        }
        for (Index k = Rstart(c1); k < Rstart(c1 + 1); ++k) {
          const Index i = Rcol(k);
          for (Index e = APstart(i); e < APstart(i + 1); ++e)
            mark(t, APcol(e)) = -1;
        }
        token.release(t);
        acc_cnt(c1) = u;
      });
  IView Sstart("amgs_Acs", nc + 1);
  const Index snnz = scanCounts(acc_cnt, Sstart, nc);
  IView Scol(Kokkos::view_alloc(std::string("amgs_Acc"), Kokkos::WithoutInitializing), snnz);
  DView Sval(Kokkos::view_alloc(std::string("amgs_Acv"), Kokkos::WithoutInitializing), snnz);
  DView cdiag("amgs_Acd", nc);
  IView nzc("amgs_nzc", nc);
  Index zeros = 0;
  Kokkos::parallel_reduce(
      "peclet::core::amgs_rap_fill", Policy(0, nc),
      KOKKOS_LAMBDA(const Index c1, Index& zc) {
        const auto t = token.acquire();
        const Index o = Sstart(c1);
        Index u = 0;
        double dsum = 0.0;
        for (Index k = Rstart(c1); k < Rstart(c1 + 1); ++k) {
          const Index i = Rcol(k);
          const double p1 = Rval(k);
          for (Index e = APstart(i); e < APstart(i + 1); ++e) {
            const Index c2 = APcol(e);
            const double v = p1 * APval(e);
            if (c2 == c1) {
              dsum += v;
              continue;
            }
            Index m = mark(t, c2);
            if (m < 0) {
              m = u++;
              mark(t, c2) = m;
              Scol(o + m) = c2;
              Sval(o + m) = 0.0;
            }
            Sval(o + m) += v;
          }
        }
        for (Index m = 0; m < u; ++m)
          mark(t, Scol(o + m)) = -1;
        token.release(t);
        sortPairs(&Scol(o), &Sval(o), u);
        cdiag(c1) = dsum;
        Index nz = 0;
        for (Index m = 0; m < u; ++m)
          if (Sval(o + m) != 0.0)
            ++nz;
        nzc(c1) = nz;
        zc += u - nz;
      },
      zeros);
  out.Ac.n = nc;
  out.Ac.diag = cdiag;
  if (zeros == 0) {
    out.Ac.start = Sstart;
    out.Ac.nbr = Scol;
    out.Ac.coef = Sval;
  } else {
    out.Ac.start = IView("amgs_Acs2", nc + 1);
    const Index nnz2 = scanCounts(nzc, out.Ac.start, nc);
    out.Ac.nbr =
        IView(Kokkos::view_alloc(std::string("amgs_Acc2"), Kokkos::WithoutInitializing), nnz2);
    out.Ac.coef =
        DView(Kokkos::view_alloc(std::string("amgs_Acv2"), Kokkos::WithoutInitializing), nnz2);
    IView s2 = out.Ac.start, c2v = out.Ac.nbr;
    DView v2 = out.Ac.coef;
    Kokkos::parallel_for(
        "peclet::core::amgs_rap_compact", Policy(0, nc), KOKKOS_LAMBDA(const Index c1) {
          Index at = s2(c1);
          for (Index m = Sstart(c1); m < Sstart(c1 + 1); ++m)
            if (Sval(m) != 0.0) {
              c2v(at) = Scol(m);
              v2(at) = Sval(m);
              ++at;
            }
        });
  }
  AMGS_T(6, tp);
  out.ok = true;
  return out;
}

/// Dense symmetric eigendecomposition by cyclic Jacobi (host, row-major n×n, symmetrised as
/// (A + Aᵀ)/2): on return V holds the eigenvectors as columns, lam the eigenvalues.
inline void jacobiEigen(std::vector<double> a, int n, std::vector<double>& V,
                        std::vector<double>& lam) {
  auto at = [n](int i, int j) { return static_cast<std::size_t>(i) * n + j; };
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double m = 0.5 * (a[at(i, j)] + a[at(j, i)]);
      a[at(i, j)] = a[at(j, i)] = m;
    }
  V.assign(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i)
    V[at(i, i)] = 1.0;
  double norm2 = 0.0;
  for (double x : a)
    norm2 += x * x;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        off += 2.0 * a[at(i, j)] * a[at(i, j)];
    if (off <= 1e-32 * norm2 || off == 0.0)
      break;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        const double apq = a[at(p, q)];
        if (apq == 0.0)
          continue;
        const double theta = (a[at(q, q)] - a[at(p, p)]) / (2.0 * apq);
        const double t =
            (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
        for (int k = 0; k < n; ++k) {  // A ← A J (columns p, q)
          const double akp = a[at(k, p)], akq = a[at(k, q)];
          a[at(k, p)] = c * akp - sn * akq;
          a[at(k, q)] = sn * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {  // A ← Jᵀ A (rows p, q)
          const double apk = a[at(p, k)], aqk = a[at(q, k)];
          a[at(p, k)] = c * apk - sn * aqk;
          a[at(q, k)] = sn * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {  // V ← V J
          const double vkp = V[at(k, p)], vkq = V[at(k, q)];
          V[at(k, p)] = c * vkp - sn * vkq;
          V[at(k, q)] = sn * vkp + c * vkq;
        }
      }
  }
  lam.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    lam[static_cast<std::size_t>(i)] = a[at(i, i)];
}

}  // namespace amg_setup
}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_GRAPH_AMG_SETUP_HPP

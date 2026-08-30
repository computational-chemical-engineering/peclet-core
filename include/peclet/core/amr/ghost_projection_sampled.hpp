// core — SAMPLED ghost-projection overlay: cut cells at MULTIPLE octree levels.
//
// The D1 machinery of docs/amr_mixed_level_cut_band_plan.md: the ghost overlay's ±2
// same-level chain entries become SAMPLE SLOTS — precomputed linear functionals of nearby
// fluid leaves. A same-level entry is the identity (today's behaviour, bit-identical on a
// uniform band); an entry across a 2:1 level boundary is a VIRTUAL SAMPLE at the uniform-grid
// position, reconstructed by DEGREE-2 least squares over fluid leaf point values (the M2
// verdict, tests/study_amr_seam_sample_order.cpp: degree-1 is an O(1) non-decaying matrix
// perturbation at 53% of the physical scale; degree-2 decays ~O(h) to 0.4%). Fallback cascade
// (fluid-only, M1-measured 0.1–0.5%): LS2 -> LS1 -> covering-leaf identity.
//
// CLASSIFICATION CONSISTENCY (the invariant that prevents double-counted fluxes): a face is
// closed in the overlay iff it is closed in the binary operator. Both derive from the SAME
// primitive — a (sub)face is open iff BOTH adjacent ACTUAL leaf centers are fluid and the
// float mean of their center samples is >= 0 (`sampledFaceOpen`; use `makeBinaryOpenFnMixed`
// for AmrPoisson::buildOpenness / AmrMultigrid::setOpenness). A row's own-face state takes
// its SIGN from this canonical rule (any-sub-face-open for a finer-across face — the closed
// sub-faces of a mixed face are then Neumann-zero, a counted Design-A-like local defect) and
// its MAGNITUDE (theta, closure weights) from the row's virtual uniform-position samples (the
// plan's D2 route, level-independent — identical to the classic builder on a uniform band).
//
// Everything is a pure function of (octree, sdf): deterministic under adaptivity (D6).
//
// LAYOUT. The builder and the `*Host` appliers are plain host C++ — they are the oracle path
// (flow_oracle.hpp `setGhostSampled`) and they also BUILD the weights the device consumes, so
// oracle==device parity holds by construction rather than by a second implementation. The
// device mirror (`GhostOverlaySampledDev`, `ghostApplyDeltaSampled`, `ghostDivergDeltaSampled`,
// `GhostGradCsrDev`) lives in the `#ifdef KOKKOS_INLINE_FUNCTION` section at the bottom, so a
// non-Kokkos TU can include this header and get the host half only. Driven from the device
// solver by `AmrFlow::setGhostSampled` (flow.hpp), single-rank — the distributed sample halo is
// a later rung.
//
// STATUS (2026-08-27, plan §Phase 2): Phase-1 rungs 1–4 and Phase 2 are done — momentum ξ-row
// seam correction, wall-aware C/F gates, the device mirror, the graded refinement policy API
// (refine.hpp `refineToSdfGraded`), family-free on a seamed mesh, and a seam offset converging
// at ~1.7–1.9 so seams do not set the order. KNOWN GAPS, in the plan's risk register: sub-face
// closures are unimplemented (the closed sub-faces of a mixed face are Neumann-zero — counted
// by `nMixedFace`, measured non-vanishing), pocket cells are not excluded from LS clouds, and
// there is no distributed sample halo.
#ifndef PECLET_CORE_AMR_GHOST_PROJECTION_SAMPLED_HPP
#define PECLET_CORE_AMR_GHOST_PROJECTION_SAMPLED_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/cf_scheme.hpp"  // CfCsr + detail::{ScalarEnt, compactCsr} (mom seam delta)
#include "peclet/core/amr/ghost_projection.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/host_parallel.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/scheme/ghost_closure.hpp"

namespace peclet::core::amr {

/// Sampled overlay: the classic per-row fields (weights, states, rescale — `base`; base.nbr is
/// UNUSED) plus one linear functional per chain slot (r*15 + a*5 + (q+2)) in CSR form. An empty
/// functional reads 0 (a solid virtual position — the masked value, as in the classic appliers).
struct GhostOverlaySampled {
  GhostOverlay base;
  std::vector<Index> sampStart;  ///< [base.n*15 + 1]
  std::vector<Index> sampIdx;
  std::vector<double> sampW;
  std::vector<int8_t> sampFluid;  ///< [base.n*15] virtual position is fluid (gradient cascade)
  std::vector<Index> rowOf;       ///< [numLeaves] row index of a leaf, -1 if none
  // build census (printed by the builder):
  long nIdentity = 0, nLS2 = 0, nLS1 = 0, nDegraded = 0, nSolidSlot = 0, nMixedFace = 0,
       nSignForced = 0;
};

namespace detail {

/// Gaussian elimination with partial pivoting (small dense normal equations).
inline bool gpsSolveDense(int n, double* A, double* b) {
  for (int k = 0; k < n; ++k) {
    int piv = k;
    for (int r = k + 1; r < n; ++r)
      if (std::fabs(A[r * n + k]) > std::fabs(A[piv * n + k]))
        piv = r;
    if (std::fabs(A[piv * n + k]) < 1e-14)
      return false;
    if (piv != k) {
      for (int c = k; c < n; ++c) {
        const double tmp = A[k * n + c];
        A[k * n + c] = A[piv * n + c];
        A[piv * n + c] = tmp;
      }
      const double tb = b[k];
      b[k] = b[piv];
      b[piv] = tb;
    }
    for (int r = k + 1; r < n; ++r) {
      const double f = A[r * n + k] / A[k * n + k];
      for (int c = k; c < n; ++c)
        A[r * n + c] -= f * A[k * n + c];
      b[r] -= f * b[k];
    }
  }
  for (int k = n - 1; k >= 0; --k) {
    for (int c = k + 1; c < n; ++c)
      b[k] -= A[k * n + c] * b[c];
    b[k] /= A[k * n + k];
  }
  return true;
}

inline void gpsMonomials(const double d[3], int deg, double* m, int& nm) {
  m[0] = 1.0;
  m[1] = d[0];
  m[2] = d[1];
  m[3] = d[2];
  nm = 4;
  if (deg >= 2) {
    m[4] = d[0] * d[0];
    m[5] = d[1] * d[1];
    m[6] = d[2] * d[2];
    m[7] = d[0] * d[1];
    m[8] = d[0] * d[2];
    m[9] = d[1] * d[2];
    nm = 10;
  }
}

}  // namespace detail

/// The canonical (sub)face openness for mixed-level cut bands: open iff both adjacent ACTUAL
/// leaf centers are fluid AND the float mean of the center samples is >= 0. Reduces to
/// makeBinaryOpenFn on a uniform finest band (centers at ±h0/2). Level-aware by construction —
/// the adjacent leaves are located through the octree, whatever their levels.
/// `origin` is the world origin of the octree's fine units (AmrFlow's origin).
template <unsigned Bits, class SdfFn>
inline auto makeBinaryOpenFnMixed(const BlockOctree<3, Bits>& t, const AmrPoisson<3, Bits>& pres,
                                  SdfFn sdfFn, double h0, Vec<3> origin) {
  return [&t, &pres, sdfFn, h0, origin](const Vec<3>& fc, int axis) -> double {
    auto centerSample = [&](int side) -> std::pair<bool, float> {
      Vec<3> probe = fc;
      probe[axis] += side * 0.25 * h0;  // strictly inside the adjacent leaf
      std::array<long, 3> q{};
      for (int d = 0; d < 3; ++d)
        q[d] = static_cast<long>(std::floor((probe[d] - origin[d]) / h0));
      const Index j = pres.probeSlot(q).first;
      if (j < 0)
        return {false, -1.0f};
      auto b = t.bounds(j);
      const double s = static_cast<double>(Index(1) << t.level(j));
      Vec<3> c{};
      for (int d = 0; d < 3; ++d)
        c[d] = origin[d] + (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
      const double sd = sdfFn(c);
      return {sd > 0.0, static_cast<float>(sd)};
    };
    const auto [flM, sM] = centerSample(-1);
    const auto [flP, sP] = centerSample(+1);
    return (flM && flP && 0.5f * (sM + sP) >= 0.0f) ? 1.0 : 0.0;
  };
}

/// Build the sampled overlay. Rows: fluid-centered leaves that are non-clean under the
/// canonical/virtual classification (see header). Requires pres.init done; does NOT require
/// openness built (the canonical rule is self-computed from the SDF).
template <unsigned Bits, class SdfFn>
inline GhostOverlaySampled buildGhostOverlaySampled(const BlockOctree<3, Bits>& t,
                                                    const AmrPoisson<3, Bits>& pres, SdfFn&& sdf,
                                                    int matrixOrder, int rhsOrder,
                                                    Vec<3> origin = Vec<3>{}) {
  GhostOverlaySampled ov;
  const Index n = t.numLeaves();
  const double h0 = pres.cellWidth(0) / static_cast<double>(Index(1) << t.level(0));

  // Leaf centers + fluid flags. Host-parallel (rung 4): per-leaf disjoint writes.
  std::vector<Vec<3>> cen(static_cast<std::size_t>(n));
  std::vector<char> fluid(static_cast<std::size_t>(n));
  hostParFor(n, [&](Index i) {
    auto b = t.bounds(i);
    const double s = static_cast<double>(Index(1) << t.level(i));
    Vec<3> c{};
    for (int d = 0; d < 3; ++d)
      c[d] = origin[d] + (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
    cen[static_cast<std::size_t>(i)] = c;
    fluid[static_cast<std::size_t>(i)] = sdf(c) > 0.0 ? 1 : 0;
  });

  // Hash bins over leaf centers for LS cloud gathering (bin 4*h0; the domain spans fineExt*h0
  // from origin — cubic domains).
  //
  // F2 FIX (docs/amr_march_perf_and_distributed_plan.md §Findings, resolved 2026-08-30): the
  // period was previously derived from the max leaf bound, which is INCLUSIVE (ext = fineExt−1),
  // and the truncating /4 then lost one whole bin — so the minimum-image period used for cloud
  // membership AND for the LS monomial offsets came out 4 fine cells short at every power-of-two
  // domain (N=64 → 60, 128 → 124, …), displacing every across-the-seam candidate. The period is
  // now the octree's fine extent — the same period probeSlot and every other wrap already uses.
  // Centred geometries (all the P2b/M2 calibration meshes) never engage the wrap and are
  // bitwise-unchanged; a cut band crossing a periodic face (the RCP bed) moves at the ~5e-5
  // relative level in Σu. Single-rank the block IS the domain; the distributed rungs (D1/D2)
  // must pass the GLOBAL fine extent through here instead of the block's.
  const double hb = 4.0 * h0;
  const auto fe = pres.fineExt();
  const long nbx = std::max<long>(
      1, static_cast<long>(std::max(std::max(fe[0], fe[1]), fe[2])) / 4);
  const double domain = static_cast<double>(nbx) * hb;  // world extent (cubic domains)
  std::vector<std::vector<Index>> bins(static_cast<std::size_t>(nbx * nbx * nbx));
  for (Index i = 0; i < n; ++i) {
    const Vec<3>& c = cen[static_cast<std::size_t>(i)];
    long bx = static_cast<long>((c[0] - origin[0]) / hb) % nbx;
    long by = static_cast<long>((c[1] - origin[1]) / hb) % nbx;
    long bz = static_cast<long>((c[2] - origin[2]) / hb) % nbx;
    bins[static_cast<std::size_t>((bz * nbx + by) * nbx + bx)].push_back(i);
  }

  // LS functional at world position p, degree deg, radius rho, scale H: returns the (idx, w)
  // list. Weight vector w_j = mono(d_j) . M^{-1} e0 with M the normal matrix.
  auto lsFunctional = [&](const Vec<3>& p, double rho, double H, int deg,
                          std::vector<Index>& idx, std::vector<double>& w) -> bool {
    idx.clear();
    w.clear();
    std::vector<Index> pts;
    const long lo[3] = {static_cast<long>(std::floor((p[0] - origin[0] - rho) / hb)),
                        static_cast<long>(std::floor((p[1] - origin[1] - rho) / hb)),
                        static_cast<long>(std::floor((p[2] - origin[2] - rho) / hb))};
    const long hi[3] = {static_cast<long>(std::floor((p[0] - origin[0] + rho) / hb)),
                        static_cast<long>(std::floor((p[1] - origin[1] + rho) / hb)),
                        static_cast<long>(std::floor((p[2] - origin[2] + rho) / hb))};
    for (long bx = lo[0]; bx <= hi[0]; ++bx)
      for (long by = lo[1]; by <= hi[1]; ++by)
        for (long bz = lo[2]; bz <= hi[2]; ++bz) {
          const long wx = (bx % nbx + nbx) % nbx, wy = (by % nbx + nbx) % nbx,
                     wz = (bz % nbx + nbx) % nbx;
          for (Index j : bins[static_cast<std::size_t>((wz * nbx + wy) * nbx + wx)]) {
            if (!fluid[static_cast<std::size_t>(j)])
              continue;
            double r2 = 0;
            for (int d = 0; d < 3; ++d) {
              double del = cen[static_cast<std::size_t>(j)][d] - p[d];
              if (del > 0.5 * domain)
                del -= domain;
              if (del < -0.5 * domain)
                del += domain;
              r2 += del * del;
            }
            if (r2 <= rho * rho)
              pts.push_back(j);
          }
        }
    const int need = deg >= 2 ? 12 : 5;
    if (static_cast<long>(pts.size()) < need)
      return false;
    double A[100] = {}, e0[10] = {};
    int nm = 0;
    double mono[10];
    for (Index j : pts) {
      double d[3];
      for (int dd = 0; dd < 3; ++dd) {
        double del = cen[static_cast<std::size_t>(j)][dd] - p[dd];
        if (del > 0.5 * domain)
          del -= domain;
        if (del < -0.5 * domain)
          del += domain;
        d[dd] = del / H;
      }
      detail::gpsMonomials(d, deg, mono, nm);
      for (int r = 0; r < nm; ++r)
        for (int c = 0; c < nm; ++c)
          A[r * nm + c] += mono[r] * mono[c];
    }
    e0[0] = 1.0;
    if (!detail::gpsSolveDense(nm, A, e0))
      return false;
    for (Index j : pts) {
      double d[3];
      for (int dd = 0; dd < 3; ++dd) {
        double del = cen[static_cast<std::size_t>(j)][dd] - p[dd];
        if (del > 0.5 * domain)
          del -= domain;
        if (del < -0.5 * domain)
          del += domain;
        d[dd] = del / H;
      }
      detail::gpsMonomials(d, deg, mono, nm);
      double wj = 0;
      for (int k = 0; k < nm; ++k)
        wj += mono[k] * e0[k];
      idx.push_back(j);
      w.push_back(wj);
    }
    return true;
  };

  ov.rowOf.assign(static_cast<std::size_t>(n), -1);
  ov.sampStart.assign(1, 0);
  std::vector<Index> sIdx;
  std::vector<double> sW;

  // ---- Rung 4 (setup-parallel plan, D2's third pattern): the single sequential-append pass is
  // split in two. PASS 1 (below, host-parallel) does ALL the geometry per leaf and stages an
  // accepted row into that leaf's OWN heap-allocated RowStage — nothing is appended to `ov`, so
  // there is no `slot = ov.base.n` cursor to serialize on. PASS 2 (serial, after) walks the
  // leaves IN ORDER and appends the staged rows, which reproduces today's row order and CSR slot
  // order EXACTLY: row order = leaf order, and within a row the 15 chain slots in (a, q) order
  // with each slot's entries in the order lsFunctional produced them. Nothing is classified
  // twice — the plan permits it (row state is a pure function of geometry) but staging is
  // cheaper.
  //
  // Only ~13% of leaves become rows, so stages are allocated per accepted row (a null pointer for
  // every clean/solid leaf) rather than as a dense array.
  //
  // NOTE the `bins` build above stays SERIAL on purpose: bin membership order sets the order the
  // LS normal-matrix sums are accumulated, i.e. the floating-point value of the weights.
  struct RowStage {
    Index cell = 0;
    float rescale = 0.0f;
    int8_t coupled = 0;
    int8_t state[6] = {};
    float th[6] = {}, w_bc[6] = {}, w_n1[6] = {}, w_n2[6] = {}, wm_n1[6] = {}, wm_n2[6] = {};
    double invh = 0.0;
    int8_t sampFluid[15] = {};
    Index sampCnt[15] = {};
    std::vector<Index> sIdx;
    std::vector<double> sW;
    long nIdentity = 0, nLS2 = 0, nLS1 = 0, nDegraded = 0, nSolidSlot = 0;
  };
  std::vector<std::unique_ptr<RowStage>> stages(static_cast<std::size_t>(n));
  // Per-leaf census slots for the two counters that are tallied over EVERY fluid leaf (clean rows
  // included), summed in leaf order in pass 2 — integers, so the total is order-independent.
  std::vector<uint8_t> sfPer(static_cast<std::size_t>(n), 0), mfPer(static_cast<std::size_t>(n), 0);

  hostParFor(n, [&](Index i) {
    if (!fluid[static_cast<std::size_t>(i)])
      return;
    const unsigned Li = t.level(i);
    const double h = pres.cellWidth(i);
    const Vec<3>& c = cen[static_cast<std::size_t>(i)];

    // Virtual uniform-position float samples (theta / closure magnitudes — the D2 route).
    float Cq[3][5], F[3][4];
    for (int a = 0; a < 3; ++a) {
      for (int q = -2; q <= 2; ++q) {
        Vec<3> p = c;
        p[a] += static_cast<double>(q) * h;
        Cq[a][q + 2] = static_cast<float>(sdf(p));
      }
      for (int m = 0; m < 4; ++m)
        F[a][m] = 0.5f * (Cq[a][m] + Cq[a][m + 1]);
    }
    // Canonical own-face openness (any-sub-face-open across finer neighbours): force the SIGN
    // of the own-face F entries to the canonical rule so overlay-closed <=> binary-closed.
    bool anyOpen[6] = {};  // slot k = 2*axis + (0 plus, 1 minus)
    int nSub[6] = {};
    const float si0 = static_cast<float>(sdf(c));
    pres.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double, double) {
      const int face = 2 * axis + (dir > 0 ? 0 : 1);
      ++nSub[face];
      if (j >= 0 && fluid[static_cast<std::size_t>(j)]) {
        const float sj = static_cast<float>(sdf(cen[static_cast<std::size_t>(j)]));
        if (0.5f * (si0 + sj) >= 0.0f)
          anyOpen[face] = true;
      }
    });
    for (int k = 0; k < 6; ++k) {
      const int a = k / 2;
      const int m = (k & 1) ? 1 : 2;  // minus own face = F[a][1], plus own face = F[a][2]
      const bool virtOpen =
          F[a][m] >= 0.0f && Cq[a][(k & 1) ? 1 : 3] > 0.0f && Cq[a][2] > 0.0f;
      if (virtOpen != anyOpen[k]) {
        // Marginal face where the virtual and canonical classifications disagree: force the
        // canonical sign (overlay-closed <=> binary-closed), keep the (small) magnitude.
        ++sfPer[static_cast<std::size_t>(i)];
        const float mag = std::fabs(F[a][m]) > 0.0f ? std::fabs(F[a][m]) : 1e-6f;
        F[a][m] = anyOpen[k] ? mag : -mag;
      }
      if (nSub[k] > 1 && !anyOpen[k])
        ++mfPer[static_cast<std::size_t>(i)];  // coarse face fully closed against finer neighbours
    }
    // Non-clean test (canonical own faces + virtual +/-1 centers), as the classic builder.
    bool clean = true;
    for (int a = 0; a < 3; ++a)
      clean = clean && Cq[a][1] >= 0.0f && Cq[a][3] >= 0.0f && F[a][1] >= 0.0f && F[a][2] >= 0.0f;
    if (clean)
      return;

    // Fill the row into a scratch ONE-row overlay (gpFillRow writes ov.field(slot), so it needs a
    // GhostOverlayRef; slot 0 of a private overlay gives the identical arithmetic).
    GhostOverlay one;
    one.cell.resize(1);
    one.rescale.resize(1);
    one.coupled.resize(1);
    one.state.resize(6);
    one.th.resize(6);
    one.w_bc.resize(6);
    one.w_n1.resize(6);
    one.w_n2.resize(6);
    one.wm_n1.resize(6);
    one.wm_n2.resize(6);
    detail::GhostOverlayRef ref{one};
    if (!scheme::gpFillRow(ref, 0, i, F, Cq, matrixOrder, rhsOrder))
      return;  // rejected row: nothing staged (the old code's resize(slot) rollback)

    auto st = std::make_unique<RowStage>();
    st->cell = one.cell[0];
    st->rescale = one.rescale[0];
    st->coupled = one.coupled[0];
    for (int k = 0; k < 6; ++k) {
      st->state[k] = one.state[static_cast<std::size_t>(k)];
      st->th[k] = one.th[static_cast<std::size_t>(k)];
      st->w_bc[k] = one.w_bc[static_cast<std::size_t>(k)];
      st->w_n1[k] = one.w_n1[static_cast<std::size_t>(k)];
      st->w_n2[k] = one.w_n2[static_cast<std::size_t>(k)];
      st->wm_n1[k] = one.wm_n1[static_cast<std::size_t>(k)];
      st->wm_n2[k] = one.wm_n2[static_cast<std::size_t>(k)];
    }
    st->invh = 1.0 / h;

    // Sample functionals for the 15 chain slots.
    for (int a = 0; a < 3; ++a)
      for (int q = -2; q <= 2; ++q) {
        const int sl = a * 5 + (q + 2);
        const Index nBefore = static_cast<Index>(st->sIdx.size());
        Vec<3> p = c;
        p[a] += static_cast<double>(q) * h;
        // Recover the covering leaf: floor the world position to fine units (a level-L cell
        // center is lo + 0.5*2^L in fine units, so the floor lands inside the cell).
        std::array<long, 3> lc{};
        for (int d = 0; d < 3; ++d)
          lc[d] = static_cast<long>(std::floor((p[d] - origin[d]) / h0));
        const Index j = pres.probeSlot(lc).first;
        const bool fluidP = sdf(p) > 0.0;
        st->sampFluid[sl] = fluidP ? 1 : 0;
        if (q == 0 || (j >= 0 && t.level(j) == Li)) {
          // own cell or same-level cover: identity (the classic chain; bit-identical band).
          if (j >= 0 && fluid[static_cast<std::size_t>(j)] == (fluidP ? 1 : 0)) {
            st->sIdx.push_back(j);
            st->sW.push_back(1.0);
            ++st->nIdentity;
          } else if (j >= 0) {
            st->sIdx.push_back(j);  // marginal float disagreement: still the covering value
            st->sW.push_back(1.0);
            ++st->nIdentity;
          }
          st->sampCnt[sl] = static_cast<Index>(st->sIdx.size()) - nBefore;
          continue;
        }
        if (!fluidP) {
          // solid virtual position: empty functional (reads the masked 0).
          ++st->nSolidSlot;
          st->sampCnt[sl] = 0;
          continue;
        }
        const double H = (j >= 0) ? pres.cellWidth(j) : h;
        const double rho = 2.2 * std::max(h, H);
        std::vector<Index> idx;
        std::vector<double> w;
        if (lsFunctional(p, rho, H, 2, idx, w)) {
          ++st->nLS2;
        } else if (lsFunctional(p, rho, H, 1, idx, w)) {
          ++st->nLS1;
        } else if (j >= 0 && fluid[static_cast<std::size_t>(j)]) {
          idx.assign(1, j);
          w.assign(1, 1.0);
          ++st->nDegraded;
        } else {
          ++st->nDegraded;  // no support at all: reads 0 (fluid-only cascade floor)
        }
        for (std::size_t k = 0; k < idx.size(); ++k) {
          st->sIdx.push_back(idx[k]);
          st->sW.push_back(w[k]);
        }
        st->sampCnt[sl] = static_cast<Index>(st->sIdx.size()) - nBefore;
      }
    stages[static_cast<std::size_t>(i)] = std::move(st);
  });

  // PASS 2 (serial, order-preserving): append the staged rows in LEAF order — the row order, the
  // 15-slot order and the sampStart/sampIdx/sampW concatenation are exactly what the old single
  // pass produced.
  for (Index i = 0; i < n; ++i) {
    ov.nSignForced += sfPer[static_cast<std::size_t>(i)];
    ov.nMixedFace += mfPer[static_cast<std::size_t>(i)];
    const RowStage* st = stages[static_cast<std::size_t>(i)].get();
    if (!st)
      continue;
    ov.base.cell.push_back(st->cell);
    ov.base.rescale.push_back(st->rescale);
    ov.base.coupled.push_back(st->coupled);
    for (int k = 0; k < 6; ++k) {
      ov.base.state.push_back(st->state[k]);
      ov.base.th.push_back(st->th[k]);
      ov.base.w_bc.push_back(st->w_bc[k]);
      ov.base.w_n1.push_back(st->w_n1[k]);
      ov.base.w_n2.push_back(st->w_n2[k]);
      ov.base.wm_n1.push_back(st->wm_n1[k]);
      ov.base.wm_n2.push_back(st->wm_n2[k]);
    }
    Index acc = static_cast<Index>(sIdx.size());
    for (int sl = 0; sl < 15; ++sl) {
      ov.sampFluid.push_back(st->sampFluid[sl]);
      acc += st->sampCnt[sl];
      ov.sampStart.push_back(acc);
    }
    sIdx.insert(sIdx.end(), st->sIdx.begin(), st->sIdx.end());
    sW.insert(sW.end(), st->sW.begin(), st->sW.end());
    ov.nIdentity += st->nIdentity;
    ov.nLS2 += st->nLS2;
    ov.nLS1 += st->nLS1;
    ov.nDegraded += st->nDegraded;
    ov.nSolidSlot += st->nSolidSlot;
    ov.base.invh.push_back(st->invh);
    ov.rowOf[static_cast<std::size_t>(i)] = ov.base.n;
    ++ov.base.n;
  }
  ov.sampIdx = std::move(sIdx);
  ov.sampW = std::move(sW);
  // `csr` is the length of the sample functional CSR — the work every overlay matvec does, and
  // the quantity Phase M's attribution turns on (an identity slot is ONE entry; a least-squares
  // slot is a whole cloud), so it is reported next to the slot census.
  std::fprintf(stderr,
               "[peclet.core.amr] sampled ghost overlay: %lld rows | slots identity %ld, LS2 %ld, "
               "LS1 %ld, degraded %ld, solid %ld | csr %lld | sign-forced faces %ld, closed mixed "
               "faces %ld\n",
               static_cast<long long>(ov.base.n), ov.nIdentity, ov.nLS2, ov.nLS1, ov.nDegraded,
               ov.nSolidSlot, static_cast<long long>(ov.sampIdx.size()), ov.nSignForced,
               ov.nMixedFace);
  return ov;
}

/// Evaluate slot (r, a, q) of a scalar leaf field. Empty functional reads 0.
inline double gpsSample(const GhostOverlaySampled& ov, Index r, int a, int q,
                        const std::vector<double>& x) {
  const std::size_t s = static_cast<std::size_t>(r * 15 + a * 5 + (q + 2));
  double v = 0.0;
  for (Index k = ov.sampStart[s]; k < ov.sampStart[s + 1]; ++k)
    v += ov.sampW[static_cast<std::size_t>(k)] *
         x[static_cast<std::size_t>(ov.sampIdx[static_cast<std::size_t>(k)])];
  return v;
}

/// Matrix overlay (sampled): y currently holds the BINARY L matvec; overwrite overlay rows with
/// y = rho·(y + invh²·Delta) — the sampled analog of ghostApplyDeltaHost.
inline void ghostApplyDeltaSampledHost(const GhostOverlaySampled& ov, const std::vector<double>& x,
                                       std::vector<double>& y) {
  const GhostOverlay& g = ov.base;
  for (Index r = 0; r < g.n; ++r) {
    const std::size_t rr = static_cast<std::size_t>(r);
    const Index c = g.cell[rr];
    if (!g.coupled[rr]) {
      y[static_cast<std::size_t>(c)] = 0.0;
      continue;
    }
    auto X = [&](int a, int q) { return gpsSample(ov, r, a, q, x); };
    double delta = 0.0;
    for (int k = 0; k < 6; ++k) {
      const int8_t st = g.state[rr * 6 + static_cast<std::size_t>(k)];
      if (st != scheme::GP_QUAD && st != scheme::GP_LIN)
        continue;
      const int a = k / 2;
      const int sgn = (k & 1) ? -1 : 1;
      const int mn = (k & 1) ? 1 : 0;
      const int mf = (k & 1) ? 2 : -1;
      const double w1 = g.wm_n1[rr * 6 + static_cast<std::size_t>(k)];
      const double w2 = g.wm_n2[rr * 6 + static_cast<std::size_t>(k)];
      delta += sgn * w1 * (X(a, mn) - X(a, mn - 1));
      if (st == scheme::GP_QUAD && w2 != 0.0)
        delta += sgn * w2 * (X(a, mf) - X(a, mf - 1));
    }
    const double ih = g.invh[rr];
    y[static_cast<std::size_t>(c)] = g.rescale[rr] * (y[static_cast<std::size_t>(c)] + ih * ih * delta);
  }
}

/// Divergence overlay (sampled): the sampled analog of ghostDivergDeltaHost (u_bc = 0).
inline void ghostDivergDeltaSampledHost(const GhostOverlaySampled& ov,
                                        const std::array<std::vector<double>, 3>& u,
                                        std::vector<double>& d) {
  const GhostOverlay& g = ov.base;
  for (Index r = 0; r < g.n; ++r) {
    const std::size_t rr = static_cast<std::size_t>(r);
    const Index c = g.cell[rr];
    if (!g.coupled[rr]) {
      d[static_cast<std::size_t>(c)] = 0.0;
      continue;
    }
    auto U = [&](int a, int m) {  // face-averaged value at face index i+m along axis a
      return 0.5 * (gpsSample(ov, r, a, m - 1, u[static_cast<std::size_t>(a)]) +
                    gpsSample(ov, r, a, m, u[static_cast<std::size_t>(a)]));
    };
    double dd = 0.0;
    for (int k = 0; k < 6; ++k) {
      const int8_t st = g.state[rr * 6 + static_cast<std::size_t>(k)];
      if (st == scheme::GP_COUPLED)
        continue;
      const int a = k / 2;
      const int sgn = (k & 1) ? -1 : 1;
      const int mg = (k & 1) ? 0 : 1;
      const int mn = (k & 1) ? 1 : 0;
      const int mf = (k & 1) ? 2 : -1;
      if (st == scheme::GP_EXPLICIT) {
        dd += sgn * U(a, mg);
        continue;
      }
      if (st == scheme::GP_BC_ONLY)
        continue;
      double val = g.w_n1[rr * 6 + static_cast<std::size_t>(k)] * U(a, mn);
      if (st == scheme::GP_QUAD)
        val += g.w_n2[rr * 6 + static_cast<std::size_t>(k)] * U(a, mf);
      dd += sgn * val;
    }
    d[static_cast<std::size_t>(c)] =
        g.rescale[rr] * (d[static_cast<std::size_t>(c)] + g.invh[rr] * dd);
  }
}

/// Momentum ξ-overlay seam correction (plan §6.2, rung 1): at overlay rows with a
/// different-level ±1 neighbour, the raw AmrCutCell row is wrong twice over — it classifies
/// and anchors the wall from the COVERING neighbour's center sample (position-inconsistent),
/// and its ξ branch was built with the GLOBAL β = μ/h0² (dimensionally off by 4^ΔL at a
/// coarser row). The corrected target row is the row-local ξ stencil (buildCutStencil with
/// β = μ/h_row², AC0 = idiag + 6β) evaluated on the overlay's VIRTUAL ±1 sample functionals —
/// i.e. exactly the uniform-band momentum closure on virtual samples (D1) — or, when the row
/// is virtually clean, the level-aware conservative regular row. Delivered as a lagged
/// deferred-correction CSR on the momentum source (the cfMom_ pattern): at the fixed point the
/// steady operator carries the corrected row exactly. Assumes u_bc = 0 (stationary walls — the
/// inhomogeneous ξ term is not corrected). Rows that are raw-regular AND virtually clean, or
/// raw-regular with a virtually-ghost face (rare marginal), are skipped and counted.
///
/// Application (oracle step): src += cfApplyHost(csr, u^n) BEFORE makeRhs — the coefficients
/// fold in 1/rscale so makeRhs's rscale multiply lands the correction at unit row scale.
template <unsigned Bits, class Mom, class SdfFn>
inline CfCsr buildMomSeamDelta(const GhostOverlaySampled& ov, const BlockOctree<3, Bits>& t,
                               const AmrPoisson<3, Bits>& pres, const Mom& mom, SdfFn&& sdf,
                               Vec<3> origin, double idiag, double mu, long* nSkipped = nullptr) {
  const Index n = t.numLeaves();
  const double h0 = pres.cellWidth(0) / static_cast<double>(Index(1) << t.level(0));
  std::vector<std::vector<detail::ScalarEnt>> per(static_cast<std::size_t>(n));
  long skipped = 0;
  for (Index i = 0; i < n; ++i) {
    const Index r = ov.rowOf[static_cast<std::size_t>(i)];
    if (r < 0)
      continue;
    const unsigned Li = t.level(i);
    bool seam = false;
    for (int k = 0; k < 6; ++k) {
      const Index nb = mom.neighborOf(i, k);
      if (nb < 0 || t.level(nb) != Li)
        seam = true;
    }
    if (!seam)
      continue;
    // Virtual classification + row-local ξ stencil.
    auto b = t.bounds(i);
    const double s = static_cast<double>(Index(1) << Li);
    Vec<3> c{};
    for (int d = 0; d < 3; ++d)
      c[d] = origin[d] + (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
    const double h = pres.cellWidth(i);
    const double sdfC = sdf(c);
    double sdfNv[6];
    bool anyGhost = false;
    for (int k = 0; k < 6; ++k) {
      Vec<3> p = c;
      p[k / 2] += ((k % 2 == 0) ? +1.0 : -1.0) * h;  // k even = plus side (AmrCutCell::neighbor)
      sdfNv[k] = sdf(p);
      if (sdfNv[k] < 0.0)
        anyGhost = true;
    }
    if (!mom.isCut(i) && !anyGhost)
      continue;  // regular both ways: the C/F-aware raw row is already the target
    if (!mom.isCut(i) && anyGhost) {
      ++skipped;  // raw-regular but virtually ghost (marginal): out of rung-1 scope
      continue;
    }
    auto push = [&](Index cell, double w) {
      if (w != 0.0)
        per[static_cast<std::size_t>(i)].push_back(detail::ScalarEnt{cell, w});
    };
    // + raw row / rscale_r (the assembled ξ row verbatim).
    const double rsr = mom.rhsScale(i);
    push(i, mom.acRaw()[static_cast<std::size_t>(i)] / rsr);
    for (int k = 0; k < 6; ++k) {
      const double a = mom.offRaw()[static_cast<std::size_t>(i) * 6 + k];
      const Index nb = mom.neighborOf(i, k);
      if (a != 0.0 && nb >= 0)
        push(nb, a / rsr);
    }
    // − virtual row / rscale_v.
    if (anyGhost) {
      const double beta = mu / (h * h);
      double ACv, offv[6], rsv = 1.0, inhomv = 0.0;
      Mom::buildCutStencil(sdfC, sdfNv, beta, idiag + 6.0 * beta, ACv, offv, rsv, inhomv);
      push(i, -ACv / rsv);
      for (int k = 0; k < 6; ++k) {
        if (offv[k] == 0.0)
          continue;
        const int q = (k % 2 == 0) ? +1 : -1;
        const std::size_t sl = static_cast<std::size_t>(r * 15 + (k / 2) * 5 + (q + 2));
        for (Index e = ov.sampStart[sl]; e < ov.sampStart[sl + 1]; ++e)
          push(ov.sampIdx[static_cast<std::size_t>(e)],
               -offv[k] / rsv * ov.sampW[static_cast<std::size_t>(e)]);
      }
    } else {
      // Virtually clean: target = the level-aware conservative regular row (assembleOperator's
      // regular branch, reproduced coefficient-for-coefficient).
      const double invV = 1.0 / mom.lap().cellVolume(i);
      double dsum = 0.0;
      mom.lap().forEachFaceNeighbor(i, [&](Index j, Real cf, int, double a) {
        push(j, -(-mu * invV * (a * cf)));  // minus the target's off-diagonal
        dsum += a * cf;
      });
      push(i, -(idiag + mu * invV * dsum));
    }
  }
  CfCsr out;
  detail::compactCsr(per, out, n);
  if (nSkipped)
    *nSkipped = skipped;
  return out;
}

/// Directional ghost cell-gradient on a sampled row (the mixed-level gradOfDir): the same
/// cascade — central where both axis virtual neighbours are fluid, 2nd-order one-sided toward
/// the fluid else (2-point fallback), 0 sandwiched — on the row's sample functionals, so the
/// gradient reads virtual uniform-position values across a seam instead of degrading. Pairing:
/// the SAME functionals feed the constraint and the gradient.
inline double gpsDirGrad(const GhostOverlaySampled& ov, Index r, const std::vector<double>& fld,
                         int c, double invh) {
  auto fl = [&](int q) {
    return ov.sampFluid[static_cast<std::size_t>(r * 15 + c * 5 + (q + 2))] != 0;
  };
  auto F = [&](int q) { return gpsSample(ov, r, c, q, fld); };
  const bool ap = fl(+1), am = fl(-1);
  if (am && ap)
    return (F(+1) - F(-1)) * 0.5 * invh;
  if (ap)
    return fl(+2) ? (-3.0 * F(0) + 4.0 * F(+1) - F(+2)) * 0.5 * invh : (F(+1) - F(0)) * invh;
  if (am)
    return fl(-2) ? (3.0 * F(0) - 4.0 * F(-1) + F(-2)) * 0.5 * invh : (F(0) - F(-1)) * invh;
  return 0.0;
}

/// Host CSR form of the SAMPLED directional gradient overlay (variable-length stencils — a
/// seam row's cascade composes with its LS sample functionals, so the fixed 9-entry
/// GhostGradOverlay cannot hold it). Covers ALL cut cells: overlay rows get the cascade over
/// sample functionals (== oracle gpsDirGrad); cut cells WITHOUT a row get the classic
/// same-level 3-point cascade (== oracle gradOfDir — the same split the oracle's gradP makes,
/// so oracle/device parity holds row-by-row). `classicOk(j, i)` is the fallback's neighbour
/// gate (fluid + same-level + not pocket).
struct GhostGradCsrHost {
  std::vector<Index> cell;   ///< [m] cut leaf
  std::vector<Index> start;  ///< [m*3 + 1] per (cell, axis)
  std::vector<Index> idx;
  std::vector<double> w;
};

template <unsigned Bits, class IsCutFn, class OkFn>
inline GhostGradCsrHost buildSampledGradCsr(const GhostOverlaySampled& ov,
                                            const BlockOctree<3, Bits>& t,
                                            const AmrPoisson<3, Bits>& pres, IsCutFn&& isCut,
                                            OkFn&& classicOk) {
  GhostGradCsrHost g;
  g.start.assign(1, 0);
  const Index n = t.numLeaves();
  auto pushSlot = [&](Index r, int a, int q, double wgt) {
    const std::size_t s = static_cast<std::size_t>(r * 15 + a * 5 + (q + 2));
    for (Index e = ov.sampStart[s]; e < ov.sampStart[s + 1]; ++e) {
      g.idx.push_back(ov.sampIdx[static_cast<std::size_t>(e)]);
      g.w.push_back(wgt * ov.sampW[static_cast<std::size_t>(e)]);
    }
  };
  auto pushCell = [&](Index j, double wgt) {
    g.idx.push_back(j);
    g.w.push_back(wgt);
  };
  for (Index i = 0; i < n; ++i) {
    if (!isCut(i))
      continue;
    g.cell.push_back(i);
    const double h = pres.cellWidth(i);
    const Index r = ov.rowOf[static_cast<std::size_t>(i)];
    for (int a = 0; a < 3; ++a) {
      if (r >= 0) {
        auto fl = [&](int q) {
          return ov.sampFluid[static_cast<std::size_t>(r * 15 + a * 5 + (q + 2))] != 0;
        };
        const bool ap = fl(+1), am = fl(-1);
        if (am && ap) {
          pushSlot(r, a, +1, 0.5 / h);
          pushSlot(r, a, -1, -0.5 / h);
        } else if (ap) {
          if (fl(+2)) {
            pushSlot(r, a, 0, -1.5 / h);
            pushSlot(r, a, +1, 2.0 / h);
            pushSlot(r, a, +2, -0.5 / h);
          } else {
            pushSlot(r, a, +1, 1.0 / h);
            pushSlot(r, a, 0, -1.0 / h);
          }
        } else if (am) {
          if (fl(-2)) {
            pushSlot(r, a, 0, 1.5 / h);
            pushSlot(r, a, -1, -2.0 / h);
            pushSlot(r, a, -2, 0.5 / h);
          } else {
            pushSlot(r, a, 0, 1.0 / h);
            pushSlot(r, a, -1, -1.0 / h);
          }
        }  // sandwiched: no entries (gradient 0)
      } else {
        // Classic same-level cascade (buildGhostGradOverlay's per-cell body).
        const Index jp = pres.periodicNeighbor(i, a, +1);
        const Index jm = pres.periodicNeighbor(i, a, -1);
        const bool ap = classicOk(jp, i), am = classicOk(jm, i);
        if (am && ap) {
          pushCell(jp, 0.5 / h);
          pushCell(jm, -0.5 / h);
        } else if (ap) {
          const Index jpp = pres.periodicNeighbor(jp, a, +1);
          if (classicOk(jpp, i)) {
            pushCell(i, -1.5 / h);
            pushCell(jp, 2.0 / h);
            pushCell(jpp, -0.5 / h);
          } else {
            pushCell(jp, 1.0 / h);
            pushCell(i, -1.0 / h);
          }
        } else if (am) {
          const Index jmm = pres.periodicNeighbor(jm, a, -1);
          if (classicOk(jmm, i)) {
            pushCell(i, 1.5 / h);
            pushCell(jm, -2.0 / h);
            pushCell(jmm, 0.5 / h);
          } else {
            pushCell(i, 1.0 / h);
            pushCell(jm, -1.0 / h);
          }
        }
      }
      g.start.push_back(static_cast<Index>(g.idx.size()));
    }
  }
  return g;
}

// ---- device mirror + kernels (Kokkos TUs only; include after a Kokkos-carrying header) ---------
#ifdef KOKKOS_INLINE_FUNCTION

/// Device mirror of GhostOverlaySampled (uploaded once per setSolid). base.nbr is not uploaded
/// (unused in sampled mode — the sample-slot CSR replaces it).
struct GhostOverlaySampledDev {
  Index n = 0;
  View<Index> cell;
  View<float> rescale;
  View<int8_t> coupled, state;
  View<float> w_n1, w_n2, wm_n1, wm_n2;
  View<double> invh;
  View<Index> sampStart, sampIdx;
  View<double> sampW;
};

inline GhostOverlaySampledDev uploadGhostOverlaySampled(const GhostOverlaySampled& h) {
  GhostOverlaySampledDev d;
  d.n = h.base.n;
  if (h.base.n == 0)
    return d;
  d.cell = toDevice(h.base.cell, "gps_cell");
  d.rescale = toDevice(h.base.rescale, "gps_rescale");
  d.coupled = toDevice(h.base.coupled, "gps_coupled");
  d.state = toDevice(h.base.state, "gps_state");
  d.w_n1 = toDevice(h.base.w_n1, "gps_wn1");
  d.w_n2 = toDevice(h.base.w_n2, "gps_wn2");
  d.wm_n1 = toDevice(h.base.wm_n1, "gps_wmn1");
  d.wm_n2 = toDevice(h.base.wm_n2, "gps_wmn2");
  d.invh = toDevice(h.base.invh, "gps_invh");
  d.sampStart = toDevice(h.sampStart, "gps_sstart");
  d.sampIdx = toDevice(h.sampIdx, "gps_sidx");
  d.sampW = toDevice(h.sampW, "gps_sw");
  return d;
}

/// Device matrix overlay (== ghostApplyDeltaSampledHost). Distinct rows per thread: no atomics.
inline void ghostApplyDeltaSampled(const GhostOverlaySampledDev& ov, View<const double> x,
                                   View<double> y) {
  if (ov.n == 0)
    return;
  auto cell = ov.cell;
  auto resc = ov.rescale;
  auto cpl = ov.coupled;
  auto st = ov.state;
  auto wm1 = ov.wm_n1;
  auto wm2 = ov.wm_n2;
  auto invh = ov.invh;
  auto ss = ov.sampStart;
  auto si = ov.sampIdx;
  auto sw = ov.sampW;
  Kokkos::parallel_for(
      "amr::gps_apply_delta", ov.n, KOKKOS_LAMBDA(const Index r) {
        const Index c = cell(r);
        if (!cpl(r)) {
          y(c) = 0.0;
          return;
        }
        auto X = [&](int a, int q) {
          const Index s = r * 15 + a * 5 + (q + 2);
          double v = 0.0;
          for (Index e = ss(s); e < ss(s + 1); ++e)
            v += sw(e) * x(si(e));
          return v;
        };
        double delta = 0.0;
        for (int k = 0; k < 6; ++k) {
          const int8_t s = st(r * 6 + k);
          if (s != scheme::GP_QUAD && s != scheme::GP_LIN)
            continue;
          const int a = k / 2;
          const int sgn = (k & 1) ? -1 : 1;
          const int mn = (k & 1) ? 1 : 0;
          const int mf = (k & 1) ? 2 : -1;
          const double w1 = wm1(r * 6 + k), w2 = wm2(r * 6 + k);
          delta += sgn * w1 * (X(a, mn) - X(a, mn - 1));
          if (s == scheme::GP_QUAD && w2 != 0.0)
            delta += sgn * w2 * (X(a, mf) - X(a, mf - 1));
        }
        const double ih = invh(r);
        y(c) = resc(r) * (y(c) + ih * ih * delta);
      });
}

/// Device divergence overlay (== ghostDivergDeltaSampledHost).
inline void ghostDivergDeltaSampled(const GhostOverlaySampledDev& ov, View<const double> u0,
                                    View<const double> u1, View<const double> u2,
                                    View<double> d) {
  if (ov.n == 0)
    return;
  auto cell = ov.cell;
  auto resc = ov.rescale;
  auto cpl = ov.coupled;
  auto st = ov.state;
  auto w1v = ov.w_n1;
  auto w2v = ov.w_n2;
  auto invh = ov.invh;
  auto ss = ov.sampStart;
  auto si = ov.sampIdx;
  auto sw = ov.sampW;
  Kokkos::parallel_for(
      "amr::gps_diverg_delta", ov.n, KOKKOS_LAMBDA(const Index r) {
        const Index c = cell(r);
        if (!cpl(r)) {
          d(c) = 0.0;
          return;
        }
        auto S = [&](int a, int q) {
          const Index s = r * 15 + a * 5 + (q + 2);
          double v = 0.0;
          for (Index e = ss(s); e < ss(s + 1); ++e) {
            const Index j = si(e);
            v += sw(e) * ((a == 0) ? u0(j) : (a == 1) ? u1(j) : u2(j));
          }
          return v;
        };
        auto U = [&](int a, int m) { return 0.5 * (S(a, m - 1) + S(a, m)); };
        double dd = 0.0;
        for (int k = 0; k < 6; ++k) {
          const int8_t s = st(r * 6 + k);
          if (s == scheme::GP_COUPLED)
            continue;
          const int a = k / 2;
          const int sgn = (k & 1) ? -1 : 1;
          const int mg = (k & 1) ? 0 : 1;
          const int mn = (k & 1) ? 1 : 0;
          const int mf = (k & 1) ? 2 : -1;
          if (s == scheme::GP_EXPLICIT) {
            dd += sgn * U(a, mg);
            continue;
          }
          if (s == scheme::GP_BC_ONLY)
            continue;
          double val = w1v(r * 6 + k) * U(a, mn);
          if (s == scheme::GP_QUAD)
            val += w2v(r * 6 + k) * U(a, mf);
          dd += sgn * val;
        }
        d(c) = resc(r) * (d(c) + invh(r) * dd);
      });
}

/// Device CSR gradient overlay (sampled mode) + upload + apply (the variable-length analog of
/// GhostGradOverlay / applyGhostGrad; overwrites gx/gy/gz on the overlay cells).
struct GhostGradCsrDev {
  Index n = 0;
  View<Index> cell, start, idx;
  View<double> w;
};

inline GhostGradCsrDev uploadGhostGradCsr(const GhostGradCsrHost& h) {
  GhostGradCsrDev d;
  d.n = static_cast<Index>(h.cell.size());
  if (d.n == 0)
    return d;
  d.cell = toDevice(h.cell, "gcs_cell");
  d.start = toDevice(h.start, "gcs_start");
  d.idx = toDevice(h.idx, "gcs_idx");
  d.w = toDevice(h.w, "gcs_w");
  return d;
}

inline void applyGhostGradCsr(const GhostGradCsrDev& ov, View<const double> f, View<double> gx,
                              View<double> gy, View<double> gz) {
  if (ov.n == 0)
    return;
  auto cell = ov.cell;
  auto start = ov.start;
  auto idx = ov.idx;
  auto w = ov.w;
  Kokkos::parallel_for(
      "amr::flow_ghostgrad_csr", ov.n, KOKKOS_LAMBDA(const Index s) {
        const Index i = cell(s);
        double g[3];
        for (int a = 0; a < 3; ++a) {
          double acc = 0.0;
          for (Index e = start(s * 3 + a); e < start(s * 3 + a + 1); ++e)
            acc += w(e) * f(idx(e));
          g[a] = acc;
        }
        gx(i) = g[0];
        gy(i) = g[1];
        gz(i) = g[2];
      });
}

#endif  // KOKKOS_INLINE_FUNCTION

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_GHOST_PROJECTION_SAMPLED_HPP

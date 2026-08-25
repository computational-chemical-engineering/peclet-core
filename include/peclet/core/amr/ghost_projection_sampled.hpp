// core — SAMPLED ghost-projection overlay: cut cells at MULTIPLE octree levels (host prototype).
//
// The D1 machinery of docs/amr_mixed_level_cut_band_plan.md, host-only (the Phase-1 oracle;
// the device port follows the plan's rungs after the Snellius gate): the ghost overlay's ±2
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
// Host-only-safe: no Kokkos section — the device mirror is Phase-1 work.
#ifndef PECLET_CORE_AMR_GHOST_PROJECTION_SAMPLED_HPP
#define PECLET_CORE_AMR_GHOST_PROJECTION_SAMPLED_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/ghost_projection.hpp"
#include "peclet/core/amr/poisson.hpp"
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

  // Leaf centers + fluid flags.
  std::vector<Vec<3>> cen(static_cast<std::size_t>(n));
  std::vector<char> fluid(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    auto b = t.bounds(i);
    const double s = static_cast<double>(Index(1) << t.level(i));
    Vec<3> c{};
    for (int d = 0; d < 3; ++d)
      c[d] = origin[d] + (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
    cen[static_cast<std::size_t>(i)] = c;
    fluid[static_cast<std::size_t>(i)] = sdf(c) > 0.0 ? 1 : 0;
  }

  // Hash bins over leaf centers for LS cloud gathering (bin 4*h0, periodic unit... the domain
  // spans fineExt*h0 from origin; bin in fine units to stay geometry-agnostic).
  const double hb = 4.0 * h0;
  long nbx = 0;
  {
    // domain extent from the octree via a probe of the wrap in probeSlot is not exposed;
    // derive from the max leaf bound at level 0 units.
    long ext = 0;
    for (Index i = 0; i < n; ++i) {
      auto b = t.bounds(i);
      for (int d = 0; d < 3; ++d)
        ext = std::max(ext, static_cast<long>(b[1][d]));
    }
    nbx = std::max<long>(1, ext / 4);  // ext fine cells / 4 per bin
  }
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

  for (Index i = 0; i < n; ++i) {
    if (!fluid[static_cast<std::size_t>(i)])
      continue;
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
        ++ov.nSignForced;
        const float mag = std::fabs(F[a][m]) > 0.0f ? std::fabs(F[a][m]) : 1e-6f;
        F[a][m] = anyOpen[k] ? mag : -mag;
      }
      if (nSub[k] > 1 && !anyOpen[k])
        ++ov.nMixedFace;  // coarse face fully closed against finer neighbours (all subs closed)
    }
    // Non-clean test (canonical own faces + virtual +/-1 centers), as the classic builder.
    bool clean = true;
    for (int a = 0; a < 3; ++a)
      clean = clean && Cq[a][1] >= 0.0f && Cq[a][3] >= 0.0f && F[a][1] >= 0.0f && F[a][2] >= 0.0f;
    if (clean)
      continue;

    const int slot = static_cast<int>(ov.base.n);
    auto resize = [&](int nRows) {
      ov.base.cell.resize(static_cast<std::size_t>(nRows));
      ov.base.rescale.resize(static_cast<std::size_t>(nRows));
      ov.base.coupled.resize(static_cast<std::size_t>(nRows));
      ov.base.state.resize(static_cast<std::size_t>(nRows) * 6);
      ov.base.th.resize(static_cast<std::size_t>(nRows) * 6);
      ov.base.w_bc.resize(static_cast<std::size_t>(nRows) * 6);
      ov.base.w_n1.resize(static_cast<std::size_t>(nRows) * 6);
      ov.base.w_n2.resize(static_cast<std::size_t>(nRows) * 6);
      ov.base.wm_n1.resize(static_cast<std::size_t>(nRows) * 6);
      ov.base.wm_n2.resize(static_cast<std::size_t>(nRows) * 6);
    };
    resize(slot + 1);
    detail::GhostOverlayRef ref{ov.base};
    if (!scheme::gpFillRow(ref, slot, i, F, Cq, matrixOrder, rhsOrder)) {
      resize(slot);
      continue;
    }

    // Sample functionals for the 15 chain slots.
    for (int a = 0; a < 3; ++a)
      for (int q = -2; q <= 2; ++q) {
        Vec<3> p = c;
        p[a] += static_cast<double>(q) * h;
        // Recover the covering leaf: floor the world position to fine units (a level-L cell
        // center is lo + 0.5*2^L in fine units, so the floor lands inside the cell).
        std::array<long, 3> lc{};
        for (int d = 0; d < 3; ++d)
          lc[d] = static_cast<long>(std::floor((p[d] - origin[d]) / h0));
        const Index j = pres.probeSlot(lc).first;
        const bool fluidP = sdf(p) > 0.0;
        ov.sampFluid.push_back(fluidP ? 1 : 0);
        if (q == 0 || (j >= 0 && t.level(j) == Li)) {
          // own cell or same-level cover: identity (the classic chain; bit-identical band).
          if (j >= 0 && fluid[static_cast<std::size_t>(j)] == (fluidP ? 1 : 0)) {
            sIdx.push_back(j);
            sW.push_back(1.0);
            ++ov.nIdentity;
          } else if (j >= 0) {
            sIdx.push_back(j);  // marginal float disagreement: still the covering value
            sW.push_back(1.0);
            ++ov.nIdentity;
          }
          ov.sampStart.push_back(static_cast<Index>(sIdx.size()));
          continue;
        }
        if (!fluidP) {
          // solid virtual position: empty functional (reads the masked 0).
          ++ov.nSolidSlot;
          ov.sampStart.push_back(static_cast<Index>(sIdx.size()));
          continue;
        }
        const double H = (j >= 0) ? pres.cellWidth(j) : h;
        const double rho = 2.2 * std::max(h, H);
        std::vector<Index> idx;
        std::vector<double> w;
        if (lsFunctional(p, rho, H, 2, idx, w)) {
          ++ov.nLS2;
        } else if (lsFunctional(p, rho, H, 1, idx, w)) {
          ++ov.nLS1;
        } else if (j >= 0 && fluid[static_cast<std::size_t>(j)]) {
          idx.assign(1, j);
          w.assign(1, 1.0);
          ++ov.nDegraded;
        } else {
          ++ov.nDegraded;  // no support at all: reads 0 (fluid-only cascade floor)
        }
        for (std::size_t k = 0; k < idx.size(); ++k) {
          sIdx.push_back(idx[k]);
          sW.push_back(w[k]);
        }
        ov.sampStart.push_back(static_cast<Index>(sIdx.size()));
      }
    ov.base.invh.push_back(1.0 / h);
    ov.rowOf[static_cast<std::size_t>(i)] = ov.base.n;
    ++ov.base.n;
  }
  ov.sampIdx = std::move(sIdx);
  ov.sampW = std::move(sW);
  std::fprintf(stderr,
               "[peclet.core.amr] sampled ghost overlay: %lld rows | slots identity %ld, LS2 %ld, "
               "LS1 %ld, degraded %ld, solid %ld | sign-forced faces %ld, closed mixed faces %ld\n",
               static_cast<long long>(ov.base.n), ov.nIdentity, ov.nLS2, ov.nLS1, ov.nDegraded,
               ov.nSolidSlot, ov.nSignForced, ov.nMixedFace);
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

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_GHOST_PROJECTION_SAMPLED_HPP

// M2 — virtual-sample order requirement (Phase 0 of docs/amr_mixed_level_cut_band_plan.md §8).
//
// Setting: the plan's D1 route replaces a ghost-overlay chain entry that crosses a 2:1 level
// boundary by a VIRTUAL SAMPLE — the field value at the entry's uniform-grid position,
// reconstructed from nearby fluid leaf values (least-squares valid-cell reconstruction, the
// chombo-discharge pattern; coarseStar is the degree-2 special case on one face). This study
// measures what polynomial degree that reconstruction needs so the (2,2) closure does not
// degrade, on a single sphere with a LATITUDE two-level map (jump plane oblique to the wall
// — a hemisphere jump, being perpendicular to the wall everywhere, produces almost no
// sample-needing rows; measured, census 99.8% axis-pass).
//
// For every SEAM row (a required chain entry crosses the level jump), the actual gpFillRow
// (2,2) matrix and divergence deltas are evaluated three ways:
//   exact — virtual samples = the analytic field at the uniform positions (the closure's own
//           truncation, identical to a uniform band; the reference),
//   LS1   — degree-1 (linear) least-squares reconstruction from fluid leaf point values,
//   LS2   — degree-2 (quadratic) reconstruction (10 monomials, cross terms included).
// Reported: the sample errors e1/e2 and the row-propagated perturbations
//   Bmat_d = rho * invh^2 * |Delta(LSd) - Delta(exact)|   (pressure-matrix overlay),
//   Bdiv_d = rho * invh   * |dd(LSd)   - dd(exact)|       (divergence overlay),
// laddered over depth (h -> h/2) with observed orders. Expectation from the C/F literature
// (Martin-Colella): e_d = O(H^{d+1}) => Bmat_1 = O(1) (a local O(1) operator perturbation on
// the codim-2 seam set), Bmat_2 = O(h); Bdiv_1 = O(h), Bdiv_2 = O(h^2). The measurement pins
// the CONSTANTS (theta-amplified closure weights, one-sided near-wall clouds) and hence the
// D1 verdict. Fields: matrix delta on the smooth manufactured phi; divergence delta on the
// exact Stokes velocity around the sphere (no-slip -- realistic near-wall gradients).
//
// A STUDY (printed ladders + soft gates), not a ctest.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef PECLET_CORE_HAVE_MORTON
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/cut_cell.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/scheme/ghost_closure.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

constexpr double kPi = 3.14159265358979323846;
const Vec<3> C0{0.513, 0.493, 0.504};  // off-lattice center (no symmetry luck)
constexpr double R0 = 0.30;

double sdfSphere(const Vec<3>& p) {
  const double dx = p[0] - C0[0], dy = p[1] - C0[1], dz = p[2] - C0[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz) - R0;
}

double phiMan(const Vec<3>& p) {
  const double tp = 2.0 * kPi;
  return std::sin(tp * p[0]) * std::cos(tp * p[1]) + std::sin(tp * p[1]) * std::cos(tp * p[2]) +
         std::sin(tp * p[2]) * std::cos(tp * p[0]);
}

/// Exact Stokes flow past the sphere, U_inf = (1,0,0): solenoidal, no-slip at r=R0.
Vec<3> stokesU(const Vec<3>& p) {
  const double dx = p[0] - C0[0], dy = p[1] - C0[1], dz = p[2] - C0[2];
  const double r2 = dx * dx + dy * dy + dz * dz;
  const double r = std::sqrt(r2);
  const double A = 3.0 * R0 / (4.0 * r);
  const double B = R0 * R0 * R0 / (4.0 * r2 * r);
  return {1.0 - A * (1.0 + dx * dx / r2) - B * (1.0 - 3.0 * dx * dx / r2),
          -(A - 3.0 * B) * dx * dy / r2, -(A - 3.0 * B) * dx * dz / r2};
}

// ---- octree (latitude two-level map; build pattern as in study_amr_seam_census) ----------------------------

Vec<3> centerOf(const BO& t, double h0, Index i) {
  auto b = t.bounds(i);
  const double s = static_cast<double>(Index(1) << t.level(i));
  Vec<3> c{};
  for (int d = 0; d < 3; ++d)
    c[d] = (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
  return c;
}

struct Geo {
  BO t;
  AmrPoisson<3, 21> pres;
  double h0;
  Index n = 0;
};

Geo buildGeo(unsigned depth, unsigned coarseLevel, double band) {
  Geo g;
  g.t = BO(IVec<3>{1, 1, 1}, depth);
  for (unsigned k = 0; k + coarseLevel < depth; ++k)
    g.t.refineIf([](Code, unsigned) { return true; });
  g.h0 = 1.0 / static_cast<double>(1L << depth);
  const double halfDiag = 0.5 * std::sqrt(3.0);
  for (;;) {
    std::vector<Code> toRefine;
    for (Index i = 0; i < g.t.numLeaves(); ++i) {
      const unsigned L = g.t.level(i);
      if (L == 0)
        continue;
      const Vec<3> c = centerOf(g.t, g.h0, i);
      // Latitude two-level map: jump plane z = C0z - 0.15 crosses the sphere at ~30 deg S,
      // where wall normals have |n_z| ~ 0.5 — z-axis closures near the plane REQUIRE entries
      // across the jump (the sample-needing population; a hemisphere jump plane, being
      // perpendicular to the wall, produces almost none — measured, census 99.8% axis-pass).
      const unsigned tgt = c[2] < C0[2] - 0.15 ? 0u : 1u;
      if (L <= tgt)
        continue;
      const double width = g.h0 * static_cast<double>(Index(1) << L);
      if (std::fabs(sdfSphere(c)) <= halfDiag * width + band * 0.5 * width)
        toRefine.push_back(g.t.code(i));
    }
    if (toRefine.empty())
      break;
    std::sort(toRefine.begin(), toRefine.end());
    toRefine.erase(std::unique(toRefine.begin(), toRefine.end()), toRefine.end());
    g.t.refineIf(
        [&](Code c, unsigned) { return std::binary_search(toRefine.begin(), toRefine.end(), c); });
  }
  g.t.balance2to1();
  g.pres.init(g.t, g.h0);
  g.pres.setOrigin(Vec<3>{0, 0, 0});
  g.n = g.t.numLeaves();
  return g;
}

// ---- least-squares reconstruction ---------------------------------------------------------------

/// Solve the (small, SPD-ish) normal equations by Gaussian elimination with partial pivoting.
bool solveDense(int n, double* A, double* b) {
  for (int k = 0; k < n; ++k) {
    int piv = k;
    for (int r = k + 1; r < n; ++r)
      if (std::fabs(A[r * n + k]) > std::fabs(A[piv * n + k]))
        piv = r;
    if (std::fabs(A[piv * n + k]) < 1e-14)
      return false;
    if (piv != k) {
      for (int c = k; c < n; ++c)
        std::swap(A[k * n + c], A[piv * n + c]);
      std::swap(b[k], b[piv]);
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

void monomials(const Vec<3>& d, int deg, double* m, int& nm) {
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

/// LS reconstruction of a field (given per-leaf point values) at position p, degree deg, from
/// fluid leaves within radius rho (coordinates scaled by H for conditioning). Returns false if
/// the cloud is too small / degenerate (caller falls back a degree — the D6 gate analog).
struct LeafField {
  const Geo* g;
  const std::vector<double>* v;
  const std::vector<char>* fluid;
};

// M2a cloud-economy knobs, mirroring buildGhostOverlaySampled's (PECLET_CORE_GPS_RHO /
// PECLET_CORE_GPS_MAXN) so this study can report the seam-reconstruction verdict PER VARIANT.
// It is the instrument that established the degree-2 requirement in the first place, and it
// costs ~2 s where the P2b march ladder costs hours.
inline double gpsRhoFactor() {
  const char* e = std::getenv("PECLET_CORE_GPS_RHO");
  const double v = e ? std::atof(e) : 0.0;
  return v > 0.0 ? v : 2.2;
}
inline long gpsMaxN() {
  const char* e = std::getenv("PECLET_CORE_GPS_MAXN");
  const long v = e ? std::atol(e) : 0;
  return v > 0 ? v : 0;
}

bool lsFit(const LeafField& lf, const std::vector<std::vector<Index>>& bins, long nb, double hb,
           const Vec<3>& p, double rho, double H, int deg, double& out) {
  double A[100] = {}, b[10] = {};
  int nm = 0, npts = 0;
  double mono[10];
  // Candidates are collected first (bin-loop order preserved) so the nearest-N cap can drop the
  // far ones before the normal equations are accumulated.
  const long capN = gpsMaxN();
  struct Cd {
    double r2;
    Index i;
  };
  std::vector<Cd> cds;
  const long b0[3] = {static_cast<long>(std::floor((p[0] - rho) / hb)),
                      static_cast<long>(std::floor((p[1] - rho) / hb)),
                      static_cast<long>(std::floor((p[2] - rho) / hb))};
  const long b1[3] = {static_cast<long>(std::floor((p[0] + rho) / hb)),
                      static_cast<long>(std::floor((p[1] + rho) / hb)),
                      static_cast<long>(std::floor((p[2] + rho) / hb))};
  for (long bx = b0[0]; bx <= b1[0]; ++bx)
    for (long by = b0[1]; by <= b1[1]; ++by)
      for (long bz = b0[2]; bz <= b1[2]; ++bz) {
        const long wx = (bx % nb + nb) % nb, wy = (by % nb + nb) % nb, wz = (bz % nb + nb) % nb;
        for (Index i : bins[static_cast<std::size_t>((wz * nb + wy) * nb + wx)]) {
          if (!(*lf.fluid)[static_cast<std::size_t>(i)])
            continue;
          Vec<3> c = centerOf(lf.g->t, lf.g->h0, i);
          Vec<3> d{};
          double r2 = 0;
          for (int dd = 0; dd < 3; ++dd) {
            double del = c[dd] - p[dd];
            if (del > 0.5)
              del -= 1.0;
            if (del < -0.5)
              del += 1.0;
            d[dd] = del / H;
            r2 += del * del;
          }
          if (r2 > rho * rho)
            continue;
          cds.push_back(Cd{r2, i});
        }
      }
  if (capN > 0 && static_cast<long>(cds.size()) > capN) {
    std::vector<Cd> byDist = cds;
    std::nth_element(byDist.begin(), byDist.begin() + capN, byDist.end(),
                     [](const Cd& a, const Cd& c) {
                       return a.r2 != c.r2 ? a.r2 < c.r2 : a.i < c.i;
                     });
    std::vector<Index> keep;
    keep.reserve(static_cast<std::size_t>(capN));
    for (long q = 0; q < capN; ++q)
      keep.push_back(byDist[static_cast<std::size_t>(q)].i);
    std::sort(keep.begin(), keep.end());
    std::vector<Cd> kept;
    for (const Cd& c : cds)
      if (std::binary_search(keep.begin(), keep.end(), c.i))
        kept.push_back(c);
    cds.swap(kept);
  }
  for (const Cd& cd : cds) {  // accumulate in the ORIGINAL bin-loop order
    const Index i = cd.i;
    Vec<3> c = centerOf(lf.g->t, lf.g->h0, i);
    Vec<3> d{};
    for (int dd = 0; dd < 3; ++dd) {
      double del = c[dd] - p[dd];
      if (del > 0.5)
        del -= 1.0;
      if (del < -0.5)
        del += 1.0;
      d[dd] = del / H;
    }
    monomials(d, deg, mono, nm);
    const double val = (*lf.v)[static_cast<std::size_t>(i)];
    for (int r = 0; r < nm; ++r) {
      for (int cc = 0; cc < nm; ++cc)
        A[r * nm + cc] += mono[r] * mono[cc];
      b[r] += mono[r] * val;
    }
    ++npts;
  }
  const int need = deg >= 2 ? 12 : 5;
  if (npts < need)
    return false;
  if (!solveDense(nm, A, b))
    return false;
  out = b[0];  // monomials are centered at p
  return true;
}

// ---- gpFillRow scratch (as in the census) -------------------------------------------------------

struct ScratchRow {
  mutable Index cellv = 0;
  mutable float rescv = 0;
  mutable int8_t coupledv = 0;
  mutable int8_t statev[6] = {};
  mutable float thv[6] = {}, wbcv[6] = {}, w1v[6] = {}, w2v[6] = {}, wm1v[6] = {}, wm2v[6] = {};
  Index& cell(int) const { return cellv; }
  float& rescale(int) const { return rescv; }
  int8_t& coupled(int) const { return coupledv; }
  int8_t& state(int k) const { return statev[k % 6]; }
  float& th(int k) const { return thv[k % 6]; }
  float& w_bc(int k) const { return wbcv[k % 6]; }
  float& w_n1(int k) const { return w1v[k % 6]; }
  float& w_n2(int k) const { return w2v[k % 6]; }
  float& wm_n1(int k) const { return wm1v[k % 6]; }
  float& wm_n2(int k) const { return wm2v[k % 6]; }
};

// Matrix delta for one row given per-(axis, q) sample values X[a][q] (ghostApplyDeltaHost's sum).
double deltaMat(const ScratchRow& row, const double X[3][5]) {
  double delta = 0.0;
  for (int k = 0; k < 6; ++k) {
    const int8_t st = row.statev[k];
    if (st != scheme::GP_QUAD && st != scheme::GP_LIN)
      continue;
    const int a = k / 2;
    const int sgn = (k & 1) ? -1 : 1;
    const int mn = (k & 1) ? 1 : 0;
    const int mf = (k & 1) ? 2 : -1;
    const double w1 = row.wm1v[k], w2 = row.wm2v[k];
    delta += sgn * w1 * (X[a][mn + 2] - X[a][mn + 1]);
    if (st == scheme::GP_QUAD && w2 != 0.0)
      delta += sgn * w2 * (X[a][mf + 2] - X[a][mf + 1]);
  }
  return delta;
}

// Divergence delta given per-(axis, q) samples of the axis-a velocity U[a][q]
// (ghostDivergDeltaHost's sum; U(a,m) = face average of cell samples m-1, m).
double deltaDiv(const ScratchRow& row, const double U[3][5]) {
  double dd = 0.0;
  for (int k = 0; k < 6; ++k) {
    const int8_t st = row.statev[k];
    if (st == scheme::GP_COUPLED)
      continue;
    const int a = k / 2;
    const int sgn = (k & 1) ? -1 : 1;
    const int mg = (k & 1) ? 0 : 1;
    const int mn = (k & 1) ? 1 : 0;
    const int mf = (k & 1) ? 2 : -1;
    auto Uf = [&](int m) { return 0.5 * (U[a][m + 1] + U[a][m + 2]); };
    if (st == scheme::GP_EXPLICIT) {
      dd += sgn * Uf(mg);
      continue;
    }
    if (st == scheme::GP_BC_ONLY)
      continue;
    double val = row.w1v[k] * Uf(mn);
    if (st == scheme::GP_QUAD)
      val += row.w2v[k] * Uf(mf);
    dd += sgn * val;
  }
  return dd;
}

struct DepthResult {
  long seamRows = 0, degFall = 0;
  double e1 = 0, e2 = 0;          // max raw sample errors
  double bm1 = 0, bm2 = 0;        // max matrix-delta perturbation (rho*invh^2 scaled)
  double bd1 = 0, bd2 = 0;        // max divergence-delta perturbation (rho*invh scaled)
  double refMat = 0, refDiv = 0;  // scale references: max |rho*invh^2*Delta_exact|, div analog
  double sm1 = 0, sm2 = 0, sd1 = 0, sd2 = 0;  // sums of squares (RMS over seam rows)
  double rm1() const { return seamRows ? std::sqrt(sm1 / seamRows) : 0.0; }
  double rm2() const { return seamRows ? std::sqrt(sm2 / seamRows) : 0.0; }
  double rd1() const { return seamRows ? std::sqrt(sd1 / seamRows) : 0.0; }
  double rd2() const { return seamRows ? std::sqrt(sd2 / seamRows) : 0.0; }
};

DepthResult runDepth(unsigned depth) {
  const unsigned coarseLevel = std::min(3u, depth - 3u);
  Geo g = buildGeo(depth, coarseLevel, 2.0);
  const auto& t = g.t;
  const auto& pres = g.pres;

  // Leaf point values + fluid flags for the two test fields.
  std::vector<char> fluid(static_cast<std::size_t>(g.n));
  std::vector<double> fPhi(static_cast<std::size_t>(g.n), 0.0);
  std::array<std::vector<double>, 3> fU;
  for (int a = 0; a < 3; ++a)
    fU[a].assign(static_cast<std::size_t>(g.n), 0.0);
  for (Index i = 0; i < g.n; ++i) {
    const Vec<3> c = centerOf(t, g.h0, i);
    const bool fl = sdfSphere(c) > 0.0;
    fluid[static_cast<std::size_t>(i)] = fl ? 1 : 0;
    if (fl) {
      fPhi[static_cast<std::size_t>(i)] = phiMan(c);
      const Vec<3> v = stokesU(c);
      for (int a = 0; a < 3; ++a)
        fU[a][static_cast<std::size_t>(i)] = v[a];
    }
  }

  // Hash grid over leaf centers (bin 4*h0).
  const double hb = 4.0 * g.h0;
  const long nb = std::lround(1.0 / hb);
  std::vector<std::vector<Index>> bins(static_cast<std::size_t>(nb * nb * nb));
  for (Index i = 0; i < g.n; ++i) {
    const Vec<3> c = centerOf(t, g.h0, i);
    long bx = static_cast<long>(c[0] / hb) % nb, by = static_cast<long>(c[1] / hb) % nb,
         bz = static_cast<long>(c[2] / hb) % nb;
    bins[static_cast<std::size_t>((bz * nb + by) * nb + bx)].push_back(i);
  }
  LeafField lfPhi{&g, &fPhi, &fluid};
  LeafField lfU[3] = {{&g, &fU[0], &fluid}, {&g, &fU[1], &fluid}, {&g, &fU[2], &fluid}};

  DepthResult res;
  for (Index i = 0; i < g.n; ++i) {
    const unsigned Li = t.level(i);
    const double h = g.h0 * static_cast<double>(Index(1) << Li);
    const Vec<3> c = centerOf(t, g.h0, i);
    if (!(sdfSphere(c) > 0.0))
      continue;
    // D2 classification at virtual uniform positions.
    float Cq[3][5], F[3][4];
    bool clean = true;
    for (int a = 0; a < 3; ++a) {
      for (int q = -2; q <= 2; ++q) {
        Vec<3> p = c;
        p[a] += static_cast<double>(q) * h;
        Cq[a][q + 2] = static_cast<float>(sdfSphere(p));
      }
      for (int m = 0; m < 4; ++m)
        F[a][m] = 0.5f * (Cq[a][m] + Cq[a][m + 1]);
      clean = clean && Cq[a][1] >= 0.0f && Cq[a][3] >= 0.0f && F[a][1] >= 0.0f && F[a][2] >= 0.0f;
    }
    if (clean)
      continue;
    ScratchRow row;
    if (!scheme::gpFillRow(row, 0, i, F, Cq, 2, 2))
      continue;
    // Chains + required entries (as in the census).
    Index ch[3][5];
    for (int a = 0; a < 3; ++a) {
      ch[a][2] = i;
      ch[a][3] = pres.periodicNeighbor(i, a, +1);
      ch[a][1] = pres.periodicNeighbor(i, a, -1);
      ch[a][4] = ch[a][3] >= 0 ? pres.periodicNeighbor(ch[a][3], a, +1) : ch[a][3];
      ch[a][0] = ch[a][1] >= 0 ? pres.periodicNeighbor(ch[a][1], a, -1) : ch[a][1];
    }
    bool req[3][5] = {};
    for (int k = 0; k < 6; ++k) {
      const int8_t st = row.statev[k];
      const int a = k / 2;
      const int s = (k & 1) ? +1 : -1;
      if (st == scheme::GP_QUAD) {
        req[a][2 + s] = req[a][2 + 2 * s] = true;
      } else if (st == scheme::GP_LIN) {
        req[a][2 + s] = true;
      } else if (st == scheme::GP_EXPLICIT) {
        req[a][2 - s] = true;
      }
    }
    bool seam = false;
    for (int a = 0; a < 3; ++a)
      for (int q = 0; q < 5; ++q)
        if (req[a][q] && q != 2 && (ch[a][q] < 0 || t.level(ch[a][q]) != Li))
          seam = true;
    if (!seam)
      continue;
    ++res.seamRows;

    // Per-(axis, q) sample values, three ways. Non-required / same-level entries use the leaf
    // point value (identity slots — no error, as D1 would). phi and each velocity component.
    double Xe[3][5], X1[3][5], X2[3][5];      // phi
    double Ue[3][3][5], U1[3][3][5], U2[3][3][5];  // [component][axis][q]
    for (int a = 0; a < 3; ++a)
      for (int q = 0; q < 5; ++q) {
        const Index j = ch[a][q];
        Vec<3> p = c;
        p[a] += static_cast<double>(q - 2) * h;
        for (int d = 0; d < 3; ++d) {
          if (p[d] >= 1.0)
            p[d] -= 1.0;
          if (p[d] < 0.0)
            p[d] += 1.0;
        }
        const bool virt = req[a][q] && q != 2 && (j < 0 || t.level(j) != Li);
        if (!virt) {
          const bool have = j >= 0;
          const double phv = have ? fPhi[static_cast<std::size_t>(j)] : 0.0;
          Xe[a][q] = X1[a][q] = X2[a][q] = phv;
          for (int cmp = 0; cmp < 3; ++cmp) {
            const double uv = have ? fU[cmp][static_cast<std::size_t>(j)] : 0.0;
            Ue[cmp][a][q] = U1[cmp][a][q] = U2[cmp][a][q] = uv;
          }
          continue;
        }
        const unsigned Lj = t.level(j);
        const double H = g.h0 * static_cast<double>(Index(1) << Lj);
        const double rho = gpsRhoFactor() * std::max(h, H);
        const bool fluidP = sdfSphere(p) > 0.0;
        // exact
        Xe[a][q] = fluidP ? phiMan(p) : 0.0;
        {
          const Vec<3> v = fluidP ? stokesU(p) : Vec<3>{};
          for (int cmp = 0; cmp < 3; ++cmp)
            Ue[cmp][a][q] = v[cmp];
        }
        // LS1 / LS2 (deg-2 falls back to deg-1, deg-1 to the covering value — D6 gates)
        auto sample = [&](const LeafField& lf, int deg, double exact) {
          double out;
          if (deg >= 2 && lsFit(lf, bins, nb, hb, p, rho, H, 2, out))
            return out;
          if (lsFit(lf, bins, nb, hb, p, rho, H, 1, out))
            return out;
          ++res.degFall;
          return j >= 0 ? (*lf.v)[static_cast<std::size_t>(j)] : exact;
        };
        if (!fluidP) {
          // solid virtual position: never read (weights zero for its face state) — keep 0.
          X1[a][q] = X2[a][q] = 0.0;
          for (int cmp = 0; cmp < 3; ++cmp)
            U1[cmp][a][q] = U2[cmp][a][q] = 0.0;
          continue;
        }
        X1[a][q] = sample(lfPhi, 1, Xe[a][q]);
        X2[a][q] = sample(lfPhi, 2, Xe[a][q]);
        res.e1 = std::max(res.e1, std::fabs(X1[a][q] - Xe[a][q]));
        res.e2 = std::max(res.e2, std::fabs(X2[a][q] - Xe[a][q]));
        for (int cmp = 0; cmp < 3; ++cmp) {
          U1[cmp][a][q] = sample(lfU[cmp], 1, Ue[cmp][a][q]);
          U2[cmp][a][q] = sample(lfU[cmp], 2, Ue[cmp][a][q]);
          res.e1 = std::max(res.e1, std::fabs(U1[cmp][a][q] - Ue[cmp][a][q]));
          res.e2 = std::max(res.e2, std::fabs(U2[cmp][a][q] - Ue[cmp][a][q]));
        }
      }
    const double invh = 1.0 / h, rr = row.rescv;
    const double me = deltaMat(row, Xe), m1 = deltaMat(row, X1), m2 = deltaMat(row, X2);
    const double pm1 = rr * invh * invh * std::fabs(m1 - me);
    const double pm2 = rr * invh * invh * std::fabs(m2 - me);
    res.bm1 = std::max(res.bm1, pm1);
    res.bm2 = std::max(res.bm2, pm2);
    res.sm1 += pm1 * pm1;
    res.sm2 += pm2 * pm2;
    res.refMat = std::max(res.refMat, rr * invh * invh * std::fabs(me));
    // divergence: samples per axis are the AXIS component (U(a,m) averages u_a).
    double De[3][5], D1[3][5], D2[3][5];
    for (int a = 0; a < 3; ++a)
      for (int q = 0; q < 5; ++q) {
        De[a][q] = Ue[a][a][q];
        D1[a][q] = U1[a][a][q];
        D2[a][q] = U2[a][a][q];
      }
    const double de = deltaDiv(row, De), d1 = deltaDiv(row, D1), d2 = deltaDiv(row, D2);
    const double pd1 = rr * invh * std::fabs(d1 - de);
    const double pd2 = rr * invh * std::fabs(d2 - de);
    res.bd1 = std::max(res.bd1, pd1);
    res.bd2 = std::max(res.bd2, pd2);
    res.sd1 += pd1 * pd1;
    res.sd2 += pd2 * pd2;
    res.refDiv = std::max(res.refDiv, rr * invh * std::fabs(de));
  }
  return res;
}

double orderOf(double prev, double cur) {
  return (prev > 0 && cur > 0) ? std::log2(prev / cur) : 0.0;
}

// ---- [B] momentum ξ-row truncation: raw (covering reads, global β) vs corrected (row-local
// virtual stencil — the buildMomSeamDelta target) against the analytic operator, on a
// manufactured field that RESPECTS the wall BC: f = sdf·phi (zero on the sphere), so the
// wall-anchored closure rows are consistent with idiag·f − mu·lap f. ------------------------------

double fBc(const Vec<3>& p) {
  return sdfSphere(p) * phiMan(p);
}
double lapFBc(const Vec<3>& p) {
  // lap(s·phi) = phi·lap s + 2 grad s . grad phi + s·lap phi, s = |r|-R0 (lap s = 2/|r|).
  const double tp = 2.0 * kPi;
  const double dx = p[0] - C0[0], dy = p[1] - C0[1], dz = p[2] - C0[2];
  const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
  const Vec<3> gs{dx / r, dy / r, dz / r};
  const Vec<3> gphi{tp * std::cos(tp * p[0]) * std::cos(tp * p[1]) -
                        tp * std::sin(tp * p[2]) * std::sin(tp * p[0]),
                    -tp * std::sin(tp * p[0]) * std::sin(tp * p[1]) +
                        tp * std::cos(tp * p[1]) * std::cos(tp * p[2]),
                    -tp * std::sin(tp * p[1]) * std::sin(tp * p[2]) +
                        tp * std::cos(tp * p[2]) * std::cos(tp * p[0])};
  // phiMan is a sum of sin·cos products in orthogonal coords: each term has lap = −2·tp²·term.
  const double lapPhiMan = -2.0 * tp * tp * phiMan(p);
  return phiMan(p) * 2.0 / r + 2.0 * (gs[0] * gphi[0] + gs[1] * gphi[1] + gs[2] * gphi[2]) +
         (r - R0) * lapPhiMan;
}

struct MomResult {
  long rows = 0;
  double eRaw = 0, eVirt = 0, eVirtLs = 0;  // max rescaled-row truncation
  double sRaw = 0, sVirt = 0, sVirtLs = 0;  // sums of squares (RMS)
  double rRaw() const { return rows ? std::sqrt(sRaw / rows) : 0.0; }
  double rVirt() const { return rows ? std::sqrt(sVirt / rows) : 0.0; }
  double rVirtLs() const { return rows ? std::sqrt(sVirtLs / rows) : 0.0; }
};

MomResult runMomDepth(unsigned depth) {
  const unsigned coarseLevel = std::min(3u, depth - 3u);
  Geo g = buildGeo(depth, coarseLevel, 2.0);
  const auto& t = g.t;
  const double idiag = 1.0, mu = 1.0;
  AmrCutCell<21> mom;
  mom.init(t, g.h0, Vec<3>{});
  mom.build(sdfSphere, idiag, mu / (g.h0 * g.h0));  // the RAW build: global finest-h beta

  // Leaf point values of f + fluid flags + hash bins (as runDepth).
  const Index n = t.numLeaves();
  std::vector<char> fluid(static_cast<std::size_t>(n));
  std::vector<double> fv(static_cast<std::size_t>(n), 0.0);
  for (Index i = 0; i < n; ++i) {
    const Vec<3> c = centerOf(t, g.h0, i);
    fluid[static_cast<std::size_t>(i)] = sdfSphere(c) > 0.0 ? 1 : 0;
    if (fluid[static_cast<std::size_t>(i)])
      fv[static_cast<std::size_t>(i)] = fBc(c);
  }
  const double hb = 4.0 * g.h0;
  const long nb = std::lround(1.0 / hb);
  std::vector<std::vector<Index>> bins(static_cast<std::size_t>(nb * nb * nb));
  for (Index i = 0; i < n; ++i) {
    const Vec<3> c = centerOf(t, g.h0, i);
    long bx = static_cast<long>(c[0] / hb) % nb, by = static_cast<long>(c[1] / hb) % nb,
         bz = static_cast<long>(c[2] / hb) % nb;
    bins[static_cast<std::size_t>((bz * nb + by) * nb + bx)].push_back(i);
  }
  LeafField lf{&g, &fv, &fluid};

  MomResult res;
  for (Index i = 0; i < n; ++i) {
    if (!mom.isCut(i) || !fluid[static_cast<std::size_t>(i)])
      continue;
    const unsigned Li = t.level(i);
    bool seam = false;
    for (int k = 0; k < 6; ++k) {
      const Index nbk = mom.neighborOf(i, k);
      if (nbk < 0 || t.level(nbk) != Li)
        seam = true;
    }
    if (!seam)
      continue;
    const Vec<3> c = centerOf(t, g.h0, i);
    const double h = g.h0 * static_cast<double>(Index(1) << Li);
    // raw row applied to f (covering reads), physical scale (/rscale).
    double raw = mom.acRaw()[static_cast<std::size_t>(i)] * fv[static_cast<std::size_t>(i)];
    for (int k = 0; k < 6; ++k) {
      const double a = mom.offRaw()[static_cast<std::size_t>(i) * 6 + k];
      const Index j = mom.neighborOf(i, k);
      if (a != 0.0 && j >= 0)
        raw += a * fv[static_cast<std::size_t>(j)];
    }
    const double rsr = mom.rhsScale(i);
    // virtual row (row-local beta) on exact / LS2 virtual samples.
    double sdfNv[6], fVirt[6], fLs[6];
    bool anyGhost = false;
    for (int k = 0; k < 6; ++k) {
      Vec<3> p = c;
      p[k / 2] += ((k % 2 == 0) ? +1.0 : -1.0) * h;
      sdfNv[k] = sdfSphere(p);
      anyGhost = anyGhost || sdfNv[k] < 0.0;
      fVirt[k] = sdfNv[k] > 0.0 ? fBc(p) : 0.0;
      double out;
      const Index j = mom.neighborOf(i, k);
      const double H = j >= 0 ? g.h0 * static_cast<double>(Index(1) << t.level(j)) : h;
      if (sdfNv[k] <= 0.0)
        fLs[k] = 0.0;
      else if (j >= 0 && t.level(j) == Li)
        fLs[k] = fv[static_cast<std::size_t>(j)];
      else if (lsFit(lf, bins, nb, hb, p, gpsRhoFactor() * std::max(h, H), H, 2, out))
        fLs[k] = out;
      else
        fLs[k] = fVirt[k];  // (degenerate cloud: exact fallback, counted elsewhere)
    }
    if (!anyGhost)
      continue;  // virtually clean: regular-row target, out of this instrument's scope
    const double beta = mu / (h * h);
    double ACv, offv[6], rsv = 1.0, inh = 0.0;
    AmrCutCell<21>::buildCutStencil(sdfSphere(c), sdfNv, beta, idiag + 6.0 * beta, ACv, offv, rsv,
                                    inh);
    double virt = ACv * fv[static_cast<std::size_t>(i)], virtLs = virt;
    for (int k = 0; k < 6; ++k) {
      virt += offv[k] * fVirt[k];
      virtLs += offv[k] * fLs[k];
    }
    // RESCALED-row truncation (the metric the solver feels: |row(f) − rscale·target|, bounded
    // weights — un-rescaling by 1/rscale would amplify sliver rows, the P5b pointwise trap).
    const double target = idiag * fv[static_cast<std::size_t>(i)] - mu * lapFBc(c);
    ++res.rows;
    const double er = std::fabs(raw - rsr * target);
    const double ev = std::fabs(virt - rsv * target);
    const double el = std::fabs(virtLs - rsv * target);
    res.eRaw = std::max(res.eRaw, er);
    res.eVirt = std::max(res.eVirt, ev);
    res.eVirtLs = std::max(res.eVirtLs, el);
    res.sRaw += er * er;
    res.sVirt += ev * ev;
    res.sVirtLs += el * el;
  }
  return res;
}

}  // namespace

int main() {
  std::printf("M2 virtual-sample order study (plan docs/amr_mixed_level_cut_band_plan.md §8):\n");
  std::printf("latitude two-level sphere (jump oblique to the wall); (2,2) closure deltas with\n");
  std::printf("exact vs LS1 vs LS2\n");
  std::printf("virtual samples at seam rows. B = row-propagated perturbation (max over rows).\n\n");
  std::printf("%6s %6s %9s %9s | %9s %9s %9s %9s %9s | %9s %9s %9s %9s %9s | %5s\n", "N", "rows",
              "e1", "e2", "Bmat1", "rmsM1", "Bmat2", "rmsM2", "refMat", "Bdiv1", "rmsD1", "Bdiv2",
              "rmsD2", "refDiv", "fall");
  std::vector<DepthResult> R;
  for (unsigned depth = 6; depth <= 9; ++depth) {
    DepthResult r = runDepth(depth);
    std::printf("%6ld %6ld %9.2e %9.2e | %9.2e %9.2e %9.2e %9.2e %9.2e | %9.2e %9.2e %9.2e %9.2e "
                "%9.2e | %5ld\n",
                1L << depth, r.seamRows, r.e1, r.e2, r.bm1, r.rm1(), r.bm2, r.rm2(), r.refMat,
                r.bd1, r.rd1(), r.bd2, r.rd2(), r.refDiv, r.degFall);
    R.push_back(r);
  }
  const auto& f0 = R.front();
  const auto& fN = R.back();
  const double rungs = static_cast<double>(R.size() - 1);
  auto agg = [&](double a, double b) { return orderOf(a, b) / rungs; };
  const double lapScale = 3.0 * (2.0 * kPi) * (2.0 * kPi);  // max |lap phiMan|
  std::printf("\naggregate orders (N=%ld -> %ld): Bmat1 %.2f (rms %.2f) | Bmat2 %.2f (rms %.2f) "
              "| Bdiv1 %.2f (rms %.2f) | Bdiv2 %.2f (rms %.2f)\n",
              1L << 6, 1L << 9, agg(f0.bm1, fN.bm1), agg(f0.rm1(), fN.rm1()), agg(f0.bm2, fN.bm2),
              agg(f0.rm2(), fN.rm2()), agg(f0.bd1, fN.bd1), agg(f0.rd1(), fN.rd1()),
              agg(f0.bd2, fN.bd2), agg(f0.rd2(), fN.rd2()));
  std::printf("physical scales at finest: |lap phi| = %.3g (Bmat1/scale = %.1f%%, Bmat2/scale = "
              "%.2f%%); refDiv = %.3g (Bdiv2/refDiv = %.3f%%)\n",
              lapScale, 100.0 * fN.bm1 / lapScale, 100.0 * fN.bm2 / lapScale, fN.refDiv,
              100.0 * fN.bd2 / fN.refDiv);
  // [B] momentum ξ-row truncation at seam rows: raw (covering reads, global β) vs the
  // buildMomSeamDelta target (row-local virtual stencil), exact and LS2 samples.
  std::printf("\n[B] momentum xi-row truncation at seam rows (f = sdf·phi, wall-consistent):\n");
  std::printf("%6s %6s | %10s %10s %10s | %10s %10s %10s\n", "N", "rows", "rawMax", "virtMax",
              "ls2Max", "rawRms", "virtRms", "ls2Rms");
  for (unsigned depth = 6; depth <= 8; ++depth) {
    MomResult m = runMomDepth(depth);
    std::printf("%6ld %6ld | %10.3e %10.3e %10.3e | %10.3e %10.3e %10.3e\n", 1L << depth,
                m.rows, m.eRaw, m.eVirt, m.eVirtLs, m.rRaw(), m.rVirt(), m.rVirtLs());
  }

  std::printf("\n==== soft gates (documenting the D1 verdict) ====\n");
  struct Gate {
    const char* name;
    bool ok;
  } gates[] = {
      {"LS1 matrix perturbation is O(1) on the physical scale (agg ord <= 0.3) — linear "
       "INSUFFICIENT for the matrix path",
       agg(f0.bm1, fN.bm1) <= 0.3},
      {"LS2 matrix perturbation decays (agg ord >= 0.6) — quadratic gives consistent seam rows",
       agg(f0.bm2, fN.bm2) >= 0.6},
      {"LS2 divergence perturbation decays at >= 1st order (rms agg)",
       agg(f0.rd2(), fN.rd2()) >= 1.0},
      {"LS2 matrix perturbation < 3% of |lap phi| at finest", fN.bm2 < 0.03 * lapScale},
  };
  int npass = 0;
  for (const auto& gt : gates) {
    std::printf("  %s  %s\n", gt.ok ? "PASS" : "FAIL", gt.name);
    npass += gt.ok ? 1 : 0;
  }
  std::printf("%d/4 gates\n", npass);
  return 0;
}
#else
#include <cstdio>
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping\n");
  return 0;
}
#endif

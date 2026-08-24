// M1 — Option-0' seam census (Phase 0 of docs/amr_mixed_level_cut_band_plan.md).
//
// Measures, on a periodic random sphere packing (RSA + engineered contact/throat pairs) with a
// GAP-GRADED surface level map (cut cells at multiple octree levels, the plan's target regime):
// how many ghost-overlay rows
//   [strict]  pass the CURRENT contract (all 3 axes' +/-2 chain same-level — today's invariant),
//   [axis]    pass the per-CLOSURE-axis relaxation (Option 0': only the entries the (2,2)
//             closures actually read must be same-level),
//   [quad]    need virtual samples and full coarseStar-quality support exists,
//   [lin]     need virtual samples with degraded (linear) tangential support,
//   [fall]    hit the fluid-only fallback cascade (a required sample's support is solid/missing).
// Plus the D4 economics: cells of the graded map vs the uniform finest band.
//
// Classification is the plan's D2 route: F/Cq float SDF samples at VIRTUAL UNIFORM positions
// (center +/- q*h(L_i) along the axis) — level-independent, the same arithmetic as
// buildGhostOverlay on a uniform band. Face states come from the REAL scheme::gpFillRow (2,2).
//
// A STUDY (printed tables), not a regression test: the numbers decide whether the sample
// machinery is a rare path or the main path (plan §8 M1). Also runs the M2 GEOMETRY as a
// reference (single sphere, hemisphere two-level map -> one clean seam ring).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef PECLET_CORE_HAVE_MORTON
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/scheme/ghost_closure.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

// ---- packing ------------------------------------------------------------------------------------

struct Sphere {
  Vec<3> c;
  double r;
};

double pdist2(const Vec<3>& a, const Vec<3>& b) {
  double s = 0.0;
  for (int d = 0; d < 3; ++d) {
    double dd = std::fabs(a[d] - b[d]);
    if (dd > 0.5)
      dd = 1.0 - dd;
    s += dd * dd;
  }
  return s;
}

struct Packing {
  std::vector<Sphere> sp;
  /// signed distance to the union (fluid > 0), min-image periodic.
  double sdf(const Vec<3>& p) const {
    double m = 1e30;
    for (const auto& s : sp)
      m = std::min(m, std::sqrt(pdist2(p, s.c)) - s.r);
    return m;
  }
  /// two-closest-surfaces gap proxy: d1 + d2 (plan §7 criterion 1).
  double gap(const Vec<3>& p) const {
    double d1 = 1e30, d2 = 1e30;
    for (const auto& s : sp) {
      const double d = std::sqrt(pdist2(p, s.c)) - s.r;
      if (d < d1) {
        d2 = d1;
        d1 = d;
      } else if (d < d2) {
        d2 = d;
      }
    }
    return d1 + d2;
  }
};

// Deterministic LCG (no <random> — reproducible across libstdc++ versions).
struct Lcg {
  unsigned long long s;
  double next() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<double>((s >> 11) & ((1ull << 53) - 1)) / static_cast<double>(1ull << 53);
  }
};

/// RSA packing with mild overlaps allowed (real contacts/cusps) + engineered throat pairs at
/// prescribed surface gaps (in units of h_fine) to exercise the level ladder.
Packing buildPacking(double hFine) {
  Packing pk;
  auto ok = [&](const Sphere& s) {
    for (const auto& q : pk.sp)
      if (std::sqrt(pdist2(s.c, q.c)) < 0.90 * (s.r + q.r))
        return false;
    return true;
  };
  // Engineered pairs: gaps 2, 6, 20 * hFine along skew axes (off-lattice, no symmetry luck).
  const double gaps[3] = {2.0 * hFine, 6.0 * hFine, 20.0 * hFine};
  const double R = 0.085;
  const Vec<3> at[3] = {{0.21, 0.24, 0.77}, {0.72, 0.71, 0.28}, {0.23, 0.74, 0.26}};
  const Vec<3> dir[3] = {{1.0, 0.3, 0.1}, {0.2, 1.0, 0.4}, {0.5, 0.2, 1.0}};
  for (int k = 0; k < 3; ++k) {
    double nn = std::sqrt(dir[k][0] * dir[k][0] + dir[k][1] * dir[k][1] + dir[k][2] * dir[k][2]);
    const double off = R + 0.5 * gaps[k];
    Sphere a{{at[k][0] - off * dir[k][0] / nn, at[k][1] - off * dir[k][1] / nn,
              at[k][2] - off * dir[k][2] / nn},
             R};
    Sphere b{{at[k][0] + off * dir[k][0] / nn, at[k][1] + off * dir[k][1] / nn,
              at[k][2] + off * dir[k][2] / nn},
             R};
    pk.sp.push_back(a);
    pk.sp.push_back(b);
  }
  Lcg rng{20260824ull};
  int placed = 0;
  for (int tries = 0; tries < 20000 && placed < 44; ++tries) {
    Sphere s{{rng.next(), rng.next(), rng.next()}, 0.07 + 0.03 * rng.next()};
    if (ok(s)) {
      pk.sp.push_back(s);
      ++placed;
    }
  }
  return pk;
}

// ---- octree building ----------------------------------------------------------------------------

struct Geo {
  BO t;
  AmrPoisson<3, 21> pres;
  double h0;  // finest width (level 0)
  Index n = 0;
};

Vec<3> centerOf(const BO& t, double h0, Index i) {
  auto b = t.bounds(i);
  const double s = static_cast<double>(Index(1) << t.level(i));
  Vec<3> c{};
  for (int d = 0; d < 3; ++d)
    c[d] = (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
  return c;
}

/// Refine near the surface toward a per-point target level; iterate to fixpoint + 2:1 balance.
/// targetFn(x) in [0, coarse] (0 = finest). Band predicate matches refineToSdf: surface within
/// halfDiag + band*h(level-1) of the center (the band margin at the level being created).
template <class SdfFn, class TargetFn>
void refineGraded(BO& t, double h0, SdfFn&& sdf, TargetFn&& targetFn, double band) {
  const double halfDiag = 0.5 * std::sqrt(3.0);
  for (;;) {
    std::vector<Code> toRefine;
    for (Index i = 0; i < t.numLeaves(); ++i) {
      const unsigned L = t.level(i);
      if (L == 0)
        continue;
      const Vec<3> c = centerOf(t, h0, i);
      const unsigned tgt = targetFn(c);
      if (L <= tgt)
        continue;
      const double width = h0 * static_cast<double>(Index(1) << L);
      const double hNew = 0.5 * width;
      if (std::fabs(sdf(c)) <= halfDiag * width + band * hNew)
        toRefine.push_back(t.code(i));
    }
    if (toRefine.empty())
      break;
    std::sort(toRefine.begin(), toRefine.end());
    toRefine.erase(std::unique(toRefine.begin(), toRefine.end()), toRefine.end());
    t.refineIf(
        [&](Code c, unsigned) { return std::binary_search(toRefine.begin(), toRefine.end(), c); });
  }
  t.balance2to1();
}

template <class SdfFn, class TargetFn>
Geo buildGeo(unsigned depth, unsigned coarseLevel, SdfFn&& sdf, TargetFn&& targetFn, double band) {
  Geo g;
  g.t = BO(IVec<3>{1, 1, 1}, depth);
  for (unsigned k = 0; k + coarseLevel < depth; ++k)
    g.t.refineIf([](Code, unsigned) { return true; });
  g.h0 = 1.0 / static_cast<double>(1L << depth);
  refineGraded(g.t, g.h0, sdf, targetFn, band);
  g.pres.init(g.t, g.h0);
  g.pres.setOrigin(Vec<3>{0, 0, 0});
  g.n = g.t.numLeaves();
  return g;
}

// ---- census -------------------------------------------------------------------------------------

// Scratch single-row overlay satisfying gpFillRow's OV concept.
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

struct Census {
  long rows = 0;          // non-clean (overlay) rows
  long strictPass = 0;    // current contract holds (all 15 entries same-level)
  long axisPass = 0;      // Option 0' holds (only required entries same-level), strict fails
  long needQuad = 0;      // virtual samples needed, full coarseStar support
  long needLin = 0;       // virtual samples needed, degraded tangential support
  long fallback = 0;      // some required sample unsupported -> fluid-only cascade
  long rowsPerLevel[8] = {};
  long leaves = 0, leavesFinest = 0;
};

Census runCensus(const Geo& g, const Packing& pk) {
  Census cs;
  cs.leaves = g.n;
  const auto& t = g.t;
  const auto& pres = g.pres;
  for (Index i = 0; i < g.n; ++i)
    if (t.level(i) == 0)
      ++cs.leavesFinest;
  for (Index i = 0; i < g.n; ++i) {
    const unsigned Li = t.level(i);
    const double h = g.h0 * static_cast<double>(Index(1) << Li);
    const Vec<3> c = centerOf(t, g.h0, i);
    if (!(pk.sdf(c) > 0.0))
      continue;  // solid-centered: decoupled, no row
    // D2 classification: float SDF at virtual uniform positions center + q*h.
    float Cq[3][5], F[3][4];
    bool clean = true;
    for (int a = 0; a < 3; ++a) {
      for (int q = -2; q <= 2; ++q) {
        Vec<3> p = c;
        p[a] += static_cast<double>(q) * h;
        Cq[a][q + 2] = static_cast<float>(pk.sdf(p));
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
    ++cs.rows;
    ++cs.rowsPerLevel[Li < 8 ? Li : 7];
    // Chains (buildGhostOverlay's hops).
    Index ch[3][5];
    for (int a = 0; a < 3; ++a) {
      ch[a][2] = i;
      ch[a][3] = pres.periodicNeighbor(i, a, +1);
      ch[a][1] = pres.periodicNeighbor(i, a, -1);
      ch[a][4] = ch[a][3] >= 0 ? pres.periodicNeighbor(ch[a][3], a, +1) : ch[a][3];
      ch[a][0] = ch[a][1] >= 0 ? pres.periodicNeighbor(ch[a][1], a, -1) : ch[a][1];
    }
    auto sameLevel = [&](Index j) { return j >= 0 && t.level(j) == Li; };
    bool strict = true;
    for (int a = 0; a < 3; ++a)
      for (int q = 0; q < 5; ++q)
        strict = strict && sameLevel(ch[a][q]);
    if (strict) {
      ++cs.strictPass;
      continue;
    }
    // Required entries: what the (2,2) matrix/divergence deltas actually read per face state.
    // k even (plus closed): QUAD -> q=-1,-2; LIN -> q=-1; EXPLICIT -> q=+1. k odd mirrored.
    bool req[3][5] = {};
    for (int k = 0; k < 6; ++k) {
      const int8_t st = row.statev[k];
      const int a = k / 2;
      const int s = (k & 1) ? +1 : -1;  // fluid side of the closed face
      if (st == scheme::GP_QUAD) {
        req[a][2 + s] = req[a][2 + 2 * s] = true;
      } else if (st == scheme::GP_LIN) {
        req[a][2 + s] = true;
      } else if (st == scheme::GP_EXPLICIT) {
        req[a][2 - s] = true;  // the across-face entry (own closed face's other cell)
      }
    }
    int worst = 0;  // 0 = axis-pass, 1 = quad sample, 2 = linear sample, 3 = fallback
    for (int a = 0; a < 3; ++a)
      for (int q = 0; q < 5; ++q) {
        if (!req[a][q] || q == 2)
          continue;
        const Index j = ch[a][q];
        if (j < 0) {
          worst = 3;
          continue;
        }
        const unsigned Lj = t.level(j);
        if (Lj == Li)
          continue;
        // Virtual sample at the uniform position: support must be fluid.
        Vec<3> pv = c;
        pv[a] += static_cast<double>(q - 2) * h;
        if (Lj < Li) {
          // finer cover: volume-average sample; region-center fluidness as the support proxy.
          worst = std::max(worst, pk.sdf(pv) > 0.0 ? 1 : 3);
        } else {
          // coarser: coarseStar prolongation; needs the coarse cell fluid + tangential
          // neighbours fluid (per tangential axis; a failing axis degrades to linear).
          if (!(pk.sdf(centerOf(t, g.h0, j)) > 0.0)) {
            worst = 3;
            continue;
          }
          int lvl = 1;
          for (int tt = 0; tt < 3; ++tt) {
            if (tt == a)
              continue;
            const Index jp = pres.periodicNeighbor(j, tt, +1);
            const Index jm = pres.periodicNeighbor(j, tt, -1);
            const bool okT = jp >= 0 && jm >= 0 && pk.sdf(centerOf(t, g.h0, jp)) > 0.0 &&
                             pk.sdf(centerOf(t, g.h0, jm)) > 0.0;
            if (!okT)
              lvl = 2;
          }
          worst = std::max(worst, lvl);
        }
      }
    if (worst == 0)
      ++cs.axisPass;
    else if (worst == 1)
      ++cs.needQuad;
    else if (worst == 2)
      ++cs.needLin;
    else
      ++cs.fallback;
  }
  return cs;
}

void printCensus(const char* name, const Census& cs, const Census* baseline) {
  std::printf("\n[%s] leaves %ld (finest %ld", name, cs.leaves, cs.leavesFinest);
  if (baseline)
    std::printf("; %.1f%% of uniform-band cells", 100.0 * cs.leaves / baseline->leaves);
  std::printf(")\n");
  auto pct = [&](long v) { return cs.rows ? 100.0 * v / cs.rows : 0.0; };
  std::printf("  overlay rows %ld: strict-pass %ld (%.1f%%) | axis-pass %ld (%.1f%%) | "
              "quad-sample %ld (%.1f%%) | lin-sample %ld (%.1f%%) | fallback %ld (%.1f%%)\n",
              cs.rows, cs.strictPass, pct(cs.strictPass), cs.axisPass, pct(cs.axisPass),
              cs.needQuad, pct(cs.needQuad), cs.needLin, pct(cs.needLin), cs.fallback,
              pct(cs.fallback));
  const long seam = cs.rows - cs.strictPass;
  std::printf("  seam rows (not strict) %ld (%.1f%% of rows); of seam: axis-pass %.1f%%, "
              "samples %.1f%%, fallback %.1f%%\n",
              seam, pct(seam), seam ? 100.0 * cs.axisPass / seam : 0.0,
              seam ? 100.0 * (cs.needQuad + cs.needLin) / seam : 0.0,
              seam ? 100.0 * cs.fallback / seam : 0.0);
  std::printf("  rows per level:");
  for (int l = 0; l < 8; ++l)
    if (cs.rowsPerLevel[l])
      std::printf("  L%d=%ld", l, cs.rowsPerLevel[l]);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  // Optional: argv[1] = packing file (unit periodic box, lines "x y z r", '#' comments) for a
  // REAL bed census (plan §8 M1 follow-up); argv[2] = octree depth (default 8).
  const unsigned depth = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 8;
  const unsigned coarseLevel = 3;  // background level (coarse bulk)
  const double nGap = 4.0;         // gap floor: gap >= nGap * h(level)
  const double band = 2.0;         // band margin in target-level cells

  std::printf("M1 seam census (plan docs/amr_mixed_level_cut_band_plan.md §8): depth %u "
              "(h0=1/%ld), background L%u, gap floor n=%.0f, band %.0f\n",
              depth, 1L << depth, coarseLevel, nGap, band);

  Packing pk;
  bool realBed = false;
  if (argc > 1) {
    std::FILE* f = std::fopen(argv[1], "r");
    if (!f) {
      std::printf("cannot open %s\n", argv[1]);
      return 1;
    }
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
      if (line[0] == '#')
        continue;
      Sphere s{};
      if (std::sscanf(line, "%lf %lf %lf %lf", &s.c[0], &s.c[1], &s.c[2], &s.r) == 4)
        pk.sp.push_back(s);
    }
    std::fclose(f);
    realBed = true;
    std::printf("packing file %s: %zu spheres, R/h0 = %.1f\n", argv[1], pk.sp.size(),
                pk.sp.empty() ? 0.0 : pk.sp[0].r * static_cast<double>(1L << depth));
  } else {
    pk = buildPacking(1.0 / static_cast<double>(1L << depth));
  }
  double phi = 0.0;
  for (const auto& s : pk.sp)
    phi += 4.0 / 3.0 * 3.14159265358979323846 * s.r * s.r * s.r;
  std::printf("packing: %zu spheres%s, phi ~= %.1f%% (overlaps uncounted)\n", pk.sp.size(),
              realBed ? "" : " (3 engineered throat pairs at 2/6/20 h0)", 100.0 * phi);

  auto sdf = [&](const Vec<3>& p) { return pk.sdf(p); };

  // Map U — uniform finest band (today's policy): baseline economics.
  Geo gU = buildGeo(depth, coarseLevel, sdf, [](const Vec<3>&) { return 0u; }, band);
  Census cU = runCensus(gU, pk);
  printCensus("map U: uniform finest band", cU, nullptr);

  // Map G — gap-graded: coarsest level whose h satisfies the gap floor (plan §7 criterion 1),
  // pointwise (NO D4 quantization/hysteresis — the worst case for seam counts; a production
  // policy only improves on this).
  const double h0 = 1.0 / static_cast<double>(1L << depth);
  auto targetG = [&](const Vec<3>& p) -> unsigned {
    const double gp = pk.gap(p);
    long L = 0;
    while (L < static_cast<long>(coarseLevel) &&
           h0 * static_cast<double>(1L << (L + 1)) * nGap <= gp)
      ++L;
    return static_cast<unsigned>(L);
  };
  Geo gG = buildGeo(depth, coarseLevel, sdf, targetG, band);
  Census cG = runCensus(gG, pk);
  printCensus("map G: gap-graded (pointwise, un-quantized)", cG, &cU);

  if (realBed) {
    std::printf("\ninterpretation: [axis-pass] rows need NO new numerics under Option 0'; "
                "[quad/lin-sample] rows need the D1 virtual-sample machinery; [fallback] rows "
                "degrade within the fluid-only cascade (plan §4.2).\n");
    return 0;
  }

  // Map Q — D4-quantized: TWO surface levels only ({0, 2} — throats/contacts finest, caps at
  // L2; the L1 transition is forced by 2:1 balance but carries no surface patch of its own),
  // with a small dilation of the fine patch for coherence. The un-quantized map G produces
  // concentric level RINGS around every contact (a seam per factor-2 of gap); collapsing the
  // ladder to two surface levels is the D4 move that removes them.
  const double rho = 8.0 * h0;
  auto targetQ = [&](const Vec<3>& p) -> unsigned {
    // fine (L0) iff any point of the dilation ball needs finer than the cap level: gap floor at
    // the cap level L2 is nGap*h(L2).
    const double capGap = nGap * h0 * 4.0;
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          Vec<3> q{p[0] + rho * dx, p[1] + rho * dy, p[2] + rho * dz};
          if (pk.gap(q) < capGap)
            return 0u;
        }
    return 2u;
  };
  Geo gQ = buildGeo(depth, coarseLevel, sdf, targetQ, band);
  Census cQ = runCensus(gQ, pk);
  printCensus("map Q: two surface levels {0,2} + dilation (rho=8 h0)", cQ, &cU);

  // M2 reference geometry: single sphere, hemisphere two-level map -> one clean seam ring.
  Packing one;
  one.sp.push_back(Sphere{{0.513, 0.493, 0.504}, 0.30});
  auto sdf1 = [&](const Vec<3>& p) { return one.sdf(p); };
  Geo gH = buildGeo(depth, coarseLevel, sdf1,
                    [](const Vec<3>& p) { return p[0] < 0.513 ? 0u : 1u; }, band);
  Census cH = runCensus(gH, one);
  Geo gH0 = buildGeo(depth, coarseLevel, sdf1, [](const Vec<3>&) { return 0u; }, band);
  Census cH0 = runCensus(gH0, one);
  printCensus("hemisphere sphere: uniform band (reference)", cH0, nullptr);
  printCensus("hemisphere sphere: two-level (one seam ring)", cH, &cH0);

  std::printf("\ninterpretation: [axis-pass] rows need NO new numerics under Option 0'; "
              "[quad/lin-sample] rows need the D1 virtual-sample machinery; [fallback] rows "
              "degrade within the fluid-only cascade (plan §4.2).\n");
  return 0;
}
#else
#include <cstdio>
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping\n");
  return 0;
}
#endif

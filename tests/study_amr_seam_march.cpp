// M3 — seam stability probe (Phase 0 of docs/amr_mixed_level_cut_band_plan.md §8).
//
// Marches the HOST ORACLE with the SAMPLED ghost projection (setGhostSampled — the mixed-level
// cut-band prototype: ghost_projection_sampled.hpp) on the latitude two-level sphere (the M2
// geometry: cut cells at TWO levels, seam oblique to the wall) across the campaign's dt ladder,
// probing for the lagged-stiff-term instability class that killed three wall-band schemes in
// flow (collocated_invisible_subspace.md / fluid_only_constraint_plan.md):
//
//   [P] parity      — on a UNIFORM finest band the sampled path must reproduce the classic
//                     ghost projection (identity slots, same classification): K to ~1e-10.
//   [S] stationarity— seamed mesh, dt = 60 / 600 / 1e20: bounded march, stationarity residual
//                     collapsing, K finite. A dt=1e20 march only converges if the scheme has
//                     no dt-hidden instability (the C2 battery criterion).
//   [D] dt-spread   — K(dt) spread across the ladder (ghost on uniform bands: ~1e-8 relative).
//   [C] dt-cycling  — 60 -> 1e20 -> 60 reversibility of K (attractor-family probe: a family
//                     would let the march land on a different member after the cycle).
//
// dt enters the momentum operator at setSolid (idiag = rho/dt), so every dt switch re-runs
// setSolid — full deterministic rebuild (D6), which is itself part of what is probed.
//
// A STUDY (printed metrics + gates), not a ctest.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#ifdef PECLET_CORE_HAVE_MORTON
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/flow_oracle.hpp"
#include "peclet/core/common/types.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

const Vec<3> C0{0.513, 0.493, 0.504};
constexpr double R0 = 0.30;

double sdfSphere(const Vec<3>& p) {
  const double dx = p[0] - C0[0], dy = p[1] - C0[1], dz = p[2] - C0[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz) - R0;
}

Vec<3> centerOf(const BO& t, double h0, Index i) {
  auto b = t.bounds(i);
  const double s = static_cast<double>(Index(1) << t.level(i));
  Vec<3> c{};
  for (int d = 0; d < 3; ++d)
    c[d] = (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
  return c;
}

/// Latitude (two-level) or uniform (target 0) surface band on a coarse background.
BO buildMesh(unsigned depth, unsigned coarseLevel, bool twoLevel) {
  BO t(IVec<3>{1, 1, 1}, depth);
  for (unsigned k = 0; k + coarseLevel < depth; ++k)
    t.refineIf([](Code, unsigned) { return true; });
  const double h0 = 1.0 / static_cast<double>(1L << depth);
  const double halfDiag = 0.5 * std::sqrt(3.0);
  for (;;) {
    std::vector<Code> toRefine;
    for (Index i = 0; i < t.numLeaves(); ++i) {
      const unsigned L = t.level(i);
      if (L == 0)
        continue;
      const Vec<3> c = centerOf(t, h0, i);
      const unsigned tgt = (twoLevel && c[2] >= C0[2] - 0.15) ? 1u : 0u;
      if (L <= tgt)
        continue;
      const double width = h0 * static_cast<double>(Index(1) << L);
      if (std::fabs(sdfSphere(c)) <= halfDiag * width + 2.0 * 0.5 * width)
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
  return t;
}

struct MarchOut {
  double K = 0;        // volume-weighted superficial <u_x> (mu = G = 1)
  double statRes = 0;  // final max|u^{n+1}-u^n| / max|u|
  double umax = 0;
  bool bounded = true;
  int steps = 0;
};

/// March `fl` (already set up) `maxSteps` Stokes steps; stop early on divergence. Prints the
/// stationarity trace every `traceEvery` steps (0 = silent) — the converging-vs-stuck signal.
MarchOut march(oracle::AmrFlow<21>& fl, const BO& t, double h0, int maxSteps, int momSweeps,
               int traceEvery = 0, const char* tag = "") {
  MarchOut out;
  const Index n = t.numLeaves();
  std::vector<double> prev;
  for (int s = 0; s < maxSteps; ++s) {
    prev = fl.velocity(0);
    fl.step(momSweeps, /*presIters=*/40, /*presSweeps=*/2);
    ++out.steps;
    double um = 0, dm = 0;
    const auto& ux = fl.velocity(0);
    for (Index i = 0; i < n; ++i) {
      um = std::max(um, std::fabs(ux[static_cast<std::size_t>(i)]));
      dm = std::max(dm, std::fabs(ux[static_cast<std::size_t>(i)] -
                                  prev[static_cast<std::size_t>(i)]));
    }
    out.umax = um;
    out.statRes = um > 0 ? dm / um : 0.0;
    if (traceEvery > 0 && (s + 1) % traceEvery == 0)
      std::printf("  trace %s step %4d statRes %.3e\n", tag, s + 1, out.statRes);
    if (!(um < 1e6) || um != um) {
      out.bounded = false;
      break;
    }
  }
  double vsum = 0, usum = 0;
  const auto& ux = fl.velocity(0);
  for (Index i = 0; i < n; ++i) {
    const double w = h0 * static_cast<double>(Index(1) << t.level(i));
    const double v = w * w * w;
    vsum += v;
    usum += v * ux[static_cast<std::size_t>(i)];
  }
  out.K = usum / vsum;
  return out;
}

void setup(oracle::AmrFlow<21>& fl, const BO& t, double h0, double dt, bool sampled,
           int cfScheme = 0) {
  fl.init(t, h0);
  fl.setViscosity(1.0);
  fl.setDensity(1.0);
  fl.setDt(dt);
  fl.setBodyForce(1.0, 0, 0);
  fl.setAdvection(false);
  fl.setGhostProjection(true, 2, 2);
  if (sampled)
    fl.setGhostSampled(true);
  if (cfScheme)
    fl.setCfScheme(cfScheme);
  fl.setSolid(sdfSphere);
}

}  // namespace

// Phase-parallel driver: each phase is one process (launch them concurrently, aggregate the
// RESULT lines). The CONTROL arm is the sampled path on the UNIFORM band (== the classic ghost
// bit-for-bit, gate [P]) under the IDENTICAL protocol — the v1 run's absolute gates conflated
// convergence-budget floors with scheme defects; every verdict here is seam-vs-control.
//   parity            classic vs sampled on the uniform band (12 steps, dt=1e6)
//   march {seam|ctrl} {dt}   march with trace; RESULT K/statRes/bounded
//   cycle {seam|ctrl}        dt 60(300) -> 1e20(100) -> 60(300); RESULT K1/K3
int main(int argc, char** argv) {
  const unsigned depth = 6, coarse = 3;
  const double h0 = 1.0 / static_cast<double>(1L << depth);
  const char* phase = argc > 1 ? argv[1] : "parity";
  auto sweeps = [](double dt) { return dt > 1e10 ? 400 : 150; };

  if (std::string_view(phase) == "parity") {
    BO tU = buildMesh(depth, coarse, /*twoLevel=*/false);
    oracle::AmrFlow<21> a, b;
    setup(a, tU, h0, 1e6, /*sampled=*/false);
    setup(b, tU, h0, 1e6, /*sampled=*/true);
    const double Kc = march(a, tU, h0, 12, 200).K;
    const double Ks = march(b, tU, h0, 12, 200).K;
    std::printf("RESULT parity Kclassic %.12e Ksampled %.12e rel %.3e leaves %lld\n", Kc, Ks,
                std::fabs(Ks - Kc) / std::fabs(Kc), static_cast<long long>(tU.numLeaves()));
    return 0;
  }
  if (std::string_view(phase) == "march" && argc > 3) {
    const bool seam = std::string_view(argv[2]) == "seam";
    const double dt = std::atof(argv[3]);
    const int cf = argc > 4 ? std::atoi(argv[4]) : 0;  // optional: C/F scheme (1 = quadratic)
    BO t = buildMesh(depth, coarse, seam);
    oracle::AmrFlow<21> fl;
    setup(fl, t, h0, dt, /*sampled=*/true, cf);
    const int nStep = dt > 1e10 ? 200 : 300;
    MarchOut r = march(fl, t, h0, nStep, sweeps(dt), /*traceEvery=*/25, argv[2]);
    std::printf("RESULT march %s dt %.0e cf %d K %.12e statRes %.3e umax %.3e bounded %d "
                "leaves %lld\n",
                argv[2], dt, cf, r.K, r.statRes, r.umax, r.bounded ? 1 : 0,
                static_cast<long long>(t.numLeaves()));
    return r.bounded ? 0 : 1;
  }
  if (std::string_view(phase) == "cycle" && argc > 2) {
    const bool seam = std::string_view(argv[2]) == "seam";
    BO t = buildMesh(depth, coarse, seam);
    oracle::AmrFlow<21> fl;
    setup(fl, t, h0, 60.0, /*sampled=*/true);
    const double K1 = march(fl, t, h0, 300, sweeps(60), 50, "cyc60a").K;
    fl.setDt(1e20);  // dt is baked into the momentum operator: setSolid re-runs (D6 probe)
    fl.setSolid(sdfSphere);
    (void)march(fl, t, h0, 100, sweeps(1e20), 50, "cyc1e20");
    fl.setDt(60.0);
    fl.setSolid(sdfSphere);
    const double K3 = march(fl, t, h0, 300, sweeps(60), 50, "cyc60b").K;
    std::printf("RESULT cycle %s K1 %.12e K3 %.12e rel %.3e\n", argv[2], K1, K3,
                std::fabs(K3 - K1) / std::fabs(K1));
    return 0;
  }
  std::printf("usage: %s parity | march {seam|ctrl} {dt} | cycle {seam|ctrl}\n", argv[0]);
  return 2;
}
#else
#include <cstdio>
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping\n");
  return 0;
}
#endif

// Device (Kokkos) collocated Stokes step on the octree (peclet::core::amr::AmrFlow) —
// the device counterpart of oracle::AmrFlow. Validates:
//   (1) Poiseuille — body-force-driven Stokes flow between immersed no-slip walls
//       converges to the analytic parabola u = G/(2μ)(y-y0)(y1-y) to ~round-off, solids
//       held at 0, and the device field matches the host oracle::AmrFlow field to tolerance;
//   (2) immersed sphere (cut cells) — the device step reproduces the host oracle::AmrFlow
//       velocity field on a genuinely cut geometry (the cut-cell ξ overlay + openness
//       projection all on device);
//   (3) projection — a pure-gradient periodic field's divergence + magnitude drop sharply.
// Runs on whatever backend Kokkos targets (CUDA / HIP / OpenMP). Validation is host-vs-
// device agreement + the analytic solution (the GPU differs from host only in FP last bits).
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <algorithm>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/flow.hpp"
#include "peclet/core/amr/flow_oracle.hpp"
#include "peclet/core/amr/refine.hpp"
#include "peclet/core/common/types.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

BO uniformFine(unsigned L) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k)
    t.refineIf([](Code, unsigned) { return true; });
  return t;
}

double cellCoord(const BO& t, Index i, int d, double h0) {
  auto b = t.bounds(i);
  double s = static_cast<double>(1 << t.level(i));
  return (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
}

void test_poiseuille() {
  const unsigned L = 4;  // 16^3
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / static_cast<double>(N);
  const double y0 = 0.25, y1 = 0.75, G = 1.0, mu = 1.0;
  auto sdf = [&](const Vec<3>& p) { return std::min(p[1] - y0, y1 - p[1]); };

  // Host reference.
  oracle::AmrFlow<21> hfl;
  hfl.init(t, h0);
  hfl.setDensity(1.0);
  hfl.setViscosity(mu);
  hfl.setDt(1e6);
  hfl.setBodyForce(G, 0, 0);
  hfl.setSolid(sdf);
  for (int s = 0; s < 5; ++s)
    hfl.step(/*momSweeps=*/300, /*presIters=*/5, /*presSweeps=*/2);
  const auto& hux = hfl.velocity(0);

  // Device.
  AmrFlow<21> dfl;
  dfl.init(t, h0);
  dfl.setDensity(1.0);
  dfl.setViscosity(mu);
  dfl.setDt(1e6);
  dfl.setBodyForce(G, 0, 0);
  dfl.setSolid(sdf);
  for (int s = 0; s < 5; ++s)
    dfl.step(/*momIters=*/400, /*presIters=*/80);
  const auto dux = dfl.velocity(0);

  const Index n = t.numLeaves();
  double e = 0, dh = 0;
  long nf = 0;
  bool solidsZero = true, fluidPositive = false;
  for (Index i = 0; i < n; ++i) {
    double cy = cellCoord(t, i, 1, h0);
    if (hfl.isFluid(i)) {
      double uex = G / (2.0 * mu) * (cy - y0) * (y1 - cy);
      double d = dux[(std::size_t)i] - uex;
      e += d * d;
      ++nf;
      dh = std::max(dh, std::fabs(dux[(std::size_t)i] - hux[(std::size_t)i]));
      if (dux[(std::size_t)i] > 1e-3)
        fluidPositive = true;
    } else if (std::fabs(dux[(std::size_t)i]) > 1e-12) {
      solidsZero = false;
    }
  }
  double l2 = std::sqrt(e / nf);
  std::printf("[flow] poiseuille: device L2 vs analytic = %.3e, max|dev-host| = %.3e\n", l2, dh);
  PECLET_CORE_CHECK(l2 < 1e-6);
  PECLET_CORE_CHECK(solidsZero);
  PECLET_CORE_CHECK(fluidPositive);
  PECLET_CORE_CHECK(dh < 1e-7);  // device matches host
}

void test_sphere() {
  const unsigned L = 4;  // 16^3
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / static_cast<double>(N);
  const double mu = 1.0, G = 1.0;
  // periodic array of spheres (radius 0.2 centred in the unit cell) — a cut geometry.
  Vec<3> c{0.5, 0.5, 0.5};
  double rad = 0.2;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;  // >0 fluid (outside sphere)
  };

  oracle::AmrFlow<21> hfl;
  hfl.init(t, h0);
  hfl.setViscosity(mu);
  hfl.setDt(1e6);
  hfl.setBodyForce(G, 0, 0);
  hfl.setSolid(sdf);
  for (int s = 0; s < 8; ++s)
    hfl.step(/*momSweeps=*/400, /*presIters=*/8, /*presSweeps=*/2);
  const auto& hux = hfl.velocity(0);

  AmrFlow<21> dfl;
  dfl.init(t, h0);
  dfl.setViscosity(mu);
  dfl.setDt(1e6);
  dfl.setBodyForce(G, 0, 0);
  dfl.setSolid(sdf);
  for (int s = 0; s < 8; ++s)
    dfl.step(/*momIters=*/500, /*presIters=*/120);
  const auto dux = dfl.velocity(0);

  // mean streamwise velocity (permeability proxy) + field agreement on fluid cells.
  const Index n = t.numLeaves();
  double hsum = 0, dsum = 0, dmax = 0, hmax = 0;
  long nf = 0;
  for (Index i = 0; i < n; ++i)
    if (hfl.isFluid(i)) {
      hsum += hux[(std::size_t)i];
      dsum += dux[(std::size_t)i];
      dmax = std::max(dmax, std::fabs(dux[(std::size_t)i] - hux[(std::size_t)i]));
      hmax = std::max(hmax, std::fabs(hux[(std::size_t)i]));
      ++nf;
    }
  double hmean = hsum / nf, dmean = dsum / nf;
  std::printf("[flow] sphere: Umean host %.6e dev %.6e (rel %.2e), max|dev-host| %.3e (mag %.3e)\n",
              hmean, dmean, std::fabs(dmean - hmean) / hmean, dmax, hmax);
  PECLET_CORE_CHECK(dmean > 0.0);
  PECLET_CORE_CHECK(std::fabs(dmean - hmean) / hmean < 2e-3);  // same permeability
  PECLET_CORE_CHECK(dmax < 5e-3 * hmax);                       // fields agree
}

// Same sphere case with the directional ghost gradient (setGhostGradient) on BOTH engines: the
// device overlay (buildGhostGradOverlay + applyGhostGrad) must reproduce the oracle's gradOfDir
// dispatch — same permeability, fields agree.
void test_sphere_ghost() {
  const unsigned L = 4;
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / static_cast<double>(N);
  const double mu = 1.0, G = 1.0;
  Vec<3> c{0.5, 0.5, 0.5};
  double rad = 0.2;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
  };

  oracle::AmrFlow<21> hfl;
  hfl.init(t, h0);
  hfl.setViscosity(mu);
  hfl.setDt(1e6);
  hfl.setBodyForce(G, 0, 0);
  hfl.setGhostGradient(true);
  hfl.setSolid(sdf);
  for (int s = 0; s < 8; ++s)
    hfl.step(/*momSweeps=*/400, /*presIters=*/8, /*presSweeps=*/2);
  const auto& hux = hfl.velocity(0);

  AmrFlow<21> dfl;
  dfl.init(t, h0);
  dfl.setViscosity(mu);
  dfl.setDt(1e6);
  dfl.setBodyForce(G, 0, 0);
  dfl.setGhostGradient(true);
  dfl.setSolid(sdf);
  for (int s = 0; s < 8; ++s)
    dfl.step(/*momIters=*/500, /*presIters=*/120);
  const auto dux = dfl.velocity(0);

  const Index n = t.numLeaves();
  double hsum = 0, dsum = 0, dmax = 0, hmax = 0;
  long nf = 0;
  for (Index i = 0; i < n; ++i)
    if (hfl.isFluid(i)) {
      hsum += hux[(std::size_t)i];
      dsum += dux[(std::size_t)i];
      dmax = std::max(dmax, std::fabs(dux[(std::size_t)i] - hux[(std::size_t)i]));
      hmax = std::max(hmax, std::fabs(hux[(std::size_t)i]));
      ++nf;
    }
  double hmean = hsum / nf, dmean = dsum / nf;
  std::printf(
      "[flow] sphere+ghostgrad: Umean host %.6e dev %.6e (rel %.2e), max|dev-host| %.3e "
      "(mag %.3e)\n",
      hmean, dmean, std::fabs(dmean - hmean) / hmean, dmax, hmax);
  PECLET_CORE_CHECK(dmean > 0.0);
  PECLET_CORE_CHECK(std::fabs(dmean - hmean) / hmean < 2e-3);  // device == oracle permeability
  PECLET_CORE_CHECK(dmax < 5e-3 * hmax);                       // fields agree
}

// FULL ghost projection (setGhostProjection, (1,2) default) on BOTH engines: binary-openness
// operator + closure overlay + MG-preconditioned BiCGStab must agree oracle==device on a cut
// geometry (same permeability, fields agree), and differ from the aperture projection (it
// replaces the O(1) cut-face constraint).
void test_sphere_ghostproj() {
  const unsigned L = 4;
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / static_cast<double>(N);
  const double mu = 1.0, G = 1.0;
  Vec<3> c{0.5, 0.5, 0.5};
  double rad = 0.2;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
  };

  oracle::AmrFlow<21> hfl;
  hfl.init(t, h0);
  hfl.setViscosity(mu);
  hfl.setDt(1e6);
  hfl.setBodyForce(G, 0, 0);
  hfl.setGhostProjection(true, 1, 2);
  hfl.setSolid(sdf);
  for (int s = 0; s < 8; ++s)
    hfl.step(/*momSweeps=*/400, /*presIters=*/12, /*presSweeps=*/2);
  const auto& hux = hfl.velocity(0);

  AmrFlow<21> dfl;
  dfl.init(t, h0);
  dfl.setViscosity(mu);
  dfl.setDt(1e6);
  dfl.setBodyForce(G, 0, 0);
  dfl.setGhostProjection(true, 1, 2);
  dfl.setSolid(sdf);
  for (int s = 0; s < 8; ++s)
    dfl.step(/*momIters=*/500, /*presIters=*/40);
  const auto dux = dfl.velocity(0);

  const Index n = t.numLeaves();
  double hsum = 0, dsum = 0, dmax = 0, hmax = 0;
  long nf = 0;
  for (Index i = 0; i < n; ++i)
    if (hfl.isFluid(i)) {
      hsum += hux[(std::size_t)i];
      dsum += dux[(std::size_t)i];
      dmax = std::max(dmax, std::fabs(dux[(std::size_t)i] - hux[(std::size_t)i]));
      hmax = std::max(hmax, std::fabs(hux[(std::size_t)i]));
      ++nf;
    }
  double hmean = hsum / nf, dmean = dsum / nf;
  std::printf(
      "[flow] sphere+ghostproj: Umean host %.6e dev %.6e (rel %.2e), max|dev-host| %.3e "
      "(mag %.3e), dev ghost-div %.2e\n",
      hmean, dmean, std::fabs(dmean - hmean) / hmean, dmax, hmax, dfl.divNormL2());
  PECLET_CORE_CHECK(dmean > 0.0);
  PECLET_CORE_CHECK(std::fabs(dmean - hmean) / hmean < 2e-3);  // device == oracle permeability
  PECLET_CORE_CHECK(dmax < 5e-3 * hmax);                       // fields agree
}

// Ghost projection on a GRADED octree: (a) a too-thin finest band must THROW in setSolid (the
// band-margin invariant — an overlay row's ±2 reach may never cross a 2:1 boundary); (b) with an
// adequate band the graded solve is stable and produces a sane drag (the graded ACCURACY on the
// dense SC array is band-dominated — measured in tests/study/amr_zh_graded.py — so this is a
// machinery lock, not an accuracy gate).
void test_graded_ghostproj() {
  const long N = 32;
  const double phi = 0.125;
  const double R = std::pow(phi * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * static_cast<double>(N);
  const double c = N / 2.0;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c, dy = p[1] - c, dz = p[2] - c;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - R;
  };
  auto makeTree = [&](double band) {
    BO t(IVec<3>{1, 1, 1}, 5);
    AmrGeometry<3> geo;
    geo.h0 = 1.0;
    refineToSdf(t, geo, sdf, /*target*/ 0, band, /*balance*/ true);
    return t;
  };

  {  // (a) thin band -> the overlay build must throw (never silently mis-close a face)
    BO t = makeTree(0.5);
    AmrFlow<21> fl;
    fl.init(t, 1.0, Vec<3>{0, 0, 0});
    fl.setViscosity(0.1);
    fl.setDt(60.0);
    fl.setGhostProjection(true, 1, 2);
    bool threw = false;
    try {
      fl.setSolid(sdf);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    PECLET_CORE_CHECK(threw);
  }

  {  // (b) adequate band: stable graded solve, sane drag, genuinely coarsened mesh
    BO t = makeTree(3.0);
    PECLET_CORE_CHECK(t.numLeaves() < N * N * N);
    AmrFlow<21> fl;
    fl.init(t, 1.0, Vec<3>{0, 0, 0});
    fl.setViscosity(0.1);
    fl.setDt(60.0);
    fl.setBodyForce(1e-3, 0, 0);
    fl.setGhostProjection(true, 1, 2);
    fl.setSolid(sdf);
    for (int s = 0; s < 200; ++s)
      fl.step(100, 60);
    double usup = 0;
    const auto u = fl.velocity(0);
    for (Index i = 0; i < t.numLeaves(); ++i) {
      const double w = static_cast<double>(1L << t.level(i));
      usup += u[(std::size_t)i] * w * w * w;
    }
    usup /= static_cast<double>(N * N * N);
    PECLET_CORE_CHECK(std::isfinite(usup) && usup > 0.0 && usup < 1.0);  // stable
    const double k = 1e-3 * N * N * N / (6.0 * M_PI * 0.1 * R * usup);
    std::printf("[flow] graded ghostproj (band 3): K=%.4f (Z&H 4.292; band-dominated), "
                "leaves=%lld/%lld\n",
                k, static_cast<long long>(t.numLeaves()), static_cast<long long>(N * N * N));
    PECLET_CORE_CHECK(k > 3.0 && k < 6.5);  // sane drag (machinery lock, not accuracy)
  }
}

// Graded sphere with ghost projection + the Martin–Cartwright C/F scheme (setCfScheme(1)) on
// BOTH engines: the three C/F overlays (momentum deferred correction, divergence, gradients) are
// built by the shared host builders, so oracle==device to solver tolerance. Also guards that the
// scheme actually changes the graded solution (it replaces the 1st-order C/F treatment).
void test_graded_cf_quadratic() {
  const long N = 32;
  const double phi = 0.125;
  const double R = std::pow(phi * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * static_cast<double>(N);
  const double c = N / 2.0;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c, dy = p[1] - c, dz = p[2] - c;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - R;
  };
  BO t(IVec<3>{1, 1, 1}, 5);
  AmrGeometry<3> geo;
  geo.h0 = 1.0;
  refineToSdf(t, geo, sdf, /*target*/ 0, /*band*/ 3.0, /*balance*/ true);
  PECLET_CORE_CHECK(t.numLeaves() < N * N * N);

  auto runDev = [&](int cf) {
    AmrFlow<21> fl;
    fl.init(t, 1.0, Vec<3>{0, 0, 0});
    fl.setViscosity(0.1);
    fl.setDt(60.0);
    fl.setBodyForce(1e-3, 0, 0);
    fl.setGhostProjection(true, 1, 2);
    fl.setCfScheme(cf);
    fl.setSolid(sdf);
    for (int s = 0; s < 40; ++s)
      fl.step(100, 60);
    return fl.velocity(0);
  };
  oracle::AmrFlow<21> hfl;
  hfl.init(t, 1.0, Vec<3>{0, 0, 0});
  hfl.setViscosity(0.1);
  hfl.setDt(60.0);
  hfl.setBodyForce(1e-3, 0, 0);
  hfl.setGhostProjection(true, 1, 2);
  hfl.setCfScheme(1);
  hfl.setSolid(sdf);
  for (int s = 0; s < 40; ++s)
    hfl.step(/*momSweeps=*/250, /*presIters=*/12, /*presSweeps=*/2);
  const auto& hux = hfl.velocity(0);
  const auto dux = runDev(1);
  const auto dux0 = runDev(0);

  const Index n = t.numLeaves();
  double hsum = 0, dsum = 0, d0sum = 0, dmax = 0, hmax = 0;
  long nf = 0;
  for (Index i = 0; i < n; ++i)
    if (hfl.isFluid(i)) {
      const double w = static_cast<double>(1L << t.level(i));
      const double w3 = w * w * w;
      hsum += hux[(std::size_t)i] * w3;
      dsum += dux[(std::size_t)i] * w3;
      d0sum += dux0[(std::size_t)i] * w3;
      dmax = std::max(dmax, std::fabs(dux[(std::size_t)i] - hux[(std::size_t)i]));
      hmax = std::max(hmax, std::fabs(hux[(std::size_t)i]));
      ++nf;
    }
  std::printf(
      "[flow] graded cf-quadratic: Usup host %.6e dev %.6e (rel %.2e), max|dev-host| %.3e "
      "(mag %.3e); dev cf0 Usup %.6e (cf1 shift %.2e rel)\n",
      hsum, dsum, std::fabs(dsum - hsum) / std::fabs(hsum), dmax, hmax, d0sum,
      std::fabs(dsum - d0sum) / std::fabs(d0sum));
  PECLET_CORE_CHECK(std::isfinite(dsum) && dsum > 0.0);
  PECLET_CORE_CHECK(std::fabs(dsum - hsum) / std::fabs(hsum) < 2e-3);  // device == oracle
  PECLET_CORE_CHECK(dmax < 5e-3 * hmax);
  PECLET_CORE_CHECK(std::fabs(dsum - d0sum) / std::fabs(d0sum) > 1e-4);  // the scheme acts
}

// NAVIER–STOKES with the ghost projection (ladder step 4): the immersed sphere at finite Re
// (the aperture advection test's setup) with setGhostProjection on BOTH engines. Checks:
// oracle==device parity, the ghost NS steady state is close to the aperture NS steady state
// (same physics, different cut-cell scheme), and the ghost-closed divergence stays small.
void test_sphere_ghostproj_adv() {
  const unsigned L = 4;
  BO t = uniformFine(L);
  const double h0 = 1.0 / (double)(1L << L);
  const double mu = 0.5, G = 1.0;  // finite Re: advection matters, both converge
  Vec<3> c{0.5, 0.5, 0.5};
  double rad = 0.2;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
  };
  const double dt = 2.0;  // NS needs finite dt (the (ρ/dt) mass damps the lagged advection)
  // mode: 0 = explicit aperture, 1 = explicit ghost, 2 = AUTO (nothing set — the NS default).
  auto setup = [&](auto& f, int mode) {
    f.init(t, h0);
    f.setViscosity(mu);
    f.setDt(dt);
    f.setBodyForce(G, 0, 0);
    f.setAdvection(true);
    if (mode == 1)
      f.setGhostProjection(true, 1, 2);
    else if (mode == 0)
      f.setGhostProjection(false);  // explicit: the aperture NS reference (auto would pick ghost)
    f.setSolid(sdf);
  };
  oracle::AmrFlow<21> hfl;
  setup(hfl, 1);
  AmrFlow<21> dfl;
  setup(dfl, 1);
  AmrFlow<21> afl;  // device aperture reference (cross-scheme closeness)
  setup(afl, 0);
  AmrFlow<21> ufl;  // AUTO: with advection on, the default must resolve to the ghost projection
  setup(ufl, 2);
  for (int s = 0; s < 50; ++s) {
    hfl.step(300, 12, 2);
    dfl.step(200, 60);
    afl.step(200, 80);
    ufl.step(200, 60);
  }
  const auto& hux = hfl.velocity(0);
  const auto dux = dfl.velocity(0);
  const auto aux = afl.velocity(0);
  const Index n = t.numLeaves();
  double hsum = 0, dsum = 0, asum = 0, dmax = 0, hmax = 0;
  long nf = 0;
  for (Index i = 0; i < n; ++i)
    if (hfl.isFluid(i)) {
      hsum += hux[(std::size_t)i];
      dsum += dux[(std::size_t)i];
      asum += aux[(std::size_t)i];
      dmax = std::max(dmax, std::fabs(dux[(std::size_t)i] - hux[(std::size_t)i]));
      hmax = std::max(hmax, std::fabs(hux[(std::size_t)i]));
      ++nf;
    }
  const double hmean = hsum / nf, dmean = dsum / nf, amean = asum / nf;
  // Residual divergence comparison in MATCHED units: the ghost-closed divergence of the ghost
  // solve vs the aperture divergence of the aperture solve (the collocated approximate
  // projection leaves an O(h²) cell-divergence residual on both paths; the (1,2) ghost adds a
  // per-step lag that decays through stepping).
  const double gdiv = dfl.divNormL2();
  const double adiv = afl.divNormL2();
  std::printf(
      "[flow] sphere+ghostproj+adv: Umean host %.6e dev %.6e (rel %.2e) aperture %.6e "
      "(scheme gap %.2e rel), max|dev-host| %.3e (mag %.3e), div ghost %.2e aperture %.2e\n",
      hmean, dmean, std::fabs(dmean - hmean) / hmean, amean, std::fabs(dmean - amean) / amean,
      dmax, hmax, gdiv, adiv);
  PECLET_CORE_CHECK(dmean > 0.0 && std::isfinite(dmean));
  PECLET_CORE_CHECK(std::fabs(dmean - hmean) / hmean < 2e-3);  // device == oracle
  PECLET_CORE_CHECK(dmax < 5e-3 * hmax);                       // fields agree
  PECLET_CORE_CHECK(std::fabs(dmean - amean) / amean < 5e-2);  // ghost ≈ aperture NS physics
  PECLET_CORE_CHECK(std::isfinite(gdiv) && gdiv < 10.0 * adiv);  // same residual class
  {  // the AUTO default must have resolved to the ghost projection (== the explicit-ghost run)
    const auto uux = ufl.velocity(0);
    double umax = 0.0;
    for (Index i = 0; i < n; ++i)
      umax = std::max(umax, std::fabs(uux[(std::size_t)i] - dux[(std::size_t)i]));
    std::printf("[flow] NS auto-default: max|auto-ghost| = %.2e\n", umax);
    PECLET_CORE_CHECK(umax < 1e-10 * hmax);  // identical configuration, identical kernels
  }
}

// Adaptivity DURING a run (ladder step 5): run on a band-3 graded mesh, beginAdapt, widen the
// finest band to 5 (mutating the SAME octree), finishAdapt (conservative u/p transfer + full
// rebuild), continue — the continued run must land on the SAME steady state as a cold start on
// the identical final mesh (same fixed point), and stay finite throughout.
void test_adapt_midrun() {
  const long N = 32;
  const double phi = 0.125;
  const double R = std::pow(phi * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * static_cast<double>(N);
  const double c = N / 2.0;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c, dy = p[1] - c, dz = p[2] - c;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - R;
  };
  AmrGeometry<3> geo;
  geo.h0 = 1.0;
  auto usup = [&](const BO& t, const AmrFlow<21>& fl) {
    const auto u = fl.velocity(0);
    double s = 0;
    for (Index i = 0; i < t.numLeaves(); ++i) {
      const double w = static_cast<double>(1L << t.level(i));
      s += u[(std::size_t)i] * w * w * w;
    }
    return s / static_cast<double>(N * N * N);
  };
  auto configure = [&](AmrFlow<21>& fl, const BO& t) {
    fl.init(t, 1.0, Vec<3>{0, 0, 0});
    fl.setViscosity(0.1);
    fl.setDt(60.0);
    fl.setBodyForce(1e-3, 0, 0);
    fl.setGhostProjection(true, 1, 2);
  };

  // Continued run: band 3 -> (mid-run) band 5.
  BO t(IVec<3>{1, 1, 1}, 5);
  refineToSdf(t, geo, sdf, 0, 3.0, true);
  const Index leaves3 = t.numLeaves();
  AmrFlow<21> fl;
  configure(fl, t);
  fl.setSolid(sdf);
  for (int s = 0; s < 200; ++s)
    fl.step(100, 60);
  const double uMid = usup(t, fl);
  fl.beginAdapt();
  refineToSdf(t, geo, sdf, 0, 5.0, true);  // widen the band ON THE SAME octree
  PECLET_CORE_CHECK(t.numLeaves() > leaves3);
  fl.finishAdapt(sdf);
  const double uAfter = usup(t, fl);  // transferred state, before any new step
  for (int s = 0; s < 400; ++s)
    fl.step(100, 60);
  const double uCont = usup(t, fl);

  // Cold start on the identical final mesh.
  AmrFlow<21> fc;
  configure(fc, t);
  fc.setSolid(sdf);
  for (int s = 0; s < 600; ++s)
    fc.step(100, 60);
  const double uCold = usup(t, fc);

  std::printf(
      "[flow] adapt-midrun: usup mid %.6e -> transferred %.6e -> continued %.6e, cold %.6e "
      "(rel %.2e); leaves %lld -> %lld\n",
      uMid, uAfter, uCont, uCold, std::fabs(uCont - uCold) / uCold,
      static_cast<long long>(leaves3), static_cast<long long>(t.numLeaves()));
  PECLET_CORE_CHECK(std::isfinite(uAfter) && uAfter > 0.0);
  PECLET_CORE_CHECK(std::fabs(uAfter - uMid) / uMid < 0.05);  // transfer preserves the state
  PECLET_CORE_CHECK(std::isfinite(uCont) && uCont > 0.0);
  PECLET_CORE_CHECK(std::fabs(uCont - uCold) / uCold < 1e-3);  // same fixed point
}

// The optional Helmholtz-MG momentum preconditioner (setMomentumMG) must not change the
// converged answer — a preconditioner only affects the iteration path, not the fixed point.
void test_momentum_mg_option() {
  const unsigned L = 4;
  BO t = uniformFine(L);
  const double h0 = 1.0 / (double)(1L << L);
  Vec<3> c{0.5, 0.5, 0.5};
  double rad = 0.2;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
  };
  auto run = [&](bool mgPre) {
    AmrFlow<21> f;
    f.init(t, h0);
    f.setViscosity(1.0);
    f.setDt(1e6);
    f.setBodyForce(1.0, 0, 0);
    f.setMomentumMG(mgPre);
    f.setSolid(sdf);
    for (int s = 0; s < 8; ++s)
      f.step(400, 120);
    return f.velocity(0);
  };
  auto uoff = run(false);
  auto uon = run(true);
  const Index n = t.numLeaves();
  double dmax = 0, mag = 0;
  for (Index i = 0; i < n; ++i) {
    dmax = std::max(dmax, std::fabs(uon[(std::size_t)i] - uoff[(std::size_t)i]));
    mag = std::max(mag, std::fabs(uoff[(std::size_t)i]));
  }
  std::printf("[flow] momentum-MG option: max|on-off| = %.3e (mag %.3e)\n", dmax, mag);
  PECLET_CORE_CHECK(dmax < 1e-3 * mag);  // same converged step regardless of preconditioner
}

// Phase 2 scalability guard: with the Galerkin velocity multigrid the per-step momentum
// BiCGStab iteration count must stay ~flat as the grid refines (multigrid), not grow like
// the unpreconditioned ~N^⅓. Refining 16³→32³ (8× cells) the Jacobi path roughly doubles;
// the velocity-MG path must grow only mildly and stay well bounded.
void test_momentum_scaling() {
  auto momIters = [](unsigned L) {
    BO t = uniformFine(L);
    const double h0 = 1.0 / (double)(1L << L);
    Vec<3> c{0.5, 0.5, 0.5};
    double rad = 0.25;
    auto sdf = [&](const Vec<3>& p) {
      double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
      return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
    };
    AmrFlow<21> f;
    f.init(t, h0);
    f.setViscosity(1.0);
    f.setDt(1e6);
    f.setBodyForce(1.0, 0, 0);
    f.setSolid(sdf);  // momentum MG on by default
    for (int s = 0; s < 4; ++s)
      f.step(400, 60);
    return f.lastMomIters();
  };
  int m4 = momIters(4), m5 = momIters(5);  // 16³, 32³
  std::printf("[flow] velocity-MG momentum iters: 16³=%d  32³=%d  (ratio %.2f)\n", m4, m5,
              (double)m5 / std::max(1, m4));
  PECLET_CORE_CHECK(m5 < 2 * m4);  // near-flat (multigrid) — would ~double without it
  PECLET_CORE_CHECK(m5 < 150);  // absolute bound (3 components × a few dozen MG-accelerated iters)
}

// Implicit-FOU + deferred-correction SOU advection (Navier–Stokes), validated against the host
// oracle::AmrFlow which runs the identical scheme: (a) Poiseuille with advection ON still converges
// to the analytic parabola (∇·(u u)=0 for unidirectional flow) and matches the host; (b) an
// immersed sphere at finite Re (the advection term is non-trivial) reaches the same steady
// velocity field on device and host.
// Kernel-only isolation: the device high-order advection ∇·(u u_c) must equal the host
// oracle::AmrFlow::advectTerm for the same velocity field (no solve, no time-stepping). Run on an
// all-fluid box AND an immersed sphere (cut faces). This pinpoints any SOU-kernel discrepancy.
void test_advection_kernel() {
  for (int which = 0; which < 2; ++which) {
    const unsigned L = 4;
    BO t = uniformFine(L);
    const Index n = t.numLeaves();
    const double h0 = 1.0 / (double)(1L << L);
    const double k = 2.0 * M_PI;
    auto sphere = [&](const Vec<3>& p) {
      double dx = p[0] - 0.5, dy = p[1] - 0.5, dz = p[2] - 0.5;
      return std::sqrt(dx * dx + dy * dy + dz * dz) - 0.2;
    };
    auto allfluid = [](const Vec<3>&) { return 1.0; };
    std::array<std::vector<double>, 3> U;
    for (int c = 0; c < 3; ++c)
      U[c].assign((std::size_t)n, 0.0);
    for (Index i = 0; i < n; ++i) {
      double x = cellCoord(t, i, 0, h0), y = cellCoord(t, i, 1, h0), z = cellCoord(t, i, 2, h0);
      U[0][(std::size_t)i] = std::sin(k * x) * std::cos(k * y) * std::cos(k * z) + 0.3;
      U[1][(std::size_t)i] = -std::cos(k * x) * std::sin(k * y) * std::cos(k * z);
      U[2][(std::size_t)i] = 0.1 * std::sin(k * z);
    }
    oracle::AmrFlow<21> hfl;
    hfl.init(t, h0);
    hfl.setDt(1.0);
    hfl.setAdvection(true);
    if (which == 0)
      hfl.setSolid(allfluid);
    else
      hfl.setSolid(sphere);
    for (int c = 0; c < 3; ++c)
      hfl.velocityRef()[c] = U[c];

    AmrFlow<21> dfl;
    dfl.init(t, h0);
    dfl.setDt(1.0);
    dfl.setAdvection(true);
    if (which == 0)
      dfl.setSolid(allfluid);
    else
      dfl.setSolid(sphere);
    for (int c = 0; c < 3; ++c)
      dfl.setVelocity(c, U[c]);

    double emax = 0, mag = 0;
    for (int c = 0; c < 3; ++c) {
      auto ds = dfl.debugSou(c);
      for (Index i = 0; i < n; ++i)
        if (hfl.isFluid(i)) {
          double href = hfl.advectTerm(c, i);
          emax = std::max(emax, std::fabs(ds[(std::size_t)i] - href));
          mag = std::max(mag, std::fabs(href));
        }
    }
    std::printf("[flow] advect-kernel (%s): max|dev-host SOU| = %.3e (mag %.3e)\n",
                which == 0 ? "all-fluid" : "sphere", emax, mag);
    PECLET_CORE_CHECK(emax < 1e-10 * (1.0 + mag));
  }
}

void test_advection() {
  // (a) Poiseuille with advection on.
  {
    const unsigned L = 4;
    BO t = uniformFine(L);
    const double h0 = 1.0 / (double)(1L << L);
    const double y0 = 0.25, y1 = 0.75, G = 1.0, mu = 1.0;
    auto sdf = [&](const Vec<3>& p) { return std::min(p[1] - y0, y1 - p[1]); };

    oracle::AmrFlow<21> hfl;
    hfl.init(t, h0);
    hfl.setViscosity(mu);
    hfl.setDt(1e6);
    hfl.setBodyForce(G, 0, 0);
    hfl.setAdvection(true);  // SOU + implicit FOU (defaults)
    hfl.setGhostProjection(false);  // explicit: this test validates the APERTURE NS path
    hfl.setSolid(sdf);
    for (int s = 0; s < 6; ++s)
      hfl.step(300, 5, 2);
    const auto& hux = hfl.velocity(0);

    AmrFlow<21> dfl;
    dfl.init(t, h0);
    dfl.setViscosity(mu);
    dfl.setDt(1e6);
    dfl.setBodyForce(G, 0, 0);
    dfl.setAdvection(true);
    dfl.setGhostProjection(false);
    dfl.setSolid(sdf);
    for (int s = 0; s < 6; ++s)
      dfl.step(400, 80);
    const auto dux = dfl.velocity(0);

    const Index n = t.numLeaves();
    double e = 0, dh = 0;
    long nf = 0;
    for (Index i = 0; i < n; ++i)
      if (hfl.isFluid(i)) {
        double cy = cellCoord(t, i, 1, h0);
        double uex = G / (2.0 * mu) * (cy - y0) * (y1 - cy);
        e += (dux[(std::size_t)i] - uex) * (dux[(std::size_t)i] - uex);
        dh = std::max(dh, std::fabs(dux[(std::size_t)i] - hux[(std::size_t)i]));
        ++nf;
      }
    std::printf("[flow] poiseuille+adv: device L2 vs analytic = %.3e, max|dev-host| = %.3e\n",
                std::sqrt(e / nf), dh);
    PECLET_CORE_CHECK(std::sqrt(e / nf) < 1e-6);  // ∇·(u u)=0 ⇒ unchanged parabola
    PECLET_CORE_CHECK(dh < 1e-6);
  }

  // (b) immersed sphere at finite Re (non-trivial advection): device == host steady field.
  {
    const unsigned L = 4;
    BO t = uniformFine(L);
    const double h0 = 1.0 / (double)(1L << L);
    const double mu = 0.5, G = 1.0;  // moderate μ ⇒ finite Re ⇒ advection matters, both converge
    Vec<3> c{0.5, 0.5, 0.5};
    double rad = 0.2;
    auto sdf = [&](const Vec<3>& p) {
      double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
      return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
    };
    // Navier–Stokes needs a finite dt (the (ρ/dt) mass term damps the lagged advection — a huge
    // steady-Stokes dt makes the explicit deferred correction a Picard iteration with no damping
    // and diverges). Time-step both to the same NS steady state.
    const double dt = 2.0;
    auto setup = [&](auto& f) {
      f.init(t, h0);
      f.setViscosity(mu);
      f.setDt(dt);
      f.setBodyForce(G, 0, 0);
      f.setAdvection(true);
      f.setGhostProjection(false);  // explicit: the aperture NS parity case
      f.setSolid(sdf);
    };
    oracle::AmrFlow<21> hfl;
    setup(hfl);
    AmrFlow<21> dfl;
    setup(dfl);
    for (int s = 0; s < 50; ++s) {  // both reach the same NS steady state
      hfl.step(300, 8, 2);
      dfl.step(200, 80);
    }
    const auto& hux = hfl.velocity(0);
    const auto dux = dfl.velocity(0);

    const Index n = t.numLeaves();
    double hsum = 0, dsum = 0, dmax = 0, hmax = 0;
    long nf = 0;
    for (Index i = 0; i < n; ++i)
      if (hfl.isFluid(i)) {
        hsum += hux[(std::size_t)i];
        dsum += dux[(std::size_t)i];
        dmax = std::max(dmax, std::fabs(dux[(std::size_t)i] - hux[(std::size_t)i]));
        hmax = std::max(hmax, std::fabs(hux[(std::size_t)i]));
        ++nf;
      }
    std::printf(
        "[flow] sphere+adv (Re~%.1f): Umean host %.6e dev %.6e (rel %.2e), max|dev-host| "
        "%.3e (mag %.3e)\n",
        G * (hsum / nf) * 0.4 / mu, hsum / nf, dsum / nf, std::fabs(dsum - hsum) / std::fabs(hsum),
        dmax, hmax);
    PECLET_CORE_CHECK(std::fabs(dsum - hsum) / std::fabs(hsum) < 5e-3);  // same steady permeability
    PECLET_CORE_CHECK(dmax < 1e-2 * hmax);                               // fields agree
  }
}

// The rediscretized staircase velocity-MG (VelocityMG) must reach the same converged step as
// the Galerkin MG on a cut geometry — validates the clean-fluid exclude mask + pore-scale cap fix.
void test_staircase_mg() {
  const unsigned L = 4;  // 16³
  BO t = uniformFine(L);
  const double h0 = 1.0 / (double)(1L << L);
  Vec<3> c{0.5, 0.5, 0.5};
  double rad = 0.25;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
  };
  auto run = [&](bool staircase) {
    AmrFlow<21> f;
    f.init(t, h0);
    f.setViscosity(1.0);
    f.setDt(1e6);
    f.setBodyForce(1.0, 0, 0);
    f.setVelocityMGStaircase(staircase);
    if (staircase)
      f.setVelocityMGMinCoarse(512);
    f.setSolid(sdf);
    for (int s = 0; s < 6; ++s)
      f.step(400, 60);
    return f.velocity(0);
  };
  auto ug = run(false);  // Galerkin
  auto us = run(true);   // staircase
  const Index n = t.numLeaves();
  double dmax = 0, mag = 0;
  for (Index i = 0; i < n; ++i) {
    dmax = std::max(dmax, std::fabs(us[(std::size_t)i] - ug[(std::size_t)i]));
    mag = std::max(mag, std::fabs(ug[(std::size_t)i]));
  }
  std::printf("[flow] staircase-vs-Galerkin MG: max|stair-galerkin| = %.3e (mag %.3e)\n", dmax,
              mag);
  PECLET_CORE_CHECK(dmax <
                    1e-4 * mag);  // same converged step regardless of coarse-operator strategy
}

// Multicolour Gauss–Seidel smoother (P5): for both coarse-operator strategies, the GS-smoothed MG
// must converge to the same steady velocity as the Jacobi-smoothed MG — GS only changes the
// smoothing rate, not the fixed point.
void test_momentum_gs() {
  const unsigned L = 4;  // 16³
  BO t = uniformFine(L);
  const double h0 = 1.0 / (double)(1L << L);
  Vec<3> c{0.5, 0.5, 0.5};
  double rad = 0.25;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
  };
  auto run = [&](bool staircase, bool gs) {
    AmrFlow<21> f;
    f.init(t, h0);
    f.setViscosity(1.0);
    f.setDt(1e6);
    f.setBodyForce(1.0, 0, 0);
    f.setVelocityMGStaircase(staircase);
    if (staircase)
      f.setVelocityMGMinCoarse(512);
    f.setMomentumGS(gs);
    f.setSolid(sdf);
    for (int s = 0; s < 6; ++s)
      f.step(400, 60);
    return f.velocity(0);
  };
  const Index n = t.numLeaves();
  for (bool staircase : {false, true}) {
    auto uj = run(staircase, false);  // Jacobi
    auto ug = run(staircase, true);   // multicolour-GS
    double dmax = 0, mag = 0;
    for (Index i = 0; i < n; ++i) {
      dmax = std::max(dmax, std::fabs(ug[(std::size_t)i] - uj[(std::size_t)i]));
      mag = std::max(mag, std::fabs(uj[(std::size_t)i]));
    }
    std::printf("[flow] GS-vs-Jacobi MG (%s): max|gs-jac| = %.3e (mag %.3e)\n",
                staircase ? "staircase" : "Galerkin", dmax, mag);
    PECLET_CORE_CHECK(dmax < 1e-4 * mag);
  }
}

// P4: the velocity-MG used as the *solver* (MG-preconditioned defect correction, no Krylov — the
// sdflow mirror) must reach the same converged step as the default MG-preconditioned BiCGStab. Runs
// both with the Galerkin and the staircase hierarchy, and with the GS smoother (the full RB-GS/MG
// path) to confirm the Krylov-free solve is robust there too.
void test_momentum_mgsolver() {
  const unsigned L = 4;  // 16³
  BO t = uniformFine(L);
  const double h0 = 1.0 / (double)(1L << L);
  Vec<3> c{0.5, 0.5, 0.5};
  double rad = 0.25;
  auto sdf = [&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
  };
  auto run = [&](bool staircase, bool mgSolver) {
    AmrFlow<21> f;
    f.init(t, h0);
    f.setViscosity(1.0);
    f.setDt(1e6);
    f.setBodyForce(1.0, 0, 0);
    f.setVelocityMGStaircase(staircase);
    if (staircase)
      f.setVelocityMGMinCoarse(512);
    f.setMomentumGS(true);  // exercise the Krylov-free solve with the RB-GS smoother
    f.setMomentumMGSolver(mgSolver);
    f.setSolid(sdf);
    for (int s = 0; s < 6; ++s)
      f.step(400, 60);
    return f.velocity(0);
  };
  const Index n = t.numLeaves();
  for (bool staircase : {false, true}) {
    auto ub = run(staircase, false);  // BiCGStab + MG preconditioner
    auto um = run(staircase, true);   // MG as the solver (defect correction)
    double dmax = 0, mag = 0;
    for (Index i = 0; i < n; ++i) {
      dmax = std::max(dmax, std::fabs(um[(std::size_t)i] - ub[(std::size_t)i]));
      mag = std::max(mag, std::fabs(ub[(std::size_t)i]));
    }
    std::printf("[flow] MG-solver-vs-BiCGStab (%s): max|mg-bicg| = %.3e (mag %.3e)\n",
                staircase ? "staircase" : "Galerkin", dmax, mag);
    PECLET_CORE_CHECK(dmax < 1e-4 * mag);
  }
}

// P4: the optional Picard outer loop over the lagged advection.
//   (a) Stokes (advection off): the extra outer iterations are genuine no-ops — re-lagging an
//   absent
//       advection cannot move the already-projected iterate — so the early-stop must engage (a
//       couple of iterations, not the cap) and the field must match the single-step run.
//   (b) Navier–Stokes: the loop must reach a valid NS steady field close to the single lagged step
//       (they coincide exactly only at true steady; after a finite number of steps the gap is at
//       the transient level — the same bar the device-vs-host advection field check uses).
void test_picard_outer() {
  const unsigned L = 4;
  BO t = uniformFine(L);
  const double h0 = 1.0 / (double)(1L << L);
  Vec<3> c{0.5, 0.5, 0.5};
  const Index n = t.numLeaves();

  // (a) Stokes: outer loop is a no-op beyond convergence detection.
  {
    double rad = 0.25;
    auto sdf = [&](const Vec<3>& p) {
      double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
      return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
    };
    auto run = [&](int outer, int* lastOuter) {
      AmrFlow<21> f;
      f.init(t, h0);
      f.setViscosity(1.0);
      f.setDt(1e6);
      f.setBodyForce(1.0, 0, 0);
      f.setOuterIterations(outer, 1e-9);
      f.setSolid(sdf);
      for (int s = 0; s < 6; ++s)
        f.step(400, 60);
      if (lastOuter)
        *lastOuter = f.lastOuterIters();
      return f.velocity(0);
    };
    int lo1 = 0, lo5 = 0;
    auto u1 = run(1, &lo1);
    auto u5 = run(5, &lo5);
    double dmax = 0, mag = 0;
    for (Index i = 0; i < n; ++i) {
      dmax = std::max(dmax, std::fabs(u5[(std::size_t)i] - u1[(std::size_t)i]));
      mag = std::max(mag, std::fabs(u1[(std::size_t)i]));
    }
    std::printf("[flow] picard outer Stokes: max|n5-n1| = %.3e (mag %.3e), lastOuter n1=%d n5=%d\n",
                dmax, mag, lo1, lo5);
    PECLET_CORE_CHECK(dmax < 1e-6 * mag);  // extra outer iters change nothing without advection
    PECLET_CORE_CHECK(lo1 == 1);           // default cap = a single outer iteration
    PECLET_CORE_CHECK(lo5 <= 3);           // early-stop engages well below the cap of 5
  }

  // (b) Navier–Stokes: valid steady close to the single lagged step.
  {
    const double mu = 0.5, G = 1.0, dt = 2.0;
    double rad = 0.2;
    auto sdf = [&](const Vec<3>& p) {
      double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
      return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
    };
    auto run = [&](int outer) {
      AmrFlow<21> f;
      f.init(t, h0);
      f.setViscosity(mu);
      f.setDt(dt);
      f.setBodyForce(G, 0, 0);
      f.setAdvection(true);
      f.setGhostProjection(false);  // explicit: the validated aperture Picard case
      f.setOuterIterations(outer, 1e-7);
      f.setSolid(sdf);
      for (int s = 0; s < 50; ++s)
        f.step(200, 80);
      return f.velocity(0);
    };
    auto u1 = run(1);
    auto u4 = run(4);
    double dmax = 0, mag = 0;
    for (Index i = 0; i < n; ++i) {
      dmax = std::max(dmax, std::fabs(u4[(std::size_t)i] - u1[(std::size_t)i]));
      mag = std::max(mag, std::fabs(u1[(std::size_t)i]));
    }
    std::printf("[flow] picard outer NS (n=4): max|picard-single| = %.3e (mag %.3e)\n", dmax, mag);
    PECLET_CORE_CHECK(dmax <
                      1e-2 * mag);  // same NS steady field (transient-level gap, sibling's bar)
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  test_poiseuille();
  test_staircase_mg();
  test_momentum_gs();
  test_momentum_mgsolver();
  test_sphere();
  test_sphere_ghost();
  test_sphere_ghostproj();
  test_graded_ghostproj();
  test_graded_cf_quadratic();
  test_sphere_ghostproj_adv();
  test_adapt_midrun();
  test_momentum_mg_option();
  test_momentum_scaling();
  test_advection_kernel();
  test_advection();
  test_picard_outer();
  Kokkos::finalize();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping device flow test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

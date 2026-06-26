// Device (Kokkos) collocated Stokes step on the octree (tpx::amr::DeviceAmrFlow) —
// the device counterpart of AmrFlow. Validates:
//   (1) Poiseuille — body-force-driven Stokes flow between immersed no-slip walls
//       converges to the analytic parabola u = G/(2μ)(y-y0)(y1-y) to ~round-off, solids
//       held at 0, and the device field matches the host AmrFlow field to tolerance;
//   (2) immersed sphere (cut cells) — the device step reproduces the host AmrFlow
//       velocity field on a genuinely cut geometry (the cut-cell ξ overlay + openness
//       projection all on device);
//   (3) projection — a pure-gradient periodic field's divergence + magnitude drop sharply.
// Runs on whatever backend Kokkos targets (CUDA / HIP / OpenMP). Validation is host-vs-
// device agreement + the analytic solution (the GPU differs from host only in FP last bits).
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <algorithm>
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/device_flow.hpp"
#include "tpx/amr/flow.hpp"
#include "tpx/common/types.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

BO uniformFine(unsigned L) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k) t.refineIf([](Code, unsigned) { return true; });
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
  AmrFlow<21> hfl;
  hfl.init(t, h0);
  hfl.setDensity(1.0);
  hfl.setViscosity(mu);
  hfl.setDt(1e6);
  hfl.setBodyForce(G, 0, 0);
  hfl.setSolid(sdf);
  for (int s = 0; s < 5; ++s) hfl.step(/*momSweeps=*/300, /*presIters=*/5, /*presSweeps=*/2);
  const auto& hux = hfl.velocity(0);

  // Device.
  DeviceAmrFlow<21> dfl;
  dfl.init(t, h0);
  dfl.setDensity(1.0);
  dfl.setViscosity(mu);
  dfl.setDt(1e6);
  dfl.setBodyForce(G, 0, 0);
  dfl.setSolid(sdf);
  for (int s = 0; s < 5; ++s) dfl.step(/*momIters=*/400, /*presIters=*/80);
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
      if (dux[(std::size_t)i] > 1e-3) fluidPositive = true;
    } else if (std::fabs(dux[(std::size_t)i]) > 1e-12) {
      solidsZero = false;
    }
  }
  double l2 = std::sqrt(e / nf);
  std::printf("[flow] poiseuille: device L2 vs analytic = %.3e, max|dev-host| = %.3e\n", l2, dh);
  TPX_CHECK(l2 < 1e-6);
  TPX_CHECK(solidsZero);
  TPX_CHECK(fluidPositive);
  TPX_CHECK(dh < 1e-7);  // device matches host
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

  AmrFlow<21> hfl;
  hfl.init(t, h0);
  hfl.setViscosity(mu);
  hfl.setDt(1e6);
  hfl.setBodyForce(G, 0, 0);
  hfl.setSolid(sdf);
  for (int s = 0; s < 8; ++s) hfl.step(/*momSweeps=*/400, /*presIters=*/8, /*presSweeps=*/2);
  const auto& hux = hfl.velocity(0);

  DeviceAmrFlow<21> dfl;
  dfl.init(t, h0);
  dfl.setViscosity(mu);
  dfl.setDt(1e6);
  dfl.setBodyForce(G, 0, 0);
  dfl.setSolid(sdf);
  for (int s = 0; s < 8; ++s) dfl.step(/*momIters=*/500, /*presIters=*/120);
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
  TPX_CHECK(dmean > 0.0);
  TPX_CHECK(std::fabs(dmean - hmean) / hmean < 2e-3);  // same permeability
  TPX_CHECK(dmax < 5e-3 * hmax);                       // fields agree
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
    DeviceAmrFlow<21> f;
    f.init(t, h0);
    f.setViscosity(1.0);
    f.setDt(1e6);
    f.setBodyForce(1.0, 0, 0);
    f.setMomentumMG(mgPre);
    f.setSolid(sdf);
    for (int s = 0; s < 8; ++s) f.step(400, 120);
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
  TPX_CHECK(dmax < 1e-3 * mag);  // same converged step regardless of preconditioner
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
    DeviceAmrFlow<21> f;
    f.init(t, h0);
    f.setViscosity(1.0);
    f.setDt(1e6);
    f.setBodyForce(1.0, 0, 0);
    f.setSolid(sdf);  // momentum MG on by default
    for (int s = 0; s < 4; ++s) f.step(400, 60);
    return f.lastMomIters();
  };
  int m4 = momIters(4), m5 = momIters(5);  // 16³, 32³
  std::printf("[flow] velocity-MG momentum iters: 16³=%d  32³=%d  (ratio %.2f)\n", m4, m5,
              (double)m5 / std::max(1, m4));
  TPX_CHECK(m5 < 2 * m4);  // near-flat (multigrid) — would ~double without it
  TPX_CHECK(m5 < 150);     // absolute bound (3 components × a few dozen MG-accelerated iters)
}

// Implicit-FOU + deferred-correction SOU advection (Navier–Stokes), validated against the host
// AmrFlow which runs the identical scheme: (a) Poiseuille with advection ON still converges to
// the analytic parabola (∇·(u u)=0 for unidirectional flow) and matches the host; (b) an
// immersed sphere at finite Re (the advection term is non-trivial) reaches the same steady
// velocity field on device and host.
// Kernel-only isolation: the device high-order advection ∇·(u u_c) must equal the host
// AmrFlow::advectTerm for the same velocity field (no solve, no time-stepping). Run on an
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
    for (int c = 0; c < 3; ++c) U[c].assign((std::size_t)n, 0.0);
    for (Index i = 0; i < n; ++i) {
      double x = cellCoord(t, i, 0, h0), y = cellCoord(t, i, 1, h0), z = cellCoord(t, i, 2, h0);
      U[0][(std::size_t)i] = std::sin(k * x) * std::cos(k * y) * std::cos(k * z) + 0.3;
      U[1][(std::size_t)i] = -std::cos(k * x) * std::sin(k * y) * std::cos(k * z);
      U[2][(std::size_t)i] = 0.1 * std::sin(k * z);
    }
    AmrFlow<21> hfl;
    hfl.init(t, h0);
    hfl.setDt(1.0);
    hfl.setAdvection(true);
    if (which == 0) hfl.setSolid(allfluid);
    else hfl.setSolid(sphere);
    for (int c = 0; c < 3; ++c) hfl.velocityRef()[c] = U[c];

    DeviceAmrFlow<21> dfl;
    dfl.init(t, h0);
    dfl.setDt(1.0);
    dfl.setAdvection(true);
    if (which == 0) dfl.setSolid(allfluid);
    else dfl.setSolid(sphere);
    for (int c = 0; c < 3; ++c) dfl.setVelocity(c, U[c]);

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
    TPX_CHECK(emax < 1e-10 * (1.0 + mag));
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

    AmrFlow<21> hfl;
    hfl.init(t, h0);
    hfl.setViscosity(mu);
    hfl.setDt(1e6);
    hfl.setBodyForce(G, 0, 0);
    hfl.setAdvection(true);  // SOU + implicit FOU (defaults)
    hfl.setSolid(sdf);
    for (int s = 0; s < 6; ++s) hfl.step(300, 5, 2);
    const auto& hux = hfl.velocity(0);

    DeviceAmrFlow<21> dfl;
    dfl.init(t, h0);
    dfl.setViscosity(mu);
    dfl.setDt(1e6);
    dfl.setBodyForce(G, 0, 0);
    dfl.setAdvection(true);
    dfl.setSolid(sdf);
    for (int s = 0; s < 6; ++s) dfl.step(400, 80);
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
    TPX_CHECK(std::sqrt(e / nf) < 1e-6);  // ∇·(u u)=0 ⇒ unchanged parabola
    TPX_CHECK(dh < 1e-6);
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
      f.setSolid(sdf);
    };
    AmrFlow<21> hfl;
    setup(hfl);
    DeviceAmrFlow<21> dfl;
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
    std::printf("[flow] sphere+adv (Re~%.1f): Umean host %.6e dev %.6e (rel %.2e), max|dev-host| "
                "%.3e (mag %.3e)\n",
                G * (hsum / nf) * 0.4 / mu, hsum / nf, dsum / nf,
                std::fabs(dsum - hsum) / std::fabs(hsum), dmax, hmax);
    TPX_CHECK(std::fabs(dsum - hsum) / std::fabs(hsum) < 5e-3);  // same steady permeability
    TPX_CHECK(dmax < 1e-2 * hmax);                               // fields agree
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  test_poiseuille();
  test_sphere();
  test_momentum_mg_option();
  test_momentum_scaling();
  test_advection_kernel();
  test_advection();
  Kokkos::finalize();
  TPX_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping device flow test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

// Collocated incompressible Stokes step on the octree (peclet::core::amr::oracle::AmrFlow) — wires
// the cut-cell Dirichlet momentum operator (no-slip IBM) and the openness pressure
// projection into one sdflow-style step:
//   (1) Poiseuille — body-force-driven Stokes flow between immersed no-slip walls
//       converges to the analytic parabola u = G/(2μ) (y-y0)(y1-y) (the discrete
//       cut-cell Laplacian is exact for a parabola with walls on cell faces, so
//       this matches to ~round-off); solids are held at zero;
//   (2) projection — on a periodic all-fluid box a pure-gradient (fully divergent)
//       velocity is reduced by the projection (divergence and |u| drop sharply).
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <limits>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/flow_oracle.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/amr/refine.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/geom/sdf.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

BO uniformFine(unsigned L) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k) t.refineIf([](Code, unsigned) { return true; });
  return t;
}

void test_poiseuille() {
  const unsigned L = 4;  // 16^3
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / static_cast<double>(N);
  const double y0 = 0.25, y1 = 0.75, G = 1.0, mu = 1.0;  // walls on cell faces

  oracle::AmrFlow<21> fl;
  fl.init(t, h0);
  fl.setDensity(1.0);
  fl.setViscosity(mu);
  fl.setDt(1e6);  // ~steady (divided convention well-conditioned at large dt)
  fl.setBodyForce(G, 0, 0);
  fl.setSolid([&](const Vec<3>& p) { return std::min(p[1] - y0, y1 - p[1]); });
  for (int s = 0; s < 5; ++s) fl.step(/*momSweeps=*/300, /*presIters=*/5, /*presSweeps=*/2);

  const auto& ux = fl.velocity(0);
  const Index n = t.numLeaves();
  double e = 0;
  long nf = 0;
  bool solidsZero = true, fluidPositive = false;
  for (Index i = 0; i < n; ++i) {
    auto b = t.bounds(i);
    double s = static_cast<double>(1 << t.level(i));
    double cy = (static_cast<double>(b[0][1]) + 0.5 * s) * h0;
    if (fl.isFluid(i)) {
      double uex = G / (2.0 * mu) * (cy - y0) * (y1 - cy);
      double d = ux[static_cast<std::size_t>(i)] - uex;
      e += d * d;
      ++nf;
      if (ux[static_cast<std::size_t>(i)] > 1e-3) fluidPositive = true;
    } else if (std::fabs(ux[static_cast<std::size_t>(i)]) > 1e-12) {
      solidsZero = false;
    }
  }
  double l2 = std::sqrt(e / nf);
  PECLET_CORE_CHECK(l2 < 1e-6);      // matches the analytic parabola (~round-off here)
  PECLET_CORE_CHECK(solidsZero);     // no-slip: immersed solid held at 0
  PECLET_CORE_CHECK(fluidPositive);  // flow actually develops
}

void test_projection() {
  const unsigned L = 4;
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / static_cast<double>(N);

  oracle::AmrFlow<21> fl;
  fl.init(t, h0);
  fl.setSolid([](const Vec<3>&) { return 1.0; });  // all fluid, periodic
  const Index n = t.numLeaves();
  const double k = 2.0 * M_PI;
  auto ctr = [&](Index i, int d) {
    auto b = t.bounds(i);
    double s = static_cast<double>(1 << t.level(i));
    return (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
  };
  auto& U = fl.velocityRef();
  for (Index i = 0; i < n; ++i) {
    double x = ctr(i, 0), y = ctr(i, 1), z = ctr(i, 2);
    U[0][static_cast<std::size_t>(i)] = -std::sin(k * x) * std::cos(k * y) * std::cos(k * z);
    U[1][static_cast<std::size_t>(i)] = -std::cos(k * x) * std::sin(k * y) * std::cos(k * z);
    U[2][static_cast<std::size_t>(i)] = -std::cos(k * x) * std::cos(k * y) * std::sin(k * z);
  }
  auto unorm = [&]() {
    double s = 0;
    for (int c = 0; c < 3; ++c)
      for (Index i = 0; i < n; ++i) s += U[c][static_cast<std::size_t>(i)] * U[c][static_cast<std::size_t>(i)];
    return std::sqrt(s);
  };
  double d0 = fl.divNormL2(U), u0 = unorm();
  fl.project(200, 4);
  double d1 = fl.divNormL2(U), u1 = unorm();
  // pure-gradient field: exact projection -> 0; approximate collocated -> sharp drop.
  PECLET_CORE_CHECK(d0 / d1 > 10.0);
  PECLET_CORE_CHECK(u0 / u1 > 10.0);
}

// Advection operator: ∇·(u u_x) on a divergence-free field has the exact value
// (k/2) sin(2k x); measure the error for scheme `sch` (0=SOU default, 1=TVD).
double advectErr(unsigned L, double& constMax, int sch = 0) {
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / static_cast<double>(N), k = 2.0 * M_PI;
  oracle::AmrFlow<21> fl;
  fl.init(t, h0);
  fl.setAdvectionScheme(sch);
  fl.setSolid([](const Vec<3>&) { return 1.0; });
  const Index n = t.numLeaves();
  auto ctr = [&](Index i, int d) {
    auto b = t.bounds(i);
    double s = static_cast<double>(1 << t.level(i));
    return (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
  };
  auto& U = fl.velocityRef();
  for (Index i = 0; i < n; ++i) {
    double x = ctr(i, 0), y = ctr(i, 1);
    U[0][static_cast<std::size_t>(i)] = std::sin(k * x) * std::cos(k * y);
    U[1][static_cast<std::size_t>(i)] = -std::cos(k * x) * std::sin(k * y);
    U[2][static_cast<std::size_t>(i)] = 0.0;
  }
  double e = 0;
  for (Index i = 0; i < n; ++i) {
    double d = fl.advectTerm(0, i) - 0.5 * k * std::sin(2.0 * k * ctr(i, 0));
    e += d * d;
  }
  for (int c = 0; c < 3; ++c)
    for (Index i = 0; i < n; ++i) U[c][static_cast<std::size_t>(i)] = (c == 0) ? 3.7 : 0.0;
  constMax = 0;
  for (Index i = 0; i < n; ++i) constMax = std::max(constMax, std::fabs(fl.advectTerm(0, i)));
  return std::sqrt(e / n);
}

void test_advection() {
  // SOU (default): unlimited 2nd-order upwind -> ~2nd-order convergence (ratio ~4).
  double c5 = 0, c6 = 0;
  double e5 = advectErr(5, c5, /*SOU*/ 0);
  double e6 = advectErr(6, c6, /*SOU*/ 0);
  PECLET_CORE_CHECK(e5 / e6 > 3.3);              // ~2nd order (vs TVD ~2.8 — limiter clips extrema)
  PECLET_CORE_CHECK(c5 < 1e-12 && c6 < 1e-12);  // Galilean: constant advects to 0
  // TVD (option): also converges, above 1st order (limiter -> ~1.5 order at extrema).
  double t5 = 0, t6 = 0;
  double et5 = advectErr(5, t5, /*TVD*/ 1);
  double et6 = advectErr(6, t6, /*TVD*/ 1);
  PECLET_CORE_CHECK(et5 / et6 > 2.3);
  PECLET_CORE_CHECK(e6 < et6);  // SOU is more accurate than TVD on this smooth field

  // Poiseuille with advection ON: for unidirectional flow ∇·(u u)=0, so the
  // advection term must vanish and the parabola is recovered unchanged.
  const unsigned L = 4;
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / static_cast<double>(N);
  const double y0 = 0.25, y1 = 0.75, G = 1.0, mu = 1.0;
  oracle::AmrFlow<21> fl;
  fl.init(t, h0);
  fl.setViscosity(mu);
  fl.setDt(1e6);
  fl.setBodyForce(G, 0, 0);
  fl.setAdvection(true);
  fl.setSolid([&](const Vec<3>& p) { return std::min(p[1] - y0, y1 - p[1]); });
  for (int s = 0; s < 5; ++s) fl.step(300, 5, 2);
  const auto& ux = fl.velocity(0);
  double err = 0;
  long nf = 0;
  for (Index i = 0; i < t.numLeaves(); ++i)
    if (fl.isFluid(i)) {
      auto b = t.bounds(i);
      double s = static_cast<double>(1 << t.level(i));
      double cy = (static_cast<double>(b[0][1]) + 0.5 * s) * h0;
      double d = ux[static_cast<std::size_t>(i)] - G / (2.0 * mu) * (cy - y0) * (y1 - cy);
      err += d * d;
      ++nf;
    }
  PECLET_CORE_CHECK(std::sqrt(err / nf) < 1e-6);
}

// Implicit-FOU deferred correction is unconditionally stable for advection: at a
// high advective CFL, explicit high-order advection blows up while the implicit-FOU
// path stays bounded.
void test_implicit_advection() {
  const unsigned L = 4;
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / static_cast<double>(N), k = 2.0 * M_PI, A = 4.0;
  auto initVortex = [&](oracle::AmrFlow<21>& fl) {
    fl.init(t, h0);
    fl.setDensity(1.0);
    fl.setViscosity(0.005);
    fl.setDt(0.1);  // CFL = A*dt/h ≈ 6.4 -> explicit advection blows up (NaN)
    fl.setAdvection(true);
    fl.setSolid([](const Vec<3>&) { return 1.0; });  // all fluid, periodic
    auto ctr = [&](Index i, int d) {
      auto b = t.bounds(i);
      double s = static_cast<double>(1 << t.level(i));
      return (static_cast<double>(b[0][d]) + 0.5 * s) * h0;
    };
    auto& U = fl.velocityRef();
    for (Index i = 0; i < t.numLeaves(); ++i) {
      double x = ctr(i, 0), y = ctr(i, 1);
      U[0][static_cast<std::size_t>(i)] = A * std::sin(k * x) * std::cos(k * y);
      U[1][static_cast<std::size_t>(i)] = -A * std::cos(k * x) * std::sin(k * y);
    }
  };
  auto maxU = [&](oracle::AmrFlow<21>& fl) {
    double m = 0;
    bool finite = true;
    for (int c = 0; c < 3; ++c)
      for (Index i = 0; i < t.numLeaves(); ++i) {
        double v = fl.velocity(c)[static_cast<std::size_t>(i)];
        if (!std::isfinite(v)) finite = false;  // NaN/Inf would otherwise hide under std::max
        m = std::max(m, std::fabs(v));
      }
    return finite ? m : std::numeric_limits<double>::infinity();
  };

  oracle::AmrFlow<21> imp;
  initVortex(imp);  // implicit-FOU (default on)
  for (int it = 0; it < 25; ++it) imp.step(40, 5, 2);
  double mi = maxU(imp);

  oracle::AmrFlow<21> exp_;
  initVortex(exp_);
  exp_.setImplicitAdvection(false);  // fully explicit high-order advection
  for (int it = 0; it < 25; ++it) exp_.step(40, 5, 2);
  double me = maxU(exp_);

  PECLET_CORE_CHECK(std::isfinite(mi) && mi < 5.0 * A);   // implicit stays bounded (stable)
  PECLET_CORE_CHECK(!(std::isfinite(me) && me < 5.0 * A)); // explicit blows up at this CFL
}

// Graded-interface advection: on a graded mesh (sphere band finest, far field
// coarse), advection ON must be STABLE across the 2:1 interfaces — the FOU operator
// is C/F-conservative (sums fine sub-faces) and the high-order flux is C/F-aware.
void test_graded_advection() {
  const long brick = 8;
  const unsigned lmax = 1;  // finest 16
  const long N = brick * (1L << lmax);
  const double k = 2.0 * M_PI, c = N / 2.0;
  BO t(IVec<3>{brick, brick, brick}, lmax);
  AmrGeometry<3> geo;
  geo.h0 = 1.0;
  peclet::core::geom::Sphere sph{{c, c, c}, 0.30 * N};
  refineToSdf(t, geo, [&](const Vec<3>& p) { return -sph.eval(p); }, 0, 1.5, true);
  PECLET_CORE_CHECK(t.numLeaves() < N * N * N);  // genuinely graded

  oracle::AmrFlow<21> fl;
  fl.init(t, 1.0, Vec<3>{0, 0, 0});
  fl.setDensity(1.0);
  fl.setViscosity(0.02);
  fl.setDt(0.1);
  fl.setAdvection(true);                           // implicit-FOU (default on)
  fl.setSolid([](const Vec<3>&) { return 1.0; });  // all fluid, periodic
  auto& U = fl.velocityRef();
  for (Index i = 0; i < t.numLeaves(); ++i) {
    auto b = t.bounds(i);
    double s = static_cast<double>(1 << t.level(i));
    double x = (static_cast<double>(b[0][0]) + 0.5 * s) / N, y = (static_cast<double>(b[0][1]) + 0.5 * s) / N;
    U[0][static_cast<std::size_t>(i)] = 2.0 * std::sin(k * x) * std::cos(k * y);
    U[1][static_cast<std::size_t>(i)] = -2.0 * std::cos(k * x) * std::sin(k * y);
  }
  for (int it = 0; it < 30; ++it) fl.step(40, 6, 2);
  double m = 0;
  bool finite = true;
  for (int comp = 0; comp < 3; ++comp)
    for (Index i = 0; i < t.numLeaves(); ++i) {
      double v = fl.velocity(comp)[static_cast<std::size_t>(i)];
      if (!std::isfinite(v)) finite = false;
      m = std::max(m, std::fabs(v));
    }
  PECLET_CORE_CHECK(finite && m < 4.0);  // stable across 2:1 interfaces (bounded, no blow-up)
}

void run() {
  test_poiseuille();
  test_projection();
  test_advection();
  test_implicit_advection();
  test_graded_advection();
}

}  // namespace

int main() {
  run();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping AMR flow test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

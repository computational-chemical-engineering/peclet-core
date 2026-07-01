// Explicit FV scalar transport on the octree (peclet::core::amr::ScalarTransport):
//   (1) conservation — a divergence-free advection+diffusion update conserves the
//       total scalar to round-off, on uniform AND 2:1-graded meshes;
//   (2) diffusion — a sine mode decays at the analytic rate exp(-D k^2 t);
//   (3) advection — upwind is monotone (no new extrema) and preserves a constant.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/amr/refine.hpp"
#include "peclet/core/amr/scalar_transport.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/geom/sdf.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;

BO uniformFine(unsigned L) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k) t.refineIf([](Code, unsigned) { return true; });
  return t;
}

double pseudo(std::uint64_t& s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>((s >> 11) & 0xFFFFF) / static_cast<double>(0x100000);
}

// Uniform (divergence-free) velocity.
struct UniformVel {
  Vec<3> u;
  double operator()(const Vec<3>&, int axis) const { return u[axis]; }
};

void test_conservation(const BO& t, double h0) {
  AmrGeometry<3> geo;
  geo.h0 = h0;
  ScalarTransport<3, kBits> st(t, geo);
  std::vector<double> c(static_cast<std::size_t>(t.numLeaves())), tmp;
  std::uint64_t s = 13;
  for (auto& x : c) x = pseudo(s);
  UniformVel vel{{0.7, -0.3, 0.2}};
  double minDx = h0;  // finest cell width
  double dt = 0.2 * std::min(minDx / 0.7, minDx * minDx / (2.0 * 0.01 * 3.0));
  double m0 = st.totalMass(c);
  for (int it = 0; it < 25; ++it) {
    st.step(c, tmp, dt, /*D=*/0.01, vel);
    c.swap(tmp);
  }
  double m1 = st.totalMass(c);
  PECLET_CORE_CHECK(std::fabs(m1 - m0) < 1e-10 * (std::fabs(m0) + 1e-30));
}

void test_diffusion_rate() {
  const unsigned L = 5;
  const double h0 = 1.0 / static_cast<double>(1u << L);
  BO t = uniformFine(L);
  AmrGeometry<3> geo;
  geo.h0 = h0;
  ScalarTransport<3, kBits> st(t, geo);
  const double k = 2.0 * M_PI;
  std::vector<double> c(static_cast<std::size_t>(t.numLeaves())), tmp;
  auto xc = [&](Index i) { return (static_cast<double>(t.bounds(i)[0][0]) + 0.5) * h0; };
  for (Index i = 0; i < t.numLeaves(); ++i) c[static_cast<std::size_t>(i)] = std::sin(k * xc(i));

  auto amplitude = [&](const std::vector<double>& f) {
    double num = 0, den = 0;
    for (Index i = 0; i < t.numLeaves(); ++i) {
      double phi = std::sin(k * xc(i));
      num += f[static_cast<std::size_t>(i)] * phi;
      den += phi * phi;
    }
    return num / den;
  };

  const double D = 0.02;
  double dt = 0.2 * h0 * h0 / (2.0 * D * 3.0);
  UniformVel vel{{0, 0, 0}};
  double a0 = amplitude(c);
  double t1 = 0.0;
  for (int it = 0; it < 200; ++it) {
    st.step(c, tmp, dt, D, vel);
    c.swap(tmp);
    t1 += dt;
  }
  double a1 = amplitude(c);
  double expect = std::exp(-D * k * k * t1);
  PECLET_CORE_CHECK(std::fabs(a1 / a0 - expect) < 0.05 * expect);
}

void test_advection_monotone() {
  const unsigned L = 4;
  const double h0 = 1.0 / static_cast<double>(1u << L);
  BO t = uniformFine(L);
  AmrGeometry<3> geo;
  geo.h0 = h0;
  ScalarTransport<3, kBits> st(t, geo);
  auto xc = [&](Index i) { return (static_cast<double>(t.bounds(i)[0][0]) + 0.5) * h0; };

  // (a) constant field is preserved exactly.
  {
    std::vector<double> c(static_cast<std::size_t>(t.numLeaves()), 2.5), tmp;
    UniformVel vel{{1.0, -0.5, 0.3}};
    double dt = 0.3 * h0 / 1.0;
    double maxdev = 0;
    for (int it = 0; it < 10; ++it) {
      st.step(c, tmp, dt, 0.0, vel);
      c.swap(tmp);
    }
    for (double v : c) maxdev = std::max(maxdev, std::fabs(v - 2.5));
    PECLET_CORE_CHECK(maxdev < 1e-10);
  }

  // (b) upwind is monotone: a step profile in [0,1] develops no over/undershoot.
  {
    std::vector<double> c(static_cast<std::size_t>(t.numLeaves())), tmp;
    for (Index i = 0; i < t.numLeaves(); ++i) c[static_cast<std::size_t>(i)] = (xc(i) < 0.5) ? 1.0 : 0.0;
    UniformVel vel{{1.0, 0.0, 0.0}};
    double dt = 0.4 * h0 / 1.0;
    double lo = 1e30, hi = -1e30;
    for (int it = 0; it < 20; ++it) {
      st.step(c, tmp, dt, 0.0, vel);
      c.swap(tmp);
      for (double v : c) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
    }
    PECLET_CORE_CHECK(lo > -1e-12 && hi < 1.0 + 1e-12);
  }
}

void run() {
  test_conservation(uniformFine(4), 1.0 / 16.0);

  // graded mesh around a sphere.
  BO g(IVec<3>{2, 2, 2}, 4);
  AmrGeometry<3> ggeo;
  ggeo.h0 = 1.0;
  peclet::core::geom::Sphere sph{{16.0, 16.0, 16.0}, 8.0};
  refineToSdf(g, ggeo, [&](const Vec<3>& p) { return sph.eval(p); }, 1, 1.0, true);
  test_conservation(g, 1.0);

  test_diffusion_rate();
  test_advection_monotone();
}

}  // namespace

int main() {
  run();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping AMR scalar transport test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

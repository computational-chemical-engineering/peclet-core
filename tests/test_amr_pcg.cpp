// Device (Kokkos) multigrid-preconditioned CG for the AMR FV Poisson
// (peclet::core::amr::PCG). Validates, on a genuinely graded octree:
//   (1) PCG drives the manufactured-RHS residual to round-off (the singular periodic
//       operator: RHS b = L·u_exact is exactly mean-zero, so CG stays in the range
//       space with nullspace projection and converges to ~machine precision);
//   (2) PCG reaches a target residual in FEWER fine-grid matvecs than plain V-cycling
//       (the point of the Krylov acceleration);
//   (3) the PCG solution matches the V-cycle's converged solution (same linear system);
//   (4) the openness (cut-cell) operator path also converges.
// Runs on whatever backend Kokkos was built for (CUDA / HIP / OpenMP). Validation is by
// convergence + tolerance (not host bit-exactness): the GPU matvec differs from the host
// in the last bit due to FMA contraction, but the iteration is mathematically identical.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/multigrid.hpp"
#include "peclet/core/amr/pcg.hpp"
#include "peclet/core/amr/poisson.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using M = BO::M;
using AP = AmrPoisson<3, kBits>;

void setDev(View<double> v, const std::vector<double>& h) {
  auto m = Kokkos::create_mirror_view(v);
  for (std::size_t i = 0; i < h.size(); ++i)
    m((Index)i) = h[i];
  Kokkos::deep_copy(v, m);
}
std::vector<double> getDev(View<double> v, Index n) {
  std::vector<double> h((std::size_t)n);
  auto m = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(m, v);
  for (Index i = 0; i < n; ++i)
    h[(std::size_t)i] = m(i);
  return h;
}

// A graded octree: 2×2×2 brick (level 3, domain 16^3), refined uniformly to level 1 then
// the lower octant to level 0, 2:1-balanced. (Same mesh as the device MG test.)
BO gradedMesh() {
  BO t(IVec<3>{2, 2, 2}, 3);
  for (int kk = 0; kk < 2; ++kk)
    t.refineIf([](Code, unsigned l) { return l > 0; });
  t.refineIf([&](Code c, unsigned) {
    auto o = M::from_code(c).decode();
    return o[0] < 8 && o[1] < 8 && o[2] < 8;
  });
  t.balance2to1();
  return t;
}

// Manufactured smooth field on leaf centroids.
std::vector<double> manufactured(const BO& t, double h0) {
  const Index n = t.numLeaves();
  std::vector<double> u((std::size_t)n);
  const double k = 2.0 * M_PI;
  for (Index i = 0; i < n; ++i) {
    auto o = M::from_code(t.code(i)).decode();
    double half = 0.5 * (double)(Index(1) << t.level(i));
    double cx = ((double)o[0] + half) * h0, cy = ((double)o[1] + half) * h0,
           cz = ((double)o[2] + half) * h0;
    u[(std::size_t)i] = std::sin(k * cx) * std::cos(k * cy) + 0.5 * std::cos(k * cz);
  }
  return u;
}

void run() {
  BO t = gradedMesh();
  const Index n = t.numLeaves();
  const double h0 = 1.0 / 16.0;

  AP ap0;
  ap0.init(t, h0);

  // Manufactured RHS b = L·u_exact (exactly mean-zero by conservation ⇒ compatible with
  // the periodic singular operator). Solving L x = b recovers u_exact up to a constant.
  std::vector<double> uex = manufactured(t, h0);
  std::vector<double> b;
  ap0.applyLaplacian(uex, b);

  auto resNorm = [&](const std::vector<double>& x) {
    std::vector<double> lu;
    ap0.applyLaplacian(x, lu);
    double s = 0.0;
    for (Index i = 0; i < n; ++i) {
      double rr = b[(std::size_t)i] - lu[(std::size_t)i];
      s += rr * rr;
    }
    return std::sqrt(s);
  };
  const double bnorm = [&] {
    double s = 0;
    for (double v : b)
      s += v * v;
    return std::sqrt(s);
  }();

  // ===== (1) PCG drives the residual to round-off =====
  Multigrid<3, kBits> mg;
  mg.build(t, h0);
  PCG<3, kBits> pcg;
  pcg.setVcycle(2, 2, 40, 0.8);
  pcg.setSingular(true);

  View<double> dx("pcg_x", (std::size_t)n), db("pcg_b", (std::size_t)n);
  setDev(db, b);
  auto R = pcg.solve(mg, dx, View<const double>(db), /*maxIters=*/200, /*tol=*/1e-12);
  auto xpcg = getDev(dx, n);
  double rpcg = resNorm(xpcg);
  std::printf("[pcg] graded: %d iters, res %.3e -> %.3e (rel %.3e)\n", R.iters, R.res0, rpcg,
              rpcg / bnorm);
  PECLET_CORE_CHECK(rpcg < bnorm * 1e-9);
  PECLET_CORE_CHECK(R.iters < 60);  // CG over MG converges in a handful of iterations

  // ===== (2) PCG beats plain V-cycling in fine matvecs, (3) same solution =====
  // Count V-cycles to reach the same relative residual PCG hit.
  const double target = std::max(rpcg, bnorm * 1e-10);
  Kokkos::deep_copy(mg.x(0), 0.0);
  setDev(mg.b(0), b);
  int vcyc = 0;
  double rv = bnorm;
  for (; vcyc < 500; ++vcyc) {
    mg.vcycle(2, 2, 40, 0.8);
    auto xv = getDev(mg.x(0), n);
    rv = resNorm(xv);
    if (rv <= target)
      break;
  }
  // Each PCG iteration ≈ 1 V-cycle (preconditioner) + 1 matvec; count PCG fine matvecs as
  // iters*(cyclesPerPrec*(pre+post+...) + 1). Compare V-cycles-to-target: PCG should need
  // markedly fewer *preconditioner V-cycles* than stationary V-cycling.
  std::printf("[pcg] to reach rel %.3e: PCG %d iters (=%d precond V-cycles) vs %d plain V-cycles\n",
              rpcg / bnorm, R.iters, R.iters, vcyc + 1);
  PECLET_CORE_CHECK(R.iters <= vcyc);  // Krylov acceleration: fewer V-cycles than stationary

  // (3) solution agreement (up to the nullspace constant): compare mean-removed fields.
  auto xv = getDev(mg.x(0), n);
  ap0.removeMean(xpcg);
  ap0.removeMean(xv);
  double dmax = 0;
  for (Index i = 0; i < n; ++i)
    dmax = std::max(dmax, std::fabs(xpcg[(std::size_t)i] - xv[(std::size_t)i]));
  std::printf("[pcg] |x_pcg - x_vcyc|_max (mean-removed) = %.3e\n", dmax);
  PECLET_CORE_CHECK(dmax < bnorm * 1e-6);

  // ===== (4) openness (cut-cell) operator path converges =====
  // A sphere-shaped openness (aperture from an SDF) on the same mesh.
  Vec<3> ctr{0.5, 0.5, 0.5};
  double rad = 0.3;
  auto openFn = [&](const Vec<3>& fc, int) -> double {
    double dx0 = fc[0] - ctr[0], dy0 = fc[1] - ctr[1], dz0 = fc[2] - ctr[2];
    double d = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0) - rad;
    // open (1) outside the sphere, closed (0) inside, smooth band of width h0.
    double a = 0.5 + d / (2.0 * h0);
    return a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);
  };
  AP apO;
  apO.init(t, h0);
  apO.buildOpenness(openFn);
  std::vector<double> bO;
  apO.applyLaplacian(uex, bO);  // mean-zero manufactured RHS for the openness operator
  auto resNormO = [&](const std::vector<double>& x) {
    std::vector<double> lu;
    apO.applyLaplacian(x, lu);
    double s = 0.0;
    for (Index i = 0; i < n; ++i) {
      double rr = bO[(std::size_t)i] - lu[(std::size_t)i];
      s += rr * rr;
    }
    return std::sqrt(s);
  };
  double bOnorm = [&] {
    double s = 0;
    for (double v : bO)
      s += v * v;
    return std::sqrt(s);
  }();

  Multigrid<3, kBits> mgO;
  mgO.build(t, h0, openFn, /*periodic=*/true);
  PCG<3, kBits> pcgO;
  pcgO.setVcycle(2, 2, 60, 0.8);
  pcgO.setSingular(true);
  View<double> dxO("pcgO_x", (std::size_t)n), dbO("pcgO_b", (std::size_t)n);
  setDev(dbO, bO);
  auto RO = pcgO.solve(mgO, dxO, View<const double>(dbO), 300, 1e-10);
  auto xO = getDev(dxO, n);
  double rO = resNormO(xO);
  std::printf("[pcg] openness: %d iters, rel res %.3e\n", RO.iters, rO / bOnorm);
  PECLET_CORE_CHECK(rO < bOnorm * 1e-6);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping device PCG test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

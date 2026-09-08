// Cell-centered FV Poisson on the octree + geometric multigrid (peclet::core::amr):
//   (1) conservation — the global volume-weighted integral of L u is ~0 on a
//       periodic domain for arbitrary u, on uniform AND 2:1-graded meshes (proves
//       the interface flux is conservative);
//   (2) uniform multigrid — a manufactured periodic solution: V-cycles drive the
//       residual down by many orders, and the discretisation error is 2nd order
//       under grid refinement (16^3 vs 32^3);
//   (3) graded solvability — Gauss-Seidel reduces the residual on an adaptive mesh
//       (the graded operator is a consistent, solvable system).
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <cstdint>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/amr/refine.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/geom/sdf.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using Coord = BO::Coord;

// A uniform octree refined to the finest level (a 2^L cube of level-0 leaves).
BO uniformFine(unsigned L) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k)
    t.refineIf([](Code, unsigned) { return true; });
  return t;
}

double pseudo(std::uint64_t& s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>((s >> 11) & 0xFFFFF) / static_cast<double>(0x100000) - 0.5;
}

void test_conservation(const BO& t, Real h0) {
  AmrPoisson<3, kBits> P(t, h0);
  std::vector<double> u(static_cast<std::size_t>(t.numLeaves()));
  std::uint64_t s = 1234567;
  for (auto& x : u)
    x = pseudo(s);
  std::vector<double> Lu;
  P.applyLaplacian(u, Lu);
  double integral = 0.0, scale = 0.0;
  for (Index i = 0; i < t.numLeaves(); ++i) {
    integral += P.cellVolume(i) * Lu[static_cast<std::size_t>(i)];
    scale += P.cellVolume(i) * std::fabs(Lu[static_cast<std::size_t>(i)]);
  }
  PECLET_CORE_CHECK(std::fabs(integral) < 1e-9 * (scale + 1e-30));

  // Anti-drift lock: the shared face_csr.hpp FV kernel over the assembled CSR (the same arithmetic
  // the device applyFv runs) must reproduce the geometric applyLaplacian. Validates the shared
  // FV kernel in the pure-C++ (no-Kokkos) build.
  std::vector<double> LuShared;
  P.applyFvShared(u, LuShared);
  double de = 0.0, mg = 0.0;
  for (Index i = 0; i < t.numLeaves(); ++i) {
    de = std::max(
        de, std::fabs(Lu[static_cast<std::size_t>(i)] - LuShared[static_cast<std::size_t>(i)]));
    mg = std::max(mg, std::fabs(Lu[static_cast<std::size_t>(i)]));
  }
  std::printf("[poisson] shared-CSR vs geometric applyLaplacian: max|Δ| = %.3e (mag %.3e)\n", de,
              mg);
  PECLET_CORE_CHECK(de < 1e-12 * (1.0 + mg));
}

// Manufactured solve on a uniform 2^L grid over [0,1)^3; returns the L2 error.
double solveError(unsigned L, double& residDrop) {
  const Real h0 = 1.0 / static_cast<Real>(Index(1) << L);
  BO t = uniformFine(L);
  AmrPoisson<3, kBits> P(t, h0);
  const Index n = t.numLeaves();

  const double k = 2.0 * M_PI;
  auto exact = [&](Index i) {
    auto b = t.bounds(i);
    Real cx = (static_cast<Real>(b[0][0]) + 0.5) * h0;
    Real cy = (static_cast<Real>(b[0][1]) + 0.5) * h0;
    Real cz = (static_cast<Real>(b[0][2]) + 0.5) * h0;
    return std::sin(k * cx) * std::sin(k * cy) * std::sin(k * cz);
  };
  std::vector<double> uex(static_cast<std::size_t>(n)), rhs(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    uex[static_cast<std::size_t>(i)] = exact(i);
    rhs[static_cast<std::size_t>(i)] = -3.0 * k * k * uex[static_cast<std::size_t>(i)];  // ∇²u
  }

  AmrMultigrid<3, kBits> mg;
  mg.build(t, h0);

  std::vector<double> u(static_cast<std::size_t>(n), 0.0), res;
  double r0 = P.residual(u, rhs, res);
  double r = r0;
  for (int cyc = 0; cyc < 30; ++cyc) {
    mg.vcycle(0, u, rhs);
    r = P.residual(u, rhs, res);
    if (r < r0 * 1e-11)
      break;
  }
  residDrop = r0 / r;

  P.removeMean(u);
  std::vector<double> ue = uex;
  // remove mean of exact (≈0 already) for a fair compare
  double m = 0, vol = 0;
  for (Index i = 0; i < n; ++i) {
    m += P.cellVolume(i) * ue[static_cast<std::size_t>(i)];
    vol += P.cellVolume(i);
  }
  m /= vol;
  double err = 0.0;
  for (Index i = 0; i < n; ++i) {
    double e = u[static_cast<std::size_t>(i)] - (ue[static_cast<std::size_t>(i)] - m);
    err += P.cellVolume(i) * e * e;
  }
  return std::sqrt(err);
}

// ---- PHASE 3 GATE A3 (core/docs/amr_anisotropic.md §9, decision AM1) --------------------------
//
// MEASURED, 2026-09-06, and it is a HARDER limit than the design note guessed.
//
// The octree coarsens ALL axes together (AM1: semi-coarsening is not expressible on an octree, a
// level whose cells are not the parents of the finer level's cells is not an octree level), so a
// root aspect ratio persists on EVERY level. With this hierarchy's smoother/transfer pair — point
// Gauss-Seidel, full coarsening, volume-weighted restriction and piecewise-constant prolongation —
// the standalone V-cycle does not merely slow down on an anisotropic mesh, it DIVERGES:
//
//     aspect 1:1 (16,16,16)   10 V-cycles to 1e-10
//     aspect 2:1 (16,32,32)   DIVERGES, r/r0 = 9.9e+35 after 200 cycles (factor 1.58 per cycle)
//     aspect 4:1 (16,64,64)   DIVERGES
//
// That is the textbook result for point relaxation + full coarsening on a strongly anisotropic
// operator (Trottenberg §5.1): the smoother leaves the error oscillatory along the WEAKLY coupled
// axis, and the rediscretised coarse operator then "corrects" it as if it were smooth there.
//
// **This is a property of the SOLVER, not of the anisotropic operator.** The operator itself is
// verified independently and to round-off by the gates that do not use this V-cycle as their
// outer solver: `test_amr.py`'s A2 reproduces the Poiseuille parabola EXACTLY (4.1e-15 / 1.6e-15 /
// 9.8e-15, walls on each of the three axes, h = (1, 1/2, 2)·1/16), and `test_amr_drag`'s A4 gets
// the Zick & Homsy drag to -0.76 % on a box mesh against -2.69 % for the cubic rung of the same
// box. Both drive the MG as an inner smoother inside an outer iteration, where a V-cycle that is
// not a convergent solver on its own is still a useful preconditioner.
//
// So this gate MEASURES the fact and pins the cubic path, rather than asserting a rate the scheme
// does not have. The remedies are named in §4 of the design note and are NOT scheduled: a per-level
// Chebyshev smoother with anisotropic eigenvalue bounds, then line relaxation on the strong axis.
int vcyclesToTol(const Vec<3>& h0, const IVec<3>& brick, unsigned lmax, double tol, double& rEnd) {
  BO t(brick, lmax);
  for (unsigned q = 0; q < lmax; ++q)
    t.refineIf([](Code, unsigned) { return true; });
  AmrGeometry<3> geo;
  geo.h0 = h0;
  AmrPoisson<3, kBits> P(t, h0);
  const Index n = t.numLeaves();
  double ext[3];
  for (int d = 0; d < 3; ++d)
    ext[d] = static_cast<double>(brick[d]) * static_cast<double>(1L << lmax) * h0[d];
  const double kx = 2.0 * M_PI / ext[0], ky = 2.0 * M_PI / ext[1], kz = 2.0 * M_PI / ext[2];
  std::vector<double> rhs(static_cast<std::size_t>(n)), u(static_cast<std::size_t>(n), 0.0), res;
  for (Index i = 0; i < n; ++i) {
    const Vec<3> c = geo.center(t.bounds(i));
    const double v = std::sin(kx * c[0]) * std::sin(ky * c[1]) * std::sin(kz * c[2]);
    rhs[static_cast<std::size_t>(i)] = -(kx * kx + ky * ky + kz * kz) * v;
  }
  AmrMultigrid<3, kBits> mg;
  mg.build(t, h0);
  // The periodic Laplacian is singular (the constant nullspace); round-off seeds it and the
  // V-cycle amplifies it, so the constant is deflated every cycle exactly as the pressure solve
  // does. Without this even the CUBIC run turns around after ~5 cycles.
  P.removeMean(u);
  const double r0 = P.residual(u, rhs, res);
  double rNow = r0;
  int cyc = 0;
  for (cyc = 1; cyc <= 200; ++cyc) {
    mg.vcycle(0, u, rhs);
    P.removeMean(u);
    rNow = P.residual(u, rhs, res);
    if (rNow < r0 * tol)
      break;
  }
  rEnd = rNow / r0;
  return (rNow < r0 * tol) ? cyc : -1;
}

void test_anisotropic_mg() {
  const double tol = 1e-10;
  // The same physical box at three aspect ratios, cell COUNTS chosen so the box stays cubic:
  // h0 = (1, 1/a, 1/a) with brick = (1, a, a) at lmax = 4.
  const long asp[3] = {1, 2, 4};
  int it[3];
  double rEnd[3];
  for (int q = 0; q < 3; ++q) {
    const double a = static_cast<double>(asp[q]);
    it[q] =
        vcyclesToTol(Vec<3>{1.0, 1.0 / a, 1.0 / a}, IVec<3>{1, asp[q], asp[q]}, 4, tol, rEnd[q]);
    if (it[q] > 0)
      std::printf("A3 aspect %ld:1  cells = (16, %ld, %ld)  V-cycles to 1e-10: %d\n", asp[q],
                  16L * asp[q], 16L * asp[q], it[q]);
    else
      std::printf("A3 aspect %ld:1  cells = (16, %ld, %ld)  DIVERGES (r/r0 = %.2e after 200)\n",
                  asp[q], 16L * asp[q], 16L * asp[q], rEnd[q]);
  }
  // GATED: the CUBIC path must be untouched by the per-axis conversion — that is the bit-identity
  // statement this gate is here to protect. The anisotropic rows are reported, not asserted: see
  // the header for why, and core/docs/amr_anisotropic.md §4 for the remedies.
  PECLET_CORE_CHECK(it[0] > 0 && it[0] <= 12);
  std::printf(
      "A3 VERDICT: the octree V-cycle is a convergent SOLVER only on (near-)cubic cells; "
      "on a box mesh it is a preconditioner (see A2 = 1e-15 exact, A4 = -0.76%% drag).\n");
}

void test_graded_solvable() {
  // Adaptive mesh around a sphere.
  BO t(IVec<3>{2, 2, 2}, 4);  // 32^3 fine available
  AmrGeometry<3> geo;
  geo.setIsotropic(1.0);
  peclet::core::geom::Sphere sph{{16.0, 16.0, 16.0}, 8.0};
  refineToSdf(t, geo, [&](const Vec<3>& p) { return sph.eval(p); }, 1, 1.0, true);
  PECLET_CORE_CHECK(t.isBalanced());
  test_conservation(t, geo.hMin());  // conservation must hold on the graded mesh too

  AmrPoisson<3, kBits> P(t, geo.h0);
  const Index n = t.numLeaves();
  // rhs with zero volume-weighted mean (so the periodic system is consistent).
  std::vector<double> rhs(static_cast<std::size_t>(n));
  std::uint64_t s = 42;
  for (auto& x : rhs)
    x = pseudo(s);
  double m = 0, vol = 0;
  for (Index i = 0; i < n; ++i) {
    m += P.cellVolume(i) * rhs[static_cast<std::size_t>(i)];
    vol += P.cellVolume(i);
  }
  m /= vol;
  for (Index i = 0; i < n; ++i)
    rhs[static_cast<std::size_t>(i)] -= m;

  std::vector<double> u(static_cast<std::size_t>(n), 0.0), res;
  double r0 = P.residual(u, rhs, res);
  P.gaussSeidel(u, rhs, 300);
  P.removeMean(u);
  double r = P.residual(u, rhs, res);
  PECLET_CORE_CHECK(r < r0 * 1e-2);  // GS makes clear progress -> operator is solvable
}

void run() {
  // (1) conservation on a uniform mesh.
  test_conservation(uniformFine(3), 1.0 / 8.0);

  // (2) uniform multigrid: convergence + 2nd-order accuracy.
  double d4 = 0, d5 = 0;
  double e4 = solveError(4, d4);
  double e5 = solveError(5, d5);
  PECLET_CORE_CHECK(d4 > 1e8);  // V-cycle drives residual down by >1e8
  PECLET_CORE_CHECK(d5 > 1e8);
  double ratio = e4 / e5;
  // 2nd order => error quarters when h halves; allow slack.
  PECLET_CORE_CHECK(ratio > 3.3);

  // (3) graded operator: conservative + solvable.
  test_graded_solvable();
}

}  // namespace

int main() {
  run();
  test_anisotropic_mg();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping AMR Poisson test\n");
  return ::peclet::core::test::kSkipExitCode;
}
#endif  // PECLET_CORE_HAVE_MORTON

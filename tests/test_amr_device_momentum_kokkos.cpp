// Device (Kokkos) momentum operator + solver for the AMR collocated flow step
// (tpx::amr::DeviceMomentumOp / DeviceMomentumSolver). Validates, against the host
// AmrCutCell on a uniform-fine sphere geometry:
//   (1) the assembled device matvec deviceApplyMom == host AmrCutCell::applyOp (to FP
//       tolerance — same coefficients; GPU differs only in the last bit by FMA);
//   (2) device weighted-Jacobi and Jacobi-preconditioned BiCGStab both solve A u = b to
//       the same solution the host serial gaussSeidel converges to;
//   (3) the matvec still matches with implicit-FOU advection assembled in
//       (buildAdvectionFou ⇒ the operator gains the upwind couplings);
//   (4) BiCGStab converges in the large-dt (weak-reaction, ill-conditioned) regime where
//       plain Jacobi stalls — the reason a Krylov accelerator is needed for momentum.
// Runs on whatever backend Kokkos targets (CUDA / HIP / OpenMP).
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/cut_cell.hpp"
#include "tpx/amr/device_momentum.hpp"
#include "tpx/common/types.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

constexpr double kLbox = 1.0;
constexpr double kR = 0.6;

BO uniformFine(unsigned L) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k) t.refineIf([](Code, unsigned) { return true; });
  return t;
}

void setDev(View<double> v, const std::vector<double>& h) {
  auto m = Kokkos::create_mirror_view(v);
  for (std::size_t i = 0; i < h.size(); ++i) m((Index)i) = h[i];
  Kokkos::deep_copy(v, m);
}
std::vector<double> getDev(View<double> v, Index n) {
  std::vector<double> h((std::size_t)n);
  auto m = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(m, v);
  for (Index i = 0; i < n; ++i) h[(std::size_t)i] = m(i);
  return h;
}

DeviceMomentumOp upload(const AmrCutCell<21>::Assembled& A) {
  DeviceMomentumOp op;
  op.n = (Index)A.diag.size();
  op.diag = toDevice(A.diag, "mom_diag");
  op.faceStart = toDevice(A.start, "mom_fstart");
  op.faceNbr = toDevice(A.nbr, "mom_fnbr");
  op.faceCoef = toDevice(A.coef, "mom_fcoef");
  return op;
}

double maxFluidErr(const std::vector<double>& a, const std::vector<double>& b,
                   const AmrCutCell<21>& cc, Index n) {
  double e = 0;
  for (Index i = 0; i < n; ++i)
    if (cc.isFluid(i)) e = std::max(e, std::fabs(a[(std::size_t)i] - b[(std::size_t)i]));
  return e;
}

void run() {
  const unsigned L = 4;
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 2.0 * kLbox / (double)N;
  const Index n = t.numLeaves();
  auto sdf = [&](const Vec<3>& p) {
    return kR - std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  };

  // Moderate-dt momentum operator: A = (ρ/dt)I − μ∇² (+ cut overlay). idiag = ρ/dt.
  const double rho = 1.0, mu = 1.0, dt = 0.25;
  const double idiag = rho / dt;
  AmrCutCell<21> cc;
  cc.init(t, h0, Vec<3>{-kLbox, -kLbox, -kLbox});
  cc.build(sdf, idiag, /*beta=*/mu / (h0 * h0), /*nsub=*/4);

  // ===== (1) matvec: device == host applyOp =====
  std::vector<double> u((std::size_t)n);
  for (Index i = 0; i < n; ++i) u[(std::size_t)i] = std::sin(0.3 * i) - 0.2 * std::cos(0.11 * i);
  {
    auto A = cc.assembleOperator();
    DeviceMomentumOp op = upload(A);
    View<double> du("u", (std::size_t)n), dAu("Au", (std::size_t)n);
    setDev(du, u);
    deviceApplyMom(op, View<const double>(du), dAu);
    auto devAu = getDev(dAu, n);
    std::vector<double> hostAu;
    cc.applyOp(u, hostAu);
    double e = 0, mag = 0;
    for (Index i = 0; i < n; ++i) {
      e = std::max(e, std::fabs(devAu[(std::size_t)i] - hostAu[(std::size_t)i]));
      mag = std::max(mag, std::fabs(hostAu[(std::size_t)i]));
    }
    std::printf("[mom] matvec max|dev-host| = %.3e (mag %.3e)\n", e, mag);
    TPX_CHECK(e < 1e-10 * (1.0 + mag));
  }

  // ===== (2) device solves reach the host gaussSeidel solution =====
  std::vector<double> src((std::size_t)n, 0.0);
  for (Index i = 0; i < n; ++i)
    if (cc.isFluid(i)) src[(std::size_t)i] = 1.0;  // unit body force
  std::vector<double> b = cc.makeRhs(src, 0.0);

  // host reference (serial GS to convergence)
  std::vector<double> uh((std::size_t)n, 0.0), res;
  double r0 = cc.residual(uh, b, res);
  for (int it = 0; it < 5000; ++it) {
    cc.gaussSeidel(uh, b, 20);
    if (cc.residual(uh, b, res) < r0 * 1e-12) break;
  }

  auto A = cc.assembleOperator();
  DeviceMomentumOp op = upload(A);
  View<double> db("b", (std::size_t)n), du("u", (std::size_t)n);
  setDev(db, b);

  DeviceMomentumSolver<21> solver;
  solver.setJacobi(2, 0.7);
  // BiCGStab
  Kokkos::deep_copy(du, 0.0);
  auto RB = solver.solveBiCGStab(op, du, View<const double>(db), 500, 1e-11);
  auto ub = getDev(du, n);
  double eb = maxFluidErr(ub, uh, cc, n);
  std::printf("[mom] BiCGStab: %d iters, res0 %.3e res %.3e, max|u-uGS| = %.3e\n", RB.iters,
              RB.res0, RB.res, eb);
  TPX_CHECK(RB.res < RB.res0 * 1e-9);
  TPX_CHECK(eb < 1e-7);

  // weighted Jacobi (diagonally dominant at dt=0.25 ⇒ converges)
  Kokkos::deep_copy(du, 0.0);
  double rj = RB.res0;
  for (int k = 0; k < 40; ++k) {
    rj = solver.solveJacobi(op, du, View<const double>(db), 50);
    if (rj < RB.res0 * 1e-9) break;
  }
  auto uj = getDev(du, n);
  double ej = maxFluidErr(uj, uh, cc, n);
  std::printf("[mom] Jacobi: res %.3e, max|u-uGS| = %.3e\n", rj, ej);
  TPX_CHECK(ej < 1e-6);

  // ===== (3) matvec still matches with implicit-FOU advection assembled in =====
  {
    std::array<std::vector<double>, 3> vel;
    for (int c = 0; c < 3; ++c) {
      vel[c].assign((std::size_t)n, 0.0);
      for (Index i = 0; i < n; ++i)
        if (cc.isFluid(i)) vel[c][(std::size_t)i] = 0.2 * std::sin(0.05 * i + c);
    }
    cc.buildAdvectionFou(vel, rho);
    auto Aa = cc.assembleOperator();
    DeviceMomentumOp opa = upload(Aa);
    View<double> dua("u", (std::size_t)n), dAu("Au", (std::size_t)n);
    setDev(dua, u);
    deviceApplyMom(opa, View<const double>(dua), dAu);
    auto devAu = getDev(dAu, n);
    std::vector<double> hostAu;
    cc.applyOp(u, hostAu);  // now includes advection
    double e = 0, mag = 0;
    for (Index i = 0; i < n; ++i) {
      e = std::max(e, std::fabs(devAu[(std::size_t)i] - hostAu[(std::size_t)i]));
      mag = std::max(mag, std::fabs(hostAu[(std::size_t)i]));
    }
    std::printf("[mom] advection matvec max|dev-host| = %.3e (mag %.3e)\n", e, mag);
    TPX_CHECK(e < 1e-10 * (1.0 + mag));
  }

  // ===== (4) large-dt (ill-conditioned) regime: BiCGStab converges where Jacobi stalls =====
  {
    const double dtBig = 1e6;
    AmrCutCell<21> cc2;
    cc2.init(t, h0, Vec<3>{-kLbox, -kLbox, -kLbox});
    cc2.build(sdf, rho / dtBig, mu / (h0 * h0), 4);
    std::vector<double> b2 = cc2.makeRhs(src, 0.0);
    auto A2 = cc2.assembleOperator();
    DeviceMomentumOp op2 = upload(A2);
    View<double> db2("b2", (std::size_t)n), du2("u2", (std::size_t)n);
    setDev(db2, b2);
    Kokkos::deep_copy(du2, 0.0);
    DeviceMomentumSolver<21> s2;
    s2.setJacobi(4, 0.7);
    auto R2 = s2.solveBiCGStab(op2, du2, View<const double>(db2), 2000, 1e-8);
    std::printf("[mom] large-dt BiCGStab: %d iters, rel res %.3e\n", R2.iters, R2.res / R2.res0);
    TPX_CHECK(R2.res < R2.res0 * 1e-6);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  TPX_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping device momentum test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

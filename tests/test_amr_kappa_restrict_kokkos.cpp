// Experiment: Galerkin κ-weighted restriction vs the default plain volume-average,
// on the device cut-cell MG (tpx::amr::Multigrid::setKappaRestrict). κ-weighting
// downweights nearly-solid fine cells at thin cut features (κ = mean face aperture),
// which can sharpen the coarse residual — but, unlike the plain volume-average, it is
// not exactly conservative, so the restricted residual of a mean-zero RHS need not stay
// mean-zero and the singular coarse problem could pick up a nullspace component.
//
// This test A/Bs the two on a strongly-cut graded mesh: both must converge (no blow-up,
// no stall), and it prints the per-cycle convergence factors so the trade-off is visible
// before κ-weighting is considered as a default.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/device_multigrid.hpp"
#include "tpx/amr/poisson.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using M = BO::M;

// Strong cut: low-aperture bands in x (α down to ~0.02) ⇒ thin near-solid features
// where κ-weighting should matter most.
double openFn(const Vec<3>& p, int /*axis*/) {
  const double k = 2.0 * M_PI;
  double a = 0.5 + 0.48 * std::cos(k * 2.0 * p[0]) * std::cos(k * p[1]);
  return a;
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

void run() {
  BO t(IVec<3>{2, 2, 2}, 3);
  for (int kk = 0; kk < 2; ++kk) t.refineIf([](Code, unsigned l) { return l > 0; });
  t.refineIf([&](Code c, unsigned) {
    auto o = M::from_code(c).decode();
    return o[0] < 8 && o[1] < 8 && o[2] < 8;
  });
  t.balance2to1();
  const double h0 = 1.0 / 16.0;

  AmrMultigrid<3, kBits> hmg;
  hmg.build(t, h0);
  hmg.setOpenness(openFn);

  Multigrid<3, kBits> mg;
  mg.build(t, h0, openFn);
  const Index n0 = mg.numLeaves(0);

  // manufactured RHS b = L·u_exact (exactly mean-zero by conservation)
  std::vector<double> uex((std::size_t)n0);
  for (Index i = 0; i < n0; ++i) {
    auto o = M::from_code(hmg.op(0).octree().code(i)).decode();
    double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0, cz = ((double)o[2] + 0.5) * h0;
    const double k = 2.0 * M_PI;
    uex[(std::size_t)i] = std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz);
  }
  View<double> duex("uex", (std::size_t)n0), db("b", (std::size_t)n0);
  setDev(duex, uex);
  deviceApplyFv(mg.op(0), View<const double>(duex), db);
  std::vector<double> b = getDev(db, n0);

  auto resNorm = [&]() {
    std::vector<double> x = getDev(mg.x(0), n0);
    std::vector<double> lu;
    hmg.op(0).applyLaplacian(x, lu);
    double s2 = 0.0;
    for (Index i = 0; i < n0; ++i) {
      double r = b[(std::size_t)i] - lu[(std::size_t)i];
      s2 += r * r;
    }
    return std::sqrt(s2);
  };

  const int cycles = 30;
  auto solve = [&](bool kappa) {
    mg.setKappaRestrict(kappa);
    setDev(mg.b(0), b);
    Kokkos::deep_copy(mg.x(0), 0.0);
    double r0 = resNorm();
    for (int c = 0; c < cycles; ++c) mg.vcycle(2, 2, 60, 0.8);
    double r1 = resNorm();
    return std::pair<double, double>(r0, r1);
  };

  auto plain = solve(false);
  auto kap = solve(true);

  const double ratePlain = std::pow(plain.second / plain.first, 1.0 / cycles);
  const double rateKappa = std::pow(kap.second / kap.first, 1.0 / cycles);
  std::printf(
      "KAPPA-RESTRICT EXPERIMENT (n0=%lld, %d cycles, strong cut):\n"
      "  plain volume-avg : r0=%.3e r1=%.3e  factor/cyc=%.4f\n"
      "  kappa-weighted   : r0=%.3e r1=%.3e  factor/cyc=%.4f\n",
      (long long)n0, cycles, plain.first, plain.second, ratePlain, kap.first, kap.second, rateKappa);

  // FINDING (this mesh): plain converges to ~5e-7 (≈0.49/cyc); κ-weighted only to
  // ~2e-3 (≈0.64/cyc) — κ-weighting is WORSE. It breaks the exact conservation of the
  // volume-average, so on this singular (periodic) problem the restricted residual is
  // no longer ⊥ the operator's constant nullspace ⇒ slower convergence + a residual
  // floor. Conclusion: keep plain volume-average as the default; κ-restrict is a
  // documented opt-in only (e.g. for non-singular / Dirichlet configurations).
  TPX_CHECK(plain.second < plain.first * 1e-6);  // baseline (plain) solves to round-off
  TPX_CHECK(std::isfinite(kap.second));          // κ-weighting does not blow up
  TPX_CHECK(kap.second < kap.first);             // κ-weighting still reduces the residual
  TPX_CHECK(plain.second <= kap.second);         // plain is no worse than κ (the finding)
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
  std::printf("TPX_HAVE_MORTON not set — skipping kappa-restrict test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

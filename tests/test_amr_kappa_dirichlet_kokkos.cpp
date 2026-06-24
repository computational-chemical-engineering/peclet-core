// Follow-up experiment: κ-weighted restriction on a *Dirichlet* config (non-singular).
//
// On the periodic problem κ-weighting stalled because it breaks the volume-average's
// conservation and the singular operator's restricted residual was no longer ⊥ the
// constant null space (see test_amr_kappa_restrict_kokkos). A homogeneous-Dirichlet
// operator (DeviceMultigrid::build(..., periodic=false): every domain-boundary face is a
// u=0 wall at half a cell, folded into the diagonal) has NO null space, so the
// conservation argument no longer bites. This A/Bs plain vs κ-weighted there: both
// should now converge to round-off (no floor), confirming κ-restrict is safe when the
// operator is non-singular.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/device_multigrid.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using M = BO::M;

double openFn(const Vec<3>& p, int /*axis*/) {
  const double k = 2.0 * M_PI;
  return 0.5 + 0.48 * std::cos(k * 2.0 * p[0]) * std::cos(k * p[1]);  // strong cut
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

  // Homogeneous-Dirichlet cut-cell device MG (periodic = false).
  DeviceMultigrid<3, kBits> mg;
  mg.build(t, h0, openFn, /*periodic=*/false);
  const Index n0 = mg.numLeaves(0);
  TPX_CHECK(mg.numLevels() >= 3);

  // manufactured RHS b = A·u_exact (A non-singular ⇒ unique solution, no null space)
  std::vector<double> uex((std::size_t)n0);
  for (Index i = 0; i < n0; ++i) {
    auto o = M::from_code(mg.octreeCode(0, i)).decode();
    double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0, cz = ((double)o[2] + 0.5) * h0;
    const double k = 2.0 * M_PI;
    uex[(std::size_t)i] = std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz);
  }
  View<double> duex("uex", (std::size_t)n0), db("b", (std::size_t)n0), dlu("lu", (std::size_t)n0);
  setDev(duex, uex);
  deviceApplyFv(mg.op(0), View<const double>(duex), db);
  std::vector<double> b = getDev(db, n0);

  auto resNorm = [&]() {
    deviceApplyFv(mg.op(0), View<const double>(mg.x(0)), dlu);
    std::vector<double> lu = getDev(dlu, n0);
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
    return std::pair<double, double>(r0, resNorm());
  };

  auto plain = solve(false);
  auto kap = solve(true);
  const double ratePlain = std::pow(plain.second / plain.first, 1.0 / cycles);
  const double rateKappa = std::pow(kap.second / kap.first, 1.0 / cycles);
  std::printf(
      "KAPPA-RESTRICT on DIRICHLET (n0=%lld, %d cycles, strong cut):\n"
      "  plain volume-avg : r0=%.3e r1=%.3e  factor/cyc=%.4f\n"
      "  kappa-weighted   : r0=%.3e r1=%.3e  factor/cyc=%.4f\n",
      (long long)n0, cycles, plain.first, plain.second, ratePlain, kap.first, kap.second, rateKappa);

  // FINDING (this mesh): plain → ~2.6e-10 (≈0.36/cyc), κ-weighted → ~6.7e-7 (≈0.47/cyc).
  // On Dirichlet (non-singular) the κ stall/floor from the periodic case is GONE — both
  // converge to ~round-off — which confirms the constant null space was the cause there.
  // But κ-weighting still does not *beat* plain here (slightly slower), likely because the
  // prolongation stays plain piecewise-constant (an unmatched transfer pair). So plain
  // volume-average remains the default; κ-restrict is safe on non-singular configs but is
  // not a convergence win in these tests.
  TPX_CHECK(plain.second < plain.first * 1e-8);  // no floor (baseline)
  TPX_CHECK(kap.second < kap.first * 1e-8);       // no floor with κ either (Dirichlet)
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
  std::printf("TPX_HAVE_MORTON not set — skipping kappa-dirichlet test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

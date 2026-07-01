// Device cut-cell MOMENTUM operator ASSEMBLY (peclet::core::amr::deviceAssembleMomentum, on the S1 CSR
// primitive) must reproduce host AmrCutCell::build (Pass 2 buildCutStencil) + assembleOperator
// bit-for-bit on the OpenMP backend: the ξ-overlay stencil rebuild (AC/off/cut/rscale), and the merged
// diag + face-CSR over the three per-cell branches (solid identity / ξ-overlay / regular ∇²·μ). This is
// the D3 anti-drift lock.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <cmath>
#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/block_octree_view.hpp"
#include "peclet/core/amr/cut_cell.hpp"
#include "peclet/core/amr/device_momentum_assembly.hpp"
#include "peclet/core/amr/momentum.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;

template <class T>
std::vector<T> down(const View<T>& d) {
  std::vector<T> h(d.extent(0));
  auto m = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(m, d);
  for (std::size_t i = 0; i < h.size(); ++i) h[i] = m(i);
  return h;
}

template <class A, class B>
int mismatch(const std::vector<A>& a, const std::vector<B>& b) {
  if (a.size() != b.size()) return -1;
  int m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (static_cast<double>(a[i]) != static_cast<double>(b[i])) ++m;
  return m;
}

void run() {
  // A 2:1-refined, balanced octree so the regular-fluid branch exercises C/F (2:1) face coupling.
  BO t(IVec<3>{2, 2, 2}, 3);
  auto refineAt = [&](std::array<BO::Coord, 3> p) {
    Index leaf = t.find(p);
    if (leaf >= 0 && t.level(leaf) > 0) {
      Code target = t.code(leaf);
      t.refineIf([&](Code c, unsigned) { return c == target; });
    }
  };
  refineAt({0, 0, 0});
  refineAt({15, 15, 15});
  refineAt({8, 8, 8});
  t.balance2to1();

  const double h0 = 0.1;
  // A sphere centred in the block ⇒ a mix of solid (inside), cut (straddling), and regular fluid cells.
  const Vec<3> org{-0.8, -0.8, -0.8};
  const double R = 0.45;
  AmrCutCell<kBits> cc;
  cc.init(t, h0, org);
  auto sdf = [&](const Vec<3>& p) {
    return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - R;  // >0 outside sphere = fluid
  };
  cc.build(sdf, /*idiag=*/2.0, /*beta=*/1.0, /*nsub=*/4);

  const Index n = t.numLeaves();
  BlockOctreeView<3, kBits> dev;
  dev.upload(t);

  // ---- (1) device ξ-stencil rebuild == host build() Pass 2 ----
  {
    const double beta = cc.beta();
    const double AC0 = cc.idiag() + 6.0 * beta;
    View<double> sdfC = toDevice(cc.sdfCRaw(), "sdfC");
    View<Index> nb = toDevice(cc.nbRaw(), "nb");
    View<char> fluid = toDevice(cc.fluidRaw(), "fl");
    View<double> AC("AC", static_cast<std::size_t>(n)), off("off", static_cast<std::size_t>(n) * 6),
        rscale("rs", static_cast<std::size_t>(n));
    View<char> cut("cut", static_cast<std::size_t>(n));
    deviceRebuildCutStencil<kBits>(n, beta, AC0, sdfC, nb, fluid, AC, off, cut, rscale);
    PECLET_CORE_CHECK_EQ(mismatch(cc.acRaw(), down(AC)), 0);
    PECLET_CORE_CHECK_EQ(mismatch(cc.offRaw(), down(off)), 0);
    PECLET_CORE_CHECK_EQ(mismatch(cc.cutRaw(), down(cut)), 0);
    PECLET_CORE_CHECK_EQ(mismatch(cc.rscaleRaw(), down(rscale)), 0);
  }

  // ---- (2) device assembled MomentumOp == host assembleOperator (diag/start/nbr/coef) ----
  const auto H = cc.assembleOperator(/*scaleAdvByRscale=*/false);
  MomentumOp op = deviceAssembleMomentum<kBits>(cc, dev, /*scaleAdvByRscale=*/false);
  PECLET_CORE_CHECK_EQ(static_cast<Index>(op.n), n);
  std::vector<double> ddiag = down(op.diag);
  std::vector<Index> dstart = down(op.faceStart);
  std::vector<Index> dnbr = down(op.faceNbr);
  std::vector<double> dcoef = down(op.faceCoef);
  std::printf("  n=%lld nnz host=%lld dev=%lld\n", static_cast<long long>(n),
              static_cast<long long>(H.nbr.size()), static_cast<long long>(dnbr.size()));
  PECLET_CORE_CHECK_EQ(mismatch(H.diag, ddiag), 0);
  PECLET_CORE_CHECK_EQ(mismatch(H.start, dstart), 0);
  PECLET_CORE_CHECK_EQ(mismatch(H.nbr, dnbr), 0);
  PECLET_CORE_CHECK_EQ(mismatch(H.coef, dcoef), 0);

  // ---- (3) deviceApplyMom == host applyOp over the assembled operator ----
  std::vector<double> x(static_cast<std::size_t>(n));
  std::uint64_t s = 0xda3e39cb94b95bdbULL;
  for (auto& v : x) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    v = static_cast<double>((s >> 11) & 0xFFFFF) / static_cast<double>(0x100000) - 0.5;
  }
  std::vector<double> hout;
  cc.applyOp(x, hout);
  View<const double> dx = toDevice(x, "x");
  View<double> dAu("Au", static_cast<std::size_t>(n));
  deviceApplyMom(op, dx, dAu);
  PECLET_CORE_CHECK_EQ(mismatch(hout, down(dAu)), 0);
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
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping device momentum assembly test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

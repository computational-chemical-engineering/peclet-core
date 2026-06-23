// Device (Kokkos) geometric-multigrid V-cycle (tpx::amr::DeviceMultigrid): a full MG
// V-cycle running in device kernels over the leaf Views must (1) match the same Jacobi
// V-cycle on the host bit-for-bit (Jacobi smoother + CSR-ordered average restriction +
// piecewise-constant prolongation are all deterministic / order-independent), and
// (2) actually solve — the Poisson residual drops by orders of magnitude in a few
// cycles. Runs on whatever backend Kokkos was built for (CUDA / HIP / OpenMP).
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <algorithm>
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

// ---- host mirror of DeviceMultigrid (identical hierarchy, transfers, smoother) ----
struct HLevel {
  BO t{IVec<3>{1, 1, 1}, 0};
  Index n = 0;
  double inv = 1.0;
  std::vector<Index> c2p, cstart, cidx;
  std::vector<double> x, b, res, ax;
};

double hlap(const BO& t, const std::vector<double>& x, Index i) {
  double s = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int d = -1; d <= 1; d += 2) {
      Index j = t.faceNeighbor(i, a, d);
      if (j >= 0) s += x[(std::size_t)j] - x[(std::size_t)i];
    }
  return s;
}

void buildHost(const BO& finest, double h0, std::vector<HLevel>& lv) {
  std::vector<BO> host;
  host.push_back(finest);
  for (;;) {
    BO c = host.back();
    Index m = c.coarsenIf([](Code, unsigned) { return true; });
    if (m == 0 || c.numLeaves() == host.back().numLeaves()) break;
    host.push_back(c);
    if (c.numLeaves() == 1) break;
  }
  lv.resize(host.size());
  for (std::size_t L = 0; L < host.size(); ++L) {
    lv[L].t = host[L];
    lv[L].n = host[L].numLeaves();
    double hL = h0 * (double)(Index(1) << L);
    lv[L].inv = 1.0 / (hL * hL);
    std::size_t n = (std::size_t)lv[L].n;
    lv[L].x.assign(n, 0.0);
    lv[L].b.assign(n, 0.0);
    lv[L].res.assign(n, 0.0);
    lv[L].ax.assign(n, 0.0);
  }
  for (std::size_t L = 0; L + 1 < host.size(); ++L) {
    BO& f = lv[L].t;
    BO& c = lv[L + 1].t;
    Index nf = f.numLeaves(), nc = c.numLeaves();
    lv[L].c2p.assign((std::size_t)nf, -1);
    std::vector<Index> cnt((std::size_t)nc, 0);
    for (Index i = 0; i < nf; ++i) {
      Code par = M::from_code(f.code(i)).ancestor(f.level(i) + 1).code();
      Index p = c.find(par);
      lv[L].c2p[(std::size_t)i] = p;
      if (p >= 0) ++cnt[(std::size_t)p];
    }
    lv[L].cstart.assign((std::size_t)nc + 1, 0);
    for (Index p = 0; p < nc; ++p) lv[L].cstart[(std::size_t)p + 1] = lv[L].cstart[(std::size_t)p] + cnt[(std::size_t)p];
    lv[L].cidx.assign((std::size_t)nf, 0);
    std::vector<Index> cur(lv[L].cstart.begin(), lv[L].cstart.end() - 1);
    for (Index i = 0; i < nf; ++i) {
      Index p = lv[L].c2p[(std::size_t)i];
      if (p >= 0) lv[L].cidx[(std::size_t)(cur[(std::size_t)p]++)] = i;
    }
  }
}

void hjac(HLevel& L, int sw, double om) {
  double diag = 2.0 * 3 * L.inv;
  for (int s = 0; s < sw; ++s) {
    for (Index i = 0; i < L.n; ++i) L.ax[(std::size_t)i] = -L.inv * hlap(L.t, L.x, i);
    for (Index i = 0; i < L.n; ++i) L.x[(std::size_t)i] += om * (L.b[(std::size_t)i] - L.ax[(std::size_t)i]) / diag;
  }
}

void hvcycle(std::vector<HLevel>& lv, std::size_t L, int pre, int post, int bot, double om) {
  if (L + 1 == lv.size()) {
    hjac(lv[L], bot, om);
    return;
  }
  hjac(lv[L], pre, om);
  for (Index i = 0; i < lv[L].n; ++i) lv[L].res[(std::size_t)i] = lv[L].b[(std::size_t)i] + lv[L].inv * hlap(lv[L].t, lv[L].x, i);
  HLevel& c = lv[L + 1];
  for (Index p = 0; p < c.n; ++p) {
    Index a = lv[L].cstart[(std::size_t)p], z = lv[L].cstart[(std::size_t)p + 1];
    double s = 0.0;
    for (Index k = a; k < z; ++k) s += lv[L].res[(std::size_t)lv[L].cidx[(std::size_t)k]];
    c.b[(std::size_t)p] = (z > a) ? s / (double)(z - a) : 0.0;
  }
  std::fill(c.x.begin(), c.x.end(), 0.0);
  hvcycle(lv, L + 1, pre, post, bot, om);
  for (Index i = 0; i < lv[L].n; ++i) {
    Index p = lv[L].c2p[(std::size_t)i];
    if (p >= 0) lv[L].x[(std::size_t)i] += c.x[(std::size_t)p];
  }
  hjac(lv[L], post, om);
}

// Smooth, mean-zero RHS keyed on each leaf centroid (compatible with the constant
// nullspace of the pure-Neumann operator).
std::vector<double> smoothRhs(const BO& t, double h0) {
  const Index n = t.numLeaves();
  std::vector<double> b((std::size_t)n);
  const double k = 2.0 * M_PI;
  double mean = 0.0;
  for (Index i = 0; i < n; ++i) {
    auto o = M::from_code(t.code(i)).decode();
    double half = 0.5 * (double)(Index(1) << t.level(i));
    double cx = ((double)o[0] + half) * h0, cy = ((double)o[1] + half) * h0, cz = ((double)o[2] + half) * h0;
    b[(std::size_t)i] = std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz);
    mean += b[(std::size_t)i];
  }
  mean /= (double)n;
  for (auto& v : b) v -= mean;
  return b;
}

void run() {
  // Graded multilevel octree: a 2×2×2 brick at level 3 (domain 16^3 fine), refined
  // uniformly down to level 1 (uniform 8^3), then the lower octant down to level 0,
  // 2:1-balanced. Coarsening yields ~4 MG levels with mixed child counts (the level-0
  // region vs untouched level-1 cells) — exercising the transfer maps both ways.
  BO t(IVec<3>{2, 2, 2}, 3);
  for (int k = 0; k < 2; ++k) t.refineIf([](Code, unsigned l) { return l > 0; });
  t.refineIf([&](Code c, unsigned) {
    auto o = M::from_code(c).decode();
    return o[0] < 8 && o[1] < 8 && o[2] < 8;  // lower octant (fine coords) → level 0
  });
  t.balance2to1();
  const Index n = t.numLeaves();
  const double h0 = 1.0 / 16.0;  // domain ≈ [0,1)^3
  const int cycles = 8;

  // ===== (1) device == host bit-for-bit, on the GRADED mesh =====
  // (exercises the transfer maps across the 2:1 interface, both directions).
  std::vector<double> b = smoothRhs(t, h0);
  DeviceMultigrid<3, kBits> mg;
  mg.build(t, h0);
  TPX_CHECK_EQ(mg.numLeaves(0), n);
  {
    auto hb = Kokkos::create_mirror_view(mg.b(0));
    for (Index i = 0; i < n; ++i) hb(i) = b[(std::size_t)i];
    Kokkos::deep_copy(mg.b(0), hb);
  }
  for (int c = 0; c < cycles; ++c) mg.vcycle();
  std::vector<double> dx((std::size_t)n);
  {
    auto hx = Kokkos::create_mirror_view(mg.x(0));
    Kokkos::deep_copy(hx, mg.x(0));
    for (Index i = 0; i < n; ++i) dx[(std::size_t)i] = hx(i);
  }
  std::vector<HLevel> lv;
  buildHost(t, h0, lv);
  lv[0].b = b;
  for (int c = 0; c < cycles; ++c) hvcycle(lv, 0, 2, 2, 30, 0.8);
  int mism = 0;
  for (Index i = 0; i < n; ++i)
    if (dx[(std::size_t)i] != lv[0].x[(std::size_t)i]) ++mism;
  TPX_CHECK_EQ(mism, 0);
  TPX_CHECK(lv.size() >= 3);

  // ===== (2) the V-cycle actually solves, on a UNIFORM mesh =====
  // Where the plain Laplacian is a consistent discretization the V-cycle converges
  // properly (the graded mesh's 2:1 interface degrades the plain operator — the
  // consistent graded operator is host AmrPoisson's quadratic flux, out of scope for
  // this device plain-Laplacian milestone). Build a uniform 16^3 (full refine to
  // level 0) and check the residual drops by orders of magnitude.
  BO tu(IVec<3>{2, 2, 2}, 3);
  for (int kk = 0; kk < 3; ++kk) tu.refineIf([](Code, unsigned l) { return l > 0; });
  const Index nu = tu.numLeaves();
  std::vector<double> bu = smoothRhs(tu, h0);
  DeviceMultigrid<3, kBits> mu;
  mu.build(tu, h0);
  {
    auto hb = Kokkos::create_mirror_view(mu.b(0));
    for (Index i = 0; i < nu; ++i) hb(i) = bu[(std::size_t)i];
    Kokkos::deep_copy(mu.b(0), hb);
  }
  for (int c = 0; c < cycles; ++c) mu.vcycle();
  std::vector<double> xu((std::size_t)nu);
  {
    auto hx = Kokkos::create_mirror_view(mu.x(0));
    Kokkos::deep_copy(hx, mu.x(0));
    for (Index i = 0; i < nu; ++i) xu[(std::size_t)i] = hx(i);
  }
  const double inv0 = 1.0 / (h0 * h0);
  double r0 = 0.0, r1 = 0.0;
  for (Index i = 0; i < nu; ++i) {
    r0 += bu[(std::size_t)i] * bu[(std::size_t)i];
    double r = bu[(std::size_t)i] + inv0 * hlap(tu, xu, i);
    r1 += r * r;
  }
  TPX_CHECK(std::sqrt(r1) < std::sqrt(r0) * 1e-2);
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
  std::printf("TPX_HAVE_MORTON not set — skipping device multigrid test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

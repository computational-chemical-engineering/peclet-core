// Device (Kokkos) geometric-multigrid V-cycle with the CONSISTENT graded operator
// (tpx::amr::Multigrid). Validates, on a genuinely graded octree:
//   (1) the device consistent FV operator (deviceApplyFv over the face CSR) ==
//       host AmrPoisson::applyLaplacian bit-for-bit (same coeffs, same 2:1 sub-faces);
//   (2) the full standard V-cycle == a host Jacobi-MG mirror bit-for-bit, AND it now
//       *converges on the graded mesh* (the plain operator stalled there — this is the
//       point of folding in the consistent operator);
//   (3) the quadratic coarse-fine correction: deviceQuadDelta == host
//       (applyLaplacianQuad − applyLaplacian) to round-off, and solveQuad (deferred
//       correction) drives the 2nd-order graded residual down.
// Runs on whatever backend Kokkos was built for (CUDA / HIP / OpenMP).
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <algorithm>
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/multigrid.hpp"
#include "tpx/amr/poisson.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using M = BO::M;
using AP = AmrPoisson<3, kBits>;

// Smooth, mean-zero RHS keyed on each leaf centroid (compatible with the constant
// nullspace of the periodic operator).
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

// ---- host Jacobi-MG mirror of Multigrid (consistent operator) ----
struct HRef {
  std::vector<BO> lvl;
  std::vector<AP> ap;
  std::vector<std::vector<Index>> c2p;
  std::vector<std::vector<double>> x, b, res, tmp;

  void build(const BO& finest, double h0) {
    lvl.push_back(finest);
    for (;;) {
      BO c = lvl.back();
      Index m = c.coarsenIf([](Code, unsigned) { return true; });
      if (m == 0 || c.numLeaves() == lvl.back().numLeaves()) break;
      lvl.push_back(c);
      if (c.numLeaves() == 1) break;
    }
    ap.resize(lvl.size());
    for (std::size_t L = 0; L < lvl.size(); ++L) {
      ap[L].init(lvl[L], h0);
      std::size_t n = (std::size_t)lvl[L].numLeaves();
      x.emplace_back(n, 0.0);
      b.emplace_back(n, 0.0);
      res.emplace_back(n, 0.0);
      tmp.emplace_back(n, 0.0);
    }
    c2p.resize(lvl.size());
    for (std::size_t L = 0; L + 1 < lvl.size(); ++L) {
      BO& f = lvl[L];
      BO& c = lvl[L + 1];
      Index nf = f.numLeaves();
      c2p[L].assign((std::size_t)nf, -1);
      for (Index i = 0; i < nf; ++i) {
        Code par = M::from_code(f.code(i)).ancestor(f.level(i) + 1).code();
        c2p[L][(std::size_t)i] = c.find(par);
      }
    }
  }
  void jac(std::size_t L, int sw, double om) {
    AP& P = ap[L];
    Index n = lvl[L].numLeaves();
    for (int s = 0; s < sw; ++s) {
      for (Index i = 0; i < n; ++i) {
        double so = 0, dg = 0;
        P.forEachFaceNeighbor(i, [&](Index j, Real c, int, double a) {
          so += a * c * x[L][(std::size_t)j];
          dg += a * c;
        });
        tmp[L][(std::size_t)i] = (so - b[L][(std::size_t)i] * P.cellVolume(i)) / dg;
      }
      for (Index i = 0; i < n; ++i) x[L][(std::size_t)i] = (1 - om) * x[L][(std::size_t)i] + om * tmp[L][(std::size_t)i];
    }
  }
  void resid(std::size_t L) {
    std::vector<double> lu;
    ap[L].applyLaplacian(x[L], lu);
    Index n = lvl[L].numLeaves();
    for (Index i = 0; i < n; ++i) res[L][(std::size_t)i] = b[L][(std::size_t)i] - lu[(std::size_t)i];
  }
  void vcyc(std::size_t L, int pre, int post, int bot, double om) {
    if (L + 1 == lvl.size()) {
      jac(L, bot, om);
      return;
    }
    jac(L, pre, om);
    resid(L);
    Index nc = lvl[L + 1].numLeaves(), nf = lvl[L].numLeaves();
    std::vector<double> cb((std::size_t)nc, 0), cn((std::size_t)nc, 0);
    for (Index i = 0; i < nf; ++i) {
      Index p = c2p[L][(std::size_t)i];
      if (p >= 0) {
        cb[(std::size_t)p] += res[L][(std::size_t)i];
        cn[(std::size_t)p] += 1;
      }
    }
    for (Index p = 0; p < nc; ++p)
      if (cn[(std::size_t)p] > 0) cb[(std::size_t)p] /= cn[(std::size_t)p];
    std::fill(x[L + 1].begin(), x[L + 1].end(), 0.0);
    b[L + 1] = cb;
    vcyc(L + 1, pre, post, bot, om);
    for (Index i = 0; i < nf; ++i) {
      Index p = c2p[L][(std::size_t)i];
      if (p >= 0) x[L][(std::size_t)i] += x[L + 1][(std::size_t)p];
    }
    jac(L, post, om);
  }
};

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
  // Graded multilevel octree: 2×2×2 brick (level 3, domain 16^3), refined uniformly
  // to level 1 then the lower octant to level 0, 2:1-balanced.
  BO t(IVec<3>{2, 2, 2}, 3);
  for (int kk = 0; kk < 2; ++kk) t.refineIf([](Code, unsigned l) { return l > 0; });
  t.refineIf([&](Code c, unsigned) {
    auto o = M::from_code(c).decode();
    return o[0] < 8 && o[1] < 8 && o[2] < 8;
  });
  t.balance2to1();
  const Index n = t.numLeaves();
  const double h0 = 1.0 / 16.0;

  Multigrid<3, kBits> mg;
  mg.build(t, h0);
  TPX_CHECK_EQ(mg.numLeaves(0), n);
  TPX_CHECK(mg.numLevels() >= 3);

  AP ap0;
  ap0.init(t, h0);

  // ===== (1) consistent operator: deviceApplyFv == host applyLaplacian (bit-exact) =====
  std::vector<double> xr((std::size_t)n);
  for (Index i = 0; i < n; ++i) xr[(std::size_t)i] = std::sin(0.3 * i) - 0.2 * std::cos(0.13 * i);
  {
    View<double> dxr("xr", (std::size_t)n), dLu("Lu", (std::size_t)n);
    setDev(dxr, xr);
    deviceApplyFv(mg.op(0), View<const double>(dxr), dLu);
    std::vector<double> hLu;
    ap0.applyLaplacian(xr, hLu);
    auto dLh = getDev(dLu, n);
    int mism = 0;
    for (Index i = 0; i < n; ++i)
      if (dLh[(std::size_t)i] != hLu[(std::size_t)i]) ++mism;
    TPX_CHECK_EQ(mism, 0);
  }

  // ===== (2) standard V-cycle: device == host bit-for-bit + converges on graded mesh =====
  std::vector<double> b = smoothRhs(t, h0);
  setDev(mg.b(0), b);
  Kokkos::deep_copy(mg.x(0), 0.0);
  HRef hr;
  hr.build(t, h0);
  hr.b[0] = b;
  const int cycles = 12;
  for (int c = 0; c < cycles; ++c) {
    mg.vcycle(2, 2, 40, 0.8);
    hr.vcyc(0, 2, 2, 40, 0.8);
  }
  auto dx = getDev(mg.x(0), n);
  int vmis = 0;
  for (Index i = 0; i < n; ++i)
    if (dx[(std::size_t)i] != hr.x[0][(std::size_t)i]) ++vmis;
  TPX_CHECK_EQ(vmis, 0);
  // converges on the graded mesh (plain operator stalled here ~0.03)
  std::vector<double> lu;
  ap0.applyLaplacian(dx, lu);
  double r0 = 0, r1 = 0;
  for (Index i = 0; i < n; ++i) {
    r0 += b[(std::size_t)i] * b[(std::size_t)i];
    double rr = b[(std::size_t)i] - lu[(std::size_t)i];
    r1 += rr * rr;
  }
  TPX_CHECK(std::sqrt(r1) < std::sqrt(r0) * 1e-3);

  // ===== (3) quadratic coarse-fine correction =====
  // deviceQuadDelta(xr) ≈ host (applyLaplacianQuad − applyLaplacian)(xr)
  {
    View<double> dxr("xr2", (std::size_t)n), ddq("dq", (std::size_t)n);
    setDev(dxr, xr);
    deviceQuadDelta(View<const Index>(mg.quadStart()), View<const Index>(mg.quadSlot()),
                    View<const double>(mg.quadCoef()), View<const double>(dxr), ddq, n);
    std::vector<double> lq, ls;
    ap0.applyLaplacianQuad(xr, lq);
    ap0.applyLaplacian(xr, ls);
    auto dqh = getDev(ddq, n);
    double maxabs = 0, maxerr = 0;
    for (Index i = 0; i < n; ++i) {
      double ref = lq[(std::size_t)i] - ls[(std::size_t)i];
      maxabs = std::max(maxabs, std::fabs(ref));
      maxerr = std::max(maxerr, std::fabs(dqh[(std::size_t)i] - ref));
    }
    TPX_CHECK(maxabs > 0.0);          // the correction is actually exercised
    TPX_CHECK(maxerr < 1e-9 * (1.0 + maxabs));
  }
  // solveQuad drives the 2nd-order graded residual down
  setDev(mg.b(0), b);
  Kokkos::deep_copy(mg.x(0), 0.0);
  double rq = mg.solveQuad(40, 1, 2, 2, 40, 0.8);
  TPX_CHECK(rq < std::sqrt(r0) * 1e-2);
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

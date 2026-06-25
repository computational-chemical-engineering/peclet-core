// Benchmark: AMR collocated Stokes flow step + Poisson solve, host (serial AmrFlow) vs
// device (DeviceAmrFlow / DevicePCG) on whatever backend Kokkos targets (OpenMP multicore
// or CUDA/HIP GPU). Reports ms/step and throughput (Mcell/s), the host/device speedup, and
// — for the Poisson half — V-cycle vs MG-PCG iterations/time to a fixed tolerance.
//
//   ./bench_amr_flow [Lmax=5] [steps=10]
//
// Guarded by TPX_HAVE_MORTON; prints a notice and exits otherwise.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef TPX_HAVE_MORTON
#include <Kokkos_Core.hpp>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/device_flow.hpp"
#include "tpx/amr/device_multigrid.hpp"
#include "tpx/amr/device_pcg.hpp"
#include "tpx/amr/flow.hpp"
#include "tpx/amr/poisson.hpp"

using namespace tpx;
using namespace tpx::amr;
using Clock = std::chrono::steady_clock;
using BO = BlockOctree<3, 21>;
using Code = BO::Code;

namespace {
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
BO uniformFine(unsigned L) {
  BO t(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k) t.refineIf([](Code, unsigned) { return true; });
  return t;
}

// Periodic sphere array (cut geometry) — the Zick&Homsy-style drag config.
auto sphereSdf(const Vec<3>& c, double rad) {
  return [c, rad](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - rad;
  };
}

// Host vs device flow-step time (only at sizes where the serial host is affordable).
void benchFlowCompare(unsigned L, int steps) {
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / (double)N;
  const Index n = t.numLeaves();
  auto sdf = sphereSdf(Vec<3>{0.5, 0.5, 0.5}, 0.25);

  AmrFlow<21> hfl;
  hfl.init(t, h0);
  hfl.setViscosity(1.0);
  hfl.setDt(1e6);
  hfl.setBodyForce(1.0, 0, 0);
  hfl.setSolid(sdf);
  hfl.step(200, 5, 2);  // warm up
  auto h0t = Clock::now();
  for (int s = 0; s < steps; ++s) hfl.step(200, 5, 2);
  double hostMs = ms(h0t, Clock::now()) / steps;

  DeviceAmrFlow<21> dfl;
  dfl.init(t, h0);
  dfl.setViscosity(1.0);
  dfl.setDt(1e6);
  dfl.setBodyForce(1.0, 0, 0);
  dfl.setSolid(sdf);
  dfl.step(200, 40);
  Kokkos::fence();
  auto d0t = Clock::now();
  for (int s = 0; s < steps; ++s) dfl.step(200, 40);
  Kokkos::fence();
  double devMs = ms(d0t, Clock::now()) / steps;

  std::printf("flow  L=%u  N=%ld^3  cells=%7ld :  host %9.2f ms/step  device %8.2f ms/step  "
              "speedup %6.1fx\n",
              L, N, (long)n, hostMs, devMs, hostMs / devMs);
}

// Device-only flow-step throughput scaling (host too slow at these sizes).
void benchFlowDevice(unsigned L, int steps) {
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / (double)N;
  const Index n = t.numLeaves();
  auto sdf = sphereSdf(Vec<3>{0.5, 0.5, 0.5}, 0.25);

  DeviceAmrFlow<21> dfl;
  dfl.init(t, h0);
  dfl.setViscosity(1.0);
  dfl.setDt(1e6);
  dfl.setBodyForce(1.0, 0, 0);
  dfl.setSolid(sdf);
  dfl.step(200, 40);
  Kokkos::fence();
  auto d0t = Clock::now();
  for (int s = 0; s < steps; ++s) dfl.step(200, 40);
  Kokkos::fence();
  double devMs = ms(d0t, Clock::now()) / steps;
  std::printf("flow  L=%u  N=%ld^3  cells=%8ld :  device %9.2f ms/step  (%7.1f Mcell/s)\n", L, N,
              (long)n, devMs, (double)n / devMs / 1e3);
}

// Poisson solver micro-benchmark: V-cycle vs MG-PCG to a fixed relative tolerance.
void benchPoisson(unsigned L) {
  BO t = uniformFine(L);
  const long N = 1L << L;
  const double h0 = 1.0 / (double)N;
  const Index n = t.numLeaves();
  AmrPoisson<3, 21> ap;
  ap.init(t, h0);
  std::vector<double> uex((std::size_t)n), b;
  const double k = 2.0 * M_PI;
  for (Index i = 0; i < n; ++i) {
    auto o = BO::M::from_code(t.code(i)).decode();
    double half = 0.5 * (1 << t.level(i));
    uex[(std::size_t)i] = std::sin(k * ((o[0] + half) * h0)) * std::cos(k * ((o[1] + half) * h0));
  }
  ap.applyLaplacian(uex, b);
  double bn = 0;
  for (double v : b) bn += v * v;
  bn = std::sqrt(bn);
  const double tol = 1e-8;

  DeviceMultigrid<3, 21> mg;
  mg.build(t, h0);
  View<double> db("b", (std::size_t)n);
  {
    auto m = Kokkos::create_mirror_view(db);
    for (Index i = 0; i < n; ++i) m(i) = b[(std::size_t)i];
    Kokkos::deep_copy(db, m);
  }
  // Device-side residual norm (no host round-trip — fair to the V-cycle loop).
  View<double> dres("res", (std::size_t)n);
  auto resDev = [&](View<double> x) {
    deviceResidualFv(mg.op(0), View<const double>(x), View<const double>(db), dres);
    return std::sqrt(dotPlain(View<const double>(dres), View<const double>(dres), n));
  };

  // plain V-cycling
  Kokkos::deep_copy(mg.x(0), 0.0);
  Kokkos::deep_copy(mg.b(0), db);
  Kokkos::fence();
  auto v0 = Clock::now();
  int nv = 0;
  for (; nv < 500; ++nv) {
    mg.vcycle(2, 2, 60, 0.8);
    if (resDev(mg.x(0)) <= tol * bn) break;
  }
  Kokkos::fence();
  double vMs = ms(v0, Clock::now());

  // MG-PCG
  DevicePCG<3, 21> pcg;
  pcg.setVcycle(2, 2, 60, 0.8);
  View<double> dx("x", (std::size_t)n);
  Kokkos::fence();
  auto p0 = Clock::now();
  auto R = pcg.solve(mg, dx, View<const double>(db), 200, tol);
  Kokkos::fence();
  double pMs = ms(p0, Clock::now());

  std::printf("pois  L=%u  cells=%ld :  V-cycle %3d cyc %7.2f ms  |  MG-PCG %3d it %7.2f ms  "
              "(%.1fx fewer cycles, %.2fx time)\n",
              L, (long)n, nv + 1, vMs, R.iters, pMs, (double)(nv + 1) / R.iters, vMs / pMs);
}
}  // namespace

int main(int argc, char** argv) {
  // args: [Lmax_device=6] [steps=10] [Lmax_host=5]
  unsigned Ldev = argc > 1 ? (unsigned)std::atoi(argv[1]) : 6;
  int steps = argc > 2 ? std::atoi(argv[2]) : 10;
  unsigned Lhost = argc > 3 ? (unsigned)std::atoi(argv[3]) : 5;
  Kokkos::initialize(argc, argv);
  {
    std::printf("# backend: %s\n", Kokkos::DefaultExecutionSpace::name());
    std::printf("# --- Poisson: V-cycle vs MG-PCG to rel 1e-8 ---\n");
    for (unsigned L = 4; L <= Ldev; ++L) benchPoisson(L);
    std::printf("# --- Flow: host (serial) vs device, where host is affordable ---\n");
    for (unsigned L = 4; L <= Lhost; ++L) benchFlowCompare(L, steps);
    std::printf("# --- Flow: device-only throughput scaling ---\n");
    for (unsigned L = 4; L <= Ldev; ++L) benchFlowDevice(L, steps);
  }
  Kokkos::finalize();
  return 0;
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping AMR flow benchmark\n");
  return 0;
}
#endif

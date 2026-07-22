// Device (Kokkos) GraphAMG apply vs the host oracle.
//
// Same model problem as test_graph_amg (3-component vector Laplacian on a periodic grid, the
// shape of the voro mesh-optimiser Hessian). The device V-cycle must reproduce the host oracle's
// apply to ~machine precision: SpMV row sums, transfers (transpose-gather restriction in host
// scatter order) and vector updates are bit-compatible; only the coarsest-level CG dot products
// reorder (parallel_reduce), so the tolerance is 1e-11 relative rather than 0.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/core/solver/graph_amg.hpp"
#include "peclet/core/solver/graph_amg_device.hpp"

using peclet::core::Index;
using peclet::core::View;
using peclet::core::solver::AmgParams;
using peclet::core::solver::GraphAMG;
using peclet::core::solver::GraphAMGDevice;
using peclet::core::solver::HostCsrOp;

// 3-component vector Laplacian on an m^3 periodic grid (DOF = 3*node + d), diagonal shift to make
// it definite.
static HostCsrOp vectorLaplacian3D(int m, double shift) {
  const Index nn = (Index)m * m * m;
  HostCsrOp A;
  A.n = 3 * nn;
  A.diag.assign((std::size_t)A.n, 6.0 + shift);
  A.start.assign((std::size_t)A.n + 1, 0);
  auto node = [&](int x, int y, int z) {
    auto w = [&](int a) { return (a % m + m) % m; };
    return (Index)w(x) + (Index)w(y) * m + (Index)w(z) * m * m;
  };
  std::vector<std::vector<Index>> nbrs((std::size_t)nn);
  for (int z = 0; z < m; ++z)
    for (int y = 0; y < m; ++y)
      for (int x = 0; x < m; ++x) {
        auto& v = nbrs[(std::size_t)node(x, y, z)];
        v = {node(x - 1, y, z), node(x + 1, y, z), node(x, y - 1, z),
             node(x, y + 1, z), node(x, y, z - 1), node(x, y, z + 1)};
      }
  for (Index i = 0; i < nn; ++i)
    for (int d = 0; d < 3; ++d)
      A.start[(std::size_t)(3 * i + d) + 1] = (Index)nbrs[(std::size_t)i].size();
  for (std::size_t r = 0; r < (std::size_t)A.n; ++r)
    A.start[r + 1] += A.start[r];
  A.nbr.resize((std::size_t)A.start[(std::size_t)A.n]);
  A.coef.assign(A.nbr.size(), -1.0);
  for (Index i = 0; i < nn; ++i)
    for (int d = 0; d < 3; ++d) {
      Index at = A.start[(std::size_t)(3 * i + d)];
      for (Index nb : nbrs[(std::size_t)i])
        A.nbr[(std::size_t)at++] = 3 * nb + d;
    }
  return A;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    const int m = 12;
    HostCsrOp A = vectorLaplacian3D(m, 0.05);
    AmgParams prm;
    prm.ndofPerNode = 3;

    GraphAMG host;
    host.build(A, prm);
    GraphAMGDevice dev;
    dev.build(A, prm);
    if (dev.numLevels() != host.numLevels()) {
      std::printf("FAILED: level count mismatch host=%d dev=%d\n", host.numLevels(),
                  dev.numLevels());
      fail = 1;
    }

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const std::size_t n = (std::size_t)A.n;
    for (int trial = 0; trial < 3 && !fail; ++trial) {
      std::vector<double> r(n), zh(n);
      for (auto& v : r)
        v = U(rng);
      host.apply(r, zh);

      View<double> dr(Kokkos::view_alloc(std::string("r"), Kokkos::WithoutInitializing), n),
          dz("z", n);
      auto hr = Kokkos::create_mirror_view(dr);
      for (std::size_t i = 0; i < n; ++i)
        hr(i) = r[i];
      Kokkos::deep_copy(dr, hr);
      dev.apply(dr, dz);
      auto hz = Kokkos::create_mirror_view(dz);
      Kokkos::deep_copy(hz, dz);

      double md = 0, mz = 0;
      for (std::size_t i = 0; i < n; ++i) {
        md = std::max(md, std::fabs(hz(i) - zh[i]));
        mz = std::max(mz, std::fabs(zh[i]));
      }
      const double rel = md / (mz + 1e-300);
      std::printf("  trial %d: n=%zu levels=%d  max|z_dev-z_host|/max|z_host| = %.3e\n", trial, n,
                  host.numLevels(), rel);
      if (!(rel < 1e-11))
        fail = 1;
    }
    std::printf(fail == 0 ? "OK: device GraphAMG apply matches the host oracle\n" : "FAILED\n");
  }
  Kokkos::finalize();
  return fail;
}

// Mesh-independence (O(N)) proof for the smoothed-aggregation graph-AMG (peclet::core::solver).
//
// The voro mesh optimiser's Gauss–Newton Hessian is a vector "graph Laplacian" (translation
// near-nullspace, 3 DOFs/node). We stand in a MODEL of it here — a 3-component vector Laplacian on
// a periodic 3D grid, assembled straight into the mesh-agnostic HostCsrOp — so the test needs no
// Kokkos and no tessellator, yet exercises exactly the structure the AMG must coarsen: nodal
// aggregation over an s=3 block operator with the uniform per-component nullspace.
//
// The contract (what the later device + MPI ports must also satisfy): as N grows 8³ → 40³ the
// AMG-preconditioned CG iteration count stays ~FLAT (mesh independent ⇒ O(N) overall), while the
// Jacobi-preconditioned CG count GROWS like N^(1/3). That growth-vs-flat gap is the whole point.
#include <cmath>
#include <cstdio>
#include <vector>

#include "peclet/core/solver/graph_amg.hpp"
#include "test_util.hpp"

using peclet::core::Index;
using peclet::core::solver::AmgParams;
using peclet::core::solver::amgPcg;
using peclet::core::solver::GraphAMG;
using peclet::core::solver::HostCsrOp;

namespace {

// A 3-component vector Laplacian on an m³ periodic grid: each component d ∈ {0,1,2} is an
// independent 7-point Laplacian, interleaved as DOF = 3·node + d (the block layout the AMG
// aggregates nodally). A small reaction term (shift·I) keeps it SPD and non-singular so CG has a
// clean convergence target; the coarsening behaviour is that of the pure Laplacian.
HostCsrOp vectorLaplacian3D(int m, double shift) {
  const Index nNode = static_cast<Index>(m) * m * m;
  const int s = 3;
  HostCsrOp A;
  A.n = nNode * s;
  A.diag.assign(static_cast<std::size_t>(A.n), 0.0);
  A.start.assign(static_cast<std::size_t>(A.n) + 1, 0);
  auto nodeAt = [&](int x, int y, int z) {
    const int xw = (x + m) % m, yw = (y + m) % m, zw = (z + m) % m;
    return static_cast<Index>((static_cast<Index>(zw) * m + yw) * m + xw);
  };
  const int off[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int z = 0; z < m; ++z)
    for (int y = 0; y < m; ++y)
      for (int x = 0; x < m; ++x) {
        const Index I = nodeAt(x, y, z);
        for (int d = 0; d < s; ++d) {
          const Index i = I * s + d;
          A.diag[static_cast<std::size_t>(i)] = 6.0 + shift;
          for (int f = 0; f < 6; ++f) {
            const Index J = nodeAt(x + off[f][0], y + off[f][1], z + off[f][2]);
            A.nbr.push_back(J * s + d);
            A.coef.push_back(-1.0);
          }
          A.start[static_cast<std::size_t>(i) + 1] = static_cast<Index>(A.nbr.size());
        }
      }
  return A;
}

// Jacobi-preconditioned CG on the same operator, for the baseline (growing) iteration count.
int jacobiPcgIters(const HostCsrOp& A, const std::vector<double>& b, double tol) {
  const std::size_t n = static_cast<std::size_t>(A.n);
  std::vector<double> x(n, 0.0), r(n), z(n), p(n), Ap(n), invD(n);
  for (std::size_t i = 0; i < n; ++i)
    invD[i] = 1.0 / A.diag[i];
  auto dot = [&](const std::vector<double>& u, const std::vector<double>& v) {
    double sdp = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      sdp += u[i] * v[i];
    return sdp;
  };
  A.apply(x, Ap);
  for (std::size_t i = 0; i < n; ++i)
    r[i] = b[i] - Ap[i];
  const double r0 = std::sqrt(dot(r, r));
  for (std::size_t i = 0; i < n; ++i)
    z[i] = invD[i] * r[i];
  p = z;
  double rz = dot(r, z);
  int it = 0;
  for (; it < 5000; ++it) {
    A.apply(p, Ap);
    const double pAp = dot(p, Ap);
    const double alpha = rz / pAp;
    for (std::size_t i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    if (std::sqrt(dot(r, r)) <= tol * r0) {
      ++it;
      break;
    }
    for (std::size_t i = 0; i < n; ++i)
      z[i] = invD[i] * r[i];
    const double rzn = dot(r, z);
    const double beta = rzn / rz;
    for (std::size_t i = 0; i < n; ++i)
      p[i] = z[i] + beta * p[i];
    rz = rzn;
  }
  return it;
}

}  // namespace

int main() {
  const double tol = 1e-8;
  const int sizes[] = {8, 16, 24, 32, 40};

  int amgIters[5] = {0}, jacIters[5] = {0};
  int idx = 0;
  for (int m : sizes) {
    HostCsrOp A = vectorLaplacian3D(m, /*shift=*/0.05);
    const std::size_t n = static_cast<std::size_t>(A.n);

    // Deterministic RHS orthogonal-ish to the near-nullspace (a smooth+rough mix).
    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i)
      b[i] = std::sin(0.7 * static_cast<double>(i)) + 0.3 * std::cos(0.03 * static_cast<double>(i));

    AmgParams prm;
    prm.ndofPerNode = 3;
    prm.chebDegree = 2;
    prm.pre = prm.post = 1;
    GraphAMG M;
    M.build(A, prm);

    std::vector<double> x(n, 0.0);
    auto R = amgPcg(A, x, b, M, /*maxIters=*/500, tol);

    // Correctness: AMG-PCG actually solved the system to tolerance.
    PECLET_CORE_CHECK(R.res <= tol * R.res0 * 1.5);
    // Residual really is small in absolute terms too (guard against a bogus res0).
    std::vector<double> Ax(n), rr(n);
    A.apply(x, Ax);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      rr[i] = b[i] - Ax[i];
      num += rr[i] * rr[i];
      den += b[i] * b[i];
    }
    PECLET_CORE_CHECK(std::sqrt(num / den) <= 1e-6);

    // Healthy hierarchy: several levels, bounded operator complexity.
    PECLET_CORE_CHECK(M.numLevels() >= 3);
    PECLET_CORE_CHECK(M.operatorComplexity() < 3.0);

    amgIters[idx] = R.iters;
    jacIters[idx] = jacobiPcgIters(A, b, tol);
    std::printf("  m=%2d  N=%7lld  AMG-PCG=%3d iters (%d lvls, opC=%.2f)   Jacobi-PCG=%4d iters\n", m,
                static_cast<long long>(A.n), R.iters, M.numLevels(), M.operatorComplexity(),
                jacIters[idx]);
    ++idx;
  }

  // --- the mesh-independence contract ---
  // AMG iteration count is ~flat: the largest grid needs no more than +6 iterations over the
  // smallest (a generous band; in practice it is within a couple).
  PECLET_CORE_CHECK(amgIters[4] <= amgIters[0] + 6);
  // And it is dramatically better than Jacobi at the largest size (the O(N) vs O(N^4/3) win).
  PECLET_CORE_CHECK(amgIters[4] * 3 < jacIters[4]);
  // Jacobi genuinely grows (guards the baseline — if it didn't grow, the model would be too easy).
  PECLET_CORE_CHECK(jacIters[4] > jacIters[0]);

  PECLET_CORE_RETURN_TEST_RESULT();
}

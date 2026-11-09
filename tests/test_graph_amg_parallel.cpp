// Parallel (Aggregation::Mis2) GraphAMG setup — design peclet-papers work/M7/DESIGN_NOTE.md §14.5
// C1/C2, results in docs/graph_amg_parallel_setup.md.
//
// Default mode (the ctest):
//   * steps 5–7 are checked BIT FOR BIT against a host reference that recomputes, from the mis2
//     level's own aggregates and λ(D_F⁻¹A_F), the strong graph, the filtered smoothed P, R = Pᵀ and
//     the Galerkin A_c with the host oracle's own code and accumulation order;
//   * aggregates are a valid partition; gate (b): PCG iterations ≤ 1.2× the host oracle's and
//     mesh-independent (N ×8 adds ≤ 2) on the four test operators;
//   * the dense coarsest level (singular Neumann operator; A·A_c⁺·b = b on the range) and the
//     forced CG fallback (denseFallback()).
// `--hash m`  : one line per operator with an FNV-1a hash of the whole hierarchy (gate (d): run at
//               1/2/4/8 threads and on the Serial backend, compare).
// `--bench m` : setup and V-cycle timings (gate (c)), best of 3.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "peclet/core/solver/graph_amg_device.hpp"

using peclet::core::Index;
using peclet::core::View;
using namespace peclet::core::solver;

namespace {

// ---------------------------------------------------------------- test operators
// Dirichlet 7-point Poisson on m³ (scalar, s = 1).
HostCsrOp poisson7(int m, bool neumann = false) {
  HostCsrOp A;
  A.n = (Index)m * m * m;
  A.diag.assign((std::size_t)A.n, 0.0);
  A.start.assign((std::size_t)A.n + 1, 0);
  const int off[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int z = 0; z < m; ++z)
    for (int y = 0; y < m; ++y)
      for (int x = 0; x < m; ++x) {
        const Index i = ((Index)z * m + y) * m + x;
        double d = 0.0;
        for (auto& o : off) {
          const int a = x + o[0], b = y + o[1], c = z + o[2];
          if (a < 0 || b < 0 || c < 0 || a >= m || b >= m || c >= m) {
            if (!neumann)
              d += 1.0;
            continue;
          }
          A.nbr.push_back(((Index)c * m + b) * m + a);
          A.coef.push_back(-1.0);
          d += 1.0;
        }
        A.diag[(std::size_t)i] = d;
        A.start[(std::size_t)i + 1] = (Index)A.nbr.size();
      }
  return A;
}

// 3-component vector Laplacian on a periodic m³ grid (3-DOF blocks), as test_graph_amg.cpp.
HostCsrOp vectorLaplacian3D(int m, double shift) {
  const Index nNode = (Index)m * m * m;
  HostCsrOp A;
  A.n = nNode * 3;
  A.diag.assign((std::size_t)A.n, 0.0);
  A.start.assign((std::size_t)A.n + 1, 0);
  auto nodeAt = [&](int x, int y, int z) {
    return (Index)((((z + m) % m) * m + (y + m) % m) * m + (x + m) % m);
  };
  const int off[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int z = 0; z < m; ++z)
    for (int y = 0; y < m; ++y)
      for (int x = 0; x < m; ++x)
        for (int d = 0; d < 3; ++d) {
          const Index i = nodeAt(x, y, z) * 3 + d;
          A.diag[(std::size_t)i] = 6.0 + shift;
          for (auto& o : off) {
            A.nbr.push_back(nodeAt(x + o[0], y + o[1], z + o[2]) * 3 + d);
            A.coef.push_back(-1.0);
          }
          A.start[(std::size_t)i + 1] = (Index)A.nbr.size();
        }
  return A;
}

// Irregular graph: a jittered m³ lattice (Dirichlet box), each node coupled to its 14 bcc-type
// lattice neighbours (6 faces + 8 corners: a Voronoi-like valence) with weight 1/|x_I − x_J|²; a
// missing neighbour outside the box is recorded with its nominal weight and direction (Dirichlet).
struct IrregularGraph {
  int m;
  std::vector<double> pos;  // 3 per node
  std::vector<Index> start, nbr;
  std::vector<double> w, dir;  // weight, unit direction (3 per edge)
  std::vector<double> bnd;     // Dirichlet diagonal per node
  std::vector<double> bndVec;  // Σ nominal w·n over the missing neighbours (3 per node)
};
IrregularGraph irregular(int m, unsigned seed) {
  IrregularGraph G;
  G.m = m;
  const Index n = (Index)m * m * m;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-0.25, 0.25);
  G.pos.resize((std::size_t)n * 3);
  for (Index I = 0; I < n; ++I) {
    const int x = (int)(I % m), y = (int)((I / m) % m), z = (int)(I / ((Index)m * m));
    G.pos[(std::size_t)I * 3 + 0] = x + U(rng);
    G.pos[(std::size_t)I * 3 + 1] = y + U(rng);
    G.pos[(std::size_t)I * 3 + 2] = z + U(rng);
  }
  G.start.assign((std::size_t)n + 1, 0);
  G.bnd.assign((std::size_t)n, 0.0);
  G.bndVec.assign((std::size_t)n * 3, 0.0);
  for (Index I = 0; I < n; ++I) {
    const int x = (int)(I % m), y = (int)((I / m) % m), z = (int)(I / ((Index)m * m));
    for (int c = -1; c <= 1; ++c)
      for (int b = -1; b <= 1; ++b)
        for (int a = -1; a <= 1; ++a) {
          const int nz = (a != 0) + (b != 0) + (c != 0);
          if (nz != 1 && nz != 3)
            continue;  // faces and corners only
          const int X = x + a, Y = y + b, Z = z + c;
          if (X < 0 || Y < 0 || Z < 0 || X >= m || Y >= m || Z >= m) {
            const double r2 = a * a + b * b + c * c, r = std::sqrt(r2);
            G.bnd[(std::size_t)I] += 1.0 / r2;
            G.bndVec[(std::size_t)I * 3 + 0] += a / (r * r2);
            G.bndVec[(std::size_t)I * 3 + 1] += b / (r * r2);
            G.bndVec[(std::size_t)I * 3 + 2] += c / (r * r2);
            continue;
          }
          const Index J = ((Index)Z * m + Y) * m + X;
          double d[3], r2 = 0;
          for (int k = 0; k < 3; ++k) {
            d[k] = G.pos[(std::size_t)J * 3 + k] - G.pos[(std::size_t)I * 3 + k];
            r2 += d[k] * d[k];
          }
          const double r = std::sqrt(r2);
          G.nbr.push_back(J);
          G.w.push_back(1.0 / r2);
          for (int k = 0; k < 3; ++k)
            G.dir.push_back(d[k] / r);
        }
    G.start[(std::size_t)I + 1] = (Index)G.nbr.size();
  }
  return G;
}

// "Voronoi H" model: 3-DOF block operator K_IJ = −w (I + n nᵀ) on the irregular graph (the
// translation near-nullspace per component, as the Gauss–Newton Hessian), Dirichlet box.
HostCsrOp voronoiH(int m) {
  const IrregularGraph G = irregular(m, 11);
  const Index nN = (Index)m * m * m;
  HostCsrOp A;
  A.n = nN * 3;
  A.diag.assign((std::size_t)A.n, 0.0);
  A.start.assign((std::size_t)A.n + 1, 0);
  for (Index I = 0; I < nN; ++I) {
    double blkD[3][3] = {};
    for (int d = 0; d < 3; ++d)
      blkD[d][d] = 2.0 * G.bnd[(std::size_t)I];
    for (Index e = G.start[(std::size_t)I]; e < G.start[(std::size_t)I + 1]; ++e)
      for (int d = 0; d < 3; ++d)
        for (int q = 0; q < 3; ++q)
          blkD[d][q] += G.w[(std::size_t)e] *
                        ((d == q) + G.dir[(std::size_t)e * 3 + d] * G.dir[(std::size_t)e * 3 + q]);
    for (int d = 0; d < 3; ++d) {
      const Index i = I * 3 + d;
      A.diag[(std::size_t)i] = blkD[d][d];
      for (int q = 0; q < 3; ++q)
        if (q != d) {
          A.nbr.push_back(I * 3 + q);
          A.coef.push_back(blkD[d][q]);
        }
      for (Index e = G.start[(std::size_t)I]; e < G.start[(std::size_t)I + 1]; ++e) {
        const Index J = G.nbr[(std::size_t)e];
        for (int q = 0; q < 3; ++q) {
          A.nbr.push_back(J * 3 + q);
          A.coef.push_back(-G.w[(std::size_t)e] * ((d == q) + G.dir[(std::size_t)e * 3 + d] *
                                                                  G.dir[(std::size_t)e * 3 + q]));
        }
      }
      A.start[(std::size_t)i + 1] = (Index)A.nbr.size();
    }
  }
  return A;
}

// "S₀-type" 2-ring operator: S₀ = C M⁻¹ Cᵀ with M = I and Cᵀ = G the discrete gradient on the
// irregular graph, g_I = Σ_J w_IJ n_IJ (p_J − p_I), Dirichlet (p = 0) outside the box: a scalar
// divergence-of-gradient ("wide Laplacian") whose rows reach two graph rings (≈ 55 nnz).
HostCsrOp twoRing(int m) {
  const IrregularGraph G = irregular(m, 23);
  const Index n = (Index)m * m * m;
  std::vector<std::map<Index, double>> S((std::size_t)n);
  std::vector<Index> cols;
  std::vector<double> gc;  // 3 per column of row block I
  for (Index I = 0; I < n; ++I) {
    cols.clear();
    gc.clear();
    double self[3];
    for (int d = 0; d < 3; ++d)
      self[d] = -G.bndVec[(std::size_t)I * 3 + d];
    for (Index e = G.start[(std::size_t)I]; e < G.start[(std::size_t)I + 1]; ++e) {
      cols.push_back(G.nbr[(std::size_t)e]);
      for (int d = 0; d < 3; ++d) {
        const double v = G.w[(std::size_t)e] * G.dir[(std::size_t)e * 3 + d];
        gc.push_back(v);
        self[d] -= v;
      }
    }
    cols.push_back(I);
    for (int d = 0; d < 3; ++d)
      gc.push_back(self[d]);
    for (std::size_t a = 0; a < cols.size(); ++a)
      for (std::size_t b = 0; b < cols.size(); ++b) {
        double v = 0;
        for (int d = 0; d < 3; ++d)
          v += gc[a * 3 + d] * gc[b * 3 + d];
        S[(std::size_t)cols[a]][cols[b]] += v;
      }
  }
  HostCsrOp A;
  A.n = n;
  A.diag.assign((std::size_t)n, 0.0);
  A.start.assign((std::size_t)n + 1, 0);
  for (Index I = 0; I < n; ++I) {
    for (auto& [J, v] : S[(std::size_t)I])
      if (J == I)
        A.diag[(std::size_t)I] = v;
      else {
        A.nbr.push_back(J);
        A.coef.push_back(v);
      }
    A.start[(std::size_t)I + 1] = (Index)A.nbr.size();
  }
  return A;
}

struct Case {
  const char* name;
  int s;
  HostCsrOp (*make)(int);
};
HostCsrOp mkPoisson(int m) {
  return poisson7(m);
}
HostCsrOp mkVec(int m) {
  return vectorLaplacian3D(m, 0.05);
}
const Case kCases[] = {{"poisson7", 1, mkPoisson},
                       {"voronoiH", 3, voronoiH},
                       {"s0_2ring", 1, twoRing},
                       {"vec3blk", 3, mkVec}};

// ---------------------------------------------------------------- helpers
template <class T>
std::vector<T> toHost(const View<T>& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  return std::vector<T>(h.data(), h.data() + h.extent(0));
}
View<double> toDevV(const std::vector<double>& v) {
  return GraphAMGDevice::toDev(v, "tv");
}

double dotD(const View<double>& a, const View<double>& b) {
  double s = 0;
  Kokkos::parallel_reduce(
      "t_dot", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, a.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, double& acc) { acc += a(i) * b(i); }, s);
  return s;
}

// Device PCG (rel. tol on ‖r‖₂) with one V-cycle per iteration; returns iterations.
int pcg(const HostCsrOp& Ah, const GraphAMGDevice& M, const std::vector<double>& bh, double tol,
        int maxIt = 300) {
  GraphAMGDevice::DLevel A;
  A.n = Ah.n;
  A.start = GraphAMGDevice::toDev(Ah.start, "s");
  A.nbr = GraphAMGDevice::toDev(Ah.nbr, "n");
  A.coef = GraphAMGDevice::toDev(Ah.coef, "c");
  A.diag = GraphAMGDevice::toDev(Ah.diag, "d");
  const std::size_t n = (std::size_t)Ah.n;
  View<double> x("x", n), r = toDevV(bh), z("z", n), p("p", n), Ap("Ap", n);
  const double r0 = std::sqrt(dotD(r, r));
  M.apply(r, z);
  Kokkos::deep_copy(p, z);
  double rz = dotD(r, z);
  int it = 0;
  for (; it < maxIt; ++it) {
    GraphAMGDevice::spmv(A, p, Ap);
    const double pAp = dotD(p, Ap);
    if (!(pAp > 0))
      break;
    const double al = rz / pAp;
    Kokkos::parallel_for(
        "t_up", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, n), KOKKOS_LAMBDA(std::size_t i) {
          x(i) += al * p(i);
          r(i) -= al * Ap(i);
        });
    if (std::sqrt(dotD(r, r)) <= tol * r0) {
      ++it;
      break;
    }
    M.apply(r, z);
    const double rzn = dotD(r, z), be = rzn / rz;
    rz = rzn;
    Kokkos::parallel_for(
        "t_p", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, n),
        KOKKOS_LAMBDA(std::size_t i) { p(i) = z(i) + be * p(i); });
  }
  return it;
}

std::vector<double> rhs(Index n, unsigned seed, bool meanFree = false) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-1, 1);
  std::vector<double> b((std::size_t)n);
  double mean = 0;
  for (auto& v : b)
    mean += (v = U(rng));
  if (meanFree)
    for (auto& v : b)
      v -= mean / (double)n;
  return b;
}

struct Fnv {
  std::uint64_t h = 1469598103934665603ULL;
  void bytes(const void* p, std::size_t n) {
    auto c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i)
      h = (h ^ c[i]) * 1099511628211ULL;
  }
  template <class T>
  void view(const View<T>& v) {
    if (v.extent(0) == 0)
      return;
    auto h2 = toHost(v);
    bytes(h2.data(), h2.size() * sizeof(T));
  }
  template <class T>
  void val(T v) {
    bytes(&v, sizeof(T));
  }
};

std::uint64_t hierarchyHash(const GraphAMGDevice& M) {
  Fnv f;
  for (const auto& L : M.levels()) {
    f.val(L.n);
    f.val(L.nc);
    f.val(L.lmax);
    f.val(L.lmaxF);
    f.view(L.start);
    f.view(L.nbr);
    f.view(L.coef);
    f.view(L.diag);
    f.view(L.invDiag);
    f.view(L.agg);
    f.view(L.Pstart);
    f.view(L.Pcol);
    f.view(L.Pval);
    f.view(L.Rstart);
    f.view(L.Rcol);
    f.view(L.Rval);
    f.view(L.denseV);
    f.view(L.denseLamInv);
  }
  return f.h;
}

// ---------------------------------------------------------------- host reference for steps 5–7
// Given a mis2 level (operator, aggregates, λ_F), recompute strength, filtered smoothed P, R = Pᵀ
// and A_c with the host oracle's code and order; compare bit for bit with the parallel result.
int checkLevelBitwise(const GraphAMGDevice::DLevel& L, const GraphAMGDevice::DLevel& C,
                      const AmgParams& prm, const char* tag) {
  const int s = prm.ndofPerNode;
  const auto start = toHost(L.start), nbr = toHost(L.nbr), agg = toHost(L.agg);
  const auto coef = toHost(L.coef), diag = toHost(L.diag);
  const Index n = L.n, nN = n / s;
  std::vector<std::map<Index, double>> blk((std::size_t)nN);
  std::vector<double> ddiag((std::size_t)nN, 0.0);
  for (Index i = 0; i < n; ++i) {
    const Index I = i / s;
    ddiag[(std::size_t)I] += diag[(std::size_t)i] * diag[(std::size_t)i];
    for (Index k = start[(std::size_t)i]; k < start[(std::size_t)i + 1]; ++k) {
      const Index J = nbr[(std::size_t)k] / s;
      const double c = coef[(std::size_t)k];
      if (J == I)
        ddiag[(std::size_t)I] += c * c;
      else
        blk[(std::size_t)I][J] += c * c;
    }
  }
  std::vector<std::vector<Index>> strong((std::size_t)nN);
  const double th2 = prm.theta * prm.theta;
  for (Index I = 0; I < nN; ++I)
    for (auto& [J, f2] : blk[(std::size_t)I])
      if (f2 >= th2 * std::sqrt(ddiag[(std::size_t)I] * ddiag[(std::size_t)J]))
        strong[(std::size_t)I].push_back(J);
  auto isStrong = [&](Index I, Index J) {
    const auto& v = strong[(std::size_t)I];
    return std::binary_search(v.begin(), v.end(), J);
  };
  Index nAgg = 0;
  for (Index a : agg) {
    if (a < 0) {
      std::printf("FAILED %s: unassigned node\n", tag);
      return 1;
    }
    nAgg = std::max(nAgg, a + 1);
  }
  std::vector<Index> cnt((std::size_t)nAgg, 0);
  for (Index a : agg)
    ++cnt[(std::size_t)a];
  for (Index a = 0; a < nAgg; ++a)
    if (cnt[(std::size_t)a] == 0) {
      std::printf("FAILED %s: empty aggregate %lld\n", tag, (long long)a);
      return 1;
    }
  if (nAgg * s != L.nc) {
    std::printf("FAILED %s: nc mismatch\n", tag);
    return 1;
  }
  std::vector<double> isc((std::size_t)nAgg);
  for (Index a = 0; a < nAgg; ++a)
    isc[(std::size_t)a] = 1.0 / std::sqrt((double)cnt[(std::size_t)a]);
  const double sc = prm.smoothOmega / L.lmaxF;
  auto p0col = [&](Index i) { return agg[(std::size_t)(i / s)] * s + i % s; };
  auto p0val = [&](Index i) { return isc[(std::size_t)agg[(std::size_t)(i / s)]]; };
  std::vector<Index> Pstart((std::size_t)n + 1, 0), Pcol;
  std::vector<double> Pval;
  std::map<Index, double> row;
  for (Index i = 0; i < n; ++i) {
    const Index I = i / s;
    double dF = diag[(std::size_t)i];
    for (Index k = start[(std::size_t)i]; k < start[(std::size_t)i + 1]; ++k) {
      const Index J = nbr[(std::size_t)k] / s;
      if (J != I && !isStrong(I, J))
        dF += coef[(std::size_t)k];
    }
    row.clear();
    row[p0col(i)] += p0val(i);
    const double f = (std::fabs(dF) > 1e-300) ? (-sc / dF) : 0.0;
    row[p0col(i)] += f * dF * p0val(i);
    for (Index k = start[(std::size_t)i]; k < start[(std::size_t)i + 1]; ++k) {
      const Index j = nbr[(std::size_t)k];
      const Index J = j / s;
      if (J != I && !isStrong(I, J))
        continue;
      row[p0col(j)] += f * coef[(std::size_t)k] * p0val(j);
    }
    for (auto& [c, v] : row)
      if (v != 0.0) {
        Pcol.push_back(c);
        Pval.push_back(v);
      }
    Pstart[(std::size_t)i + 1] = (Index)Pcol.size();
  }
  if (Pstart != toHost(L.Pstart) || Pcol != toHost(L.Pcol) || Pval != toHost(L.Pval)) {
    std::printf("FAILED %s: smoothed P differs from the host-order reference\n", tag);
    return 1;
  }
  // R = Pᵀ, entries per coarse row in increasing fine index
  const Index nc = L.nc;
  std::vector<Index> Rstart((std::size_t)nc + 1, 0), Rcol(Pcol.size());
  std::vector<double> Rval(Pval.size());
  for (Index c : Pcol)
    ++Rstart[(std::size_t)c + 1];
  for (Index c = 0; c < nc; ++c)
    Rstart[(std::size_t)c + 1] += Rstart[(std::size_t)c];
  std::vector<Index> cur(Rstart.begin(), Rstart.end() - 1);
  for (Index i = 0; i < n; ++i)
    for (Index k = Pstart[(std::size_t)i]; k < Pstart[(std::size_t)i + 1]; ++k) {
      const Index at = cur[(std::size_t)Pcol[(std::size_t)k]]++;
      Rcol[(std::size_t)at] = i;
      Rval[(std::size_t)at] = Pval[(std::size_t)k];
    }
  if (Rstart != toHost(L.Rstart) || Rcol != toHost(L.Rcol) || Rval != toHost(L.Rval)) {
    std::printf("FAILED %s: R differs from Pᵀ\n", tag);
    return 1;
  }
  // A_c = PᵀAP, the host oracle's loop
  std::vector<std::map<Index, double>> Ac((std::size_t)nc);
  std::map<Index, double> apRow;
  for (Index i = 0; i < n; ++i) {
    apRow.clear();
    for (Index k = Pstart[(std::size_t)i]; k < Pstart[(std::size_t)i + 1]; ++k)
      apRow[Pcol[(std::size_t)k]] += diag[(std::size_t)i] * Pval[(std::size_t)k];
    for (Index k = start[(std::size_t)i]; k < start[(std::size_t)i + 1]; ++k) {
      const Index j = nbr[(std::size_t)k];
      for (Index kk = Pstart[(std::size_t)j]; kk < Pstart[(std::size_t)j + 1]; ++kk)
        apRow[Pcol[(std::size_t)kk]] += coef[(std::size_t)k] * Pval[(std::size_t)kk];
    }
    for (Index k = Pstart[(std::size_t)i]; k < Pstart[(std::size_t)i + 1]; ++k) {
      auto& r1 = Ac[(std::size_t)Pcol[(std::size_t)k]];
      for (auto& [c2, v] : apRow)
        r1[c2] += Pval[(std::size_t)k] * v;
    }
  }
  std::vector<double> cd((std::size_t)nc, 0.0), cv;
  std::vector<Index> cs((std::size_t)nc + 1, 0), cn;
  for (Index c = 0; c < nc; ++c) {
    for (auto& [c2, v] : Ac[(std::size_t)c])
      if (c2 == c)
        cd[(std::size_t)c] = v;
      else if (v != 0.0) {
        cn.push_back(c2);
        cv.push_back(v);
      }
    cs[(std::size_t)c + 1] = (Index)cn.size();
  }
  if (C.n != nc || cd != toHost(C.diag) || cs != toHost(C.start) || cn != toHost(C.nbr) ||
      cv != toHost(C.coef)) {
    std::printf("FAILED %s: A_c differs from the host-order Galerkin product\n", tag);
    return 1;
  }
  return 0;
}

AmgParams paramsFor(const Case& c, Aggregation ag, CoarseSolve cs = CoarseSolve::Cg) {
  AmgParams p;
  p.ndofPerNode = c.s;
  p.aggregation = ag;
  p.coarseSolve = cs;
  return p;
}

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int runHash(int m) {
  for (const Case& c : kCases) {
    const HostCsrOp A = c.make(m);
    for (CoarseSolve cs : {CoarseSolve::Cg, CoarseSolve::Dense}) {
      GraphAMGDevice M;
      M.build(A, paramsFor(c, Aggregation::Mis2, cs));
      std::printf("hash %s m=%d coarse=%s levels=%d %016llx\n", c.name, m,
                  cs == CoarseSolve::Dense ? "dense" : "cg", M.numLevels(),
                  (unsigned long long)hierarchyHash(M));
    }
  }
  return 0;
}

int runBench(int m) {
  for (const Case& c : kCases) {
    const HostCsrOp A = c.make(m);
    const auto b = rhs(A.n, 5);
    double tg = 1e30, tp = 1e30, tv = 1e30;
    GraphAMGDevice G, P;
    for (int rep = 0; rep < 3; ++rep) {
      double t0 = now();
      G.build(A, paramsFor(c, Aggregation::Greedy));
      tg = std::min(tg, now() - t0);
      DeviceCsr d;
      d.n = A.n;
      d.diag = GraphAMGDevice::toDev(A.diag, "d");
      d.start = GraphAMGDevice::toDev(A.start, "s");
      d.nbr = GraphAMGDevice::toDev(A.nbr, "n");
      d.coef = GraphAMGDevice::toDev(A.coef, "c");
      Kokkos::fence();
      t0 = now();
      P.build(d, paramsFor(c, Aggregation::Mis2, CoarseSolve::Dense));
      Kokkos::fence();
      tp = std::min(tp, now() - t0);
    }
    View<double> r = toDevV(b), z("z", (std::size_t)A.n);
    for (int rep = 0; rep < 3; ++rep) {
      Kokkos::fence();
      const double t0 = now();
      for (int k = 0; k < 20; ++k)
        P.apply(r, z);
      Kokkos::fence();
      tv = std::min(tv, (now() - t0) / 20);
    }
#ifdef PECLET_CORE_AMGS_PROFILE
    for (double& v : amg_setup::gProf)
      v = 0;
    P.build(A, paramsFor(c, Aggregation::Mis2, CoarseSolve::Dense));
    std::printf(
        "prof %s ms: strength %.1f mis2+agg %.1f filter+lamF %.1f P %.1f R %.1f AP %.1f RAP %.1f\n",
        c.name, 1e3 * amg_setup::gProf[0], 1e3 * amg_setup::gProf[1], 1e3 * amg_setup::gProf[2],
        1e3 * amg_setup::gProf[3], 1e3 * amg_setup::gProf[4], 1e3 * amg_setup::gProf[5],
        1e3 * amg_setup::gProf[6]);
#endif
    const int ig = pcg(A, G, b, 1e-8), ip = pcg(A, P, b, 1e-8);
    std::printf(
        "bench %s m=%d n=%lld thr=%d | greedy setup %.2f ms it %d | mis2+dense setup %.2f ms it "
        "%d levels %d vcycle %.3f ms setup/vcycle %.1f\n",
        c.name, m, (long long)A.n, (int)Kokkos::DefaultExecutionSpace().concurrency(), 1e3 * tg, ig,
        1e3 * tp, ip, P.numLevels(), 1e3 * tv, tp / tv);
  }
  return 0;
}

int runGate() {
  int fail = 0;
  // (1) steps 5–7 bitwise vs the host-order reference, on every level, every operator
  for (const Case& c : kCases) {
    const HostCsrOp A = c.make(10);
    const AmgParams prm = paramsFor(c, Aggregation::Mis2);
    GraphAMGDevice M;
    M.build(A, prm);
    for (int L = 0; L + 1 < M.numLevels(); ++L) {
      const std::string tag = std::string(c.name) + " L" + std::to_string(L);
      fail |= checkLevelBitwise(M.levels()[(std::size_t)L], M.levels()[(std::size_t)L + 1], prm,
                                tag.c_str());
    }
    std::printf("  %s: %d levels, steps 5-7 bitwise vs host-order reference: %s\n", c.name,
                M.numLevels(), fail ? "FAIL" : "ok");
  }
  // (2) gate (b): iterations ≤ 1.2× the host oracle's; N ×8 adds ≤ 2
  for (const Case& c : kCases) {
    int prev = -1;
    for (int m : {12, 24}) {
      const HostCsrOp A = c.make(m);
      const auto b = rhs(A.n, 3);
      GraphAMGDevice G, P, D;
      G.build(A, paramsFor(c, Aggregation::Greedy));
      P.build(A, paramsFor(c, Aggregation::Mis2));
      D.build(A, paramsFor(c, Aggregation::Mis2, CoarseSolve::Dense));
      const int ig = pcg(A, G, b, 1e-8), ip = pcg(A, P, b, 1e-8), id = pcg(A, D, b, 1e-8);
      const bool ok = ip <= 1.2 * ig + 1e-9 && id <= 1.2 * ig + 1e-9 &&
                      (prev < 0 || ip - prev <= 2) && D.coarseSolveUsed() == CoarseSolve::Dense;
      std::printf(
          "  %s m=%d n=%lld: iters greedy %d | mis2 %d (%d lv) | mis2+dense %d (%d lv) %s\n",
          c.name, m, (long long)A.n, ig, ip, P.numLevels(), id, D.numLevels(), ok ? "ok" : "FAIL");
      if (!ok)
        fail = 1;
      prev = ip;
    }
  }
  // (3) dense coarsest on a singular (Neumann) operator: A·A_c⁺·b = b on the range
  {
    const HostCsrOp A = poisson7(5, true);  // 125 DOFs ≤ 200 ⇒ one level, dense
    AmgParams prm;
    prm.aggregation = Aggregation::Mis2;
    prm.coarseSolve = CoarseSolve::Dense;
    GraphAMGDevice M;
    M.build(A, prm);
    const auto b = rhs(A.n, 9, true);
    View<double> r = toDevV(b), z("z", (std::size_t)A.n);
    M.apply(r, z);
    const auto zh = toHost(z);
    std::vector<double> Az((std::size_t)A.n);
    A.apply(zh, Az);
    double e = 0, nb = 0;
    for (std::size_t i = 0; i < Az.size(); ++i) {
      e = std::max(e, std::fabs(Az[i] - b[i]));
      nb = std::max(nb, std::fabs(b[i]));
    }
    const bool ok = M.numLevels() == 1 && M.coarseSolveUsed() == CoarseSolve::Dense &&
                    !M.denseFallback() && e <= 1e-12 * nb;
    std::printf("  dense Neumann n=%lld: max|A A+ b - b|/max|b| = %.2e %s\n", (long long)A.n,
                e / nb, ok ? "ok" : "FAIL");
    if (!ok)
      fail = 1;
    // Neumann PCG with the dense bottom through a multilevel hierarchy
    const HostCsrOp A2 = poisson7(16, true);
    GraphAMGDevice M2;
    M2.build(A2, prm);
    const int it = pcg(A2, M2, rhs(A2.n, 4, true), 1e-8);
    const bool ok2 = M2.coarseSolveUsed() == CoarseSolve::Dense && it > 0 && it < 40;
    std::printf("  dense Neumann PCG n=%lld levels=%d iters=%d %s\n", (long long)A2.n,
                M2.numLevels(), it, ok2 ? "ok" : "FAIL");
    if (!ok2)
      fail = 1;
  }
  // (4) forced fallback: maxLevels = 1 leaves a 1728-DOF coarsest level ⇒ CG, flagged
  for (Aggregation ag : {Aggregation::Greedy, Aggregation::Mis2}) {
    const HostCsrOp A = poisson7(12);
    AmgParams prm;
    prm.aggregation = ag;
    prm.coarseSolve = CoarseSolve::Dense;
    prm.maxLevels = 1;
    GraphAMGDevice M;
    M.build(A, prm);
    const int it = pcg(A, M, rhs(A.n, 2), 1e-8);
    const bool ok = M.denseFallback() && M.coarseSolveUsed() == CoarseSolve::Cg && it <= 3;
    std::printf("  dense fallback (%s, maxLevels=1, n=%lld): fallback=%d used=%s iters=%d %s\n",
                ag == Aggregation::Mis2 ? "mis2" : "greedy", (long long)A.n, (int)M.denseFallback(),
                M.coarseSolveUsed() == CoarseSolve::Dense ? "dense" : "cg", it, ok ? "ok" : "FAIL");
    if (!ok)
      fail = 1;
  }
  // (5) the host oracle rejects the device-only options
  {
    bool threw = false;
    try {
      GraphAMG h;
      AmgParams p;
      p.aggregation = Aggregation::Mis2;
      h.build(poisson7(4), p);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) {
      std::printf("FAILED: host oracle accepted Aggregation::Mis2\n");
      fail = 1;
    }
  }
  std::printf(fail ? "FAILED\n" : "OK: parallel GraphAMG setup\n");
  return fail;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    if (argc >= 3 && std::strcmp(argv[1], "--hash") == 0)
      rc = runHash(std::atoi(argv[2]));
    else if (argc >= 3 && std::strcmp(argv[1], "--bench") == 0)
      rc = runBench(std::atoi(argv[2]));
    else
      rc = runGate();
  }
  Kokkos::finalize();
  return rc;
}

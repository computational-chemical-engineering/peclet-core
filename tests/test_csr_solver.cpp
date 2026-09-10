// The face-CSR solver layer (peclet::core::solver: face_csr / coloring / csr_operator /
// csr_bicgstab / vector_ops), lifted out of the AMR tree (QUALITY_PLAN G.2) and exercised here
// without any octree: a 3-D 7-point reaction–diffusion operator assembled as a diagonal + face CSR
// on a periodic n³ grid.
//   (1) the host row kernels and the device matvec agree on the same CSR (bit-exact on host
//       backends; FMA contraction on GPUs — checked to tolerance);
//   (2) greedyColoring is a proper colouring of the SYMMETRISED adjacency (no two cells sharing an
//       edge, in either direction, get one colour; every cell is coloured exactly once) — including
//       a structurally asymmetric CSR, the case the symmetrisation exists for;
//   (3) weighted Jacobi and the symmetric multicolour GS sweep both reduce the residual, GS faster;
//   (4) Jacobi-preconditioned BiCGStab and the defect-correction solve reach the tolerance and
//       recover a manufactured solution, on the diagonally dominant (small-dt-like) operator and
//       on the weak-reaction (large-dt-like) one where plain Jacobi stalls.
// Runs on whatever backend Kokkos targets (CUDA / HIP / OpenMP).
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/coloring.hpp"
#include "peclet/core/solver/csr_bicgstab.hpp"
#include "peclet/core/solver/csr_operator.hpp"
#include "peclet/core/solver/face_csr.hpp"
#include "peclet/core/solver/vector_ops.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::solver;

namespace {

struct HostCsr {
  Index n = 0;
  std::vector<double> diag, coef;
  std::vector<Index> start, nbr;
};

// (reaction + 6·kappa) on the diagonal, −kappa per face neighbour: A = c·I − kappa·∇²_h on a
// periodic m³ grid, x-fastest.
HostCsr laplacian(Index m, double reaction, double kappa) {
  HostCsr A;
  A.n = m * m * m;
  A.diag.assign(static_cast<std::size_t>(A.n), reaction + 6.0 * kappa);
  A.start.assign(static_cast<std::size_t>(A.n) + 1, 0);
  for (Index i = 0; i < A.n; ++i) {
    const Index x = i % m, y = (i / m) % m, z = i / (m * m);
    const Index nb[6] = {((x + 1) % m) + y * m + z * m * m, ((x + m - 1) % m) + y * m + z * m * m,
                         x + ((y + 1) % m) * m + z * m * m, x + ((y + m - 1) % m) * m + z * m * m,
                         x + y * m + ((z + 1) % m) * m * m, x + y * m + ((z + m - 1) % m) * m * m};
    for (Index k = 0; k < 6; ++k) {
      A.nbr.push_back(nb[k]);
      A.coef.push_back(-kappa);
    }
    A.start[static_cast<std::size_t>(i) + 1] = static_cast<Index>(A.nbr.size());
  }
  return A;
}

MomentumOp upload(const HostCsr& A) {
  MomentumOp op;
  op.n = A.n;
  op.diag = toDevice(A.diag, "t_diag");
  op.faceStart = toDevice(A.start, "t_start");
  op.faceNbr = toDevice(A.nbr, "t_nbr");
  op.faceCoef = toDevice(A.coef, "t_coef");
  return op;
}

std::vector<double> download(View<double> v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}

double residualNorm(const MomentumOp& op, View<double> u, View<const double> b) {
  View<double> r("t_r", static_cast<std::size_t>(op.n));
  residualMom(op, View<const double>(u), b, r);
  return std::sqrt(dotPlain(View<const double>(r), View<const double>(r), op.n));
}

// A proper colouring: no symmetrised edge joins two cells of one colour; every cell once.
bool properColoring(const HostCsr& A, const Coloring& col) {
  std::vector<int> color(static_cast<std::size_t>(A.n), -1);
  auto hidx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), col.idx);
  for (int c = 0; c < col.nColors; ++c)
    for (Index k = col.hStart[static_cast<std::size_t>(c)];
         k < col.hStart[static_cast<std::size_t>(c) + 1]; ++k) {
      const Index i = hidx(k);
      if (color[static_cast<std::size_t>(i)] != -1)
        return false;  // coloured twice
      color[static_cast<std::size_t>(i)] = c;
    }
  for (Index i = 0; i < A.n; ++i) {
    if (color[static_cast<std::size_t>(i)] < 0)
      return false;  // never coloured
    for (Index k = A.start[static_cast<std::size_t>(i)];
         k < A.start[static_cast<std::size_t>(i) + 1]; ++k)
      if (color[static_cast<std::size_t>(A.nbr[static_cast<std::size_t>(k)])] ==
          color[static_cast<std::size_t>(i)])
        return false;  // an edge (either direction) inside one colour
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const Index m = 12;
  const HostCsr A = laplacian(m, 4.0, 1.0);  // strongly diagonally dominant
  const MomentumOp op = upload(A);
  const Index n = A.n;

  // Manufactured solution and RHS b = A u_exact (host row kernel through HostArr).
  std::vector<double> ue(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i)
    ue[static_cast<std::size_t>(i)] =
        std::sin(0.37 * static_cast<double>(i)) + 0.1 * static_cast<double>(i % 7);
  FaceCsrOpT<HostArr<double>, HostArr<Index>> hop;
  hop.n = n;
  hop.diag = HostArr<double>(A.diag.data());
  hop.coef = HostArr<double>(A.coef.data());
  hop.start = HostArr<Index>(A.start.data());
  hop.nbr = HostArr<Index>(A.nbr.data());
  std::vector<double> bh(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i)
    bh[static_cast<std::size_t>(i)] = faceCsrApplyRow(hop, i, HostArr<double>(ue.data()));

  // (1) device matvec == host row kernel.
  View<double> ud = toDevice(ue, "t_u"), Au("t_Au", static_cast<std::size_t>(n));
  applyMom(op, View<const double>(ud), Au);
  {
    const auto h = download(Au);
    double maxErr = 0.0;
    for (Index i = 0; i < n; ++i)
      maxErr = std::max(
          maxErr, std::fabs(h[static_cast<std::size_t>(i)] - bh[static_cast<std::size_t>(i)]));
    std::printf("device matvec vs host row kernel: max |diff| = %.3e\n", maxErr);
    PECLET_CORE_CHECK(maxErr < 1e-13);
  }

  // (2) colouring: the symmetric 7-point graph, and an asymmetric CSR (extra one-way entries).
  Coloring col = greedyColoring(A.start, A.nbr, n);
  std::printf("greedyColoring: %d colours on the periodic %lld^3 7-point graph\n", col.nColors,
              static_cast<long long>(m));
  PECLET_CORE_CHECK(col.nColors >= 2 && col.nColors <= 8);
  PECLET_CORE_CHECK(properColoring(A, col));
  {
    HostCsr B = A;
    // one-way references i -> i+17 (no back edge): symmetrisation must still separate them.
    std::vector<Index> s2(1, 0), nb2;
    for (Index i = 0; i < n; ++i) {
      for (Index k = A.start[static_cast<std::size_t>(i)];
           k < A.start[static_cast<std::size_t>(i) + 1]; ++k)
        nb2.push_back(A.nbr[static_cast<std::size_t>(k)]);
      if (i % 5 == 0)
        nb2.push_back((i + 17) % n);
      s2.push_back(static_cast<Index>(nb2.size()));
    }
    B.start = s2;
    B.nbr = nb2;
    B.coef.assign(nb2.size(), -1.0);
    Coloring c2 = greedyColoring(B.start, B.nbr, n);
    PECLET_CORE_CHECK(properColoring(B, c2));
    // Deterministic: the same CSR colours identically.
    Coloring c3 = greedyColoring(B.start, B.nbr, n);
    PECLET_CORE_CHECK(c3.nColors == c2.nColors && c3.hStart == c2.hStart);
  }

  // (3) smoothers reduce the residual; the symmetric multicolour GS beats one Jacobi sweep.
  View<double> b = toDevice(bh, "t_b");
  View<const double> bc(b);
  {
    View<double> uj("t_uj", static_cast<std::size_t>(n)), tmp("t_tmp", static_cast<std::size_t>(n));
    View<double> ug("t_ug", static_cast<std::size_t>(n));
    const double r0 = residualNorm(op, uj, bc);
    for (int s = 0; s < 5; ++s)
      jacobiMom(op, uj, bc, tmp, 0.7);
    const double rj = residualNorm(op, uj, bc);
    for (int s = 0; s < 5; ++s)
      multicolorGSMom(op, ug, bc, col, 1.0);
    const double rg = residualNorm(op, ug, bc);
    std::printf("5 sweeps: r0 %.3e  jacobi %.3e  multicolour SGS %.3e\n", r0, rj, rg);
    PECLET_CORE_CHECK(rj < 0.5 * r0);
    PECLET_CORE_CHECK(rg < rj);
  }

  // (4) BiCGStab / defect correction on the dominant and on the weak-reaction operator.
  for (double reaction : {4.0, 1e-3}) {
    const HostCsr Aw = laplacian(m, reaction, 1.0);
    const MomentumOp opw = upload(Aw);
    hop.diag = HostArr<double>(Aw.diag.data());
    std::vector<double> bw(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i)
      bw[static_cast<std::size_t>(i)] = faceCsrApplyRow(hop, i, HostArr<double>(ue.data()));
    View<double> bwd = toDevice(bw, "t_bw");
    MomentumSolver solver;
    solver.setJacobi(2, 0.7);
    View<double> x("t_x", static_cast<std::size_t>(n));
    const auto R = solver.solveBiCGStab(opw, x, View<const double>(bwd), 500, 1e-10);
    const auto xh = download(x);
    double err = 0.0;
    for (Index i = 0; i < n; ++i)
      err = std::max(err,
                     std::fabs(xh[static_cast<std::size_t>(i)] - ue[static_cast<std::size_t>(i)]));
    std::printf("BiCGStab (reaction %g): %d iters, res %.3e -> %.3e, max err %.3e\n", reaction,
                R.iters, R.res0, R.res, err);
    PECLET_CORE_CHECK(R.res <= 1e-10 * R.res0);
    PECLET_CORE_CHECK(err < 1e-7);
    if (reaction > 1.0) {
      View<double> y("t_y", static_cast<std::size_t>(n));
      const auto D = solver.solveDefectCorrection(opw, y, View<const double>(bwd), 200, 1e-8);
      std::printf("defect correction: %d iters, res %.3e -> %.3e\n", D.iters, D.res0, D.res);
      PECLET_CORE_CHECK(D.res <= 1e-8 * D.res0);
      // The MG-preconditioner hook: any z = M⁻¹ r callable. Plain Jacobi sweeps as a stand-in.
      MomentumSolver s2;
      View<double> tmp("t_tmp2", static_cast<std::size_t>(n));
      s2.setPreconditioner([&](View<const double> r, View<double> z) {
        Kokkos::deep_copy(z, 0.0);
        for (int s = 0; s < 3; ++s)
          jacobiMom(opw, z, r, tmp, 0.7);
      });
      View<double> x2("t_x2", static_cast<std::size_t>(n));
      const auto R2 = s2.solveBiCGStab(opw, x2, View<const double>(bwd), 500, 1e-10);
      PECLET_CORE_CHECK(R2.res <= 1e-10 * R2.res0);
    }
  }

  // vector_ops: the elementwise primitives against a host evaluation.
  {
    std::vector<double> ph(static_cast<std::size_t>(n)), rh(static_cast<std::size_t>(n)),
        vh(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
      ph[static_cast<std::size_t>(i)] = 0.5 * static_cast<double>(i % 11);
      rh[static_cast<std::size_t>(i)] = 1.0 - 0.01 * static_cast<double>(i % 13);
      vh[static_cast<std::size_t>(i)] = 0.25 * static_cast<double>(i % 3);
    }
    View<double> p = toDevice(ph, "t_p"), r = toDevice(rh, "t_r2"), v = toDevice(vh, "t_v");
    bicgPUpdate(p, View<const double>(r), View<const double>(v), 0.3, 0.8, n);
    axpy(p, -2.0, View<const double>(v), n);
    zpby(p, View<const double>(r), 0.5, n);
    negate(p, n);
    const auto got = download(p);
    double maxErr = 0.0;
    for (Index i = 0; i < n; ++i) {
      const std::size_t s = static_cast<std::size_t>(i);
      double e = rh[s] + 0.3 * (ph[s] - 0.8 * vh[s]);
      e += -2.0 * vh[s];
      e = rh[s] + 0.5 * e;
      e = -e;
      maxErr = std::max(maxErr, std::fabs(got[s] - e));
    }
    PECLET_CORE_CHECK(maxErr < 1e-14);
    const double d = dotPlain(View<const double>(r), View<const double>(v), n);
    double dh = 0.0;
    for (Index i = 0; i < n; ++i)
      dh += rh[static_cast<std::size_t>(i)] * vh[static_cast<std::size_t>(i)];
    PECLET_CORE_CHECK(std::fabs(d - dh) < 1e-9 * std::fabs(dh));
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

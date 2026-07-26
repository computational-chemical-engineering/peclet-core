// Distributed cut-cell momentum operator + solve (docs/amr_distributed_flow.md, rung 2):
// AmrCutCell built per rank with the LeafHalo resolver seam (probes that exit the ORB block
// resolve to ghost slots; SDF/openness evaluated in the GLOBAL frame so every rank samples
// bit-identical geometry), then the device BiCGStab with a halo refresh before every matvec /
// preconditioner sweep and Allreduce'd dots. On a graded octree with a finest cut band around a
// sphere it validates, at np = 1,2,4,8:
//   (1) the distributed operator CSR row-for-row BIT-EXACT against the single-rank AmrCutCell
//       on the whole domain (rows mapped by global Morton code; ghost columns mapped through
//       their anchors) — diag, off-diagonal (code, coef) sets, rscale, fluid/cut flags;
//   (2) np=1: the distributed solve reproduces the single-rank solve BIT-exactly (the resolver
//       resolves every wrapped probe locally; the halo refresh and the 1-rank Allreduce are
//       exact no-ops);
//   (3) np>1: the distributed BiCGStab converges and matches the single-rank solution to
//       Krylov tolerance (the iterate sequence differs only through dot reduction order).
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <algorithm>
#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <utility>
#include <vector>

#include "peclet/core/common/view.hpp"

#include "peclet/core/amr/cut_cell.hpp"
#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/leaf_halo.hpp"
#include "peclet/core/amr/momentum.hpp"
#include "peclet/core/common/mpi.hpp"

using namespace peclet::core;
using namespace peclet::core::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;
using Coord = DO::Coord;

// Fluid outside a sphere at the domain centre.
double sphereSdf(const Vec<3>& p) {
  const double dx = p[0] - 0.5, dy = p[1] - 0.5, dz = p[2] - 0.5;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - 0.2;
}

double fAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0,
         cz = ((double)o[2] + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cx);
}

// Refine a finest band around the sphere surface (rank-independent policy: world-coord SDF at
// the cell centre), then cross-block 2:1 balance. Cut cells land well inside the finest band.
void makeMesh(DO& d, double h0) {
  for (int pass = 0; pass < 2; ++pass) {
    d.local().refineIf([&](Code c, unsigned lvl) -> bool {
      if (lvl == 0)
        return false;
      auto o = M::from_code(c).decode();
      const double s = static_cast<double>(1u << lvl);
      Vec<3> ctr{};
      for (int a = 0; a < 3; ++a)
        ctr[a] = (static_cast<double>((long)o[a] + d.blockFineOrigin()[a]) + 0.5 * s) * h0;
      const double w = s * h0;
      return std::fabs(sphereSdf(ctr)) < w + 2.0 * h0;
    });
  }
  d.balance();
}

// The distributed AmrCutCell build: resolver → LeafHalo registry, miss-collect fixpoint,
// ghost metadata refreshed per round, GLOBAL-frame SDF sampling.
void buildDistributed(DO& d, LeafHalo<3, kBits>& halo, AmrCutCell<kBits>& mom, double h0,
                      double idiag, double beta) {
  halo.init(d);
  mom.init(d.local(), h0, Vec<3>{0.0, 0.0, 0.0});  // GLOBAL origin
  std::array<long, 3> shift{};
  for (int a = 0; a < 3; ++a)
    shift[a] = d.blockFineOrigin()[a];
  mom.setFrameShift(shift);
  mom.setResolver([&halo, shift](const std::array<long, 3>& p) -> Index {
    std::array<long, 3> g = p;
    for (int a = 0; a < 3; ++a)
      g[a] += shift[a];
    return halo.resolveGlobal(g);
  });
  for (;;) {
    std::vector<std::array<long, 3>> glo(static_cast<std::size_t>(halo.numGhosts()));
    std::vector<unsigned> glv(static_cast<std::size_t>(halo.numGhosts()));
    for (Index g = 0; g < halo.numGhosts(); ++g) {
      for (int a = 0; a < 3; ++a)
        glo[static_cast<std::size_t>(g)][a] =
            static_cast<long>(halo.ghostCoord(g)[a]) - shift[a];
      glv[static_cast<std::size_t>(g)] =
          static_cast<unsigned>(halo.level(halo.numLocal() + g));
    }
    mom.setGhosts(std::move(glo), std::move(glv));
    mom.build(sphereSdf, idiag, beta);
    if (halo.resolveMisses() == 0)
      break;
  }
  halo.finalize();
}

MomentumOp upload(const AmrCutCell<kBits>::Assembled& A, Index n) {
  MomentumOp op;
  op.n = n;
  op.diag = toDevice(A.diag, "mo_diag");
  op.faceStart = toDevice(A.start, "mo_start");
  op.faceNbr = toDevice(A.nbr, "mo_nbr");
  op.faceCoef = toDevice(A.coef, "mo_coef");
  return op;
}

std::vector<double> down(const View<double>& d) {
  std::vector<double> h(d.extent(0));
  auto m = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(m, d);
  for (std::size_t i = 0; i < h.size(); ++i)
    h[i] = m(i);
  return h;
}

void run() {
  const long Nr = 4;  // 4^3 roots, lmax 2 ⇒ 16^3 fine, periodic [0,1)^3
  const unsigned lmax = 2;
  const double h0 = 1.0 / (Nr * (1 << lmax));
  const double idiag = 0.2, beta = 1.0;
  AmrGeometry<3> geo;
  geo.h0 = h0;
  const std::array<bool, 3> per{true, true, true};
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // ---- distributed (COMM_WORLD) ----
  DO world;
  world.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeMesh(world, h0);
  const Index n = world.local().numLeaves();
  LeafHalo<3, kBits> halo;
  AmrCutCell<kBits> mom;
  buildDistributed(world, halo, mom, h0, idiag, beta);
  auto Aw = mom.assembleOperator();

  // ---- single-rank reference (whole domain on COMM_SELF, plain AmrCutCell) ----
  DO self;
  self.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_SELF);
  makeMesh(self, h0);
  const Index ns = self.local().numLeaves();
  AmrCutCell<kBits> momS;
  momS.init(self.local(), h0, Vec<3>{0.0, 0.0, 0.0});
  momS.build(sphereSdf, idiag, beta);
  auto As = momS.assembleOperator();

  long lw = (long)n, gw = 0;
  MPI_Allreduce(&lw, &gw, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  PECLET_CORE_CHECK(gw == (long)ns);
  if (size > 1)
    PECLET_CORE_CHECK(halo.numGhosts() > 0);  // the seam is actually exercised

  // ---- (1) operator CSR row-for-row bit-exact vs the single-rank reference ----
  auto codeOfExt = [&](Index slot) -> Code {
    if (slot < n)
      return world.globalCode(slot);
    return M::encode(halo.ghostCoord(slot - n)).code();
  };
  Index mismRow = 0;
  for (Index i = 0; i < n; ++i) {
    const Index si = self.local().find(world.globalCode(i));
    PECLET_CORE_CHECK(si >= 0);
    bool ok = Aw.diag[(std::size_t)i] == As.diag[(std::size_t)si] &&
              mom.rhsScale(i) == momS.rhsScale(si) && mom.isFluid(i) == momS.isFluid(si) &&
              mom.isCut(i) == momS.isCut(si);
    std::vector<std::pair<Code, double>> rw, rs;
    for (Index k = Aw.start[(std::size_t)i]; k < Aw.start[(std::size_t)i + 1]; ++k)
      rw.emplace_back(codeOfExt(Aw.nbr[(std::size_t)k]), Aw.coef[(std::size_t)k]);
    for (Index k = As.start[(std::size_t)si]; k < As.start[(std::size_t)si + 1]; ++k)
      rs.emplace_back(self.globalCode(As.nbr[(std::size_t)k]), As.coef[(std::size_t)k]);
    std::sort(rw.begin(), rw.end());
    std::sort(rs.begin(), rs.end());
    ok = ok && rw.size() == rs.size();
    if (ok)
      for (std::size_t k = 0; k < rw.size(); ++k)
        ok = ok && rw[k].first == rs[k].first && rw[k].second == rs[k].second;
    if (!ok)
      ++mismRow;
  }
  PECLET_CORE_CHECK_EQ(mismRow, 0);

  // ---- (2)+(3) distributed device BiCGStab vs the single-rank solve ----
  LeafHaloExchange ex;
  ex.init(halo);
  MomentumOp opW = upload(Aw, n), opS = upload(As, ns);

  std::vector<double> srcW((std::size_t)n), srcS((std::size_t)ns);
  for (Index i = 0; i < n; ++i)
    srcW[(std::size_t)i] = fAt(world.globalCode(i), h0);
  for (Index i = 0; i < ns; ++i)
    srcS[(std::size_t)i] = fAt(self.globalCode(i), h0);
  View<double> bW = toDevice(mom.makeRhs(srcW, 0.0), "bW");
  View<double> bS = toDevice(momS.makeRhs(srcS, 0.0), "bS");

  MomentumSolver<kBits> solW, solS;
  solW.setJacobi(2, 0.7);
  solS.setJacobi(2, 0.7);
  solW.setDistributed([&ex](View<double> v) { ex.exchange(v); },
                      [](double s) {
                        double g = 0.0;
                        MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                        return g;
                      },
                      halo.extendedSize());

  View<double> uW("uW", (std::size_t)halo.extendedSize()), uS("uS", (std::size_t)ns);
  Kokkos::deep_copy(uW, 0.0);
  Kokkos::deep_copy(uS, 0.0);
  auto rW = solW.solveBiCGStab(opW, uW, View<const double>(bW), 400, 1e-11);
  auto rS = solS.solveBiCGStab(opS, uS, View<const double>(bS), 400, 1e-11);
  PECLET_CORE_CHECK(rW.res <= 1e-10 * rW.res0 || rW.res <= 1e-13);
  PECLET_CORE_CHECK(rS.res <= 1e-10 * rS.res0 || rS.res <= 1e-13);

  std::vector<double> hW = down(uW), hS = down(uS);
  double umax = 0.0;
  for (Index i = 0; i < ns; ++i)
    umax = std::max(umax, std::fabs(hS[(std::size_t)i]));
  double dmax = 0.0;
  for (Index i = 0; i < n; ++i) {
    const Index si = self.local().find(world.globalCode(i));
    dmax = std::max(dmax, std::fabs(hW[(std::size_t)i] - hS[(std::size_t)si]));
  }
  double gdmax = 0.0;
  MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (size == 1) {
    PECLET_CORE_CHECK(gdmax == 0.0);  // np=1: distributed == single-rank bit-for-bit
  } else {
    PECLET_CORE_CHECK(gdmax <= 1e-7 * umax);  // Krylov tolerance (dot order differs)
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int fails = peclet::core::test::g_failures, total = 0;
  MPI_Reduce(&fails, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (rank == 0) {
    if (total == 0) {
      std::printf("OK\n");
      return 0;
    }
    std::fprintf(stderr, "%d failure(s)\n", total);
    return 1;
  }
  return 0;
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping distributed momentum test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

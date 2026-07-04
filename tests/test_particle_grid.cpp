// Trilinear particle<->grid interpolation (interp/particle_grid.hpp). Gather is exact for a linear
// field; scatter is the transpose of gather (conserves the deposited total and satisfies the
// adjoint identity <gather(f),q> == <f,scatter(q)>).
#include "test_util.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

#include "peclet/core/interp/particle_grid.hpp"

namespace {
using peclet::core::interp::GridMap;
using View = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;
using PosV = Kokkos::View<double* [3], Kokkos::DefaultExecutionSpace::memory_space>;

void run() {
  const int nx = 6, ny = 5, nz = 4, g = 2;
  const int ex = nx + 2 * g, ey = ny + 2 * g, ez = nz + 2 * g;
  const double ox = 3.0, oy = -1.0, oz = 0.5, h = 0.25;
  GridMap m{ox, oy, oz, 1.0 / h, 1.0 / h, 1.0 / h, ex, ey, ez, g};

  // linear field f = a + b*X + c*Y + d*Z at cell centres X = ox + (i+0.5)*h
  const double a = 1.5, b = 2.0, c = -0.7, d = 0.3;
  View field("f", (std::size_t)ex * ey * ez);
  {
    auto hf = Kokkos::create_mirror_view(field);  // fill the FULL padded block (incl. ghosts) with
    for (int kk = 0; kk < ez; ++kk)               // the linear field, so boundary-overhang particles
      for (int jj = 0; jj < ey; ++jj)             // interpolate exactly too (ghost fill is the
        for (int ii = 0; ii < ex; ++ii) {         // caller's job in production — here it isolates
          const double X = ox + (ii - g + 0.5) * h, Y = oy + (jj - g + 0.5) * h,  // the interpolation)
                       Z = oz + (kk - g + 0.5) * h;
          hf[(std::size_t)ii + (std::size_t)jj * ex + (std::size_t)kk * ex * ey] =
              a + b * X + c * Y + d * Z;
        }
    Kokkos::deep_copy(field, hf);
  }

  const int NP = 5;
  PosV pos("pos", NP);
  {
    auto hp = Kokkos::create_mirror_view(pos);
    double pts[NP][3] = {{3.4, -0.6, 0.8}, {3.9, -0.2, 1.1}, {3.55, -0.85, 0.55},
                         {4.2, -0.4, 1.3}, {3.1, -0.95, 0.6}};
    for (int p = 0; p < NP; ++p)
      for (int c2 = 0; c2 < 3; ++c2)
        hp(p, c2) = pts[p][c2];
    Kokkos::deep_copy(pos, hp);
  }

  // gather: exact for the linear field
  View gout("gout", NP);
  peclet::core::interp::trilinearGather(NP, pos, field, gout, m);
  auto hg = Kokkos::create_mirror_view(gout);
  Kokkos::deep_copy(hg, gout);
  auto hp = Kokkos::create_mirror_view(pos);
  Kokkos::deep_copy(hp, pos);
  double gmax = 0;
  for (int p = 0; p < NP; ++p) {
    const double want = a + b * hp(p, 0) + c * hp(p, 1) + d * hp(p, 2);
    gmax = std::fmax(gmax, std::fabs(hg(p) - want));
  }
  std::printf("gather linear-field max-err %.2e\n", gmax);
  PECLET_CORE_CHECK(gmax < 1e-12);

  // scatter: deposit q; total conserved and adjoint identity vs gather
  View q("q", NP), dep("dep", (std::size_t)ex * ey * ez);
  {
    auto hq = Kokkos::create_mirror_view(q);
    for (int p = 0; p < NP; ++p)
      hq(p) = 1.0 + 0.5 * p;
    Kokkos::deep_copy(q, hq);
  }
  peclet::core::interp::trilinearScatterAtomic(NP, pos, q, dep, m);
  auto hd = Kokkos::create_mirror_view(dep);
  Kokkos::deep_copy(hd, dep);
  double sumDep = 0, sumQ = 0, adjG = 0, adjS = 0;
  auto hq = Kokkos::create_mirror_view(q);
  Kokkos::deep_copy(hq, q);
  for (std::size_t i = 0; i < hd.extent(0); ++i)
    sumDep += hd(i);
  for (int p = 0; p < NP; ++p) {
    sumQ += hq(p);
    adjG += hg(p) * hq(p);  // <gather(f), q>
  }
  auto hf = Kokkos::create_mirror_view(field);
  Kokkos::deep_copy(hf, field);
  for (std::size_t i = 0; i < hd.extent(0); ++i)
    adjS += hf(i) * hd(i);  // <f, scatter(q)>
  std::printf("scatter conservation %.2e  adjoint |gather.q - f.scatter| %.2e\n",
              std::fabs(sumDep - sumQ), std::fabs(adjG - adjS));
  PECLET_CORE_CHECK(std::fabs(sumDep - sumQ) < 1e-12);
  PECLET_CORE_CHECK(std::fabs(adjG - adjS) < 1e-11);
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  PECLET_CORE_RETURN_TEST_RESULT();
}

// pvFitAdd == pvFitTerm + pvFitAccum, BITWISE, on the default execution space (CUDA / HIP /
// OpenMP) and on the host space (flow doc/vof_step_performance_design.md §4.7, §5.11, WO-7a).
//
// Three accumulations of the same stencils, each on the same backend:
//   ref   a frozen copy of pvFitAdd as it was before the split (below, verbatim body);
//   add   today's pvFitAdd (the composition);
//   team  the device fallback's shape: one kernel writes every stencil cell's PvTerm to memory,
//         a second kernel folds the accepted terms with pvFitAccum in the canonical order.
// 10^5 random cases (800 stencils x 5^3 cells), random planes, frames, origins, metrics and guards,
// with a planted share of rejected polygons (zero normal, plane missing the cell, normal below
// cosMin, outside the Wendland support). Every accumulator byte and every accept flag must match.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/vof/curvature.hpp"

using namespace peclet::core::vof;

namespace {

// --- the pre-split pvFitAdd (core 6859d4e), verbatim apart from its name -------------------------
KOKKOS_INLINE_FUNCTION bool pvFitAddReference(PvFit& f, double mx, double my, double mz,
                                              double alpha, const double off[3],
                                              const double org[3], const double t1[3],
                                              const double t2[3], const double nn[3], double dW,
                                              double cosMin, const VofMetric& g) {
  const double n2 = mx * mx + my * my + mz * mz;
  if (!(n2 > 0.0))
    return false;
  // Phase 3 (V2.4): the polygon's own normal enters the fit as a PHYSICAL direction (the frame is
  // physical), so it is pulled back through H^-1 and renormalized. At g.h = {1,1,1} the pullback
  // is `m/1.0` and `invn` is the same reciprocal square root as before — bitwise.
  const double mi[3] = {mx / g.h[0], my / g.h[1], mz / g.h[2]};
  const double invn = 1.0 / Kokkos::sqrt(mi[0] * mi[0] + mi[1] * mi[1] + mi[2] * mi[2]);
  const double np[3] = {(mi[0] * t1[0] + mi[1] * t1[1] + mi[2] * t1[2]) * invn,
                        (mi[0] * t2[0] + mi[1] * t2[1] + mi[2] * t2[2]) * invn,
                        (mi[0] * nn[0] + mi[1] * nn[1] + mi[2] * nn[2]) * invn};
  if (!(np[2] > cosMin))
    return false;

  double v[8][3];
  const int nv = plicPolygon(mx, my, mz, alpha, v);
  if (nv < 3)
    return false;

  // cell-local [0,1]^3 -> target-centred cell units -> the fit frame
  double xy[8][2];
  double zc = 0.0;
  double px = 0.0, py = 0.0;
  for (int k = 0; k < nv; ++k) {
    const double Xi[3] = {off[0] + v[k][0] - 0.5 - org[0], off[1] + v[k][1] - 0.5 - org[1],
                          off[2] + v[k][2] - 0.5 - org[2]};
    double X[3];
    g.toPhys(Xi, X);  // index displacement -> physical (identity at g.h = {1,1,1})
    xy[k][0] = X[0] * t1[0] + X[1] * t1[1] + X[2] * t1[2];
    xy[k][1] = X[0] * t2[0] + X[1] * t2[1] + X[2] * t2[2];
    const double z = X[0] * nn[0] + X[1] * nn[1] + X[2] * nn[2];
    px += xy[k][0];
    py += xy[k][1];
    zc += z;
  }
  const double invv = 1.0 / static_cast<double>(nv);
  px *= invv;
  py *= invv;
  zc *= invv;

  double s[6];
  polygonMoments2d(xy, nv, s);
  if (!(Kokkos::fabs(s[0]) > 1e-14))
    return false;

  // the polygon's own plane in the fit frame: z' = b0 + b1 x' + b2 y'
  const double b1 = -np[0] / np[2], b2 = -np[1] / np[2];
  const double b0 = zc - b1 * px - b2 * py;

  const double r = Kokkos::sqrt(px * px + py * py + zc * zc);
  const double w = wendlandWeight(r, dW);
  if (!(w > 0.0))
    return false;

  const double B = b0 * s[0] + b1 * s[1] + b2 * s[2];
  for (int i = 0; i < 6; ++i) {
    f.b[i] += w * s[i] * B;
    for (int j = 0; j < 6; ++j)
      f.A[i][j] += w * s[i] * s[j];
  }
  ++f.npoly;
  return true;
}

constexpr int kSide = 5, kCells = kSide * kSide * kSide, kStencils = 800;
constexpr int kCases = kStencils * kCells;  // 100000

struct Stencil {
  double org[3], t1[3], t2[3], nn[3], dW, cosMin;
  VofMetric g;
};
struct Cell {
  double mx, my, mz, alpha, off[3];
};

struct Rng {  // splitmix64: fixed seed, identical sequence on every host
  std::uint64_t s;
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double uni() { return (next() >> 11) * 0x1.0p-53; }  // [0,1)
  double sym() { return 2.0 * uni() - 1.0; }           // [-1,1)
};

void unit(double v[3]) {
  const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  for (int a = 0; a < 3; ++a)
    v[a] /= n;
}

void makeCases(std::vector<Stencil>& st, std::vector<Cell>& ce) {
  Rng r{20261002ULL};
  st.resize(kStencils);
  ce.resize(kCases);
  for (int s = 0; s < kStencils; ++s) {
    Stencil& S = st[s];
    double nn[3] = {r.sym(), r.sym(), r.sym()};
    unit(nn);
    // t1 = any unit vector orthogonal to nn, t2 = nn x t1
    double a[3] = {1, 0, 0};
    if (std::fabs(nn[0]) > 0.6)
      a[0] = 0, a[1] = 1;
    const double d = a[0] * nn[0] + a[1] * nn[1] + a[2] * nn[2];
    double t1[3] = {a[0] - d * nn[0], a[1] - d * nn[1], a[2] - d * nn[2]};
    unit(t1);
    const double t2[3] = {nn[1] * t1[2] - nn[2] * t1[1], nn[2] * t1[0] - nn[0] * t1[2],
                          nn[0] * t1[1] - nn[1] * t1[0]};
    for (int k = 0; k < 3; ++k) {
      S.nn[k] = nn[k];
      S.t1[k] = t1[k];
      S.t2[k] = t2[k];
      S.org[k] = 0.5 * r.sym();
    }
    S.dW = (s % 50 == 7) ? 0.0 : 1.5 + 2.0 * r.uni();  // a few unweighted stencils (d <= 0)
    const double cm[4] = {0.0, 0.2, 0.5, -1.0};
    S.cosMin = cm[s % 4];
    if (s % 3 != 0)  // two thirds anisotropic
      for (int k = 0; k < 3; ++k)
        S.g.h[k] = 0.5 + 1.5 * r.uni();
    for (int k = 0; k < kCells; ++k) {
      Cell& C = ce[s * kCells + k];
      C.off[0] = k % kSide - 2;
      C.off[1] = (k / kSide) % kSide - 2;
      C.off[2] = k / (kSide * kSide) - 2;
      const int kind = static_cast<int>(r.next() % 20);
      if (kind == 0) {  // degenerate normal
        C.mx = C.my = C.mz = 0.0;
      } else if (kind <= 3) {  // random direction (many fall below cosMin)
        C.mx = r.sym(), C.my = r.sym(), C.mz = r.sym();
      } else {  // near the target normal, in index space
        C.mx = nn[0] * S.g.h[0] + 0.3 * r.sym();
        C.my = nn[1] * S.g.h[1] + 0.3 * r.sym();
        C.mz = nn[2] * S.g.h[2] + 0.3 * r.sym();
      }
      if (kind == 1) {  // plane well outside the cube: no polygon
        C.alpha = 3.0 + r.uni();
      } else {  // plane through a random point of the cell
        const double p[3] = {r.uni(), r.uni(), r.uni()};
        C.alpha = C.mx * p[0] + C.my * p[1] + C.mz * p[2];
      }
    }
  }
}

struct Result {
  std::vector<PvFit> ref, add, team;
  std::vector<int> fref, fadd, fterm;
};

template <class Exec>
Result run(const std::vector<Stencil>& st, const std::vector<Cell>& ce) {
  using Mem = typename Exec::memory_space;
  Kokkos::View<Stencil*, Mem> S("st", kStencils);
  Kokkos::View<Cell*, Mem> C("ce", kCases);
  {
    auto hs = Kokkos::create_mirror_view(S);
    auto hc = Kokkos::create_mirror_view(C);
    for (int i = 0; i < kStencils; ++i)
      hs(i) = st[i];
    for (int i = 0; i < kCases; ++i)
      hc(i) = ce[i];
    Kokkos::deep_copy(S, hs);
    Kokkos::deep_copy(C, hc);
  }
  Kokkos::View<PvFit*, Mem> ref("ref", kStencils), add("add", kStencils), team("team", kStencils);
  Kokkos::View<int*, Mem> fref("fref", kCases), fadd("fadd", kCases), fterm("fterm", kCases);
  Kokkos::View<PvTerm*, Mem> terms("terms", kCases);

  // ref and add: one thread per stencil, cells in canonical order (as curvFallbackCell does).
  // Separate kernels, so the compiler cannot merge the two identical computations into one.
  Kokkos::parallel_for(
      "pvfit_ref", Kokkos::RangePolicy<Exec>(0, kStencils), KOKKOS_LAMBDA(const int s) {
        const Stencil& T = S(s);
        PvFit f;
        pvFitInit(f);
        for (int k = 0; k < kCells; ++k) {
          const Cell& c = C(s * kCells + k);
          fref(s * kCells + k) = pvFitAddReference(f, c.mx, c.my, c.mz, c.alpha, c.off, T.org, T.t1,
                                                   T.t2, T.nn, T.dW, T.cosMin, T.g);
        }
        ref(s) = f;
      });
  Kokkos::parallel_for(
      "pvfit_add", Kokkos::RangePolicy<Exec>(0, kStencils), KOKKOS_LAMBDA(const int s) {
        const Stencil& T = S(s);
        PvFit f;
        pvFitInit(f);
        for (int k = 0; k < kCells; ++k) {
          const Cell& c = C(s * kCells + k);
          fadd(s * kCells + k) = pvFitAdd(f, c.mx, c.my, c.mz, c.alpha, c.off, T.org, T.t1, T.t2,
                                          T.nn, T.dW, T.cosMin, T.g);
        }
        add(s) = f;
      });
  // team shape: the map over every case, then the ordered fold
  Kokkos::parallel_for(
      "pvfit_term", Kokkos::RangePolicy<Exec>(0, kCases), KOKKOS_LAMBDA(const int i) {
        const Stencil& T = S(i / kCells);
        const Cell& c = C(i);
        PvTerm t;
        fterm(i) = pvFitTerm(t, c.mx, c.my, c.mz, c.alpha, c.off, T.org, T.t1, T.t2, T.nn, T.dW,
                             T.cosMin, T.g);
        terms(i) = t;
      });
  Kokkos::parallel_for(
      "pvfit_accum", Kokkos::RangePolicy<Exec>(0, kStencils), KOKKOS_LAMBDA(const int s) {
        PvFit f;
        pvFitInit(f);
        for (int k = 0; k < kCells; ++k)
          if (terms(s * kCells + k).ok)
            pvFitAccum(f, terms(s * kCells + k));
        team(s) = f;
      });
  Kokkos::fence();

  Result R;
  auto pull = [](auto v, auto& out) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    out.assign(h.data(), h.data() + h.extent(0));
  };
  pull(ref, R.ref);
  pull(add, R.add);
  pull(team, R.team);
  pull(fref, R.fref);
  pull(fadd, R.fadd);
  pull(fterm, R.fterm);
  return R;
}

bool sameFit(const PvFit& a, const PvFit& b) {
  return std::memcmp(a.A, b.A, sizeof a.A) == 0 && std::memcmp(a.b, b.b, sizeof a.b) == 0 &&
         a.npoly == b.npoly;
}

int check(const char* name, const Result& R) {
  int badFit = 0, badFlag = 0, accepted = 0, fitted = 0;
  for (int s = 0; s < kStencils; ++s) {
    badFit += !sameFit(R.ref[s], R.add[s]) || !sameFit(R.ref[s], R.team[s]);
    fitted += R.ref[s].npoly >= 6;
  }
  for (int i = 0; i < kCases; ++i) {
    badFlag += (R.fref[i] != R.fadd[i]) || (R.fref[i] != R.fterm[i]);
    accepted += R.fref[i];
  }
  const int rejected = kCases - accepted;
  std::printf(
      "[%s] %d cases: %d accepted, %d rejected; %d/%d stencils with npoly >= 6; "
      "accumulator mismatches %d, flag mismatches %d\n",
      name, kCases, accepted, rejected, fitted, kStencils, badFit, badFlag);
  // non-vacuous: both branches exercised in volume, and most stencils carry a real fit
  const bool coverage = accepted > kCases / 5 && rejected > kCases / 20 && fitted > kStencils / 2;
  if (!coverage)
    std::printf("[%s] FAIL: coverage too thin to mean anything\n", name);
  return badFit + badFlag + (coverage ? 0 : 1);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int bad = 0;
  {
    std::vector<Stencil> st;
    std::vector<Cell> ce;
    makeCases(st, ce);
    bad += check(Kokkos::DefaultExecutionSpace::name(), run<Kokkos::DefaultExecutionSpace>(st, ce));
    bad += check(Kokkos::DefaultHostExecutionSpace::name(),
                 run<Kokkos::DefaultHostExecutionSpace>(st, ce));
  }
  Kokkos::finalize();
  std::printf(bad ? "FAIL\n" : "PASS\n");
  return bad ? 1 : 0;
}

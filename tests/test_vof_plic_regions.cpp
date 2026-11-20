// PLIC region fractions (peclet/core/vof/plic_regions.hpp; flow doc/vof_sharp_viscous_stress.md
// §5.2, WO-1): box × box face fraction, hat × box shear-link fraction, cell midsection.
//
//   (1) Monte-Carlo quadrature: 10^4 random planes (incl. a planted share of zero components),
//       2·10^6 samples per plane and region, |Δ| ≤ 3e-3 (statistical; σ ≤ 3.6e-4).
//   (2) box × box = the thin-slab limit of plicSlabVolume (q0 = b0 = 0) and of plicBoxVolume
//       (general q0, b0), within the a-priori bound δ²/24·max|f''| + 1e-11.
//   (3) hat × box with m_b = 0 = the 1-D hat CDF, ≤ 1e-14.
//   (4) plicVolume(m, plicAlpha(m, C)) = C (round-off) on reconstructed planes; midsection =
//       face fraction at ξ_p = 1/2 bitwise; complement f(m, α) + f(−m, −α) = 1.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <Kokkos_Core.hpp>

#include "peclet/core/vof/plic_regions.hpp"

using namespace peclet::core::vof;

namespace {

KOKKOS_INLINE_FUNCTION std::uint64_t splitmix(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
KOKKOS_INLINE_FUNCTION double uni(std::uint64_t& s) {
  return (double)(splitmix(s) >> 11) * (1.0 / 9007199254740992.0);
}

struct Plane {
  double m[3], alpha;
  int p, h;           // plane axis, hat axis (h != p)
  double xi, q0, b0;  // region placement (box × box uses q0, b0; hat × box uses qc = q0, b0)
};

// A random plane through (or near) a random region: components in [-1, 1], ~1/4 of the planes with
// one or two zero components; the plane passes through a random point of the region's bounding box.
Plane randomPlane(std::uint64_t& s) {
  Plane P;
  for (int k = 0; k < 3; ++k)
    P.m[k] = 2.0 * uni(s) - 1.0;
  const double r = uni(s);
  if (r < 0.15)
    P.m[(int)(3 * uni(s)) % 3] = 0.0;
  else if (r < 0.25) {
    const int k = (int)(3 * uni(s)) % 3;
    P.m[k] = 0.0;
    P.m[(k + 1) % 3] = 0.0;
  }
  P.p = (int)(3 * uni(s)) % 3;
  P.h = (P.p + 1 + (int)(2 * uni(s)) % 2) % 3;
  P.xi = 3.0 * uni(s) - 1.0;
  P.q0 = 3.0 * uni(s) - 1.0;
  P.b0 = 3.0 * uni(s) - 1.0;
  double x[3];
  for (int k = 0; k < 3; ++k)
    x[k] = 4.0 * uni(s) - 1.5;
  x[P.p] = P.xi;
  P.alpha = P.m[0] * x[0] + P.m[1] * x[1] + P.m[2] * x[2];
  return P;
}

int fails = 0;
void check(bool ok, const char* what, int k, double got, double want) {
  if (!ok) {
    if (fails < 20)
      std::printf("FAIL %s plane %d: got %.17g want %.17g (diff %.3e)\n", what, k, got, want,
                  got - want);
    ++fails;
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int kPlanes = 10000;
    constexpr long kSamples = 2000000;
    std::uint64_t seed = 20261031ULL;
    double maxMc[3] = {0, 0, 0}, maxSlab = 0, maxHat = 0, maxRt = 0;

    for (int k = 0; k < kPlanes; ++k) {
      const Plane P = randomPlane(seed);
      const double m0 = P.m[0], m1 = P.m[1], m2 = P.m[2], al = P.alpha, xi = P.xi, q0 = P.q0,
                   b0 = P.b0;
      const int p = P.p, h = P.h;
      const int qa = (p == 0) ? 1 : 0, ba = (p == 2) ? 1 : 2;  // box × box axes (increasing)
      const int hb = 3 - p - h;                                // hat × box: the box axis
      // (1) Monte Carlo, three regions in one pass: counts of samples on the colour-1 side.
      double cnt[3] = {0, 0, 0};
      Kokkos::parallel_reduce(
          "plic_regions_mc", Kokkos::RangePolicy<>(0, kSamples),
          KOKKOS_LAMBDA(long n, double& cf, double& ch, double& cm) {
            std::uint64_t st = 0x5851f42d4c957f2dULL * (std::uint64_t)(k + 1) + (std::uint64_t)n;
            splitmix(st);
            const double mm[3] = {m0, m1, m2};
            const double u1 = uni(st), u2 = uni(st), u3 = uni(st);
            double x[3];
            x[p] = xi;  // face: q ∈ [q0, q0+1], b ∈ [b0, b0+1]
            x[qa] = q0 + u1;
            x[ba] = b0 + u2;
            if (mm[0] * x[0] + mm[1] * x[1] + mm[2] * x[2] < al)
              cf += 1.0;
            x[p] = xi;  // hat: x_h = qc − 1 + u1 + u3 (hat on [qc−1, qc+1]), box on hb
            x[h] = q0 - 1.0 + u1 + u3;
            x[hb] = b0 + u2;
            if (mm[0] * x[0] + mm[1] * x[1] + mm[2] * x[2] < al)
              ch += 1.0;
            x[p] = 0.5;  // midsection of the unit cell
            x[qa] = u1;
            x[ba] = u2;
            if (mm[0] * x[0] + mm[1] * x[1] + mm[2] * x[2] < al)
              cm += 1.0;
          },
          cnt[0], cnt[1], cnt[2]);
      const double fF = plicFaceFraction(P.m, al, p, xi, q0, b0);
      const double fH = plicHatBoxFraction(P.m, al, p, xi, h, q0, b0);
      const double fM = plicMidsectionFraction(P.m, al, p);
      const double ref[3] = {cnt[0] / kSamples, cnt[1] / kSamples, cnt[2] / kSamples};
      const double got[3] = {fF, fH, fM};
      const char* nm[3] = {"mc face", "mc hatbox", "mc midsection"};
      for (int r = 0; r < 3; ++r) {
        const double d = std::fabs(got[r] - ref[r]);
        maxMc[r] = std::fmax(maxMc[r], d);
        check(d <= 3e-3 && got[r] >= 0.0 && got[r] <= 1.0, nm[r], k, got[r], ref[r]);
      }

      // (4a) midsection == face fraction at ξ_p = 1/2, q0 = b0 = 0, bitwise; (4b) complement.
      check(fM == plicFaceFraction(P.m, al, p, 0.5, 0.0, 0.0), "midsection bitwise", k, fM,
            plicFaceFraction(P.m, al, p, 0.5, 0.0, 0.0));
      const double mn[3] = {-m0, -m1, -m2};
      const double cF = plicFaceFraction(mn, -al, p, xi, q0, b0);
      const double cH = plicHatBoxFraction(mn, -al, p, xi, h, q0, b0);
      // (a region parallel to the plane is all-or-nothing and the plane itself is measure zero:
      // the complement holds only when the region's in-plane normal components are not both 0)
      if (P.m[qa] != 0.0 || P.m[ba] != 0.0)
        check(std::fabs(fF + cF - 1.0) <= 1e-14, "face complement", k, fF + cF, 1.0);
      if (P.m[h] != 0.0 || P.m[hb] != 0.0)
        check(std::fabs(fH + cH - 1.0) <= 1e-14, "hatbox complement", k, fH + cH, 1.0);

      // (2) thin-slab limit, generic planes only (every |m_k| ≥ 0.05 of the L1 norm): the slab
      // average of f over [ξ − δ/2, ξ + δ/2] differs from f(ξ) by ≤ δ²/24 · max|f''|, and f'' is
      // the ξ-derivative of the cut chord length, |f''| ≤ m_p² / (|m_q| |m_b|).
      const double l1 = std::fabs(m0) + std::fabs(m1) + std::fabs(m2);
      if (std::fabs(m0) >= 0.05 * l1 && std::fabs(m1) >= 0.05 * l1 && std::fabs(m2) >= 0.05 * l1) {
        const double dl = 1e-4;
        const double tol =
            dl * dl / 24.0 * P.m[p] * P.m[p] / (std::fabs(P.m[qa]) * std::fabs(P.m[ba])) + 1e-11;
        const double xs = 0.1 + 0.8 * uni(seed);  // inside the unit cell: plicSlabVolume's frame
        const double slab = plicSlabVolume(m0, m1, m2, al, p, xs - 0.5 * dl, xs + 0.5 * dl) / dl;
        const double f0 = plicFaceFraction(P.m, al, p, xs, 0.0, 0.0);
        maxSlab = std::fmax(maxSlab, std::fabs(slab - f0));
        check(std::fabs(slab - f0) <= tol, "slab limit", k, f0, slab);
        double lo[3], hi[3];
        lo[p] = xi - 0.5 * dl;
        hi[p] = xi + 0.5 * dl;
        lo[qa] = q0;
        hi[qa] = q0 + 1.0;
        lo[ba] = b0;
        hi[ba] = b0 + 1.0;
        const double box =
            plicBoxVolume(m0, m1, m2, al, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]) / dl;
        maxSlab = std::fmax(maxSlab, std::fabs(box - fF));
        check(std::fabs(box - fF) <= tol, "box limit", k, fF, box);
      }

      // (3) hat × box with m_b = 0 is the 1-D hat CDF (support [qc−1, qc+1], unit half-width).
      if (P.m[h] != 0.0) {
        double mz[3] = {m0, m1, m2};
        mz[hb] = 0.0;
        const double fz = plicHatBoxFraction(mz, al, p, xi, h, q0, b0);
        const double sc = (al - mz[p] * xi) / mz[h] - q0;  // the cut, relative to qc
        const double t = std::fmin(std::fmax(sc, -1.0), 1.0);
        double cdf = (t <= 0.0) ? 0.5 * (t + 1.0) * (t + 1.0) : 1.0 - 0.5 * (1.0 - t) * (1.0 - t);
        if (mz[h] < 0.0)
          cdf = 1.0 - cdf;  // colour-1 side is x_h > cut
        maxHat = std::fmax(maxHat, std::fabs(fz - cdf));
        check(std::fabs(fz - cdf) <= 1e-14, "hat cdf", k, fz, cdf);
      }

      // (4c) a reconstructed plane carries its colour: plicVolume(m, plicAlpha(m, C)) = C.
      const double C = uni(seed);
      if (l1 > 0.0) {
        const double a = plicAlpha(m0, m1, m2, C);
        const double v = plicVolume(m0, m1, m2, a);
        maxRt = std::fmax(maxRt, std::fabs(v - C));
        check(std::fabs(v - C) <= 1e-13, "plicVolume(plicAlpha(C)) = C", k, v, C);
      }
    }
    std::printf(
        "plic_regions: %d planes x %ld samples: max |MC - f| face %.2e hatbox %.2e midsection "
        "%.2e (tol 3e-3); thin-slab %.2e; hat CDF %.2e (tol 1e-14); C round trip %.2e; %s (%d "
        "fails)\n",
        kPlanes, kSamples, maxMc[0], maxMc[1], maxMc[2], maxSlab, maxHat, maxRt,
        fails ? "FAIL" : "PASS", fails);
  }
  Kokkos::finalize();
  return fails ? 1 : 0;
}

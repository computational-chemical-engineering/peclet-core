// AndersonCore (peclet::core::solver, solver/anderson.hpp) on synthetic maps: the core unit tests
// U1–U6, U4b and U10 of flow doc/steady_acceleration.md §8 (rev 2), the §6.3 memory formula, and
// the host linear algebra (Jacobi eigendecomposition). Device kernels on whatever backend Kokkos
// was built for; single rank (U7 is test_anderson_mpi.cpp).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "anderson_test_maps.hpp"
#include "peclet/core/solver/anderson.hpp"
#include "test_util.hpp"

using peclet::core::Index;
using peclet::core::IVec;
using peclet::core::View;
using peclet::core::solver::AndersonCore;
using peclet::core::solver::AndersonRole;
using peclet::core::solver::AndersonState;
using peclet::core::test::applyLinear;
using peclet::core::test::Box;
using peclet::core::test::download;
using peclet::core::test::LinearMap;
using peclet::core::test::makeLinearMap;
namespace da = peclet::core::solver::detail::anderson;

namespace {

AndersonState oneField(const Box& b, View<double> f, AndersonRole role = AndersonRole::Velocity) {
  AndersonState s;
  s.fields = {f};
  s.roles = {role};
  s.extent = b.e;
  s.ghost = b.g;
  return s;
}

/// Digest of every iterate and every decision a test produces (FNV-1a over the bytes of the
/// state fields, γ, the residual, the column count and the status after each complete()). Printed
/// per test; two builds that print the same digest produced bit-identical iterates.
struct Digest {
  std::uint64_t h = 1469598103934665603ull;
  void add(const void* p, std::size_t n) {
    const auto* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i)
      h = (h ^ c[i]) * 1099511628211ull;
  }
};
Digest g_digest;

void record(const AndersonCore& acc) {
  for (const auto& f : acc.state().fields) {
    const auto v = download(f);
    g_digest.add(v.data(), v.size() * sizeof(double));
  }
  const auto gamma = acc.gamma();
  g_digest.add(gamma.data(), gamma.size() * sizeof(double));
  const double res = acc.residual();
  const int cols = acc.numColumns(), status = static_cast<int>(acc.status());
  g_digest.add(&res, sizeof res);
  g_digest.add(&cols, sizeof cols);
  g_digest.add(&status, sizeof status);
}

void complete(AndersonCore& acc) {
  acc.complete();
  record(acc);
}

/// One map evaluation through the split interface; returns whether the input was mixed.
template <class Map>
bool evaluate(AndersonCore& acc, bool accelerate, Map&& map) {
  const bool mixed = acc.prepare(accelerate);
  map();
  complete(acc);
  return mixed;
}

template <class Test>
void runTest(const char* name, Test&& test) {
  g_digest = Digest{};
  test();
  std::printf("digest %s %016llx\n", name, static_cast<unsigned long long>(g_digest.h));
}

bool allFinite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

// ---- U1: linear contraction ---------------------------------------------------------------------
// The note's original U1 (an evenly spread spectrum on [0, 0.996] to 1e-12 in 80 steps) was a spec
// error. Coordinator decision 2026-10-02: the premise (D12, §1.1, the measured checkerboard) is ONE
// isolated slow mode over a fast bulk (U1a); three slow modes need more columns than m = 5 gives
// (measured 330 steps; m = 8 makes 77), kept as the regression U1c; a continuum is the documented
// limitation, kept as the regression U1b.

/// 10^4 diagonal entries: a seeded uniform bulk in [0, 0.5] plus isolated slow modes, spread over
/// the box.
LinearMap slowModeMap(const Box& b, const std::vector<double>& slow) {
  std::vector<double> lam(static_cast<std::size_t>(b.nPad), 0.0);
  std::vector<Index> innerIdx;
  for (Index i = 0; i < b.nPad; ++i)
    if (b.inner(i))
      innerIdx.push_back(i);
  std::mt19937_64 rng(20261002);
  std::uniform_real_distribution<double> bulk(0.0, 0.5);
  for (const Index i : innerIdx)
    lam[static_cast<std::size_t>(i)] = bulk(rng);
  for (std::size_t k = 0; k < slow.size(); ++k)
    lam[static_cast<std::size_t>(innerIdx[(k + 1) * innerIdx.size() / 4])] = slow[k];
  return makeLinearMap(lam, std::vector<double>(b.nPad, 1.0));
}

/// Accelerated steps (window 5) until the residual is <= 1e-12, at most maxSteps.
int stepsToTolerance(const Box& b, const LinearMap& map, int maxSteps, AndersonCore::Status& status,
                     int& restarts, double& residual) {
  View<double> x("u1_x", b.nPad);
  AndersonCore acc(oneField(b, x), 5);
  int steps = 0;
  while (steps < maxSteps && !(acc.residual() <= 1e-12)) {
    evaluate(acc, true, [&] { applyLinear(map, x); });
    ++steps;
  }
  status = acc.status();
  restarts = acc.numRestarts();
  residual = acc.residual();
  return steps;
}

// U1a premise: one isolated mode 0.996. Window 5 reaches 1e-12 within 80 steps; the plain march
// does not in 3000.
void testU1a() {
  const Box b(25, 20, 20, 2);  // 10^4 inner entries
  const LinearMap map = slowModeMap(b, {0.996});
  AndersonCore::Status status{};
  int restarts = 0;
  double res = 0.0;
  const int steps = stepsToTolerance(b, map, 1000, status, restarts, res);
  std::printf("U1a: one slow mode: window 5 reaches residual %.3e after %d steps (restarts %d)\n",
              res, steps, restarts);
  PECLET_CORE_CHECK(res <= 1e-12);
  PECLET_CORE_CHECK(steps <= 80);
  View<double> x("u1a_xp", b.nPad);
  AndersonCore acc(oneField(b, x), 5);
  for (int k = 0; k < 3000; ++k)
    evaluate(acc, false, [&] { applyLinear(map, x); });
  std::printf("U1a: plain march after 3000 steps: residual %.3e\n", acc.residual());
  PECLET_CORE_CHECK(acc.residual() > 1e-12);
}

// U1c regression: three slow modes {0.996, 0.99, 0.98}: window 5 within 400 steps (measured
// 330 host / 332 CUDA), no restart, still active.
void testU1c() {
  const Box b(25, 20, 20, 2);
  const LinearMap map = slowModeMap(b, {0.996, 0.99, 0.98});
  AndersonCore::Status status{};
  int restarts = 0;
  double res = 0.0;
  const int steps = stepsToTolerance(b, map, 1000, status, restarts, res);
  std::printf(
      "U1c: three slow modes: window 5 reaches residual %.3e after %d steps (restarts %d)\n", res,
      steps, restarts);
  PECLET_CORE_CHECK(res <= 1e-12);
  PECLET_CORE_CHECK(steps <= 400);
  PECLET_CORE_CHECK(restarts == 0);
  PECLET_CORE_CHECK(status == AndersonCore::Status::Active);
}

// U1b continuum regression: linspace [0, 0.996] — Anderson stays active, reaches <= 1e-6 by step
// 400, and is never behind the plain march at every 50th step.
void testU1b() {
  const Box b(25, 20, 20, 2);
  const LinearMap map =
      makeLinearMap(peclet::core::test::u1Spectrum(b), std::vector<double>(b.nPad, 1.0));
  View<double> xa("u1b_xa", b.nPad), xp("u1b_xp", b.nPad);
  AndersonCore acc(oneField(b, xa), 5), plain(oneField(b, xp), 5);
  bool ahead = true;
  for (int k = 1; k <= 400; ++k) {
    evaluate(acc, true, [&] { applyLinear(map, xa); });
    evaluate(plain, false, [&] { applyLinear(map, xp); });
    if (k % 50 == 0) {
      std::printf("  U1b step %3d: anderson %.3e  plain %.3e\n", k, acc.residual(),
                  plain.residual());
      ahead = ahead && acc.residual() <= plain.residual();
    }
  }
  std::printf(
      "U1b: continuum, step 400 residual %.3e (status %s), anderson <= plain at every "
      "50th step %d\n",
      acc.residual(), acc.statusName(), ahead ? 1 : 0);
  PECLET_CORE_CHECK(acc.residual() <= 1e-6);
  PECLET_CORE_CHECK(acc.status() == AndersonCore::Status::Active);
  PECLET_CORE_CHECK(ahead);
}

// ---- U2: affine hull — every output has Σ_inner x = S; mixed states keep it ---------------------
void testU2() {
  const Box b(16, 12, 10, 2);
  std::vector<double> lam(b.nPad, 0.0), c(b.nPad, 0.0);
  for (Index i = 0; i < b.nPad; ++i)
    if (b.inner(i)) {
      lam[i] = 0.99 * std::fmod(0.618034 * static_cast<double>(i), 1.0);
      c[i] = 1.0 + 0.5 * std::sin(0.1 * static_cast<double>(i));
    }
  const LinearMap lin = makeLinearMap(lam, c);
  const double S = 3.0 * static_cast<double>(b.nInner);
  const double invN = 1.0 / static_cast<double>(b.nInner);
  View<double> mask("u2_mask", b.nPad);
  {
    std::vector<double> mh(b.nPad, 0.0);
    for (Index i = 0; i < b.nPad; ++i)
      mh[i] = b.inner(i) ? 1.0 : 0.0;
    mask = peclet::core::toDevice(mh, "u2_mask");
  }
  auto innerSum = [&](View<const double> v) {
    const auto h = download(v);
    long double s = 0.0L;
    for (Index i = 0; i < b.nPad; ++i)
      if (b.inner(i))
        s += h[i];
    return static_cast<double>(s);
  };
  // g = y − (Σ_inner y − S)/N on inner entries, y = Λx + c.
  View<double> x("u2_x", b.nPad);
  auto map = [&] {
    applyLinear(lin, x);
    const double shift = (innerSum(x) - S) * invN;
    View<double> xv = x;
    View<const double> mk = mask;
    Kokkos::parallel_for(
        "u2_shift", x.extent(0), KOKKOS_LAMBDA(const std::size_t i) { xv(i) -= shift * mk(i); });
  };
  AndersonCore acc(oneField(b, x), 5);
  double worstOut = 0.0, worstMix = 0.0;
  int nMixed = 0;
  for (int k = 0; k < 60; ++k) {
    const bool mixed = acc.prepare(true);
    if (mixed) {
      worstMix = std::max(worstMix, std::abs(innerSum(x) - S));
      ++nMixed;
    }
    map();
    worstOut = std::max(worstOut, std::abs(innerSum(x) - S));
    complete(acc);
  }
  std::printf("U2: %d mixed states, max |Σx − S|/|S| = %.3e (outputs %.3e), residual %.3e\n",
              nMixed, worstMix / S, worstOut / S, acc.residual());
  PECLET_CORE_CHECK(nMixed > 10);
  PECLET_CORE_CHECK(worstMix <= 1e-14 * std::abs(S));
}

// ---- U3: a NaN at the 3rd mixed iterate restores the last output
// ---------------------------------
void testU3() {
  const Box b(12, 10, 8, 2);
  const LinearMap lin =
      makeLinearMap(peclet::core::test::u1Spectrum(b), std::vector<double>(b.nPad, 1.0));
  // (a) the map returns a NaN
  {
    View<double> x("u3_x", b.nPad);
    AndersonCore acc(oneField(b, x), 5);
    int nMixed = 0;
    std::vector<double> lastOut;
    bool hit = false;
    for (int k = 0; k < 40 && !hit; ++k) {
      const bool mixed = acc.prepare(true);
      nMixed += mixed ? 1 : 0;
      applyLinear(lin, x);
      if (mixed && nMixed == 3) {
        View<double> xv = x;
        const Index i = b.pad(3, 4, 5);
        Kokkos::parallel_for(
            "u3_nan", 1,
            KOKKOS_LAMBDA(const int) { xv(i) = std::numeric_limits<double>::quiet_NaN(); });
        hit = true;
      }
      complete(acc);
      if (!hit)
        lastOut = download(x);
    }
    PECLET_CORE_CHECK(hit);
    const auto restored = download(x);
    bool bitwise = restored.size() == lastOut.size();
    for (std::size_t i = 0; bitwise && i < restored.size(); ++i)
      bitwise = std::memcmp(&restored[i], &lastOut[i], sizeof(double)) == 0;
    std::printf("U3a: NaN at mixed iterate 3 -> status %s (%s), restored bitwise %d\n",
                acc.statusName(), acc.reason().c_str(), bitwise ? 1 : 0);
    PECLET_CORE_CHECK(bitwise);
    PECLET_CORE_CHECK(acc.status() == AndersonCore::Status::Disabled);
    // the next plain step works: it is the plain map of the restored state
    std::vector<double> expect = restored;
    const auto lh = download(lin.lam);
    for (std::size_t i = 0; i < expect.size(); ++i)
      expect[i] = lh[i] * restored[i] + 1.0;
    PECLET_CORE_CHECK(!evaluate(acc, false, [&] { applyLinear(lin, x); }));
    const auto next = download(x);
    double err = 0.0, ref = 0.0;
    for (std::size_t i = 0; i < next.size(); ++i) {
      err = std::max(err, std::abs(next[i] - expect[i]));
      ref = std::max(ref, std::abs(expect[i]));
    }
    // (a device backend may contract λx + c into an FMA, so allow the last bit)
    PECLET_CORE_CHECK(allFinite(next) && err <= 4.0 * std::numeric_limits<double>::epsilon() * ref);
    PECLET_CORE_CHECK(acc.status() == AndersonCore::Status::Disabled);
  }
  // (b) the caller's step throws at the 3rd mixed iterate
  {
    View<double> x("u3b_x", b.nPad);
    AndersonCore acc(oneField(b, x), 5);
    int nMixed = 0;
    std::vector<double> lastOut;
    bool absorbed = false;
    for (int k = 0; k < 40 && !absorbed; ++k) {
      const bool mixed = acc.prepare(true);
      nMixed += mixed ? 1 : 0;
      try {
        applyLinear(lin, x);
        if (mixed && nMixed == 3)
          throw std::runtime_error("synthetic failure");
        complete(acc);
        lastOut = download(x);
      } catch (const std::exception& ex) {
        absorbed = acc.stepFailed(ex.what());
      }
    }
    const auto restored = download(x);
    bool bitwise = absorbed && restored.size() == lastOut.size();
    for (std::size_t i = 0; bitwise && i < restored.size(); ++i)
      bitwise = std::memcmp(&restored[i], &lastOut[i], sizeof(double)) == 0;
    std::printf("U3b: throw at mixed iterate 3 -> absorbed %d, status %s, restored bitwise %d\n",
                absorbed ? 1 : 0, acc.statusName(), bitwise ? 1 : 0);
    PECLET_CORE_CHECK(bitwise);
    PECLET_CORE_CHECK(acc.status() == AndersonCore::Status::Disabled);
    PECLET_CORE_CHECK(acc.reason().find("synthetic failure") != std::string::npos);
    // a failure of a PLAIN step is the caller's own: not absorbed
    View<double> y("u3b_y", b.nPad);
    AndersonCore plain(oneField(b, y), 5);
    plain.prepare(false);
    PECLET_CORE_CHECK(!plain.stepFailed("plain failure"));
    PECLET_CORE_CHECK(plain.status() == AndersonCore::Status::Active);
  }
}

// ---- U4 (rev 2): the core converges on an unstable map and judges nothing ----------------------
// J: the bulk evenly spread on [0, 0.9], one outlier entry, and a 2×2 block on two entries; c = 1.
// U4: outlier 1.02 and a rotation of modulus 1.01 (ρ(J) = 1.02, the plain march diverges). U4b:
// outlier 0.5 and the non-normal block [[0.99, 1], [0, 0.99]] (ρ(J) = 0.99, the plain march
// converges). 400 step(true) calls with no stop on each: status Active after every call, 0
// restarts, residual ≤ 1e-10 within 300 (U4) / 150 (U4b) calls (rev-2 oracle: 162 / 91 calls,
// round-off sensitive). U4 pins the GMRES-like property the design relies on (§2.4): stability
// evidence must come from the caller's plain steps. U4b is the counterexample that removed the
// rev-1 Ritz guard: on this STABLE map it declared "unstable" at call 13 (radius 1.027).
struct U4Result {
  int engagedAt = -1, steps = 0, firstBelow = -1, numRestarts = 0;
  bool alwaysActive = true;
  double minResidual = std::numeric_limits<double>::infinity();
};

U4Result runU4(double outlier, double b00, double b01, double b10, double b11, int calls,
               double target) {
  const Box b(25, 20, 20, 2);
  std::vector<double> lam(b.nPad, 0.0);
  std::vector<Index> innerIdx;
  for (Index i = 0; i < b.nPad; ++i)
    if (b.inner(i))
      innerIdx.push_back(i);
  const Index nBulk = static_cast<Index>(innerIdx.size()) - 3;
  for (Index k = 0; k < nBulk; ++k)
    lam[innerIdx[k]] = 0.9 * static_cast<double>(k) / static_cast<double>(nBulk - 1);
  lam[innerIdx[nBulk]] = outlier;
  LinearMap map = makeLinearMap(lam, std::vector<double>(b.nPad, 1.0));
  map.i0 = innerIdx[nBulk + 1];
  map.i1 = innerIdx[nBulk + 2];
  map.b00 = b00;
  map.b01 = b01;
  map.b10 = b10;
  map.b11 = b11;
  View<double> x("u4_x", b.nPad);
  AndersonCore acc(oneField(b, x), 5);
  U4Result r;
  for (int k = 1; k <= calls; ++k) {
    evaluate(acc, true, [&] { applyLinear(map, x); });
    r.steps = k;
    if (r.engagedAt < 0 && acc.engaged())
      r.engagedAt = k;
    r.alwaysActive = r.alwaysActive && acc.status() == AndersonCore::Status::Active;
    r.minResidual = std::min(r.minResidual, acc.residual());
    if (r.firstBelow < 0 && acc.residual() <= target)
      r.firstBelow = k;
  }
  r.numRestarts = acc.numRestarts();
  return r;
}

void testU4() {
  const double ra = 1.01 * std::cos(0.5), rb = 1.01 * std::sin(0.5);
  const U4Result r = runU4(1.02, ra, -rb, rb, ra, 400, 1e-10);
  std::printf(
      "U4: unstable map (1.02, rotation 1.01): engaged at %d, active after every call %d, "
      "%d restarts, residual <= 1e-10 at call %d, min residual %.3e over %d calls\n",
      r.engagedAt, r.alwaysActive ? 1 : 0, r.numRestarts, r.firstBelow, r.minResidual, r.steps);
  PECLET_CORE_CHECK(r.alwaysActive);
  PECLET_CORE_CHECK(r.numRestarts == 0);
  PECLET_CORE_CHECK(r.firstBelow >= 1 && r.firstBelow <= 300);
}

void testU4b() {
  const U4Result r = runU4(0.5, 0.99, 1.0, 0.0, 0.99, 400, 1e-10);
  std::printf(
      "U4b: stable non-normal map ([[0.99, 1], [0, 0.99]]): engaged at %d, active after every "
      "call %d, %d restarts, residual <= 1e-10 at call %d, min residual %.3e over %d calls\n",
      r.engagedAt, r.alwaysActive ? 1 : 0, r.numRestarts, r.firstBelow, r.minResidual, r.steps);
  PECLET_CORE_CHECK(r.alwaysActive);
  PECLET_CORE_CHECK(r.numRestarts == 0);
  PECLET_CORE_CHECK(r.firstBelow >= 1 && r.firstBelow <= 150);
}

// ---- U5: rank-1 differences — columns drop, γ stays finite
// ---------------------------------------
void testU5() {
  const Box b(10, 8, 6, 2);
  // g(x) = a + v·φ(t(x)), t(x) = ⟨v, x − a⟩/⟨v, v⟩ (inner), φ(t) = 0.9 t + 0.1 sin t + 1: every
  // output lies on the line a + v t, so every difference is a multiple of v.
  std::vector<double> ah(b.nPad, 0.0), vh(b.nPad, 0.0);
  double vv = 0.0;
  for (Index i = 0; i < b.nPad; ++i) {
    ah[i] = 0.5 + 0.25 * std::cos(0.3 * static_cast<double>(i));
    vh[i] = 1.0 + 0.5 * std::sin(0.7 * static_cast<double>(i));
    if (b.inner(i))
      vv += vh[i] * vh[i];
  }
  View<double> a = peclet::core::toDevice(ah, "u5_a"), v = peclet::core::toDevice(vh, "u5_v");
  View<double> x("u5_x", b.nPad);
  Kokkos::deep_copy(x, a);
  auto map = [&] {
    const auto xh = download(x);
    double t = 0.0;
    for (Index i = 0; i < b.nPad; ++i)
      if (b.inner(i))
        t += vh[i] * (xh[i] - ah[i]);
    t /= vv;
    const double phi = 0.9 * t + 0.1 * std::sin(t) + 1.0;
    View<double> xv = x;
    View<const double> av = a, vvw = v;
    Kokkos::parallel_for(
        "u5_map", x.extent(0),
        KOKKOS_LAMBDA(const std::size_t i) { xv(i) = av(i) + vvw(i) * phi; });
  };
  AndersonCore acc(oneField(b, x), 5);
  // Above the noise floor every difference is a multiple of v, so at most one column survives;
  // at round-off the differences are noise (full rank) and the window may refill.
  int maxCols = 0, nMixed = 0, nRank1 = 0;
  bool finite = true;
  for (int k = 0; k < 40; ++k) {
    nMixed += evaluate(acc, true, map) ? 1 : 0;
    if (acc.engaged() && acc.residual() >= AndersonCore::kNoiseFloor) {
      maxCols = std::max(maxCols, acc.numColumns());
      ++nRank1;
    }
    for (double gk : acc.gamma())
      finite = finite && std::isfinite(gk);
    finite = finite && allFinite(download(x));
  }
  std::printf(
      "U5: rank-1 differences: %d mixes, max columns over %d engaged steps above the noise "
      "floor %d, residual %.3e, status %s\n",
      nMixed, nRank1, maxCols, acc.residual(), acc.statusName());
  PECLET_CORE_CHECK(nMixed > 0);
  PECLET_CORE_CHECK(nRank1 >= 3);  // several columns were formed, and dropped
  PECLET_CORE_CHECK(maxCols <= 1);
  PECLET_CORE_CHECK(finite);
  PECLET_CORE_CHECK(acc.status() == AndersonCore::Status::Active);
}

// ---- U6: change detection — an inner write resets, a ghost write does not ----------------------
void testU6() {
  const Box b(12, 10, 8, 2);
  const LinearMap lin =
      makeLinearMap(peclet::core::test::u1Spectrum(b), std::vector<double>(b.nPad, 1.0));
  View<double> x("u6_x", b.nPad);
  AndersonCore acc(oneField(b, x), 5);
  for (int k = 0; k < 8; ++k)
    evaluate(acc, true, [&] { applyLinear(lin, x); });
  PECLET_CORE_CHECK(acc.pending());
  auto poke = [&](Index i) {
    View<double> xv = x;
    Kokkos::parallel_for("u6_poke", 1, KOKKOS_LAMBDA(const int) { xv(i) += 1.0; });
  };
  // ghost write only: no reset, the mix is applied
  poke(0);  // padded entry (0,0,0) is a ghost
  const int resets0 = acc.numResets();
  bool mixed = acc.prepare(true);
  applyLinear(lin, x);
  complete(acc);
  std::printf("U6: ghost write -> mixed %d, resets %d -> %d\n", mixed ? 1 : 0, resets0,
              acc.numResets());
  PECLET_CORE_CHECK(mixed);
  PECLET_CORE_CHECK(acc.numResets() == resets0);
  PECLET_CORE_CHECK(acc.pending());
  // inner write: reset, no mix
  poke(b.pad(4, 5, 3));
  mixed = acc.prepare(true);
  applyLinear(lin, x);
  complete(acc);
  std::printf("U6: inner write -> mixed %d, resets %d -> %d, columns %d\n", mixed ? 1 : 0, resets0,
              acc.numResets(), acc.numColumns());
  PECLET_CORE_CHECK(!mixed);
  PECLET_CORE_CHECK(acc.numResets() == resets0 + 1);
  PECLET_CORE_CHECK(acc.numColumns() == 0);
}

// ---- the §6.3 memory formula, and the descriptor checks -----------------------------------------
void testMemory() {
  const Box b(9, 7, 5, 2);
  for (int nf = 1; nf <= 7; nf += 3)
    for (int m = 1; m <= 8; ++m) {
      AndersonState s;
      s.extent = b.e;
      s.ghost = b.g;
      for (int f = 0; f < nf; ++f) {
        s.fields.push_back(View<double>("mem_f", b.nPad));
        s.roles.push_back((f == 1 || f >= 4) ? AndersonRole::Carried : AndersonRole::Velocity);
      }
      AndersonCore acc(s, m);
      const std::size_t formula = static_cast<std::size_t>(2 * m + 3) *
                                  static_cast<std::size_t>(nf) * 8u *
                                  static_cast<std::size_t>(b.nPad);
      PECLET_CORE_CHECK(acc.memoryBytes() == formula);
      PECLET_CORE_CHECK(AndersonCore::memoryBytesFor(m, nf, b.nPad) == formula);
    }
  std::printf("memory: memoryBytes() == (2m+3)*n_s*8*n_pad for m = 1..8, n_s = 1, 4, 7\n");
  // descriptor validation
  View<double> f("val_f", b.nPad), g("val_g", b.nPad + 1);
  auto throwsInvalid = [](auto&& make) {
    try {
      make();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  PECLET_CORE_CHECK(throwsInvalid([&] { AndersonCore a(oneField(b, f), 0); }));
  PECLET_CORE_CHECK(throwsInvalid([&] { AndersonCore a(oneField(b, f), 9); }));
  PECLET_CORE_CHECK(throwsInvalid([&] { AndersonCore a(oneField(b, f), 5, 0.0); }));
  PECLET_CORE_CHECK(throwsInvalid([&] { AndersonCore a(oneField(b, g), 5); }));
  PECLET_CORE_CHECK(  // the revision-0 Pressure role (= 1) no longer exists
      throwsInvalid([&] { AndersonCore a(oneField(b, f, static_cast<AndersonRole>(1))); }));
}

// ---- the metric: velocity at unit weight, fluid-centred gauge-free pressure at cP² ------------
void testMetric() {
  // Rev 1 (D3): the metric is the velocity at unit weight over every inner entry. From an input
  // x0 the residual is ‖g_u − x0_u‖/‖g_u‖ over inner Velocity entries: ghosts and Carried fields
  // (any values, moved or not) do not enter.
  const Box b(8, 6, 4, 2);
  std::vector<double> u0(b.nPad), uh(b.nPad), ph(b.nPad);
  for (Index i = 0; i < b.nPad; ++i) {
    u0[i] = 0.5 * std::cos(0.13 * static_cast<double>(i));
    uh[i] = std::sin(0.37 * static_cast<double>(i));
    ph[i] = std::cos(0.21 * static_cast<double>(i));
  }
  auto residualOf = [&](double carriedShift, double ghostValue) {
    View<double> u = peclet::core::toDevice(u0, "m_u"), p("m_p", b.nPad);
    AndersonState s;
    s.fields = {u, p};
    s.roles = {AndersonRole::Velocity, AndersonRole::Carried};
    s.extent = b.e;
    s.ghost = b.g;
    AndersonCore acc(s, 3);
    std::vector<double> ug = uh, pg = ph;
    for (Index i = 0; i < b.nPad; ++i) {
      pg[i] += carriedShift;
      if (!b.inner(i))
        ug[i] = ghostValue;
    }
    acc.prepare(false);
    Kokkos::deep_copy(u, peclet::core::toDevice(ug, "m_ug"));
    Kokkos::deep_copy(p, peclet::core::toDevice(pg, "m_pg"));
    complete(acc);
    return acc.residual();
  };
  double rr = 0.0, uu = 0.0;
  for (Index i = 0; i < b.nPad; ++i)
    if (b.inner(i)) {
      rr += (uh[i] - u0[i]) * (uh[i] - u0[i]);
      uu += uh[i] * uh[i];
    }
  const double ref = std::sqrt(rr / uu);
  const double r0 = residualOf(0.0, 0.0);
  const double r1 = residualOf(123.0, -55.0);
  std::printf("metric: residual %.15f (host reference %.15f), Carried and ghosts changed %.15f\n",
              r0, ref, r1);
  PECLET_CORE_CHECK(std::abs(r0 / ref - 1.0) < 1e-13);
  PECLET_CORE_CHECK(r1 == r0);
}

// ---- host linear algebra ------------------------------------------------------------------------
void testLinearAlgebra() {
  // Jacobi: V Λ Vᵀ reproduces a symmetric 6×6 matrix.
  const int n = 6;
  da::Mat a = {}, a0 = {}, v = {};
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j)
      a[i][j] = a[j][i] = (i == j ? 4.0 + i : 1.0 / (1.0 + i + j)) + 0.1 * std::sin(i * 3.0 + j);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      a0[i][j] = a[i][j];
  double lam[da::kMaxWindow];
  da::jacobiEigen(n, a, v, lam);
  double err = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k)
        s += v[i][k] * lam[k] * v[j][k];
      err = std::max(err, std::abs(s - a0[i][j]));
    }
  std::printf("jacobi: max |V L V^T - A| = %.3e\n", err);
  PECLET_CORE_CHECK(err < 1e-13);
}

// ---- U10 (rev 1): Carried fields — mixed, never measured
// ----------------------------------------- Velocity + Carried. For 10 steps the map moves both
// (velocity: x ← Λx + c; Carried: an affine map whose output satisfies Σ_inner p = S); afterwards
// it moves only the Carried field. The residual is then 0, and every mixed Carried state keeps
// Σ_inner p = S (U2 for Carried).
void testU10() {
  const Box b(16, 12, 10, 2);
  const LinearMap lu =
      makeLinearMap(peclet::core::test::u1Spectrum(b), std::vector<double>(b.nPad, 1.0));
  std::vector<double> lam(b.nPad, 0.0), c(b.nPad, 0.0), mh(b.nPad, 0.0);
  for (Index i = 0; i < b.nPad; ++i)
    if (b.inner(i)) {
      lam[i] = 0.99 * std::fmod(0.618034 * static_cast<double>(i), 1.0);
      c[i] = 1.0 + 0.5 * std::sin(0.1 * static_cast<double>(i));
      mh[i] = 1.0;
    }
  const LinearMap lp = makeLinearMap(lam, c);
  View<double> mask = peclet::core::toDevice(mh, "u10_mask");
  const double S = 3.0 * static_cast<double>(b.nInner);
  const double invN = 1.0 / static_cast<double>(b.nInner);
  auto innerSum = [&](View<const double> v) {
    const auto h = download(v);
    long double sum = 0.0L;
    for (Index i = 0; i < b.nPad; ++i)
      if (b.inner(i))
        sum += h[i];
    return static_cast<double>(sum);
  };
  View<double> u("u10_u", b.nPad), p("u10_p", b.nPad);
  AndersonState st;
  st.fields = {u, p};
  st.roles = {AndersonRole::Velocity, AndersonRole::Carried};
  st.extent = b.e;
  st.ghost = b.g;
  AndersonCore acc(st, 5);
  double worstMix = 0.0, worstRes = 0.0;
  int nMixed = 0;
  for (int k = 1; k <= 40; ++k) {
    if (acc.prepare(true)) {
      worstMix = std::max(worstMix, std::abs(innerSum(p) - S));
      ++nMixed;
    }
    if (k <= 10)
      applyLinear(lu, u);
    applyLinear(lp, p);
    const double shift = (innerSum(p) - S) * invN;
    View<double> pv = p;
    View<const double> mk = mask;
    Kokkos::parallel_for(
        "u10_shift", p.extent(0), KOKKOS_LAMBDA(const std::size_t i) { pv(i) -= shift * mk(i); });
    complete(acc);
    if (k > 10)
      worstRes = std::max(worstRes, acc.residual());
  }
  std::printf(
      "U10: %d mixed states, max |Σp − S|/|S| over them = %.3e; max residual after step "
      "10 = %.3e, status %s\n",
      nMixed, worstMix / S, worstRes, acc.statusName());
  PECLET_CORE_CHECK(nMixed >= 5);
  PECLET_CORE_CHECK(worstRes == 0.0);
  PECLET_CORE_CHECK(worstMix <= 1e-14 * std::abs(S));
  PECLET_CORE_CHECK(acc.status() == AndersonCore::Status::Active);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  testLinearAlgebra();
  testMemory();
  runTest("metric", testMetric);
  runTest("U1a", testU1a);
  runTest("U1b", testU1b);
  runTest("U1c", testU1c);
  runTest("U2", testU2);
  runTest("U3", testU3);
  runTest("U4", testU4);
  runTest("U4b", testU4b);
  runTest("U5", testU5);
  runTest("U6", testU6);
  runTest("U10", testU10);
  PECLET_CORE_RETURN_TEST_RESULT();
}

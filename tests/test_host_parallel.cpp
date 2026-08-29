// Unit test for the host-parallel shim (docs/amr_setup_parallel_plan.md rung 0.5).
//
// The shim's whole contract is "bitwise identical to the serial loop, for any thread count".
// This exercises both halves against an explicit serial reference IN THE SAME PROCESS:
//
//   1. hostParFor over a synthetic disjoint-write fill whose per-element value is a chain of
//      transcendental + fma ops (i.e. the kind of arithmetic the real builders do, so a compiler
//      that reassociated something under the parallel policy would show up). Compared with
//      memcmp, not a tolerance.
//   2. hostParScan over integer counts, giving CSR row offsets — the rung-4 pattern. An integer
//      scan is exact regardless of how the partial sums are combined, and that is the reason the
//      two-pass builders can assign row order in parallel and still reproduce the serial CSR.
//   3. hostParFor over the scan's offsets: each i fills its own [start[i], start[i+1]) slice, so
//      the concatenated payload must equal the serial append order (the row-order contract).
//
// When Kokkos is compiled in and initialized, 1-3 really run parallel (thread count from
// OMP_NUM_THREADS); when it is not, they run the serial fallback and the test still asserts the
// reference semantics. Both are worth gating.
#include <cmath>
#include <cstring>
#include <vector>

#include "peclet/core/common/host_parallel.hpp"
#include "test_util.hpp"

#if PECLET_CORE_HOST_PARALLEL_KOKKOS
#include <Kokkos_Core.hpp>
#endif

namespace {

using peclet::core::hostParFor;
using peclet::core::hostParScan;
using peclet::core::Index;

// A deliberately fussy per-element expression: several libm calls plus an fma, so the value
// depends on the exact instruction sequence the compiler picked. Same function for both paths.
double payload(Index i) {
  const double x = 1e-3 * static_cast<double>(i) + 0.25;
  const double a = std::sin(x) * std::log(1.0 + x);
  return std::fma(a, std::sqrt(x), std::cos(2.0 * x) / (1.0 + a * a));
}

Index countOf(Index i) { return (i * 7919) % 13; }  // 0..12, irregular — a realistic row width

void run() {
  const Index n = 200003;  // prime-ish, so the last chunk of any schedule is ragged

  // ---- 1. disjoint-write fill -----------------------------------------------------------------
  std::vector<double> ref(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i)
    ref[static_cast<std::size_t>(i)] = payload(i);

  std::vector<double> got(static_cast<std::size_t>(n), 0.0);
  hostParFor(n, [&](Index i) { got[static_cast<std::size_t>(i)] = payload(i); });
  PECLET_CORE_CHECK(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(double)) == 0);

  // ---- 2. integer exclusive scan ---------------------------------------------------------------
  std::vector<Index> cnt(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i)
    cnt[static_cast<std::size_t>(i)] = countOf(i);

  std::vector<Index> startRef(static_cast<std::size_t>(n) + 1, 0);
  for (Index i = 0; i < n; ++i)
    startRef[static_cast<std::size_t>(i) + 1] =
        startRef[static_cast<std::size_t>(i)] + cnt[static_cast<std::size_t>(i)];

  std::vector<Index> start(static_cast<std::size_t>(n) + 1, -1);
  hostParScan(n, [&](Index i, Index& acc, bool final) {
    const Index c = cnt[static_cast<std::size_t>(i)];
    if (final)
      start[static_cast<std::size_t>(i)] = acc;
    acc += c;
  });
  start[static_cast<std::size_t>(n)] = startRef[static_cast<std::size_t>(n)];
  PECLET_CORE_CHECK(std::memcmp(startRef.data(), start.data(), start.size() * sizeof(Index)) == 0);

  // ---- 3. two-pass fill at the scanned offsets (the CSR row-order contract) ---------------------
  const Index nz = startRef[static_cast<std::size_t>(n)];
  PECLET_CORE_CHECK(nz > 0);
  std::vector<double> csrRef;
  csrRef.reserve(static_cast<std::size_t>(nz));
  for (Index i = 0; i < n; ++i)  // the serial APPEND order this must reproduce
    for (Index k = 0; k < cnt[static_cast<std::size_t>(i)]; ++k)
      csrRef.push_back(payload(i) + static_cast<double>(k));

  std::vector<double> csr(static_cast<std::size_t>(nz), 0.0);
  hostParFor(n, [&](Index i) {
    const Index base = start[static_cast<std::size_t>(i)];
    for (Index k = 0; k < cnt[static_cast<std::size_t>(i)]; ++k)
      csr[static_cast<std::size_t>(base + k)] = payload(i) + static_cast<double>(k);
  });
  PECLET_CORE_CHECK_EQ((long long)csrRef.size(), (long long)csr.size());
  PECLET_CORE_CHECK(std::memcmp(csrRef.data(), csr.data(), csr.size() * sizeof(double)) == 0);

  std::printf("host_parallel: backend %s, n=%lld nz=%lld\n",
              peclet::core::hostParallelActive() ? "kokkos-host" : "serial-fallback", (long long)n,
              (long long)nz);
}

}  // namespace

int main(int argc, char** argv) {
#if PECLET_CORE_HOST_PARALLEL_KOKKOS
  Kokkos::initialize(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  run();
#if PECLET_CORE_HOST_PARALLEL_KOKKOS
  Kokkos::finalize();
#endif
  PECLET_CORE_RETURN_TEST_RESULT();
}

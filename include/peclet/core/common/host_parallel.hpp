// core — the host-parallel shim for the shared HOST builders (setup-parallel plan D1', rung 0.5).
//
// The AMR setup builders (amr/momentum.hpp, amr/poisson.hpp, amr/cf_scheme.hpp,
// amr/ghost_projection_sampled.hpp) are host-by-design: their weights are shared VERBATIM with the
// serial host oracle (parity by construction), so they cannot move to a device kernel without
// breaking that contract. They are also the whole of `AmrFlow::setSolid`'s cost at bed scale.
// Rather than sprinkle OpenMP pragmas (a second parallel model, and a dead end for the later
// device-assembly campaign), the suite's ONE parallel model — Kokkos — is used over its HOST
// execution space. This header is the single place that dispatch lives.
//
//   hostParFor(n, body)          — body(i) for i in [0, n), any order. The caller MUST guarantee
//                                  the writes are DISJOINT across i (each iteration touches only
//                                  its own slots). Under that contract the result is bitwise
//                                  identical to the serial loop for ANY thread count and ANY
//                                  schedule — determinism by construction, never by scheduling
//                                  policy (plan D2).
//   hostParScan(n, body)         — exclusive scan, body(i, update, final) with `update` an Index.
//                                  INTEGER only: an integer scan is exact regardless of the order
//                                  the partial sums are combined, which is what lets the two-pass
//                                  CSR builders assign row/slot offsets in parallel and still get
//                                  today's serial row order.
//
// Two deliberate properties:
//
//   * Kokkos presence is detected per BUILD TREE (`__has_include(<Kokkos_Core.hpp>)`), not per
//     include order. Keying on KOKKOS_VERSION — as common/portable.hpp does for PECLET_HD, where
//     the two branches are ABI-compatible — would make this header's inline functions resolve
//     differently in TUs that reach it before vs after Kokkos_Core.hpp: an ODR violation. The
//     header pulls Kokkos in itself so every TU in a Kokkos build agrees.
//   * `Kokkos::is_initialized()` is checked at the call. Several host-oracle ctests
//     (test_amr_drag, test_amr_flow, test_amr_cf_vector, ...) link a Kokkos-enabled tpx_core but
//     never call Kokkos::initialize — they exercise the oracle, not the device. Those must keep
//     working, so an uninitialized Kokkos falls back to the serial loop instead of aborting.
//
// Thread count comes from the Kokkos host backend (OpenMP => OMP_NUM_THREADS). With OpenMP now in
// the default prefix, ALWAYS pin it for test batteries: `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`.
//
// Design: docs/amr_setup_parallel_plan.md (D1' piece 3, D2).
#ifndef PECLET_CORE_COMMON_HOST_PARALLEL_HPP
#define PECLET_CORE_COMMON_HOST_PARALLEL_HPP

#if !defined(PECLET_CORE_HOST_PARALLEL_KOKKOS)
#if defined(__has_include)
#if __has_include(<Kokkos_Core.hpp>)
#define PECLET_CORE_HOST_PARALLEL_KOKKOS 1
#else
#define PECLET_CORE_HOST_PARALLEL_KOKKOS 0
#endif
#elif defined(KOKKOS_VERSION)
#define PECLET_CORE_HOST_PARALLEL_KOKKOS 1
#else
#define PECLET_CORE_HOST_PARALLEL_KOKKOS 0
#endif
#endif  // !defined(PECLET_CORE_HOST_PARALLEL_KOKKOS)

// common/view.hpp, not <Kokkos_Core.hpp> directly: several AMR headers gate their device mirrors on
// `#ifdef KOKKOS_INLINE_FUNCTION` and, inside those blocks, use peclet::core::View / toDevice
// without including view.hpp themselves (they rely on the includer having pulled it in first).
// Bringing Kokkos into a TU therefore has to bring View along, or those blocks light up
// undefined in the host-oracle TUs that reach this header.
#if PECLET_CORE_HOST_PARALLEL_KOKKOS
#include "peclet/core/common/view.hpp"
#endif

#include <utility>

#include "peclet/core/common/types.hpp"

namespace peclet::core {

/// Is the parallel backend actually live for this process? (Kokkos compiled in AND initialized.)
/// Exposed so a builder can assert/report; the shim itself already falls back transparently.
inline bool hostParallelActive() {
#if PECLET_CORE_HOST_PARALLEL_KOKKOS
  return Kokkos::is_initialized();
#else
  return false;
#endif
}

/// Parallel `for (Index i = 0; i < n; ++i) body(i);` over the Kokkos HOST execution space.
/// CONTRACT: iterations must write disjoint memory (see the header note) — the shim does not and
/// cannot check it, and a violation is a data race, not merely a nondeterminism.
template <class Body>
inline void hostParFor(Index n, Body&& body) {
  if (n <= 0)
    return;
#if PECLET_CORE_HOST_PARALLEL_KOKKOS
  if (Kokkos::is_initialized()) {
    using HostExec = Kokkos::DefaultHostExecutionSpace;
    Kokkos::parallel_for(
        "peclet::core::hostParFor", Kokkos::RangePolicy<HostExec>(0, n),
        [&body](const Index i) { body(i); });
    Kokkos::fence("peclet::core::hostParFor");
    return;
  }
#endif
  for (Index i = 0; i < n; ++i)
    body(i);
}

/// Exclusive integer scan over the host space: `body(i, update, final)` where `update` carries the
/// running sum of everything BEFORE `i` when `final` is true. Semantics are Kokkos
/// parallel_scan's, and the serial fallback reproduces them exactly.
///
/// Typical use (two-pass CSR): pass counts in, read row offsets out.
///   hostParScan(n, [&](Index i, Index& acc, bool final) {
///     const Index c = count[i];
///     if (final) start[i] = acc;
///     acc += c;
///   });
template <class Body>
inline void hostParScan(Index n, Body&& body) {
  if (n <= 0)
    return;
#if PECLET_CORE_HOST_PARALLEL_KOKKOS
  if (Kokkos::is_initialized()) {
    using HostExec = Kokkos::DefaultHostExecutionSpace;
    Kokkos::parallel_scan(
        "peclet::core::hostParScan", Kokkos::RangePolicy<HostExec>(0, n),
        [&body](const Index i, Index& update, const bool final) { body(i, update, final); });
    Kokkos::fence("peclet::core::hostParScan");
    return;
  }
#endif
  Index update = 0;
  for (Index i = 0; i < n; ++i)
    body(i, update, true);
}

}  // namespace peclet::core

#endif  // PECLET_CORE_COMMON_HOST_PARALLEL_HPP

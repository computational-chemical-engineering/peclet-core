// transport-core — device CSR-fill primitive (S1 / D1): the reusable count → scan → fill backbone
// for assembling a face-CSR sparse operator on the device.
//
// Every AMR operator (the FV pressure Laplacian, the cut-cell momentum operator, the face-geometry
// tables) is a per-cell list of faces with a neighbour index and a coefficient. Assembling such a CSR
// in parallel is the standard three-step pattern:
//   1. count   — each cell counts its faces                       (parallel_for)
//   2. scan    — exclusive prefix-sum the counts into row offsets (parallel_scan)
//   3. fill    — each cell writes its faces into its OWN slice    (parallel_for, atomic-free)
// Step 3 is deterministic (own-slice, no atomics, fixed emit order), so on the OpenMP backend the
// device-assembled CSR is bit-for-bit identical to the host serial assembler — the suite's
// anti-drift contract. (GPU is tolerance-not-bit-exact by the documented FMA convention.)
//
// The per-cell face traversal is supplied by the caller as an `Emit` functor with a templated
// `operator()(Index i, Sink& s)` that calls `s(neighbourLeaf, coef)` once per face in a fixed order.
// The same functor drives both the count pass (a counting sink) and the fill pass (a writing sink),
// so the emit order is guaranteed identical between the two — exactly as the host assembler reuses one
// `forEachFaceNeighbor` lambda for both passes.
//
// Requires a Kokkos build; included only by the device-assembly headers (themselves PECLET_CORE_HAVE_MORTON).
#ifndef PECLET_CORE_AMR_CSR_HPP
#define PECLET_CORE_AMR_CSR_HPP

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::core::amr {

/// Sink passed to the emit functor during the COUNT pass: just tallies faces.
struct CsrCountSink {
  Index count = 0;
  KOKKOS_INLINE_FUNCTION void operator()(Index /*nbr*/, double /*coef*/) { ++count; }
};

/// Sink passed to the emit functor during the FILL pass: writes each face into the cell's own slice
/// [start_i, start_{i+1}) at the running cursor `k` (no atomics — deterministic).
struct CsrFillSink {
  View<Index> nbr;
  View<double> coef;
  Index k = 0;
  KOKKOS_INLINE_FUNCTION void operator()(Index j, double c) {
    nbr(k) = j;
    coef(k) = c;
    ++k;
  }
};

/// CSR row offsets (size n+1) from a per-cell count functor, via one exclusive prefix scan, with the
/// total written to `nTotal`. The standalone scan half of S1 — for assemblers whose per-face payload is
/// richer than (nbr, coef) (e.g. the face-geometry tables), which then run their own own-slice fill
/// over these offsets. `CountFn` is device-callable: KOKKOS Index operator()(Index i) const.
template <class CountFn>
View<Index> scanOffsets(Index n, const CountFn& countFn, Index& nTotal) {
  View<Index> start(Kokkos::view_alloc("peclet::core::amr::csr_off", Kokkos::WithoutInitializing),
                    static_cast<std::size_t>(n) + 1);
  Kokkos::parallel_scan(
      "peclet::core::amr::csr_off_scan", n + 1,
      KOKKOS_LAMBDA(const Index i, Index& partial, const bool final_pass) {
        const Index c = (i < n) ? countFn(i) : Index(0);
        if (final_pass) start(i) = partial;
        partial += c;
      });
  Kokkos::deep_copy(nTotal, Kokkos::subview(start, n));
  return start;
}

/// An assembled face-CSR: row offsets (size n+1), neighbour index + coefficient per face (size nFaces).
struct Csr {
  View<Index> start;   ///< CSR row offsets, size n+1; start(n) == nFaces
  View<Index> nbr;     ///< neighbour leaf per face, size nFaces
  View<double> coef;   ///< coefficient per face, size nFaces
  Index nFaces = 0;
};

/// Build a face-CSR on device from a per-cell `emit` functor.
///
/// `Emit` must be a trivially-copyable, device-callable object with:
///     template <class Sink> KOKKOS_INLINE_FUNCTION void operator()(Index i, Sink& s) const;
/// calling `s(neighbourLeaf, coef)` once per face of cell `i`, in a fixed deterministic order.
///
/// The count is folded directly into the prefix scan (each cell's traversal runs once in the scan and
/// once in the fill — the same two traversals the host assembler does). Offsets are written by the
/// scan's `final` pass; the fill writes each cell's own slice with no atomics.
template <class Emit>
Csr buildFaceCsr(Index n, const Emit& emit) {
  Csr csr;
  View<Index> start(Kokkos::view_alloc("peclet::core::amr::csr_start", Kokkos::WithoutInitializing),
                    static_cast<std::size_t>(n) + 1);
  Kokkos::parallel_scan(
      "peclet::core::amr::csr_scan", n + 1,
      KOKKOS_LAMBDA(const Index i, Index& partial, const bool final_pass) {
        Index c = 0;
        if (i < n) {
          CsrCountSink s;
          emit(i, s);
          c = s.count;
        }
        if (final_pass) start(i) = partial;  // exclusive prefix: offset before adding this cell
        partial += c;
      });
  Kokkos::deep_copy(csr.nFaces, Kokkos::subview(start, n));
  csr.start = start;
  csr.nbr = View<Index>(Kokkos::view_alloc("peclet::core::amr::csr_nbr", Kokkos::WithoutInitializing),
                        static_cast<std::size_t>(csr.nFaces));
  csr.coef = View<double>(Kokkos::view_alloc("peclet::core::amr::csr_coef", Kokkos::WithoutInitializing),
                          static_cast<std::size_t>(csr.nFaces));
  View<Index> nbr = csr.nbr;
  View<double> coef = csr.coef;
  Kokkos::parallel_for(
      "peclet::core::amr::csr_fill", n, KOKKOS_LAMBDA(const Index i) {
        CsrFillSink sink{nbr, coef, start(i)};
        emit(i, sink);
      });
  return csr;
}

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_AMR_CSR_HPP

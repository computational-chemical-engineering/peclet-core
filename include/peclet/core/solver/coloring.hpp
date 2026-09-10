// core — greedy graph colouring of a face CSR for race-free multicolour Gauss–Seidel sweeps.
//
// Lifted verbatim from the AMR tree (peclet/core/amr/momentum.hpp, 2026-09-10; QUALITY_PLAN G.2):
// on a 2:1-graded octree or a Voronoi-cell mesh the face-adjacency graph needs a general greedy
// colouring (~6–8 colours); a colour is a set of cells with no shared face, so all of one colour
// update in parallel reading the already-updated other colours — a true GS sweep, deterministic
// (fixed cell order ⇒ fixed colouring). Mesh-agnostic: operates on a face CSR, no octree types.
// Consumers: the AMR momentum multigrid (peclet-amr) and voro's mesh optimiser.
#ifndef PECLET_CORE_SOLVER_COLORING_HPP
#define PECLET_CORE_SOLVER_COLORING_HPP

#include <cstddef>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::core::solver {

/// A graph colouring of a face CSR: cells grouped by colour. `hStart` (host, size nColors+1) slices
/// `idx` (device, cells in colour order). Rebuilt when the connectivity changes (adapt / re-tess).
struct Coloring {
  std::vector<Index> hStart;
  View<Index> idx;
  int nColors = 1;
};

/// Greedy colouring of the face CSR (`start`/`nbr`, host): each cell gets the smallest colour not
/// used by any face neighbour, so cells of one colour share no edge (race-free parallel GS sweep).
/// Deterministic from the natural cell order.
///
/// The adjacency is **symmetrised** first (undirected: i conflicts with j if i∈nbr(j) OR j∈nbr(i)).
/// The assembled cut-cell operator's CSR can be structurally *asymmetric* — the ξ-polynomial
/// Dirichlet overlay adds extrapolation entries a cut cell references but its target doesn't
/// reference back — and colouring only the outgoing edges would then leave two mutually-adjacent
/// cells the same colour, a data race that makes the GS sweep a non-deterministic (inconsistent)
/// operator and silently breaks the BiCGStab it preconditions (false convergence to NaN at scale).
/// Symmetrising is the correctness guard; it costs one O(nnz) host pass at build/adapt time.
inline Coloring greedyColoring(const std::vector<Index>& start, const std::vector<Index>& nbr,
                               Index n) {
  // Build the symmetric (undirected) adjacency in CSR form.
  std::vector<Index> deg(static_cast<std::size_t>(n) + 1, 0);
  for (Index i = 0; i < n; ++i)
    for (Index k = start[static_cast<std::size_t>(i)]; k < start[static_cast<std::size_t>(i) + 1];
         ++k) {
      ++deg[static_cast<std::size_t>(i) + 1];
      ++deg[static_cast<std::size_t>(nbr[static_cast<std::size_t>(k)]) + 1];
    }
  for (Index i = 0; i < n; ++i)
    deg[static_cast<std::size_t>(i) + 1] += deg[static_cast<std::size_t>(i)];
  std::vector<Index> aStart(deg);  // copy of the offsets
  std::vector<Index> aNbr(static_cast<std::size_t>(deg[static_cast<std::size_t>(n)]));
  std::vector<Index> acur(deg.begin(), deg.end() - 1);
  for (Index i = 0; i < n; ++i)
    for (Index k = start[static_cast<std::size_t>(i)]; k < start[static_cast<std::size_t>(i) + 1];
         ++k) {
      const Index j = nbr[static_cast<std::size_t>(k)];
      aNbr[static_cast<std::size_t>(acur[static_cast<std::size_t>(i)]++)] = j;
      aNbr[static_cast<std::size_t>(acur[static_cast<std::size_t>(j)]++)] = i;
    }
  std::vector<int> color(static_cast<std::size_t>(n), -1);
  std::vector<int> stamp;  // stamp[c]==i ⇒ colour c forbidden for cell i (avoids per-cell clears)
  int nColors = 1;
  for (Index i = 0; i < n; ++i) {
    for (Index k = aStart[static_cast<std::size_t>(i)]; k < aStart[static_cast<std::size_t>(i) + 1];
         ++k) {
      const int nc = color[static_cast<std::size_t>(aNbr[static_cast<std::size_t>(k)])];
      if (nc >= 0) {
        if (static_cast<std::size_t>(nc) >= stamp.size())
          stamp.resize(static_cast<std::size_t>(nc) + 1, -1);
        stamp[static_cast<std::size_t>(nc)] = static_cast<int>(i);
      }
    }
    int c = 0;
    while (c < static_cast<int>(stamp.size()) &&
           stamp[static_cast<std::size_t>(c)] == static_cast<int>(i))
      ++c;
    color[static_cast<std::size_t>(i)] = c;
    if (c + 1 > nColors)
      nColors = c + 1;
  }
  Coloring col;
  col.nColors = nColors;
  col.hStart.assign(static_cast<std::size_t>(nColors) + 1, 0);
  for (Index i = 0; i < n; ++i)
    ++col.hStart[static_cast<std::size_t>(color[static_cast<std::size_t>(i)]) + 1];
  for (int c = 0; c < nColors; ++c)
    col.hStart[static_cast<std::size_t>(c) + 1] += col.hStart[static_cast<std::size_t>(c)];
  std::vector<Index> idx(static_cast<std::size_t>(n));
  std::vector<Index> cur(col.hStart.begin(), col.hStart.end() - 1);
  for (Index i = 0; i < n; ++i) {
    const int c = color[static_cast<std::size_t>(i)];
    idx[static_cast<std::size_t>(cur[static_cast<std::size_t>(c)]++)] = i;
  }
  col.idx = toDevice(idx, "gs_coloring");
  return col;
}

}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_COLORING_HPP

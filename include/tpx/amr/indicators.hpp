// transport-core — solution-adaptive refinement indicators for a BlockOctree.
//
// Geometry-based refinement lives in refine.hpp (refineToSdf). This header adds
// *solution*-based error indicators: per-leaf scalars derived from a leaf field
// that flag where the mesh should be refined (large indicator) or may be coarsened
// (small indicator). They drive the dynamic-AMR cycle in adapt.hpp.
//
// The headline criterion is the Löhner (1987) indicator — a dimensionless,
// normalized second difference that is the standard robust feature detector for
// AMR (it tracks curvature / steep fronts and self-normalizes so a single
// threshold works across scales). A raw second-derivative magnitude is also
// provided for when an un-normalized curvature measure is wanted.
//
// Neighbour values come from BlockOctree::faceNeighbor (the covering same/coarser/
// corner-finer neighbour under 2:1 balance); a face with no neighbour (block /
// domain edge) contributes nothing on that axis.
//
// Header-only, guarded by TPX_HAVE_MORTON.
#ifndef TPX_AMR_INDICATORS_HPP
#define TPX_AMR_INDICATORS_HPP

#ifdef TPX_HAVE_MORTON

#include <cmath>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/common/types.hpp"

namespace tpx::amr {

/// Löhner normalized second-difference indicator E_i ∈ [0,1], per leaf, for scalar
/// `u` (indexed by leaf slot). Along each axis with both neighbours present:
///   Δ²   = u₊ − 2u_i + u₋
///   norm = |u₊ − u_i| + |u_i − u₋| + ε(|u₊| + 2|u_i| + |u₋|)
/// and E_i = sqrt( Σ_axis Δ²² / Σ_axis norm² ). The ε term (filter) suppresses
/// refinement on ripples whose amplitude is at the noise level (typical ε≈0.01).
/// Refine where E_i exceeds a threshold (~0.2–0.5); coarsen where it is small.
template <int Dim, unsigned Bits>
std::vector<double> lohnerIndicator(const BlockOctree<Dim, Bits>& t,
                                    const std::vector<double>& u, double eps = 0.01) {
  const Index n = t.numLeaves();
  std::vector<double> e(static_cast<std::size_t>(n), 0.0);
  for (Index i = 0; i < n; ++i) {
    const double ui = u[static_cast<std::size_t>(i)];
    double num2 = 0.0, den2 = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const Index jp = t.faceNeighbor(i, axis, +1);
      const Index jm = t.faceNeighbor(i, axis, -1);
      if (jp < 0 || jm < 0) continue;
      const double up = u[static_cast<std::size_t>(jp)];
      const double um = u[static_cast<std::size_t>(jm)];
      const double d2 = up - 2.0 * ui + um;
      const double nrm = std::fabs(up - ui) + std::fabs(ui - um) +
                         eps * (std::fabs(up) + 2.0 * std::fabs(ui) + std::fabs(um));
      num2 += d2 * d2;
      den2 += nrm * nrm;
    }
    e[static_cast<std::size_t>(i)] = (den2 > 0.0) ? std::sqrt(num2 / den2) : 0.0;
  }
  return e;
}

/// Raw second-derivative magnitude indicator, per leaf: the L2 norm over axes of the
/// undivided second difference |u₊ − 2u_i + u₋| (so ∝ |∂²u|·h²). Un-normalized — the
/// threshold is problem-dependent; prefer `lohnerIndicator` unless a dimensional
/// curvature is specifically wanted.
template <int Dim, unsigned Bits>
std::vector<double> secondDiffIndicator(const BlockOctree<Dim, Bits>& t,
                                        const std::vector<double>& u) {
  const Index n = t.numLeaves();
  std::vector<double> e(static_cast<std::size_t>(n), 0.0);
  for (Index i = 0; i < n; ++i) {
    const double ui = u[static_cast<std::size_t>(i)];
    double s = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const Index jp = t.faceNeighbor(i, axis, +1);
      const Index jm = t.faceNeighbor(i, axis, -1);
      if (jp < 0 || jm < 0) continue;
      const double d2 = u[static_cast<std::size_t>(jp)] - 2.0 * ui + u[static_cast<std::size_t>(jm)];
      s += d2 * d2;
    }
    e[static_cast<std::size_t>(i)] = std::sqrt(s);
  }
  return e;
}

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_INDICATORS_HPP

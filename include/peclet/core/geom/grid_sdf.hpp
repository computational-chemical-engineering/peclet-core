/// @file grid_sdf.hpp
/// @brief Sampled (grid) signed-distance field with trilinear interpolation.
///
/// This is how real geometry enters the Eulerian/Lagrangian solvers: an SDF sampled on a regular
/// grid (x-fastest, matching the suite indexing and `flow`/`dem`'s VTI fields). Values keep the
/// suite sign convention (negative inside solid). Out-of-domain queries clamp to the nearest
/// in-domain sample. Satisfies the same `peclet::core::geom::Sdf` concept as the analytic
/// primitives, so solvers consume analytic and sampled geometry through one interface.
#ifndef PECLET_CORE_GEOM_GRID_SDF_HPP
#define PECLET_CORE_GEOM_GRID_SDF_HPP

#include <algorithm>
#include <vector>

#include "peclet/core/common/types.hpp"

namespace peclet::core::geom {

/// A signed-distance field sampled on a regular axis-aligned grid (negative inside solid).
///
/// Models the `peclet::core::geom::Sdf` concept via `eval()`. Storage is x-fastest, matching the
/// suite indexing convention and the VTI fields produced by `flow`/`dem`.
struct GridSdf {
  std::vector<float> values;  ///< Sample values, x-fastest: idx = i + j*nx + k*nx*ny.
  IVec<3> dims{};             ///< Sample count per axis (nx, ny, nz).
  Vec<3> origin{};            ///< World position of sample (0,0,0).
  Vec<3> spacing{1, 1, 1};    ///< World-space distance between samples per axis.

  /// Raw sample lookup at integer grid index (i,j,k); no bounds checking.
  double at(Index i, Index j, Index k) const {
    return static_cast<double>(values[i + dims[0] * (j + dims[1] * k)]);
  }

  /// Trilinearly-interpolated signed distance at world point @p p. Queries outside the sampled box
  /// clamp to the nearest in-domain sample.
  double eval(const Vec<3>& p) const {
    double g[3];
    Index i0[3], i1[3];
    double f[3];
    for (int d = 0; d < 3; ++d) {
      double c = (p[d] - origin[d]) / spacing[d];
      c = std::clamp(c, 0.0, static_cast<double>(dims[d] - 1));
      i0[d] = static_cast<Index>(std::floor(c));
      i1[d] = std::min<Index>(i0[d] + 1, dims[d] - 1);
      f[d] = c - static_cast<double>(i0[d]);
    }
    double c000 = at(i0[0], i0[1], i0[2]), c100 = at(i1[0], i0[1], i0[2]);
    double c010 = at(i0[0], i1[1], i0[2]), c110 = at(i1[0], i1[1], i0[2]);
    double c001 = at(i0[0], i0[1], i1[2]), c101 = at(i1[0], i0[1], i1[2]);
    double c011 = at(i0[0], i1[1], i1[2]), c111 = at(i1[0], i1[1], i1[2]);
    double c00 = c000 * (1 - f[0]) + c100 * f[0];
    double c10 = c010 * (1 - f[0]) + c110 * f[0];
    double c01 = c001 * (1 - f[0]) + c101 * f[0];
    double c11 = c011 * (1 - f[0]) + c111 * f[0];
    double c0 = c00 * (1 - f[1]) + c10 * f[1];
    double c1 = c01 * (1 - f[1]) + c11 * f[1];
    return c0 * (1 - f[2]) + c1 * f[2];
  }
};

/// Sample any analytic SDF onto a grid, producing a GridSdf (e.g. to bake geometry for a solver).
template <typename S>
GridSdf sample(const S& shape, IVec<3> dims, Vec<3> origin, Vec<3> spacing) {
  GridSdf g;
  g.dims = dims;
  g.origin = origin;
  g.spacing = spacing;
  g.values.resize(static_cast<std::size_t>(dims[0]) * dims[1] * dims[2]);
  for (Index k = 0; k < dims[2]; ++k)
    for (Index j = 0; j < dims[1]; ++j)
      for (Index i = 0; i < dims[0]; ++i) {
        Vec<3> p{origin[0] + i * spacing[0], origin[1] + j * spacing[1],
                 origin[2] + k * spacing[2]};
        g.values[i + dims[0] * (j + dims[1] * k)] = static_cast<float>(shape.eval(p));
      }
  return g;
}

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_GRID_SDF_HPP

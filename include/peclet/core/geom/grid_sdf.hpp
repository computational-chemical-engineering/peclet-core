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
#include "peclet/core/geom/scene.hpp"

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

  /// Descriptor form of this field, for the shared sampler. Built from `spacing` once per call
  /// here (this is host setup-side code, not a device inner loop); a hot path should hold the
  /// descriptor instead -- see the rung-3 note in docs/ANALYTIC_SDF_GEOMETRY.md.
  GridDesc<double> desc() const {
    GridDesc<double> d;
    d.nx = static_cast<int>(dims[0]);
    d.ny = static_cast<int>(dims[1]);
    d.nz = static_cast<int>(dims[2]);
    d.offset = 0;
    d.origin = Vec3<double>{origin[0], origin[1], origin[2]};
    d.invSpacing = Vec3<double>{spacing[0] != 0.0 ? 1.0 / spacing[0] : 0.0,
                                spacing[1] != 0.0 ? 1.0 / spacing[1] : 0.0,
                                spacing[2] != 0.0 ? 1.0 / spacing[2] : 0.0};
    d.extension = GridExtension::kClamp;  // flat at the box face -- this type's documented contract
    return d;
  }

  /// Trilinearly-interpolated signed distance at world point @p p. Queries outside the sampled box
  /// clamp to the nearest in-domain sample (GridExtension::kClamp).
  ///
  /// DELEGATES to peclet::core::geom::sampleGrid, which is now the suite's single trilinear
  /// grid-SDF routine (it also serves dem's object/container policies). NOTE this changed the
  /// last bit of the result: the old body divided by `spacing`, the shared routine multiplies by
  /// its inverse, and a/s != a*(1/s) in floating point. Deliberate -- multiplying is what dem's
  /// bit-exactness contract is written against and is cheaper on device.
  double eval(const Vec<3>& p) const {
    return sampleGrid(Vec3<double>{p[0], p[1], p[2]}, desc(), PoolPtr<float>{values.data()});
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

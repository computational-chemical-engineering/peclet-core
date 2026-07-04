// core — generic trilinear particle<->grid interpolation (the P2G / G2P primitive for CFD-DEM).
//
// Physics-free: gather a cell-centred grid field to particle locations (grid->particle, G2P) and
// scatter a per-particle scalar onto the grid with the transpose weights (particle->grid, P2G, via
// atomic_add so overlapping particles accumulate). Both are Kokkos parallel_fors over the device
// views the caller supplies, so the coupling never leaves the device.
//
// Layout contract (matches flow's fields): the field is a FLAT x-fastest buffer of padded extents
// (ex,ey,ez) = (nx+2g, ny+2g, nz+2g); a cell-centred inner sample (i,j,k) in [0,nx)x[0,ny)x[0,nz)
// lives at flat index (i+g) + (j+g)*ex + (k+g)*ex*ey and represents the value at physical position
// origin + (i+0.5)*h. With ghost width g>=1 the 8-point trilinear stencil of any particle inside
// [origin, origin+L] stays in bounds (the half-cell overhang lands on a ghost cell); indices are
// clamped defensively regardless.
//
// Header-only, Kokkos required. Templated on the view types so it composes with dem's float particle
// SoA and flow's double grid fields (interpolation arithmetic is done in double).
#ifndef PECLET_CORE_INTERP_PARTICLE_GRID_HPP
#define PECLET_CORE_INTERP_PARTICLE_GRID_HPP

#include <Kokkos_Core.hpp>

namespace peclet::core::interp {

// Origin, inverse spacing, padded extents and ghost width of a cell-centred grid field.
struct GridMap {
  double ox, oy, oz;     // physical origin (low corner of cell (0,0,0))
  double idx, idy, idz;  // inverse spacing 1/h per axis
  int ex, ey, ez;        // PADDED extents (inner + 2*ghost)
  int g;                 // ghost width
};

namespace detail {
// Cell-centred trilinear stencil at physical coordinate p along one axis: base inner index i0 and
// weight w in [0,1) so value = (1-w)*f[i0] + w*f[i0+1]. si = (p-origin)*inv - 0.5 (cell centres are
// half a cell in from the low corner).
KOKKOS_INLINE_FUNCTION void axisStencil(double p, double origin, double inv, int nInner, int& i0,
                                        double& w) {
  const double si = (p - origin) * inv - 0.5;
  double fi = Kokkos::floor(si);
  i0 = (int)fi;
  w = si - fi;
  if (i0 < -1) {
    i0 = -1;
    w = 0.0;
  }
  if (i0 > nInner - 1) {
    i0 = nInner - 1;
    w = 1.0;
  }
}
}  // namespace detail

// G2P: out(p) = trilinear interpolation of the cell-centred field at pos(p). PosView is (N,3) float
// or double; FieldView is a flat View<T*>; OutView is a per-particle View<Tout*>.
template <class PosView, class FieldView, class OutView>
void trilinearGather(int nParticles, PosView pos, FieldView field, OutView out, GridMap m) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int ex = m.ex, ey = m.ey, ez = m.ez, g = m.g;
  const int nx = ex - 2 * g, ny = ey - 2 * g, nz = ez - 2 * g;
  Kokkos::parallel_for(
      "peclet::core::interp::gather", Kokkos::RangePolicy<Exec>(0, nParticles), KOKKOS_LAMBDA(int p) {
        int i0, j0, k0;
        double wx, wy, wz;
        detail::axisStencil((double)pos(p, 0), m.ox, m.idx, nx, i0, wx);
        detail::axisStencil((double)pos(p, 1), m.oy, m.idy, ny, j0, wy);
        detail::axisStencil((double)pos(p, 2), m.oz, m.idz, nz, k0, wz);
        const long sx = 1, sy = ex, sz = (long)ex * ey;
        const long b = (long)(i0 + g) + (long)(j0 + g) * sy + (long)(k0 + g) * sz;
        auto F = [&](long o) { return (double)field(b + o); };
        const double c00 = F(0) * (1 - wx) + F(sx) * wx;
        const double c10 = F(sy) * (1 - wx) + F(sy + sx) * wx;
        const double c01 = F(sz) * (1 - wx) + F(sz + sx) * wx;
        const double c11 = F(sz + sy) * (1 - wx) + F(sz + sy + sx) * wx;
        const double c0 = c00 * (1 - wy) + c10 * wy;
        const double c1 = c01 * (1 - wy) + c11 * wy;
        out(p) = c0 * (1 - wz) + c1 * wz;
      });
}

// P2G: atomic_add q(p) onto the 8 surrounding cell-centred field cells with the SAME trilinear
// weights (the transpose of gather — Σ over particles conserves the deposited quantity). field must
// be pre-zeroed by the caller. Atomics => run-to-run order dependence (results tolerance-, not
// bit-exact).
template <class PosView, class QView, class FieldView>
void trilinearScatterAtomic(int nParticles, PosView pos, QView q, FieldView field, GridMap m) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int ex = m.ex, ey = m.ey, ez = m.ez, g = m.g;
  const int nx = ex - 2 * g, ny = ey - 2 * g, nz = ez - 2 * g;
  using T = typename FieldView::value_type;
  Kokkos::parallel_for(
      "peclet::core::interp::scatter", Kokkos::RangePolicy<Exec>(0, nParticles),
      KOKKOS_LAMBDA(int p) {
        int i0, j0, k0;
        double wx, wy, wz;
        detail::axisStencil((double)pos(p, 0), m.ox, m.idx, nx, i0, wx);
        detail::axisStencil((double)pos(p, 1), m.oy, m.idy, ny, j0, wy);
        detail::axisStencil((double)pos(p, 2), m.oz, m.idz, nz, k0, wz);
        const long sx = 1, sy = ex, sz = (long)ex * ey;
        const long b = (long)(i0 + g) + (long)(j0 + g) * sy + (long)(k0 + g) * sz;
        const double qv = (double)q(p);
        const double w[2] = {1.0 - wx, wx}, wj[2] = {1.0 - wy, wy}, wk[2] = {1.0 - wz, wz};
        for (int dk = 0; dk < 2; ++dk)
          for (int dj = 0; dj < 2; ++dj)
            for (int di = 0; di < 2; ++di) {
              const long o = b + di * sx + dj * sy + dk * sz;
              Kokkos::atomic_add(&field(o), (T)(qv * w[di] * wj[dj] * wk[dk]));
            }
      });
}

}  // namespace peclet::core::interp

#endif  // PECLET_CORE_INTERP_PARTICLE_GRID_HPP

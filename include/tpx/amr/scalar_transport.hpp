// transport-core — explicit finite-volume scalar transport on a BlockOctree.
//
// Solves  dc/dt + div(u c) = D lap(c)  on the adaptive octree (periodic), the
// canonical "AMR for scalar transport" target. Reusable shared infra: sdflow (or
// any consumer) can advect a species/level-set on an AMR grid through this, and
// later couple it to a structured hydro solver via leaf point-location.
//
// Discretisation (cell-centered FV, explicit Euler):
//   * advection — first-order upwind face flux. The outward normal velocity at a
//     face selects the upwind cell, so the same physical face uses the same upwind
//     value from both sides => conservative and monotone (no new extrema).
//   * diffusion — the conservative two-point flux of AmrPoisson.
//   * 2:1 interfaces — the coarse side sums over all fine sub-faces (each at its
//     own face centre / velocity), so mass is conserved across refinement jumps.
// The face-normal velocity is sampled from a user callable vel(faceCentreWorld,
// axis) -> Real; pass a divergence-free field for a conservative, monotone update.
//
// Header-only, guarded by TPX_HAVE_MORTON. Serial/host first.
#ifndef TPX_AMR_SCALAR_TRANSPORT_HPP
#define TPX_AMR_SCALAR_TRANSPORT_HPP

#ifdef TPX_HAVE_MORTON

#include <array>
#include <cmath>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/common/types.hpp"

namespace tpx::amr {

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class ScalarTransport {
 public:
  using Octree = BlockOctree<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;
  using Coord = typename Octree::Coord;

  ScalarTransport() = default;
  ScalarTransport(const Octree& t, const AmrGeometry<Dim>& geo) { init(t, geo); }

  void init(const Octree& t, const AmrGeometry<Dim>& geo) {
    t_ = &t;
    geo_ = geo;
    for (int d = 0; d < Dim; ++d) fineExt_[d] = static_cast<Coord>(t.brick()[d] * (Index(1) << t.lmax()));
  }

  Index numLeaves() const { return t_->numLeaves(); }
  Real cellWidth(Index i) const { return geo_.h0 * static_cast<Real>(Index(1) << t_->level(i)); }
  Real cellVolume(Index i) const {
    Real w = cellWidth(i), v = 1;
    for (int d = 0; d < Dim; ++d) v *= w;
    return v;
  }

  /// Total scalar content sum_i V_i c_i (conserved by a divergence-free update).
  double totalMass(const std::vector<double>& c) const {
    double s = 0.0;
    for (Index i = 0; i < numLeaves(); ++i) s += cellVolume(i) * c[static_cast<std::size_t>(i)];
    return s;
  }

  /// One explicit Euler step: cOut = c + dt*(-div(u c) + D lap(c)). `vel(fc,axis)`
  /// returns the axis-component of velocity at world face centre `fc`.
  template <class VelFn>
  void step(const std::vector<double>& c, std::vector<double>& cOut, double dt, double D,
            VelFn&& vel) const {
    const Index n = numLeaves();
    cOut.assign(static_cast<std::size_t>(n), 0.0);
    for (Index i = 0; i < n; ++i) {
      const double ci = c[static_cast<std::size_t>(i)];
      double flux = 0.0;  // sum of outward (advective + diffusive) fluxes
      forEachFace(i, [&](Index j, int axis, int oSign, Real area, const Vec<Dim>& fc, Real dist) {
        const double cj = c[static_cast<std::size_t>(j)];
        const double unOut = oSign * static_cast<double>(vel(fc, axis));
        const double cUp = (unOut > 0.0) ? ci : cj;        // upwind
        flux += area * unOut * cUp;                        // advective outflow
        flux += -D * area * (cj - ci) / dist;              // diffusive outflow
      });
      cOut[static_cast<std::size_t>(i)] = ci - dt * flux / cellVolume(i);
    }
  }

  /// Visit each (sub)face of leaf `i`. fn(neighbour, axis, outwardSign, areaPhys,
  /// faceCentreWorld, centreDistPhys). Periodic; 2:1 interfaces enumerated.
  template <class Fn>
  void forEachFace(Index i, Fn&& fn) const {
    auto b = t_->bounds(i);
    const auto& lo = b[0];
    const unsigned Li = t_->level(i);
    const Coord si = Coord(Coord(1) << Li);
    for (int axis = 0; axis < Dim; ++axis)
      for (int dir = -1; dir <= 1; dir += 2) {
        const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                  : static_cast<long>(lo[axis]) - 1;
        const long facePlane = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                         : static_cast<long>(lo[axis]);
        std::array<Coord, Dim> p = lo;
        p[axis] = wrap(pc, axis);
        Index j0 = t_->find(M::encode(p).code());
        const unsigned Lj = t_->level(j0);
        if (Lj >= Li) {
          fn(j0, axis, dir, faceArea(si), faceCentre(lo, axis, facePlane, std::array<Coord, Dim>{}, si),
             0.5 * (static_cast<Real>(si) + static_cast<Real>(Coord(Coord(1) << Lj))) * geo_.h0);
        } else {
          const Coord sj = Coord(si >> 1);
          const int nsub = 1 << (Dim - 1);
          for (int k = 0; k < nsub; ++k) {
            std::array<Coord, Dim> q = lo;
            q[axis] = wrap(pc, axis);
            std::array<Coord, Dim> off{};
            int bit = 0;
            for (int t = 0; t < Dim; ++t) {
              if (t == axis) continue;
              off[t] = ((k >> bit) & 1) ? sj : Coord(0);
              q[t] = wrap(static_cast<long>(lo[t]) + static_cast<long>(off[t]), t);
              ++bit;
            }
            Index jj = t_->find(M::encode(q).code());
            fn(jj, axis, dir, faceArea(sj), faceCentre(lo, axis, facePlane, off, sj),
               0.5 * (static_cast<Real>(si) + static_cast<Real>(sj)) * geo_.h0);
          }
        }
      }
  }

 private:
  Coord wrap(long c, int axis) const {
    long e = static_cast<long>(fineExt_[axis]);
    return static_cast<Coord>(((c % e) + e) % e);
  }
  Real faceArea(Coord s) const {
    Real a = 1;
    for (int d = 0; d < Dim - 1; ++d) a *= static_cast<Real>(s) * geo_.h0;
    return a;
  }
  // World centre of a (sub)face: `facePlane` is the shared-plane coord (fine) on
  // `axis`; the sub-face spans width `tw` from lo+off in each tangential axis.
  Vec<Dim> faceCentre(const std::array<Coord, Dim>& lo, int axis, long facePlane,
                      const std::array<Coord, Dim>& off, Coord tw) const {
    Vec<Dim> fc{};
    for (int d = 0; d < Dim; ++d) {
      if (d == axis)
        fc[d] = geo_.origin[d] + static_cast<Real>(facePlane) * geo_.h0;
      else
        fc[d] = geo_.origin[d] +
                (static_cast<Real>(lo[d] + off[d]) + 0.5 * static_cast<Real>(tw)) * geo_.h0;
    }
    return fc;
  }

  const Octree* t_ = nullptr;
  AmrGeometry<Dim> geo_{};
  std::array<Coord, Dim> fineExt_{};
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_SCALAR_TRANSPORT_HPP

// core — signed-distance geometry for solids (see suite/docs/CONVENTIONS.md §2).
//
// Sign convention (load-bearing, shared across the suite): sdf < 0 inside solid, > 0 in fluid/void,
// 0 on the surface; the gradient points outward (into the fluid). Analytic primitives mirror the
// shapes packing-gpu already uses (sphere, box, hollow cylinder); a generic finite-difference
// gradient covers any shape, and a `Sdf` concept lets solvers accept analytic or grid SDFs uniformly.
//
// C++17-clean below the concept so this header can be pulled into CUDA translation units; the concept
// itself is guarded for C++20 host use.
#ifndef PECLET_CORE_GEOM_SDF_HPP
#define PECLET_CORE_GEOM_SDF_HPP

#include <algorithm>
#include <cmath>

#include "peclet/core/common/types.hpp"

namespace peclet::core::geom {

namespace detail {
inline Vec<3> sub(const Vec<3>& a, const Vec<3>& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline double norm(const Vec<3>& a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
}  // namespace detail

/// Solid ball: negative inside.
struct Sphere {
  Vec<3> center{};
  double radius = 1.0;
  double eval(const Vec<3>& p) const { return detail::norm(detail::sub(p, center)) - radius; }
};

/// Axis-aligned solid box of half-extents `half`: standard exact box SDF, negative inside.
struct Box {
  Vec<3> center{};
  Vec<3> half{0.5, 0.5, 0.5};
  double eval(const Vec<3>& p) const {
    Vec<3> d = detail::sub(p, center);
    Vec<3> q{std::fabs(d[0]) - half[0], std::fabs(d[1]) - half[1], std::fabs(d[2]) - half[2]};
    double outside = detail::norm({std::max(q[0], 0.0), std::max(q[1], 0.0), std::max(q[2], 0.0)});
    double inside = std::min(std::max(q[0], std::max(q[1], q[2])), 0.0);
    return outside + inside;
  }
};

/// Solid hollow cylinder (a tube wall) of given outer/inner radius and height about `axis`.
/// Modelled as the CSG intersection {r<=rOuter} ∩ {r>=rInner} ∩ {|z|<=height/2}; the shell material
/// is the solid (negative). Sign-exact; distance is the standard max-of-halfspaces approximation.
struct HollowCylinder {
  Vec<3> center{};
  double rOuter = 1.0;
  double rInner = 0.5;
  double height = 1.0;
  int axis = 2;
  double eval(const Vec<3>& p) const {
    Vec<3> d = detail::sub(p, center);
    int a0 = (axis + 1) % 3, a1 = (axis + 2) % 3;
    double r = std::sqrt(d[a0] * d[a0] + d[a1] * d[a1]);
    double z = d[axis];
    return std::max({r - rOuter, rInner - r, std::fabs(z) - height * 0.5});
  }
};

/// Negation: the solid and the void swap (sdf -> -sdf). Useful to make a container's interior the
/// fluid (e.g. solid everywhere outside a box).
template <typename S>
struct Complement {
  S shape;
  double eval(const Vec<3>& p) const { return -shape.eval(p); }
};

/// Generic outward normal via central differences; works for any shape with eval().
template <typename S>
Vec<3> gradient(const S& shape, const Vec<3>& p, double h = 1e-4) {
  Vec<3> g{};
  for (int i = 0; i < 3; ++i) {
    Vec<3> pp = p, pm = p;
    pp[i] += h;
    pm[i] -= h;
    g[i] = (shape.eval(pp) - shape.eval(pm)) / (2.0 * h);
  }
  return g;
}

#if defined(__cpp_concepts)
/// A shape usable as a solid: it can report a signed distance at a point.
template <typename S>
concept Sdf = requires(const S s, Vec<3> p) {
  { s.eval(p) } -> std::convertible_to<double>;
};
#endif

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_SDF_HPP

// core — implicit quadrature over {phi > 0}: face apertures and cell volume fractions for
// ANALYTIC geometry, the general replacement for the sphere-only closed form
// (flow/scripts/exact_apertures_spheres.py).
//
// WHAT THIS IS. Given any device-callable field phi (a SceneQueryView, a single leaf, a lambda),
// compute the fraction of an axis-aligned face or box on which phi > 0. The construction is
// Saye's dimension reduction (SIAM J. Sci. Comput. 37(2), 2015) at its simplest useful level:
//
//   - pick a HEIGHT DIRECTION in which the interface is (locally) a graph;
//   - integrate over the remaining directions with Gauss-Legendre;
//   - along each height line, locate the sign changes by bracketed bisection and sum the
//     lengths of the positive sub-intervals.
//
// The inner 1-D measure is exact to rounding, so all the error lives in the outer Gauss rule, and
// its integrand -- the positive length as a function of the transverse coordinate -- is smooth
// exactly where the interface is a single graph over the height axis.
//
// THE OUTER RULE IS SPLIT AT THE KINKS, and that is what makes the method worth having. The length
// function has a kink wherever the interface crosses an EDGE of the face, and integrating straight
// through those pins the error at ~1e-3 regardless of h: an aperture is a FRACTION, so a fixed kink
// in a fixed normalized integrand costs a fixed amount and refining the grid does not help. Locating
// the crossings on the two edge functions (two bracketed bisections each) and running the Gauss rule
// piecewise removes the obstruction: measured on a sphere against the closed form, the mean face
// error goes 8.5e-04 -> 3.9e-12 at N=16 and 5.4e-04 -> 3.9e-12 at N=64, against 3.8e-02 and 4.7e-02
// for the sampled-SDF linear estimator the CFD otherwise uses. Pass breakpoints=false to reproduce
// the unsplit behaviour.
//
// cellVolumeFraction does NOT do this: its outer rule is two-dimensional and the kinks are CURVES,
// not points. Its accuracy is correspondingly weaker (~1e-6 relative on a sphere at 24^3) and more
// nodes, not subdivision, is the lever there.
//
// *** NOT CERTIFIED. *** Unlike the candidate-grid pruning in scene_query.hpp, this carries NO
// exactness guarantee, and the failure modes are structural, not rounding:
//
//   1. MULTIPLE CROSSINGS PER LINE are handled (every bracketed sign change is refined and the
//      positive sub-intervals accumulate), but only those a fixed initial subdivision RESOLVES. A
//      feature thinner than the sampling stride between two bracketing points is missed entirely.
//   2. WHERE THE INTERFACE EXITS THE FACE the length function has a kink or a square-root corner,
//      and the outer Gauss rule drops from spectral to roughly second order. That is the generic
//      case for a sphere cutting a grid face, so second order is what the gate MEASURES -- do not
//      expect the spectral rate the smooth case would give.
//   3. NON-GRAPH interfaces (a thin neck, two surfaces crossing the same line non-monotonically,
//      a corner of a CSG difference) violate the premise of the height reduction. The result stays
//      bounded in [0,1] and consistent, but the order claim is void.
//   4. The height direction is chosen ONCE per face from the gradient at its centre. A face whose
//      interface changes orientation across it gets a poor choice and loses order.
//
// Consumers must therefore treat this as a high-accuracy ESTIMATOR, not as an exact aperture:
// keep the sub-resolution floor the cut-cell pressure operator needs (alpha ~ 1e-12 rows destroy
// its conditioning -- measured), and do not build a conservation argument on it.
//
// Host- and device-callable (PECLET_HD); phi is a template parameter so it inlines.
#ifndef PECLET_CORE_GEOM_QUADRATURE_HPP
#define PECLET_CORE_GEOM_QUADRATURE_HPP

#include "peclet/core/common/portable.hpp"  // PECLET_HD, Vec3, Quat
#include "peclet/core/common/types.hpp"

namespace peclet::core::geom {

/// Gauss-Legendre nodes/weights on [0,1] for n = 1..8. Returned by reference into a static
/// constexpr table so the caller pays nothing; n outside the range clamps.
namespace quad_detail {

template <class Real>
PECLET_HD void gaussLegendre01(int n, int i, Real& x, Real& w) {
  // nodes/weights on [-1,1], mapped to [0,1] as x = (1+t)/2, w = wt/2
  constexpr double T1[1] = {0.0};
  constexpr double W1[1] = {2.0};
  constexpr double T2[2] = {-0.5773502691896257, 0.5773502691896257};
  constexpr double W2[2] = {1.0, 1.0};
  constexpr double T3[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
  constexpr double W3[3] = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
  constexpr double T4[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
                            0.8611363115940526};
  constexpr double W4[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
                            0.3478548451374538};
  constexpr double T5[5] = {-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831,
                            0.9061798459386640};
  constexpr double W5[5] = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                            0.4786286704993665, 0.2369268850561891};
  constexpr double T6[6] = {-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                            0.2386191860831969,  0.6612093864662645,  0.9324695142031521};
  constexpr double W6[6] = {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                            0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
  constexpr double T8[8] = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
                            -0.1834346424956498, 0.1834346424956498,  0.5255324099163290,
                            0.7966664774136267,  0.9602898564975363};
  constexpr double W8[8] = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
                            0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                            0.2223810344533745, 0.1012285362903763};
  const double* T = T4;
  const double* W = W4;
  int m = 4;
  if (n <= 1) { T = T1; W = W1; m = 1; }
  else if (n == 2) { T = T2; W = W2; m = 2; }
  else if (n == 3) { T = T3; W = W3; m = 3; }
  else if (n == 4) { T = T4; W = W4; m = 4; }
  else if (n == 5) { T = T5; W = W5; m = 5; }
  else if (n == 6) { T = T6; W = W6; m = 6; }
  else { T = T8; W = W8; m = 8; }
  const int k = i < 0 ? 0 : (i >= m ? m - 1 : i);
  x = Real(0.5) * (Real(1) + Real(T[k]));
  w = Real(0.5) * Real(W[k]);
}

PECLET_HD int gaussCount(int n) {
  return n <= 1 ? 1 : (n == 2 ? 2 : (n == 3 ? 3 : (n == 4 ? 4 : (n == 5 ? 5 : (n == 6 ? 6 : 8)))));
}

/// Length of {phi > 0} along the segment p0 + s*d, s in [0,1], by bracketed bisection.
/// `nseg` initial subintervals: a feature thinner than 1/nseg of the segment is INVISIBLE here --
/// failure mode 1 in the header comment.
template <class Real, class Phi>
PECLET_HD Real positiveLength(const Phi& phi, Vec3<Real> p0, Vec3<Real> d, int nseg) {
  const int NS = nseg < 2 ? 2 : (nseg > 64 ? 64 : nseg);
  auto at = [&](Real s) {
    return phi(Vec3<Real>{p0.x + s * d.x, p0.y + s * d.y, p0.z + s * d.z});
  };
  auto root = [&](Real a, Real b, Real fa) {
    for (int it = 0; it < 40; ++it) {  // ~1 ulp of the segment parameter
      const Real m = Real(0.5) * (a + b);
      const Real fm = at(m);
      if ((fm > Real(0)) == (fa > Real(0))) { a = m; fa = fm; }
      else { b = m; }
    }
    return Real(0.5) * (a + b);
  };
  Real acc = 0;
  Real sPrev = 0, fPrev = at(Real(0));
  Real openAt = fPrev > Real(0) ? Real(0) : Real(-1);  // start of the current positive run
  for (int i = 1; i <= NS; ++i) {
    const Real s = Real(i) / Real(NS);
    const Real f = at(s);
    if ((f > Real(0)) != (fPrev > Real(0))) {
      const Real r = root(sPrev, s, fPrev);
      if (fPrev > Real(0)) {          // positive run ends here
        acc += r - openAt;
        openAt = Real(-1);
      } else {                        // positive run starts here
        openAt = r;
      }
    }
    sPrev = s;
    fPrev = f;
  }
  if (openAt >= Real(0))
    acc += Real(1) - openAt;
  return acc < Real(0) ? Real(0) : (acc > Real(1) ? Real(1) : acc);
}

}  // namespace quad_detail

/// Fluid fraction ({phi > 0}) of the axis-aligned FACE with corner `origin`, normal `axis`, and
/// side lengths `hu`, `hv` along the two axes following `axis` cyclically. `order` is the number
/// of Gauss nodes across the face (4 by default); `nseg` the initial bracketing subdivision along
/// the height direction.
///
/// The height direction is the in-face axis along which the field varies faster at the face
/// centre, so the interface is a graph over the other one. That choice is made ONCE per face --
/// failure mode 4.
template <class Real, class Phi>
PECLET_HD Real faceAperture(const Phi& phi, Vec3<Real> origin, int axis, Real hu, Real hv,
                            int order = 4, int nseg = 8, bool breakpoints = true) {
  const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
  Real e1[3] = {0, 0, 0}, e2[3] = {0, 0, 0};
  e1[a1] = hu;
  e2[a2] = hv;
  auto at = [&](Real u, Real v) {
    return phi(Vec3<Real>{origin.x + u * e1[0] + v * e2[0], origin.y + u * e1[1] + v * e2[1],
                          origin.z + u * e1[2] + v * e2[2]});
  };
  // height direction: the in-face axis with the larger central difference
  const Real h = Real(0.25);
  const Real du = at(Real(0.5) + h, Real(0.5)) - at(Real(0.5) - h, Real(0.5));
  const Real dv = at(Real(0.5), Real(0.5) + h) - at(Real(0.5), Real(0.5) - h);
  const bool heightIsV = (dv < Real(0) ? -dv : dv) >= (du < Real(0) ? -du : du);

  // Transverse coordinate t; the measured length L(t) along the height direction. L is smooth
  // WHERE THE INTERFACE STAYS INSIDE THE FACE and has a KINK at every t where it crosses one of
  // the two edges t -> (t, 0) and t -> (t, 1). Integrating straight through those kinks is what
  // pins the error at ~1e-3 no matter how fine the grid: the aperture is a FRACTION, so a fixed
  // kink in a fixed normalized integrand gives a fixed error, and refining h does not help --
  // only more nodes do. Splitting the rule AT the kinks removes the obstruction entirely, and the
  // breakpoints cost two bracketed bisections each on the same edge functions.
  auto edge = [&](Real t, int which) {
    return heightIsV ? at(t, which ? Real(1) : Real(0)) : at(which ? Real(1) : Real(0), t);
  };
  Real brk[10];
  int nb = 0;
  const int NS = nseg < 2 ? 2 : (nseg > 32 ? 32 : nseg);
  for (int w = 0; w < 2 && breakpoints && nb < 8; ++w) {
    Real tPrev = 0, fPrev = edge(Real(0), w);
    for (int i = 1; i <= NS && nb < 8; ++i) {
      const Real t = Real(i) / Real(NS);
      const Real f = edge(t, w);
      if ((f > Real(0)) != (fPrev > Real(0))) {
        Real a = tPrev, b = t, fa = fPrev;
        for (int it = 0; it < 40; ++it) {
          const Real m = Real(0.5) * (a + b);
          const Real fm = edge(m, w);
          if ((fm > Real(0)) == (fa > Real(0))) { a = m; fa = fm; }
          else { b = m; }
        }
        brk[nb++] = Real(0.5) * (a + b);
      }
      tPrev = t;
      fPrev = f;
    }
  }
  // sort the breakpoints (at most 8; insertion sort is the right tool at this size)
  for (int i = 1; i < nb; ++i) {
    const Real key = brk[i];
    int j = i - 1;
    while (j >= 0 && brk[j] > key) { brk[j + 1] = brk[j]; --j; }
    brk[j + 1] = key;
  }
  brk[nb] = Real(1);

  const int nq = quad_detail::gaussCount(order);
  Real acc = 0;
  Real lo = 0;
  for (int seg = 0; seg <= nb; ++seg) {
    const Real hi = brk[seg];
    const Real len = hi - lo;
    if (len > Real(0)) {
      for (int i = 0; i < nq; ++i) {
        Real t, w;
        quad_detail::gaussLegendre01<Real>(order, i, t, w);
        const Real tt = lo + t * len;
        Vec3<Real> p0, d;
        if (heightIsV) {
          p0 = Vec3<Real>{origin.x + tt * e1[0], origin.y + tt * e1[1], origin.z + tt * e1[2]};
          d = Vec3<Real>{e2[0], e2[1], e2[2]};
        } else {
          p0 = Vec3<Real>{origin.x + tt * e2[0], origin.y + tt * e2[1], origin.z + tt * e2[2]};
          d = Vec3<Real>{e1[0], e1[1], e1[2]};
        }
        acc += w * len * quad_detail::positiveLength(phi, p0, d, nseg);
      }
    }
    lo = hi;
  }
  return acc < Real(0) ? Real(0) : (acc > Real(1) ? Real(1) : acc);
}

/// Fluid fraction ({phi > 0}) of the axis-aligned BOX [origin, origin+h]. Same construction one
/// dimension up: a tensor Gauss rule over the two transverse axes, exact 1-D measure along the
/// height axis. Same non-certification.
template <class Real, class Phi>
PECLET_HD Real cellVolumeFraction(const Phi& phi, Vec3<Real> origin, Vec3<Real> h, int order = 4,
                                  int nseg = 8) {
  auto at = [&](Real a, Real b, Real c) {
    return phi(Vec3<Real>{origin.x + a * h.x, origin.y + b * h.y, origin.z + c * h.z});
  };
  const Real q = Real(0.25);
  const Real gx = at(Real(0.5) + q, Real(0.5), Real(0.5)) - at(Real(0.5) - q, Real(0.5), Real(0.5));
  const Real gy = at(Real(0.5), Real(0.5) + q, Real(0.5)) - at(Real(0.5), Real(0.5) - q, Real(0.5));
  const Real gz = at(Real(0.5), Real(0.5), Real(0.5) + q) - at(Real(0.5), Real(0.5), Real(0.5) - q);
  auto ab = [](Real v) { return v < Real(0) ? -v : v; };
  int hi = 0;
  if (ab(gy) >= ab(gx) && ab(gy) >= ab(gz)) hi = 1;
  else if (ab(gz) >= ab(gx) && ab(gz) >= ab(gy)) hi = 2;
  const int t1 = (hi + 1) % 3, t2 = (hi + 2) % 3;
  const Real hh[3] = {h.x, h.y, h.z};

  const int nq = quad_detail::gaussCount(order);
  Real acc = 0;
  for (int i = 0; i < nq; ++i) {
    Real ti, wi;
    quad_detail::gaussLegendre01<Real>(order, i, ti, wi);
    for (int j = 0; j < nq; ++j) {
      Real tj, wj;
      quad_detail::gaussLegendre01<Real>(order, j, tj, wj);
      Real o[3] = {origin.x, origin.y, origin.z};
      o[t1] += ti * hh[t1];
      o[t2] += tj * hh[t2];
      Real d[3] = {0, 0, 0};
      d[hi] = hh[hi];
      acc += wi * wj *
             quad_detail::positiveLength(phi, Vec3<Real>{o[0], o[1], o[2]},
                                         Vec3<Real>{d[0], d[1], d[2]}, nseg);
    }
  }
  return acc < Real(0) ? Real(0) : (acc > Real(1) ? Real(1) : acc);
}

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_QUADRATURE_HPP

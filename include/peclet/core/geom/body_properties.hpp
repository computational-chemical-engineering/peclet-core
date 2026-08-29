// core — rigid-body mass properties of an implicit solid {phi < 0}: volume, mass, centre of mass,
// the full inertia tensor, and its principal decomposition (three moments + a QUATERNION), by
// composite implicit quadrature on the same backbone as geom/quadrature.hpp.
//
// WHY THIS EXISTS. The DEM particle pipeline needs mass properties at preprocessing time, and its
// existing voxel integrator (peclet.dem.particle_builder) is percent-level at default settings and
// SYSTEMATICALLY biased when the field is a non-true-distance bound (measured +3.5% inertia on an
// ellipsoid) -- exactly what core's non-certified leaves and CSG results emit. This integrator only
// consumes SIGNS and bracketed roots: along each quadrature line the positive-set boundary is
// located by bisection, and every leaf in the vocabulary is sign-exact, so bound-only fields lose
// NOTHING here. The 1-D moments over each interior run are analytic (∫x^k dx over [a,b]), so the
// error lives in the transverse Gauss rule per cell, whose kinks are CURVES (where the interface
// crosses cell faces) -- the same obstruction cellVolumeFraction documents, and unlike the face
// apertures' kink POINTS it is not removed by splitting. MEASURED grade (ctest geom_body):
// ~1e-4 relative at n=8, ~4e-6 at n=32, converging ~O(n^-2), with ZERO systematic bias on
// bound-only fields -- against the voxel integrator's 3e-3-with-+3.5e-2-bias. If a consumer ever
// needs more, the recorded lever is nested 1-D splitting in the transverse plane (bracket the
// interface's zeros on the cell's cap faces per row); no consumer does today.
//
// FRAME CONTRACT. Everything is reported in the INPUT frame of phi:
//     p_input = com + R * p_body,     R = `rotation` (columns = principal axes), also as `quat`.
// A shape whose canonical frame is NOT its principal frame is therefore fully supported at the
// measurement level; re-expressing the shape ITSELF in the principal frame is the consumer's move
// (SceneBuilder::addReframed does it exactly for node trees -- one composed transform, no
// resampling).
//
// DEGENERATE FRAMES. Within a cluster of (near-)equal principal moments any orthonormal basis is
// principal, and a raw eigensolver returns an arbitrary one -- a cube comes back edge-down. Inside
// each cluster the basis is rotated onto the input axes with the largest projection into that
// subspace (orthogonal Procrustes), so a sphere or cube reports the IDENTITY rotation and a
// z-aligned cylinder reports a z-aligned frame. Ported from peclet.dem.particle_builder, where the
// behaviour is documented and relied on.
//
// Host-only (preprocessing); `Phi` is any callable Vec3<Real> -> Real, negative inside --
// a lambda, a SceneQueryView, a DeviceScene host mirror.
#ifndef PECLET_CORE_GEOM_BODY_PROPERTIES_HPP
#define PECLET_CORE_GEOM_BODY_PROPERTIES_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "peclet/core/geom/quadrature.hpp"

namespace peclet::core::geom {

template <class Real>
struct BodyProperties {
  Real volume = 0;               ///< |{phi < 0}|
  Real mass = 0;                 ///< density * volume
  Vec3<Real> com{0, 0, 0};       ///< centre of mass, input frame
  Real inertia[3][3] = {};       ///< full tensor about the COM, input frame, at the given density
  Real principal[3] = {};        ///< principal moments; principal[k] pairs with column k of R
  Real rotation[3][3] = {};      ///< columns = principal axes in the input frame
  Quat<Real> quat{0, 0, 0, 1};   ///< the same rotation, p_input = com + rotate(quat, p_body)
};

namespace body_detail {

/// Moments of the NEGATIVE set {phi < 0} along p0 + s*d, s in [0,1], in the PARAMETER s:
/// m0 = ∫ 1 ds, m1 = ∫ s ds, m2 = ∫ s² ds over the interior runs. Same bracketed bisection as
/// quadrature.hpp's positiveLength (an initial nseg subdivision brackets, 40 bisections refine to
/// ~1 ulp of the parameter); the per-run integrals are analytic. NOTE the sign convention: bodies
/// are the interior (phi < 0), where the aperture code measured the fluid (phi > 0).
template <class Real, class Phi>
void interiorMoments(const Phi& phi, Vec3<Real> p0, Vec3<Real> d, int nseg, Real& m0, Real& m1,
                     Real& m2) {
  const int NS = nseg < 2 ? 2 : (nseg > 64 ? 64 : nseg);
  auto at = [&](Real s) {
    return phi(Vec3<Real>{p0.x + s * d.x, p0.y + s * d.y, p0.z + s * d.z});
  };
  auto root = [&](Real a, Real b, Real fa) {
    for (int it = 0; it < 40; ++it) {
      const Real m = Real(0.5) * (a + b);
      const Real fm = at(m);
      if ((fm < Real(0)) == (fa < Real(0))) {
        a = m;
        fa = fm;
      } else {
        b = m;
      }
    }
    return Real(0.5) * (a + b);
  };
  m0 = m1 = m2 = Real(0);
  auto accumulate = [&](Real a, Real b) {
    m0 += b - a;
    m1 += Real(0.5) * (b * b - a * a);
    m2 += (b * b * b - a * a * a) / Real(3);
  };
  Real sPrev = 0, fPrev = at(Real(0));
  Real openAt = fPrev < Real(0) ? Real(0) : Real(-1);
  for (int i = 1; i <= NS; ++i) {
    const Real s = Real(i) / Real(NS);
    const Real f = at(s);
    if ((f < Real(0)) != (fPrev < Real(0))) {
      const Real r = root(sPrev, s, fPrev);
      if (fPrev < Real(0)) {
        accumulate(openAt, r);
        openAt = Real(-1);
      } else {
        openAt = r;
      }
    }
    sPrev = s;
    fPrev = f;
  }
  if (openAt >= Real(0))
    accumulate(openAt, Real(1));
}

/// Jacobi eigensolver for a symmetric 3x3: A = V diag(w) V^T, V orthonormal columns. Small, exact
/// enough (50 sweeps to machine precision), no dependencies.
template <class Real>
void eigSym3(const Real A[3][3], Real w[3], Real V[3][3]) {
  Real a[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      a[i][j] = A[i][j];
      V[i][j] = (i == j) ? Real(1) : Real(0);
    }
  for (int sweep = 0; sweep < 50; ++sweep) {
    Real off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
    if (off == Real(0))
      break;
    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        if (a[p][q] == Real(0))
          continue;
        const Real theta = (a[q][q] - a[p][p]) / (Real(2) * a[p][q]);
        const Real t = (theta >= Real(0) ? Real(1) : Real(-1)) /
                       (std::fabs(theta) + std::sqrt(theta * theta + Real(1)));
        const Real c = Real(1) / std::sqrt(t * t + Real(1)), s = t * c;
        for (int k = 0; k < 3; ++k) {
          const Real akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const Real apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const Real vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
  }
  for (int i = 0; i < 3; ++i)
    w[i] = a[i][i];
  // sort ascending, carrying columns
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (w[j] < w[i]) {
        std::swap(w[i], w[j]);
        for (int k = 0; k < 3; ++k)
          std::swap(V[k][i], V[k][j]);
      }
}

/// Within each cluster of (near-)equal eigenvalues, rotate the eigenbasis onto the input axes with
/// the largest projection into that subspace (orthogonal Procrustes on the k x k projection). See
/// the header comment; a fully degenerate tensor yields exactly the identity.
template <class Real>
void resolveDegenerateFrame(Real w[3], Real V[3][3], Real relTol = Real(1e-2)) {
  const Real scale = std::max(std::max(std::fabs(w[0]), std::fabs(w[2])), Real(1e-30));
  int i = 0;
  while (i < 3) {
    int j = i + 1;
    while (j < 3 && (w[j] - w[j - 1]) <= relTol * scale)
      ++j;
    const int k = j - i;
    if (k > 1) {
      // projection sizes of each input axis e_r into the subspace = row norms of V[:, i:j]
      Real rn[3];
      for (int r = 0; r < 3; ++r) {
        rn[r] = Real(0);
        for (int c = i; c < j; ++c)
          rn[r] += V[r][c] * V[r][c];
      }
      int sel[3], ns = 0;  // the k axes with the largest projections, in index order
      for (int pick = 0; pick < k; ++pick) {
        int best = -1;
        for (int r = 0; r < 3; ++r) {
          bool used = false;
          for (int u = 0; u < ns; ++u)
            used = used || (sel[u] == r);
          if (!used && (best < 0 || rn[r] > rn[best]))
            best = r;
        }
        sel[ns++] = best;
      }
      for (int u = 0; u < ns; ++u)
        for (int v = u + 1; v < ns; ++v)
          if (sel[v] < sel[u])
            std::swap(sel[u], sel[v]);
      // M = V_sub^T E_sel (k x k); its polar factor rotates V_sub onto the chosen axes
      Real M[3][3] = {};
      for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c)
          M[r][c] = V[sel[c]][i + r];
      // polar factor via Jacobi on M^T M: U = M (M^T M)^{-1/2}
      Real MtM[3][3] = {};
      for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c)
          for (int t = 0; t < k; ++t)
            MtM[r][c] += M[t][r] * M[t][c];
      for (int r = k; r < 3; ++r)
        MtM[r][r] = Real(1);
      Real ew[3], EV[3][3];
      eigSym3(MtM, ew, EV);
      Real inv[3][3] = {};  // (M^T M)^{-1/2}
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
          for (int t = 0; t < 3; ++t)
            inv[r][c] += EV[r][t] * EV[c][t] / std::sqrt(std::max(ew[t], Real(1e-30)));
      Real U[3][3] = {};
      for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c)
          for (int t = 0; t < k; ++t)
            U[r][c] += M[r][t] * inv[t][c];
      // new subspace basis: V_sub <- V_sub U
      Real NV[3][3];
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < k; ++c) {
          NV[r][c] = Real(0);
          for (int t = 0; t < k; ++t)
            NV[r][c] += V[r][i + t] * U[t][c];
        }
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < k; ++c)
          V[r][i + c] = NV[r][c];
    }
    i = j;
  }
}

/// Rotation matrix (columns orthonormal, det +1) -> quaternion (x, y, z, w), Shepperd's method.
template <class Real>
Quat<Real> quatFromRotation(const Real R[3][3]) {
  Quat<Real> q;
  const Real tr = R[0][0] + R[1][1] + R[2][2];
  if (tr > Real(0)) {
    Real s = std::sqrt(tr + Real(1)) * Real(2);
    q.w = Real(0.25) * s;
    q.x = (R[2][1] - R[1][2]) / s;
    q.y = (R[0][2] - R[2][0]) / s;
    q.z = (R[1][0] - R[0][1]) / s;
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    Real s = std::sqrt(Real(1) + R[0][0] - R[1][1] - R[2][2]) * Real(2);
    q.w = (R[2][1] - R[1][2]) / s;
    q.x = Real(0.25) * s;
    q.y = (R[0][1] + R[1][0]) / s;
    q.z = (R[0][2] + R[2][0]) / s;
  } else if (R[1][1] > R[2][2]) {
    Real s = std::sqrt(Real(1) + R[1][1] - R[0][0] - R[2][2]) * Real(2);
    q.w = (R[0][2] - R[2][0]) / s;
    q.x = (R[0][1] + R[1][0]) / s;
    q.y = Real(0.25) * s;
    q.z = (R[1][2] + R[2][1]) / s;
  } else {
    Real s = std::sqrt(Real(1) + R[2][2] - R[0][0] - R[1][1]) * Real(2);
    q.w = (R[1][0] - R[0][1]) / s;
    q.x = (R[0][2] + R[2][0]) / s;
    q.y = (R[1][2] + R[2][1]) / s;
    q.z = Real(0.25) * s;
  }
  return q;
}

}  // namespace body_detail

/// Mass properties of {phi < 0} over the box [lo, hi], composite implicit quadrature: an n^3 cell
/// grid; per cell a tensor Gauss rule (order nodes per transverse axis, height axis chosen from
/// the central gradient like cellVolumeFraction) with analytic 1-D moments along each line. All
/// ten raw moments (volume, 3 first, 6 second) come from the same line walks in one pass.
///
/// The box MUST contain the whole solid; anything outside is not seen (that is what the bounding
/// sphere of instanceBound / SceneBuilder is for). Cost = n^3 * order^2 lines, each ~nseg + 40
/// bisection evaluations near the surface; preprocessing-grade, seconds at the defaults.
template <class Real, class Phi>
BodyProperties<Real> bodyProperties(const Phi& phi, Vec3<Real> lo, Vec3<Real> hi, int n = 32,
                                    int order = 5, int nseg = 8, Real density = Real(1)) {
  namespace bd = body_detail;
  if (!(hi.x > lo.x && hi.y > lo.y && hi.z > lo.z))
    throw std::invalid_argument("bodyProperties: empty box");
  if (n < 1)
    throw std::invalid_argument("bodyProperties: n must be >= 1");
  const Real H[3] = {(hi.x - lo.x) / n, (hi.y - lo.y) / n, (hi.z - lo.z) / n};
  const Real LO[3] = {lo.x, lo.y, lo.z};
  const int nq = quad_detail::gaussCount(order);

  // raw moments in the input frame: V, M[a] = ∫x_a, S[a][b] = ∫x_a x_b (symmetric)
  Real V = 0, M[3] = {0, 0, 0}, S[3][3] = {};

  for (int ci = 0; ci < n; ++ci)
    for (int cj = 0; cj < n; ++cj)
      for (int ck = 0; ck < n; ++ck) {
        const Real o[3] = {LO[0] + ci * H[0], LO[1] + cj * H[1], LO[2] + ck * H[2]};
        // height axis: the axis of the strongest central difference at the cell centre
        auto at = [&](Real a, Real b, Real c) {
          return phi(Vec3<Real>{o[0] + a * H[0], o[1] + b * H[1], o[2] + c * H[2]});
        };
        const Real q = Real(0.25);
        const Real g0 = at(Real(0.5) + q, Real(0.5), Real(0.5)) -
                        at(Real(0.5) - q, Real(0.5), Real(0.5));
        const Real g1 = at(Real(0.5), Real(0.5) + q, Real(0.5)) -
                        at(Real(0.5), Real(0.5) - q, Real(0.5));
        const Real g2 = at(Real(0.5), Real(0.5), Real(0.5) + q) -
                        at(Real(0.5), Real(0.5), Real(0.5) - q);
        auto ab = [](Real v) { return v < Real(0) ? -v : v; };
        int hA = 0;
        if (ab(g1) >= ab(g0) && ab(g1) >= ab(g2))
          hA = 1;
        else if (ab(g2) >= ab(g0) && ab(g2) >= ab(g1))
          hA = 2;
        const int t1 = (hA + 1) % 3, t2 = (hA + 2) % 3;
        const Real area = H[t1] * H[t2];

        for (int qi = 0; qi < nq; ++qi) {
          Real ti, wi;
          quad_detail::gaussLegendre01<Real>(order, qi, ti, wi);
          for (int qj = 0; qj < nq; ++qj) {
            Real tj, wj;
            quad_detail::gaussLegendre01<Real>(order, qj, tj, wj);
            Real p0[3] = {o[0], o[1], o[2]};
            p0[t1] += ti * H[t1];
            p0[t2] += tj * H[t2];
            Real d[3] = {0, 0, 0};
            d[hA] = H[hA];
            Real m0, m1, m2;
            bd::interiorMoments(phi, Vec3<Real>{p0[0], p0[1], p0[2]},
                                Vec3<Real>{d[0], d[1], d[2]}, nseg, m0, m1, m2);
            if (m0 == Real(0))
              continue;
            const Real W = wi * wj * area;
            const Real x0 = p0[hA], h = H[hA];
            const Real L0 = h * m0;
            const Real L1 = h * (x0 * m0 + h * m1);
            const Real L2 = h * (x0 * x0 * m0 + Real(2) * x0 * h * m1 + h * h * m2);
            const Real tw[3] = {p0[0], p0[1], p0[2]};  // transverse coords are constant on the line
            V += W * L0;
            M[hA] += W * L1;
            M[t1] += W * tw[t1] * L0;
            M[t2] += W * tw[t2] * L0;
            S[hA][hA] += W * L2;
            S[t1][t1] += W * tw[t1] * tw[t1] * L0;
            S[t2][t2] += W * tw[t2] * tw[t2] * L0;
            S[t1][t2] += W * tw[t1] * tw[t2] * L0;
            S[hA][t1] += W * tw[t1] * L1;
            S[hA][t2] += W * tw[t2] * L1;
          }
        }
      }
  // symmetrise the off-diagonal accumulators (each pair was accumulated once)
  S[1][0] = S[0][1] += S[1][0];
  S[2][0] = S[0][2] += S[2][0];
  S[2][1] = S[1][2] += S[2][1];

  BodyProperties<Real> bp;
  bp.volume = V;
  bp.mass = density * V;
  if (!(V > Real(0)))
    throw std::runtime_error(
        "bodyProperties: the field encloses no volume in this box -- check the sign convention "
        "(negative inside) and that the box contains the solid");
  bp.com = Vec3<Real>{M[0] / V, M[1] / V, M[2] / V};
  const Real c[3] = {bp.com.x, bp.com.y, bp.com.z};
  // inertia about the COM: I_ab = rho * [ delta_ab * sum_k (S_kk - V c_k^2) - (S_ab - V c_a c_b) ]
  Real cs[3][3];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      cs[a][b] = S[a][b] - V * c[a] * c[b];
  const Real trace = cs[0][0] + cs[1][1] + cs[2][2];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      bp.inertia[a][b] = density * ((a == b ? trace : Real(0)) - cs[a][b]);

  Real w[3], R[3][3];
  bd::eigSym3(bp.inertia, w, R);
  bd::resolveDegenerateFrame(w, R);
  // keep a proper rotation
  const Real det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
                   R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
                   R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
  if (det < Real(0))
    for (int r = 0; r < 3; ++r)
      R[r][0] = -R[r][0];
  for (int k = 0; k < 3; ++k)
    bp.principal[k] = w[k];
  for (int r = 0; r < 3; ++r)
    for (int cc = 0; cc < 3; ++cc)
      bp.rotation[r][cc] = R[r][cc];
  bp.quat = bd::quatFromRotation(R);
  return bp;
}

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_BODY_PROPERTIES_HPP

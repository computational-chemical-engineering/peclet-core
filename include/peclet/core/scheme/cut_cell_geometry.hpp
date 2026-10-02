/// @file
/// @brief core — the cut-cell geometric record of one cell from a piecewise-linear (PL) model of
/// its signed distance field: fluid fraction, face apertures, facet area vectors and centroids.
///
/// Design: flow `doc/scalar_ibm_design.md` §2 (decision D4, register entry "Scalar cut-cell
/// geometry is the fan-tetrahedron PL model"). The record is container-free — scalars and small
/// local arrays, no `Kokkos::View`, no grid indexing — so ONE copy serves flow's structured grid
/// (WO-2) and amr's leaves (§7.3).
///
/// **The PL model.** A cell carries 15 samples of the SDF φ (internal length units): its 8 corners,
/// its 6 face centres and its centre. Each face is split into 4 triangles (corner, next corner,
/// face centre) — the triangle fan of flow's `ccFaceOpenMS` — and each triangle is coned to the
/// cell centre, giving 24 tetrahedra of volume V/24. φ is linear on each. Shared faces carry
/// identical fans, so the model is conforming across cells.
///
/// **"Fluid" means φ > 0 strictly**: a vertex with φ = 0 is solid, so a wall lying exactly on a
/// face closes that face and κ = 0 never coexists with an open face.
///
/// κ, the six apertures and the facet pieces all come from that one polyhedron, so the PL
/// divergence theorem holds to round-off:
///
///     Σ_a A_a (aperture_{a+} − aperture_{a−}) e_a = areaPL          (A_a = face area)
///
/// **Conventions.** Frame = the cell centre at the origin, internal lengths, cell size h[3].
/// Corner index c = bx + 2·by + 4·bz (x fastest, the suite's axis order; bit set = high side),
/// i.e. corner c sits at ((bx − ½)h_x, (by − ½)h_y, (bz − ½)h_z). Face order −x, +x, −y, +y, −z,
/// +z. Area vectors point INTO the fluid. Everything is double (D7).
///
/// Host/device: `PECLET_CORE_CC_HD` (from `scheme/cut_cell_closure.hpp`) is
/// `KOKKOS_INLINE_FUNCTION` when Kokkos_Core.hpp was included BEFORE this header, plain `inline`
/// otherwise (host-only oracle builds).
///
/// Reserved, NOT implemented (design §7.1, §7.4): the plane source
/// `cutCellGeometryFromPlane(const double m[3], double alpha, const double h[3])` for PLIC cells.
#ifndef PECLET_CORE_SCHEME_CUT_CELL_GEOMETRY_HPP
#define PECLET_CORE_SCHEME_CUT_CELL_GEOMETRY_HPP

#include <cmath>

#include "peclet/core/scheme/cut_cell_closure.hpp"  // PECLET_CORE_CC_HD

namespace peclet::core::scheme {

/// Aperture snap threshold at BOTH ends (§2.3): a < 1e-3 → 0, a > 1 − 1e-3 → 1. The 1e-3 floor is
/// flow's pressure constant; snapping both ends keeps a_s = 1 − a^snap exact (linear exactness).
inline constexpr double kApertureSnap = 1e-3;
/// A facet piece with area < kPieceDropRel · min_a A_a is dropped (§2.2).
inline constexpr double kPieceDropRel = 1e-14;
/// ρ = |areaPL| / areaSum at or above which the cell carries ONE facet (§2.2).
inline constexpr double kSingleFacetRho = 0.5;

/// Facet classification of a cut cell (§2.2).
enum class CutCellKind : int {
  none = 0,       ///< no facet (uncut, or every piece below the drop threshold)
  single = 1,     ///< one facet (ρ ≥ 0.5, or the two-group split left a group empty)
  gap = 2,        ///< two facets, fluid between two walls
  thinSolid = 3,  ///< two facets, solid between two fluids (a counted resolution error)
};

/// The geometric record of one cell (§2.1). Double, cell-centred frame, internal lengths.
struct CutCellGeometry {
  double kappa;                ///< fluid volume fraction ∈ [0, 1]
  double aperture[6];          ///< open fraction of each face, −x,+x,−y,+y,−z,+z, UNSNAPPED
  double areaPL[3];            ///< net PL facet area vector (into the fluid)
  double areaSum;              ///< Σ |piece|
  int nFacet;                  ///< 0, 1 or 2
  double facetArea[2][3];      ///< per facet: area vector (into the fluid); unused slots 0
  double facetCentroid[2][3];  ///< per facet: centroid relative to the cell centre; unused 0
  CutCellKind kind;
};

namespace detail {

PECLET_CORE_CC_HD double ccgSqrt(double x) {
#ifdef KOKKOS_INLINE_FUNCTION
  return Kokkos::sqrt(x);
#else
  return std::sqrt(x);
#endif
}

PECLET_CORE_CC_HD double ccgAbs(double x) {
  return x < 0.0 ? -x : x;
}

PECLET_CORE_CC_HD double ccgMaxAbs3(double a, double b, double c) {
  double m = ccgAbs(a);
  m = ccgAbs(b) > m ? ccgAbs(b) : m;
  return ccgAbs(c) > m ? ccgAbs(c) : m;
}

PECLET_CORE_CC_HD double ccgMaxAbs4(double a, double b, double c, double d) {
  const double m = ccgMaxAbs3(a, b, c);
  return ccgAbs(d) > m ? ccgAbs(d) : m;
}

// Triangle, one positive vertex x > 0 >= y, z: F = x^2 / ((x - y)(x - z)).
PECLET_CORE_CC_HD double triOnePositive(double x, double y, double z) {
  const double den = (x - y) * (x - z);
  if (den > 0.0)
    return (x * x) / den;
  // Underflow only (every |value| below ~1e-154): F is scale-invariant, so rescale once.
  const double s = 1.0 / ccgMaxAbs3(x, y, z);
  x *= s;
  y *= s;
  z *= s;
  return (x * x) / ((x - y) * (x - z));
}

// Tetrahedron, one positive vertex a > 0 >= b1, b2, b3: F = a^3 / ((a-b1)(a-b2)(a-b3)).
PECLET_CORE_CC_HD double tetOnePositive(double a, double b1, double b2, double b3) {
  const double den = ((a - b1) * (a - b2)) * (a - b3);
  if (den > 0.0)
    return (a * a * a) / den;
  const double s = 1.0 / ccgMaxAbs4(a, b1, b2, b3);  // underflow only; F is scale-invariant
  a *= s;
  b1 *= s;
  b2 *= s;
  b3 *= s;
  return (a * a * a) / (((a - b1) * (a - b2)) * (a - b3));
}

// Tetrahedron, two positive vertices a1, a2 > 0 >= b1, b2 (§2.2; every term non-negative):
// F = [a1²a2² − (a1+a2)(b1+b2)a1a2 + b1b2(a1²+a1a2+a2²)] / [(a1−b1)(a1−b2)(a2−b1)(a2−b2)].
PECLET_CORE_CC_HD double tetTwoPositive(double a1, double a2, double b1, double b2) {
  double den = ((a1 - b1) * (a1 - b2)) * ((a2 - b1) * (a2 - b2));
  if (!(den > 0.0)) {
    const double s = 1.0 / ccgMaxAbs4(a1, a2, b1, b2);  // underflow only; F is scale-invariant
    a1 *= s;
    a2 *= s;
    b1 *= s;
    b2 *= s;
    den = ((a1 - b1) * (a1 - b2)) * ((a2 - b1) * (a2 - b2));
  }
  const double p = a1 * a2;
  const double num = p * p - (a1 + a2) * (b1 + b2) * p + b1 * b2 * (a1 * a1 + p + a2 * a2);
  return num / den;
}

}  // namespace detail

/// Fraction of a triangle on which the linear interpolant of its vertex values (a, b, c) is
/// strictly positive. Robust case formulas (§2.2), no subtraction of close numbers; the vertex
/// rotation mirrors flow's `ccTriFrac`.
PECLET_CORE_CC_HD double triPositiveFraction(double a, double b, double c) {
  const bool pa = a > 0.0, pb = b > 0.0, pc = c > 0.0;
  const int np = (pa ? 1 : 0) + (pb ? 1 : 0) + (pc ? 1 : 0);
  if (np == 3)
    return 1.0;
  if (np == 0)
    return 0.0;
  if (np == 1) {  // rotate the positive vertex to the front
    if (pa)
      return detail::triOnePositive(a, b, c);
    if (pb)
      return detail::triOnePositive(b, c, a);
    return detail::triOnePositive(c, a, b);
  }
  // np == 2: the fraction of the single non-positive vertex's mirror, F = 1 − b²/((b−a1)(b−a2))
  // = 1 − triOnePositive(−b, −a1, −a2) (the negative part as a one-positive triangle of −φ).
  if (!pa)
    return 1.0 - detail::triOnePositive(-a, -b, -c);
  if (!pb)
    return 1.0 - detail::triOnePositive(-b, -c, -a);
  return 1.0 - detail::triOnePositive(-c, -a, -b);
}

/// Fraction of a tetrahedron on which the linear interpolant of its vertex values is strictly
/// positive (§2.2). Vertices with value exactly 0 count as non-positive (solid).
PECLET_CORE_CC_HD double tetPositiveFraction(double v0, double v1, double v2, double v3) {
  const double v[4] = {v0, v1, v2, v3};
  double pos[4], neg[4];
  int np = 0, nn = 0;
  for (int k = 0; k < 4; ++k) {
    if (v[k] > 0.0)
      pos[np++] = v[k];
    else
      neg[nn++] = v[k];
  }
  if (np == 0)
    return 0.0;
  if (np == 4)
    return 1.0;
  if (np == 1)
    return detail::tetOnePositive(pos[0], neg[0], neg[1], neg[2]);
  if (np == 3)  // 1 − b³/((b−a1)(b−a2)(b−a3)) = 1 − tetOnePositive(−b, −a1, −a2, −a3)
    return 1.0 - detail::tetOnePositive(-neg[0], -pos[0], -pos[1], -pos[2]);
  return detail::tetTwoPositive(pos[0], pos[1], neg[0], neg[1]);
}

/// PL aperture of one face from its fan (§2.2): the corner values in loop order c00, c10, c11, c01
/// over the face's tangent axes (t1 < t2) and the face-centre value cc,
/// ¼ (F(c00,c10,cc) + F(c10,c11,cc) + F(c11,c01,cc) + F(c01,c00,cc)), summed in that order. The
/// face kernel of a container and `cutCellGeometryFanTet` both call this, so a face's aperture is
/// the same bits from either cell and from the face kernel.
PECLET_CORE_CC_HD double fanFaceAperture(double v00, double v10, double v11, double v01,
                                         double cc) {
  return 0.25 * (triPositiveFraction(v00, v10, cc) + triPositiveFraction(v10, v11, cc) +
                 triPositiveFraction(v11, v01, cc) + triPositiveFraction(v01, v00, cc));
}

/// Snap a face aperture at both ends (§2.3): a < 1e-3 → 0, a > 1 − 1e-3 → 1, else a.
PECLET_CORE_CC_HD double snapAperture(double a) {
  if (a < kApertureSnap)
    return 0.0;
  if (a > 1.0 - kApertureSnap)
    return 1.0;
  return a;
}

/// Face area of axis a for cell size h: the product of the two other spacings.
PECLET_CORE_CC_HD double cutCellFaceArea(int a, const double h[3]) {
  return a == 0 ? h[1] * h[2] : (a == 1 ? h[0] * h[2] : h[0] * h[1]);
}

/// The area vector the apertures imply through the divergence theorem (§2.3):
/// out_a = A_a (ap[2a+1] − ap[2a]), A_a the a-face area. Applied to the UNSNAPPED apertures it is
/// `areaPL` (the PL identity); applied to the snapped ones it is the operator's A⃗^snap.
PECLET_CORE_CC_HD void facetAreaVector(const double ap[6], const double h[3], double out[3]) {
  for (int a = 0; a < 3; ++a)
    out[a] = cutCellFaceArea(a, h) * (ap[2 * a + 1] - ap[2 * a]);
}

namespace detail {

// Position of corner c (bits x, y, z = bit 0, 1, 2) relative to the cell centre.
PECLET_CORE_CC_HD void ccgCornerPos(int c, const double h[3], double x[3]) {
  for (int a = 0; a < 3; ++a)
    x[a] = (((c >> a) & 1) ? 0.5 : -0.5) * h[a];
}

// The two tangent axes of a face of axis a, ascending (the ccFaceOpenMS t1, t2).
PECLET_CORE_CC_HD void ccgTangents(int a, int& t1, int& t2) {
  t1 = (a == 0) ? 1 : 0;
  t2 = (a == 2) ? 1 : 2;
}

// Corner k (0..3) of face f's fan loop: c00, c10, c11, c01 in the (t1, t2) bits.
PECLET_CORE_CC_HD int ccgFanCorner(int f, int k) {
  const int a = f >> 1, side = f & 1;
  int t1, t2;
  ccgTangents(a, t1, t2);
  const int u = (k == 1 || k == 2) ? 1 : 0;
  const int w = (k >= 2) ? 1 : 0;
  return (side << a) | (u << t1) | (w << t2);
}

// Tetrahedron (f, k): face f's fan triangle k = (corner k, corner k+1, face centre) coned to the
// cell centre. Vertex values and positions in that order.
PECLET_CORE_CC_HD void ccgFanTet(int f, int k, const double corner[8], const double face[6],
                                 double centre, const double h[3], double val[4],
                                 double pos[4][3]) {
  const int c0 = ccgFanCorner(f, k), c1 = ccgFanCorner(f, (k + 1) & 3);
  val[0] = corner[c0];
  val[1] = corner[c1];
  val[2] = face[f];
  val[3] = centre;
  ccgCornerPos(c0, h, pos[0]);
  ccgCornerPos(c1, h, pos[1]);
  const int a = f >> 1;
  for (int d = 0; d < 3; ++d) {
    pos[2][d] = 0.0;
    pos[3][d] = 0.0;
  }
  pos[2][a] = ((f & 1) ? 0.5 : -0.5) * h[a];
}

// Edge crossing x = x_p + t (x_n − x_p), t = φ_p / (φ_p − φ_n) ∈ (0, 1].
PECLET_CORE_CC_HD void ccgCross(double vp, double vn, const double xp[3], const double xn[3],
                                double x[3]) {
  const double t = vp / (vp - vn);
  for (int d = 0; d < 3; ++d)
    x[d] = xp[d] + t * (xn[d] - xp[d]);
}

PECLET_CORE_CC_HD void ccgCrossProd(const double u[3], const double v[3], double w[3]) {
  w[0] = u[1] * v[2] - u[2] * v[1];
  w[1] = u[2] * v[0] - u[0] * v[2];
  w[2] = u[0] * v[1] - u[1] * v[0];
}

PECLET_CORE_CC_HD double ccgNorm(const double v[3]) {
  return ccgSqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// The zero-set piece of one tetrahedron with 0 < n+ < 4 (§2.2): area vector `vec` oriented toward
// a positive vertex, its magnitude `area`, and centroid `cen`. Returns false when the tet is not
// cut (n+ = 0 or 4).
PECLET_CORE_CC_HD bool ccgTetPiece(const double val[4], const double pos[4][3], double vec[3],
                                   double& area, double cen[3]) {
  int ip[4], in[4];
  int np = 0, nn = 0;
  for (int k = 0; k < 4; ++k) {
    if (val[k] > 0.0)
      ip[np++] = k;
    else
      in[nn++] = k;
  }
  if (np == 0 || np == 4)
    return false;
  const int pRef = ip[0];  // the positive vertex the area vector is oriented toward
  double X[4][3];
  if (np == 2) {
    // Quad X11, X12, X22, X21 (X_ij on edge p_i – n_j); area ½ (X2 − X0) × (X3 − X1).
    ccgCross(val[ip[0]], val[in[0]], pos[ip[0]], pos[in[0]], X[0]);
    ccgCross(val[ip[0]], val[in[1]], pos[ip[0]], pos[in[1]], X[1]);
    ccgCross(val[ip[1]], val[in[1]], pos[ip[1]], pos[in[1]], X[2]);
    ccgCross(val[ip[1]], val[in[0]], pos[ip[1]], pos[in[0]], X[3]);
    double d02[3], d13[3];
    for (int d = 0; d < 3; ++d) {
      d02[d] = X[2][d] - X[0][d];
      d13[d] = X[3][d] - X[1][d];
    }
    ccgCrossProd(d02, d13, vec);
    for (int d = 0; d < 3; ++d)
      vec[d] *= 0.5;
    // Centroid: area-weighted mean of the triangles (X0, X1, X2) and (X0, X2, X3).
    double d01[3], d03[3], tA[3], tB[3];
    for (int d = 0; d < 3; ++d) {
      d01[d] = X[1][d] - X[0][d];
      d03[d] = X[3][d] - X[0][d];
    }
    ccgCrossProd(d01, d02, tA);
    ccgCrossProd(d02, d03, tB);
    const double wA = 0.5 * ccgNorm(tA), wB = 0.5 * ccgNorm(tB);
    const double wS = wA + wB;
    for (int d = 0; d < 3; ++d) {
      const double cA = (X[0][d] + X[1][d] + X[2][d]) / 3.0;
      const double cB = (X[0][d] + X[2][d] + X[3][d]) / 3.0;
      cen[d] = wS > 0.0 ? (wA * cA + wB * cB) / wS : 0.25 * (X[0][d] + X[1][d] + X[2][d] + X[3][d]);
    }
  } else {
    // Triangle: n+ = 1 → crossings on (p, n1), (p, n2), (p, n3); n+ = 3 → on (p1, n), (p2, n),
    // (p3, n). Area ½ (X1 − X0) × (X2 − X0).
    for (int k = 0; k < 3; ++k) {
      const int p = (np == 1) ? ip[0] : ip[k];
      const int n = (np == 1) ? in[k] : in[0];
      ccgCross(val[p], val[n], pos[p], pos[n], X[k]);
    }
    double d01[3], d02[3];
    for (int d = 0; d < 3; ++d) {
      d01[d] = X[1][d] - X[0][d];
      d02[d] = X[2][d] - X[0][d];
    }
    ccgCrossProd(d01, d02, vec);
    for (int d = 0; d < 3; ++d) {
      vec[d] *= 0.5;
      cen[d] = (X[0][d] + X[1][d] + X[2][d]) / 3.0;
    }
  }
  // Orient toward the positive vertex.
  const double dot = vec[0] * (pos[pRef][0] - X[0][0]) + vec[1] * (pos[pRef][1] - X[0][1]) +
                     vec[2] * (pos[pRef][2] - X[0][2]);
  if (dot < 0.0)
    for (int d = 0; d < 3; ++d)
      vec[d] = -vec[d];
  area = ccgNorm(vec);
  return true;
}

}  // namespace detail

/// The cut-cell record of one cell from its 15 PL samples (§2.2): corner[8] (index bx+2by+4bz),
/// face[6] (−x,+x,−y,+y,−z,+z), the centre value, and the cell size h[3] (internal lengths).
///
/// - κ = (1/24) Σ_tets F_tet; aperture_f = ¼ Σ_{4 fan triangles of f} F_tri (unsnapped).
/// - Pieces: the zero set of each cut tetrahedron; pieces with area < 1e-14 · min_a A_a dropped.
///   areaPL = Σ piece vectors, areaSum = Σ |piece|, ρ = |areaPL| / areaSum.
/// - ρ ≥ 0.5 → `single` (facet 0 = areaPL at the areaSum-weighted piece centroid). Otherwise the
///   pieces split into two groups by the sign of n_piece · n_ref (n_ref = the largest piece; a
///   piece with n_piece · n_ref ≥ 0 joins the reference group A, which is facet 0) — `gap` if
///   (x̄_B − x̄_A) · n_A > 0, else `thinSolid`; an empty group falls back to `single`.
PECLET_CORE_CC_HD CutCellGeometry cutCellGeometryFanTet(const double corner[8],
                                                        const double face[6], double centre,
                                                        const double h[3]) {
  CutCellGeometry g;
  g.kappa = 0.0;
  g.areaSum = 0.0;
  g.nFacet = 0;
  g.kind = CutCellKind::none;
  for (int d = 0; d < 3; ++d) {
    g.areaPL[d] = 0.0;
    g.facetArea[0][d] = g.facetArea[1][d] = 0.0;
    g.facetCentroid[0][d] = g.facetCentroid[1][d] = 0.0;
  }

  // Apertures: face f's fan (c00, c10, cc), (c10, c11, cc), (c11, c01, cc), (c01, c00, cc),
  // summed in that order — the same triangles and order from both cells sharing the face.
  for (int f = 0; f < 6; ++f) {
    const double cc = face[f];
    const double v00 = corner[detail::ccgFanCorner(f, 0)];
    const double v10 = corner[detail::ccgFanCorner(f, 1)];
    const double v11 = corner[detail::ccgFanCorner(f, 2)];
    const double v01 = corner[detail::ccgFanCorner(f, 3)];
    g.aperture[f] = fanFaceAperture(v00, v10, v11, v01, cc);
  }

  // Uniform sign over all 15 samples: κ = 1 or 0, no facet (the apertures above are then exactly
  // 1 or 0 as well).
  bool anyPos = centre > 0.0, anyNonPos = !(centre > 0.0);
  for (int c = 0; c < 8; ++c) {
    anyPos = anyPos || corner[c] > 0.0;
    anyNonPos = anyNonPos || !(corner[c] > 0.0);
  }
  for (int f = 0; f < 6; ++f) {
    anyPos = anyPos || face[f] > 0.0;
    anyNonPos = anyNonPos || !(face[f] > 0.0);
  }
  if (!anyPos || !anyNonPos) {
    g.kappa = anyPos ? 1.0 : 0.0;
    return g;
  }

  double minA = cutCellFaceArea(0, h);
  minA = cutCellFaceArea(1, h) < minA ? cutCellFaceArea(1, h) : minA;
  minA = cutCellFaceArea(2, h) < minA ? cutCellFaceArea(2, h) : minA;
  const double dropArea = kPieceDropRel * minA;

  // Pass 1: κ, the net area vector, Σ|piece|, the weighted centroid and the largest piece.
  double ksum = 0.0;
  double wc[3] = {0.0, 0.0, 0.0};
  double refVec[3] = {0.0, 0.0, 0.0};
  double refArea = -1.0;
  for (int f = 0; f < 6; ++f) {
    for (int k = 0; k < 4; ++k) {
      double val[4], pos[4][3];
      detail::ccgFanTet(f, k, corner, face, centre, h, val, pos);
      ksum += tetPositiveFraction(val[0], val[1], val[2], val[3]);
      double vec[3], cen[3], area;
      if (!detail::ccgTetPiece(val, pos, vec, area, cen) || area < dropArea)
        continue;
      for (int d = 0; d < 3; ++d) {
        g.areaPL[d] += vec[d];
        wc[d] += area * cen[d];
      }
      g.areaSum += area;
      if (area > refArea) {
        refArea = area;
        for (int d = 0; d < 3; ++d)
          refVec[d] = vec[d];
      }
    }
  }
  g.kappa = ksum / 24.0;
  if (!(g.areaSum > 0.0))
    return g;  // every piece below the drop threshold: no facet

  const double rho = detail::ccgNorm(g.areaPL) / g.areaSum;
  if (rho < kSingleFacetRho) {
    // Pass 2: split the pieces into the reference group A (n·n_ref >= 0) and B.
    double vA[3] = {0.0, 0.0, 0.0}, vB[3] = {0.0, 0.0, 0.0};
    double cA[3] = {0.0, 0.0, 0.0}, cB[3] = {0.0, 0.0, 0.0};
    double sA = 0.0, sB = 0.0;
    for (int f = 0; f < 6; ++f) {
      for (int k = 0; k < 4; ++k) {
        double val[4], pos[4][3];
        detail::ccgFanTet(f, k, corner, face, centre, h, val, pos);
        double vec[3], cen[3], area;
        if (!detail::ccgTetPiece(val, pos, vec, area, cen) || area < dropArea)
          continue;
        const double dot = vec[0] * refVec[0] + vec[1] * refVec[1] + vec[2] * refVec[2];
        double* v = dot >= 0.0 ? vA : vB;
        double* c = dot >= 0.0 ? cA : cB;
        for (int d = 0; d < 3; ++d) {
          v[d] += vec[d];
          c[d] += area * cen[d];
        }
        (dot >= 0.0 ? sA : sB) += area;
      }
    }
    if (sA > 0.0 && sB > 0.0) {
      for (int d = 0; d < 3; ++d) {
        g.facetArea[0][d] = vA[d];
        g.facetArea[1][d] = vB[d];
        g.facetCentroid[0][d] = cA[d] / sA;
        g.facetCentroid[1][d] = cB[d] / sB;
      }
      const double sep = (g.facetCentroid[1][0] - g.facetCentroid[0][0]) * vA[0] +
                         (g.facetCentroid[1][1] - g.facetCentroid[0][1]) * vA[1] +
                         (g.facetCentroid[1][2] - g.facetCentroid[0][2]) * vA[2];
      g.nFacet = 2;
      g.kind = sep > 0.0 ? CutCellKind::gap : CutCellKind::thinSolid;
      return g;
    }
  }
  g.nFacet = 1;
  g.kind = CutCellKind::single;
  for (int d = 0; d < 3; ++d) {
    g.facetArea[0][d] = g.areaPL[d];
    g.facetCentroid[0][d] = wc[d] / g.areaSum;
  }
  return g;
}

}  // namespace peclet::core::scheme

#endif  // PECLET_CORE_SCHEME_CUT_CELL_GEOMETRY_HPP

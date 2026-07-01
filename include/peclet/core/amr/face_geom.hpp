// core — FaceGeom: the static (sub)face geometry CSR of the collocated AMR projection.
//
// Factored out of flow.hpp into its own header so the device face-geometry assembler
// (facegeom_assembly.hpp) and the flow driver (flow.hpp) can both name the type without a
// circular include (flow.hpp consumes the assembler, the assembler produces a FaceGeom).
//
// Per (sub)face (matching AmrPoisson::forEachFaceFull, 2:1 sub-faces): neighbour, axis, dir, α·area
// (divergence weight), raw area (advection flux), face-normal distance, openness α (gradient gate),
// and the two SOU upstream-probe leaves. Per cell: 1/V and a fluid flag.
//
// Requires a Kokkos build (View).
#ifndef PECLET_CORE_AMR_FACE_GEOM_HPP
#define PECLET_CORE_AMR_FACE_GEOM_HPP

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::core::amr {

struct FaceGeom {
  View<Index> start;       ///< CSR row offsets, size n+1
  View<Index> nbr;         ///< neighbour leaf per face, size nFaces
  View<int> axis;          ///< face axis 0/1/2, size nFaces
  View<int> dir;           ///< face direction +1/-1, size nFaces
  View<double> alphaArea;  ///< α·area (physical) per face, size nFaces
  View<double> rawArea;    ///< raw face area (physical, no openness) per face — advection flux
  View<double> dist;       ///< face-normal distance (physical) per face, size nFaces
  View<double> alpha;      ///< openness per face (gradient gate), size nFaces
  View<Index> upupI;       ///< upstream-of-i probe (periodicNeighbor(i,axis,−dir)) — SOU, size nFaces
  View<Index> upupJ;       ///< upstream-of-j probe (periodicNeighbor(j,axis,+dir)) — SOU, size nFaces
  View<double> invVol;     ///< 1/V_i per cell, size n
  View<char> fluid;        ///< per-cell fluid flag, size n
  Index n = 0;
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_AMR_FACE_GEOM_HPP

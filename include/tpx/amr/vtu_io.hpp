// transport-core — VTK UnstructuredGrid (.vtu) output for an adaptive octree.
//
// A graded octree can't be a single VTK ImageData (vti_io.hpp), so each leaf is
// emitted as one cell — a hexahedron (3D) or quad (2D) — carrying a per-leaf
// CellData scalar. Points are written per-cell (not shared), which keeps the
// writer trivial and is what ParaView expects for a cell-wise AMR dump. ASCII,
// human-readable, openable in ParaView.
//
// Header-only, guarded by TPX_HAVE_MORTON.
#ifndef TPX_AMR_VTU_IO_HPP
#define TPX_AMR_VTU_IO_HPP

#ifdef TPX_HAVE_MORTON

#include <array>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/common/types.hpp"

namespace tpx::amr {

namespace detail {
// Corner offsets in VTK cell-point order. 3D: VTK_HEXAHEDRON (=12); 2D: VTK_QUAD (=9).
template <int Dim>
inline const std::array<std::array<int, Dim>, (1 << Dim)>& vtkCorners();

template <>
inline const std::array<std::array<int, 3>, 8>& vtkCorners<3>() {
  static const std::array<std::array<int, 3>, 8> c{{{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 0, 1},
                                                    {1, 0, 1},
                                                    {1, 1, 1},
                                                    {0, 1, 1}}};
  return c;
}
template <>
inline const std::array<std::array<int, 2>, 4>& vtkCorners<2>() {
  static const std::array<std::array<int, 2>, 4> c{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
  return c;
}
}  // namespace detail

/// Write a BlockOctree (+ world geometry + a per-leaf scalar) as a VTK
/// UnstructuredGrid (.vtu), ASCII, one cell per leaf.
template <int Dim, unsigned Bits>
void writeVtu(const std::string& path, const BlockOctree<Dim, Bits>& t,
              const AmrGeometry<Dim>& geo, const std::string& name,
              const std::vector<double>& cellData) {
  static_assert(Dim == 2 || Dim == 3, "writeVtu supports 2D/3D");
  const Index nLeaf = t.numLeaves();
  if (static_cast<Index>(cellData.size()) != nLeaf)
    throw std::runtime_error("writeVtu: cellData size != numLeaves");

  std::ofstream f(path);
  if (!f) throw std::runtime_error("writeVtu: cannot open " + path);

  constexpr int NP = 1 << Dim;          // points per cell
  const int cellType = (Dim == 3) ? 12 : 9;
  const auto& corners = detail::vtkCorners<Dim>();
  const std::size_t npts = static_cast<std::size_t>(nLeaf) * NP;

  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
    << "  <UnstructuredGrid>\n"
    << "    <Piece NumberOfPoints=\"" << npts << "\" NumberOfCells=\"" << nLeaf << "\">\n";

  // --- points (per-cell, in VTK corner order; always 3 components) ---
  f << "      <Points>\n"
    << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  f.setf(std::ios::scientific);
  f.precision(9);
  for (Index i = 0; i < nLeaf; ++i) {
    auto b = t.bounds(i);
    const auto lo = b[0];
    const Index sz = Index(1) << t.level(i);
    for (int c = 0; c < NP; ++c) {
      Real xyz[3] = {0, 0, 0};
      for (int d = 0; d < Dim; ++d)
        xyz[d] = geo.origin[d] + static_cast<Real>(lo[d] + corners[c][d] * sz) * geo.h0;
      f << xyz[0] << ' ' << xyz[1] << ' ' << xyz[2] << '\n';
    }
  }
  f << "        </DataArray>\n      </Points>\n";

  // --- cells ---
  f << "      <Cells>\n"
    << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for (std::size_t p = 0; p < npts; ++p) f << p << (((p + 1) % NP == 0) ? '\n' : ' ');
  f << "        </DataArray>\n"
    << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
  for (Index i = 0; i < nLeaf; ++i) f << static_cast<std::size_t>(i + 1) * NP << '\n';
  f << "        </DataArray>\n"
    << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (Index i = 0; i < nLeaf; ++i) f << cellType << '\n';
  f << "        </DataArray>\n      </Cells>\n";

  // --- cell data ---
  f << "      <CellData Scalars=\"" << name << "\">\n"
    << "        <DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";
  for (Index i = 0; i < nLeaf; ++i) f << cellData[static_cast<std::size_t>(i)] << '\n';
  f << "        </DataArray>\n      </CellData>\n";

  f << "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
}

/// Convenience overload taking a LeafField<double>.
template <int Dim, unsigned Bits>
void writeVtu(const std::string& path, const BlockOctree<Dim, Bits>& t,
              const AmrGeometry<Dim>& geo, const std::string& name,
              const LeafField<double>& field) {
  writeVtu(path, t, geo, name, field.values);
}

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_VTU_IO_HPP

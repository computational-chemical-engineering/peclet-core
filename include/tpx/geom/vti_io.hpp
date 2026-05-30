// transport-core — VTK ImageData (.vti) I/O for sampled SDF / scalar grid fields.
//
// Writes/reads a GridSdf as an ImageData PointData scalar in ASCII format — human-readable, exact
// float round-trip, and openable in ParaView. This is the interchange format cfd-gpu and packing-gpu
// already use for SDF/field grids; consolidating their readers onto this is a follow-up. The reader
// is tolerant (attribute scan, not a full XML parse) but expects the ASCII inline layout this writer
// emits; binary/base64 "appended" VTI (what some existing files use) is a noted TODO.
#ifndef TPX_GEOM_VTI_IO_HPP
#define TPX_GEOM_VTI_IO_HPP

#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>

#include "tpx/common/types.hpp"
#include "tpx/geom/grid_sdf.hpp"

namespace tpx::geom {

/// Write a GridSdf as a VTK ImageData (.vti), PointData Float32 scalar, ASCII.
inline void writeVti(const std::string& path, const GridSdf& g, const std::string& name = "sdf") {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("writeVti: cannot open " + path);
  const Index nx = g.dims[0], ny = g.dims[1], nz = g.dims[2];
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
    << "  <ImageData WholeExtent=\"0 " << nx - 1 << " 0 " << ny - 1 << " 0 " << nz - 1 << "\""
    << " Origin=\"" << g.origin[0] << " " << g.origin[1] << " " << g.origin[2] << "\""
    << " Spacing=\"" << g.spacing[0] << " " << g.spacing[1] << " " << g.spacing[2] << "\">\n"
    << "    <Piece Extent=\"0 " << nx - 1 << " 0 " << ny - 1 << " 0 " << nz - 1 << "\">\n"
    << "      <PointData Scalars=\"" << name << "\">\n"
    << "        <DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
  f.setf(std::ios::scientific);
  f.precision(9);  // 9 significant digits round-trips float32 exactly
  for (std::size_t i = 0; i < g.values.size(); ++i) {
    f << g.values[i] << (((i + 1) % static_cast<std::size_t>(nx) == 0) ? '\n' : ' ');
  }
  f << "\n        </DataArray>\n      </PointData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n";
}

namespace detail {
inline std::string vtiAttr(const std::string& s, const std::string& key) {
  std::string needle = key + "=\"";
  auto p = s.find(needle);
  if (p == std::string::npos) throw std::runtime_error("readVti: missing attribute " + key);
  p += needle.size();
  auto e = s.find('"', p);
  return s.substr(p, e - p);
}
}  // namespace detail

/// Read a VTK ImageData (.vti) written by writeVti (ASCII PointData scalar) into a GridSdf.
inline GridSdf readVti(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("readVti: cannot open " + path);
  std::stringstream buf;
  buf << f.rdbuf();
  const std::string s = buf.str();

  GridSdf g;
  Index x0, x1, y0, y1, z0, z1;
  std::istringstream(detail::vtiAttr(s, "WholeExtent")) >> x0 >> x1 >> y0 >> y1 >> z0 >> z1;
  g.dims = {x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1};
  std::istringstream(detail::vtiAttr(s, "Origin")) >> g.origin[0] >> g.origin[1] >> g.origin[2];
  std::istringstream(detail::vtiAttr(s, "Spacing")) >> g.spacing[0] >> g.spacing[1] >> g.spacing[2];

  auto da = s.find("<DataArray");
  if (da == std::string::npos) throw std::runtime_error("readVti: no DataArray");
  auto open = s.find('>', da);
  auto close = s.find("</DataArray>", open);
  std::istringstream data(s.substr(open + 1, close - open - 1));

  std::size_t n = static_cast<std::size_t>(g.dims[0]) * g.dims[1] * g.dims[2];
  g.values.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    float v = 0.0f;
    data >> v;
    g.values[i] = v;
  }
  return g;
}

}  // namespace tpx::geom

#endif  // TPX_GEOM_VTI_IO_HPP

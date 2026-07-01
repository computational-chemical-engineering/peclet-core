/// @file vti_io.hpp
/// @brief VTK ImageData (.vti) I/O for sampled SDF / scalar / vector grid fields.
///
/// Writes/reads a GridSdf as an ImageData PointData scalar in ASCII format — human-readable, exact
/// float round-trip, and openable in ParaView. This is the interchange format `flow` and `dem`
/// already use for SDF/field grids; consolidating their readers onto this is a follow-up. The reader
/// is tolerant (attribute scan, not a full XML parse) but expects the ASCII inline layout this writer
/// emits; binary/base64 "appended" VTI (what some existing files use) is a noted TODO.
#ifndef PECLET_CORE_GEOM_VTI_IO_HPP
#define PECLET_CORE_GEOM_VTI_IO_HPP

#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>

#include "peclet/core/common/types.hpp"
#include "peclet/core/geom/grid_sdf.hpp"

namespace peclet::core::geom {

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

/// A sampled vector field on a regular grid (3 interleaved components per point, x-fastest).
struct VtiVector {
  IVec<3> dims{};
  Vec<3> origin{};
  Vec<3> spacing{1, 1, 1};
  std::vector<float> values;  // size = 3 * dims[0]*dims[1]*dims[2], interleaved (vx,vy,vz)
};

/// Write a 3-component vector field as a VTK ImageData PointData vector (ASCII), ParaView-openable.
inline void writeVtiVector(const std::string& path, const VtiVector& g,
                           const std::string& name = "vector") {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("writeVtiVector: cannot open " + path);
  const Index nx = g.dims[0], ny = g.dims[1], nz = g.dims[2];
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
    << "  <ImageData WholeExtent=\"0 " << nx - 1 << " 0 " << ny - 1 << " 0 " << nz - 1 << "\""
    << " Origin=\"" << g.origin[0] << " " << g.origin[1] << " " << g.origin[2] << "\""
    << " Spacing=\"" << g.spacing[0] << " " << g.spacing[1] << " " << g.spacing[2] << "\">\n"
    << "    <Piece Extent=\"0 " << nx - 1 << " 0 " << ny - 1 << " 0 " << nz - 1 << "\">\n"
    << "      <PointData Vectors=\"" << name << "\">\n"
    << "        <DataArray type=\"Float32\" Name=\"" << name
    << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  f.setf(std::ios::scientific);
  f.precision(9);
  std::size_t npts = static_cast<std::size_t>(nx) * ny * nz;
  for (std::size_t p = 0; p < npts; ++p) {
    f << g.values[3 * p] << ' ' << g.values[3 * p + 1] << ' ' << g.values[3 * p + 2] << '\n';
  }
  f << "        </DataArray>\n      </PointData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n";
}

/// Read a vector-field VTI written by writeVtiVector.
inline VtiVector readVtiVector(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("readVtiVector: cannot open " + path);
  std::stringstream buf;
  buf << f.rdbuf();
  const std::string s = buf.str();

  VtiVector g;
  Index x0, x1, y0, y1, z0, z1;
  std::istringstream(detail::vtiAttr(s, "WholeExtent")) >> x0 >> x1 >> y0 >> y1 >> z0 >> z1;
  g.dims = {x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1};
  std::istringstream(detail::vtiAttr(s, "Origin")) >> g.origin[0] >> g.origin[1] >> g.origin[2];
  std::istringstream(detail::vtiAttr(s, "Spacing")) >> g.spacing[0] >> g.spacing[1] >> g.spacing[2];

  auto da = s.find("<DataArray");
  auto open = s.find('>', da);
  auto close = s.find("</DataArray>", open);
  std::istringstream data(s.substr(open + 1, close - open - 1));
  std::size_t n3 = 3u * static_cast<std::size_t>(g.dims[0]) * g.dims[1] * g.dims[2];
  g.values.resize(n3);
  for (std::size_t i = 0; i < n3; ++i) {
    float v = 0.0f;
    data >> v;
    g.values[i] = v;
  }
  return g;
}

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_VTI_IO_HPP

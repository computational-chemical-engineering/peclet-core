// transport-core — Python surface for the AMR octree (tpx::amr).
//
// A pybind11 module exposing the host adaptive-mesh-refinement path: the per-block BlockOctree
// (serial) and the MPI DistributedOctree (ORB over root cells), so an mpi4py driver can build a
// graded octree, refine it to a signed-distance surface, read leaf geometry + a per-leaf field as
// numpy, load-rebalance the distributed octree, gather face-neighbour values, and export VTU.
//
// Conventions (see ../docs/CONVENTIONS.md):
//   * 3D only here (Bits=21, codes in a 64-bit int). x-fastest where a linear index appears.
//   * Leaf arrays are in the octree's Z-order slot order; every per-leaf numpy array (centers,
//     sizes, levels, codes, fields) is indexed by that same slot, length num_leaves.
//   * Geometry: a leaf's world centre = origin + h0 * fine_centre; a leaf at refinement `level`
//     is h0 * 2**level wide. Root cells sit at level=lmax; level decreases toward 0 (finest).
//
// AMR is guarded by TPX_HAVE_MORTON, so this module REQUIRES the morton sibling checkout — its
// CMake points the include path at ../../morton/include and defines TPX_HAVE_MORTON. MPI is assumed
// already initialized by the host (import mpi4py.MPI first); the distributed class uses
// MPI_COMM_WORLD and never calls MPI_Init/Finalize.
//
// Build: see python/CMakeLists.txt (the tpx_amr target).
#include <mpi.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/distributed_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/amr/refine.hpp"
#include "tpx/amr/vtu_io.hpp"
#include "tpx/common/types.hpp"
#include "tpx/geom/sdf.hpp"

namespace py = pybind11;
using namespace tpx;

namespace {

using BO = amr::BlockOctree<3>;       // 3D, Bits=21 (default) — codes fit a uint64
using DO = amr::DistributedOctree<3>;
using Code = BO::Code;
using Coord = BO::Coord;

// ---- numpy leaf-geometry helpers (one row per leaf, Z-order slot order) ------------------------

// World centre of every leaf -> (N,3) float64.
py::array_t<double> leafCenters(const BO& t, const amr::AmrGeometry<3>& geo) {
  const Index n = t.numLeaves();
  py::array_t<double> a({(py::ssize_t)n, (py::ssize_t)3});
  auto r = a.mutable_unchecked<2>();
  for (Index i = 0; i < n; ++i) {
    Vec<3> c = geo.center(t.bounds(i));
    r(i, 0) = c[0];
    r(i, 1) = c[1];
    r(i, 2) = c[2];
  }
  return a;
}

// World width of every leaf (h0 * 2**level) -> (N,) float64.
py::array_t<double> leafSizes(const BO& t, const amr::AmrGeometry<3>& geo) {
  const Index n = t.numLeaves();
  py::array_t<double> a(n);
  auto r = a.mutable_unchecked<1>();
  for (Index i = 0; i < n; ++i) r(i) = geo.leafSize(t.level(i));
  return a;
}

// Refinement level of every leaf -> (N,) int32 (lmax at the root, 0 finest).
py::array_t<std::int32_t> leafLevels(const BO& t) {
  const Index n = t.numLeaves();
  py::array_t<std::int32_t> a(n);
  auto r = a.mutable_unchecked<1>();
  for (Index i = 0; i < n; ++i) r(i) = static_cast<std::int32_t>(t.level(i));
  return a;
}

// Block-local Morton origin code of every leaf -> (N,) uint64.
py::array_t<std::uint64_t> leafCodes(const BO& t) {
  const Index n = t.numLeaves();
  py::array_t<std::uint64_t> a(n);
  auto r = a.mutable_unchecked<1>();
  for (Index i = 0; i < n; ++i) r(i) = static_cast<std::uint64_t>(t.code(i));
  return a;
}

// Validate a per-leaf field numpy array and view it as a contiguous vector<double>.
std::vector<double> asField(const BO& t, py::array_t<double> field, const char* who) {
  auto f = field.unchecked<1>();
  if (f.shape(0) != t.numLeaves())
    throw std::runtime_error(std::string(who) + ": field length != num_leaves");
  std::vector<double> v(static_cast<std::size_t>(t.numLeaves()));
  for (Index i = 0; i < t.numLeaves(); ++i) v[static_cast<std::size_t>(i)] = f(i);
  return v;
}

// ---- serial single-block octree ----------------------------------------------------------------

// A per-block adaptive octree with its world placement (origin + uniform finest spacing h0).
// Wraps tpx::amr::BlockOctree<3> + AmrGeometry<3>: build a uniform brick, refine toward a surface,
// query leaves, and read leaf geometry / fields as numpy. The serial / single-rank form; for the
// distributed (MPI) octree use DistributedOctree below.
class Octree {
 public:
  Octree(std::array<long, 3> brick, unsigned lmax, std::array<double, 3> origin, double h0) {
    t_.init(IVec<3>{brick[0], brick[1], brick[2]}, lmax);
    geo_.origin = {origin[0], origin[1], origin[2]};
    geo_.h0 = h0;
  }

  // ---- introspection ----
  Index num_leaves() const { return t_.numLeaves(); }
  unsigned lmax() const { return t_.lmax(); }
  double h0() const { return geo_.h0; }
  std::array<double, 3> origin() const { return {geo_.origin[0], geo_.origin[1], geo_.origin[2]}; }
  bool is_balanced() const { return t_.isBalanced(); }

  py::array_t<double> centers() const { return leafCenters(t_, geo_); }
  py::array_t<double> sizes() const { return leafSizes(t_, geo_); }
  py::array_t<std::int32_t> levels() const { return leafLevels(t_); }
  py::array_t<std::uint64_t> codes() const { return leafCodes(t_); }

  // ---- queries ----
  // Index of the leaf containing world point (x,y,z), or -1 if outside the block.
  Index find(std::array<double, 3> x) const {
    std::array<Coord, 3> fine{};
    for (int d = 0; d < 3; ++d) {
      double f = (x[d] - geo_.origin[d]) / geo_.h0;
      if (f < 0) return -1;
      fine[d] = static_cast<Coord>(f);
    }
    return t_.find(fine);
  }

  // ---- refinement ----
  // Refine leaves the sphere's surface passes through (plus a band) down to target_level.
  Index refine_to_sphere(std::array<double, 3> center, double radius, unsigned target_level,
                         double band, bool balance) {
    geom::Sphere s{{center[0], center[1], center[2]}, radius};
    return amr::refineToSdf(t_, geo_, [&](const Vec<3>& p) { return s.eval(p); }, target_level,
                            band, balance);
  }

  // Refine toward an arbitrary signed-distance field given as a Python callable f(x,y,z)->distance
  // (suite SDF sign: <0 inside solid). Convenient for scripting; the callback runs under the GIL.
  Index refine_to_sdf(std::function<double(double, double, double)> sdf, unsigned target_level,
                      double band, bool balance) {
    return amr::refineToSdf(t_, geo_, [&](const Vec<3>& p) { return sdf(p[0], p[1], p[2]); },
                            target_level, band, balance);
  }

  // Split a single leaf by index; returns True if it was split (level>0).
  bool refine_leaf(Index i) { return t_.refineLeaf(i); }

  // Enforce 2:1 (graded) balance; returns the number of refinements performed.
  Index balance() { return t_.balance2to1(); }

  // ---- output ----
  // Write the octree + a per-leaf scalar field (N,) as a VTK UnstructuredGrid (.vtu, ASCII).
  void write_vtu(const std::string& path, const std::string& name, py::array_t<double> field) {
    amr::writeVtu(path, t_, geo_, name, asField(t_, field, "write_vtu"));
  }

 private:
  BO t_;
  amr::AmrGeometry<3> geo_;
};

// ---- distributed (MPI) octree ------------------------------------------------------------------

// The MPI octree: an ORB block decomposition of a global root grid, one BlockOctree per rank.
// Build it collectively (every rank constructs the same global geometry; ORB hands each rank a
// block), refine the local octree toward a global surface, restore cross-block 2:1 balance,
// load-rebalance leaves+fields onto a weighted ORB, gather face-neighbour field values across the
// owner-based halo, and read this rank's local leaf geometry / fields as numpy. Uses MPI_COMM_WORLD.
class DistributedOctree {
 public:
  DistributedOctree(std::array<long, 3> global_root_size, unsigned lmax,
                    std::array<double, 3> origin, double h0, std::array<bool, 3> periodic) {
    amr::AmrGeometry<3> g;
    g.origin = {origin[0], origin[1], origin[2]};
    g.h0 = h0;
    d_.init(IVec<3>{global_root_size[0], global_root_size[1], global_root_size[2]}, lmax, g,
            {periodic[0], periodic[1], periodic[2]}, MPI_COMM_WORLD);
  }

  // ---- introspection ----
  int rank() const { return d_.rank(); }
  int size() const { return d_.size(); }
  Index num_leaves() const { return d_.local().numLeaves(); }
  unsigned lmax() const { return d_.lmax(); }
  double h0() const { return d_.h0(); }
  std::array<long, 3> block_origin_root() const {
    const auto& b = d_.blockOriginRoot();
    return {b[0], b[1], b[2]};
  }
  std::array<long, 3> block_brick() const {
    const auto& b = d_.blockBrick();
    return {b[0], b[1], b[2]};
  }
  std::array<long, 3> global_root_size() const {
    const auto& b = d_.globalRootSize();
    return {b[0], b[1], b[2]};
  }

  // World leaf geometry of THIS rank's block (centres in global world coordinates).
  py::array_t<double> centers() const { return leafCenters(d_.local(), d_.localGeometry()); }
  py::array_t<double> sizes() const { return leafSizes(d_.local(), d_.localGeometry()); }
  py::array_t<std::int32_t> levels() const { return leafLevels(d_.local()); }
  py::array_t<std::uint64_t> codes() const { return leafCodes(d_.local()); }

  // ---- refinement ----
  // Refine the local octree toward a GLOBAL sphere surface, then (if balance) restore cross-block
  // 2:1 balance collectively. Returns the local refinement count. Collective when balance=True.
  Index refine_to_sphere(std::array<double, 3> center, double radius, unsigned target_level,
                         double band, bool balance) {
    geom::Sphere s{{center[0], center[1], center[2]}, radius};
    auto lgeo = d_.localGeometry();
    // Refine the block locally without the within-block balance; the distributed balance() below
    // grades across block boundaries too.
    Index n = amr::refineToSdf(d_.local(), lgeo, [&](const Vec<3>& p) { return s.eval(p); },
                               target_level, band, /*balance=*/false);
    if (balance) d_.balance();
    return n;
  }

  // Cross-block 2:1 balance (collective). Returns the local number of refinements performed.
  Index balance() { return d_.balance(); }

  // ---- load balancing ----
  // Re-decompose by leaf COUNT (weighted ORB) so each rank holds a near-equal share, migrating
  // leaves AND their fields. `fields` is (N,K) float64 (K per-leaf columns, e.g. a solution and a
  // marker); returns this rank's (M,K) columns after migration. Pure redistribution — the global
  // leaf+field set is unchanged; only ownership moves. Collective. After it, num_leaves / centers
  // reflect the new partition.
  py::array_t<double> rebalance(py::array_t<double> fields) {
    auto in = fields.unchecked<2>();
    const Index n = d_.local().numLeaves();
    if (in.shape(0) != n) throw std::runtime_error("rebalance: fields.shape[0] != num_leaves");
    const int K = static_cast<int>(in.shape(1));
    std::vector<std::vector<double>> cols(static_cast<std::size_t>(K),
                                          std::vector<double>(static_cast<std::size_t>(n)));
    for (Index i = 0; i < n; ++i)
      for (int c = 0; c < K; ++c) cols[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] = in(i, c);

    d_.rebalance(cols);  // mutates the octree in place and swaps in the new columns

    const Index m = d_.local().numLeaves();
    py::array_t<double> out({(py::ssize_t)m, (py::ssize_t)K});
    auto o = out.mutable_unchecked<2>();
    for (Index i = 0; i < m; ++i)
      for (int c = 0; c < K; ++c) o(i, c) = cols[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)];
    return out;
  }

  // ---- halo ----
  // For each local leaf and each of the 6 faces, the neighbouring leaf's field value, gathered
  // across the owner-based halo. Returns (N,6) float64 laid out as [+x,-x,+y,-y,+z,-z]; domain
  // boundaries (no neighbour) carry `sentinel`. Collective. `field` is this rank's per-leaf (N,).
  py::array_t<double> face_neighbor_gather(py::array_t<double> field, double sentinel) {
    std::vector<double> f = asField(d_.local(), field, "face_neighbor_gather");
    std::vector<double> g = d_.faceNeighborGather(f, sentinel);
    const Index n = d_.local().numLeaves();
    constexpr int F = 6;  // 2*Dim
    py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)F});
    auto o = out.mutable_unchecked<2>();
    for (Index i = 0; i < n; ++i)
      for (int s = 0; s < F; ++s) o(i, s) = g[static_cast<std::size_t>(i) * F + s];
    return out;
  }

  // ---- output ----
  // Write THIS rank's local octree + a per-leaf scalar (N,) as a .vtu (one file per rank).
  void write_vtu(const std::string& path, const std::string& name, py::array_t<double> field) {
    amr::writeVtu(path, d_.local(), d_.localGeometry(), name,
                  asField(d_.local(), field, "write_vtu"));
  }

 private:
  DO d_;
};

}  // namespace

PYBIND11_MODULE(tpx_amr, m) {
  m.doc() =
      "transport-core adaptive-mesh-refinement octree (host path): per-block BlockOctree (serial) "
      "and DistributedOctree (MPI ORB). Build a graded octree, refine to an SDF surface, read leaf "
      "geometry + per-leaf fields as numpy, load-rebalance, gather face neighbours, export VTU.";

  py::class_<Octree>(m, "Octree",
                     "Serial single-block adaptive octree with a world placement (origin + finest "
                     "spacing h0). Leaves are addressed in Z-order slot order; every per-leaf array "
                     "is indexed by that slot.")
      .def(py::init<std::array<long, 3>, unsigned, std::array<double, 3>, double>(),
           py::arg("brick"), py::arg("lmax"), py::arg("origin") = std::array<double, 3>{0, 0, 0},
           py::arg("h0") = 1.0,
           "Build a uniform octree of `brick` root cells per axis, each refinable `lmax` levels "
           "deep. `origin` is the block's lower corner and `h0` the finest (level-0) cell width.")
      .def_property_readonly("num_leaves", &Octree::num_leaves, "Number of leaves (Z-order slots).")
      .def_property_readonly("lmax", &Octree::lmax, "Root-cell level (max refinement depth).")
      .def_property_readonly("h0", &Octree::h0, "Finest (level-0) cell width in world units.")
      .def_property_readonly("origin", &Octree::origin, "Block lower corner in world coordinates.")
      .def("is_balanced", &Octree::is_balanced,
           "True iff every face-adjacent leaf pair differs by at most one level (2:1).")
      .def("centers", &Octree::centers, "Leaf world centres, (num_leaves, 3) float64.")
      .def("sizes", &Octree::sizes, "Leaf world widths h0*2**level, (num_leaves,) float64.")
      .def("levels", &Octree::levels, "Leaf refinement levels, (num_leaves,) int32 (0 = finest).")
      .def("codes", &Octree::codes, "Leaf block-local Morton origin codes, (num_leaves,) uint64.")
      .def("find", &Octree::find, py::arg("x"),
           "Index of the leaf containing world point x=(x,y,z), or -1 if outside the block.")
      .def("refine_to_sphere", &Octree::refine_to_sphere, py::arg("center"), py::arg("radius"),
           py::arg("target_level") = 0u, py::arg("band") = 1.0, py::arg("balance") = true,
           "Refine leaves the sphere surface passes through (plus `band`*h0) down to target_level; "
           "optionally restore 2:1 balance. Returns the number of refinements performed.")
      .def("refine_to_sdf", &Octree::refine_to_sdf, py::arg("sdf"), py::arg("target_level") = 0u,
           py::arg("band") = 1.0, py::arg("balance") = true,
           "Refine toward an arbitrary signed-distance field given as a callable f(x,y,z)->distance "
           "(suite sign: <0 inside solid), down to target_level. Returns refinements performed.")
      .def("refine_leaf", &Octree::refine_leaf, py::arg("i"),
           "Split leaf `i` into its 8 children; returns True if it was split (level>0).")
      .def("balance", &Octree::balance,
           "Enforce 2:1 graded balance to a fixpoint; returns refinements performed.")
      .def("write_vtu", &Octree::write_vtu, py::arg("path"), py::arg("name"), py::arg("field"),
           "Write the octree + a per-leaf scalar field (num_leaves,) as a VTK UnstructuredGrid "
           "(.vtu, ASCII, one cell per leaf), openable in ParaView.");

  py::class_<DistributedOctree>(
      m, "DistributedOctree",
      "MPI octree: an ORB block decomposition of a global root grid (one BlockOctree per rank, over "
      "MPI_COMM_WORLD). Construct it collectively; refine/balance/rebalance/face_neighbor_gather are "
      "collective. Per-leaf arrays describe THIS rank's local block in global world coordinates.")
      .def(py::init<std::array<long, 3>, unsigned, std::array<double, 3>, double,
                    std::array<bool, 3>>(),
           py::arg("global_root_size"), py::arg("lmax"),
           py::arg("origin") = std::array<double, 3>{0, 0, 0}, py::arg("h0") = 1.0,
           py::arg("periodic") = std::array<bool, 3>{true, true, true},
           "Decompose `global_root_size` root cells (each `lmax` levels deep) across the ranks of "
           "MPI_COMM_WORLD via ORB. `origin`/`h0` place the global grid; `periodic` per axis.")
      .def_property_readonly("rank", &DistributedOctree::rank, "This process's MPI rank.")
      .def_property_readonly("size", &DistributedOctree::size, "Number of ranks (blocks).")
      .def_property_readonly("num_leaves", &DistributedOctree::num_leaves,
                             "Leaves owned by this rank.")
      .def_property_readonly("lmax", &DistributedOctree::lmax, "Root-cell level.")
      .def_property_readonly("h0", &DistributedOctree::h0, "Finest cell width in world units.")
      .def_property_readonly("block_origin_root", &DistributedOctree::block_origin_root,
                             "This rank's block lower corner, in global root-cell coordinates.")
      .def_property_readonly("block_brick", &DistributedOctree::block_brick,
                             "This rank's block size in root cells per axis.")
      .def_property_readonly("global_root_size", &DistributedOctree::global_root_size,
                             "Global grid size in root cells per axis.")
      .def("centers", &DistributedOctree::centers,
           "Local leaf world centres, (num_leaves, 3) float64 (global coordinates).")
      .def("sizes", &DistributedOctree::sizes, "Local leaf world widths, (num_leaves,) float64.")
      .def("levels", &DistributedOctree::levels, "Local leaf levels, (num_leaves,) int32.")
      .def("codes", &DistributedOctree::codes,
           "Local leaf block-local Morton origin codes, (num_leaves,) uint64.")
      .def("refine_to_sphere", &DistributedOctree::refine_to_sphere, py::arg("center"),
           py::arg("radius"), py::arg("target_level") = 0u, py::arg("band") = 1.0,
           py::arg("balance") = true,
           "Refine the local block toward a GLOBAL sphere surface down to target_level, then (if "
           "balance) restore cross-block 2:1 balance collectively. Returns the local count.")
      .def("balance", &DistributedOctree::balance,
           "Restore cross-block 2:1 graded balance (collective). Returns this rank's refinements.")
      .def("rebalance", &DistributedOctree::rebalance, py::arg("fields"),
           "Re-decompose by leaf count (weighted ORB) and migrate leaves + their fields. `fields` "
           "is (num_leaves, K) float64; returns this rank's (M, K) columns after migration. Pure "
           "redistribution; the partition is updated in place (collective).")
      .def("face_neighbor_gather", &DistributedOctree::face_neighbor_gather, py::arg("field"),
           py::arg("sentinel") = 0.0,
           "For each local leaf, the field value across each of its 6 faces, gathered over the "
           "owner-based halo. `field` is (num_leaves,); returns (num_leaves, 6) laid out "
           "[+x,-x,+y,-y,+z,-z]; domain boundaries carry `sentinel` (collective).")
      .def("write_vtu", &DistributedOctree::write_vtu, py::arg("path"), py::arg("name"),
           py::arg("field"),
           "Write this rank's local octree + a per-leaf scalar (num_leaves,) as a .vtu (one file "
           "per rank; combine in ParaView).");
}

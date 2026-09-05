// core — Python surface for the AMR octree (peclet::core::amr).
//
// A nanobind module exposing the host adaptive-mesh-refinement path: the per-block BlockOctree
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
// Per-leaf arrays are returned via the shared peclet::core::python::vector_to_ndarray (capsule-backed, no
// extra copy); the Flow path returns host fields the same way. AMR is guarded by PECLET_CORE_HAVE_MORTON, so
// this module REQUIRES the morton sibling checkout — its CMake points the include path at
// ../../morton/include and defines PECLET_CORE_HAVE_MORTON. MPI is assumed already initialized by the host
// (import mpi4py.MPI first); the distributed class uses MPI_COMM_WORLD and never calls Init/Finalize.
//
// Build: see python/CMakeLists.txt (the tpx_amr target).
#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "peclet/core/amr/adapt.hpp"
#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/flow.hpp"  // the canonical (device) AmrFlow exposed to Python
#include "peclet/core/geom/scene_query.hpp"  // SphereBedQuery (Layer 2-for-core)
#include "peclet/core/amr/distributed_adapt.hpp"
#include "peclet/core/amr/distributed_octree.hpp"
#include "peclet/core/amr/flow_oracle.hpp"  // oracle::AmrFlow (dev-only reference; not exposed)
#include "peclet/core/amr/indicators.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/amr/poisson.hpp"
#include "peclet/core/amr/refine.hpp"
#include "peclet/core/amr/vtu_io.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/geom/sdf.hpp"
#include "peclet/core/python/kokkos_teardown.hpp"
#include "peclet/core/python/ndarray_interop.hpp"

namespace nb = nanobind;
using namespace peclet::core;

// The wrapper classes live in a NAMED namespace on purpose: they have virtual methods (Releasable),
// and with an anonymous namespace hipcc's host/device split leaves their vtables unemitted, so the
// HIP link fails with `ld.lld: undefined hidden symbol: vtable for (anonymous namespace)::Octree`
// (and Poisson / Flow / DistributedOctree). OpenMP/CUDA builds are indifferent.
namespace peclet::core::pybind_amr {

using BO = amr::BlockOctree<3>;       // 3D, Bits=21 (default) — codes fit a uint64
using DO = amr::DistributedOctree<3>;
using Code = BO::Code;
using Coord = BO::Coord;
using DArray = nb::ndarray<double, nb::c_contig>;

// ---- per-leaf array helpers (capsule-backed; no extra copy; Z-order slot order) ----------------

template <class T>
nb::ndarray<nb::numpy, T> vec1(std::vector<T> v) {
  const std::size_t n = v.size();
  return peclet::core::python::vector_to_ndarray<T>(std::move(v), {n}, {1});
}
template <class T>
nb::ndarray<nb::numpy, T> vec2(std::vector<T> v, std::size_t cols) {
  const std::size_t n = v.size() / cols;
  return peclet::core::python::vector_to_ndarray<T>(std::move(v), {n, cols},
                                           {static_cast<std::int64_t>(cols), 1});
}

// World centre of every leaf -> (N,3) float64.
nb::ndarray<nb::numpy, double> leafCenters(const BO& t, const amr::AmrGeometry<3>& geo) {
  const Index n = t.numLeaves();
  std::vector<double> d(static_cast<std::size_t>(n) * 3);
  for (Index i = 0; i < n; ++i) {
    Vec<3> c = geo.center(t.bounds(i));
    d[i * 3 + 0] = c[0];
    d[i * 3 + 1] = c[1];
    d[i * 3 + 2] = c[2];
  }
  return vec2<double>(std::move(d), 3);
}

// World width of every leaf (h0 * 2**level) -> (N,) float64.
nb::ndarray<nb::numpy, double> leafSizes(const BO& t, const amr::AmrGeometry<3>& geo) {
  const Index n = t.numLeaves();
  std::vector<double> d(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) d[i] = geo.leafSize(t.level(i));
  return vec1<double>(std::move(d));
}

// Refinement level of every leaf -> (N,) int32 (lmax at the root, 0 finest).
nb::ndarray<nb::numpy, std::int32_t> leafLevels(const BO& t) {
  const Index n = t.numLeaves();
  std::vector<std::int32_t> d(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) d[i] = static_cast<std::int32_t>(t.level(i));
  return vec1<std::int32_t>(std::move(d));
}

// Block-local Morton origin code of every leaf -> (N,) uint64.
nb::ndarray<nb::numpy, std::uint64_t> leafCodes(const BO& t) {
  const Index n = t.numLeaves();
  std::vector<std::uint64_t> d(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) d[i] = static_cast<std::uint64_t>(t.code(i));
  return vec1<std::uint64_t>(std::move(d));
}

// A contiguous vector<double> as a (N,) float64 numpy array.
nb::ndarray<nb::numpy, double> vecToArray(std::vector<double> v) { return vec1<double>(std::move(v)); }

// Validate a per-leaf field numpy array and view it as a contiguous vector<double>.
std::vector<double> asField(const BO& t, const DArray& field, const char* who) {
  if (static_cast<Index>(field.shape(0)) != t.numLeaves())
    throw std::runtime_error(std::string(who) + ": field length != num_leaves");
  const double* f = field.data();
  return std::vector<double>(f, f + static_cast<std::size_t>(t.numLeaves()));
}

// ---- teardown registry -------------------------------------------------------------------------
// The suite-wide pattern (peclet/core/python/kokkos_teardown.hpp): Kokkos::finalize() MUST run from
// the atexit hook (else cudaErrorCudartUnloading on CUDA), but any wrapper that still holds Views at
// that point -- an Octree/Flow at script or notebook scope, the normal case -- is destroyed after
// finalize and Kokkos::abort()s the process. Every wrapper below is a Releasable (registered on
// construction); the hook drops their Views (release()) BEFORE finalize.
using peclet::core::python::Releasable;

// ---- serial single-block octree ----------------------------------------------------------------

// A per-block adaptive octree with its world placement (origin + uniform finest spacing h0).
// Wraps peclet::core::amr::BlockOctree<3> + AmrGeometry<3>: build a uniform brick, refine toward a surface,
// query leaves, and read leaf geometry / fields as numpy. The serial / single-rank form; for the
// distributed (MPI) octree use DistributedOctree below.
class Octree : public Releasable {
 public:
  Octree(std::array<long, 3> brick, unsigned lmax, std::array<double, 3> origin, double h0) {
    t_.init(IVec<3>{brick[0], brick[1], brick[2]}, lmax);
    geo_.origin = {origin[0], origin[1], origin[2]};
    geo_.h0 = h0;
  }
  void release() noexcept override { t_ = BO{}; }

  // ---- introspection ----
  Index num_leaves() const { return t_.numLeaves(); }
  unsigned lmax() const { return t_.lmax(); }
  double h0() const { return geo_.h0; }
  std::array<double, 3> origin() const { return {geo_.origin[0], geo_.origin[1], geo_.origin[2]}; }
  bool is_balanced() const { return t_.isBalanced(); }

  nb::ndarray<nb::numpy, double> centers() const { return leafCenters(t_, geo_); }
  nb::ndarray<nb::numpy, double> sizes() const { return leafSizes(t_, geo_); }
  nb::ndarray<nb::numpy, std::int32_t> levels() const { return leafLevels(t_); }
  nb::ndarray<nb::numpy, std::uint64_t> codes() const { return leafCodes(t_); }

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

  // Graded surface refinement: a per-point TARGET LEVEL callable instead of one global target
  // (the mixed-level cut band, plan §7). Pair with Flow.set_ghost_sampled(True).
  Index refine_to_sdf_graded(std::function<double(double, double, double)> sdf,
                             std::function<unsigned(double, double, double)> target_level,
                             double band, bool balance) {
    return amr::refineToSdfGraded(
        t_, geo_, [&](const Vec<3>& p) { return sdf(p[0], p[1], p[2]); },
        [&](const Vec<3>& p) { return target_level(p[0], p[1], p[2]); }, band, balance);
  }

  // The plan's gap-width floor (§7 criterion 1) driving refine_to_sdf_graded in one call:
  // target level = coarsest L with n*h_L <= gap(x), clamped to [0, coarsest_level].
  Index refine_to_gap_floor(std::function<double(double, double, double)> sdf,
                            std::function<double(double, double, double)> gap,
                            unsigned coarsest_level, double n, double band, bool balance) {
    auto tgt = amr::gapFloorTarget<3>([&](const Vec<3>& p) { return gap(p[0], p[1], p[2]); },
                                      geo_.h0, coarsest_level, n);
    return amr::refineToSdfGraded(
        t_, geo_, [&](const Vec<3>& p) { return sdf(p[0], p[1], p[2]); }, tgt, band, balance);
  }

  // Split a single leaf by index; returns True if it was split (level>0).
  bool refine_leaf(Index i) { return t_.refineLeaf(i); }

  // Enforce 2:1 (graded) balance; returns the number of refinements performed.
  Index balance() { return t_.balance2to1(); }

  // Internal (not bound to Python): give the Poisson solver the underlying octree + geometry.
  const BO& octreeRef() const { return t_; }
  const amr::AmrGeometry<3>& geoRef() const { return geo_; }

  // ---- solution-adaptive refinement ----
  // Löhner normalized-second-difference indicator E in [0,1] per leaf from a scalar field (N,);
  // large E marks steep features to refine, small E marks smooth regions to coarsen.
  nb::ndarray<nb::numpy, double> lohner_indicator(DArray field, double eps) {
    return vecToArray(amr::lohnerIndicator(t_, asField(t_, field, "lohner_indicator"), eps));
  }

  // One solution-adaptive step driven by the Löhner indicator: refine leaves where the indicator
  // exceeds refine_thresh (down to finest_level), coarsen sibling groups all below coarsen_thresh,
  // restore 2:1 balance, and conservatively remap `field` (N,) onto the new mesh. MUTATES the octree
  // in place (num_leaves/centers/... then reflect the new mesh) and returns the remapped field (M,).
  nb::ndarray<nb::numpy, double> adapt(DArray field, double refine_thresh, double coarsen_thresh,
                                       unsigned finest_level, double eps, bool linear) {
    auto fv = asField(t_, field, "adapt");
    auto r = amr::adapt(t_, fv, refine_thresh, coarsen_thresh, finest_level, eps, linear);
    t_ = std::move(r.octree);
    return vecToArray(std::move(r.field));
  }

  // ---- output ----
  // Write the octree + a per-leaf scalar field (N,) as a VTK UnstructuredGrid (.vtu, ASCII).
  void write_vtu(const std::string& path, const std::string& name, DArray field) {
    amr::writeVtu(path, t_, geo_, name, asField(t_, field, "write_vtu"));
  }

 private:
  BO t_;
  amr::AmrGeometry<3> geo_;
};

// ---- geometric-multigrid Poisson solver --------------------------------------------------------

// Cell-centered finite-volume Poisson solver (Lu = rhs) on an Octree, via a geometric multigrid
// V-cycle (peclet::core::amr::AmrMultigrid). The operator L is the conservative two-point FV Laplacian
// (negative-definite, suite sign convention); on a graded octree it is consistent and second-order
// in the bulk (first-order at coarse/fine faces). `periodic=True` solves the singular periodic
// problem (the constant null space is removed each cycle); the manufactured RHS b = apply(u_exact)
// is exactly mean-zero, so the residual drives to round-off. The hierarchy snapshots the octree at
// construction. (Cut-cell openness is handled by the flow solver, which sets it consistently.)
class Poisson : public Releasable {
 public:
  Poisson(const Octree& oct, bool periodic) : n_(oct.octreeRef().numLeaves()) {
    mg_.build(oct.octreeRef(), oct.geoRef().h0);
    mg_.setPeriodic(periodic);
  }
  void release() noexcept override { mg_ = amr::AmrMultigrid<3>{}; }

  Index num_leaves() const { return n_; }
  std::size_t num_levels() const { return mg_.numLevels(); }

  // L applied to u (the FV Laplacian); use it to manufacture a RHS b = apply(u_exact). (N,)->(N,).
  nb::ndarray<nb::numpy, double> apply(DArray u) {
    std::vector<double> out;
    mg_.op(0).applyLaplacian(toVec(u, "apply"), out);
    return vec1<double>(std::move(out));
  }

  // Volume-weighted L2 residual norm sqrt(sum V*(rhs - L u)^2).
  double residual(DArray u, DArray rhs) {
    std::vector<double> res;
    return mg_.op(0).residual(toVec(u, "residual"), toVec(rhs, "residual"), res);
  }

  // Solve L u = rhs with up to `cycles` V-cycles (pre/post smoothing sweeps each), starting from
  // x0 (or 0). Stops early once the residual <= tol (tol<=0 disables). Returns
  // (u (N,), final_residual, cycles_done).
  nb::tuple solve(DArray rhs, std::optional<DArray> x0, int cycles, int pre, int post, double tol) {
    std::vector<double> rv = toVec(rhs, "solve");
    std::vector<double> u(static_cast<std::size_t>(n_), 0.0);
    if (x0) u = toVec(*x0, "solve");
    std::vector<double> res;
    double r = mg_.op(0).residual(u, rv, res);
    int done = 0;
    for (int k = 0; k < cycles; ++k) {
      mg_.vcycle(0, u, rv, pre, post);
      r = mg_.op(0).residual(u, rv, res);
      ++done;
      if (tol > 0.0 && r <= tol) break;
    }
    return nb::make_tuple(vec1<double>(std::move(u)), r, done);
  }

 private:
  std::vector<double> toVec(const DArray& a, const char* who) const {
    if (static_cast<Index>(a.shape(0)) != n_)
      throw std::runtime_error(std::string(who) + ": length != num_leaves");
    const double* p = a.data();
    return std::vector<double>(p, p + static_cast<std::size_t>(n_));
  }

  amr::AmrMultigrid<3> mg_;
  Index n_;
};

// ---- collocated incompressible flow ------------------------------------------------------------

// Collocated (cell-centered) incompressible Stokes / Navier-Stokes step on an Octree with a cut-cell
// immersed boundary (peclet::core::amr::AmrFlow). Each step() is: an implicit backward-Euler viscous momentum
// predictor with no-slip (u=0) Dirichlet cut-cell IBM on the SDF solid, then the Almgren-Bell-Colella
// approximate projection in incremental-rotational form (openness-weighted pressure Poisson). Stokes
// by default; set_advection(True) adds explicit high-order (SOU / Koren-TVD) momentum advection.
// Driven by a body force; iterate step() to steady state. Velocities/pressure are per-leaf (num_leaves,)
// in Z-order slots. The octree is borrowed by reference (kept alive for the Flow's lifetime).
//
// Resolve the immersed boundary in a uniformly-finest band: the cut-cell and ±2 advection stencils
// assume same-level neighbours, so keep the solid surface off 2:1 interfaces.
class Flow : public Releasable {
 public:
  Flow(const Octree& oct, double rho, double mu, double dt) : n_(oct.octreeRef().numLeaves()) {
    flow_.init(oct.octreeRef(), oct.geoRef().h0, oct.geoRef().origin);
    flow_.setDensity(rho);
    flow_.setViscosity(mu);
    flow_.setDt(dt);
    const auto& t = oct.octreeRef();
    for (int d = 0; d < 3; ++d) {
      extent_[d] = static_cast<double>(t.brick()[d]) * static_cast<double>(Index(1) << t.lmax()) *
                   oct.geoRef().h0;
      origin_[d] = oct.geoRef().origin[d];
    }
  }
  // Distributed (mpi4py): run this solver on one ORB block of a DistributedOctree — the whole
  // step then executes multi-rank through the ±2 LeafHalo (docs/amr_distributed_flow.md).
  // Defined after DistributedOctree below; keep_alive on the binding pins the octree.
  Flow(class DistributedOctree& d, double rho, double mu, double dt);
  // Distributed load rebalance: weighted-ORB re-decomposition migrating u/p with the leaves +
  // full rebuild (collective; distributed constructor only).
  void rebalance_mpi(std::function<double(double, double, double)> sdf) {
    nb::gil_scoped_release nogil;  // see set_solid
    flow_.rebalanceMpi([&](const Vec<3>& p) { return sdf(p[0], p[1], p[2]); });
    n_ = flow_.numLeaves();
  }
  void release() noexcept override { flow_ = amr::AmrFlow<>{}; }

  Index num_leaves() const { return n_; }
  void set_body_force(double fx, double fy, double fz) { flow_.setBodyForce(fx, fy, fz); }
  void set_advection(bool on) { flow_.setAdvection(on); }
  void set_ghost_gradient(bool on) { flow_.setGhostGradient(on); }
  void set_aperture_order(int order) { flow_.setApertureOrder(order); }
  void set_ghost_projection(bool on, int matrix_order, int rhs_order) {
    flow_.setGhostProjection(on, matrix_order, rhs_order);
  }
  void set_ghost_sampled(bool on) { flow_.setGhostSampled(on); }
  void set_cf_scheme(int scheme) { flow_.setCfScheme(scheme); }

  // Adaptivity during a run: snapshot -> (externally mutate the Octree: adapt / refine_to_* /
  // balance) -> rebuild + conservative field transfer. num_leaves is refreshed.
  void begin_adapt() { flow_.beginAdapt(); }
  void set_pressure(nb::ndarray<double, nb::c_contig> h) {
    flow_.setPressure(std::vector<double>(h.data(), h.data() + h.shape(0)));
  }
  void set_dt(double dt) { flow_.setDt(dt); }
  void finish_adapt(std::function<double(double, double, double)> sdf) {
    nb::gil_scoped_release nogil;  // see set_solid
    flow_.finishAdapt([&](const Vec<3>& p) { return sdf(p[0], p[1], p[2]); });
    n_ = flow_.numLeaves();
  }
  void set_advection_scheme(int s) { flow_.setAdvectionScheme(s); }
  void set_implicit_advection(bool on) { flow_.setImplicitAdvection(on); }

  // Build the cut-cell operators from a signed-distance callable f(x,y,z) (>0 in fluid, <0 in solid),
  // and zero the velocity / pressure fields. Call before stepping; re-call to change the geometry.
  void set_solid(std::function<double(double, double, double)> sdf) {
    // The host operator builders sample the SDF from Kokkos host parallel regions. nanobind's
    // std::function wrapper takes the GIL for every call, so with the calling thread still
    // holding it while it waits at the parallel region's barrier, the worker threads block
    // forever (measured: a hang under the OpenMP host backend at OMP_NUM_THREADS > 1). Release
    // it here; the callback re-acquires it per sample, which serialises the sampling but
    // cannot deadlock. The same holds for finish_adapt / rebalance_mpi, which rebuild through
    // setSolid.
    nb::gil_scoped_release nogil;
    flow_.setSolid([&](const Vec<3>& p) { return sdf(p[0], p[1], p[2]); });
  }

  // Native union-of-spheres SDF — the porous-media geometry evaluated in C++ instead of through
  // a Python callback. set_solid samples the SDF tens of times per leaf (operator build, overlay
  // classification at virtual positions, the openness probe), so at bed scale the callback, not
  // the solve, dominates: measured >1h43m of pure numpy on an 11.35M-leaf 180-sphere bed before
  // the GPU was touched once. `centers` is (M,3) and `radii` is (M,) or a single scalar-valued
  // (1,) array; `periodic` applies the minimum-image convention over the octree's own extent.
  void set_solid_spheres(nb::ndarray<const double, nb::c_contig> centers,
                         nb::ndarray<const double, nb::c_contig> radii, bool periodic) {
    if (centers.ndim() != 2 || centers.shape(1) != 3)
      throw std::runtime_error("set_solid_spheres: centers must be (M, 3)");
    const std::size_t m = centers.shape(0);
    if (radii.ndim() != 1 || (radii.shape(0) != m && radii.shape(0) != 1))
      throw std::runtime_error("set_solid_spheres: radii must be (M,) or (1,)");
    std::vector<double> cx(m), cy(m), cz(m), rr(m);
    const double* cp = centers.data();
    const double* rp = radii.data();
    const bool oneR = radii.shape(0) == 1;
    for (std::size_t i = 0; i < m; ++i) {
      cx[i] = cp[3 * i];
      cy[i] = cp[3 * i + 1];
      cz[i] = cp[3 * i + 2];
      rr[i] = oneR ? rp[0] : rp[i];
    }
    // Layer 2-for-core (docs/AMR_GEOMETRY_SETUP_REQUIREMENTS.md): the per-query brute-force scan
    // over all M spheres — measured 1.18 us/eval at M=180, 94% of the whole setSolid — is retired
    // for the candidate-grid + equal-radius query. BIT-IDENTICAL to the old callback (value, not
    // just sign) by construction and by the geom_scene_query ctest; the templated setSolid inlines
    // the query, so there is no std::function indirection left either.
    geom::SphereBedQuery q(std::move(cx), std::move(cy), std::move(cz), std::move(rr),
                           Vec<3>{origin_[0], origin_[1], origin_[2]},
                           Vec<3>{extent_[0], extent_[1], extent_[2]}, periodic);
    flow_.setSolid(q);
  }

  // Momentum solver controls (device path): velocity multigrid + smoother selection.
  void set_momentum_mg(bool on) { flow_.setMomentumMG(on); }
  void set_momentum_gs(bool on) { flow_.setMomentumGS(on); }
  void set_velocity_mg_staircase(bool on) { flow_.setVelocityMGStaircase(on); }
  void set_momentum_mg_solver(bool on) { flow_.setMomentumMGSolver(on); }
  void set_outer_iterations(int n, double tol) { flow_.setOuterIterations(n, tol); }

  // Advance one collocated projection step (Stokes, or NS if advection is on).
  void step(int mom_iters, int pres_iters) { flow_.step(mom_iters, pres_iters); }

  // Per-leaf velocity component c (0=x,1=y,2=z) -> (num_leaves,) float64.
  nb::ndarray<nb::numpy, double> velocity(int c) const {
    if (c < 0 || c > 2) throw std::runtime_error("velocity: component must be 0..2");
    const auto& v = flow_.velocity(c);
    return vec1<double>(std::vector<double>(v.begin(), v.end()));
  }

  // All three velocity components -> (num_leaves, 3) float64. Packed on-device into one interleaved
  // buffer, so the device->host boundary is crossed once (G6) instead of three times.
  nb::ndarray<nb::numpy, double> velocities() const { return vec2<double>(flow_.velocities(), 3); }

  // Per-leaf fluid mask (False inside the solid) -> (num_leaves,) bool.
  nb::ndarray<nb::numpy, bool> is_fluid() const {
    auto* buf = new std::vector<std::uint8_t>(static_cast<std::size_t>(n_));
    for (Index i = 0; i < n_; ++i) (*buf)[static_cast<std::size_t>(i)] = flow_.isFluid(i) ? 1 : 0;
    nb::capsule owner(buf, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
    return nb::ndarray<nb::numpy, bool>(reinterpret_cast<bool*>(buf->data()),
                                        {static_cast<std::size_t>(n_)}, owner, {1});
  }

  // Volume-weighted L2 norm of the residual cell divergence (a projection-quality diagnostic).
  double divergence_norm() { return flow_.divNormL2(); }
  // L2 norm of the divergence of the ABC divergence-free FACE field (≈ the pressure-solve residual,
  // far below divergence_norm — including across 2:1 interfaces).
  double divergence_norm_face() { return flow_.divNormFace(); }

  // Per-leaf pressure (incremental-rotational p) -> (num_leaves,) float64.
  nb::ndarray<nb::numpy, double> pressure() const { return vec1<double>(flow_.pressure()); }
  // ABC divergence-free FACE velocity, one value per CSR (sub)face -> (num_faces,) float64.
  nb::ndarray<nb::numpy, double> face_field() const { return vec1<double>(flow_.faceField()); }
  // Write velocity component c (0=x,1=y,2=z) from a (num_leaves,) array — ICs / restart / warm-start.
  void set_velocity(int c, nb::ndarray<double, nb::c_contig> h) {
    flow_.setVelocity(c, std::vector<double>(h.data(), h.data() + h.shape(0)));
  }
  // Pressure projection only (no momentum) — project an externally-set velocity to divergence-free.
  void project(int pres_iters) { flow_.project(pres_iters); }
  // Solver iteration counts from the last step (convergence / perf diagnostics).
  int last_mom_iters() const { return flow_.lastMomIters(); }
  int last_pres_iters() const { return flow_.lastPresIters(); }
  int last_outer_iters() const { return flow_.lastOuterIters(); }

 private:
  amr::AmrFlow<> flow_;  // the canonical device (Kokkos) AMR flow
  Index n_;
  std::array<double, 3> extent_{0.0, 0.0, 0.0};  // world box size (min-image for set_solid_spheres)
  std::array<double, 3> origin_{0.0, 0.0, 0.0};  // world box origin (candidate-grid coverage)
};

// ---- distributed (MPI) octree ------------------------------------------------------------------

// The MPI octree: an ORB block decomposition of a global root grid, one BlockOctree per rank.
// Build it collectively (every rank constructs the same global geometry; ORB hands each rank a
// block), refine the local octree toward a global surface, restore cross-block 2:1 balance,
// load-rebalance leaves+fields onto a weighted ORB, gather face-neighbour field values across the
// owner-based halo, and read this rank's local leaf geometry / fields as numpy. Uses MPI_COMM_WORLD.
class DistributedOctree : public Releasable {
 public:
  DistributedOctree(std::array<long, 3> global_root_size, unsigned lmax,
                    std::array<double, 3> origin, double h0, std::array<bool, 3> periodic) {
    amr::AmrGeometry<3> g;
    g.origin = {origin[0], origin[1], origin[2]};
    g.h0 = h0;
    d_.init(IVec<3>{global_root_size[0], global_root_size[1], global_root_size[2]}, lmax, g,
            {periodic[0], periodic[1], periodic[2]}, MPI_COMM_WORLD);
  }
  void release() noexcept override { d_ = DO{}; }

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
  nb::ndarray<nb::numpy, double> centers() const { return leafCenters(d_.local(), d_.localGeometry()); }
  nb::ndarray<nb::numpy, double> sizes() const { return leafSizes(d_.local(), d_.localGeometry()); }
  nb::ndarray<nb::numpy, std::int32_t> levels() const { return leafLevels(d_.local()); }
  nb::ndarray<nb::numpy, std::uint64_t> codes() const { return leafCodes(d_.local()); }

  // ---- refinement ----
  // Refine the local octree toward a GLOBAL sphere surface, then (if balance) restore cross-block
  // 2:1 balance collectively. Returns the local refinement count. Collective when balance=True.
  Index refine_to_sphere(std::array<double, 3> center, double radius, unsigned target_level,
                         double band, bool balance) {
    geom::Sphere s{{center[0], center[1], center[2]}, radius};
    auto lgeo = d_.localGeometry();
    Index n = amr::refineToSdf(d_.local(), lgeo, [&](const Vec<3>& p) { return s.eval(p); },
                               target_level, band, /*balance=*/false);
    if (balance) d_.balance();
    return n;
  }

  // Refine the local octree toward an arbitrary GLOBAL surface given as a Python SDF callable
  // f(x,y,z) (suite sign: <0 inside solid), then (if balance) restore cross-block 2:1 balance
  // collectively. The distributed analogue of the serial Octree.refine_to_sdf. Collective when balance.
  Index refine_to_sdf(std::function<double(double, double, double)> sdf, unsigned target_level,
                      double band, bool balance) {
    auto lgeo = d_.localGeometry();
    Index n = amr::refineToSdf(d_.local(), lgeo,
                               [&](const Vec<3>& p) { return sdf(p[0], p[1], p[2]); }, target_level,
                               band, /*balance=*/false);
    if (balance) d_.balance();
    return n;
  }

  // Cross-block 2:1 balance (collective). Returns the local number of refinements performed.
  Index balance() { return d_.balance(); }

  // ---- load balancing ----
  // Re-decompose by leaf COUNT (weighted ORB) so each rank holds a near-equal share, migrating
  // leaves AND their fields. `fields` is (N,K) float64 (K per-leaf columns); returns this rank's
  // (M,K) columns after migration. Pure redistribution; the partition is updated in place. Collective.
  nb::ndarray<nb::numpy, double> rebalance(DArray fields) {
    const Index n = d_.local().numLeaves();
    if (static_cast<Index>(fields.shape(0)) != n)
      throw std::runtime_error("rebalance: fields.shape[0] != num_leaves");
    const int K = static_cast<int>(fields.shape(1));
    const double* in = fields.data();
    std::vector<std::vector<double>> cols(static_cast<std::size_t>(K),
                                          std::vector<double>(static_cast<std::size_t>(n)));
    for (Index i = 0; i < n; ++i)
      for (int c = 0; c < K; ++c)
        cols[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] = in[i * K + c];

    d_.rebalance(cols);  // mutates the octree in place and swaps in the new columns

    const Index m = d_.local().numLeaves();
    std::vector<double> out(static_cast<std::size_t>(m) * static_cast<std::size_t>(K));
    for (Index i = 0; i < m; ++i)
      for (int c = 0; c < K; ++c)
        out[i * K + c] = cols[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)];
    return vec2<double>(std::move(out), static_cast<std::size_t>(K));
  }

  // ---- halo ----
  // For each local leaf and each of the 6 faces, the neighbouring leaf's field value, gathered
  // across the owner-based halo. Returns (N,6) float64 laid out as [+x,-x,+y,-y,+z,-z]; domain
  // boundaries (no neighbour) carry `sentinel`. Collective. `field` is this rank's per-leaf (N,).
  nb::ndarray<nb::numpy, double> face_neighbor_gather(DArray field, double sentinel) {
    std::vector<double> f = asField(d_.local(), field, "face_neighbor_gather");
    std::vector<double> g = d_.faceNeighborGather(f, sentinel);
    return vec2<double>(std::move(g), 6);  // 2*Dim faces
  }

  // ---- solution-adaptive refinement ----
  // Löhner indicator per local leaf, evaluated across the owner-based halo so cross-block neighbours
  // contribute exactly as in a whole-domain solve. `field` is this rank's (num_leaves,). Collective.
  nb::ndarray<nb::numpy, double> lohner_indicator(DArray field, double eps) {
    auto fv = asField(d_.local(), field, "lohner_indicator");
    return vecToArray(amr::lohnerIndicatorDistributed(d_, fv, eps));
  }

  // One distributed solution-adaptive step: refine/coarsen each block's local octree from the
  // Löhner indicator, restore cross-block 2:1 balance, and conservatively remap `field` (num_leaves,)
  // onto the new local mesh. MUTATES the octree in place (keeping ORB ownership); returns the remapped
  // local field (M,). Bit-identical across rank counts. Collective.
  nb::ndarray<nb::numpy, double> adapt(DArray field, double refine_thresh, double coarsen_thresh,
                                       unsigned finest_level, double eps, bool linear) {
    auto fv = asField(d_.local(), field, "adapt");
    return vecToArray(
        amr::distributedAdapt(d_, fv, refine_thresh, coarsen_thresh, finest_level, eps, linear));
  }

  // ---- output ----
  // Write THIS rank's local octree + a per-leaf scalar (N,) as a .vtu (one file per rank).
  void write_vtu(const std::string& path, const std::string& name, DArray field) {
    amr::writeVtu(path, d_.local(), d_.localGeometry(), name,
                  asField(d_.local(), field, "write_vtu"));
  }

  /// The wrapped core octree (the distributed Flow constructor hooks into it).
  DO& ref() { return d_; }

 private:
  DO d_;
};

// Distributed Flow constructor (out of line: DistributedOctree is declared after Flow).
inline Flow::Flow(DistributedOctree& d, double rho, double mu, double dt)
    : n_(d.ref().local().numLeaves()) {
  flow_.initMpi(d.ref());
  flow_.setDensity(rho);
  flow_.setViscosity(mu);
  flow_.setDt(dt);
}

}  // namespace peclet::core::pybind_amr

NB_MODULE(amr, m) {
  using namespace peclet::core::pybind_amr;
  // The Flow path runs Kokkos kernels — initialise the device runtime on import (the backend/arch is
  // fixed by the prefix the module was built against); the release-then-finalize atexit hook,
  // finalize() and execution_space are the suite-wide pattern (teardown registry above). Per-leaf
  // arrays are host-vector-backed (no Views).
  peclet::core::python::install(m);
  m.attr("__doc__") =
      "core adaptive-mesh-refinement: per-block BlockOctree (serial) and DistributedOctree "
      "(MPI ORB) for the mesh, plus the device (Kokkos) AmrFlow cut-cell Stokes/Navier-Stokes solver. "
      "Build a graded octree, refine to an SDF surface, read leaf geometry + per-leaf fields as numpy, "
      "load-rebalance, gather face neighbours, export VTU, and run the flow step on device.";

  nb::class_<Octree>(m, "Octree",
                     "Serial single-block adaptive octree with a world placement (origin + finest "
                     "spacing h0). Leaves are addressed in Z-order slot order; every per-leaf array "
                     "is indexed by that slot.")
      .def(nb::init<std::array<long, 3>, unsigned, std::array<double, 3>, double>(),
           nb::arg("brick"), nb::arg("lmax"), nb::arg("origin") = std::array<double, 3>{0, 0, 0},
           nb::arg("h0") = 1.0,
           "Build a uniform octree of `brick` root cells per axis, each refinable `lmax` levels "
           "deep. `origin` is the block's lower corner and `h0` the finest (level-0) cell width.")
      .def_prop_ro("num_leaves", &Octree::num_leaves, "Number of leaves (Z-order slots).")
      .def_prop_ro("lmax", &Octree::lmax, "Root-cell level (max refinement depth).")
      .def_prop_ro("h0", &Octree::h0, "Finest (level-0) cell width in world units.")
      .def_prop_ro("origin", &Octree::origin, "Block lower corner in world coordinates.")
      .def("is_balanced", &Octree::is_balanced,
           "True iff every face-adjacent leaf pair differs by at most one level (2:1).")
      .def("centers", &Octree::centers, "Leaf world centres, (num_leaves, 3) float64.")
      .def("sizes", &Octree::sizes, "Leaf world widths h0*2**level, (num_leaves,) float64.")
      .def("levels", &Octree::levels, "Leaf refinement levels, (num_leaves,) int32 (0 = finest).")
      .def("codes", &Octree::codes, "Leaf block-local Morton origin codes, (num_leaves,) uint64.")
      .def("find", &Octree::find, nb::arg("x"),
           "Index of the leaf containing world point x=(x,y,z), or -1 if outside the block.")
      .def("refine_to_sphere", &Octree::refine_to_sphere, nb::arg("center"), nb::arg("radius"),
           nb::arg("target_level") = 0u, nb::arg("band") = 1.0, nb::arg("balance") = true,
           "Refine leaves the sphere surface passes through (plus `band`*h0) down to target_level; "
           "optionally restore 2:1 balance. Returns the number of refinements performed.")
      .def("refine_to_sdf", &Octree::refine_to_sdf, nb::arg("sdf"), nb::arg("target_level") = 0u,
           nb::arg("band") = 1.0, nb::arg("balance") = true,
           "Refine toward an arbitrary signed-distance field given as a callable f(x,y,z)->distance "
           "(suite sign: <0 inside solid), down to target_level. Returns refinements performed.")
      .def("refine_to_sdf_graded", &Octree::refine_to_sdf_graded, nb::arg("sdf"),
           nb::arg("target_level"), nb::arg("band") = 2.0, nb::arg("balance") = true,
           "GRADED surface refinement (the mixed-level cut band, "
           "docs/amr_mixed_level_cut_band_plan.md §7): `target_level` is a callable "
           "f(x,y,z)->level giving the COARSEST acceptable level at a world point (0 = finest), "
           "so cut cells end up at SEVERAL levels — fine in throats/contacts, coarse on smooth "
           "caps. The band margin is measured in cells of the level being created (not in h0 as "
           "in refine_to_sdf). Requires Flow.set_ghost_sampled(True) — the classic overlay "
           "contracts a uniform finest band and raises on these level jumps. Returns "
           "refinements performed.")
      .def("refine_to_gap_floor", &Octree::refine_to_gap_floor, nb::arg("sdf"), nb::arg("gap"),
           nb::arg("coarsest_level"), nb::arg("n") = 4.0, nb::arg("band") = 2.0,
           nb::arg("balance") = true,
           "refine_to_sdf_graded driven by the plan's GAP-WIDTH FLOOR (§7 criterion 1): the "
           "target level at a point is the coarsest L with n*h_L <= gap(x), clamped to "
           "[0, coarsest_level]. `gap` is the local fluid-gap proxy f(x,y,z)->width — for a "
           "sphere packing the two-closest-surfaces sum d1+d2; a medial-axis or peclet.pnm "
           "throat-radius field substitutes verbatim. n=4 per the M1/M2 measurements. This is "
           "the AMReX multi-valued-cell rule inverted: coarsening never merges or disconnects "
           "fluid.")
      .def("refine_leaf", &Octree::refine_leaf, nb::arg("i"),
           "Split leaf `i` into its 8 children; returns True if it was split (level>0).")
      .def("balance", &Octree::balance,
           "Enforce 2:1 graded balance to a fixpoint; returns refinements performed.")
      .def("lohner_indicator", &Octree::lohner_indicator, nb::arg("field"), nb::arg("eps") = 0.01,
           "Löhner normalized-second-difference feature indicator E in [0,1] per leaf from a scalar "
           "field (num_leaves,); large E = steep feature (refine), small = smooth (coarsen).")
      .def("adapt", &Octree::adapt, nb::arg("field"), nb::arg("refine_thresh"),
           nb::arg("coarsen_thresh"), nb::arg("finest_level") = 0u, nb::arg("eps") = 0.01,
           nb::arg("linear") = true,
           "Solution-adaptive step (Löhner-driven): refine where the indicator > refine_thresh (to "
           "finest_level), coarsen sibling groups all < coarsen_thresh, 2:1-balance, and "
           "conservatively remap `field`. MUTATES the octree in place; returns the remapped field "
           "(M,). `linear` uses minmod-limited prolongation (else piecewise-constant).")
      .def("write_vtu", &Octree::write_vtu, nb::arg("path"), nb::arg("name"), nb::arg("field"),
           "Write the octree + a per-leaf scalar field (num_leaves,) as a VTK UnstructuredGrid "
           "(.vtu, ASCII, one cell per leaf), openable in ParaView.");

  nb::class_<Poisson>(
      m, "Poisson",
      "Cell-centered finite-volume Poisson solver (L u = rhs) on an Octree, by a geometric-multigrid "
      "V-cycle. L is the conservative two-point FV Laplacian (suite sign). The hierarchy snapshots "
      "the octree at construction; per-leaf arrays are (num_leaves,) float64 in Z-order slots.")
      .def(nb::init<const Octree&, bool>(), nb::arg("octree"), nb::arg("periodic") = true,
           "Build the multigrid hierarchy from `octree`. periodic=True solves the singular periodic "
           "problem (constant null space removed each cycle).")
      .def_prop_ro("num_leaves", &Poisson::num_leaves, "Leaves on the finest level.")
      .def_prop_ro("num_levels", &Poisson::num_levels, "Number of multigrid levels.")
      .def("apply", &Poisson::apply, nb::arg("u"),
           "L applied to u (the FV Laplacian); use b = apply(u_exact) to manufacture a RHS. "
           "(num_leaves,) -> (num_leaves,).")
      .def("residual", &Poisson::residual, nb::arg("u"), nb::arg("rhs"),
           "Volume-weighted L2 residual norm sqrt(sum V*(rhs - L u)^2).")
      .def("solve", &Poisson::solve, nb::arg("rhs"), nb::arg("x0") = std::nullopt,
           nb::arg("cycles") = 20, nb::arg("pre") = 2, nb::arg("post") = 2, nb::arg("tol") = 0.0,
           "Solve L u = rhs with up to `cycles` V-cycles (pre/post Gauss-Seidel sweeps), from x0 or "
           "0, stopping once residual <= tol (tol<=0 disables). Returns (u (num_leaves,), "
           "final_residual, cycles_done).");

  nb::class_<Flow>(
      m, "Flow",
      "Collocated incompressible Stokes/Navier-Stokes step on an Octree with a cut-cell immersed "
      "boundary (no-slip on an SDF solid). step() = implicit viscous momentum predictor + "
      "Almgren-Bell-Colella rotational projection. Drive with a body force and iterate to steady "
      "state; velocities are per-leaf (num_leaves,) in Z-order slots.")
      .def(nb::init<const Octree&, double, double, double>(), nb::arg("octree"),
           nb::arg("density") = 1.0, nb::arg("viscosity") = 1.0, nb::arg("dt") = 1e6,
           nb::keep_alive<1, 2>(),  // keep the octree alive for the Flow's lifetime (borrowed by ref)
           "Create a flow on `octree` with the given density, viscosity and time step. A large dt "
           "drives straight to the steady (Stokes) solution. The octree is borrowed by reference.")
      .def(nb::init<DistributedOctree&, double, double, double>(), nb::arg("octree"),
           nb::arg("density") = 1.0, nb::arg("viscosity") = 1.0, nb::arg("dt") = 1e6,
           nb::keep_alive<1, 2>(),
           "DISTRIBUTED (mpi4py, collective): run this solver on one ORB block of a "
           "DistributedOctree — the whole step (momentum, pressure, overlays, adaptivity) then "
           "executes multi-rank through the ±2 leaf ghost registry. Per-leaf arrays are this "
           "rank's LOCAL leaves. setSolid/step/project/begin_adapt/finish_adapt/rebalance_mpi "
           "are collective. np=1 is bit-identical to the single-rank constructor.")
      .def("rebalance_mpi", &Flow::rebalance_mpi, nb::arg("sdf"),
           "DISTRIBUTED: weighted-ORB load rebalance — migrates the leaves WITH the state (u, p) "
           "to the new owners and rebuilds every solver structure (collective). num_leaves is "
           "refreshed.")
      .def_prop_ro("num_leaves", &Flow::num_leaves, "Number of leaves.")
      .def("set_solid", &Flow::set_solid, nb::arg("sdf"),
           "Build the cut-cell operators from a signed-distance callable f(x,y,z) (>0 fluid, <0 "
           "solid) and zero the fields. Call before stepping; re-call to change the geometry.")
      .def("set_solid_spheres", &Flow::set_solid_spheres, nb::arg("centers"), nb::arg("radii"),
           nb::arg("periodic") = true,
           "set_solid for a UNION OF SPHERES, evaluated natively instead of through a Python "
           "callback — the porous-media geometry. `centers` is (M,3), `radii` is (M,) or (1,); "
           "`periodic` uses the minimum-image convention over the octree's own box extent. "
           "Prefer this to set_solid at bed scale: set_solid samples the SDF tens of times per "
           "leaf (operator build, overlay classification at virtual positions, the openness "
           "probe), so a Python callback dominates everything else — measured >1h43m of pure "
           "numpy on an 11.35M-leaf 180-sphere bed before the GPU ran a single kernel.")
      .def("set_body_force", &Flow::set_body_force, nb::arg("fx"), nb::arg("fy"), nb::arg("fz"),
           "Set the per-volume body force (e.g. a pressure gradient) driving the flow.")
      .def("set_advection", &Flow::set_advection, nb::arg("on"),
           "Enable explicit momentum advection (Navier-Stokes); off = Stokes.")
      .def("set_ghost_gradient", &Flow::set_ghost_gradient, nb::arg("on"),
           "Directional ghost cell-gradient on cut cells for the pressure predictor and the "
           "projection's cell correction (2nd-order one-sided, never reads decoupled solid "
           "pressure — removes the gauge-dependent O(1/h) cut-cell gradient error of the plain "
           "ABC gradient). The aperture projection itself is unchanged. Call before set_solid.")
      .def("set_aperture_order", &Flow::set_aperture_order, nb::arg("order"),
           "Aperture estimator for the (fallback) aperture projection: 2 = analytic "
           "marching-squares (DEFAULT since 2026-08-26), 1 = legacy one-sample model. "
           "Call before set_solid.")
      .def("set_ghost_projection", &Flow::set_ghost_projection, nb::arg("on"),
           nb::arg("matrix_order") = 2, nb::arg("rhs_order") = 2,
           "DEFAULT since 2026-08-25 (AUTO: ghost, with an aperture fallback + stderr notice when "
           "the finest band is too thin): the fluid-only constraint scheme — family-free, unconditionally "
           "stable, protocol-independent (flow's attractor-campaign verdicts; == flow's "
           "set_collocated_scheme('ghost')). FULL directional ghost-cell projection (the AMR "
           "port): binary-openness pressure operator + wall-anchored closure overlay on the "
           "finest-band rows, MG-preconditioned BiCGStab, ghost-closed divergence constraint; "
           "implies set_ghost_gradient. (matrix_order, rhs_order) closure orders: (2, 2) "
           "default and the only pair cleared for production — the (1, 2) mixed form is "
           "march-UNSTABLE above ~2000 spheres (flow hardening Phase A), kept callable for "
           "parity records only. Raises if the finest band is too thin (a closure would cross "
           "a 2:1 boundary). Call before set_solid.")
      .def("set_ghost_sampled", &Flow::set_ghost_sampled, nb::arg("on"),
           "MIXED-LEVEL CUT BAND (docs/amr_mixed_level_cut_band_plan.md): allow cut cells at "
           "MULTIPLE octree levels — the finest-band contract is dropped. Chain entries that "
           "cross a 2:1 boundary become degree-2 LS virtual samples at the uniform closure "
           "positions (identity weights at same level, so a uniform finest band is BIT-IDENTICAL "
           "to set_ghost_sampled(False)); face classification uses the level-aware canonical "
           "openness; the momentum xi-row seam correction and the wall-aware C/F tangential "
           "fallback ride along. Implies the ghost projection (engages when the resolved scheme "
           "is ghost — the AUTO default or an explicit set_ghost_projection(True)). Single-rank "
           "only (the distributed sample halo is a later rung). Call before set_solid.")
      .def("set_pressure", &Flow::set_pressure, nb::arg("values"),
           "Write the accumulated rotational pressure from a (num_leaves,) array — restart, or "
           "re-accumulation policies after finish_adapt (at steady-state dt the transferred p is "
           "the load-bearing state; zeroing it after a coarsening adapt lets it re-accumulate "
           "cleanly).")
      .def("set_dt", &Flow::set_dt, nb::arg("dt"),
           "Change the time step. dt is BAKED INTO the momentum operator at build time "
           "(idiag = rho/dt), so a set_dt must be followed by set_solid or the operator stays "
           "stale. set_solid reallocates and zeroes u and p, so a dt SWITCH mid-march is: read "
           "velocity()/pressure() -> set_dt -> set_solid -> set_velocity()/set_pressure(). That "
           "sequence is the dt-cycling protocol of the attractor-family batteries.")
      .def("begin_adapt", &Flow::begin_adapt,
           "Snapshot the octree topology + (u, p) ahead of an external mesh mutation (adapt / "
           "refine_to_sphere / refine_to_sdf / balance on the SAME Octree object). Pair with "
           "finish_adapt.")
      .def("finish_adapt", &Flow::finish_adapt, nb::arg("sdf"),
           "Rebuild the solver on the mutated octree and conservatively transfer the snapshotted "
           "u and p onto it (minmod-limited linear remap). `sdf` is re-sampled on the new leaves "
           "(pass the same geometry callable as set_solid); keep the cut band at the finest "
           "level on the new mesh (re-run refine_to_sdf on the geometry band after a "
           "solution-driven adapt). The advecting face field restarts from the cell average for "
           "one step.")
      .def("set_cf_scheme", &Flow::set_cf_scheme, nb::arg("scheme"),
           "Coarse/fine (2:1) interface scheme: 0 = standard two-point flux (default, 1st-order "
           "at level boundaries), 1 = Martin-Cartwright tangential quadratic (2nd-order; applied "
           "to the momentum diffusion, the divergence constraint, and the pressure gradients — "
           "the pressure matrix/MG stays standard, which does not move the steady solution). "
           "Works with both the aperture and the ghost projection. Call before set_solid.")
      .def("set_advection_scheme", &Flow::set_advection_scheme, nb::arg("scheme"),
           "High-order advection flux: 0 = second-order upwind (default), 1 = Koren TVD.")
      .def("set_implicit_advection", &Flow::set_implicit_advection, nb::arg("on"),
           "Implicit first-order-upwind deferred-correction advection (default on): unconditionally "
           "stable. Off = fully explicit high-order advection.")
      .def("set_momentum_mg", &Flow::set_momentum_mg, nb::arg("on"),
           "Use the Galerkin velocity multigrid as the momentum solve preconditioner (default on; "
           "makes the momentum solve scale with resolution). Call before set_solid.")
      .def("set_momentum_gs", &Flow::set_momentum_gs, nb::arg("on"),
           "Use the symmetric multicolour Gauss-Seidel smoother in the momentum multigrid (default "
           "off = weighted Jacobi). Call before set_solid.")
      .def("set_velocity_mg_staircase", &Flow::set_velocity_mg_staircase, nb::arg("on"),
           "Use the rediscretised staircase velocity-MG instead of Galerkin (default off).")
      .def("set_momentum_mg_solver", &Flow::set_momentum_mg_solver, nb::arg("on"),
           "Solve the momentum predictor with the velocity-MG as the solver (no Krylov), mirroring "
           "flow's velocity solve (default off = BiCgStab with the MG as preconditioner).")
      .def("set_outer_iterations", &Flow::set_outer_iterations, nb::arg("n"), nb::arg("tol") = 1e-6,
           "Picard outer iterations over the lagged advection per step (default 1).")
      .def("step", &Flow::step, nb::arg("mom_iters") = 100, nb::arg("pres_iters") = 60,
           "Advance one collocated projection step on device: `mom_iters` momentum solver iterations "
           "(BiCGStab/MG), `pres_iters` pressure MG-PCG iterations.")
      .def("velocity", &Flow::velocity, nb::arg("component"),
           "Per-leaf velocity component (0=x,1=y,2=z), (num_leaves,) float64.")
      .def("velocities", &Flow::velocities,
           "All three velocity components, (num_leaves, 3) float64.")
      .def("is_fluid", &Flow::is_fluid, "Per-leaf fluid mask (False in the solid), (num_leaves,) bool.")
      .def("divergence_norm", &Flow::divergence_norm,
           "Volume-weighted L2 norm of the residual cell divergence (projection-quality diagnostic).")
      .def("divergence_norm_face", &Flow::divergence_norm_face,
           "L2 norm of the divergence of the ABC divergence-free FACE field (≈ pressure-solve "
           "residual, far below divergence_norm — including across 2:1 interfaces).")
      .def("pressure", &Flow::pressure, "Per-leaf pressure (incremental-rotational p), (num_leaves,) float64.")
      .def("face_field", &Flow::face_field,
           "ABC divergence-free FACE velocity, one value per CSR (sub)face (conservative flux / "
           "streamline post-processing).")
      .def("set_velocity", &Flow::set_velocity, nb::arg("component"), nb::arg("values"),
           "Write velocity component c (0=x,1=y,2=z) from a (num_leaves,) array — initial conditions, "
           "restart, or warm-start. Call before step()/project().")
      .def("project", &Flow::project, nb::arg("pres_iters") = 60,
           "Pressure projection only (no momentum solve) — project an externally-set velocity field to "
           "divergence-free. Returns nothing; read the result via velocity()/velocities().")
      .def("last_mom_iters", &Flow::last_mom_iters,
           "Total momentum BiCGStab iterations (summed over the 3 components) of the last step.")
      .def("last_pres_iters", &Flow::last_pres_iters, "Pressure PCG iterations of the last step.")
      .def("last_outer_iters", &Flow::last_outer_iters,
           "Picard outer iterations actually run in the last step (1 unless set_outer_iterations(>1)).");

  nb::class_<DistributedOctree>(
      m, "DistributedOctree",
      "MPI octree: an ORB block decomposition of a global root grid (one BlockOctree per rank, over "
      "MPI_COMM_WORLD). Construct it collectively; refine/balance/rebalance/face_neighbor_gather are "
      "collective. Per-leaf arrays describe THIS rank's local block in global world coordinates.")
      .def(nb::init<std::array<long, 3>, unsigned, std::array<double, 3>, double,
                    std::array<bool, 3>>(),
           nb::arg("global_root_size"), nb::arg("lmax"),
           nb::arg("origin") = std::array<double, 3>{0, 0, 0}, nb::arg("h0") = 1.0,
           nb::arg("periodic") = std::array<bool, 3>{true, true, true},
           "Decompose `global_root_size` root cells (each `lmax` levels deep) across the ranks of "
           "MPI_COMM_WORLD via ORB. `origin`/`h0` place the global grid; `periodic` per axis.")
      .def_prop_ro("rank", &DistributedOctree::rank, "This process's MPI rank.")
      .def_prop_ro("size", &DistributedOctree::size, "Number of ranks (blocks).")
      .def_prop_ro("num_leaves", &DistributedOctree::num_leaves, "Leaves owned by this rank.")
      .def_prop_ro("lmax", &DistributedOctree::lmax, "Root-cell level.")
      .def_prop_ro("h0", &DistributedOctree::h0, "Finest cell width in world units.")
      .def_prop_ro("block_origin_root", &DistributedOctree::block_origin_root,
                   "This rank's block lower corner, in global root-cell coordinates.")
      .def_prop_ro("block_brick", &DistributedOctree::block_brick,
                   "This rank's block size in root cells per axis.")
      .def_prop_ro("global_root_size", &DistributedOctree::global_root_size,
                   "Global grid size in root cells per axis.")
      .def("centers", &DistributedOctree::centers,
           "Local leaf world centres, (num_leaves, 3) float64 (global coordinates).")
      .def("sizes", &DistributedOctree::sizes, "Local leaf world widths, (num_leaves,) float64.")
      .def("levels", &DistributedOctree::levels, "Local leaf levels, (num_leaves,) int32.")
      .def("codes", &DistributedOctree::codes,
           "Local leaf block-local Morton origin codes, (num_leaves,) uint64.")
      .def("refine_to_sphere", &DistributedOctree::refine_to_sphere, nb::arg("center"),
           nb::arg("radius"), nb::arg("target_level") = 0u, nb::arg("band") = 1.0,
           nb::arg("balance") = true,
           "Refine the local block toward a GLOBAL sphere surface down to target_level, then (if "
           "balance) restore cross-block 2:1 balance collectively. Returns the local count.")
      .def("refine_to_sdf", &DistributedOctree::refine_to_sdf, nb::arg("sdf"),
           nb::arg("target_level") = 0u, nb::arg("band") = 1.0, nb::arg("balance") = true,
           "Refine the local block toward an arbitrary GLOBAL surface given as a callable f(x,y,z)->"
           "distance (suite sign: <0 inside solid) — the distributed analogue of Octree.refine_to_sdf, "
           "for rings / packed beds / any non-sphere geometry. Collective when balance=True.")
      .def("balance", &DistributedOctree::balance,
           "Restore cross-block 2:1 graded balance (collective). Returns this rank's refinements.")
      .def("rebalance", &DistributedOctree::rebalance, nb::arg("fields"),
           "Re-decompose by leaf count (weighted ORB) and migrate leaves + their fields. `fields` "
           "is (num_leaves, K) float64; returns this rank's (M, K) columns after migration. Pure "
           "redistribution; the partition is updated in place (collective).")
      .def("face_neighbor_gather", &DistributedOctree::face_neighbor_gather, nb::arg("field"),
           nb::arg("sentinel") = 0.0,
           "For each local leaf, the field value across each of its 6 faces, gathered over the "
           "owner-based halo. `field` is (num_leaves,); returns (num_leaves, 6) laid out "
           "[+x,-x,+y,-y,+z,-z]; domain boundaries carry `sentinel` (collective).")
      .def("lohner_indicator", &DistributedOctree::lohner_indicator, nb::arg("field"),
           nb::arg("eps") = 0.01,
           "Löhner feature indicator per local leaf, evaluated across the owner-based halo so "
           "cross-block neighbours count exactly as in a whole-domain solve. `field` is "
           "(num_leaves,); returns (num_leaves,) (collective).")
      .def("adapt", &DistributedOctree::adapt, nb::arg("field"), nb::arg("refine_thresh"),
           nb::arg("coarsen_thresh"), nb::arg("finest_level") = 0u, nb::arg("eps") = 0.01,
           nb::arg("linear") = true,
           "Distributed solution-adaptive step (Löhner-driven): refine/coarsen each block, restore "
           "cross-block 2:1 balance, and conservatively remap `field` (num_leaves,) onto the new "
           "local mesh. MUTATES the octree in place (keeping ORB ownership); returns the remapped "
           "local field (M,). Bit-identical across rank counts (collective).")
      .def("write_vtu", &DistributedOctree::write_vtu, nb::arg("path"), nb::arg("name"),
           nb::arg("field"),
           "Write this rank's local octree + a per-leaf scalar (num_leaves,) as a .vtu (one file "
           "per rank; combine in ParaView).");
}

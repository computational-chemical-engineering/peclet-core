// peclet.core.geom — Python authoring for the shared analytic-SDF scene layer.
//
// The missing ergonomic half of suite/docs/ANALYTIC_SDF_GEOMETRY.md: SceneBuilder has been C++
// only, so every Python consumer (flow's set_scene, dem's add_analytic_wall / add_scene_shape,
// the campaign gates) hand-assembled the flat (node_ints, node_reals, inst_ints, inst_reals)
// arrays. This module binds the builder itself, batch evaluation, lattice baking (the dem
// grid-particle path), and geom::bodyProperties — mass, COM, full inertia tensor, and the
// principal decomposition as three moments plus a QUATERNION — with addReframed closing the
// non-principal-reference-frame question exactly (one composed transform, no resampling).
//
// Host-only: no Kokkos, no MPI. Preprocessing tooling by design.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "peclet/core/geom/body_properties.hpp"
#include "peclet/core/geom/scene_builder.hpp"
#include "peclet/core/geom/scene_query.hpp"

namespace nb = nanobind;
namespace g = peclet::core::geom;
using peclet::core::Quat;
using peclet::core::Vec3;

namespace {

int kindFromName(const std::string& k) {
  if (k == "sphere") return g::kSphere;
  if (k == "box") return g::kBox;
  if (k == "hollow_cylinder") return g::kHollowCylinder;
  if (k == "hollow_cylinder_shell") return g::kHollowCylinderShell;
  if (k == "capsule") return g::kCapsule;
  if (k == "torus") return g::kTorus;
  if (k == "cone") return g::kCone;
  if (k == "ellipsoid") return g::kEllipsoid;
  if (k == "superquadric") return g::kSuperquadric;
  throw std::invalid_argument(
      "unknown leaf kind '" + k +
      "' (sphere, box, hollow_cylinder, hollow_cylinder_shell, capsule, torus, cone, ellipsoid, "
      "superquadric; grids go through add_grid)");
}

g::Transform<double> makeTransform(std::array<double, 3> t, std::array<double, 4> q, double s) {
  g::Transform<double> tr;
  tr.translation = Vec3<double>{t[0], t[1], t[2]};
  tr.rotation = Quat<double>{q[0], q[1], q[2], q[3]};
  tr.scale = s;
  return tr;
}

template <class T>
nb::ndarray<nb::numpy, T> toArray(std::vector<T>&& v, std::initializer_list<std::size_t> shape) {
  auto* held = new std::vector<T>(std::move(v));
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  std::vector<std::size_t> sh(shape);
  return nb::ndarray<nb::numpy, T>(held->data(), sh.size(), sh.data(), owner);
}

/// A thin owner: the builder plus a cached view for evaluation.
struct PyScene {
  g::SceneBuilder<double> b;

  auto rootEval(int root) const {
    const g::SceneView<double> sv = b.view();
    if (root < 0 || root >= sv.nodeCount)
      throw std::out_of_range("root node index out of range");
    return [sv, root](Vec3<double> p) {
      return g::evalTree<double>(g::TablePtr<g::ShapeNode<double>>{sv.nodes}, sv.nodeCount, root, p,
                                 g::TablePtr<g::GridDesc<double>>{sv.grids},
                                 g::PoolPtr<float>{sv.samples});
    };
  }
};

}  // namespace

NB_MODULE(geom, m) {
  m.doc() =
      "Analytic-SDF scene authoring: SceneBuilder (leaves + CSG + transforms + instancing), batch "
      "evaluation, lattice baking, and rigid-body mass properties (mass, COM, inertia tensor, "
      "principal moments + quaternion) by implicit quadrature.";

  nb::class_<PyScene>(m, "SceneBuilder")
      .def(nb::init<>())
      .def(
          "add_leaf",
          [](PyScene& s, const std::string& kind, std::vector<double> params,
             std::array<double, 3> translation, std::array<double, 4> rotation, double scale) {
            std::initializer_list<double> il{};  // addLeaf takes an initializer_list; go via node
            g::ShapeNode<double> nd;
            nd.kind = kindFromName(kind);
            for (std::size_t i = 0; i < params.size() && i < 8; ++i)
              nd.params[i] = params[i];
            nd.transform = makeTransform(translation, rotation, scale);
            (void)il;
            return s.b.addNode(nd);
          },
          nb::arg("kind"), nb::arg("params"),
          nb::arg("translation") = std::array<double, 3>{0, 0, 0},
          nb::arg("rotation") = std::array<double, 4>{0, 0, 0, 1}, nb::arg("scale") = 1.0,
          "Add a leaf primitive; returns its node index. kind: sphere [r], box [hx,hy,hz], "
          "hollow_cylinder [rOuter,height,thickness] (y axis, distance-exact), "
          "hollow_cylinder_shell [rOuter,rInner,height] (z axis, sign-exact), capsule "
          "[r,halfLength] (y), torus [R,r] (y), cone [rBottom,rTop,halfHeight] (y), ellipsoid "
          "[rx,ry,rz] (BOUND), superquadric [rx,ry,rz,e] (BOUND). rotation is a quaternion "
          "(x, y, z, w).")
      .def(
          "add_union",
          [](PyScene& s, int a, int b2, std::array<double, 3> t, std::array<double, 4> q,
             double sc) { return s.b.addUnion(a, b2, makeTransform(t, q, sc)); },
          nb::arg("a"), nb::arg("b"), nb::arg("translation") = std::array<double, 3>{0, 0, 0},
          nb::arg("rotation") = std::array<double, 4>{0, 0, 0, 1}, nb::arg("scale") = 1.0)
      .def(
          "add_intersection",
          [](PyScene& s, int a, int b2, std::array<double, 3> t, std::array<double, 4> q,
             double sc) { return s.b.addIntersection(a, b2, makeTransform(t, q, sc)); },
          nb::arg("a"), nb::arg("b"), nb::arg("translation") = std::array<double, 3>{0, 0, 0},
          nb::arg("rotation") = std::array<double, 4>{0, 0, 0, 1}, nb::arg("scale") = 1.0)
      .def(
          "add_difference",
          [](PyScene& s, int a, int b2, std::array<double, 3> t, std::array<double, 4> q,
             double sc) { return s.b.addDifference(a, b2, makeTransform(t, q, sc)); },
          nb::arg("a"), nb::arg("b"), nb::arg("translation") = std::array<double, 3>{0, 0, 0},
          nb::arg("rotation") = std::array<double, 4>{0, 0, 0, 1}, nb::arg("scale") = 1.0,
          "a minus b.")
      .def(
          "add_reframed",
          [](PyScene& s, int root, std::array<double, 3> t, std::array<double, 4> q, double sc) {
            return s.b.addReframed(root, makeTransform(t, q, sc));
          },
          nb::arg("root"), nb::arg("translation") = std::array<double, 3>{0, 0, 0},
          nb::arg("rotation") = std::array<double, 4>{0, 0, 0, 1}, nb::arg("scale") = 1.0,
          "Deep-copy the subtree and pre-compose this transform onto the copied root, so "
          "eval_new(p) = eval_old(toLocal(W, p)) -- i.e. this PLACES the copy at W. With the "
          "inverse principal transform from body_properties (rotation = conjugate of its quat, "
          "translation = -com rotated by the conjugate... see principal_frame()), the copy's "
          "canonical frame IS the principal body frame, exactly, with no resampling.")
      .def(
          "principal_frame",
          [](PyScene& s, int root, std::array<double, 3> lo, std::array<double, 3> hi, int n,
             int order, int nseg) {
            const auto bp = g::bodyProperties<double>(s.rootEval(root),
                                                      Vec3<double>{lo[0], lo[1], lo[2]},
                                                      Vec3<double>{hi[0], hi[1], hi[2]}, n, order,
                                                      nseg);
            // W with toLocal(W, p_body) = com + R p_body: q_W = conj(q_R), t_W = invR(q_R, -com)
            const Quat<double> qc{-bp.quat.x, -bp.quat.y, -bp.quat.z, bp.quat.w};
            const Vec3<double> tw = peclet::core::invRotate(
                bp.quat, Vec3<double>{-bp.com.x, -bp.com.y, -bp.com.z});
            g::Transform<double> W;
            W.rotation = qc;
            W.translation = tw;
            return s.b.addReframed(root, W);
          },
          nb::arg("root"), nb::arg("lo"), nb::arg("hi"), nb::arg("n") = 32, nb::arg("order") = 5,
          nb::arg("nseg") = 8,
          "Measure the subtree's mass properties and return a NEW root whose canonical frame is "
          "the principal body frame (COM at the origin, axes principal) -- the one-call answer to "
          "'my shape's reference frame is not its principal frame'. The original subtree is "
          "untouched.")
      .def(
          "add_instance",
          [](PyScene& s, int root, std::array<double, 3> t, std::array<double, 4> q, double sc,
             std::array<double, 3> lin, std::array<double, 3> ang, std::array<double, 3> cen,
             int material) {
            return s.b.addInstance(root, makeTransform(t, q, sc),
                                   Vec3<double>{lin[0], lin[1], lin[2]},
                                   Vec3<double>{ang[0], ang[1], ang[2]},
                                   Vec3<double>{cen[0], cen[1], cen[2]}, material);
          },
          nb::arg("root"), nb::arg("translation") = std::array<double, 3>{0, 0, 0},
          nb::arg("rotation") = std::array<double, 4>{0, 0, 0, 1}, nb::arg("scale") = 1.0,
          nb::arg("lin_vel") = std::array<double, 3>{0, 0, 0},
          nb::arg("ang_vel") = std::array<double, 3>{0, 0, 0},
          nb::arg("center") =
              std::array<double, 3>{std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN()},
          nb::arg("material") = -1,
          "Place a tree in the world; returns the instance index. What flow's set_scene and the "
          "resolved coupling consume. `center` is the centre of rotation for ang_vel: leave it NaN "
          "(the default) and it FOLLOWS the body (the translation, re-anchored on every "
          "set_instance_transform); give any finite point and it is PINNED there in world "
          "coordinates -- (0, 0, 0) included. (Raw instance arrays keep the legacy reading of an "
          "all-zero centre as 'follows the body'; pin a world-origin centre from a raw array through "
          "flow's set_instance_motion(center=...).)")
      .def(
          "encode",
          [](PyScene& s) {
            std::vector<int> ni, ii;
            std::vector<double> nr, ir;
            s.b.encode(ni, nr, ii, ir);
            return nb::make_tuple(toArray(std::move(ni), {ni.size()}),
                                  toArray(std::move(nr), {nr.size()}),
                                  toArray(std::move(ii), {ii.size()}),
                                  toArray(std::move(ir), {ir.size()}));
          },
          "The flat (node_ints, node_reals, inst_ints, inst_reals) arrays -- exactly what "
          "flow.set_scene, dem.add_analytic_wall and dem.add_scene_shape take.")
      .def(
          "eval",
          [](PyScene& s, nb::ndarray<double, nb::shape<-1, 3>, nb::c_contig> pts) {
            const g::SceneView<double> sv = s.b.view();
            const std::size_t n = pts.shape(0);
            std::vector<double> out(n);
            const double* p = pts.data();
            for (std::size_t i = 0; i < n; ++i)
              out[i] = g::evalScene<double>(sv, Vec3<double>{p[3 * i], p[3 * i + 1], p[3 * i + 2]});
            return toArray(std::move(out), {n});
          },
          nb::arg("points"),
          "Signed distance of the whole scene (min over instances) at (N,3) world points. "
          "Authoring/debug; solvers evaluate on device.")
      .def(
          "eval_root",
          [](PyScene& s, int root, nb::ndarray<double, nb::shape<-1, 3>, nb::c_contig> pts) {
            auto f = s.rootEval(root);
            const std::size_t n = pts.shape(0);
            std::vector<double> out(n);
            const double* p = pts.data();
            for (std::size_t i = 0; i < n; ++i)
              out[i] = f(Vec3<double>{p[3 * i], p[3 * i + 1], p[3 * i + 2]});
            return toArray(std::move(out), {n});
          },
          nb::arg("root"), nb::arg("points"),
          "Signed distance of ONE subtree, in its own canonical frame, at (N,3) points.")
      .def(
          "eval_root_grad",
          [](PyScene& s, int root, nb::ndarray<double, nb::shape<-1, 3>, nb::c_contig> pts) {
            const g::SceneView<double> sv = s.b.view();
            if (root < 0 || root >= sv.nodeCount)
              throw std::out_of_range("root node index out of range");
            const std::size_t n = pts.shape(0);
            std::vector<double> val(n), grd(3 * n);
            const double* p = pts.data();
            for (std::size_t i = 0; i < n; ++i) {
              Vec3<double> gv;
              val[i] = g::evalTreeGrad<double>(
                  g::TablePtr<g::ShapeNode<double>>{sv.nodes}, sv.nodeCount, root,
                  Vec3<double>{p[3 * i], p[3 * i + 1], p[3 * i + 2]},
                  g::TablePtr<g::GridDesc<double>>{sv.grids}, g::PoolPtr<float>{sv.samples}, gv);
              grd[3 * i] = gv.x;
              grd[3 * i + 1] = gv.y;
              grd[3 * i + 2] = gv.z;
            }
            return nb::make_tuple(toArray(std::move(val), {n}), toArray(std::move(grd), {n, 3}));
          },
          nb::arg("root"), nb::arg("points"),
          "Value AND analytic gradient of one subtree at (N,3) canonical points, one traversal: "
          "(values (N,), gradients (N,3), unnormalised). At CSG ridges the gradient is the ACTIVE "
          "branch's exact normal (deterministic left tie-break), not a finite-difference smear.")
      .def(
          "bake",
          [](PyScene& s, int root, std::array<double, 3> origin, std::array<double, 3> spacing,
             std::array<int, 3> dims) {
            auto f = s.rootEval(root);
            const int nx = dims[0], ny = dims[1], nz = dims[2];
            if (nx < 2 || ny < 2 || nz < 2)
              throw std::invalid_argument("bake: dims must be >= 2 per axis");
            std::vector<float> out((std::size_t)nx * ny * nz);
            for (int k = 0; k < nz; ++k)
              for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                  out[(std::size_t)i + (std::size_t)j * nx + (std::size_t)k * nx * ny] =
                      (float)f(Vec3<double>{origin[0] + i * spacing[0], origin[1] + j * spacing[1],
                                            origin[2] + k * spacing[2]});
            return toArray(std::move(out), {(std::size_t)nx * ny * nz});
          },
          nb::arg("root"), nb::arg("origin"), nb::arg("spacing"), nb::arg("dims"),
          "Sample a subtree on a lattice: flat float32, x-fastest (idx = i + j*nx + k*nx*ny), at "
          "nodes origin + (i,j,k)*spacing -- the layout dem's grid-SDF particles and shell "
          "generation consume.")
      .def(
          "body_properties",
          [](PyScene& s, int root, std::array<double, 3> lo, std::array<double, 3> hi, int n,
             int order, int nseg, double density) {
            const auto bp = g::bodyProperties<double>(s.rootEval(root),
                                                      Vec3<double>{lo[0], lo[1], lo[2]},
                                                      Vec3<double>{hi[0], hi[1], hi[2]}, n, order,
                                                      nseg, density);
            nb::dict d;
            d["volume"] = bp.volume;
            d["mass"] = bp.mass;
            d["com"] = std::array<double, 3>{bp.com.x, bp.com.y, bp.com.z};
            std::vector<double> I(9), R(9);
            for (int r = 0; r < 3; ++r)
              for (int c = 0; c < 3; ++c) {
                I[(std::size_t)(3 * r + c)] = bp.inertia[r][c];
                R[(std::size_t)(3 * r + c)] = bp.rotation[r][c];
              }
            d["inertia_tensor"] = toArray(std::move(I), {3, 3});
            d["principal"] =
                std::array<double, 3>{bp.principal[0], bp.principal[1], bp.principal[2]};
            d["rotation"] = toArray(std::move(R), {3, 3});
            d["quat"] = std::array<double, 4>{bp.quat.x, bp.quat.y, bp.quat.z, bp.quat.w};
            return d;
          },
          nb::arg("root"), nb::arg("lo"), nb::arg("hi"), nb::arg("n") = 32, nb::arg("order") = 5,
          nb::arg("nseg") = 8, nb::arg("density") = 1.0,
          "Mass properties of {subtree < 0} over [lo, hi] (which MUST contain the solid), constant "
          "density, by implicit quadrature: dict with volume, mass, com, inertia_tensor (3,3 about "
          "the COM), principal (3, ascending), rotation (3,3; columns = principal axes, "
          "p_input = com + R p_body) and quat (x,y,z,w). Sign-exact bracketing means bound-only "
          "leaves (ellipsoid, superquadric, CSG) carry NO systematic bias; measured ~4e-6 relative "
          "at n=32 (ctest geom_body).")
      .def("num_nodes",
           [](PyScene& s) { return (int)s.b.nodes().size(); })
      .def("num_instances",
           [](PyScene& s) { return (int)s.b.instances().size(); });
}

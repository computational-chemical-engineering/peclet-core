// core — host-side scene assembly + the flat encoding the Python bindings speak.
//
// Layer 0 rung 5 of suite/docs/ANALYTIC_SDF_GEOMETRY.md. Two things live here:
//
//   SceneBuilder<Real>  owns the node / grid / sample / instance arrays while a scene is being
//                       composed, and hands out a non-owning SceneView over them. Host-only
//                       (std::vector); device consumers copy the arrays into Views and build their
//                       own SceneView over .data().
//
//   THE FLAT ENCODING   a fixed, documented layout of plain int and Real arrays. This is the
//                       contract a nanobind layer implements: numpy arrays in, POD records out,
//                       no per-field binding code and no reflection. Strides are compile-time
//                       constants so a binding can size its arrays without guessing.
//
// FLAT NODE ENCODING — per node, kNodeIntStride ints and kNodeRealStride reals:
//   ints  [0] kind (ShapeKind)   [1] aux0   [2] aux1
//   reals [0..7]  params[8]
//         [8..10] transform.translation (x, y, z)
//         [11..14] transform.rotation   (x, y, z, w)   -- dem's quaternion order, NOT w-first
//         [15]    transform.scale       (isotropic; contract 1)
//
// FLAT INSTANCE ENCODING — per instance, kInstanceIntStride ints and kInstanceRealStride reals:
//   ints  [0] shapeRoot   [1] materialId
//   reals [0..2]  transform.translation      [3..6] transform.rotation      [7] transform.scale
//         [8..10] linVel   [11..13] angVel   [14..16] center
//
// GRIDS are encoded by their descriptor fields plus one shared sample pool; addGrid() appends to
// the pool and returns the node index, so a caller never computes an offset by hand.
//
// C++20 host code (std::vector); the POD types it fills are the C++17-clean ones from scene.hpp.
#ifndef PECLET_CORE_GEOM_SCENE_BUILDER_HPP
#define PECLET_CORE_GEOM_SCENE_BUILDER_HPP

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/core/geom/scene.hpp"

namespace peclet::core::geom {

inline constexpr int kNodeIntStride = 3;
inline constexpr int kNodeRealStride = 16;
inline constexpr int kInstanceIntStride = 2;
inline constexpr int kInstanceRealStride = 17;

// --- flat <-> POD conversion (the binding boundary) --------------------------------------------

template <class Real>
void encodeNode(const ShapeNode<Real>& n, int* ints, Real* reals) {
  ints[0] = n.kind;
  ints[1] = n.aux0;
  ints[2] = n.aux1;
  for (int i = 0; i < 8; ++i)
    reals[i] = n.params[i];
  reals[8] = n.transform.translation.x;
  reals[9] = n.transform.translation.y;
  reals[10] = n.transform.translation.z;
  reals[11] = n.transform.rotation.x;
  reals[12] = n.transform.rotation.y;
  reals[13] = n.transform.rotation.z;
  reals[14] = n.transform.rotation.w;
  reals[15] = n.transform.scale;
}

template <class Real>
ShapeNode<Real> decodeNode(const int* ints, const Real* reals) {
  ShapeNode<Real> n;
  n.kind = ints[0];
  n.aux0 = ints[1];
  n.aux1 = ints[2];
  for (int i = 0; i < 8; ++i)
    n.params[i] = reals[i];
  n.transform.translation = Vec3<Real>{reals[8], reals[9], reals[10]};
  n.transform.rotation = Quat<Real>{reals[11], reals[12], reals[13], reals[14]};
  n.transform.scale = reals[15];
  return n;
}

template <class Real>
void encodeInstance(const Instance<Real>& v, int* ints, Real* reals) {
  ints[0] = v.shapeRoot;
  ints[1] = v.materialId;
  reals[0] = v.transform.translation.x;
  reals[1] = v.transform.translation.y;
  reals[2] = v.transform.translation.z;
  reals[3] = v.transform.rotation.x;
  reals[4] = v.transform.rotation.y;
  reals[5] = v.transform.rotation.z;
  reals[6] = v.transform.rotation.w;
  reals[7] = v.transform.scale;
  reals[8] = v.linVel.x;
  reals[9] = v.linVel.y;
  reals[10] = v.linVel.z;
  reals[11] = v.angVel.x;
  reals[12] = v.angVel.y;
  reals[13] = v.angVel.z;
  reals[14] = v.center.x;
  reals[15] = v.center.y;
  reals[16] = v.center.z;
}

template <class Real>
Instance<Real> decodeInstance(const int* ints, const Real* reals) {
  Instance<Real> v;
  v.shapeRoot = ints[0];
  v.materialId = ints[1];
  v.transform.translation = Vec3<Real>{reals[0], reals[1], reals[2]};
  v.transform.rotation = Quat<Real>{reals[3], reals[4], reals[5], reals[6]};
  v.transform.scale = reals[7];
  v.linVel = Vec3<Real>{reals[8], reals[9], reals[10]};
  v.angVel = Vec3<Real>{reals[11], reals[12], reals[13]};
  v.center = Vec3<Real>{reals[14], reals[15], reals[16]};
  return v;
}

// --- host assembly -----------------------------------------------------------------------------

/// Owns a scene under construction and hands out a non-owning SceneView over it.
///
/// Every add* returns the new NODE INDEX, so composing a tree reads bottom-up:
///   const int shaft  = b.addLeaf(kHollowCylinder, {0.3, 3.0, 0.3});
///   const int blade  = b.addLeaf(kBox, {0.9, 0.1, 0.25}, {{1.0, 0.5, 0.0}});
///   const int stirrer = b.addUnion(shaft, blade);
///   b.addInstance(stirrer, {}, /*angVel=*/{0, 10.0, 0});   // spinning about y
template <class Real>
class SceneBuilder {
 public:
  /// Analytic leaf. `params` follows the per-kind layout documented on ShapeKind.
  int addLeaf(int kind, std::initializer_list<Real> params, Transform<Real> tr = {}) {
    if (kind >= kCsgBase)
      throw std::invalid_argument("SceneBuilder::addLeaf: kind is a CSG operator, not a leaf");
    if (params.size() > 8)
      throw std::invalid_argument("SceneBuilder::addLeaf: at most 8 parameters");
    ShapeNode<Real> n;
    n.kind = kind;
    int i = 0;
    for (Real v : params)
      n.params[i++] = v;
    n.transform = tr;
    nodes_.push_back(n);
    return static_cast<int>(nodes_.size()) - 1;
  }

  /// Sampled field: copies `samples` into the shared pool, registers a descriptor, returns the
  /// node index. The caller never computes a pool offset.
  int addGrid(const std::vector<float>& samples, int nx, int ny, int nz, Vec3<Real> origin,
              Vec3<Real> spacing, GridExtension ext, Transform<Real> tr = {}) {
    if (nx < 2 || ny < 2 || nz < 2)
      throw std::invalid_argument("SceneBuilder::addGrid: dims must be >= 2 on each axis");
    if (static_cast<std::size_t>(nx) * ny * nz != samples.size())
      throw std::invalid_argument("SceneBuilder::addGrid: samples.size() must equal nx*ny*nz");
    GridDesc<Real> d;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.offset = static_cast<int>(pool_.size());
    d.origin = origin;
    d.invSpacing = Vec3<Real>{spacing.x != Real(0) ? Real(1) / spacing.x : Real(0),
                              spacing.y != Real(0) ? Real(1) / spacing.y : Real(0),
                              spacing.z != Real(0) ? Real(1) / spacing.z : Real(0)};
    d.extension = ext;
    pool_.insert(pool_.end(), samples.begin(), samples.end());
    grids_.push_back(d);
    ShapeNode<Real> n;
    n.kind = kGrid;
    n.aux0 = static_cast<int>(grids_.size()) - 1;
    n.transform = tr;
    nodes_.push_back(n);
    return static_cast<int>(nodes_.size()) - 1;
  }

  int addUnion(int a, int b, Transform<Real> tr = {}) { return addCsg(kUnion, a, b, tr); }
  int addIntersection(int a, int b, Transform<Real> tr = {}) {
    return addCsg(kIntersection, a, b, tr);
  }
  /// a MINUS b.
  int addDifference(int a, int b, Transform<Real> tr = {}) { return addCsg(kDifference, a, b, tr); }

  /// Place a tree in the world. Returns the instance index.
  int addInstance(int shapeRoot, Transform<Real> tr = {}, Vec3<Real> linVel = {0, 0, 0},
                  Vec3<Real> angVel = {0, 0, 0}, Vec3<Real> center = {0, 0, 0},
                  int materialId = -1) {
    requireNode(shapeRoot, "addInstance");
    Instance<Real> v;
    v.shapeRoot = shapeRoot;
    v.materialId = materialId;
    v.transform = tr;
    v.linVel = linVel;
    v.angVel = angVel;
    v.center = center;
    instances_.push_back(v);
    return static_cast<int>(instances_.size()) - 1;
  }

  /// Non-owning view over this builder's arrays. Invalidated by any further add*.
  SceneView<Real> view() const {
    SceneView<Real> s;
    s.nodes = nodes_.data();
    s.nodeCount = static_cast<int>(nodes_.size());
    s.grids = grids_.data();
    s.gridCount = static_cast<int>(grids_.size());
    s.samples = pool_.data();
    s.sampleCount = static_cast<long>(pool_.size());
    s.instances = instances_.data();
    s.instanceCount = static_cast<int>(instances_.size());
    return s;
  }

  const std::vector<ShapeNode<Real>>& nodes() const { return nodes_; }
  const std::vector<GridDesc<Real>>& grids() const { return grids_; }
  const std::vector<float>& samples() const { return pool_; }
  const std::vector<Instance<Real>>& instances() const { return instances_; }

  // --- flat encoding out / in (the binding boundary) -------------------------------------------

  void encode(std::vector<int>& nodeInts, std::vector<Real>& nodeReals, std::vector<int>& instInts,
              std::vector<Real>& instReals) const {
    nodeInts.resize(nodes_.size() * kNodeIntStride);
    nodeReals.resize(nodes_.size() * kNodeRealStride);
    for (std::size_t i = 0; i < nodes_.size(); ++i)
      encodeNode(nodes_[i], &nodeInts[i * kNodeIntStride], &nodeReals[i * kNodeRealStride]);
    instInts.resize(instances_.size() * kInstanceIntStride);
    instReals.resize(instances_.size() * kInstanceRealStride);
    for (std::size_t i = 0; i < instances_.size(); ++i)
      encodeInstance(instances_[i], &instInts[i * kInstanceIntStride],
                     &instReals[i * kInstanceRealStride]);
  }

  /// Rebuild from the flat encoding (grids and the sample pool are carried across verbatim).
  static SceneBuilder decode(const std::vector<int>& nodeInts, const std::vector<Real>& nodeReals,
                             const std::vector<int>& instInts, const std::vector<Real>& instReals,
                             const std::vector<GridDesc<Real>>& grids,
                             const std::vector<float>& pool) {
    if (nodeInts.size() % kNodeIntStride || nodeReals.size() % kNodeRealStride)
      throw std::invalid_argument(
          "SceneBuilder::decode: node arrays are not a whole number of "
          "records");
    const std::size_t nn = nodeInts.size() / kNodeIntStride;
    if (nodeReals.size() / kNodeRealStride != nn)
      throw std::invalid_argument("SceneBuilder::decode: node int/real counts disagree");
    SceneBuilder b;
    b.grids_ = grids;
    b.pool_ = pool;
    b.nodes_.reserve(nn);
    for (std::size_t i = 0; i < nn; ++i)
      b.nodes_.push_back(
          decodeNode<Real>(&nodeInts[i * kNodeIntStride], &nodeReals[i * kNodeRealStride]));
    const std::size_t ni = instInts.size() / kInstanceIntStride;
    b.instances_.reserve(ni);
    for (std::size_t i = 0; i < ni; ++i)
      b.instances_.push_back(decodeInstance<Real>(&instInts[i * kInstanceIntStride],
                                                  &instReals[i * kInstanceRealStride]));
    return b;
  }

 private:
  int addCsg(int kind, int a, int b, Transform<Real> tr) {
    requireNode(a, "addCsg");
    requireNode(b, "addCsg");
    ShapeNode<Real> n;
    n.kind = kind;
    n.aux0 = a;
    n.aux1 = b;
    n.transform = tr;
    nodes_.push_back(n);
    return static_cast<int>(nodes_.size()) - 1;
  }
  void requireNode(int i, const char* who) const {
    if (i < 0 || i >= static_cast<int>(nodes_.size()))
      throw std::invalid_argument(std::string("SceneBuilder::") + who +
                                  ": node index out of range");
  }

  std::vector<ShapeNode<Real>> nodes_;
  std::vector<GridDesc<Real>> grids_;
  std::vector<float> pool_;
  std::vector<Instance<Real>> instances_;
};

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_SCENE_BUILDER_HPP

// core — the OWNING device scene + batched point evaluation (Layer 2-for-core).
//
// Two things the rest of the suite kept hand-rolling, now in one place:
//
//   DeviceScene<Real>   owns the node / instance / grid-descriptor / sample-pool Views of one
//                       scene, built from a SceneBuilder or the flat encoding, and hands out the
//                       non-owning SceneView (raw pointers) that kernels capture by value. This is
//                       the type flow's setScene, dem's analytic walls and voro's SdfScene are
//                       meant to sit on instead of each decoding + uploading privately.
//
//   evalPoints          BATCHED evaluation over arbitrary point arrays — the API decision the AMR
//                       requirements single out (docs/AMR_GEOMETRY_SETUP_REQUIREMENTS.md item 1):
//                       a device View of points in, a device View of distances out, one
//                       parallel_for; plus the host-pointer form for oracles and today's
//                       single-threaded builders. Points are ARBITRARY coordinates — leaf centers,
//                       face centers, ±h_L probes, virtual uniform positions, coarse MG faces —
//                       never a lattice assumption.
//
// The per-point functions come from scene.hpp / scene_query.hpp and are PECLET_HD, so host serial,
// host parallel and device batched paths all evaluate the SAME expressions — bit-identical per
// point across drivers on the same backend, which is what the parity gates assume.
//
// Requires Kokkos (this header owns Views); the per-point machinery it drives does not.
#ifndef PECLET_CORE_GEOM_DEVICE_SCENE_HPP
#define PECLET_CORE_GEOM_DEVICE_SCENE_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <vector>

#include "peclet/core/geom/scene_builder.hpp"
#include "peclet/core/geom/scene_query.hpp"

namespace peclet::core::geom {

/// Owning device-resident scene. Copyable (Views are shared_ptr-like); view() is cheap.
template <class Real, class MemSpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class DeviceScene {
 public:
  DeviceScene() = default;

  /// Upload from a host SceneBuilder (the composition path).
  static DeviceScene from(const SceneBuilder<Real>& b) {
    DeviceScene d;
    d.upload(b.nodes(), b.grids(), b.samples(), b.instances());
    return d;
  }

  /// Upload from the flat encoding (the binding path; see scene_builder.hpp for the layout).
  static DeviceScene fromFlat(const std::vector<int>& nodeInts, const std::vector<Real>& nodeReals,
                              const std::vector<int>& instInts,
                              const std::vector<Real>& instReals) {
    SceneBuilder<Real> b = SceneBuilder<Real>::decode(nodeInts, nodeReals, instInts, instReals,
                                                      /*grids=*/{}, /*pool=*/{});
    return from(b);
  }

  /// Non-owning bundle over the device arrays; capture by value into kernels.
  SceneView<Real> view() const {
    SceneView<Real> s;
    s.nodes = nodes_.data();
    s.nodeCount = static_cast<int>(nodes_.extent(0));
    s.grids = grids_.data();
    s.gridCount = static_cast<int>(grids_.extent(0));
    s.samples = samples_.data();
    s.sampleCount = static_cast<long>(samples_.extent(0));
    s.instances = insts_.data();
    s.instanceCount = static_cast<int>(insts_.extent(0));
    return s;
  }

 private:
  void upload(const std::vector<ShapeNode<Real>>& nodes, const std::vector<GridDesc<Real>>& grids,
              const std::vector<float>& pool, const std::vector<Instance<Real>>& insts) {
    if (nodes.empty())
      throw std::invalid_argument("DeviceScene: empty node table");
    nodes_ = Kokkos::View<ShapeNode<Real>*, MemSpace>("sceneNodes", nodes.size());
    insts_ =
        Kokkos::View<Instance<Real>*, MemSpace>("sceneInsts", insts.empty() ? 1 : insts.size());
    grids_ =
        Kokkos::View<GridDesc<Real>*, MemSpace>("sceneGrids", grids.empty() ? 1 : grids.size());
    samples_ = Kokkos::View<float*, MemSpace>("scenePool", pool.empty() ? 1 : pool.size());
    auto up = [](auto view, const auto& host) {
      auto h = Kokkos::create_mirror_view(view);
      for (std::size_t i = 0; i < host.size(); ++i)
        h(i) = host[i];
      Kokkos::deep_copy(view, h);
    };
    up(nodes_, nodes);
    up(insts_, insts);
    up(grids_, grids);
    up(samples_, pool);
    if (insts.empty())
      instCount0_ = true;
  }

  Kokkos::View<ShapeNode<Real>*, MemSpace> nodes_;
  Kokkos::View<Instance<Real>*, MemSpace> insts_;
  Kokkos::View<GridDesc<Real>*, MemSpace> grids_;
  Kokkos::View<float*, MemSpace> samples_;
  bool instCount0_ = false;

 public:
  bool empty() const { return nodes_.extent(0) == 0; }
  int instanceCount() const { return instCount0_ ? 0 : static_cast<int>(insts_.extent(0)); }
};

// ---------------------------------------------------------------------------------------------
// Batched drivers
// ---------------------------------------------------------------------------------------------

/// Batched sphere-union evaluation, device: pts is (N,3), out is (N), any layouts/memory spaces
/// reachable from ExecSpace. The union/grid views must point into ExecSpace-accessible memory.
template <class ExecSpace, class Real, class PtsView, class OutView>
void evalSphereUnionPoints(const ExecSpace& space, const SphereUnionView<Real>& u,
                           const PeriodicBox<Real>& box, const CandidateGridView<Real>& grid,
                           const PtsView& pts, const OutView& out) {
  const std::size_t n = pts.extent(0);
  Kokkos::parallel_for(
      "peclet::core::geom::evalSphereUnionPoints", Kokkos::RangePolicy<ExecSpace>(space, 0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        out(i) = evalSphereUnionGrid(
            u, Vec3<Real>{Real(pts(i, 0)), Real(pts(i, 1)), Real(pts(i, 2))}, box, grid);
      });
}

/// Host-pointer form (the oracle / today's single-threaded builders): plain loop, no Kokkos types
/// in the signature, same per-point expressions.
template <class Real>
void evalSphereUnionPointsHost(const SphereUnionView<Real>& u, const PeriodicBox<Real>& box,
                               const CandidateGridView<Real>& grid, const Real* pts, std::size_t n,
                               Real* out) {
  for (std::size_t i = 0; i < n; ++i)
    out[i] =
        evalSphereUnionGrid(u, Vec3<Real>{pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]}, box, grid);
}

/// Batched general-scene evaluation (union over instances), device.
template <class ExecSpace, class Real, class PtsView, class OutView>
void evalScenePoints(const ExecSpace& space, const SceneView<Real>& sc, const PtsView& pts,
                     const OutView& out) {
  const std::size_t n = pts.extent(0);
  Kokkos::parallel_for(
      "peclet::core::geom::evalScenePoints", Kokkos::RangePolicy<ExecSpace>(space, 0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        out(i) = evalScene(sc, Vec3<Real>{Real(pts(i, 0)), Real(pts(i, 1)), Real(pts(i, 2))});
      });
}

template <class Real>
void evalScenePointsHost(const SceneView<Real>& sc, const Real* pts, std::size_t n, Real* out) {
  for (std::size_t i = 0; i < n; ++i)
    out[i] = evalScene(sc, Vec3<Real>{pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]});
}

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_DEVICE_SCENE_HPP

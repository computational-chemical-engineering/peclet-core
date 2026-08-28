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

// ---------------------------------------------------------------------------------------------
// SceneQueryDevice: the owning accelerated query every device consumer shares
// ---------------------------------------------------------------------------------------------

/// Owns everything a SceneQueryView needs on ONE memory space: the DeviceScene, the extracted
/// sphere-union arrays when the scene is a plain sphere union (mode selected ONCE, here), the
/// candidate-grid CSR, and the periodicity. flow's set_scene, the resolved-CFD-DEM coupling and
/// the AMR device builders are meant to hold one of these instead of private uploads.
template <class Real, class MemSpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class SceneQueryDevice {
 public:
  SceneQueryDevice() = default;

  /// Build from a host SceneBuilder. `origin`/`extent` bound the candidate grid's coverage
  /// (queries outside fall back to full scans — exact); `accelerate=false` skips the grid (right
  /// for a handful of instances, e.g. a single stirrer).
  static SceneQueryDevice build(const SceneBuilder<Real>& b, Vec3<Real> origin, Vec3<Real> extent,
                                PeriodicBox<Real> box, bool accelerate = true, int nbHint = 0) {
    SceneQueryDevice d;
    d.box_ = box;
    d.scene_ = DeviceScene<Real, MemSpace>::from(b);
    // host-side view for extraction + candidate build (pointers into the builder's host arrays)
    const SceneView<Real> hostView = b.view();
    std::vector<Real> cx, cy, cz, r;
    if (extractSphereUnion(hostView, cx, cy, cz, r)) {
      d.uploadUnion(cx, cy, cz, r);
      if (accelerate) {
        SphereUnionView<Real> hu;
        hu.cx = cx.data();
        hu.cy = cy.data();
        hu.cz = cz.data();
        hu.r = r.data();
        hu.n = static_cast<int>(cx.size());
        hu.equalR = d.uMeta_.equalR;
        hu.r0 = d.uMeta_.r0;
        d.uploadGrid(buildSphereCandidateGrid(hu, origin, extent, box, nbHint));
      }
    } else if (accelerate && hostView.instanceCount > 8) {
      // General-grid construction is nbins x nCertified tree walks; refuse silly costs rather
      // than silently taking minutes — unaccelerated full scans stay exact.
      const long nb = 32L * 32L * 32L;  // upper estimate of the default bin count
      if ((long)hostView.instanceCount * nb <= 50L * 1000L * 1000L)
        d.uploadGrid(buildSceneCandidateGrid(hostView, origin, extent, box));
    }
    return d;
  }

  /// POD view for kernels; valid while this object lives.
  SceneQueryView<Real> view() const {
    SceneQueryView<Real> v;
    v.box = box_;
    if (uMeta_.n > 0) {
      v.u = uMeta_;
      v.u.cx = cx_.data();
      v.u.cy = cy_.data();
      v.u.cz = cz_.data();
      v.u.r = r_.data();
    } else {
      v.scene = scene_.view();
    }
    if (gMeta_.nx > 0) {
      v.grid = gMeta_;
      v.grid.offsets = gOff_.data();
      v.grid.items = gItems_.data();
      v.grid.always = gAlways_.data();
    }
    return v;
  }

  bool sphereFast() const { return uMeta_.n > 0; }
  bool accelerated() const { return gMeta_.nx > 0; }
  const DeviceScene<Real, MemSpace>& scene() const { return scene_; }

 private:
  void uploadUnion(const std::vector<Real>& cx, const std::vector<Real>& cy,
                   const std::vector<Real>& cz, const std::vector<Real>& r) {
    auto up = [](const char* nm, const std::vector<Real>& h) {
      Kokkos::View<Real*, MemSpace> v(nm, h.size());
      auto m = Kokkos::create_mirror_view(v);
      for (std::size_t i = 0; i < h.size(); ++i)
        m(i) = h[i];
      Kokkos::deep_copy(v, m);
      return v;
    };
    cx_ = up("sq_cx", cx);
    cy_ = up("sq_cy", cy);
    cz_ = up("sq_cz", cz);
    r_ = up("sq_r", r);
    uMeta_.n = static_cast<int>(cx.size());
    uMeta_.equalR = true;
    for (std::size_t i = 1; i < r.size(); ++i)
      if (r[i] != r[0])
        uMeta_.equalR = false;
    uMeta_.r0 = r.empty() ? Real(0) : r[0];
  }
  void uploadGrid(const CandidateGrid<Real>& g) {
    gMeta_ = g.meta;
    gMeta_.alwaysCount = static_cast<int>(g.always.size());
    auto up = [](const char* nm, const std::vector<int>& h) {
      Kokkos::View<int*, MemSpace> v(nm, h.empty() ? 1 : h.size());
      auto m = Kokkos::create_mirror_view(v);
      for (std::size_t i = 0; i < h.size(); ++i)
        m(i) = h[i];
      Kokkos::deep_copy(v, m);
      return v;
    };
    gOff_ = up("sq_off", g.offsets);
    gItems_ = up("sq_items", g.items);
    gAlways_ = up("sq_always", g.always);
  }

  DeviceScene<Real, MemSpace> scene_;
  Kokkos::View<Real*, MemSpace> cx_, cy_, cz_, r_;
  Kokkos::View<int*, MemSpace> gOff_, gItems_, gAlways_;
  SphereUnionView<Real> uMeta_{};
  CandidateGridView<Real> gMeta_{};
  PeriodicBox<Real> box_{};
};

/// Batched driver over a SceneQueryView (the form the restructured AMR builders and flow's
/// sampling kernels consume).
template <class ExecSpace, class Real, class PtsView, class OutView>
void evalQueryPoints(const ExecSpace& space, const SceneQueryView<Real>& q, const PtsView& pts,
                     const OutView& out) {
  const std::size_t n = pts.extent(0);
  Kokkos::parallel_for(
      "peclet::core::geom::evalQueryPoints", Kokkos::RangePolicy<ExecSpace>(space, 0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        out(i) = q.eval(Vec3<Real>{Real(pts(i, 0)), Real(pts(i, 1)), Real(pts(i, 2))});
      });
}

}  // namespace peclet::core::geom

#endif  // PECLET_CORE_GEOM_DEVICE_SCENE_HPP

// Bitwise reference driver of the code-health gate (.codehealth/config.toml, tier state_hash).
//
//   state_hash_driver <case> <out-prefix>
//
// Runs one fixed-input case and writes its raw output bytes to <out-prefix>.r<rank> (one file per
// rank); tests/regression/state_hash.py hashes the files with SHA-256 in rank order. A refactor
// that moves host code verbatim leaves every file byte-identical. Run at OMP_NUM_THREADS=1.
//
// Cases (serial ones write on rank 0 only):
//   body     geom::bodyProperties on a sphere (fully degenerate frame), a finite cylinder (one
//            degenerate pair) and a rotated box, plus body_detail::resolveDegenerateFrame on
//            hand-made spectra
//   cutcell  scheme::cutCellGeometryFanTet and geom::faceAperture on three SDF lattices (a sphere,
//            a thin fluid layer between two walls, a thin
//            solid slab)
//   scene    geom::buildSphereCandidateGrid and geom::buildSceneCandidateGrid, periodic and not
//   amg      solver::GraphAMG::build (every level's operator and transfer) and one apply
//   halo     halo::GridHalo exchange (blocking, split, tagged, stacked; Full and Faces) after
//            halo::GridHaloTopology::buildTopology, the whole local field per rank
//   migrate  halo::ParticleMigrator::migrate over 6 steps of moving particles, the local
//            particles per rank sorted by id after each step
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/geom/body_properties.hpp"
#include "peclet/core/geom/quadrature.hpp"
#include "peclet/core/geom/scene_builder.hpp"
#include "peclet/core/geom/scene_query.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"
#include "peclet/core/halo/particle_migrator.hpp"
#include "peclet/core/scheme/cut_cell_geometry.hpp"
#include "peclet/core/solver/graph_amg.hpp"

using namespace peclet::core;

namespace {

/// Raw little-endian bytes of everything a case produces, in production order.
struct Bytes {
  std::vector<char> b;
  template <class T>
  void put(const T& v) {
    const char* p = reinterpret_cast<const char*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
  }
  template <class T>
  void put(const std::vector<T>& v) {
    put(static_cast<std::uint64_t>(v.size()));
    const char* p = reinterpret_cast<const char*>(v.data());
    b.insert(b.end(), p, p + v.size() * sizeof(T));
  }
};

/// Deterministic generator (splitmix64), uniform in [lo, hi).
struct Rng {
  std::uint64_t s;
  double u(double lo, double hi) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return lo + (hi - lo) * static_cast<double>(z >> 11) * 0x1.0p-53;
  }
};

using V3 = Vec3<double>;

// ---- body -------------------------------------------------------------------------------------

void putBody(Bytes& out, const geom::BodyProperties<double>& bp) {
  out.put(bp.volume);
  out.put(bp.mass);
  out.put(bp.com);
  out.put(bp.inertia);
  out.put(bp.principal);
  out.put(bp.rotation);
  out.put(bp.quat);
}

void caseBody(Bytes& out) {
  const double R = 0.35;
  const V3 C{0.11, -0.07, 0.19};
  auto sphere = [&](V3 p) {
    const double dx = p.x - C.x, dy = p.y - C.y, dz = p.z - C.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - R;
  };
  putBody(out, geom::bodyProperties<double>(sphere, V3{-0.6, -0.6, -0.6}, V3{0.9, 0.6, 0.9}, 16));
  // finite cylinder along z (radius 0.3, half height 0.45): two equal principal moments
  auto cylinder = [](V3 p) {
    const double dr = std::sqrt(p.x * p.x + p.y * p.y) - 0.3, dz = std::fabs(p.z) - 0.45;
    const double ox = std::fmax(dr, 0.0), oz = std::fmax(dz, 0.0);
    return std::sqrt(ox * ox + oz * oz) + std::fmin(std::fmax(dr, dz), 0.0);
  };
  putBody(out, geom::bodyProperties<double>(cylinder, V3{-0.4, -0.4, -0.55}, V3{0.4, 0.4, 0.55}, 14,
                                            4, 6, 2.5));
  // rotated, offset box: three distinct moments
  const double hx = 0.4, hy = 0.25, hz = 0.15, c = std::cos(0.3), s = std::sin(0.3);
  const Quat<double> q{s / std::sqrt(2.0), s / std::sqrt(2.0), 0.0, c};
  const V3 B{0.05, 0.02, -0.03};
  auto box = [&](V3 p) {
    const V3 b = invRotate(q, V3{p.x - B.x, p.y - B.y, p.z - B.z});
    const double qx = std::fabs(b.x) - hx, qy = std::fabs(b.y) - hy, qz = std::fabs(b.z) - hz;
    const double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0), oz = std::fmax(qz, 0.0);
    return std::sqrt(ox * ox + oy * oy + oz * oz) +
           std::fmin(std::fmax(qx, std::fmax(qy, qz)), 0.0);
  };
  putBody(out, geom::bodyProperties<double>(box, V3{-0.6, -0.6, -0.6}, V3{0.7, 0.6, 0.6}, 18));

  // resolveDegenerateFrame alone: a rotated basis under a triple, a low pair, a high pair and a
  // separated spectrum
  const double a = 0.4, ca = std::cos(a), sa = std::sin(a), cb = std::cos(1.1), sb = std::sin(1.1);
  const double R0[3][3] = {{ca * cb, -sa, ca * sb}, {sa * cb, ca, sa * sb}, {-sb, 0.0, cb}};
  const double spectra[4][3] = {
      {2.0, 2.0, 2.0}, {1.0, 1.005, 3.0}, {0.5, 2.0, 2.01}, {1.0, 2.0, 3.0}};
  for (const auto& sp : spectra) {
    double w[3] = {sp[0], sp[1], sp[2]}, V[3][3];
    std::memcpy(V, R0, sizeof V);
    geom::body_detail::resolveDegenerateFrame(w, V);
    out.put(w);
    out.put(V);
  }
}

// ---- cutcell ----------------------------------------------------------------------------------

template <class Phi>
void cutCellLattice(Bytes& out, const Phi& phi, int n, const double h[3], bool breakpoints) {
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double xc[3] = {(i + 0.5) * h[0], (j + 0.5) * h[1], (k + 0.5) * h[2]};
        double corner[8], face[6];
        for (int cIdx = 0; cIdx < 8; ++cIdx)
          corner[cIdx] =
              phi(V3{xc[0] + ((cIdx & 1) - 0.5) * h[0], xc[1] + (((cIdx >> 1) & 1) - 0.5) * h[1],
                     xc[2] + (((cIdx >> 2) & 1) - 0.5) * h[2]});
        for (int f = 0; f < 6; ++f) {
          double x[3] = {xc[0], xc[1], xc[2]};
          x[f / 2] += ((f & 1) ? 0.5 : -0.5) * h[f / 2];
          face[f] = phi(V3{x[0], x[1], x[2]});
        }
        const double centre = phi(V3{xc[0], xc[1], xc[2]});
        const scheme::CutCellGeometry g = scheme::cutCellGeometryFanTet(corner, face, centre, h);
        out.put(g.kappa);
        out.put(g.aperture);
        out.put(g.areaPL);
        out.put(g.areaSum);
        out.put(g.nFacet);
        out.put(g.facetArea);
        out.put(g.facetCentroid);
        out.put(static_cast<int>(g.kind));
        const V3 lo{i * h[0], j * h[1], k * h[2]};
        for (int axis = 0; axis < 3; ++axis)
          out.put(geom::faceAperture<double>(phi, lo, axis, h[(axis + 1) % 3], h[(axis + 2) % 3], 4,
                                             8, breakpoints));
      }
}

void caseCutCell(Bytes& out) {
  const int n = 14;
  const double h[3] = {1.0 / n, 1.1 / n, 0.9 / n};
  auto sphere = [](V3 p) {
    const double dx = p.x - 0.52, dy = p.y - 0.47, dz = p.z - 0.44;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - 0.31;
  };
  cutCellLattice(out, sphere, n, h, true);
  // a fluid layer of a quarter cell between two walls, tilted the other way: gap cells
  auto layer = [](V3 p) {
    const double nx = -0.4, ny = 0.3, nz = 0.8660254037844386;
    return 0.009 - std::fabs(nx * (p.x - 0.5) + ny * (p.y - 0.45) + nz * (p.z - 0.55));
  };
  cutCellLattice(out, layer, n, h, true);
  // a slab of a quarter cell, tilted: thin-solid cells
  auto slab = [](V3 p) {
    const double nx = 0.3, ny = 0.2, nz = 0.9327379053088815;
    return std::fabs(nx * (p.x - 0.5) + ny * (p.y - 0.5) + nz * (p.z - 0.45)) - 0.009;
  };
  cutCellLattice(out, slab, n, h, false);
}

// ---- scene ------------------------------------------------------------------------------------

void putGrid(Bytes& out, const geom::CandidateGrid<double>& g) {
  out.put(g.offsets);
  out.put(g.items);
  out.put(g.always);
  out.put(g.instBoundR);
  const geom::CandidateGridView<double>& m = g.meta;
  out.put(m.nx);
  out.put(m.ny);
  out.put(m.nz);
  out.put(m.ox);
  out.put(m.oy);
  out.put(m.oz);
  out.put(m.bx);
  out.put(m.by);
  out.put(m.bz);
  out.put(m.wrap);
}

void caseScene(Bytes& out) {
  Rng rng{20261118};
  const int ns = 60;
  std::vector<double> cx(ns), cy(ns), cz(ns), r(ns);
  for (int i = 0; i < ns; ++i) {
    cx[i] = rng.u(0, 1);
    cy[i] = rng.u(0, 1);
    cz[i] = rng.u(0, 1);
    r[i] = rng.u(0.03, 0.09);
  }
  geom::SphereUnionView<double> u;
  u.cx = cx.data();
  u.cy = cy.data();
  u.cz = cz.data();
  u.r = r.data();
  u.n = ns;
  for (bool periodic : {true, false}) {
    const geom::PeriodicBox<double> box{1.0, 1.0, 1.0, periodic};
    for (int hint : {0, 7})
      putGrid(out, geom::buildSphereCandidateGrid(u, V3{0, 0, 0}, V3{1, 1, 1}, box, hint));
  }
  geom::SphereUnionView<double> ue = u;  // equal radii
  ue.equalR = true;
  ue.r0 = 0.06;
  putGrid(out, geom::buildSphereCandidateGrid(ue, V3{0, 0, 0}, V3{1, 1, 1},
                                              geom::PeriodicBox<double>{1, 1, 1, true}));

  // a mixed scene: certified spheres and a CSG stirrer, a non-certified ellipsoid
  geom::SceneBuilder<double> b;
  const int sph = b.addLeaf(geom::kSphere, {0.06});
  const int shaft = b.addLeaf(geom::kHollowCylinder, {0.03, 0.4, 0.03});
  const int blade =
      b.addLeaf(geom::kBox, {0.12, 0.02, 0.03}, geom::Transform<double>{V3{0.13, 0.0, 0.0}});
  const int stir = b.addUnion(shaft, blade);
  const int ell = b.addLeaf(geom::kEllipsoid, {0.09, 0.05, 0.04});
  for (int i = 0; i < 30; ++i)
    b.addInstance(sph, geom::Transform<double>{V3{rng.u(0, 1), rng.u(0, 1), rng.u(0, 1)}});
  const double a45 = 0.7853981633974483;
  b.addInstance(stir,
                geom::Transform<double>{V3{0.5, 0.5, 0.5},
                                        Quat<double>{0, std::sin(a45), 0, std::cos(a45)}, 1.2});
  b.addInstance(ell, geom::Transform<double>{V3{0.2, 0.7, 0.3}});
  const geom::SceneView<double> sv = b.view();
  for (bool periodic : {true, false}) {
    const geom::PeriodicBox<double> box{1.0, 1.0, 1.0, periodic};
    for (double hint : {0.0, 0.08})
      putGrid(out, geom::buildSceneCandidateGrid(sv, V3{0, 0, 0}, V3{1, 1, 1}, box, hint));
  }
}

// ---- amg --------------------------------------------------------------------------------------

/// Periodic 3D Laplacian, `s` uncoupled components per node, anisotropic weights.
solver::HostCsrOp laplacian(int mx, int my, int mz, int s, const double w[3], double shift) {
  solver::HostCsrOp A;
  const Index nNode = static_cast<Index>(mx) * my * mz;
  A.n = nNode * s;
  A.diag.assign(static_cast<std::size_t>(A.n), 0.0);
  A.start.assign(static_cast<std::size_t>(A.n) + 1, 0);
  auto node = [&](int x, int y, int z) {
    return static_cast<Index>(((z + mz) % mz * my + (y + my) % my) * mx + (x + mx) % mx);
  };
  const int off[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int z = 0; z < mz; ++z)
    for (int y = 0; y < my; ++y)
      for (int x = 0; x < mx; ++x)
        for (int d = 0; d < s; ++d) {
          const Index i = node(x, y, z) * s + d;
          A.diag[static_cast<std::size_t>(i)] = 2.0 * (w[0] + w[1] + w[2]) + shift;
          for (int f = 0; f < 6; ++f) {
            A.nbr.push_back(node(x + off[f][0], y + off[f][1], z + off[f][2]) * s + d);
            A.coef.push_back(-w[f / 2]);
          }
          A.start[static_cast<std::size_t>(i) + 1] = static_cast<Index>(A.nbr.size());
        }
  return A;
}

void caseAmg(Bytes& out) {
  const double iso[3] = {1.0, 1.0, 1.0}, aniso[3] = {1.0, 0.05, 3.0};
  struct Cfg {
    int m[3], s;
    const double* w;
    double shift;
  };
  const Cfg cfgs[] = {{{12, 10, 14}, 1, aniso, 0.02}, {{8, 8, 8}, 3, iso, 0.05}};
  for (const Cfg& c : cfgs) {
    const solver::HostCsrOp A = laplacian(c.m[0], c.m[1], c.m[2], c.s, c.w, c.shift);
    solver::AmgParams prm;
    prm.ndofPerNode = c.s;
    solver::GraphAMG M;
    M.build(A, prm);
    out.put(M.numLevels());
    for (const auto& lv : M.levels()) {
      out.put(lv.A.n);
      out.put(lv.A.diag);
      out.put(lv.A.start);
      out.put(lv.A.nbr);
      out.put(lv.A.coef);
      out.put(lv.Pstart);
      out.put(lv.Pcol);
      out.put(lv.Pval);
      out.put(lv.nc);
    }
    std::vector<double> rhs(static_cast<std::size_t>(A.n)), z;
    for (std::size_t i = 0; i < rhs.size(); ++i)
      rhs[i] =
          std::sin(0.7 * static_cast<double>(i)) + 0.3 * std::cos(0.03 * static_cast<double>(i));
    M.apply(rhs, z);
    out.put(z);
  }
}

// ---- halo -------------------------------------------------------------------------------------

constexpr int kDim = 3;

void haloRun(Bytes& out, int rank, int size, IVec<kDim> gsize, int ghost,
             std::array<bool, kDim> periodic, halo::Neighborhood nb) {
  decomp::BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);
  halo::GridHaloTopology<kDim> topo;
  topo.buildTopology(dec, rank, ghost, periodic, MPI_COMM_WORLD, nb);
  const auto& idx = topo.indexer();
  const Index n = idx.numCellsInclGhost();
  std::vector<double> a0(static_cast<std::size_t>(n), -1.0);
  idx.forEachInner([&](const IVec<kDim>& lmd) {
    IVec<kDim> g{};
    for (int i = 0; i < kDim; ++i)
      g[i] = lmd[i] + idx.originInclGhost()[i];
    a0[static_cast<std::size_t>(idx.localMdToLocal(lmd))] =
        static_cast<double>(dec.linearGlobal(g)) + 0.25;
  });
  View<double> f("f", static_cast<std::size_t>(n));
  auto hf = Kokkos::create_mirror_view(f);
  auto load = [&](double scale) {
    for (Index c = 0; c < n; ++c)
      hf(c) = a0[static_cast<std::size_t>(c)] * scale;
    Kokkos::deep_copy(f, hf);
  };
  auto dump = [&] {
    Kokkos::deep_copy(hf, f);
    out.put(static_cast<std::uint64_t>(n));
    const char* p = reinterpret_cast<const char*>(hf.data());
    out.b.insert(out.b.end(), p, p + n * sizeof(double));
  };
  halo::GridHalo<double> h;
  h.init(topo);
  // three tag-0 exchanges (both shared-memory slot parities), the split form, and a tagged one
  for (int it = 0; it < 3; ++it) {
    load(1.0 + it);
    h.exchange(f);
    dump();
  }
  load(-0.5);
  h.exchangeBegin(f);
  h.exchangeEnd(f);
  dump();
  load(3.0);
  h.exchange(f, 5);
  dump();

  // two fields stacked into one view
  const Index stride = n + 3;
  View<double> fs("fs", static_cast<std::size_t>(2 * stride));
  auto hs = Kokkos::create_mirror_view(fs);
  for (Index c = 0; c < 2 * stride; ++c)
    hs(c) = -2.0;
  for (Index c = 0; c < n; ++c) {
    hs(c) = a0[static_cast<std::size_t>(c)];
    hs(stride + c) = 2.0 * a0[static_cast<std::size_t>(c)] + 1.0;
  }
  Kokkos::deep_copy(fs, hs);
  halo::GridHalo<double> hk;
  hk.init(topo, 2, stride);
  hk.exchange(fs);
  hk.exchange(fs);
  Kokkos::deep_copy(hs, fs);
  out.put(static_cast<std::uint64_t>(2 * stride));
  const char* p = reinterpret_cast<const char*>(hs.data());
  out.b.insert(out.b.end(), p, p + 2 * stride * sizeof(double));
}

void caseHalo(Bytes& out, int rank, int size) {
  // ~2600 ghost cells at np 1 (parallel gather), a few hundred per rank at np 8 (serial copy)
  haloRun(out, rank, size, IVec<kDim>{24, 20, 16}, 1, {true, true, true}, halo::Neighborhood::Full);
  haloRun(out, rank, size, IVec<kDim>{8, 6, 4}, 1, {true, true, true}, halo::Neighborhood::Full);
  haloRun(out, rank, size, IVec<kDim>{40, 36, 32}, 2, {true, false, true},
          halo::Neighborhood::Full);
  haloRun(out, rank, size, IVec<kDim>{20, 16, 12}, 2, {false, true, true},
          halo::Neighborhood::Faces);
}

// ---- migrate ----------------------------------------------------------------------------------

void caseMigrate(Bytes& out, int rank, int size) {
  const IVec<kDim> gsize{20, 16, 12};
  decomp::BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);
  halo::DomainMap<kDim> map;
  for (int i = 0; i < kDim; ++i) {
    map.origin[i] = -1.0;
    map.cellSize[i] = 0.5;
    map.periodic[i] = true;
  }
  halo::ParticleMigrator<kDim> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);
  const double L[3] = {gsize[0] * 0.5, gsize[1] * 0.5, gsize[2] * 0.5};
  const std::int64_t N = 3000;
  const std::size_t stride = 2 * sizeof(std::int64_t);
  std::vector<Vec<kDim>> pos;
  std::vector<char> payload;
  Rng rng{77};
  for (std::int64_t id = 0; id < N; ++id) {
    Vec<kDim> x{};
    for (int i = 0; i < kDim; ++i)
      x[i] = map.origin[i] + rng.u(0, L[i]);
    if (id % size != rank)
      continue;
    pos.push_back(x);
    const std::int64_t rec[2] = {id, 7 * id + 1};
    const char* p = reinterpret_cast<const char*>(rec);
    payload.insert(payload.end(), p, p + stride);
  }
  for (int step = 0; step < 6; ++step) {
    const std::size_t nOut = mig.migrate(pos, payload, stride);
    out.put(static_cast<std::uint64_t>(nOut));
    std::vector<std::size_t> perm(pos.size());
    for (std::size_t k = 0; k < perm.size(); ++k)
      perm[k] = k;
    auto idOf = [&](std::size_t k) {
      std::int64_t id;
      std::memcpy(&id, &payload[k * stride], sizeof id);
      return id;
    };
    std::sort(perm.begin(), perm.end(),
              [&](std::size_t a, std::size_t b) { return idOf(a) < idOf(b); });
    out.put(static_cast<std::uint64_t>(perm.size()));
    for (std::size_t k : perm) {
      out.put(pos[k]);
      out.b.insert(out.b.end(), &payload[k * stride], &payload[k * stride] + stride);
    }
    // move every particle by a deterministic displacement of up to 3 cells, wrapped periodically
    for (std::size_t k = 0; k < pos.size(); ++k) {
      const std::int64_t id = idOf(k);
      for (int i = 0; i < kDim; ++i) {
        const double d = 1.5 * std::sin(0.37 * static_cast<double>(id) + 1.3 * (step + 1) + i);
        double x = pos[k][i] + d - map.origin[i];
        x -= std::floor(x / L[i]) * L[i];
        pos[k][i] = map.origin[i] + x;
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rc = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
      if (argc != 3)
        throw std::invalid_argument("usage: state_hash_driver <case> <out-prefix>");
      const std::string c = argv[1];
      Bytes out;
      const bool serial = c == "body" || c == "cutcell" || c == "scene" || c == "amg";
      if (serial && rank == 0) {
        if (c == "body")
          caseBody(out);
        else if (c == "cutcell")
          caseCutCell(out);
        else if (c == "scene")
          caseScene(out);
        else
          caseAmg(out);
      } else if (c == "halo") {
        caseHalo(out, rank, size);
      } else if (c == "migrate") {
        caseMigrate(out, rank, size);
      } else if (!serial) {
        throw std::invalid_argument("state_hash_driver: unknown case " + c);
      }
      if (!serial || rank == 0) {
        std::ofstream os(std::string(argv[2]) + ".r" + std::to_string(rank), std::ios::binary);
        os.write(out.b.data(), static_cast<std::streamsize>(out.b.size()));
        if (!os)
          throw std::runtime_error("state_hash_driver: write failed");
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "rank %d: %s\n", rank, e.what());
      rc = 1;
    }
  }
  if (rc)
    MPI_Abort(MPI_COMM_WORLD, rc);
  MPI_Finalize();
  return rc;
}

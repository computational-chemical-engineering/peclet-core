// Layer 0 rung 1 gate (suite/docs/ANALYTIC_SDF_GEOMETRY.md): the runtime node layer
// (peclet/core/geom/scene.hpp) reproduces the compile-time leaves exactly, composes conformal
// transforms correctly, evaluates CSG trees iteratively without recursion, and samples grid fields
// bit-exactly in BOTH off-grid extension policies.
//
// As in rung 0, the grid oracles are verbatim transcriptions of dem's sampleGridSdf /
// sampleWallSdf (dem/src/narrowphase.hpp) into this file; the cross-check against dem's compiled
// code is rung 3's gate.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "peclet/core/geom/primitives.hpp"
#include "peclet/core/geom/scene.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;
namespace prim = peclet::core::geom::prim;

static bool bitEqualF(float a, float b) {
  std::uint32_t ba, bb;
  std::memcpy(&ba, &a, sizeof ba);
  std::memcpy(&bb, &b, sizeof bb);
  return ba == bb;
}

// --- Reference: verbatim transcription of dem sampleGridSdf / sampleWallSdf (narrowphase.hpp) ---
// The two differ only in the sign of the clamp residual; kept as two functions so the transcription
// is a literal diff against the origin.
struct RefGrid {
  int nx, ny, nz, off;
  float ox, oy, oz, ivx, ivy, ivz;
};

static float ref_sample(float px, float py, float pz, const RefGrid& d, const float* grid,
                        bool object) {
  const float fx = (px - d.ox) * d.ivx;
  const float fy = (py - d.oy) * d.ivy;
  const float fz = (pz - d.oz) * d.ivz;
  const float cx = std::fmin(std::fmax(fx, 0.0f), (float)(d.nx - 1));
  const float cy = std::fmin(std::fmax(fy, 0.0f), (float)(d.ny - 1));
  const float cz = std::fmin(std::fmax(fz, 0.0f), (float)(d.nz - 1));
  const int ix = (int)cx, iy = (int)cy, iz = (int)cz;
  const int ix1 = ix < d.nx - 1 ? ix + 1 : ix;
  const int iy1 = iy < d.ny - 1 ? iy + 1 : iy;
  const int iz1 = iz < d.nz - 1 ? iz + 1 : iz;
  const float tx = cx - ix, ty = cy - iy, tz = cz - iz;
  const long nxny = (long)d.nx * d.ny;
  const int off = d.off;
  auto at = [&](int x, int y, int z) { return grid[off + (long)z * nxny + (long)y * d.nx + x]; };
  const float c00 = at(ix, iy, iz) * (1 - tx) + at(ix1, iy, iz) * tx;
  const float c10 = at(ix, iy1, iz) * (1 - tx) + at(ix1, iy1, iz) * tx;
  const float c01 = at(ix, iy, iz1) * (1 - tx) + at(ix1, iy, iz1) * tx;
  const float c11 = at(ix, iy1, iz1) * (1 - tx) + at(ix1, iy1, iz1) * tx;
  const float c0 = c00 * (1 - ty) + c10 * ty;
  const float c1 = c01 * (1 - ty) + c11 * ty;
  const float val = c0 * (1 - tz) + c1 * tz;
  const float rx = (d.ivx > 0.0f) ? (fx - cx) / d.ivx : 0.0f;
  const float ry = (d.ivy > 0.0f) ? (fy - cy) / d.ivy : 0.0f;
  const float rz = (d.ivz > 0.0f) ? (fz - cz) / d.ivz : 0.0f;
  const float res = std::sqrt(rx * rx + ry * ry + rz * rz);
  return object ? val + res : val - res;
}

struct Rng {
  std::uint32_t s = 987654321u;
  float u(float lo, float hi) {
    s = s * 1664525u + 1013904223u;
    const float t = static_cast<float>((s >> 8) & 0xFFFFFFu) / static_cast<float>(0x1000000u);
    return lo + (hi - lo) * t;
  }
};

using NodesF = TablePtr<ShapeNode<float>>;
using GridsF = TablePtr<GridDesc<float>>;
using PoolF = PoolPtr<float>;

int main() {
  Rng rng;
  const GridsF noGrids{nullptr};
  const PoolF noPool{nullptr};

  // ---------------------------------------------------------------------------------------
  // (1) A node wrapping a single primitive, identity transform, == the direct leaf call, bit-exact.
  // ---------------------------------------------------------------------------------------
  {
    std::vector<ShapeNode<float>> nv(4);
    nv[0].kind = kSphere;
    nv[0].params[0] = 1.25f;
    nv[1].kind = kBox;
    nv[1].params[0] = 0.75f;
    nv[1].params[1] = 1.10f;
    nv[1].params[2] = 0.40f;
    nv[2].kind = kHollowCylinder;
    nv[2].params[0] = 1.30f;
    nv[2].params[1] = 1.60f;
    nv[2].params[2] = 0.50f;
    nv[3].kind = kHollowCylinderShell;
    nv[3].params[0] = 1.30f;
    nv[3].params[1] = 0.80f;
    nv[3].params[2] = 1.60f;
    const NodesF nodes{nv.data()};

    prim::Sphere<float> lSph{1.25f};
    prim::Box<float> lBox{0.75f, 1.10f, 0.40f};
    prim::HollowCylinder<float> lHc{1.30f, 1.60f, 0.50f};
    prim::HollowCylinderShell<float> lSh{1.30f, 0.80f, 1.60f};

    int n = 0;
    for (int i = 0; i < 3000; ++i) {
      const Vec3<float> p{rng.u(-3, 3), rng.u(-3, 3), rng.u(-3, 3)};
      PECLET_CORE_CHECK(bitEqualF(evalTree<float>(nodes, 4, 0, p, noGrids, noPool), lSph.eval(p)));
      PECLET_CORE_CHECK(bitEqualF(evalTree<float>(nodes, 4, 1, p, noGrids, noPool), lBox.eval(p)));
      PECLET_CORE_CHECK(bitEqualF(evalTree<float>(nodes, 4, 2, p, noGrids, noPool), lHc.eval(p)));
      PECLET_CORE_CHECK(bitEqualF(evalTree<float>(nodes, 4, 3, p, noGrids, noPool), lSh.eval(p)));
      ++n;
    }
    PECLET_CORE_CHECK(n == 3000);
  }

  // ---------------------------------------------------------------------------------------
  // (2) Conformal transform: a translated + rotated + scaled sphere is still an exact distance,
  //     and matches the closed-form answer |p - c| - s*R.
  // ---------------------------------------------------------------------------------------
  {
    std::vector<ShapeNode<double>> nv(1);
    nv[0].kind = kSphere;
    nv[0].params[0] = 1.0;  // unit ball, scaled up by the transform
    const double s = 2.5;
    const Vec3<double> c{0.7, -1.3, 0.4};
    // 60 degrees about a tilted axis — a sphere is rotation-invariant, so the rotation must have
    // NO effect on the value. That is exactly what makes it a good check of the quaternion path.
    const double ang = 1.0471975511965976, ax = 0.6, ay = 0.8, az = 0.0;
    nv[0].transform.translation = c;
    nv[0].transform.rotation =
        Quat<double>{ax * std::sin(ang / 2), ay * std::sin(ang / 2), az * std::sin(ang / 2),
                     std::cos(ang / 2)};
    nv[0].transform.scale = s;
    const TablePtr<ShapeNode<double>> nodes{nv.data()};
    const TablePtr<GridDesc<double>> g{nullptr};
    const PoolPtr<double> pl{nullptr};

    for (int i = 0; i < 2000; ++i) {
      const Vec3<double> p{rng.u(-6, 6), rng.u(-6, 6), rng.u(-6, 6)};
      const double got = evalTree<double>(nodes, 1, 0, p, g, pl);
      const Vec3<double> d = sub(p, c);
      const double want = std::sqrt(dot(d, d)) - s * 1.0;
      PECLET_CORE_CHECK(std::fabs(got - want) < 1e-12);
    }
  }

  // ---------------------------------------------------------------------------------------
  // (3) CSG: a stirrer (shaft cylinder UNION two transformed blade boxes) matches a brute-force
  //     composition of the same leaves, and the iterative walk handles a right-leaning tree.
  // ---------------------------------------------------------------------------------------
  {
    // nodes: 0 = union(1, 2); 1 = shaft; 2 = union(3, 4); 3 = blade A; 4 = blade B
    std::vector<ShapeNode<double>> nv(5);
    nv[0].kind = kUnion;
    nv[0].aux0 = 1;
    nv[0].aux1 = 2;

    nv[1].kind = kHollowCylinder;  // shaft: a thick tube about y
    nv[1].params[0] = 0.30;
    nv[1].params[1] = 3.00;
    nv[1].params[2] = 0.30;

    nv[2].kind = kUnion;
    nv[2].aux0 = 3;
    nv[2].aux1 = 4;

    nv[3].kind = kBox;  // blade A, offset out along +x
    nv[3].params[0] = 0.90;
    nv[3].params[1] = 0.10;
    nv[3].params[2] = 0.25;
    nv[3].transform.translation = Vec3<double>{1.0, 0.5, 0.0};

    nv[4].kind = kBox;  // blade B, opposite side and rotated 90 deg about y
    nv[4].params[0] = 0.90;
    nv[4].params[1] = 0.10;
    nv[4].params[2] = 0.25;
    nv[4].transform.translation = Vec3<double>{-1.0, -0.5, 0.0};
    nv[4].transform.rotation = Quat<double>{0, std::sin(0.7853981633974483), 0,
                                            std::cos(0.7853981633974483)};
    const TablePtr<ShapeNode<double>> nodes{nv.data()};
    const TablePtr<GridDesc<double>> g{nullptr};
    const PoolPtr<double> pl{nullptr};

    prim::HollowCylinder<double> shaft{0.30, 3.00, 0.30};
    prim::Box<double> blade{0.90, 0.10, 0.25};

    for (int i = 0; i < 4000; ++i) {
      const Vec3<double> p{rng.u(-4, 4), rng.u(-4, 4), rng.u(-4, 4)};
      // brute force: transform into each blade's frame by hand and take the min of the three
      const double dShaft = shaft.eval(p);
      const double dA = blade.eval(sub(p, nv[3].transform.translation));
      const double dB =
          blade.eval(invRotate(nv[4].transform.rotation, sub(p, nv[4].transform.translation)));
      const double want = std::fmin(dShaft, std::fmin(dA, dB));
      const double got = evalTree<double>(nodes, 5, 0, p, g, pl);
      PECLET_CORE_CHECK(std::fabs(got - want) < 1e-14);
    }

    // intersection and difference against the same two leaves
    std::vector<ShapeNode<double>> iv(3);
    iv[0].aux0 = 1;
    iv[0].aux1 = 2;
    iv[1].kind = kSphere;
    iv[1].params[0] = 1.0;
    iv[2].kind = kBox;
    iv[2].params[0] = iv[2].params[1] = iv[2].params[2] = 0.8;
    prim::Sphere<double> ls{1.0};
    prim::Box<double> lb{0.8, 0.8, 0.8};
    const TablePtr<ShapeNode<double>> inodes{iv.data()};
    for (int i = 0; i < 2000; ++i) {
      const Vec3<double> p{rng.u(-3, 3), rng.u(-3, 3), rng.u(-3, 3)};
      const double a = ls.eval(p), b = lb.eval(p);
      iv[0].kind = kIntersection;
      PECLET_CORE_CHECK(evalTree<double>(inodes, 3, 0, p, g, pl) == std::fmax(a, b));
      iv[0].kind = kDifference;
      PECLET_CORE_CHECK(evalTree<double>(inodes, 3, 0, p, g, pl) == std::fmax(a, -b));
      iv[0].kind = kUnion;
      PECLET_CORE_CHECK(evalTree<double>(inodes, 3, 0, p, g, pl) == std::fmin(a, b));
    }
  }

  // ---------------------------------------------------------------------------------------
  // (4) Grid leaf: both extension policies reproduce their dem origin bit-exactly, and the two
  //     policies genuinely disagree outside the stored box (the 71k-lost-grains guard).
  // ---------------------------------------------------------------------------------------
  {
    const int NX = 7, NY = 6, NZ = 5;
    std::vector<float> pool((std::size_t)NX * NY * NZ);
    // A sphere sampled on the lattice, so the field is a real SDF rather than noise.
    const float sp = 0.35f, o0 = -1.0f;
    for (int k = 0; k < NZ; ++k)
      for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
          const float x = o0 + sp * i, y = o0 + sp * j, z = o0 + sp * k;
          pool[(std::size_t)i + (std::size_t)j * NX + (std::size_t)k * NX * NY] =
              std::sqrt(x * x + y * y + z * z) - 0.6f;
        }

    RefGrid rg{NX, NY, NZ, 0, o0, o0, o0, 1.0f / sp, 1.0f / sp, 1.0f / sp};
    GridDesc<float> dObj;
    dObj.nx = NX;
    dObj.ny = NY;
    dObj.nz = NZ;
    dObj.offset = 0;
    dObj.origin = Vec3<float>{o0, o0, o0};
    dObj.invSpacing = Vec3<float>{1.0f / sp, 1.0f / sp, 1.0f / sp};
    dObj.extension = GridExtension::kObject;
    GridDesc<float> dCon = dObj;
    dCon.extension = GridExtension::kContainer;

    std::vector<GridDesc<float>> gv{dObj, dCon};
    const GridsF grids{gv.data()};
    const PoolF pl{pool.data()};

    std::vector<ShapeNode<float>> nv(2);
    nv[0].kind = kGrid;
    nv[0].aux0 = 0;  // object policy
    nv[1].kind = kGrid;
    nv[1].aux0 = 1;  // container policy
    const NodesF nodes{nv.data()};

    int outside = 0;
    for (int i = 0; i < 5000; ++i) {
      // range deliberately overshoots the stored box on all sides so the residual branch fires
      const Vec3<float> p{rng.u(-2.5f, 2.0f), rng.u(-2.5f, 2.0f), rng.u(-2.5f, 2.0f)};
      const float gotO = evalTree<float>(nodes, 2, 0, p, grids, pl);
      const float gotC = evalTree<float>(nodes, 2, 1, p, grids, pl);
      PECLET_CORE_CHECK(bitEqualF(gotO, ref_sample(p.x, p.y, p.z, rg, pool.data(), true)));
      PECLET_CORE_CHECK(bitEqualF(gotC, ref_sample(p.x, p.y, p.z, rg, pool.data(), false)));
      if (gotO != gotC) {
        ++outside;
        // object extension must read FARTHER (more positive) than container everywhere the
        // residual is non-zero -- this is the sign asymmetry that keeps a grain from escaping.
        PECLET_CORE_CHECK(gotO > gotC);
      }
    }
    PECLET_CORE_CHECK(outside > 500);  // the off-box branch really was exercised
    std::printf("  grid: %d of 5000 probes outside the stored box\n", outside);
  }

  // ---------------------------------------------------------------------------------------
  // (5) Malformed trees degrade to "infinitely far", never corrupt memory.
  // ---------------------------------------------------------------------------------------
  {
    std::vector<ShapeNode<double>> nv(2);
    nv[0].kind = kUnion;
    nv[0].aux0 = 1;
    nv[0].aux1 = 99;  // out of range
    nv[1].kind = kSphere;
    nv[1].params[0] = 1.0;
    const TablePtr<ShapeNode<double>> nodes{nv.data()};
    const TablePtr<GridDesc<double>> g{nullptr};
    const PoolPtr<double> pl{nullptr};
    const Vec3<double> p{0.1, 0.2, 0.3};
    PECLET_CORE_CHECK(evalTree<double>(nodes, 2, 0, p, g, pl) == 1e9);   // bad child
    PECLET_CORE_CHECK(evalTree<double>(nodes, 2, -1, p, g, pl) == 1e9);  // bad root
    PECLET_CORE_CHECK(evalTree<double>(nodes, 2, 7, p, g, pl) == 1e9);   // root past the end

    // a chain deeper than kMaxTreeDepth must bail rather than overrun the frame stack
    std::vector<ShapeNode<double>> deep(kMaxTreeDepth + 4);
    for (int i = 0; i < kMaxTreeDepth + 3; ++i) {
      deep[i].kind = kUnion;
      deep[i].aux0 = i + 1;
      deep[i].aux1 = i + 1;
    }
    deep[kMaxTreeDepth + 3].kind = kSphere;
    deep[kMaxTreeDepth + 3].params[0] = 1.0;
    const TablePtr<ShapeNode<double>> dn{deep.data()};
    PECLET_CORE_CHECK(evalTree<double>(dn, kMaxTreeDepth + 4, 0, p, g, pl) == 1e9);
  }

  // dem's ShapeKind numbering is preserved so its Python shape_type integers survive rung 3.
  static_assert(kGrid == 0 && kSphere == 1 && kHollowCylinder == 2 && kBox == 3,
                "core ShapeKind must keep dem's SHAPE_GRID_SDF/SPHERE/HOLLOW_CYLINDER/BOX values");

  PECLET_CORE_RETURN_TEST_RESULT();
}

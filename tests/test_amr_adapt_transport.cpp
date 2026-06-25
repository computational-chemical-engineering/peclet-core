// End-to-end adaptive transport: the dynamic-AMR infrastructure (indicators + adapt +
// transferField) driving a real time-dependent solver (ScalarTransport). A Gaussian
// blob is advected in a divergence-free (uniform) periodic velocity while the mesh is
// periodically re-adapted to track it. Validates that the whole pipeline composes:
//   (1) total mass Σ V·c is conserved across the entire run — through every transport
//       step AND every adapt/remap (the key infrastructure guarantee);
//   (2) the mesh tracks the moving blob (the finest cells stay near its centre);
//   (3) the adaptive mesh stays far smaller than a uniform-fine grid.
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <array>
#include <cmath>
#include <vector>

#include "tpx/amr/adapt.hpp"
#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/amr/scalar_transport.hpp"
#include "tpx/common/types.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using M = BO::M;
constexpr double kN = 32.0;  // 32^3 fine over [0,1)^3 (brick 4^3, lmax 3)

std::array<double, 3> centroidW(const BO& t, Index i) {
  auto o = M::from_code(t.code(i)).decode();
  double s = static_cast<double>(Index(1) << t.level(i));
  return {((double)o[0] + 0.5 * s) / kN, ((double)o[1] + 0.5 * s) / kN, ((double)o[2] + 0.5 * s) / kN};
}

// Periodic Gaussian blob centred at cx (in x; y,z centred at 0.5), width σ.
double blob(const std::array<double, 3>& p, double cx) {
  auto per = [](double d) {  // shortest periodic distance on [0,1)
    d -= std::floor(d + 0.5);
    return d;
  };
  const double sig = 0.07;
  double dx = per(p[0] - cx), dy = per(p[1] - 0.5), dz = per(p[2] - 0.5);
  return std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sig * sig));
}
std::vector<double> sampleBlob(const BO& t, double cx) {
  std::vector<double> c((std::size_t)t.numLeaves());
  for (Index i = 0; i < t.numLeaves(); ++i) c[(std::size_t)i] = blob(centroidW(t, i), cx);
  return c;
}
double cellVolRel(const BO& t, Index i) {
  double s = static_cast<double>(Index(1) << t.level(i));
  return s * s * s;
}
double mass(const BO& t, const std::vector<double>& c) {
  double s = 0.0;
  for (Index i = 0; i < t.numLeaves(); ++i) s += cellVolRel(t, i) * c[(std::size_t)i];
  return s;
}
BO baseMesh() {
  BO t(IVec<3>{4, 4, 4}, 3);
  t.refineIf([](Code, unsigned l) { return l > 1; });  // uniform base level 1 (16^3)
  return t;
}

void run() {
  AmrGeometry<3> geo;
  geo.h0 = 1.0 / kN;

  const double U = 1.0;                 // x-velocity (world/time), divergence-free
  auto vel = [&](const Vec<3>&, int axis) { return axis == 0 ? U : 0.0; };
  const double hFine = geo.h0;          // finest spacing
  const double dt = 0.4 * hFine / U;    // CFL
  const int steps = 16, adaptEvery = 4;
  const double cx0 = 0.3;

  // Initial mesh: refine the base around the initial blob.
  BO t = baseMesh();
  std::vector<double> c = sampleBlob(t, cx0);
  for (int s = 0; s < 3; ++s) {
    auto r = adapt(t, c, /*refineThresh=*/0.2, /*coarsenThresh=*/0.03, /*finestLevel=*/0);
    t = std::move(r.octree);
    c = sampleBlob(t, cx0);  // re-sample exact on the refined initial mesh
  }
  const double m0 = mass(t, c);
  double massMaxDev = 0.0;
  std::size_t maxLeaves = (std::size_t)t.numLeaves();

  for (int step = 0; step < steps; ++step) {
    ScalarTransport<3, kBits> st(t, geo);
    std::vector<double> cn;
    st.step(c, cn, dt, /*D=*/0.0, vel);  // pure advection (upwind)
    c.swap(cn);
    massMaxDev = std::max(massMaxDev, std::fabs(mass(t, c) - m0));

    if ((step + 1) % adaptEvery == 0) {
      auto r = adapt(t, c, 0.2, 0.03, /*finestLevel=*/0);  // conservative remap
      t = std::move(r.octree);
      c = std::move(r.field);
      massMaxDev = std::max(massMaxDev, std::fabs(mass(t, c) - m0));
      maxLeaves = std::max(maxLeaves, (std::size_t)t.numLeaves());
    }
  }

  // (1) mass conserved through all steps + adapts
  TPX_CHECK(massMaxDev < 1e-9 * std::fabs(m0));

  // (2) the blob advected to the right place: mass-weighted x-centroid ≈ cx0 + U*T
  // (blob stays interior, no wrap), and the finest cells are present near it.
  const double T = steps * dt;
  const double cxEnd = cx0 + U * T;
  double sw = 0.0, sx = 0.0;
  Index nFinest = 0;
  double fineCx = 0.0, fineW = 0.0;
  for (Index i = 0; i < t.numLeaves(); ++i) {
    double w = cellVolRel(t, i) * c[(std::size_t)i];
    double x = centroidW(t, i)[0];
    sw += w;
    sx += w * x;
    if (t.level(i) == 0) {
      ++nFinest;
      fineW += 1.0;
      fineCx += x;
    }
  }
  const double xc = sx / sw;            // scalar mass-weighted x-centroid
  const double fc = fineCx / fineW;     // mean x of the finest cells
  TPX_CHECK(std::fabs(xc - cxEnd) < 0.05);  // advection moved the blob correctly
  TPX_CHECK(nFinest > 0);
  TPX_CHECK(std::fabs(fc - cxEnd) < 0.1);   // finest cells follow the blob

  // (3) adaptive mesh stays smaller than uniform-fine (32^3 = 32768)
  TPX_CHECK(maxLeaves < 32768);
}

}  // namespace

int main() {
  run();
  TPX_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping adaptive transport test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

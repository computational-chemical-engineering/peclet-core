// peclet::core::Box / UniformGrid — the physical domain (peclet/core/domain.hpp).
//
// Checks:
//  - cellUnits(cells) is exactly the historical cell-unit grid: spacing 1.0 bitwise, origin 0,
//    cell centre (i+1/2) — the guarantee the solvers' extent=None path relies on;
//  - spacing = extent/cells per axis, including anisotropic;
//  - cellCentre / faceCentre / nodePosition on the MAC convention;
//  - indexOf ∘ physicalOf and physicalOf ∘ indexOf are the identity to round-off;
//  - cellOf lands on the right cell at the centres and at the lower face;
//  - periodic wrap and minimum image on a shifted, anisotropic box;
//  - isIsotropic() / minSpacing() / cellVolume() / numCells().
#include <cmath>

#include "peclet/core/domain.hpp"
#include "test_util.hpp"

using namespace peclet::core;

static void checkClose(Real a, Real b, Real tol, const char* what) {
  if (!(std::fabs(a - b) <= tol)) {
    std::fprintf(stderr, "CHECK_CLOSE failed (%s): %.17g vs %.17g\n", what, a, b);
    ++::peclet::core::test::g_failures;
  }
}

int main() {
  // --- 1. cell units are exact ------------------------------------------------------------
  {
    const auto g = UniformGrid<3>::cellUnits(IVec<3>{16, 32, 8});
    const auto h = g.spacing();
    // Bitwise 1.0: the extent=None path of every solver depends on this.
    PECLET_CORE_CHECK(h[0] == 1.0 && h[1] == 1.0 && h[2] == 1.0);
    PECLET_CORE_CHECK(g.minSpacing() == 1.0);
    PECLET_CORE_CHECK(g.isIsotropic());
    PECLET_CORE_CHECK(g.cellVolume() == 1.0);
    PECLET_CORE_CHECK_EQ(g.numCells(), Index(16 * 32 * 8));
    const auto c = g.cellCentre(IVec<3>{3, 4, 5});
    PECLET_CORE_CHECK(c[0] == 3.5 && c[1] == 4.5 && c[2] == 5.5);
    const auto o = g.origin();
    PECLET_CORE_CHECK(o[0] == 0.0 && o[1] == 0.0 && o[2] == 0.0);
    const auto L = g.length();
    PECLET_CORE_CHECK(L[0] == 16.0 && L[1] == 32.0 && L[2] == 8.0);
    PECLET_CORE_CHECK_EQ(g.resolution()[1], Index(32));
    PECLET_CORE_CHECK_EQ(UniformGrid<3>::dim(), 3);
  }

  // --- 2. anisotropic spacing, shifted origin ----------------------------------------------
  Box<3> box;
  box.origin = Vec<3>{-0.25, 1.0, 3.5};
  box.length = Vec<3>{2.0, 0.6, 8.0};
  box.periodic = {true, false, true};
  const UniformGrid<3> g(IVec<3>{20, 12, 40}, box);
  {
    const auto h = g.spacing();
    checkClose(h[0], 0.1, 1e-15, "hx");
    checkClose(h[1], 0.05, 1e-15, "hy");
    checkClose(h[2], 0.2, 1e-15, "hz");
    checkClose(g.minSpacing(), 0.05, 1e-15, "minSpacing");
    PECLET_CORE_CHECK(!g.isIsotropic());
    checkClose(g.cellVolume(), 0.1 * 0.05 * 0.2, 1e-17, "cellVolume");
    checkClose(g.box.volume(), 2.0 * 0.6 * 8.0, 1e-14, "boxVolume");
    const auto u = g.box.upper();
    checkClose(u[0], 1.75, 1e-15, "upper x");
    checkClose(u[1], 1.6, 1e-15, "upper y");
    checkClose(u[2], 11.5, 1e-14, "upper z");
    const auto ctr = g.box.centre();
    checkClose(ctr[0], 0.75, 1e-15, "centre x");
  }

  // --- 3. cell / face / node positions ------------------------------------------------------
  {
    const IVec<3> idx{7, 3, 11};
    const auto h = g.spacing();
    const auto c = g.cellCentre(idx);
    checkClose(c[0], -0.25 + 7.5 * h[0], 1e-15, "cellCentre x");
    checkClose(c[1], 1.0 + 3.5 * h[1], 1e-15, "cellCentre y");
    checkClose(c[2], 3.5 + 11.5 * h[2], 1e-14, "cellCentre z");

    const auto n = g.nodePosition(idx);
    checkClose(n[0], -0.25 + 7.0 * h[0], 1e-15, "node x");

    // x-normal face: on the node in x, cell-centred in y and z.
    const auto fx = g.faceCentre(0, idx);
    checkClose(fx[0], n[0], 1e-16, "faceCentre x (x)");
    checkClose(fx[1], c[1], 1e-16, "faceCentre x (y)");
    checkClose(fx[2], c[2], 1e-16, "faceCentre x (z)");
    // z-normal face: cell-centred in x and y, on the node in z.
    const auto fz = g.faceCentre(2, idx);
    checkClose(fz[0], c[0], 1e-16, "faceCentre z (x)");
    checkClose(fz[2], n[2], 1e-16, "faceCentre z (z)");
  }

  // --- 4. index <-> physical round trips ----------------------------------------------------
  {
    const Vec<3> xi{7.25, 3.5, 11.125};
    const auto p = g.physicalOf(xi);
    const auto back = g.indexOf(p);
    for (int a = 0; a < 3; ++a)
      checkClose(back[a], xi[a], 1e-12, "index round trip");

    const Vec<3> pt{0.375, 1.4375, 9.75};
    const auto xi2 = g.indexOf(pt);
    const auto p2 = g.physicalOf(xi2);
    for (int a = 0; a < 3; ++a)
      checkClose(p2[a], pt[a], 1e-14, "physical round trip");

    // cellOf at the centre and at the lower face of a known cell.
    const IVec<3> idx{7, 3, 11};
    const auto ci = g.cellOf(g.cellCentre(idx));
    PECLET_CORE_CHECK(ci[0] == idx[0] && ci[1] == idx[1] && ci[2] == idx[2]);
    // A quarter of a cell above the lower node is unambiguously inside the same cell. (Exactly ON
    // the node is a round-off coin flip -- (origin + i*h - origin)/h need not be exactly i in
    // floating point -- so cellOf is only asked about points strictly inside.)
    const auto h = g.spacing();
    const auto nd = g.nodePosition(idx);
    const auto ni = g.cellOf(Vec<3>{nd[0] + 0.25 * h[0], nd[1] + 0.25 * h[1], nd[2] + 0.25 * h[2]});
    PECLET_CORE_CHECK(ni[0] == idx[0] && ni[1] == idx[1] && ni[2] == idx[2]);

    PECLET_CORE_CHECK(g.contains(g.cellCentre(IVec<3>{0, 0, 0})));
    PECLET_CORE_CHECK(g.contains(g.cellCentre(IVec<3>{19, 11, 39})));
    PECLET_CORE_CHECK(!g.contains(g.box.upper()));
    PECLET_CORE_CHECK(!g.contains(Vec<3>{-0.3, 1.1, 4.0}));
  }

  // --- 5. periodic wrap and minimum image ---------------------------------------------------
  {
    // x is periodic with length 2 starting at -0.25: 1.9 folds to -0.1.
    const auto w = g.wrap(Vec<3>{1.9, 1.3, 3.4});
    checkClose(w[0], -0.1, 1e-14, "wrap x");
    checkClose(w[1], 1.3, 1e-16, "wrap y (non-periodic, untouched)");
    // z is periodic with length 8 starting at 3.5: 3.4 folds to 11.4.
    checkClose(w[2], 11.4, 1e-13, "wrap z");
    // Already-inside points are unchanged to round-off.
    const Vec<3> in{0.5, 1.3, 7.0};
    const auto w2 = g.wrap(in);
    for (int a = 0; a < 3; ++a)
      checkClose(w2[a], in[a], 1e-15, "wrap identity inside");

    // Minimum image: separation 1.8 across a periodic length-2 axis is -0.2.
    const auto d = g.minImage(Vec<3>{-0.2, 1.05, 4.0}, Vec<3>{1.6, 1.55, 4.0});
    checkClose(d[0], -0.2, 1e-14, "minImage x");
    checkClose(d[1], 0.5, 1e-15, "minImage y (non-periodic)");
    checkClose(d[2], 0.0, 1e-16, "minImage z");
    // A separation across the periodic z axis (length 8): 7.0 -> -1.0.
    const auto d2 = g.minImage(Vec<3>{0.0, 1.0, 4.0}, Vec<3>{0.0, 1.0, 11.0});
    checkClose(d2[2], -1.0, 1e-14, "minImage z wrapped");
  }

  // --- 6. Box periodicity flags survive cellUnits -------------------------------------------
  {
    const auto g2 = UniformGrid<3>::cellUnits(IVec<3>{4, 4, 4}, {true, false, true});
    PECLET_CORE_CHECK(g2.periodic(0) && !g2.periodic(1) && g2.periodic(2));
  }

  PECLET_CORE_RETURN_TEST_RESULT();
}

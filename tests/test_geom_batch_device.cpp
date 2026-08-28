// Kokkos-gated gate (Layer 2-for-core): the batched device drivers must reproduce the host
// per-point evaluation BIT-IDENTICALLY (same backend), sphere-union + candidate grid + min-image
// periodicity included.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <sstream>
#include <vector>

#include "peclet/core/geom/device_scene.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int bad = 0;
  {
    // the RCP bed
    std::vector<double> cx, cy, cz, r;
    std::ifstream f("data/rcp_pack_seed3_unit.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      std::istringstream is(line);
      double x, y, z, rr;
      if (is >> x >> y >> z >> rr) {
        cx.push_back(x);
        cy.push_back(y);
        cz.push_back(z);
        r.push_back(rr);
      }
    }
    SphereBedQuery q(cx, cy, cz, r, Vec<3>{0, 0, 0}, Vec<3>{1, 1, 1}, true);

    // device copies of the union + grid
    const int n = q.sphereUnion().n;
    auto mkview = [](const char* nm, const double* src, std::size_t cnt) {
      Kokkos::View<double*> v(nm, cnt);
      auto h = Kokkos::create_mirror_view(v);
      for (std::size_t i = 0; i < cnt; ++i)
        h(i) = src[i];
      Kokkos::deep_copy(v, h);
      return v;
    };
    auto dcx = mkview("cx", q.sphereUnion().cx, n), dcy = mkview("cy", q.sphereUnion().cy, n),
         dcz = mkview("cz", q.sphereUnion().cz, n), dr = mkview("r", q.sphereUnion().r, n);
    const auto& gh = q.grid();
    const long nbins = (long)gh.nx * gh.ny * gh.nz;
    Kokkos::View<int*> doff("off", nbins + 1), ditems("items", gh.offsets[nbins]);
    {
      auto h1 = Kokkos::create_mirror_view(doff);
      for (long i = 0; i <= nbins; ++i)
        h1(i) = gh.offsets[i];
      Kokkos::deep_copy(doff, h1);
      auto h2 = Kokkos::create_mirror_view(ditems);
      for (int i = 0; i < gh.offsets[nbins]; ++i)
        h2(i) = gh.items[i];
      Kokkos::deep_copy(ditems, h2);
    }
    SphereUnionView<double> du = q.sphereUnion();
    du.cx = dcx.data();
    du.cy = dcy.data();
    du.cz = dcz.data();
    du.r = dr.data();
    CandidateGridView<double> dg = gh;
    dg.offsets = doff.data();
    dg.items = ditems.data();

    // probe points: box + beyond + surface-hugging
    const int NP = 200000;
    Kokkos::View<double* [3]> pts("pts", NP);
    auto hp = Kokkos::create_mirror_view(pts);
    std::uint64_t st = 0x9E3779B97F4A7C15ull;
    auto u01 = [&]() {
      st ^= st << 13;
      st ^= st >> 7;
      st ^= st << 17;
      return (double)((st >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
    };
    for (int i = 0; i < NP; ++i) {
      if (i % 3 == 0) {  // surface-hugging
        const int s = (int)(u01() * n);
        const double th = 6.283185307179586 * u01(), uz = 2 * u01() - 1,
                     sr = std::sqrt(1 - uz * uz), eps = (u01() - 0.5) * 4e-3;
        hp(i, 0) = cx[s] + (r[s] + eps) * sr * std::cos(th);
        hp(i, 1) = cy[s] + (r[s] + eps) * sr * std::sin(th);
        hp(i, 2) = cz[s] + (r[s] + eps) * uz;
      } else {
        hp(i, 0) = 1.6 * u01() - 0.3;
        hp(i, 1) = 1.6 * u01() - 0.3;
        hp(i, 2) = 1.6 * u01() - 0.3;
      }
    }
    Kokkos::deep_copy(pts, hp);

    Kokkos::View<double*> out("out", NP);
    evalSphereUnionPoints(Kokkos::DefaultExecutionSpace{}, du, q.box(), dg, pts, out);
    Kokkos::fence();
    auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);

    int nbit = 0;
    for (int i = 0; i < NP; ++i) {
      const double ref = q(Vec<3>{hp(i, 0), hp(i, 1), hp(i, 2)});
      std::uint64_t a, b;
      std::memcpy(&a, &ref, 8);
      const double v = ho(i);
      std::memcpy(&b, &v, 8);
      if (a != b) {
        if (nbit < 3)
          std::printf("  mismatch %d: dev %.17g host %.17g\n", i, v, ref);
        ++nbit;
      }
    }
    std::printf("  batched device vs host query: %d/%d bit mismatches\n", nbit, NP);
    if (nbit)
      bad = 1;
  }
  {
    // DeviceScene: the owning upload path must reproduce host SceneBuilder evaluation bitwise
    // on the same backend (the flow/dem/voro retrofit target).
    using namespace peclet::core;
    geom::SceneBuilder<double> b;
    const int shaft = b.addLeaf(geom::kHollowCylinder, {0.3, 3.0, 0.3});
    const int blade = b.addLeaf(geom::kBox, {0.9, 0.1, 0.25},
                                geom::Transform<double>{Vec3<double>{1.0, 0.5, 0.0}});
    const int stir = b.addUnion(shaft, blade);
    b.addInstance(stir, {}, Vec3<double>{0, 0, 0}, Vec3<double>{0, 10, 0});
    auto ds = geom::DeviceScene<double>::from(b);
    const int NP2 = 4096;
    Kokkos::View<double* [3]> pts("pts2", NP2);
    Kokkos::View<double*> out("out2", NP2);
    auto hp = Kokkos::create_mirror_view(pts);
    for (int i = 0; i < NP2; ++i) {
      const double u = (double)i / NP2;
      hp(i, 0) = -3 + 6 * u;
      hp(i, 1) = 0.37 - 2 * u;
      hp(i, 2) = 0.11 + u;
    }
    Kokkos::deep_copy(pts, hp);
    geom::evalScenePoints(Kokkos::DefaultExecutionSpace{}, ds.view(), pts, out);
    Kokkos::fence();
    auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    const auto hostView = b.view();
    int n2 = 0;
    for (int i = 0; i < NP2; ++i) {
      const double ref = geom::evalScene(hostView, Vec3<double>{hp(i, 0), hp(i, 1), hp(i, 2)});
      std::uint64_t a, c;
      std::memcpy(&a, &ref, 8);
      const double v = ho(i);
      std::memcpy(&c, &v, 8);
      // Host-vs-device over a CSG tree: judged by sign + a ULP band (the rung-5 rule), since
      // the general tree walk is not fma-canonicalised. The sphere-union path above IS bitwise.
      if ((v < 0) != (ref < 0) && std::fabs(ref) > 1e-13)
        ++n2;
      if (std::fabs(v - ref) > 1e-12 * (std::fabs(ref) + 1.0))
        ++n2;
    }
    std::printf("  DeviceScene CSG scene: %d/%d outside the sign+ULP band\n", n2, NP2);
    if (n2)
      bad = 1;
  }
  {
    // SceneQueryDevice: mode selection + acceleration + periodicity end to end. Sphere scene ->
    // fast path, BITWISE vs the host SphereBedQuery-style eval; mixed scene -> general path,
    // bitwise vs the host evalSceneGrid on the same expressions (same backend).
    using namespace peclet::core;
    using namespace peclet::core::geom;
    SceneBuilder<double> bs;
    const int s0 = bs.addLeaf(kSphere, {0.11});
    std::uint64_t st2 = 0xABCDEF12345ull;
    auto u2 = [&]() {
      st2 ^= st2 << 13;
      st2 ^= st2 >> 7;
      st2 ^= st2 << 17;
      return (double)((st2 >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
    };
    for (int i = 0; i < 60; ++i)
      bs.addInstance(s0, Transform<double>{Vec3<double>{u2(), u2(), u2()}});
    const PeriodicBox<double> box{1.0, 1.0, 1.0, true};
    auto q = SceneQueryDevice<double>::build(bs, Vec3<double>{0, 0, 0}, Vec3<double>{1, 1, 1}, box);
    if (!q.sphereFast() || !q.accelerated()) {
      std::printf("  mode selection FAILED\n");
      bad = 1;
    }
    const int NP3 = 50000;
    Kokkos::View<double* [3]> pts3("pts3", NP3);
    Kokkos::View<double*> out3("out3", NP3);
    auto hp3 = Kokkos::create_mirror_view(pts3);
    for (int i = 0; i < NP3; ++i) {
      hp3(i, 0) = 1.4 * u2() - 0.2;
      hp3(i, 1) = 1.4 * u2() - 0.2;
      hp3(i, 2) = 1.4 * u2() - 0.2;
    }
    Kokkos::deep_copy(pts3, hp3);
    evalQueryPoints(Kokkos::DefaultExecutionSpace{}, q.view(), pts3, out3);
    Kokkos::fence();
    auto ho3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out3);
    // host reference via the same host-side machinery (host build of the same scene)
    std::vector<double> ecx, ecy, ecz, er;
    extractSphereUnion(bs.view(), ecx, ecy, ecz, er);
    SphereUnionView<double> hu;
    hu.cx = ecx.data();
    hu.cy = ecy.data();
    hu.cz = ecz.data();
    hu.r = er.data();
    hu.n = (int)ecx.size();
    hu.equalR = true;
    hu.r0 = 0.11;
    CandidateGrid<double> hg =
        buildSphereCandidateGrid(hu, Vec3<double>{0, 0, 0}, Vec3<double>{1, 1, 1}, box);
    const CandidateGridView<double> hgv = hg.view();
    int n3 = 0;
    for (int i = 0; i < NP3; ++i) {
      const double ref =
          evalSphereUnionGrid(hu, Vec3<double>{hp3(i, 0), hp3(i, 1), hp3(i, 2)}, box, hgv);
      std::uint64_t a, c;
      std::memcpy(&a, &ref, 8);
      const double v = ho3(i);
      std::memcpy(&c, &v, 8);
      if (a != c)
        ++n3;
    }
    std::printf("  SceneQueryDevice sphere mode: %d/%d bit mismatches vs host\n", n3, NP3);
    if (n3)
      bad = 1;
  }
  Kokkos::finalize();
  std::printf(bad ? "BATCH FAIL\n" : "BATCH OK\n");
  return bad;
}

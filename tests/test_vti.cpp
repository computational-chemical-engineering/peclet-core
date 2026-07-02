// Round-trip a sampled SDF through VTI (.vti) I/O: sample an analytic sphere onto a grid, write,
// read back, and require identical dims/origin/spacing and bit-exact (float) values.
#include <cmath>
#include <cstdio>
#include <string>

#include "peclet/core/common/types.hpp"
#include "peclet/core/geom/grid_sdf.hpp"
#include "peclet/core/geom/sdf.hpp"
#include "peclet/core/geom/vti_io.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::core::geom;

int main() {
  Sphere s{{0.1, -0.2, 0.3}, 1.5};
  IVec<3> dims{13, 11, 9};
  GridSdf g = sample(s, dims, {-2.0, -2.0, -2.0}, {0.3, 0.35, 0.4});

  std::string path = "test_vti_roundtrip.vti";
  writeVti(path, g, "sdf");
  GridSdf r = readVti(path);
  std::remove(path.c_str());

  PECLET_CORE_CHECK_EQ(r.dims[0], g.dims[0]);
  PECLET_CORE_CHECK_EQ(r.dims[1], g.dims[1]);
  PECLET_CORE_CHECK_EQ(r.dims[2], g.dims[2]);
  for (int i = 0; i < 3; ++i) {
    PECLET_CORE_CHECK(std::fabs(r.origin[i] - g.origin[i]) < 1e-6);
    PECLET_CORE_CHECK(std::fabs(r.spacing[i] - g.spacing[i]) < 1e-6);
  }
  PECLET_CORE_CHECK_EQ(static_cast<Index>(r.values.size()), static_cast<Index>(g.values.size()));
  int mism = 0;
  for (std::size_t i = 0; i < g.values.size(); ++i) {
    if (r.values[i] != g.values[i])
      ++mism;  // 9-digit ASCII round-trips float32 exactly
  }
  PECLET_CORE_CHECK_EQ(mism, 0);

  // Read-back field still evaluates as a sane SDF (sign agreement with analytic, away from
  // surface).
  PECLET_CORE_CHECK(r.eval({0.1, -0.2, 0.3}) < 0);  // center -> inside
  PECLET_CORE_CHECK(r.eval({1.9, -0.2, 0.3}) > 0);  // well outside

  // --- Vector VTI round-trip ---
  VtiVector vec;
  vec.dims = {5, 4, 3};
  vec.origin = {0, 0, 0};
  vec.spacing = {1, 1, 1};
  std::size_t np = 5 * 4 * 3;
  vec.values.resize(3 * np);
  for (std::size_t p = 0; p < np; ++p) {
    vec.values[3 * p] = 0.1f * p;
    vec.values[3 * p + 1] = -0.2f * p;
    vec.values[3 * p + 2] = 0.3f * p;
  }
  std::string vpath = "test_vti_vec.vti";
  writeVtiVector(vpath, vec, "vel");
  VtiVector vr = readVtiVector(vpath);
  std::remove(vpath.c_str());
  PECLET_CORE_CHECK_EQ(vr.dims[0], vec.dims[0]);
  PECLET_CORE_CHECK_EQ(vr.dims[1], vec.dims[1]);
  PECLET_CORE_CHECK_EQ(vr.dims[2], vec.dims[2]);
  int vmism = 0;
  for (std::size_t i = 0; i < vec.values.size(); ++i)
    if (vr.values[i] != vec.values[i])
      ++vmism;
  PECLET_CORE_CHECK_EQ(vmism, 0);

  PECLET_CORE_RETURN_TEST_RESULT();
}

// Synthetic fixed-point maps on padded device boxes for the AndersonCore tests (test_anderson.cpp,
// test_anderson_mpi.cpp; flow doc/steady_acceleration.md §8, U1–U7).
#ifndef PECLET_CORE_TEST_ANDERSON_TEST_MAPS_HPP
#define PECLET_CORE_TEST_ANDERSON_TEST_MAPS_HPP

#include <cmath>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/anderson.hpp"

namespace peclet::core::test {

/// A padded box: inner n[0]×n[1]×n[2], ghost width g, x fastest.
struct Box {
  IVec<3> n{};
  int g = 2;
  IVec<3> e{};
  Index nPad = 0;
  Index nInner = 0;
  Box(Index nx, Index ny, Index nz, int ghost) : n{nx, ny, nz}, g(ghost) {
    e = {nx + 2 * ghost, ny + 2 * ghost, nz + 2 * ghost};
    nPad = e[0] * e[1] * e[2];
    nInner = nx * ny * nz;
  }
  Index pad(Index x, Index y, Index z) const {
    return (x + g) + (y + g) * e[0] + (z + g) * e[0] * e[1];
  }
  bool inner(Index i) const {
    const Index x = i % e[0], y = (i / e[0]) % e[1], z = i / (e[0] * e[1]);
    return x >= g && x < e[0] - g && y >= g && y < e[1] - g && z >= g && z < e[2] - g;
  }
};

/// x ← Λx + c on every padded entry (Λ diagonal), then an optional general 2×2 block B on entries
/// (i0, i1): (g_i0, g_i1) += B·(x_i0, x_i1) (their diagonal entries are 0). A rotation of modulus
/// ρ is B = [[ρ cos θ, −ρ sin θ], [ρ sin θ, ρ cos θ]] (U4); a Jordan-type block [[λ, κ], [0, λ]] is
/// non-normal (U4b).
struct LinearMap {
  View<double> lam, c, tmp;
  Index i0 = -1, i1 = -1;
  double b00 = 0.0, b01 = 0.0, b10 = 0.0, b11 = 0.0;
};

inline LinearMap makeLinearMap(const std::vector<double>& lam, const std::vector<double>& c) {
  LinearMap m;
  m.lam = toDevice(lam, "t_lam");
  m.c = toDevice(c, "t_c");
  m.tmp = View<double>("t_tmp", lam.size());
  return m;
}

inline void applyLinear(const LinearMap& m, View<double> buf) {
  View<double> tmp = m.tmp;
  View<const double> lam = m.lam, c = m.c;
  Kokkos::deep_copy(tmp, buf);
  Kokkos::parallel_for(
      "t_linear", buf.extent(0),
      KOKKOS_LAMBDA(const std::size_t i) { buf(i) = lam(i) * tmp(i) + c(i); });
  if (m.i0 >= 0) {
    const Index i0 = m.i0, i1 = m.i1;
    const double b00 = m.b00, b01 = m.b01, b10 = m.b10, b11 = m.b11;
    Kokkos::parallel_for(
        "t_block", 1, KOKKOS_LAMBDA(const int) {
          const double a = tmp(i0), b = tmp(i1);
          buf(i0) += b00 * a + b01 * b;
          buf(i1) += b10 * a + b11 * b;
        });
  }
}

/// U1 spectrum: inner entry k of nInner (x-fastest inner order) gets 0.996·k/(nInner−1); ghosts 0.
inline std::vector<double> u1Spectrum(const Box& b) {
  std::vector<double> lam(static_cast<std::size_t>(b.nPad), 0.0);
  Index k = 0;
  for (Index z = 0; z < b.n[2]; ++z)
    for (Index y = 0; y < b.n[1]; ++y)
      for (Index x = 0; x < b.n[0]; ++x, ++k)
        lam[static_cast<std::size_t>(b.pad(x, y, z))] =
            0.996 * static_cast<double>(k) / static_cast<double>(b.nInner - 1);
  return lam;
}

inline std::vector<double> download(View<const double> v) {
  std::vector<double> h(v.extent(0));
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(h.data(),
                                                                                       h.size());
  Kokkos::deep_copy(hv, v);
  return h;
}

}  // namespace peclet::core::test

#endif  // PECLET_CORE_TEST_ANDERSON_TEST_MAPS_HPP

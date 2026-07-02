// Portable device-resident block-octree queries (peclet::core::amr::BlockOctreeView):
// the device point-location and face-neighbour walk must match the host
// BlockOctree bit-for-bit, on whatever backend Kokkos was built for (CUDA / HIP /
// OpenMP). Compiled as CXX — Kokkos routes it to the device compiler.
//
// Guarded by PECLET_CORE_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef PECLET_CORE_HAVE_MORTON
#include <array>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/block_octree_view.hpp"

using namespace peclet::core;
using peclet::core::amr::BlockOctree;
using peclet::core::amr::BlockOctreeView;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using Coord = BO::Coord;

void run() {
  // A non-trivial, balanced mesh.
  BO t(IVec<3>{4, 4, 4}, 2);
  auto refineAt = [&](std::array<Coord, 3> p) {
    Index leaf = t.find(p);
    if (leaf >= 0 && t.level(leaf) > 0) {
      Code target = t.code(leaf);
      t.refineIf([&](Code c, unsigned) { return c == target; });
    }
  };
  refineAt({0, 0, 0});
  refineAt({3, 0, 0});
  t.balance2to1();
  PECLET_CORE_CHECK(t.isBalanced());

  BlockOctreeView<3, kBits> dev;
  dev.upload(t);
  PECLET_CORE_CHECK_EQ((long long)dev.numLeaves(), (long long)t.numLeaves());

  // ---- point location: device locate(probe) == host find(probe) ----
  std::vector<Code> probes;
  std::uint64_t st = 7;
  const Coord span = 16;  // brick(4) * 2^lmax(4) = 16 fine units per axis
  for (int i = 0; i < 4096; ++i) {
    std::array<Coord, 3> p{};
    for (int d = 0; d < 3; ++d) {
      st = st * 6364136223846793005ULL + 1442695040888963407ULL;
      p[d] = static_cast<Coord>((st >> 33) % span);
    }
    probes.push_back(BO::M::encode(p).code());
  }
  View<Code> dProbe = toDevice(probes, "probes");
  View<Index> dOut("locate_out", probes.size());
  Kokkos::parallel_for(
      "amr_locate", probes.size(), KOKKOS_LAMBDA(const int i) { dOut(i) = dev.locate(dProbe(i)); });
  auto hOut = Kokkos::create_mirror_view(dOut);
  Kokkos::deep_copy(hOut, dOut);
  int mism = 0;
  for (std::size_t i = 0; i < probes.size(); ++i)
    if (hOut(i) != t.find(probes[i]))
      ++mism;
  PECLET_CORE_CHECK_EQ(mism, 0);

  // ---- face neighbours: device == host for every leaf, every face ----
  const Index nleaf = t.numLeaves();
  View<Index> dNbr("nbr_out", static_cast<std::size_t>(nleaf) * 6);
  Kokkos::parallel_for(
      "amr_nbr", nleaf, KOKKOS_LAMBDA(const Index i) {
        for (int axis = 0; axis < 3; ++axis) {
          dNbr(i * 6 + axis * 2 + 0) = dev.faceNeighbor(i, axis, +1);
          dNbr(i * 6 + axis * 2 + 1) = dev.faceNeighbor(i, axis, -1);
        }
      });
  auto hNbr = Kokkos::create_mirror_view(dNbr);
  Kokkos::deep_copy(hNbr, dNbr);
  int nmis = 0;
  for (Index i = 0; i < nleaf; ++i)
    for (int axis = 0; axis < 3; ++axis) {
      if (hNbr(i * 6 + axis * 2 + 0) != t.faceNeighbor(i, axis, +1))
        ++nmis;
      if (hNbr(i * 6 + axis * 2 + 1) != t.faceNeighbor(i, axis, -1))
        ++nmis;
    }
  PECLET_CORE_CHECK_EQ(nmis, 0);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  PECLET_CORE_RETURN_TEST_RESULT();
}
#else
int main() {
  std::printf("PECLET_CORE_HAVE_MORTON not set — skipping device block octree test\n");
  return 0;
}
#endif  // PECLET_CORE_HAVE_MORTON

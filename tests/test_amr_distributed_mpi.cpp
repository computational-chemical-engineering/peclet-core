// Distributed adaptive octree (tpx::amr::DistributedOctree) vs an independent
// serial reference (the whole domain as one block), at np = 1,2,4:
//   (1) after per-block SDF refinement + cross-block balance(), the global leaf
//       set (code, level) is identical to the serially refined+balanced octree —
//       i.e. distributed 2:1 balance == serial 2:1 balance;
//   (2) faceNeighborGather returns, for every leaf/face, the same neighbour value
//       a serial faceNeighbor lookup would — i.e. the owner-based ghost exchange
//       is correct (local + cross-rank).
//
// Guarded by TPX_HAVE_MORTON; a no-op pass without the morton sibling checkout.
#include "test_util.hpp"

#ifdef TPX_HAVE_MORTON
#include <algorithm>
#include <cmath>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/distributed_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/amr/refine.hpp"
#include "tpx/common/mpi.hpp"
#include "tpx/geom/sdf.hpp"

using namespace tpx;
using namespace tpx::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;

// Problem definition shared by the distributed run and the serial reference.
struct Problem {
  IVec<3> rootSize{4, 4, 4};
  unsigned lmax = 3;  // 32^3 fine domain
  unsigned target = 1;
  tpx::geom::Sphere sph{{16.0, 16.0, 16.0}, 8.0};
};

BO serialReference(const Problem& pr) {
  BO t(pr.rootSize, pr.lmax);
  AmrGeometry<3> geo;  // origin 0, h0 1
  auto sdf = [&](const Vec<3>& p) { return pr.sph.eval(p); };
  refineToSdf(t, geo, sdf, pr.target, /*band=*/1.0, /*balance=*/false);
  t.balance2to1();
  return t;
}

void run(MPI_Comm comm) {
  const Problem pr;

  // ---- distributed: decompose, refine each block by the global SDF, balance ----
  DistributedOctree<3, kBits> dist;
  AmrGeometry<3> gGeo;  // global origin 0, h0 1
  dist.init(pr.rootSize, pr.lmax, gGeo, {false, false, false}, comm);
  {
    auto lGeo = dist.localGeometry();
    auto sdf = [&](const Vec<3>& p) { return pr.sph.eval(p); };
    refineToSdf(dist.local(), lGeo, sdf, pr.target, /*band=*/1.0, /*balance=*/false);
  }
  dist.balance();

  // ---- serial reference (computed on every rank) ----
  BO ref = serialReference(pr);

  // ---- (1) global leaf set equality ----
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const Index nloc = dist.local().numLeaves();
  std::vector<unsigned long long> gcodes(static_cast<std::size_t>(nloc));
  std::vector<int> glev(static_cast<std::size_t>(nloc));
  for (Index i = 0; i < nloc; ++i) {
    gcodes[static_cast<std::size_t>(i)] = static_cast<unsigned long long>(dist.globalCode(i));
    glev[static_cast<std::size_t>(i)] = static_cast<int>(dist.local().level(i));
  }

  int ncount = static_cast<int>(nloc);
  std::vector<int> counts(static_cast<std::size_t>(size));
  MPI_Gather(&ncount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
  std::vector<int> displs(static_cast<std::size_t>(size), 0);
  int totalLeaves = 0;
  if (rank == 0) {
    for (int r = 0; r < size; ++r) {
      displs[static_cast<std::size_t>(r)] = totalLeaves;
      totalLeaves += counts[static_cast<std::size_t>(r)];
    }
  }
  std::vector<unsigned long long> allCodes(rank == 0 ? static_cast<std::size_t>(totalLeaves) : 0);
  std::vector<int> allLev(rank == 0 ? static_cast<std::size_t>(totalLeaves) : 0);
  MPI_Gatherv(gcodes.data(), ncount, MPI_UNSIGNED_LONG_LONG, allCodes.data(), counts.data(),
              displs.data(), MPI_UNSIGNED_LONG_LONG, 0, comm);
  MPI_Gatherv(glev.data(), ncount, MPI_INT, allLev.data(), counts.data(), displs.data(), MPI_INT, 0,
              comm);

  if (rank == 0) {
    TPX_CHECK_EQ((long long)totalLeaves, (long long)ref.numLeaves());
    std::vector<std::pair<unsigned long long, int>> got(static_cast<std::size_t>(totalLeaves));
    for (int i = 0; i < totalLeaves; ++i)
      got[static_cast<std::size_t>(i)] = {allCodes[static_cast<std::size_t>(i)],
                                          allLev[static_cast<std::size_t>(i)]};
    std::sort(got.begin(), got.end());
    bool match = (static_cast<Index>(got.size()) == ref.numLeaves());
    for (Index i = 0; match && i < ref.numLeaves(); ++i) {
      if (got[static_cast<std::size_t>(i)].first != static_cast<unsigned long long>(ref.code(i)) ||
          got[static_cast<std::size_t>(i)].second != static_cast<int>(ref.level(i)))
        match = false;
    }
    TPX_CHECK(match);
  }

  // ---- (2) face-neighbour gather matches serial faceNeighbor ----
  // field value of a leaf = its global Morton code (exact in double for this size).
  std::vector<double> field(static_cast<std::size_t>(nloc));
  for (Index i = 0; i < nloc; ++i) field[static_cast<std::size_t>(i)] = static_cast<double>(dist.globalCode(i));
  std::vector<double> gathered = dist.faceNeighborGather(field);

  // serial codes (sorted) for binary-search lookup of a leaf by global code.
  const auto& refCodes = ref.codes();
  int mism = 0;
  for (Index i = 0; i < nloc; ++i) {
    Code gc = dist.globalCode(i);
    auto it = std::lower_bound(refCodes.begin(), refCodes.end(), gc);
    if (it == refCodes.end() || *it != gc) {
      ++mism;
      continue;
    }
    Index js = static_cast<Index>(it - refCodes.begin());
    for (int axis = 0; axis < 3; ++axis)
      for (int dir = -1; dir <= 1; dir += 2) {
        int slot = static_cast<int>(i) * 6 + 2 * axis + (dir > 0 ? 0 : 1);
        Index jn = ref.faceNeighbor(js, axis, dir);
        double expect =
            (jn < 0) ? DistributedOctree<3, kBits>::kNoNeighbor : static_cast<double>(ref.code(jn));
        if (gathered[static_cast<std::size_t>(slot)] != expect) ++mism;
      }
  }
  TPX_CHECK_EQ(mism, 0);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  run(MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int fails = tpx::test::g_failures;
  int total = 0;
  MPI_Reduce(&fails, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (rank == 0) {
    if (total == 0) {
      std::printf("OK\n");
      return 0;
    }
    std::fprintf(stderr, "%d failure(s)\n", total);
    return 1;
  }
  return 0;
}
#else
int main() {
  std::printf("TPX_HAVE_MORTON not set — skipping distributed AMR test\n");
  return 0;
}
#endif  // TPX_HAVE_MORTON

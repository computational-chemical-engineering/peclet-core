// The Repartition stage (amr/docs/amr_mg_core_boundary.md §11.2-11.3, gate G-B2): makeStageComm
// and RedistributeTopology moving a level from its current decomposition onto a fresh
// proportional ORB of the same grid on the first np_L ranks.
//
//   1  makeStageComm: group IS the parent handle (not freed with the StageComm), sub is the
//      owners [0, np_L) in rank order and MPI_COMM_NULL elsewhere, members empty, myGroup -1.
//   2  forward == a one-shot redistributeGridFields over the same boxes, BITWISE — whole padded
//      buffers (ghosts untouched), 1-3 fields in one call, ghost width 1 and 2, special values
//      (-0, subnormal, NaN, inf) included; the source both proportional and WEIGHTED, the target
//      proportional on np_L in {1, 2, np/2, np}.
//   3  backward(forward(x)) == x exactly, every case; backward writes the inner cells only.
//   4  inactive ranks (rank >= np_L) receive nothing: no target cells, an empty dst accepted.
//   5  the segments tile: numSourceCells == this rank's current block, numTargetCells == its
//      target block (0 when inactive) — build() throws otherwise.
//   6  layout-agnostic: random-permutation srcIndex / dstIndex into sparse buffers, 3-D and 2-D.
//   7  through the policy: a weighted level whose sibling merge exceeds maxBlockCells gets a
//      Repartition target from chooseStageTarget, and the stage built from it round-trips.
//   8  the per-topology tag id: [0, kTagSpan) builds, anything else throws.
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/decomp/grid_redistribute.hpp"
#include "peclet/core/decomp/redistribute_topology.hpp"
#include "peclet/core/decomp/stage_comm.hpp"
#include "peclet/core/decomp/stage_target.hpp"
#include "test_util.hpp"

using peclet::core::Index;
using peclet::core::IVec;
using peclet::core::Real;
using peclet::core::decomp::Block;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::decomp::chooseStageTarget;
using peclet::core::decomp::largestBlockCells;
using peclet::core::decomp::makeStageComm;
using peclet::core::decomp::redistributeGridFields;
using peclet::core::decomp::RedistributeTopology;
using peclet::core::decomp::StageComm;
using peclet::core::decomp::StageKind;
using peclet::core::decomp::StageTarget;
using Dec = BlockDecomposer<3>;

namespace {

int rank = 0, nranks = 1;

// ---- deterministic "random" values, identical on every rank ------------------------------------
std::uint64_t mix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
// A double from (seed, key): mostly ordinary values, with -0, subnormals, inf and NaN mixed in.
double valueOf(std::uint64_t seed, std::uint64_t key) {
  const std::uint64_t h = mix(seed * 0x100000001B3ull ^ key);
  switch (h % 23) {
    case 0:
      return -0.0;
    case 1:
      return std::numeric_limits<double>::denorm_min() * static_cast<double>(h >> 40);
    case 2:
      return std::numeric_limits<double>::infinity();
    case 3: {
      const std::uint64_t bits = 0x7FF8000000000000ull | (h >> 20);  // NaN with a payload
      double d;
      std::memcpy(&d, &bits, sizeof d);
      return d;
    }
    default: {
      double d;
      std::uint64_t bits =
          (h & 0x800FFFFFFFFFFFFFull) | (std::uint64_t{0x3F0 + (h >> 52) % 32} << 52);
      std::memcpy(&d, &bits, sizeof d);
      return d;
    }
  }
}
bool sameBits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}
bool allRanks(bool ok) {
  int v = ok ? 1 : 0, all = 0;
  MPI_Allreduce(&v, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return all == 1;
}
void fail(const std::string& what) {
  if (rank == 0)
    std::fprintf(stderr, "FAILED: %s\n", what.c_str());
  ++::peclet::core::test::g_failures;
}
std::vector<Index> permutation(std::size_t n, std::uint64_t seed) {
  std::vector<Index> p(n);
  std::iota(p.begin(), p.end(), Index{0});
  for (std::size_t i = n; i > 1; --i)
    std::swap(p[i - 1], p[mix(seed * 7919ull + i) % i]);
  return p;
}
template <int Dim>
Index cellsOf(const Block<Dim>& b) {
  Index n = 1;
  for (int d = 0; d < Dim; ++d)
    n *= b.size[d];
  return n;
}

// ---- a padded x-fastest block buffer (redistributeGridFields' layout) --------------------------
struct Padded {
  Block<3> b;
  int g = 1;
  IVec<3> ext{};
  Padded(const Block<3>& blk, int gw) : b(blk), g(gw) {
    for (int k = 0; k < 3; ++k)
      ext[k] = blk.size[k] + 2 * gw;
  }
  std::size_t n() const { return static_cast<std::size_t>(ext[0] * ext[1] * ext[2]); }
  Index idx(const IVec<3>& gc) const {
    return (gc[0] - b.origin[0] + g) + (gc[1] - b.origin[1] + g) * ext[0] +
           (gc[2] - b.origin[2] + g) * ext[0] * ext[1];
  }
  template <class F>
  void forInner(F&& f) const {
    for (Index z = 0; z < b.size[2]; ++z)
      for (Index y = 0; y < b.size[1]; ++y)
        for (Index x = 0; x < b.size[0]; ++x)
          f(IVec<3>{b.origin[0] + x, b.origin[1] + y, b.origin[2] + z});
  }
};
// Field on a padded block: inner cells from the global function, ghosts rank-specific garbage.
std::vector<double> levelField(const Padded& p, const IVec<3>& G, std::uint64_t seed) {
  std::vector<double> v(p.n());
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = valueOf(seed ^ 0xABCDEFull, static_cast<std::uint64_t>(rank) * 1000003ull + i);
  p.forInner([&](const IVec<3>& gc) {
    v[static_cast<std::size_t>(p.idx(gc))] =
        valueOf(seed, static_cast<std::uint64_t>(gc[0] + G[0] * (gc[1] + G[1] * gc[2])));
  });
  return v;
}
std::vector<double> sentinel(std::size_t n, std::uint64_t seed) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = valueOf(seed, 0xFFFF0000ull + i);
  return v;
}

StageTarget<3> repartitionOnto(const IVec<3>& G, int npL) {
  StageTarget<3> t;
  t.kind = StageKind::Repartition;
  t.dec.init(static_cast<std::size_t>(npL), G);
  t.ownerOf.resize(static_cast<std::size_t>(npL));
  std::iota(t.ownerOf.begin(), t.ownerOf.end(), 0);
  return t;
}

std::vector<int> targetRankCounts() {
  std::vector<int> v{1, 2, std::max(1, nranks / 2), nranks};
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  v.erase(std::remove_if(v.begin(), v.end(), [](int n) { return n > nranks; }), v.end());
  return v;
}

std::vector<std::pair<std::string, Dec>> sources() {
  std::vector<std::pair<std::string, Dec>> v;
  const std::size_t P = static_cast<std::size_t>(nranks);
  for (const IVec<3>& G : {IVec<3>{24, 20, 18}, IVec<3>{32, 32, 32}, IVec<3>{17, 13, 11}}) {
    const std::string tag =
        std::to_string(G[0]) + "x" + std::to_string(G[1]) + "x" + std::to_string(G[2]);
    v.push_back({"proportional " + tag, Dec(P, G)});
    std::vector<Real> w(static_cast<std::size_t>(G[0] * G[1] * G[2]));
    for (std::size_t i = 0; i < w.size(); ++i)
      w[i] =
          0.25 + static_cast<double>(mix(i + 977) % 1000) / 100.0 +
          ((i % static_cast<std::size_t>(G[0])) < static_cast<std::size_t>(G[0] / 3) ? 6.0 : 0.0);
    v.push_back({"weighted " + tag, Dec(P, G, w)});
  }
  return v;
}

int gCases = 0;

// ---- 1 : the communicators ----------------------------------------------------------------------
void testComm() {
  for (int npL : targetRankCounts()) {
    ++gCases;
    const StageTarget<3> t = repartitionOnto(IVec<3>{16, 12, 10}, npL);
    bool ok = true;
    {
      StageComm c = makeStageComm(MPI_COMM_WORLD, t);
      ok = c.kind == StageKind::Repartition && c.group == MPI_COMM_WORLD &&
           c.parent == MPI_COMM_WORLD && c.members.empty() && c.myGroup == -1 &&
           c.active == (rank < npL) && c.myTargetBlock == (rank < npL ? rank : -1);
      if (rank < npL) {
        int sr = -1, ss = 0;
        MPI_Comm_rank(c.sub, &sr);
        MPI_Comm_size(c.sub, &ss);
        ok = ok && sr == rank && ss == npL;
      } else {
        ok = ok && c.sub == MPI_COMM_NULL;
      }
      StageComm moved = std::move(c);  // ownership moves with the struct
      ok = ok && moved.group == MPI_COMM_WORLD && c.group == MPI_COMM_NULL;
    }
    // the parent survives the StageComm (group == parent is not freed)
    int sz = 0;
    ok = ok && MPI_Comm_size(MPI_COMM_WORLD, &sz) == MPI_SUCCESS && sz == nranks;
    if (!allRanks(ok))
      fail("makeStageComm Repartition, np_L=" + std::to_string(npL));
  }
  // a Repartition target must own [0, np_L) by identity
  StageTarget<3> bad = repartitionOnto(IVec<3>{16, 12, 10}, 1);
  bad.ownerOf = {nranks - 1};
  bool threw = false;
  if (nranks > 1) {
    try {
      StageComm c = makeStageComm(MPI_COMM_WORLD, bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
  }
  if (!allRanks(threw == (nranks > 1)))
    fail("makeStageComm accepted a non-identity Repartition owner map");
}

// ---- 2 + 3 + 4 + 5 : against redistributeGridFields, and the round trip -------------------------
void testAgainstOneShot() {
  for (const auto& [name, src] : sources())
    for (int npL : targetRankCounts())
      for (int g : {1, 2})
        for (int nF : {1, 2, 3}) {
          ++gCases;
          const IVec<3> G = src.globalSize();
          const StageTarget<3> t = repartitionOnto(G, npL);
          StageComm c = makeStageComm(MPI_COMM_WORLD, t);
          const bool active = rank < npL;
          const Padded lv(src.block(static_cast<std::size_t>(rank)), g);
          const Padded tb(active ? t.dec.block(static_cast<std::size_t>(rank)) : Block<3>{}, g);

          RedistributeTopology<3, double> topo;
          topo.build(
              src, t, c, [&](const IVec<3>& gc) { return lv.idx(gc); },
              [&](const IVec<3>& gc) { return tb.idx(gc); }, /*id=*/g);
          bool ok = topo.kind() == StageKind::Repartition && topo.active() == active &&
                    topo.numSourceCells() == static_cast<std::size_t>(cellsOf(lv.b)) &&
                    topo.numTargetCells() ==
                        (active ? static_cast<std::size_t>(cellsOf(tb.b)) : std::size_t{0});

          std::vector<std::vector<double>> in, dst, ref;
          for (int f = 0; f < nF; ++f) {
            in.push_back(levelField(lv, G, 100 + f));
            dst.push_back(active ? sentinel(tb.n(), 200 + f) : std::vector<double>{});
            ref.push_back(active ? sentinel(tb.n(), 200 + f) : std::vector<double>{1.0});
          }
          std::vector<const double*> ip;
          std::vector<double*> dp, rp;
          for (int f = 0; f < nF; ++f) {
            ip.push_back(in[f].data());
            if (active)
              dp.push_back(dst[f].data());
            rp.push_back(ref[f].data());  // never written on an inactive rank
          }
          topo.forward(ip, dp);
          redistributeGridFields<double>(src, t.dec, rank, g, ip, rp, MPI_COMM_WORLD);
          if (active)
            for (int f = 0; f < nF; ++f)
              ok = ok && sameBits(dst[f], ref[f]);

          // the round trip, into a copy whose inner cells are wiped (ghosts must survive)
          std::vector<std::vector<double>> back = in;
          for (auto& b : back)
            lv.forInner([&](const IVec<3>& gc) { b[static_cast<std::size_t>(lv.idx(gc))] = 7.0; });
          std::vector<const double*> dcp;
          std::vector<double*> bp;
          for (int f = 0; f < nF; ++f) {
            if (active)
              dcp.push_back(dst[f].data());
            bp.push_back(back[f].data());
          }
          topo.backward(dcp, bp);
          for (int f = 0; f < nF; ++f)
            ok = ok && sameBits(back[f], in[f]);
          if (!allRanks(ok))
            fail(name + " -> np_L=" + std::to_string(npL) + " g=" + std::to_string(g) +
                 " fields=" + std::to_string(nF));
        }
}

// ---- 6 : layout-agnostic, 3-D and 2-D
// ------------------------------------------------------------
template <int Dim>
bool permutedRoundTrip(const BlockDecomposer<Dim>& src, int npL, std::uint64_t seed) {
  StageTarget<Dim> t;
  t.kind = StageKind::Repartition;
  t.dec.init(static_cast<std::size_t>(npL), src.globalSize());
  t.ownerOf.resize(static_cast<std::size_t>(npL));
  std::iota(t.ownerOf.begin(), t.ownerOf.end(), 0);
  StageComm c = makeStageComm(MPI_COMM_WORLD, t);
  const IVec<Dim> G = src.globalSize();
  auto lin = [](const IVec<Dim>& gc, const Block<Dim>& b) {
    Index i = 0;
    for (int d = Dim - 1; d >= 0; --d)
      i = i * b.size[d] + (gc[d] - b.origin[d]);
    return i;
  };
  auto glin = [&](const IVec<Dim>& gc) {
    Index i = 0;
    for (int d = Dim - 1; d >= 0; --d)
      i = i * G[d] + gc[d];
    return i;
  };
  const bool active = rank < npL;
  const Block<Dim> sb = src.block(static_cast<std::size_t>(rank));
  const Block<Dim> db = active ? t.dec.block(static_cast<std::size_t>(rank)) : Block<Dim>{};
  const std::vector<Index> sp =
      permutation(static_cast<std::size_t>(3 * cellsOf(sb)), seed + 1000 + rank);
  const std::vector<Index> dpm =
      permutation(active ? static_cast<std::size_t>(3 * cellsOf(db)) : 0, seed + 2000 + rank);
  auto srcIndex = [&](const IVec<Dim>& gc) { return sp[static_cast<std::size_t>(lin(gc, sb))]; };
  auto dstIndex = [&](const IVec<Dim>& gc) { return dpm[static_cast<std::size_t>(lin(gc, db))]; };
  RedistributeTopology<Dim, double> topo;
  topo.build(src, t, c, srcIndex, dstIndex, /*id=*/3);
  std::vector<double> in(static_cast<std::size_t>(3 * cellsOf(sb)), 5.0), dst;
  if (active)
    dst.assign(static_cast<std::size_t>(3 * cellsOf(db)), -5.0);
  IVec<Dim> end{};
  for (int d = 0; d < Dim; ++d)
    end[d] = sb.origin[d] + sb.size[d];
  peclet::core::forEachInBox<Dim>(sb.origin, end, [&](const IVec<Dim>& gc) {
    in[static_cast<std::size_t>(srcIndex(gc))] =
        valueOf(seed, static_cast<std::uint64_t>(glin(gc)));
  });
  std::vector<double*> dp;
  std::vector<const double*> dcp;
  if (active) {
    dp.push_back(dst.data());
    dcp.push_back(dst.data());
  }
  topo.forward({in.data()}, dp);
  bool ok = true;
  if (active) {
    std::vector<double> expect(dst.size(), -5.0);
    for (int d = 0; d < Dim; ++d)
      end[d] = db.origin[d] + db.size[d];
    peclet::core::forEachInBox<Dim>(db.origin, end, [&](const IVec<Dim>& gc) {
      expect[static_cast<std::size_t>(dstIndex(gc))] =
          valueOf(seed, static_cast<std::uint64_t>(glin(gc)));
    });
    ok = sameBits(dst, expect);
  }
  std::vector<double> back(in.size(), 5.0);
  topo.backward(dcp, {back.data()});
  return ok && sameBits(back, in);
}

void testLayoutAgnostic() {
  for (const auto& [name, src] : sources())
    for (int npL : targetRankCounts()) {
      ++gCases;
      if (!allRanks(permutedRoundTrip<3>(src, npL, 31)))
        fail("permuted " + name + " -> np_L=" + std::to_string(npL));
    }
  const BlockDecomposer<2> src2(static_cast<std::size_t>(nranks), IVec<2>{26, 18});
  std::vector<Real> w2(26 * 18);
  for (std::size_t i = 0; i < w2.size(); ++i)
    w2[i] = 1.0 + static_cast<double>(mix(i) % 7);
  const BlockDecomposer<2> src2w(static_cast<std::size_t>(nranks), IVec<2>{26, 18}, w2);
  for (int npL : targetRankCounts()) {
    gCases += 2;
    const bool plain = permutedRoundTrip<2>(src2, npL, 43);  // both collective: no short-circuit
    const bool weighted = permutedRoundTrip<2>(src2w, npL, 47);
    if (!allRanks(plain && weighted))
      fail("2-D permuted -> np_L=" + std::to_string(npL));
  }
}

// ---- 7 : through the policy
// ----------------------------------------------------------------------
bool can(Index d) {
  return (d % 2 == 0) && (d / 2 >= 2);
}
bool flowLiftable(const Dec& d) {  // flow's per-axis rule (mac_cutcell_mg.hpp)
  for (int ax = 0; ax < 3; ++ax)
    if (can(d.globalSize()[ax]))
      for (std::size_t b = 0; b < d.numBlocks(); ++b)
        if ((d.origins()[b][ax] % 2) || (d.sizes()[b][ax] % 2))
          return false;
  return true;
}
void testThroughPolicy() {
  if (nranks < 2)
    return;
  ++gCases;
  const IVec<3> G{48, 48, 48};
  std::vector<Real> w(static_cast<std::size_t>(G[0] * G[1] * G[2]));
  // a particle heap (the probe's shape, tilt 0.3): every ORB split lands off the halvings, and at
  // every np >= 2 the sibling merge the search finds exceeds one level-0 block -> Repartition
  for (std::size_t i = 0; i < w.size(); ++i) {
    const double x = static_cast<double>(i % 48), y = static_cast<double>((i / 48) % 48),
                 z = static_cast<double>(i / (48 * 48));
    const double zs =
        17.0 * (1.0 + 0.3 * (0.5 - (x + 0.5) / 48.0) + 0.3 * (0.5 - (y + 0.5) / 48.0));
    w[i] = 1.0 + (z + 0.5 < zs ? 3.0 + static_cast<double>(mix(i) % 5) : 0.0);
  }
  const Dec cur(static_cast<std::size_t>(nranks), G, w);
  const Index cap = largestBlockCells(cur);
  const StageTarget<3> t = chooseStageTarget(cur, G, flowLiftable, 0, cap);
  bool ok = !flowLiftable(cur) && t.kind == StageKind::Repartition;
  if (ok) {
    StageComm c = makeStageComm(MPI_COMM_WORLD, t);
    const bool active = c.active;
    const Padded lv(cur.block(static_cast<std::size_t>(rank)), 1);
    const Padded tb(active ? t.dec.block(static_cast<std::size_t>(rank)) : Block<3>{}, 1);
    RedistributeTopology<3, double> topo;
    topo.build(
        cur, t, c, [&](const IVec<3>& gc) { return lv.idx(gc); },
        [&](const IVec<3>& gc) { return tb.idx(gc); }, /*id=*/0);
    std::vector<double> in = levelField(lv, G, 900), mid, back = sentinel(lv.n(), 901);
    std::vector<double> expect = back;
    lv.forInner([&](const IVec<3>& gc) {
      const std::size_t i = static_cast<std::size_t>(lv.idx(gc));
      expect[i] = in[i];
    });
    if (active)
      mid.assign(tb.n(), 0.0);
    topo.forward({in.data()}, active ? std::vector<double*>{mid.data()} : std::vector<double*>{});
    topo.backward(active ? std::vector<const double*>{mid.data()} : std::vector<const double*>{},
                  {back.data()});
    ok = sameBits(back, expect);
  }
  if (rank == 0)
    std::printf("  policy on a weighted 48^3 level, np=%d, maxBlockCells %lld: kind %d on %zu\n",
                nranks, (long long)cap, (int)t.kind, t.dec.numBlocks());
  if (!allRanks(ok))
    fail("through the policy");
}

// ---- 8 : the tag id
// ------------------------------------------------------------------------------
void testTagId() {
  ++gCases;
  using Topo = RedistributeTopology<3, double>;
  const Dec src(static_cast<std::size_t>(nranks), IVec<3>{12, 10, 8});
  const StageTarget<3> t = repartitionOnto(src.globalSize(), 1);
  StageComm c = makeStageComm(MPI_COMM_WORLD, t);
  auto slot = [](const IVec<3>& gc) { return gc[0] + 12 * (gc[1] + 10 * gc[2]); };
  bool ok = true;
  for (int id : {-1, 0, Topo::kTagSpan - 1, Topo::kTagSpan}) {
    bool threw = false;
    try {
      Topo topo;
      topo.build(src, t, c, slot, slot, id);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    ok = ok && threw == (id < 0 || id >= Topo::kTagSpan);
  }
  ok = ok && Topo::kTagBase >= 1 && Topo::kTagBase + Topo::kTagSpan - 1 < 11;  // below AMR's 11
  if (!allRanks(ok))
    fail("the per-topology tag id range");
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  try {
    testComm();
    testAgainstOneShot();
    testLayoutAgnostic();
    testThroughPolicy();
    testTagId();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "rank %d: exception: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int fails = ::peclet::core::test::g_failures, all = 0;
  MPI_Allreduce(&fails, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("np=%d: %d repartition cases, %s\n", nranks, gCases, all == 0 ? "OK" : "FAILED");
  MPI_Finalize();
  return all == 0 ? 0 : 1;
}

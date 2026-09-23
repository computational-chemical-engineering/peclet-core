// makeStageComm + RedistributeTopology + gatherByGlobalId (peclet/core/decomp/) against the two
// reference implementations they replace, TRANSCRIBED below (core links neither):
//
//   flow's Telescope (flow/src/mac_cutcell_mg.hpp): the MPI_Comm_split pair and member lists of
//     CutcellMG::initMpi, and the bodies of teleGather / teleScatterAdd;
//   amr's ReplicatedTailStage (amr/include/peclet/amr/mg_stage.hpp): the global-id maps and the
//     Allgatherv of build / moveUp, and moveDown's local pick.
//
//   1  makeStageComm == flow's communicators: congruent group and sub communicators, the same
//      member list, owner == flow's root, for every sibling-merge depth of several partitions.
//   2  SiblingMerge forward == flow's teleGather and backward == flow's teleScatterAdd, BITWISE —
//      whole padded buffers including untouched ghosts, three fields in one call, ghost width 1
//      and 2, special values (-0, subnormal, NaN, inf) included.
//   3  Replicated forward == amr's moveUp and backward == amr's moveDown, BITWISE, with amr-like
//      permuted row orders on both sides.
//   4  backward(forward(x)) == x exactly, both kinds, every depth.
//   5  gatherByGlobalId == a direct reference, for double and for a struct; duplicate ids throw.
//   6  Layout-agnostic: random-permutation srcIndex / dstIndex (a hidden x-fastest assumption
//      fails this), 2-D as well as 3-D.
#include <mpi.h>

#include <algorithm>
#include <cmath>
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
#include "peclet/core/decomp/gather_by_global_id.hpp"
#include "peclet/core/decomp/redistribute_topology.hpp"
#include "peclet/core/decomp/stage_comm.hpp"
#include "peclet/core/decomp/stage_target.hpp"
#include "test_util.hpp"

using peclet::core::Index;
using peclet::core::IVec;
using peclet::core::Real;
using peclet::core::decomp::Block;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::decomp::gatherByGlobalId;
using peclet::core::decomp::makeStageComm;
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
// `withNaN = false` for values that are ADDED in a check: IEEE addition is commutative except for
// the payload of a NaN + NaN, and the compiler may swap the operands of one of two otherwise
// identical `a += b` loops — which is the compiler's freedom, not a difference in the movement.
double valueOf(std::uint64_t seed, std::uint64_t key, bool withNaN = true) {
  const std::uint64_t h = mix(seed * 0x100000001B3ull ^ key);
  switch (withNaN ? h % 23 : 4 + h % 19) {
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

// ---- a padded x-fastest block buffer: flow's level layout --------------------------------------
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
};
// Field f of a level on a padded block: inner cells from the global function, ghosts garbage that
// must never move.
std::vector<double> levelField(const Padded& p, const IVec<3>& G, std::uint64_t seed,
                               bool withNaN = true) {
  std::vector<double> v(p.n());
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = valueOf(seed ^ 0xABCDEFull, static_cast<std::uint64_t>(rank) * 1000003ull + i, withNaN);
  for (Index z = 0; z < p.b.size[2]; ++z)
    for (Index y = 0; y < p.b.size[1]; ++y)
      for (Index x = 0; x < p.b.size[0]; ++x) {
        const IVec<3> gc{p.b.origin[0] + x, p.b.origin[1] + y, p.b.origin[2] + z};
        v[static_cast<std::size_t>(p.idx(gc))] = valueOf(
            seed, static_cast<std::uint64_t>(gc[0] + G[0] * (gc[1] + G[1] * gc[2])), withNaN);
      }
  return v;
}
std::vector<double> sentinel(std::size_t n, std::uint64_t seed) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = valueOf(seed, 0xFFFF0000ull + i);
  return v;
}

// ---- flow's Telescope, transcribed (CutcellMG::initMpi, teleGather, teleScatterAdd) -----------
struct FlowTele {
  MPI_Comm groupComm = MPI_COMM_NULL, subComm = MPI_COMM_NULL;
  bool root = false;
  int nMembers = 1, g = 1;
  IVec<3> mInner{}, mOg{}, mExt{};
  std::vector<IVec<3>> memO, memS;
  std::vector<int> counts, displs;
  std::vector<int> memberRanks;  // not in flow: for the member-list comparison
  ~FlowTele() {
    if (groupComm != MPI_COMM_NULL)
      MPI_Comm_free(&groupComm);
    if (subComm != MPI_COMM_NULL)
      MPI_Comm_free(&subComm);
  }
};
void flowInit(FlowTele& T, const Dec& curDec, const Dec& cand, const std::vector<int>& groupOf,
              const std::vector<int>& rootOf, int curRank, MPI_Comm curComm, int g) {
  const int myGroup = groupOf[(std::size_t)curRank];
  T.root = (curRank == rootOf[(std::size_t)myGroup]);
  MPI_Comm_split(curComm, myGroup, curRank, &T.groupComm);
  MPI_Comm_split(curComm, T.root ? 0 : MPI_UNDEFINED, myGroup, &T.subComm);
  T.g = g;
  const auto mb = cand.block((std::size_t)myGroup);
  T.mOg = mb.origin;
  T.mInner = mb.size;
  T.mExt = IVec<3>{T.mInner[0] + 2 * T.g, T.mInner[1] + 2 * T.g, T.mInner[2] + 2 * T.g};
  int disp = 0;
  for (std::size_t b = 0; b < curDec.numBlocks(); ++b)
    if (groupOf[b] == myGroup) {
      const auto fb = curDec.block(b);
      T.memO.push_back(fb.origin);
      T.memS.push_back(fb.size);
      const int n = (int)(fb.size[0] * fb.size[1] * fb.size[2]);
      T.counts.push_back(n);
      T.displs.push_back(disp);
      T.memberRanks.push_back((int)b);
      disp += n;
    }
  T.nMembers = (int)T.counts.size();
}
// teleGather with the Kokkos mirrors replaced by host vectors (lv = this rank's level block).
void flowGather(const FlowTele& T, const Padded& lv, const std::vector<double>& hs,
                std::vector<double>& hd) {
  const int g = lv.g;
  const IVec<3> inner = lv.b.size, ext = lv.ext;
  const std::size_t nIn = (std::size_t)inner[0] * inner[1] * inner[2];
  std::vector<double> sb(nIn);
  for (int k = 0; k < inner[2]; ++k)
    for (int j = 0; j < inner[1]; ++j)
      for (int i = 0; i < inner[0]; ++i)
        sb[(std::size_t)i + (std::size_t)j * inner[0] + (std::size_t)k * inner[0] * inner[1]] =
            hs[(long)(i + g) + (long)(j + g) * ext[0] + (long)(k + g) * (long)ext[0] * ext[1]];
  std::vector<double> rb;
  if (T.root)
    rb.resize((std::size_t)T.displs.back() + (std::size_t)T.counts.back());
  MPI_Gatherv(sb.data(), (int)nIn, MPI_DOUBLE, T.root ? rb.data() : nullptr, T.counts.data(),
              T.displs.data(), MPI_DOUBLE, 0, T.groupComm);
  if (!T.root)
    return;
  for (int m = 0; m < T.nMembers; ++m) {
    const IVec<3> o = T.memO[m], sz = T.memS[m];
    const double* q = rb.data() + T.displs[m];
    for (int k = 0; k < sz[2]; ++k)
      for (int j = 0; j < sz[1]; ++j)
        for (int i = 0; i < sz[0]; ++i) {
          const long x = o[0] - T.mOg[0] + i + g, y = o[1] - T.mOg[1] + j + g,
                     z = o[2] - T.mOg[2] + k + g;
          hd[x + y * T.mExt[0] + z * T.mExt[0] * T.mExt[1]] =
              q[(std::size_t)i + (std::size_t)j * sz[0] + (std::size_t)k * sz[0] * sz[1]];
        }
  }
}
// teleScatterAdd, host vectors; also hands back the scattered values `rbOut` (pre-add).
void flowScatterAdd(const FlowTele& T, const Padded& lv, const std::vector<double>& hs,
                    std::vector<double>& hd, std::vector<double>& rbOut) {
  const int g = lv.g;
  std::vector<double> sb;
  if (T.root) {
    sb.resize((std::size_t)T.displs.back() + (std::size_t)T.counts.back());
    for (int m = 0; m < T.nMembers; ++m) {
      const IVec<3> o = T.memO[m], sz = T.memS[m];
      double* q = sb.data() + T.displs[m];
      for (int k = 0; k < sz[2]; ++k)
        for (int j = 0; j < sz[1]; ++j)
          for (int i = 0; i < sz[0]; ++i) {
            const long x = o[0] - T.mOg[0] + i + g, y = o[1] - T.mOg[1] + j + g,
                       z = o[2] - T.mOg[2] + k + g;
            q[(std::size_t)i + (std::size_t)j * sz[0] + (std::size_t)k * sz[0] * sz[1]] =
                hs[x + y * T.mExt[0] + z * T.mExt[0] * T.mExt[1]];
          }
    }
  }
  const IVec<3> inner = lv.b.size, ext = lv.ext;
  const std::size_t nIn = (std::size_t)inner[0] * inner[1] * inner[2];
  std::vector<double> rb(nIn);
  MPI_Scatterv(T.root ? sb.data() : nullptr, T.counts.data(), T.displs.data(), MPI_DOUBLE,
               rb.data(), (int)nIn, MPI_DOUBLE, 0, T.groupComm);
  for (int k = 0; k < inner[2]; ++k)
    for (int j = 0; j < inner[1]; ++j)
      for (int i = 0; i < inner[0]; ++i)
        hd[(long)(i + g) + (long)(j + g) * ext[0] + (long)(k + g) * (long)ext[0] * ext[1]] +=
            rb[(std::size_t)i + (std::size_t)j * inner[0] + (std::size_t)k * inner[0] * inner[1]];
  rbOut = rb;
}

// The partitions every sibling-merge check runs on (one block per rank).
std::vector<std::pair<std::string, Dec>> partitions() {
  std::vector<std::pair<std::string, Dec>> v;
  const std::size_t P = (std::size_t)nranks;
  v.push_back({"plain 24^3 (flow's starved case)", Dec(P, IVec<3>{24, 24, 24})});
  v.push_back({"plain 32^3", Dec(P, IVec<3>{32, 32, 32})});
  v.push_back({"plain 20x14x9", Dec(P, IVec<3>{20, 14, 9})});
  const IVec<3> gw{18, 12, 10};
  std::vector<Real> w((std::size_t)(gw[0] * gw[1] * gw[2]));
  for (std::size_t i = 0; i < w.size(); ++i)
    w[i] = 0.5 + (double)(mix(i) % 1000) / 250.0;
  v.push_back({"weighted 18x12x10", Dec(P, gw, w)});
  return v;
}

// Every sibling-merge target of `cur`: agglomerated(d) for each d (d = treeDepth is the identity).
std::vector<StageTarget<3>> siblingTargets(const Dec& cur) {
  std::vector<StageTarget<3>> ts;
  for (int d = cur.treeDepth(); d >= 0; --d) {
    StageTarget<3> t;
    t.kind = StageKind::SiblingMerge;
    t.dec = cur.agglomerated(d, &t.groupOf, &t.ownerOf);
    ts.push_back(std::move(t));
  }
  return ts;
}

int gCases = 0;

// ---- 1 + 2 + 4 : sibling merge vs flow
// -----------------------------------------------------------
void testSiblingMerge() {
  for (const auto& [name, cur] : partitions())
    for (const StageTarget<3>& t : siblingTargets(cur))
      for (int g : {1, 2}) {
        ++gCases;
        const IVec<3> G = cur.globalSize();
        FlowTele T;
        flowInit(T, cur, t.dec, t.groupOf, t.ownerOf, rank, MPI_COMM_WORLD, g);
        StageComm c = makeStageComm(MPI_COMM_WORLD, t);

        // 1: the communicators
        bool ok = c.active == T.root && c.members == T.memberRanks &&
                  c.myGroup == t.groupOf[(std::size_t)rank] &&
                  c.myTargetBlock == (T.root ? c.myGroup : -1);
        int cmp = MPI_UNEQUAL;
        MPI_Comm_compare(c.group, T.groupComm, &cmp);
        ok = ok && cmp == MPI_CONGRUENT;
        if (T.root) {
          MPI_Comm_compare(c.sub, T.subComm, &cmp);
          int sr = -1;
          MPI_Comm_rank(c.sub, &sr);
          ok = ok && cmp == MPI_CONGRUENT && sr == c.myTargetBlock;
        } else {
          ok = ok && c.sub == MPI_COMM_NULL;
        }
        int gr = -1;
        MPI_Comm_rank(c.group, &gr);
        ok = ok && (gr == 0) == T.root;

        // 2: the movement, three fields in one call vs three flow gathers
        const Padded lv(cur.block((std::size_t)rank), g);
        const Padded mb(t.dec.block((std::size_t)c.myGroup), g);
        std::vector<std::vector<double>> src, refDst, dst;
        for (int f = 0; f < 3; ++f) {
          src.push_back(levelField(lv, G, 100 + f));
          refDst.push_back(c.active ? sentinel(mb.n(), 200 + f) : std::vector<double>{});
          dst.push_back(refDst.back());
        }
        for (int f = 0; f < 3; ++f)
          flowGather(T, lv, src[f], refDst[f]);
        RedistributeTopology<3, double> topo;
        topo.build(
            cur, t, c, [&](const IVec<3>& gc) { return lv.idx(gc); },
            [&](const IVec<3>& gc) { return mb.idx(gc); });
        std::vector<const double*> sp;
        std::vector<double*> dp;
        for (int f = 0; f < 3; ++f) {
          sp.push_back(src[f].data());
          if (c.active)
            dp.push_back(dst[f].data());
        }
        topo.forward(sp, dp);
        for (int f = 0; f < 3; ++f)
          ok = ok && sameBits(dst[f], refDst[f]);

        // backward vs teleScatterAdd: the scattered values, and the level after the add
        std::vector<std::vector<double>> corr, lvl, lvlRef, back;
        for (int f = 0; f < 3; ++f) {
          corr.push_back(c.active ? levelField(mb, G, 300 + f, false) : std::vector<double>{});
          lvlRef.push_back(levelField(lv, G, 400 + f, false));
          lvl.push_back(lvlRef.back());
          back.push_back(sentinel(lv.n(), 500 + f));
        }
        std::vector<std::vector<double>> rb(3);
        for (int f = 0; f < 3; ++f)
          flowScatterAdd(T, lv, corr[f], lvlRef[f], rb[f]);
        std::vector<const double*> cp;
        std::vector<double*> bp;
        for (int f = 0; f < 3; ++f) {
          if (c.active)
            cp.push_back(corr[f].data());
          bp.push_back(back[f].data());
        }
        topo.backward(cp, bp);
        for (int f = 0; f < 3; ++f) {
          std::vector<double> scattered, sentinelOk = sentinel(lv.n(), 500 + f);
          std::size_t k = 0;
          for (Index z = 0; z < lv.b.size[2]; ++z)
            for (Index y = 0; y < lv.b.size[1]; ++y)
              for (Index x = 0; x < lv.b.size[0]; ++x, ++k) {
                const Index i =
                    lv.idx(IVec<3>{lv.b.origin[0] + x, lv.b.origin[1] + y, lv.b.origin[2] + z});
                scattered.push_back(back[f][(std::size_t)i]);
                lvl[f][(std::size_t)i] += back[f][(std::size_t)i];  // flow's add, caller-side
                sentinelOk[(std::size_t)i] = back[f][(std::size_t)i];
              }
          ok = ok && sameBits(scattered, rb[f]) && sameBits(lvl[f], lvlRef[f]) &&
               sameBits(back[f], sentinelOk);  // ghosts of `back` untouched
        }

        // 4: the round trip, through fresh target buffers
        std::vector<std::vector<double>> mid, rt;
        std::vector<double*> mp, rp;
        std::vector<const double*> mcp;
        for (int f = 0; f < 3; ++f) {
          mid.push_back(c.active ? sentinel(mb.n(), 600 + f) : std::vector<double>{});
          rt.push_back(src[f]);
        }
        for (int f = 0; f < 3; ++f) {
          if (c.active) {
            mp.push_back(mid[f].data());
            mcp.push_back(mid[f].data());
          }
          rp.push_back(rt[f].data());
        }
        topo.forward(sp, mp);
        for (int f = 0; f < 3; ++f)  // wipe the inner cells so the round trip must restore them
          for (Index z = 0; z < lv.b.size[2]; ++z)
            for (Index y = 0; y < lv.b.size[1]; ++y)
              for (Index x = 0; x < lv.b.size[0]; ++x)
                rt[f][(std::size_t)lv.idx(
                    IVec<3>{lv.b.origin[0] + x, lv.b.origin[1] + y, lv.b.origin[2] + z})] = 7.0;
        topo.backward(mcp, rp);
        for (int f = 0; f < 3; ++f)
          ok = ok && sameBits(rt[f], src[f]);

        if (!allRanks(ok)) {
          if (rank == 0)
            std::fprintf(stderr, "sibling merge FAILED: %s, %zu target blocks, g=%d\n",
                         name.c_str(), t.dec.numBlocks(), g);
          ++::peclet::core::test::g_failures;
        }
      }
}

// ---- 3 + 4 : replicated vs amr's ReplicatedTailStage --------------------------------------------
// amr's global id (x-fastest).
long long gidOf(const IVec<3>& g, const IVec<3>& G) {
  long long id = 0;
  for (int d = 2; d >= 0; --d)
    id = id * static_cast<long long>(G[d]) + static_cast<long long>(g[d]);
  return id;
}
// A permutation of [0, n), the same on every rank for the same seed.
std::vector<Index> permutation(std::size_t n, std::uint64_t seed) {
  std::vector<Index> p(n);
  std::iota(p.begin(), p.end(), Index{0});
  for (std::size_t i = n; i > 1; --i)
    std::swap(p[i - 1], p[mix(seed * 7919ull + i) % i]);
  return p;
}

void testReplicated() {
  for (const auto& [name, cur] : partitions()) {
    ++gCases;
    const IVec<3> G = cur.globalSize();
    const Index nt = G[0] * G[1] * G[2];
    const Block<3> blk = cur.block((std::size_t)rank);
    const Index n = blk.size[0] * blk.size[1] * blk.size[2];

    // amr's layouts: this rank's rows in some leaf order (here a permutation of the block's
    // x-fastest cells), the tail's rows in the gathered octree's order (a permutation of the grid).
    std::vector<IVec<3>> cellOfRow((std::size_t)n);
    {
      const std::vector<Index> p = permutation((std::size_t)n, 17 + rank);
      Index k = 0;
      for (Index z = 0; z < blk.size[2]; ++z)
        for (Index y = 0; y < blk.size[1]; ++y)
          for (Index x = 0; x < blk.size[0]; ++x, ++k)
            cellOfRow[(std::size_t)p[(std::size_t)k]] =
                IVec<3>{blk.origin[0] + x, blk.origin[1] + y, blk.origin[2] + z};
    }
    const std::vector<Index> rowOfGid = permutation((std::size_t)nt, 29);

    // --- amr's build, transcribed ---
    std::vector<long long> gid((std::size_t)n);
    for (Index i = 0; i < n; ++i)
      gid[(std::size_t)i] = gidOf(cellOfRow[(std::size_t)i], G);
    std::vector<int> counts((std::size_t)nranks, 0), displs((std::size_t)nranks, 0);
    int ni = (int)n;
    MPI_Allgather(&ni, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int tot = 0;
    for (int r = 0; r < nranks; ++r) {
      displs[(std::size_t)r] = tot;
      tot += counts[(std::size_t)r];
    }
    std::vector<long long> gidAll((std::size_t)tot);
    MPI_Allgatherv(gid.data(), ni, MPI_LONG_LONG, gidAll.data(), counts.data(), displs.data(),
                   MPI_LONG_LONG, MPI_COMM_WORLD);
    std::vector<Index> rowOfGathered((std::size_t)tot), rowOfLocal((std::size_t)n);
    for (int k = 0; k < tot; ++k)
      rowOfGathered[(std::size_t)k] = rowOfGid[(std::size_t)gidAll[(std::size_t)k]];
    for (Index i = 0; i < n; ++i)
      rowOfLocal[(std::size_t)i] = rowOfGid[(std::size_t)gid[(std::size_t)i]];

    // --- amr's moveUp / moveDown, transcribed ---
    std::vector<double> src((std::size_t)n);
    for (Index i = 0; i < n; ++i)
      src[(std::size_t)i] = valueOf(700, (std::uint64_t)gid[(std::size_t)i]);
    std::vector<double> recv((std::size_t)tot), host((std::size_t)nt, 0.0);
    MPI_Allgatherv(src.data(), ni, MPI_DOUBLE, recv.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    for (std::size_t k = 0; k < recv.size(); ++k)
      host[(std::size_t)rowOfGathered[k]] = recv[k];
    std::vector<double> tx((std::size_t)nt), refDown((std::size_t)n);
    for (Index r = 0; r < nt; ++r)
      tx[(std::size_t)r] = valueOf(800, (std::uint64_t)r);
    for (Index i = 0; i < n; ++i)
      refDown[(std::size_t)i] = tx[(std::size_t)rowOfLocal[(std::size_t)i]];

    // --- core ---
    auto never = [](const Dec&) { return false; };
    const StageTarget<3> t = peclet::core::decomp::chooseStageTarget(cur, G, never, 0);
    bool ok = (nranks == 1) ? t.kind == StageKind::InPlace : t.kind == StageKind::Replicated;
    StageTarget<3> rt = t;
    rt.kind = StageKind::Replicated;  // np=1: move it anyway (one rank, one block)
    rt.dec.init(1, G);
    rt.ownerOf = {0};
    rt.groupOf.clear();
    StageComm c = makeStageComm(MPI_COMM_WORLD, rt);
    ok = ok && c.active && c.myTargetBlock == 0 && (int)c.members.size() == nranks;
    int cmp = MPI_UNEQUAL;
    MPI_Comm_compare(c.sub, MPI_COMM_WORLD, &cmp);
    ok = ok && cmp == MPI_CONGRUENT;
    // the row of a global cell on this rank, and in the tail
    std::vector<Index> rowOfCell((std::size_t)n);
    for (Index i = 0; i < n; ++i) {
      const IVec<3>& gc = cellOfRow[(std::size_t)i];
      rowOfCell[(std::size_t)(
          (gc[0] - blk.origin[0]) +
          blk.size[0] * ((gc[1] - blk.origin[1]) + blk.size[1] * (gc[2] - blk.origin[2])))] = i;
    }
    RedistributeTopology<3, double> topo;
    topo.build(
        cur, rt, c,
        [&](const IVec<3>& gc) {
          return rowOfCell[(std::size_t)(
              (gc[0] - blk.origin[0]) +
              blk.size[0] * ((gc[1] - blk.origin[1]) + blk.size[1] * (gc[2] - blk.origin[2])))];
        },
        [&](const IVec<3>& gc) { return rowOfGid[(std::size_t)gidOf(gc, G)]; });
    std::vector<double> dst((std::size_t)nt, 0.0), down((std::size_t)n, -1.0);
    topo.forward({src.data()}, {dst.data()});
    ok = ok && sameBits(dst, host);
    topo.backward({tx.data()}, {down.data()});
    ok = ok && sameBits(down, refDown);
    // 4: round trip
    std::vector<double> rtv((std::size_t)n, 3.0);
    topo.backward({dst.data()}, {rtv.data()});
    ok = ok && sameBits(rtv, src);
    if (!allRanks(ok)) {
      if (rank == 0)
        std::fprintf(stderr, "replicated FAILED: %s\n", name.c_str());
      ++::peclet::core::test::g_failures;
    }
  }
}

// ---- 5 : gatherByGlobalId ----------------------------------------------------------------------
struct Pair {
  int a;
  float b;
};
void testGatherByGlobalId() {
  for (long long nGlobal : {0LL, 1LL, 37LL, 1000LL}) {
    ++gCases;
    // ids dealt irregularly: a global permutation cut at pseudo-random points (rank 0 may get none)
    const std::vector<Index> perm = permutation((std::size_t)nGlobal, 41);
    std::vector<long long> cut((std::size_t)nranks + 1, 0);
    for (int r = 1; r < nranks; ++r)
      cut[(std::size_t)r] = (long long)(mix((std::uint64_t)r * 13) % (std::uint64_t)(nGlobal + 1));
    cut[(std::size_t)nranks] = nGlobal;
    std::sort(cut.begin(), cut.end());
    std::vector<long long> ids;
    std::vector<double> vals;
    std::vector<Pair> pvals;
    for (long long k = cut[(std::size_t)rank]; k < cut[(std::size_t)rank + 1]; ++k) {
      const long long id = perm[(std::size_t)k];
      ids.push_back(id);
      vals.push_back(valueOf(900, (std::uint64_t)id));
      pvals.push_back(Pair{(int)(id * 3 + 1), (float)id * 0.5f});
    }
    std::vector<double> out;
    std::vector<Pair> pout;
    gatherByGlobalId(ids, vals, nGlobal, out, MPI_COMM_WORLD);
    gatherByGlobalId(ids, pvals, nGlobal, pout, MPI_COMM_WORLD);
    std::vector<double> ref((std::size_t)nGlobal);
    for (long long id = 0; id < nGlobal; ++id)
      ref[(std::size_t)id] = valueOf(900, (std::uint64_t)id);
    bool ok = sameBits(out, ref) && pout.size() == (std::size_t)nGlobal;
    for (long long id = 0; ok && id < nGlobal; ++id)
      ok = pout[(std::size_t)id].a == (int)(id * 3 + 1) &&
           pout[(std::size_t)id].b == (float)id * 0.5f;
    if (!allRanks(ok)) {
      if (rank == 0)
        std::fprintf(stderr, "gatherByGlobalId FAILED at nGlobal=%lld\n", nGlobal);
      ++::peclet::core::test::g_failures;
    }
  }
  // a duplicated id (and hence a missing one) throws on every rank
  std::vector<long long> ids{0};
  std::vector<double> vals{1.0};
  bool threw = false;
  std::vector<double> out;
  try {
    gatherByGlobalId(ids, vals, (long long)nranks, out, MPI_COMM_WORLD);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!allRanks(threw == (nranks > 1))) {
    if (rank == 0)
      std::fprintf(stderr, "gatherByGlobalId: duplicate ids not rejected\n");
    ++::peclet::core::test::g_failures;
  }
}

// ---- 6 : layout-agnostic, 3-D and 2-D
// ------------------------------------------------------------
template <int Dim>
bool permutedRoundTrip(const BlockDecomposer<Dim>& cur, const StageTarget<Dim>& t,
                       std::uint64_t seed) {
  StageComm c = makeStageComm(MPI_COMM_WORLD, t);
  const IVec<Dim> G = cur.globalSize();
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
  auto cells = [](const Block<Dim>& b) {
    Index n = 1;
    for (int d = 0; d < Dim; ++d)
      n *= b.size[d];
    return n;
  };
  const Block<Dim> sb = cur.block((std::size_t)rank);
  const int tb = (t.kind == StageKind::Replicated) ? 0 : c.myGroup;
  const Block<Dim> db = t.dec.block((std::size_t)tb);
  // slots: a random permutation of a buffer 3x the block (so slots are sparse, not a prefix)
  const std::vector<Index> sp = permutation((std::size_t)(3 * cells(sb)), seed + 1000 + rank);
  const std::vector<Index> dpm = permutation((std::size_t)(3 * cells(db)), seed + 2000 + tb);
  auto srcIndex = [&](const IVec<Dim>& gc) { return sp[(std::size_t)lin(gc, sb)]; };
  auto dstIndex = [&](const IVec<Dim>& gc) { return dpm[(std::size_t)lin(gc, db)]; };
  RedistributeTopology<Dim, double> topo;
  topo.build(cur, t, c, srcIndex, dstIndex);
  std::vector<double> src((std::size_t)(3 * cells(sb)), 5.0), dst;
  if (c.active)
    dst.assign((std::size_t)(3 * cells(db)), -5.0);
  IVec<Dim> end{};
  for (int d = 0; d < Dim; ++d)
    end[d] = sb.origin[d] + sb.size[d];
  peclet::core::forEachInBox<Dim>(sb.origin, end, [&](const IVec<Dim>& gc) {
    src[(std::size_t)srcIndex(gc)] = valueOf(seed, (std::uint64_t)glin(gc));
  });
  std::vector<double*> dp;
  std::vector<const double*> dcp;
  if (c.active) {
    dp.push_back(dst.data());
    dcp.push_back(dst.data());
  }
  topo.forward({src.data()}, dp);
  bool ok = true;
  if (c.active) {
    std::vector<double> expect((std::size_t)(3 * cells(db)), -5.0);
    for (int d = 0; d < Dim; ++d)
      end[d] = db.origin[d] + db.size[d];
    peclet::core::forEachInBox<Dim>(db.origin, end, [&](const IVec<Dim>& gc) {
      expect[(std::size_t)dstIndex(gc)] = valueOf(seed, (std::uint64_t)glin(gc));
    });
    ok = sameBits(dst, expect);
  }
  std::vector<double> back((std::size_t)(3 * cells(sb)), 5.0);
  topo.backward(dcp, {back.data()});
  return ok && sameBits(back, src);
}

void testLayoutAgnostic() {
  for (const auto& [name, cur] : partitions()) {
    for (const StageTarget<3>& t : siblingTargets(cur)) {
      ++gCases;
      if (!allRanks(permutedRoundTrip<3>(cur, t, 31))) {
        if (rank == 0)
          std::fprintf(stderr, "permuted sibling merge FAILED: %s\n", name.c_str());
        ++::peclet::core::test::g_failures;
      }
    }
    StageTarget<3> r;
    r.kind = StageKind::Replicated;
    r.dec.init(1, cur.globalSize());
    r.ownerOf = {0};
    ++gCases;
    if (!allRanks(permutedRoundTrip<3>(cur, r, 37))) {
      if (rank == 0)
        std::fprintf(stderr, "permuted replicated FAILED: %s\n", name.c_str());
      ++::peclet::core::test::g_failures;
    }
  }
  const BlockDecomposer<2> cur2((std::size_t)nranks, IVec<2>{26, 18});
  for (int d = cur2.treeDepth(); d >= 0; --d) {
    StageTarget<2> t;
    t.kind = StageKind::SiblingMerge;
    t.dec = cur2.agglomerated(d, &t.groupOf, &t.ownerOf);
    ++gCases;
    if (!allRanks(permutedRoundTrip<2>(cur2, t, 43))) {
      if (rank == 0)
        std::fprintf(stderr, "2-D permuted sibling merge FAILED at depth %d\n", d);
      ++::peclet::core::test::g_failures;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  try {
    testSiblingMerge();
    testReplicated();
    testGatherByGlobalId();
    testLayoutAgnostic();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "rank %d: exception: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int fails = ::peclet::core::test::g_failures, all = 0;
  MPI_Allreduce(&fails, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("np=%d: %d stage cases, %s\n", nranks, gCases, all == 0 ? "OK" : "FAILED");
  MPI_Finalize();
  return all == 0 ? 0 : 1;
}

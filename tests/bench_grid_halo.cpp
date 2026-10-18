// bench_grid_halo — GridHalo cost per exchange on a multigrid level hierarchy (flow
// doc/vof_distributed_host_comm_design.md §6 M2). A measurement instrument, not a gate (label
// `bench`): it prints one line per (level, mode) and checks only that every ghost cell received the
// owner's value.
//
// The hierarchy is the one flow's CutcellMG builds for a global grid (default 128x96x64, the bubble
// column): level 0 is the aligned ORB of the global grid over all ranks (coarsenAlignment, as
// CutcellMG::decomposition), every coarse level is the previous one coarsened IN PLACE by 2 on each
// axis whose blocks are all even. Ghost width 1, every axis periodic (the halo knows nothing about
// domain BCs), MPI_COMM_WORLD.
//
// Modes per level:
//   b2b   exchangeBegin -> exchangeEnd, back to back;
//   D=x   exchangeBegin -> a synthetic interior kernel that keeps every thread busy for x us ->
//         exchangeEnd (does the wait fall with D? if the transfer only progresses inside MPI, no).
// Per exchange and rank: begin (self-copy + pack + post), wait (exchangeEnd entry -> the unpack
// kernel's begin, seen through a Kokkos Tools callback), unpack (rest of exchangeEnd). lateness =
// max over the receive partners of (partner's post-complete time - own wait entry), on the
// node-wide steady clock; protocol = wait - max(0, lateness): the part of the wait not explained by
// a partner that had not posted yet. `late_post` / `prot_post` measure lateness from the own post
// instead.
//
//   mpirun -np P [--bind-to none + explicit pinning] bench_grid_halo [--gsize X Y Z] [--levels N]
//          [--reps R] [--warm W] [--D 0,20,50,100] [--tag T]
// --tag selects the exchange path (GridHalo passes it to the messages; a nonzero tag keeps the
// Isend path once persistent requests exist). The MPI library's eager limit is varied from the
// launcher's environment (e.g. OMPI_MCA_btl_sm_eager_limit), a diagnostic only.
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::GridHalo;
using peclet::core::halo::GridHaloTopology;

namespace {

double nowUs() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Kokkos Tools callback: the time the unpack kernel of the current exchangeEnd begins.
double gUnpackBegin = -1.0;
void onBeginFor(const char* name, const uint32_t, uint64_t* kid) {
  *kid = 0;
  if (std::strcmp(name, "peclet::core::halo::unpack") == 0)
    gUnpackBegin = nowUs();
}
void onEndFor(const uint64_t) {}

// CutcellMG::coarsenAlignment: the per-axis power of two the ORB aligns to so the levels nest.
IVec<3> coarsenAlignment(IVec<3> g) {
  auto can = [](Index d) { return (d % 2 == 0) && (d / 2 >= 2); };
  IVec<3> a{1, 1, 1};
  for (bool any = true; any;) {
    any = false;
    for (int k = 0; k < 3; ++k)
      if (can(g[k])) {
        a[k] *= 2;
        g[k] /= 2;
        any = true;
      }
  }
  for (int k = 0; k < 3; ++k)  // CutcellMG caps the alignment at 16 (2^(default nLevels - 1))
    if (a[k] > 16)
      a[k] = 16;
  return a;
}

double mean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v)
    s += x;
  return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}
double median(std::vector<double> v) {
  if (v.empty())
    return 0.0;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    IVec<3> gsize{128, 96, 64};
    int nLevels = 4, reps = 2000, warm = 200, tag = 0;
    std::vector<double> Ds{0, 20, 50, 100};
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--gsize" && i + 3 < argc) {
        for (int k = 0; k < 3; ++k)
          gsize[k] = std::atol(argv[++i]);
      } else if (a == "--levels" && i + 1 < argc) {
        nLevels = std::atoi(argv[++i]);
      } else if (a == "--reps" && i + 1 < argc) {
        reps = std::atoi(argv[++i]);
      } else if (a == "--warm" && i + 1 < argc) {
        warm = std::atoi(argv[++i]);
      } else if (a == "--tag" && i + 1 < argc) {
        tag = std::atoi(argv[++i]);
      } else if (a == "--D" && i + 1 < argc) {
        Ds.clear();
        std::string s = argv[++i];
        for (std::size_t p = 0; p <= s.size();) {
          std::size_t q = s.find(',', p);
          if (q == std::string::npos)
            q = s.size();
          if (q > p)
            Ds.push_back(std::atof(s.substr(p, q - p).c_str()));
          p = q + 1;
        }
      }
    }
    Kokkos::Tools::Experimental::set_begin_parallel_for_callback(onBeginFor);
    Kokkos::Tools::Experimental::set_end_parallel_for_callback(onEndFor);
    const int T = Kokkos::DefaultHostExecutionSpace().concurrency();
    if (rank == 0)
      std::printf(
          "bench_grid_halo np=%d threads/rank=%d gsize=%ldx%ldx%ld reps=%d warm=%d tag=%d\n", size,
          T, (long)gsize[0], (long)gsize[1], (long)gsize[2], reps, warm, tag);

    BlockDecomposer<3> dec(static_cast<std::size_t>(size), gsize, coarsenAlignment(gsize));
    const std::array<bool, 3> per{true, true, true};
    for (int L = 0; L < nLevels; ++L) {
      if (L > 0) {
        IVec<3> r{1, 1, 1};
        for (int k = 0; k < 3; ++k) {
          bool even = true;
          for (std::size_t b = 0; b < dec.sizes().size(); ++b)
            even = even && dec.sizes()[b][k] % 2 == 0 && dec.origins()[b][k] % 2 == 0;
          if (even)
            r[k] = 2;
        }
        if (r[0] == 1 && r[1] == 1 && r[2] == 1)
          break;
        dec = dec.coarsened(r);
      }
      GridHaloTopology<3> topo;
      topo.buildTopology(dec, rank, /*ghost=*/1, per, MPI_COMM_WORLD);
      const auto flat = topo.flatten();
      const auto& idx = topo.indexer();
      const Index n = idx.numCellsInclGhost();
      long sendB = 0;
      for (int c : flat.sendCounts)
        sendB += c * static_cast<long>(sizeof(double));
      const int nSend = static_cast<int>(flat.sendRanks.size());
      const int nRecv = static_cast<int>(flat.recvRanks.size());
      std::vector<int> partners(flat.recvRanks.begin(), flat.recvRanks.end());
      std::sort(partners.begin(), partners.end());
      partners.erase(std::unique(partners.begin(), partners.end()), partners.end());
      partners.erase(std::remove(partners.begin(), partners.end(), rank), partners.end());

      // Field: inner = global linear index, ghosts = -1; the exchange must fill every ghost with
      // it.
      View<double> f(Kokkos::view_alloc("bench::field", Kokkos::WithoutInitializing),
                     static_cast<std::size_t>(n));
      auto hf = Kokkos::create_mirror_view(f);
      for (Index c = 0; c < n; ++c)
        hf(c) = -1.0;
      idx.forEachInner([&](const IVec<3>& lmd) {
        IVec<3> g{};
        for (int k = 0; k < 3; ++k)
          g[k] = lmd[k] + idx.originInclGhost()[k];
        hf(idx.localMdToLocal(lmd)) = static_cast<double>(dec.linearGlobal(g));
      });
      Kokkos::deep_copy(f, hf);
      GridHalo<double> halo;
      halo.init(topo);
      halo.exchange(f, tag);
      Kokkos::deep_copy(hf, f);
      int bad = 0;
      const IVec<3> eg = idx.sizeInclGhost();
      idx.forEachAll([&](const IVec<3>& lmd) {
        bool inner = true;
        for (int k = 0; k < 3; ++k)
          inner = inner && lmd[k] >= 1 && lmd[k] < eg[k] - 1;
        if (inner)
          return;
        IVec<3> g{};
        for (int k = 0; k < 3; ++k) {
          g[k] = lmd[k] + idx.originInclGhost()[k];
          g[k] = ((g[k] % dec.globalSize()[k]) + dec.globalSize()[k]) % dec.globalSize()[k];
        }
        if (hf(idx.localMdToLocal(lmd)) != static_cast<double>(dec.linearGlobal(g)))
          ++bad;
      });
      fail += bad;

      const auto bs = dec.sizes()[static_cast<std::size_t>(rank)];
      std::vector<double> modes{-1.0};  // -1 = back to back
      modes.insert(modes.end(), Ds.begin(), Ds.end());
      for (double D : modes) {
        std::vector<double> tBeg(reps), tWait(reps), tUnp(reps), tPost(reps), tWait0(reps),
            tTot(reps);
        MPI_Barrier(MPI_COMM_WORLD);
        for (int it = -warm; it < reps; ++it) {
          const double t0 = nowUs();
          halo.exchangeBegin(f, tag);
          const double t1 = nowUs();
          if (D >= 0) {
            const double until = t1 + D;
            // Host threads busy for D us, nobody inside MPI (as during flow's interior sweep).
            Kokkos::parallel_for("bench::interior",
                                 Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, T),
                                 [=](const int) {
                                   while (nowUs() < until) {
                                   }
                                 });
          }
          const double t2 = nowUs();
          gUnpackBegin = -1.0;
          halo.exchangeEnd(f);
          const double t3 = nowUs();
          if (it < 0)
            continue;
          const double tu = gUnpackBegin > 0 ? gUnpackBegin : t3;
          tBeg[it] = t1 - t0;
          tWait[it] = tu - t2;
          tUnp[it] = t3 - tu;
          tPost[it] = t1;
          tWait0[it] = t2;
          tTot[it] = (t1 - t0) + (t3 - t2);
        }
        // Partners' post times: everyone does the same exchanges in the same order.
        std::vector<double> all(static_cast<std::size_t>(reps) * size);
        MPI_Allgather(tPost.data(), reps, MPI_DOUBLE, all.data(), reps, MPI_DOUBLE, MPI_COMM_WORLD);
        std::vector<double> late(reps), prot(reps), lateP(reps), protP(reps);
        for (int it = 0; it < reps; ++it) {
          double mx = -1e300;
          for (int p : partners)
            mx = std::max(mx, all[static_cast<std::size_t>(p) * reps + it]);
          if (partners.empty())
            mx = tPost[it];
          late[it] = mx - tWait0[it];
          prot[it] = tWait[it] - std::max(0.0, late[it]);
          lateP[it] = mx - tPost[it];
          protP[it] = tWait[it] - std::max(0.0, lateP[it]);
        }
        // Per rank: means (they add up to time per exchange); then mean / max over ranks.
        double loc[9] = {mean(tTot), mean(tBeg),  mean(tWait), mean(tUnp),  mean(late),
                         mean(prot), mean(lateP), mean(protP), median(tTot)};
        double sum[9], mx[9];
        MPI_Reduce(loc, sum, 9, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(loc, mx, 9, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        int nb[3] = {nSend, nRecv, static_cast<int>(sendB)}, nbx[3];
        MPI_Reduce(nb, nbx, 3, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
          char mode[32];
          if (D < 0)
            std::snprintf(mode, sizeof mode, "b2b");
          else
            std::snprintf(mode, sizeof mode, "D=%g", D);
          const double P = size;
          std::printf(
              "HALO L%d %-6s np=%d T=%d block0=%ldx%ldx%ld sends=%d recvs=%d sendB0=%ld "
              "maxsends=%d "
              "maxsendB=%d | us/exch mean %.2f [max-rank %.2f] med %.2f | begin %.2f wait %.2f "
              "[max %.2f] unpack %.2f | late %.2f prot %.2f [max %.2f] | late_post %.2f "
              "prot_post %.2f\n",
              L, mode, size, T, (long)bs[0], (long)bs[1], (long)bs[2], nSend, nRecv, sendB, nbx[0],
              nbx[2], sum[0] / P, mx[0], sum[8] / P, sum[1] / P, sum[2] / P, mx[2], sum[3] / P,
              sum[4] / P, sum[5] / P, mx[5], sum[6] / P, sum[7] / P);
          std::fflush(stdout);
        }
      }
    }
    int tot = 0;
    MPI_Allreduce(&fail, &tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    fail = tot;
    if (rank == 0)
      std::printf("bench_grid_halo: %s (%d wrong ghost values)\n", fail ? "FAIL" : "ok", fail);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail ? 1 : 0;
}

// Composition capstone: distributed scalar diffusion AROUND an SDF-described solid obstacle.
//
// This exercises geometry + decomposition + halo together — the suite's actual purpose (transport
// of a field through a domain containing solids). A sphere (peclet::core::geom, negative inside)
// sits in the middle of a periodic box; cells inside the solid are held at zero (a simple Dirichlet
// immersed boundary), fluid cells diffuse with a 7-point stencil. The distributed run (block
// decomposition + GridHaloTopology each step) must match an independent serial integration
// cell-for-cell.
#include <mpi.h>

#include <array>
#include <cstdio>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/geom/sdf.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"

using namespace peclet::core;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::geom::Sphere;
using peclet::core::halo::GridFieldView;
using peclet::core::halo::GridHaloTopology;

static constexpr int kDim = 3;
static constexpr double kCoeff = 0.1;
static constexpr int kSteps = 40;

static double initField(const BlockDecomposer<kDim>& dec, const IVec<kDim>& g) {
  std::uint64_t h = static_cast<std::uint64_t>(dec.linearGlobal(g)) * 2654435761u + 999u;
  h ^= h >> 33;
  return static_cast<double>(h & 0xFFFFFFu) / static_cast<double>(0x1000000u);
}

// Cell center is at integer coord + 0.5 (cell-centered field; unit spacing, origin 0).
static bool isSolid(const Sphere& s, const IVec<kDim>& g) {
  Vec<3> c{static_cast<double>(g[0]) + 0.5, static_cast<double>(g[1]) + 0.5,
           static_cast<double>(g[2]) + 0.5};
  return s.eval(c) < 0.0;
}

static std::vector<double> serialReference(const BlockDecomposer<kDim>& dec, const Sphere& s) {
  const IVec<kDim>& n = dec.globalSize();
  Index total = 1;
  for (int i = 0; i < kDim; ++i)
    total *= n[i];
  std::vector<double> u(total), un(total);
  IVec<kDim> bgn{}, end{};
  for (int i = 0; i < kDim; ++i)
    end[i] = n[i];
  forEachInBox<kDim>(bgn, end, [&](const IVec<kDim>& g) {
    u[dec.linearGlobal(g)] = isSolid(s, g) ? 0.0 : initField(dec, g);
  });
  for (int step = 0; step < kSteps; ++step) {
    forEachInBox<kDim>(bgn, end, [&](const IVec<kDim>& g) {
      Index c = dec.linearGlobal(g);
      if (isSolid(s, g)) {
        un[c] = 0.0;
        return;
      }
      double lap = 0.0;
      for (int i = 0; i < kDim; ++i) {
        IVec<kDim> gp = g, gm = g;
        gp[i] = wrap(g[i] + 1, n[i]);
        gm[i] = wrap(g[i] - 1, n[i]);
        lap += u[dec.linearGlobal(gp)] + u[dec.linearGlobal(gm)];
      }
      lap -= 2.0 * kDim * u[c];
      un[c] = u[c] + kCoeff * lap;
    });
    std::swap(u, un);
  }
  return u;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  IVec<kDim> gsize{32, 24, 20};
  Sphere obstacle{{16.0, 12.0, 10.0}, 6.0};
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gsize);
  std::vector<double> ref = serialReference(dec, obstacle);

  const std::array<bool, kDim> periodic{true, true, true};
  GridHaloTopology<kDim> halo;
  halo.buildTopology(dec, rank, /*ghost=*/1, periodic, MPI_COMM_WORLD);
  const auto& idx = halo.indexer();

  std::vector<double> u(idx.numCellsInclGhost(), 0.0), un(idx.numCellsInclGhost(), 0.0);
  idx.forEachInner([&](const IVec<kDim>& lmd) {
    IVec<kDim> g{};
    for (int i = 0; i < kDim; ++i)
      g[i] = lmd[i] + idx.originInclGhost()[i];
    u[idx.localMdToLocal(lmd)] = isSolid(obstacle, g) ? 0.0 : initField(dec, g);
  });

  for (int step = 0; step < kSteps; ++step) {
    GridFieldView<double> fv{u.data()};
    halo.exchangePersistent(fv);
    idx.forEachInner([&](const IVec<kDim>& lmd) {
      IVec<kDim> g{};
      for (int i = 0; i < kDim; ++i)
        g[i] = lmd[i] + idx.originInclGhost()[i];
      Index c = idx.localMdToLocal(lmd);
      if (isSolid(obstacle, g)) {
        un[c] = 0.0;
        return;
      }
      double lap = 0.0;
      for (int i = 0; i < kDim; ++i) {
        IVec<kDim> lp = lmd, lm = lmd;
        lp[i] += 1;
        lm[i] -= 1;
        lap += u[idx.localMdToLocal(lp)] + u[idx.localMdToLocal(lm)];
      }
      lap -= 2.0 * kDim * u[c];
      un[c] = u[c] + kCoeff * lap;
    });
    std::swap(u, un);
  }

  int fail = 0;
  idx.forEachInner([&](const IVec<kDim>& lmd) {
    IVec<kDim> g{};
    for (int i = 0; i < kDim; ++i)
      g[i] = lmd[i] + idx.originInclGhost()[i];
    if (std::fabs(u[idx.localMdToLocal(lmd)] - ref[dec.linearGlobal(g)]) > 1e-9)
      ++fail;
  });

  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): diffusion around SDF solid matches serial over %d steps\n", size,
                  kSteps);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d\n", size, totalFail);
  }
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}

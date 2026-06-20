// MPI + Kokkos correctness of the device-driven Lagrangian ghost halo.
//
// Same setup as test_particle_halo_mpi.cpp, but run the persistent forward/reverse exchanges BOTH on
// the CPU (ParticleHalo) and on the device (DeviceParticleHaloKokkos), and require the device result
// to match the CPU result bit-for-bit:
//   forward(id)        -> each ghost carries its owner's id,
//   reverse(ones, sum) -> each owned particle accumulates a count of how many ranks ghost it.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/common/view.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_halo.hpp"
#include "tpx/halo/particle_halo_kokkos.hpp"
#include "tpx/halo/particle_migrator.hpp"

using namespace tpx;
using tpx::decomp::BlockDecomposer;
using tpx::halo::DeviceParticleHaloKokkos;
using tpx::halo::DomainMap;
using tpx::halo::ParticleHalo;
using tpx::halo::ParticleMigrator;

static double frac(std::uint64_t x, int s) {
  x ^= (std::uint64_t)s * 2654435761u;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return (double)(x & 0xFFFFFF) / (double)0x1000000;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  int size = 1;
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const double dmin[3] = {0, 0, 0}, dsize[3] = {10, 8, 6};
    IVec<3> gsize{40, 32, 24};
    const double rcut = 0.9;
    BlockDecomposer<3> dec(static_cast<std::size_t>(size), gsize);
    DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = dmin[i];
      map.cellSize[i] = dsize[i] / gsize[i];
      map.periodic[i] = true;
    }
    ParticleMigrator<3> mig;
    mig.init(dec, rank, map, MPI_COMM_WORLD);

    const std::int64_t N = 3000;
    const std::size_t stride = sizeof(std::int64_t);
    std::vector<Vec<3>> pos;
    std::vector<char> payload;
    for (std::int64_t id = rank; id < N; id += size) {
      pos.push_back({dmin[0] + frac(id, 0) * dsize[0], dmin[1] + frac(id, 1) * dsize[1],
                     dmin[2] + frac(id, 2) * dsize[2]});
      std::size_t off = payload.size();
      payload.resize(off + stride);
      std::memcpy(&payload[off], &id, stride);
    }
    mig.migrate(pos, payload, stride);
    const std::size_t Nown = pos.size();
    std::vector<double> ownId(Nown);
    for (std::size_t i = 0; i < Nown; ++i) {
      std::int64_t id;
      std::memcpy(&id, &payload[i * stride], stride);
      ownId[i] = (double)id;
    }

    ParticleHalo<3> halo;
    halo.init(mig);
    halo.build(pos, rcut);
    const std::size_t G = halo.numGhost();

    // --- CPU reference exchanges ---
    std::vector<double> ghCpu(G), ownCntCpu(Nown, 0.0), ones(G, 1.0);
    halo.forward(ownId.data(), ghCpu.data());
    halo.reverse(ones.data(), ownCntCpu.data());

    // --- device exchanges ---
    DeviceParticleHaloKokkos<3> dhalo;
    dhalo.init(halo);

    View<double> dOwn(Kokkos::view_alloc("own", Kokkos::WithoutInitializing), Nown);
    View<double> dGhost(Kokkos::view_alloc("ghost", Kokkos::WithoutInitializing), G);
    {
      auto hOwn = Kokkos::create_mirror_view(dOwn);
      for (std::size_t i = 0; i < Nown; ++i) hOwn(i) = ownId[i];
      Kokkos::deep_copy(dOwn, hOwn);
    }
    dhalo.forward(dOwn, dGhost);
    std::vector<double> ghDev(G);
    {
      auto hGhost = Kokkos::create_mirror_view(dGhost);
      Kokkos::deep_copy(hGhost, dGhost);
      for (std::size_t i = 0; i < G; ++i) ghDev[i] = hGhost(i);
    }

    View<double> dGones(Kokkos::view_alloc("gones", Kokkos::WithoutInitializing), G);
    View<double> dOwnCnt("owncnt", Nown);  // zero-initialised
    Kokkos::deep_copy(dGones, 1.0);
    dhalo.reverse(dGones, dOwnCnt);
    std::vector<double> ownCntDev(Nown);
    {
      auto hCnt = Kokkos::create_mirror_view(dOwnCnt);
      Kokkos::deep_copy(hCnt, dOwnCnt);
      for (std::size_t i = 0; i < Nown; ++i) ownCntDev[i] = hCnt(i);
    }

    // --- device must match CPU bit-for-bit ---
    for (std::size_t i = 0; i < G; ++i)
      if (ghDev[i] != ghCpu[i]) ++fail;
    for (std::size_t i = 0; i < Nown; ++i)
      if (ownCntDev[i] != ownCntCpu[i]) ++fail;
  }

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (total == 0)
      std::printf("OK (np=%d): Kokkos particle forward/reverse match CPU ParticleHalo\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d mismatches\n", size, total);
  }

  Kokkos::finalize();
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}

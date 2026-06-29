// Device-resident particle migration (tpx::halo::ParticleMigratorDevice, D1): particles live in device
// Views; the periodic wrap + ORB owner lookup run on device, departing particles are device-packed into
// compact buffers, only those cross MPI (NBX), arrivals unpack back on device. Same correctness contract
// as the host migrator (test_particle_migration): over random-walk + migrate steps, globally and every
// step — count is conserved (== N), every local particle is owned by this rank, and the id multiset is
// exactly {0..N-1} (SUM + XOR reductions). np = 1,2,4,8.
//
// Guarded by TPX_HAVE_MORTON (for the Kokkos-on-MPI build wiring); a no-op otherwise.
#include "test_util.hpp"

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "tpx/common/mpi.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_migrator.hpp"
#include "tpx/halo/particle_migrator_device.hpp"

using namespace tpx;
using namespace tpx::halo;

namespace {

constexpr int Dim = 3;
constexpr Index N = 512;        // global particle count
constexpr Index G = 8;          // 8^3 cell grid, [0,8)^3 periodic, unit cells

// Deterministic per-(id,step) pseudo-random displacement in (-0.5, 0.5) per axis.
KOKKOS_INLINE_FUNCTION double jitter(std::uint64_t id, int step, int axis) {
  std::uint64_t s = id * 0x9e3779b97f4a7c15ULL + static_cast<std::uint64_t>(step) * 0x100000001b3ULL +
                    static_cast<std::uint64_t>(axis) * 0xff51afd7ed558ccdULL;
  s ^= s >> 33;
  s *= 0xff51afd7ed558ccdULL;
  s ^= s >> 29;
  return static_cast<double>((s >> 11) & 0xFFFFF) / static_cast<double>(0x100000) - 0.5;
}

void run() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  decomp::BlockDecomposer<Dim> dec(static_cast<std::size_t>(size), IVec<Dim>{G, G, G});
  DomainMap<Dim> map;
  for (int d = 0; d < Dim; ++d) {
    map.origin[d] = 0.0;
    map.cellSize[d] = 1.0;
    map.periodic[d] = true;
  }
  ParticleMigratorDevice<Dim> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);
  ParticleMigrator<Dim> hmig;  // host reference for the ownerOf check
  hmig.init(dec, rank, map, MPI_COMM_WORLD);

  const std::size_t stride = sizeof(std::int64_t);  // payload = the particle id
  // Capacity N: a rank can hold at most all N particles after migration.
  View<double> pos("pos", static_cast<std::size_t>(N) * Dim);
  View<char> pay("pay", static_cast<std::size_t>(N) * stride);

  // Seed: ids rank, rank+size, ... start scattered; positions hashed from the id.
  std::vector<double> ip;
  std::vector<char> iy;
  Index cnt0 = 0;
  for (std::int64_t id = rank; id < N; id += size) {
    for (int d = 0; d < Dim; ++d) {
      std::uint64_t s = static_cast<std::uint64_t>(id) * 7 + static_cast<std::uint64_t>(d) * 131;
      s ^= s >> 31;
      ip.push_back(static_cast<double>(s % (G * 1000)) / 1000.0);  // in [0,G)
    }
    char b[sizeof(std::int64_t)];
    std::memcpy(b, &id, stride);
    for (std::size_t k = 0; k < stride; ++k) iy.push_back(b[k]);
    ++cnt0;
  }
  Kokkos::deep_copy(Kokkos::subview(pos, std::pair<std::size_t, std::size_t>(0, ip.size())),
                    Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        ip.data(), ip.size()));
  Kokkos::deep_copy(Kokkos::subview(pay, std::pair<std::size_t, std::size_t>(0, iy.size())),
                    Kokkos::View<const char*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        iy.data(), iy.size()));
  Index n = cnt0;

  // Expected invariants of the full id set {0..N-1}.
  std::int64_t expSum = N * (N - 1) / 2, expXor = 0;
  for (std::int64_t id = 0; id < N; ++id) expXor ^= id;

  for (int step = 0; step < 6; ++step) {
    // random-walk on device, then migrate on device.
    Kokkos::parallel_for(
        "walk", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const Index i) {
          std::uint64_t id = 0;  // reconstruct the little-endian id from the byte payload (device-safe)
          for (std::size_t k = 0; k < stride; ++k)
            id |= static_cast<std::uint64_t>(static_cast<unsigned char>(pay(i * stride + k))) << (8 * k);
          for (int d = 0; d < Dim; ++d) pos(i * Dim + d) += jitter(id, step, d);
        });
    n = mig.migrate(pos, pay, n, stride);

    // (a) count conserved.
    long ln = static_cast<long>(n), gtot = 0;
    MPI_Allreduce(&ln, &gtot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    TPX_CHECK_EQ(gtot, static_cast<long>(N));

    // download for the per-particle checks.
    std::vector<double> hp(static_cast<std::size_t>(n) * Dim);
    std::vector<std::int64_t> hid(static_cast<std::size_t>(n));
    {
      auto mp = Kokkos::create_mirror_view(pos);
      auto my = Kokkos::create_mirror_view(pay);
      Kokkos::deep_copy(mp, pos);
      Kokkos::deep_copy(my, pay);
      for (Index i = 0; i < n; ++i) {
        for (int d = 0; d < Dim; ++d) hp[static_cast<std::size_t>(i) * Dim + d] = mp(i * Dim + d);
        std::memcpy(&hid[static_cast<std::size_t>(i)], &my(i * stride), stride);
      }
    }
    // (b) every local particle owned by this rank.
    int fail = 0;
    std::int64_t localSum = 0, localXor = 0;
    for (Index i = 0; i < n; ++i) {
      Vec<Dim> x;
      for (int d = 0; d < Dim; ++d) x[d] = hp[static_cast<std::size_t>(i) * Dim + d];
      if (hmig.ownerOf(x) != rank) ++fail;
      localSum += hid[static_cast<std::size_t>(i)];
      localXor ^= hid[static_cast<std::size_t>(i)];
    }
    TPX_CHECK_EQ(fail, 0);
    // (c) id multiset == {0..N-1}.
    std::int64_t gSum = 0, gXor = 0;
    MPI_Allreduce(&localSum, &gSum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localXor, &gXor, 1, MPI_INT64_T, MPI_BXOR, MPI_COMM_WORLD);
    TPX_CHECK_EQ(gSum, expSum);
    TPX_CHECK_EQ(gXor, expXor);
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int fails = tpx::test::g_failures, total = 0;
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

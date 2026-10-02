// AndersonCore under MPI (flow doc/steady_acceleration.md §8 U7, §6.1 determinism contract).
//
// U1's linear contraction on a 25×20×20 global box, split into z-slabs over np = 1, 2, 4 ranks, is
// accelerated with the collectives of solver/anderson_mpi.hpp and compared against a serial run of
// the same map on the whole box (an AndersonCore with an empty AndersonComm, in the same process):
//   - γ, the column count and the status are bitwise equal on every rank after every step;
//   - after 20 steps the iterates agree with the serial run to 1e-13 (relative to max|x|), and at
//     np = 1 they are bit-identical (an Allreduce / Bcast on one rank is the identity).
// A second case adds two Carried fields (mixed with the same γ on every rank, never measured).
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "anderson_test_maps.hpp"
#include "peclet/core/solver/anderson.hpp"
#include "peclet/core/solver/anderson_mpi.hpp"
#include "test_util.hpp"

using peclet::core::Index;
using peclet::core::View;
using peclet::core::solver::AndersonCore;
using peclet::core::solver::AndersonRole;
using peclet::core::solver::AndersonState;
using peclet::core::test::applyLinear;
using peclet::core::test::Box;
using peclet::core::test::download;
using peclet::core::test::LinearMap;
using peclet::core::test::makeLinearMap;

namespace {

constexpr Index kNx = 25, kNy = 20, kNz = 20;  // 10^4 global inner entries
constexpr int kSteps = 20;

/// Per-field value at global inner cell (x, y, z): the U1 spectrum for field 0, and fixed
/// spectra / sources / SDF signs for the extra fields of case B.
double lamOf(int field, Index x, Index y, Index z) {
  const Index k = x + kNx * (y + kNy * z);
  const double s = static_cast<double>(k) / static_cast<double>(kNx * kNy * kNz - 1);
  if (field == 0)
    return 0.996 * s;
  if (field == 1)
    return 0.95 * std::fmod(0.618034 * static_cast<double>(k), 1.0);
  return 0.5 * s;
}
double srcOf(int field, Index x, Index y, Index z) {
  if (field == 0)
    return 1.0;
  return 1.0 + 0.3 * std::sin(0.11 * static_cast<double>(x + 3 * y + 7 * z) + field);
}

/// The state of a z-slab [z0, z0 + nz) of the global box, with its own padded box and maps.
struct Slab {
  Box box;
  Index z0;
  std::vector<View<double>> fields;
  std::vector<LinearMap> maps;
  Slab(int nFields, Index z0_, Index nz) : box(kNx, kNy, nz, 2), z0(z0_) {
    for (int f = 0; f < nFields; ++f) {
      std::vector<double> lam(static_cast<std::size_t>(box.nPad), 0.0),
          c(static_cast<std::size_t>(box.nPad), 1.0);
      for (Index z = 0; z < nz; ++z)
        for (Index y = 0; y < kNy; ++y)
          for (Index x = 0; x < kNx; ++x) {
            const auto i = static_cast<std::size_t>(box.pad(x, y, z));
            lam[i] = lamOf(f, x, y, z0 + z);
            c[i] = srcOf(f, x, y, z0 + z);
          }
      maps.push_back(makeLinearMap(lam, c));
      fields.push_back(View<double>("slab_field", static_cast<std::size_t>(box.nPad)));
    }
  }
  AndersonState state(const std::vector<AndersonRole>& roles) const {
    AndersonState s;
    s.fields = fields;
    s.roles = roles;
    s.extent = box.e;
    s.ghost = box.g;
    return s;
  }
  /// FNV-1a over every iterate (all fields, padded) and γ / residual / columns / status after
  /// each step: printed per rank, so two builds can be compared bit for bit.
  std::uint64_t digest = 1469598103934665603ull;
  void add(const void* p, std::size_t n) {
    const auto* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i)
      digest = (digest ^ c[i]) * 1099511628211ull;
  }
  void step(AndersonCore& acc) {
    acc.prepare(true);
    for (std::size_t f = 0; f < fields.size(); ++f)
      applyLinear(maps[f], fields[f]);
    acc.complete();
    for (const auto& f : fields) {
      const auto v = download(f);
      add(v.data(), v.size() * sizeof(double));
    }
    const auto g = acc.gamma();
    add(g.data(), g.size() * sizeof(double));
    const double res = acc.residual();
    const int cols = acc.numColumns(), status = static_cast<int>(acc.status());
    add(&res, sizeof res);
    add(&cols, sizeof cols);
    add(&status, sizeof status);
  }
};

bool sameOnAllRanks(const void* data, int bytes, MPI_Comm comm) {
  std::vector<char> mine(static_cast<const char*>(data), static_cast<const char*>(data) + bytes);
  std::vector<char> root = mine;
  MPI_Bcast(root.data(), bytes, MPI_BYTE, 0, comm);
  int differ = std::memcmp(root.data(), mine.data(), static_cast<std::size_t>(bytes)) != 0 ? 1 : 0;
  int any = 0;
  MPI_Allreduce(&differ, &any, 1, MPI_INT, MPI_MAX, comm);
  return any == 0;
}

void runCase(const char* name, const std::vector<AndersonRole>& roles, int rank, int size) {
  const int nf = static_cast<int>(roles.size());
  const Index nzLocal = kNz / size;
  Slab dist(nf, rank * nzLocal, nzLocal);
  Slab serial(nf, 0, kNz);
  AndersonState ds = dist.state(roles);
  ds.comm = peclet::core::solver::andersonComm(MPI_COMM_WORLD);
  AndersonCore accD(ds, 5);
  AndersonCore accS(serial.state(roles), 5);

  bool consistent = true;
  for (int k = 0; k < kSteps; ++k) {
    dist.step(accD);
    serial.step(accS);
    const std::vector<double> g = accD.gamma();
    double packet[2 + AndersonCore::kMaxWindow] = {};
    packet[0] = accD.numColumns();
    packet[1] = static_cast<double>(accD.status());
    for (std::size_t j = 0; j < g.size(); ++j)
      packet[2 + j] = g[j];
    consistent = consistent && sameOnAllRanks(packet, sizeof packet, MPI_COMM_WORLD);
  }

  // Iterates against the serial run (inner entries of this slab).
  double maxDiff = 0.0, maxRef = 0.0;
  for (int f = 0; f < nf; ++f) {
    const auto hd = download(dist.fields[static_cast<std::size_t>(f)]);
    const auto hs = download(serial.fields[static_cast<std::size_t>(f)]);
    for (Index z = 0; z < nzLocal; ++z)
      for (Index y = 0; y < kNy; ++y)
        for (Index x = 0; x < kNx; ++x) {
          const double a = hd[static_cast<std::size_t>(dist.box.pad(x, y, z))];
          const double b = hs[static_cast<std::size_t>(serial.box.pad(x, y, dist.z0 + z))];
          maxDiff = std::max(maxDiff, std::abs(a - b));
          maxRef = std::max(maxRef, std::abs(b));
        }
  }
  double loc[2] = {maxDiff, maxRef}, glob[2] = {0.0, 0.0};
  MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(
        "U7 %s np=%d: %d steps, residual %.6e (serial %.6e), columns %d, gamma equal on all "
        "ranks %d, max|x_np - x_1| / max|x_1| = %.3e\n",
        name, size, kSteps, accD.residual(), accS.residual(), accD.numColumns(), consistent ? 1 : 0,
        glob[0] / glob[1]);
  std::vector<unsigned long long> all(static_cast<std::size_t>(size));
  const unsigned long long mine = dist.digest;
  MPI_Gather(&mine, 1, MPI_UNSIGNED_LONG_LONG, all.data(), 1, MPI_UNSIGNED_LONG_LONG, 0,
             MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("digest U7 %s np=%d serial %016llx ranks", name, size,
                static_cast<unsigned long long>(serial.digest));
    for (const unsigned long long d : all)
      std::printf(" %016llx", d);
    std::printf("\n");
  }
  PECLET_CORE_CHECK(consistent);
  PECLET_CORE_CHECK(accD.status() == AndersonCore::Status::Active);
  PECLET_CORE_CHECK(glob[0] <= 1e-13 * glob[1]);
  if (size == 1) {
    PECLET_CORE_CHECK(glob[0] == 0.0);  // np = 1 is bit-identical to serial
    PECLET_CORE_CHECK(accD.residual() == accS.residual());
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int failures = 0;
  Kokkos::initialize(argc, argv);
  {
    if (kNz % size != 0) {
      if (rank == 0)
        std::fprintf(stderr, "test_anderson_mpi: np must divide %lld\n", (long long)kNz);
      peclet::core::test::g_failures = 1;
    } else {
      runCase("U1", {AndersonRole::Velocity}, rank, size);
      runCase("velocity+carried",
              {AndersonRole::Velocity, AndersonRole::Carried, AndersonRole::Carried}, rank, size);
    }
    failures = peclet::core::test::g_failures;
  }
  Kokkos::finalize();
  int any = 0;
  MPI_Allreduce(&failures, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  if (any == 0 && rank == 0)
    std::printf("OK\n");
  return any == 0 ? 0 : 1;
}

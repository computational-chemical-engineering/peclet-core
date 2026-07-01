// transport-core — minimal Python surface for the Lagrangian halo (migration + ghosts).
//
// A nanobind module exposing tpx::halo::ParticleMigrator so an mpi4py driver can decompose a periodic
// domain and migrate/ghost particles between ranks. Particles are passed as numpy arrays: positions
// (N,3) float64 and an arbitrary per-particle payload (N,K) float64 (pack velocity, orientation, id,
// etc. into the K columns). MPI is assumed already initialized by the host (import mpi4py.MPI first);
// this module uses MPI_COMM_WORLD and never calls MPI_Init/Finalize.
//
// Host-only (no Kokkos): arrays are returned via a capsule-backed nanobind ndarray (the same
// owner-capsule idea as transport-core's Kokkos bridge, minus the device path).
// Reusable by dem and vorflow. Build: see python/CMakeLists.txt.
#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_halo_topology.hpp"
#include "tpx/halo/particle_migrator.hpp"
#include "tpx/halo/particle_rebalance.hpp"

namespace nb = nanobind;
using namespace tpx;

namespace {

// A flat row-major buffer -> (rows,cols) float64 numpy array, moved into the array's backing store.
nb::ndarray<nb::numpy, double> mat(std::vector<double>&& d, std::size_t rows, std::size_t cols) {
  auto* held = new std::vector<double>(std::move(d));
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  return nb::ndarray<nb::numpy, double>(held->data(), {rows, cols}, owner,
                                        {static_cast<std::int64_t>(cols), 1});
}

using DArray = nb::ndarray<double, nb::c_contig>;

}  // namespace

class Migrator {
 public:
  Migrator(std::array<double, 3> origin, std::array<double, 3> size, std::array<long, 3> gsize,
           std::array<bool, 3> periodic) {
    int sz = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &sz);
    dec_.init(static_cast<std::size_t>(sz), IVec<3>{gsize[0], gsize[1], gsize[2]});
    tpx::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = origin[i];
      map.cellSize[i] = size[i] / static_cast<double>(gsize[i]);
      map.periodic[i] = periodic[i];
    }
    mig_.init(dec_, rank_, map, MPI_COMM_WORLD);
  }

  int rank() const { return rank_; }
  int owner_of(std::array<double, 3> x) const { return mig_.ownerOf(Vec<3>{x[0], x[1], x[2]}); }

  // (pos (N,3), pay (N,K)) -> (pos2 (M,3), pay2 (M,K)) after reassigning ownership.
  nb::tuple migrate(DArray pos, DArray pay) {
    std::vector<Vec<3>> pv;
    std::vector<char> payload;
    std::size_t K = unpack(pos, pay, pv, payload);
    mig_.migrate(pv, payload, K * sizeof(double));
    return pack(pv, payload, K);
  }

  // Dynamic load re-balancing: re-decompose by per-block particle COUNT (weighted ORB) so each rank
  // holds a near-equal share, then migrate. (pos (N,3), pay (N,K)) -> (pos2 (M,3), pay2 (M,K)). A pure
  // redistribution — the global particle set is unchanged; only ownership moves. The decomposition is
  // updated in place, so subsequent owner_of()/migrate() calls use the new (balanced) partition.
  nb::tuple rebalance(DArray pos, DArray pay) {
    std::vector<Vec<3>> pv;
    std::vector<char> payload;
    std::size_t K = unpack(pos, pay, pv, payload);
    tpx::halo::rebalanceByParticleCount(dec_, mig_, pv, payload, K * sizeof(double), MPI_COMM_WORLD);
    return pack(pv, payload, K);
  }

  // copies of particles within rcut of this rank's block (periodic images handled).
  nb::tuple gather_ghosts(DArray pos, DArray pay, double rcut) {
    std::vector<Vec<3>> pv;
    std::vector<char> payload;
    std::size_t K = unpack(pos, pay, pv, payload);
    std::vector<Vec<3>> gpv;
    std::vector<char> gpay;
    mig_.gatherGhosts(pv, payload, K * sizeof(double), rcut, gpv, gpay);
    return pack(gpv, gpay, K);
  }

 private:
  static std::size_t unpack(const DArray& pos, const DArray& pay, std::vector<Vec<3>>& pv,
                            std::vector<char>& payload) {
    const std::size_t N = static_cast<std::size_t>(pos.shape(0));
    const std::size_t K = static_cast<std::size_t>(pay.shape(1));
    const double* P = pos.data();
    const double* Y = pay.data();
    pv.resize(N);
    payload.resize(N * K * sizeof(double));
    for (std::size_t i = 0; i < N; ++i) {
      pv[i] = {P[i * 3 + 0], P[i * 3 + 1], P[i * 3 + 2]};
      std::memcpy(&payload[i * K * sizeof(double)], &Y[i * K], K * sizeof(double));
    }
    return K;
  }
  static nb::tuple pack(const std::vector<Vec<3>>& pv, const std::vector<char>& payload,
                        std::size_t K) {
    const std::size_t M = pv.size();
    std::vector<double> op(M * 3), oy(M * K);
    for (std::size_t i = 0; i < M; ++i) {
      op[i * 3 + 0] = pv[i][0];
      op[i * 3 + 1] = pv[i][1];
      op[i * 3 + 2] = pv[i][2];
      std::memcpy(&oy[i * K], &payload[i * K * sizeof(double)], K * sizeof(double));
    }
    return nb::make_tuple(mat(std::move(op), M, 3), mat(std::move(oy), M, K));
  }

  int rank_ = 0;
  tpx::decomp::BlockDecomposer<3> dec_;
  tpx::halo::ParticleMigrator<3> mig_;
};

// Persistent owner<->ghost halo: build the correspondence once, then do cheap forward/reverse over
// the fixed topology each step (scheme C / conservative-flux exchange) instead of re-gathering full
// ghost state. Vec3 fields only (positions, velocities, forces). Wraps tpx::halo::ParticleHaloTopology.
struct V3 {
  double v[3];
  V3& operator+=(const V3& o) {
    for (int k = 0; k < 3; ++k) v[k] += o.v[k];
    return *this;
  }
};

class Halo {
 public:
  Halo(std::array<double, 3> origin, std::array<double, 3> size, std::array<long, 3> gsize,
       std::array<bool, 3> periodic) {
    int sz = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &sz);
    dec_.init(static_cast<std::size_t>(sz), IVec<3>{gsize[0], gsize[1], gsize[2]});
    tpx::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = origin[i];
      map.cellSize[i] = size[i] / static_cast<double>(gsize[i]);
      map.periodic[i] = periodic[i];
    }
    mig_.init(dec_, rank_, map, MPI_COMM_WORLD);
    halo_.init(mig_);
  }

  int rank() const { return rank_; }
  int owner_of(std::array<double, 3> x) const { return mig_.ownerOf(Vec<3>{x[0], x[1], x[2]}); }

  // (re)establish the correspondence from this rank's owned positions (N,3); returns ghost count.
  int build(DArray pos, double rcut) {
    n_owned_ = static_cast<std::size_t>(pos.shape(0));
    const double* P = pos.data();
    std::vector<Vec<3>> pv(n_owned_);
    for (std::size_t i = 0; i < n_owned_; ++i) pv[i] = {P[i * 3 + 0], P[i * 3 + 1], P[i * 3 + 2]};
    halo_.build(pv, rcut);
    n_ghost_ = halo_.numGhost();
    return static_cast<int>(n_ghost_);
  }
  int num_ghost() const { return static_cast<int>(n_ghost_); }
  int num_owned() const { return static_cast<int>(n_owned_); }

  // owned (N,3) -> ghost (G,3): xyz + periodic image shift (use for positions).
  nb::ndarray<nb::numpy, double> forward_positions(DArray owned) {
    auto o = in(owned);
    std::vector<V3> g(n_ghost_);
    halo_.forwardPositions(reinterpret_cast<Vec<3>*>(o.data()), reinterpret_cast<Vec<3>*>(g.data()));
    return out(g);
  }
  // owned (N,3) -> ghost (G,3): verbatim (velocities, ...).
  nb::ndarray<nb::numpy, double> forward(DArray owned) {
    auto o = in(owned);
    std::vector<V3> g(n_ghost_);
    halo_.forward(o.data(), g.data());
    return out(g);
  }
  // ghost (G,3) summed onto owned (N,3); returns owned + the reversed ghost contributions.
  nb::ndarray<nb::numpy, double> reverse(DArray ghost, DArray owned) {
    auto g = in(ghost);
    auto o = in(owned);  // copy; reverse() adds in place
    halo_.reverse(g.data(), o.data());
    return out(o);
  }

 private:
  static std::vector<V3> in(const DArray& a) {
    const std::size_t n = static_cast<std::size_t>(a.shape(0));
    const double* P = a.data();
    std::vector<V3> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = {{P[i * 3 + 0], P[i * 3 + 1], P[i * 3 + 2]}};
    return v;
  }
  static nb::ndarray<nb::numpy, double> out(const std::vector<V3>& v) {
    std::vector<double> d(v.size() * 3);
    for (std::size_t i = 0; i < v.size(); ++i)
      for (int k = 0; k < 3; ++k) d[i * 3 + k] = v[i].v[k];
    return mat(std::move(d), v.size(), 3);
  }
  int rank_ = 0;
  std::size_t n_owned_ = 0, n_ghost_ = 0;
  tpx::decomp::BlockDecomposer<3> dec_;
  tpx::halo::ParticleMigrator<3> mig_;
  tpx::halo::ParticleHaloTopology<3> halo_;
};

NB_MODULE(mpi, m) {
  m.attr("__doc__") = "transport-core Lagrangian halo (block decomposition + particle migration/ghosts)";
  nb::class_<Migrator>(m, "Migrator")
      .def(nb::init<std::array<double, 3>, std::array<double, 3>, std::array<long, 3>,
                    std::array<bool, 3>>(),
           nb::arg("origin"), nb::arg("size"), nb::arg("gsize"), nb::arg("periodic"))
      .def("migrate", &Migrator::migrate, nb::arg("positions"), nb::arg("payload"),
           "Reassign every particle to the rank owning its (wrapped) position; returns this rank's "
           "(positions (M,3), payload (M,K)) after the exchange.")
      .def("rebalance", &Migrator::rebalance, nb::arg("positions"), nb::arg("payload"),
           "Re-decompose by particle count (weighted ORB) so each rank holds a near-equal share, then "
           "migrate. Pure redistribution (count/payload preserved); the partition is updated in place. "
           "Returns this rank's (positions (M,3), payload (M,K)).")
      .def("gather_ghosts", &Migrator::gather_ghosts, nb::arg("positions"), nb::arg("payload"),
           nb::arg("rcut"),
           "Copies of particles within rcut of this rank's block (periodic images handled); returns the "
           "(ghost positions (G,3), ghost payload (G,K)).")
      .def("owner_of", &Migrator::owner_of, nb::arg("x"),
           "Rank that owns the block containing position x (after periodic wrap / boundary clamp).")
      .def_prop_ro("rank", &Migrator::rank, "This process's MPI rank.");

  nb::class_<Halo>(m, "Halo")
      .def(nb::init<std::array<double, 3>, std::array<double, 3>, std::array<long, 3>,
                    std::array<bool, 3>>(),
           nb::arg("origin"), nb::arg("size"), nb::arg("gsize"), nb::arg("periodic"))
      .def("build", &Halo::build, nb::arg("positions"), nb::arg("rcut"),
           "Establish the owner<->ghost correspondence over this rank's owned positions")
      .def("forward_positions", &Halo::forward_positions, nb::arg("owned"),
           "owned (N,3) -> ghost (G,3) with the periodic image shift (positions)")
      .def("forward", &Halo::forward, nb::arg("owned"),
           "owned (N,3) -> ghost (G,3) verbatim (velocities, ...)")
      .def("reverse", &Halo::reverse, nb::arg("ghost"), nb::arg("owned"),
           "ghost (G,3) summed onto owned (N,3); returns owned + reversed contributions")
      .def("owner_of", &Halo::owner_of, nb::arg("x"))
      .def("num_ghost", &Halo::num_ghost)
      .def("num_owned", &Halo::num_owned)
      .def_prop_ro("rank", &Halo::rank);
}

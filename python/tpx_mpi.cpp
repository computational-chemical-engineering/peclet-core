// transport-core — minimal Python surface for the Lagrangian halo (migration + ghosts).
//
// A pybind11 module exposing tpx::halo::ParticleMigrator so an mpi4py driver can decompose a periodic
// domain and migrate/ghost particles between ranks. Particles are passed as numpy arrays: positions
// (N,3) float64 and an arbitrary per-particle payload (N,K) float64 (pack velocity, orientation, id,
// etc. into the K columns). MPI is assumed already initialized by the host (import mpi4py.MPI first);
// this module uses MPI_COMM_WORLD and never calls MPI_Init/Finalize.
//
// Reusable by packing-gpu and voronoi_dynamics. Build: see python/CMakeLists.txt.
#include <mpi.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_migrator.hpp"

namespace py = pybind11;
using namespace tpx;

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
  py::tuple migrate(py::array_t<double> pos, py::array_t<double> pay) {
    std::vector<Vec<3>> pv;
    std::vector<char> payload;
    std::size_t K = unpack(pos, pay, pv, payload);
    mig_.migrate(pv, payload, K * sizeof(double));
    return pack(pv, payload, K);
  }

  // copies of particles within rcut of this rank's block (periodic images handled).
  py::tuple gather_ghosts(py::array_t<double> pos, py::array_t<double> pay, double rcut) {
    std::vector<Vec<3>> pv;
    std::vector<char> payload;
    std::size_t K = unpack(pos, pay, pv, payload);
    std::vector<Vec<3>> gpv;
    std::vector<char> gpay;
    mig_.gatherGhosts(pv, payload, K * sizeof(double), rcut, gpv, gpay);
    return pack(gpv, gpay, K);
  }

 private:
  static std::size_t unpack(py::array_t<double>& pos, py::array_t<double>& pay,
                            std::vector<Vec<3>>& pv, std::vector<char>& payload) {
    auto P = pos.unchecked<2>();
    auto Y = pay.unchecked<2>();
    std::size_t N = static_cast<std::size_t>(P.shape(0));
    std::size_t K = static_cast<std::size_t>(Y.shape(1));
    pv.resize(N);
    payload.resize(N * K * sizeof(double));
    for (std::size_t i = 0; i < N; ++i) {
      pv[i] = {P(i, 0), P(i, 1), P(i, 2)};
      for (std::size_t k = 0; k < K; ++k) {
        double v = Y(i, k);
        std::memcpy(&payload[(i * K + k) * sizeof(double)], &v, sizeof(double));
      }
    }
    return K;
  }
  static py::tuple pack(const std::vector<Vec<3>>& pv, const std::vector<char>& payload,
                        std::size_t K) {
    std::size_t M = pv.size();
    py::array_t<double> op({(py::ssize_t)M, (py::ssize_t)3});
    py::array_t<double> oy({(py::ssize_t)M, (py::ssize_t)K});
    auto OP = op.mutable_unchecked<2>();
    auto OY = oy.mutable_unchecked<2>();
    for (std::size_t i = 0; i < M; ++i) {
      OP(i, 0) = pv[i][0];
      OP(i, 1) = pv[i][1];
      OP(i, 2) = pv[i][2];
      for (std::size_t k = 0; k < K; ++k) {
        double v;
        std::memcpy(&v, &payload[(i * K + k) * sizeof(double)], sizeof(double));
        OY(i, k) = v;
      }
    }
    return py::make_tuple(op, oy);
  }

  int rank_ = 0;
  tpx::decomp::BlockDecomposer<3> dec_;
  tpx::halo::ParticleMigrator<3> mig_;
};

PYBIND11_MODULE(tpx_mpi, m) {
  m.doc() = "transport-core Lagrangian halo (block decomposition + particle migration/ghosts)";
  py::class_<Migrator>(m, "Migrator")
      .def(py::init<std::array<double, 3>, std::array<double, 3>, std::array<long, 3>,
                    std::array<bool, 3>>(),
           py::arg("origin"), py::arg("size"), py::arg("gsize"), py::arg("periodic"))
      .def("migrate", &Migrator::migrate, py::arg("positions"), py::arg("payload"))
      .def("gather_ghosts", &Migrator::gather_ghosts, py::arg("positions"), py::arg("payload"),
           py::arg("rcut"))
      .def("owner_of", &Migrator::owner_of, py::arg("x"))
      .def_property_readonly("rank", &Migrator::rank);
}

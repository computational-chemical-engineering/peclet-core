// core — the MPI side of the Anderson accelerator: AndersonComm from an MPI_Comm.
//
// solver/anderson.hpp is MPI-free and takes its collectives as callables; this header builds them
// (MPI_Allreduce SUM of doubles, MPI_Bcast of bytes from rank 0) and is therefore on the MPI side
// of core's boundary (it includes common/mpi.hpp; docs/CORE_BOUNDARY.md §2.1). Built against the
// no-MPI stub (PECLET_CORE_NO_MPI) both collectives are single-rank identities.
#ifndef PECLET_CORE_SOLVER_ANDERSON_MPI_HPP
#define PECLET_CORE_SOLVER_ANDERSON_MPI_HPP

#include <cstddef>
#include <vector>

#include "peclet/core/common/mpi.hpp"
#include "peclet/core/solver/anderson.hpp"

namespace peclet::core::solver {

/// The collectives of `comm` for an AndersonState. The communicator must outlive the accelerator.
inline AndersonComm andersonComm(MPI_Comm comm) {
  AndersonComm c;
  MPI_Comm_rank(comm, &c.rank);
  c.sumAll = [comm](double* data, int count) {
    std::vector<double> send(data, data + count);  // no MPI_IN_PLACE: the no-MPI stub lacks it
    MPI_Allreduce(send.data(), data, count, MPI_DOUBLE, MPI_SUM, comm);
  };
  c.broadcast = [comm](void* data, std::size_t bytes) {
    MPI_Bcast(data, static_cast<int>(bytes), MPI_BYTE, 0, comm);
  };
  return c;
}

}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_ANDERSON_MPI_HPP

// core — Anderson acceleration of a steady march: the grid-agnostic data path.
//
// Design: flow doc/steady_acceleration.md (D13: AndersonCore lives in core from the start; §3 the
// state and the metric; §4 the algorithm; §6 MPI and GPU). The caller owns a fixed-point map
// x ↦ g(x) that reads and writes a set of padded device buffers in place (flow: one
// `Solver::step()`); this header owns the history, the reductions, the least squares, the
// safeguards and the lazy mix of those buffers.
//
// MPI-free: a distributed run passes its collectives as two callables in `AndersonComm` (the
// pattern of csr_bicgstab.hpp's setDistributed), built from an MPI_Comm by
// solver/anderson_mpi.hpp, which is the MPI side. An empty `AndersonComm` is one rank, and an
// MPI_Allreduce / MPI_Bcast on one rank is the identity, so np = 1 is bit-identical to serial.
#ifndef PECLET_CORE_SOLVER_ANDERSON_HPP
#define PECLET_CORE_SOLVER_ANDERSON_HPP

#include <cstddef>
#include <functional>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::core::solver {

/// What a state field is to the metric (design §3.1).
enum class AndersonRole : int {
  Velocity = 0,  ///< unit weight, every inner entry
  Pressure = 1,  ///< weight cP², fluid-centred inner entries only, fluid mean removed if gauged
  Carried = 2,   ///< mixed, stored and differenced like the others, but not measured
};

/// The collectives of a distributed run. Both callables empty = a single rank (serial, or np = 1
/// without them). `sumAll(data, n)` replaces data[0..n) by its global sum on every rank
/// (MPI_Allreduce SUM); `broadcast(data, bytes)` copies rank 0's bytes to every rank (MPI_Bcast).
/// solver/anderson_mpi.hpp builds one from an MPI_Comm.
struct AndersonComm {
  std::function<void(double* data, int count)> sumAll;
  std::function<void(void* data, std::size_t bytes)> broadcast;
  int rank = 0;  ///< this rank in the communicator; rank 0's decisions win (design §6.1)
};

/// The state descriptor (design §5.1): exactly what the caller's step reads across steps, as full
/// padded x-fastest buffers over one box, plus what the metric needs. Grid-agnostic; the caller's
/// parameter signature (flow: dt, ρ, μ, F) is NOT here — the adapter checks it.
struct AndersonState {
  /// The state buffers, each over the full padded box (extent(0) == extent[0]·extent[1]·extent[2]),
  /// read and written in place by the caller's step. The mix writes them too.
  std::vector<View<double>> fields;
  /// One role per field. At most one Pressure field.
  std::vector<AndersonRole> roles;
  /// Cell-centred signed distance over the same padded box; only its sign is used (fluid where
  /// > 0). Required when a Pressure field is present, otherwise may be empty.
  View<const double> sdf;
  /// Padded extents e (inner + 2·ghost per axis); inner entries are ghost ≤ i < e − ghost.
  IVec<3> extent{};
  /// Ghost width G.
  int ghost = 2;
  /// Pressure metric weight c_P = 1/(μ + ρ/Δt) in the caller's internal units (design §3.2).
  double cP = 1.0;
  /// The pressure has a constant nullspace (no Dirichlet-pressure face): its fluid mean is removed
  /// from the metric, with the mean over the GLOBAL count of inner fluid cells.
  bool gauged = true;
  /// Collectives; empty = a single rank.
  AndersonComm comm;
};

}  // namespace peclet::core::solver

#endif  // PECLET_CORE_SOLVER_ANDERSON_HPP

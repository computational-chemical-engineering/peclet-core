// core — the id-keyed replicated gather (amr/docs/amr_mg_core_boundary.md §3.1, §4): every rank
// contributes (global id, value) pairs and every rank receives the whole global vector, indexed by
// id. The one-liner that flow's GraphAMG bottom, amr's ReplicatedTailStage and voro's coming
// GraphAMG bottom each wrote inline. Pure copies, bitwise.
#ifndef PECLET_CORE_DECOMP_GATHER_BY_GLOBAL_ID_HPP
#define PECLET_CORE_DECOMP_GATHER_BY_GLOBAL_ID_HPP

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "peclet/core/common/mpi.hpp"

namespace peclet::core::decomp {

/// `ids[i]` is the global id of `vals[i]` on this rank. On return `out` (size nGlobal, on every
/// rank) holds out[id] = the value some rank contributed for id. The ids over all ranks must be a
/// permutation of [0, nGlobal) — each id exactly once — or this throws (on every rank).
/// Collective on `comm`: one Allgather of the counts, one Allgatherv each of ids and values.
template <class T>
void gatherByGlobalId(const std::vector<long long>& ids, const std::vector<T>& vals,
                      long long nGlobal, std::vector<T>& out, MPI_Comm comm) {
  static_assert(std::is_trivially_copyable_v<T>, "gatherByGlobalId moves T as raw bytes");
  if (ids.size() != vals.size())
    throw std::invalid_argument("gatherByGlobalId: ids and vals differ in length");
  if (ids.size() > static_cast<std::size_t>(INT_MAX / sizeof(T)) ||
      ids.size() > static_cast<std::size_t>(INT_MAX / sizeof(long long)))
    throw std::overflow_error("gatherByGlobalId: a rank's contribution exceeds INT_MAX bytes");
  int size = 1;
  MPI_Comm_size(comm, &size);
  const int n = static_cast<int>(ids.size());
  std::vector<int> counts(static_cast<std::size_t>(size)), displs(static_cast<std::size_t>(size));
  MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  long long tot = 0;
  for (int r = 0; r < size; ++r) {
    displs[static_cast<std::size_t>(r)] = static_cast<int>(tot);
    tot += counts[static_cast<std::size_t>(r)];
    if (tot > INT_MAX)
      throw std::overflow_error("gatherByGlobalId: the global vector exceeds INT_MAX entries");
  }
  if (tot != nGlobal)
    throw std::invalid_argument("gatherByGlobalId: the ids do not cover [0, nGlobal) once each");
  std::vector<long long> idAll(static_cast<std::size_t>(tot));
  MPI_Allgatherv(ids.data(), n, MPI_LONG_LONG, idAll.data(), counts.data(), displs.data(),
                 MPI_LONG_LONG, comm);
  std::vector<T> vAll(static_cast<std::size_t>(tot));
  std::vector<int> bc(counts.size()), bd(displs.size());
  for (std::size_t r = 0; r < counts.size(); ++r) {
    if (static_cast<std::size_t>(counts[r]) > static_cast<std::size_t>(INT_MAX) / sizeof(T) ||
        static_cast<std::size_t>(displs[r]) > static_cast<std::size_t>(INT_MAX) / sizeof(T))
      throw std::overflow_error("gatherByGlobalId: the global vector exceeds INT_MAX bytes");
    bc[r] = counts[r] * static_cast<int>(sizeof(T));
    bd[r] = displs[r] * static_cast<int>(sizeof(T));
  }
  MPI_Allgatherv(vals.data(), n * static_cast<int>(sizeof(T)), MPI_BYTE, vAll.data(), bc.data(),
                 bd.data(), MPI_BYTE, comm);
  out.assign(static_cast<std::size_t>(nGlobal), T{});
  std::vector<char> seen(static_cast<std::size_t>(nGlobal), 0);
  for (std::size_t k = 0; k < idAll.size(); ++k) {
    const long long id = idAll[k];
    if (id < 0 || id >= nGlobal || seen[static_cast<std::size_t>(id)])
      throw std::invalid_argument("gatherByGlobalId: the ids do not cover [0, nGlobal) once each");
    seen[static_cast<std::size_t>(id)] = 1;
    out[static_cast<std::size_t>(id)] = vAll[k];
  }
}

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_DECOMP_GATHER_BY_GLOBAL_ID_HPP

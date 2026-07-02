// core — particle-count load re-balancing for the Lagrangian path.
//
// The Eulerian/AMR side rebalances by re-decomposing on a per-cell *work* weight and migrating the
// owned data (see peclet::core::amr::DistributedOctree::rebalance). This is the Lagrangian
// counterpart: when particles drift (e.g. a dense DEM packing densifies and skews the per-block
// count), the equal-cell ORB no longer balances the load. rebalanceByParticleCount() bins the
// particles onto the decomposition grid, re-decomposes with the *weighted* ORB so every block
// carries a near-equal particle count, and migrates particles to their new owners with the existing
// ParticleMigrator.
//
// It is a pure redistribution: the global particle set is unchanged (count conserved, positions and
// payloads preserved); only ownership moves. The BlockDecomposer is re-initialised in place, so a
// ParticleMigrator (or GridHaloTopology) holding a pointer to it transparently sees the new
// partition. MPI-optional: a single rank keeps every particle and the decomposition is unchanged.
#ifndef PECLET_CORE_HALO_PARTICLE_REBALANCE_HPP
#define PECLET_CORE_HALO_PARTICLE_REBALANCE_HPP

#include <vector>

#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_migrator.hpp"

namespace peclet::core::halo {

namespace detail {
template <class T>
struct NonDeduced {
  using type = T;
};
}  // namespace detail

/// Re-decompose `dec` so each block holds a near-equal particle count, then migrate the particles
/// (positions + opaque fixed-stride payload) to their new owners via `mig` (which must reference
/// `dec`). `weightOut`, when non-null, receives the agreed global per-cell count grid (x-fastest).
/// Returns this rank's new local particle count. (`pos` is a non-deduced parameter — `Dim` is fixed
/// by `dec`/`mig` — because `int Dim` cannot deduce from a `std::array`'s `size_t` bound.)
template <int Dim>
std::size_t rebalanceByParticleCount(decomp::BlockDecomposer<Dim>& dec, ParticleMigrator<Dim>& mig,
                                     std::vector<typename detail::NonDeduced<Vec<Dim>>::type>& pos,
                                     std::vector<char>& payload, std::size_t stride,
                                     MPI_Comm comm = MPI_COMM_WORLD,
                                     std::vector<double>* weightOut = nullptr) {
  // 1. Particle count per global decomposition cell, agreed across ranks.
  const IVec<Dim>& gsize = dec.globalSize();
  std::size_t ncells = 1;
  for (int d = 0; d < Dim; ++d)
    ncells *= static_cast<std::size_t>(gsize[d]);
  std::vector<double> localCount(ncells, 0.0), weight(ncells, 0.0);
  for (const Vec<Dim>& x : pos)
    localCount[static_cast<std::size_t>(dec.linearGlobal(mig.cellOf(x)))] += 1.0;
  MPI_Allreduce(localCount.data(), weight.data(), static_cast<int>(ncells), MPI_DOUBLE, MPI_SUM,
                comm);

  // 2. Weighted re-decomposition over the same grid (in place: mig still points at dec).
  int sz = 1;
  MPI_Comm_size(comm, &sz);
  dec.init(static_cast<std::size_t>(sz), gsize, weight);
  if (weightOut)
    *weightOut = std::move(weight);

  // 3. Migrate particles to their new owners. ParticleMigrator already reassigns by ownerOf, which
  //    now reflects the re-weighted partition.
  return mig.migrate(pos, payload, stride);
}

}  // namespace peclet::core::halo

#endif  // PECLET_CORE_HALO_PARTICLE_REBALANCE_HPP

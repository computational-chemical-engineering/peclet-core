# transport-core

Shared infrastructure for the transport-phenomena simulation suite (see `../docs/` for the suite-wide
[architecture](../docs/ARCHITECTURE.md), [conventions](../docs/CONVENTIONS.md),
[style](../docs/STYLE.md), [interfaces](../docs/INTERFACES.md) and [roadmap](../docs/ROADMAP.md)).

It provides the pieces every method code (`sdflow`, `dem`, `vorflow`, …) should
share: a common MPI **block domain decomposition**, an efficient **asynchronous ghost-layer
exchange** (CPU + portable Kokkos GPU), **particle migration**, **dynamic load balancing**, unified
**SDF geometry** (`tpx::geom`), an **AMR octree** flow subsystem (`tpx::amr`), and **nanobind Python
bindings**. Header-only C++20 (the device side, compiled through Kokkos, is also C++20; only the
`morton` dependency pins C++17 — see `../docs/STYLE.md`). Cut-cell IBM is not a standalone shared
module: it currently lives inside the AMR flow solver (`tpx::amr`) and in `sdflow`.

## What works today

- `tpx::decomp::BlockDecomposer<Dim>` — orthogonal recursive bisection of a global cell grid into
  rank-owned blocks; `ownerOf()` tree-walk; x-fastest global/local linear indexing.
- `tpx::decomp::BlockIndexer<Dim>` — local↔global indexing for a block with a ghost layer.
- `tpx::halo::NbxEngine` — nonblocking-consensus sparse exchange (Issend + Ibarrier), for dynamic
  patterns.
- `tpx::halo::GridHaloTopology<Dim>` (`grid_halo_topology.hpp`) — asynchronous ghost-layer exchange
  with **topology separated from exchange** and a **field-agnostic** pack/unpack interface.
  `buildTopology()` runs once; two interchangeable exchange engines give identical results:
  - `exchangeNbx()` / `start()`+`wait()` — NBX, supports compute/comm overlap.
  - `exchangePersistent()` — `MPI_Neighbor_alltoallv` on a cached distributed-graph communicator;
    fastest for the static neighbour pattern of a fixed grid.
- `tpx::halo::GridFieldView<T>` — wraps a contiguous local array as an exchangeable field.
- `tpx::halo::GridHalo<T>` (`grid_halo.hpp`) — portable **GPU-resident** ghost-layer exchange (Kokkos:
  CUDA / HIP / OpenMP). Built once from a host `GridHaloTopology<Dim>::flatten()`; pack / unpack /
  periodic self-copy run as `Kokkos::parallel_for` over the device `tpx::View<T>` field, so the full
  field never crosses the bus — only the compact halo buffers are host-staged for MPI by default, with
  an opt-in GPU-aware path (env `TPX_GPU_AWARE_MPI`). Bit-for-bit identical to the CPU exchange.
- `tpx::halo::ParticleMigrator<Dim>` — Lagrangian particle migration to owning ranks (NBX), the
  dynamic counterpart to the Eulerian grid halo.
- `tpx::halo::ParticleHaloTopology<Dim>` (`particle_halo_topology.hpp`) — persistent Lagrangian ghost
  halo: `forward` (owner→ghost), `reverse` (ghost→owner, accumulate) and `forwardPositions` (periodic
  image shift). `tpx::halo::ParticleHalo<Dim>` (`particle_halo.hpp`) is its GPU-resident Kokkos driver
  (on-device gather/scatter, host-staged or GPU-aware MPI), consumed by dem's distributed step.
- `tpx::halo::rebalanceByParticleCount(...)` (`particle_rebalance.hpp`) — **dynamic load balancing**
  for the Lagrangian path: re-inits the decomposition in place with the **weighted ORB**
  (`BlockDecomposer::init(numBlocks, globalSize, weights)`) and migrates. The Eulerian/AMR counterpart
  is `tpx::amr::DistributedOctree::rebalance`.
- `tpx::geom` (`sdf.hpp`, `grid_sdf.hpp`, `vti_io.hpp`) — shared SDF solids: analytic primitives +
  trilinear `GridSdf` behind one `Sdf` concept, with VTI (.vti) read/write.
- `tpx::amr` (`include/tpx/amr/`) — block-local-Morton **AMR octree** flow subsystem: `tpx::amr::AmrFlow`
  (collocated projection Navier–Stokes), device + distributed multigrid (`pcg.hpp`, `multigrid.hpp`,
  `velocity_mg.hpp`, `distributed_*.hpp`), cut-cell IBM (`cut_cell.hpp`) and solution-adaptive refinement
  (`adapt.hpp`, `indicators.hpp`). See `docs/amr_collocated_projection.md`.
- **Python bindings** (`python/tpx_mpi.cpp`, `python/tpx_amr.cpp`) — **nanobind** modules over the
  shared zero-copy `View`↔ndarray bridge (`include/tpx/python/ndarray_interop.hpp`). `tpx_mpi` exposes
  the host Lagrangian halo / migration / rebalance; `tpx_amr` exposes the device AMR flow.

Validated end-to-end by distributed explicit heat-diffusion solvers (plain, and **around an SDF solid
obstacle**) matching a serial reference cell-for-cell across ranks, and consumed by the validated
`sdflow` and `dem` distributed solvers. 26 ctests pass (`np` 1–8 CPU, 1–4 GPU).

## Build / test / benchmark

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure          # 26 ctests: serial + MPI (np=1,2,4,8)

# halo microbenchmark: weak scaling, NBX vs persistent
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300  # cells/rank/axis, ghost, iters
```

Requires MPI (OpenMPI/MPICH) and a C++20 compiler. `morton` is picked up automatically if
checked out as a sibling directory (enables `TPX_HAVE_MORTON`).

## Status

Complete and in production. The block decomposition, the async ghost-layer exchange (CPU + portable
Kokkos GPU, host-staged and opt-in GPU-aware), particle migration, dynamic load balancing (weighted
ORB + AMR/Lagrangian rebalancing), SDF geometry, the AMR octree flow subsystem (device + distributed
multigrid, collocated projection), and the nanobind Python bindings are all shipped and tested
(26 ctests, `np` 1–8 CPU / 1–4 GPU). `sdflow` (distributed cut-cell IBM Navier–Stokes) and `dem`
(distributed XPBD with load rebalancing) are validated consumers. CUDA is retired; Kokkos
(CUDA / HIP / OpenMP) is the canonical device path. Remaining work is at-scale multi-GPU tuning.

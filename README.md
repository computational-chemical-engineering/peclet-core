# transport-core

Shared infrastructure for the transport-phenomena simulation suite (see `../docs/` for the suite-wide
[architecture](../docs/ARCHITECTURE.md), [conventions](../docs/CONVENTIONS.md),
[style](../docs/STYLE.md), [interfaces](../docs/INTERFACES.md) and [roadmap](../docs/ROADMAP.md)).

It provides the pieces every method code (`sdflow`, `dem`, `voronoi_dynamics`, …) should
share: a common MPI **block domain decomposition** and an efficient **asynchronous ghost-layer
exchange**, plus (planned) unified SDF geometry, IBM, and Python bindings. Header-only C++20 (the
device boundary stays C++17-compatible).

## What works today

- `tpx::decomp::BlockDecomposer<Dim>` — orthogonal recursive bisection of a global cell grid into
  rank-owned blocks; `ownerOf()` tree-walk; x-fastest global/local linear indexing.
- `tpx::decomp::BlockIndexer<Dim>` — local↔global indexing for a block with a ghost layer.
- `tpx::halo::NbxEngine` — nonblocking-consensus sparse exchange (Issend + Ibarrier), for dynamic
  patterns.
- `tpx::halo::GridHalo<Dim>` — asynchronous ghost-layer exchange with **topology separated from
  exchange** and a **field-agnostic** pack/unpack interface. Two interchangeable engines:
  - `exchangeNbx()` / `start()`+`wait()` — NBX, supports compute/comm overlap.
  - `exchangePersistent()` — `MPI_Neighbor_alltoallv` on a cached distributed-graph communicator;
    fastest for the static neighbour pattern of a fixed grid.
- `tpx::halo::GridFieldView<T>` — wraps a contiguous local array as an exchangeable field.
- `tpx::halo::ParticleMigrator<Dim>` — Lagrangian particle migration to owning ranks (NBX), the
  dynamic counterpart to the Eulerian grid halo.
- `tpx::halo::DeviceGridExchangeKokkos<T>` (`grid_halo_kokkos.hpp`) — portable GPU-resident ghost-layer
  exchange (Kokkos: CUDA / HIP / OpenMP): pack/unpack/self-copy as `parallel_for`, host-staged MPI (opt-in
  GPU-aware); the field stays on the device. `tpx::halo::DeviceParticleHaloKokkos<Dim>` is the Lagrangian
  counterpart. (The legacy native-CUDA `grid_halo_cuda.cuh` was retired.)

- `tpx::geom` (`sdf.hpp`, `grid_sdf.hpp`, `vti_io.hpp`) — shared SDF solids: analytic primitives +
  trilinear `GridSdf` behind one `Sdf` concept, with VTI (.vti) read/write.

Validated end-to-end by distributed explicit heat-diffusion solvers (plain, and **around an SDF solid
obstacle**) matching a serial reference cell-for-cell across ranks. 22/22 ctest pass (`np` 1–8 CPU,
1–4 GPU).

## Build / test / benchmark

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure          # serial + MPI (np=1,2,4,8)

# halo microbenchmark: weak scaling, NBX vs persistent
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300  # cells/rank/axis, ghost, iters
```

Requires MPI (OpenMPI/MPICH) and a C++20 compiler. `morton` is picked up automatically if
checked out as a sibling directory (enables `TPX_HAVE_MORTON`).

## Status

Phase 0–1 of the roadmap: decomposition + CPU async halo, validated and benchmarked. Next:
GPU-aware exchange (CUDA-aware MPI + device pack/unpack), particle migration, then wiring `sdflow`
in as the first Eulerian consumer.

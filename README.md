# transport-core

Shared infrastructure for the transport-phenomena simulation suite (see `../docs/` for the suite-wide
[architecture](../docs/ARCHITECTURE.md), [conventions](../docs/CONVENTIONS.md),
[style](../docs/STYLE.md), [interfaces](../docs/INTERFACES.md) and [roadmap](../docs/ROADMAP.md)).

It provides the pieces every method code (`cfd-gpu`, `packing-gpu`, `voronoi_dynamics`, …) should
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
- `tpx::halo::DeviceGridExchange<T>` (`grid_halo_cuda.cuh`) — GPU-resident ghost-layer exchange:
  pack/unpack/self-copy as CUDA kernels, host-staged MPI; the field never leaves the GPU.

Validated end-to-end by a distributed explicit heat-diffusion solver that matches a serial reference
cell-for-cell across ranks. 16/16 ctest pass (`np` 1–8 CPU, 1–4 GPU).

## Build / test / benchmark

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure          # serial + MPI (np=1,2,4,8)

# halo microbenchmark: weak scaling, NBX vs persistent
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300  # cells/rank/axis, ghost, iters
```

Requires MPI (OpenMPI/MPICH) and a C++20 compiler. `morton_arithmetic` is picked up automatically if
checked out as a sibling directory (enables `TPX_HAVE_MORTON`).

## Status

Phase 0–1 of the roadmap: decomposition + CPU async halo, validated and benchmarked. Next:
GPU-aware exchange (CUDA-aware MPI + device pack/unpack), particle migration, then wiring `cfd-gpu`
in as the first Eulerian consumer.

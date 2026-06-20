# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`transport-core` is the shared infrastructure library for the transport-phenomena simulation suite
(sibling repos under `../`: `sdflow`, `dem`, `vorflow`, `morton`). The suite-wide design contract lives in `../docs/` — read
`../docs/ARCHITECTURE.md`, `CONVENTIONS.md`, `STYLE.md`, `INTERFACES.md`, `ROADMAP.md` before
cross-cutting changes. Header-only C++20; anything that must compile as CUDA device code stays
C++17-compatible.

## Build / test / benchmark

```bash
# CPU library + tests (no device dependency):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure   # serial + MPI halo + particle migration + diffusion

# Portable Kokkos device halo (CUDA / HIP / OpenMP) -- opt-in, find_package(Kokkos):
export PATH=/usr/local/cuda-13.2/bin:$PATH    # if the Kokkos install targets the CUDA backend
cmake -S . -B build_kokkos -DTPX_ENABLE_KOKKOS=ON \
  -DCMAKE_PREFIX_PATH=../extern/install/nvidia-cuda
cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure  # + GPU halo np=1,2,4
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300
```

The Kokkos halo path is provisioned via `find_package(Kokkos CONFIG)` against a cluster module or the
suite's local install prefix (`../tools/bootstrap_deps.sh`). The legacy native-CUDA halo was retired.

## Architecture

Header-only under `include/tpx/`:

- `common/types.hpp` — `Index` (int64), `Real` (double), `IVec<Dim>`/`Vec<Dim>`, `wrap()`,
  compile-time `forEachInBox`. **Convention: x-fastest linear index** `I = x + y*nx + z*nx*ny`
  (matches sdflow and `../docs/CONVENTIONS.md`). Keep this header C++17-clean (pulled into `.cu`).
- `decomp/block_decomposer.hpp` — ORB decomposition (ported & modernized from
  `../block_decomposer/src/BlockDecomposer.hpp`). `ownerOf()` walks the implicit binary tree
  (children at `2i+1`/`2i+2`, leaves carry the block index) and is the key primitive for halo
  topology. `linearGlobal`/`multiGlobal` are x-fastest and mutually inverse.
- `decomp/block_indexer.hpp` — local↔global indexing for an extended (inner+ghost) block.
- `halo/nbx.hpp` — `NbxEngine`: canonical NBX (Issend + Ibarrier consensus). Reimplements the engine
  from `../block_decomposer/src/MPISync.hpp`. Use for dynamic/sparse exchange.
- `halo/grid_halo.hpp` — `GridHalo<Dim>`: the ghost-layer exchange. **Topology** (who owns each ghost
  cell, established via one NBX round so owners learn what to send) is built once in
  `buildTopology()`; **exchange** runs every step. Field-agnostic: any type with
  `bytesPerElem()`/`pack(localIdx,dst)`/`unpack(localIdx,src)` works. Two engines give identical
  results — `exchangeNbx`/`start`+`wait` (overlap-capable) and `exchangePersistent`
  (`MPI_Neighbor_alltoallv`, faster for static grids). `flatten()` exposes a device-friendly topology.
- `halo/particle_migrator.hpp` — `ParticleMigrator<Dim>`: Lagrangian counterpart. Reassigns particles
  (positions + opaque fixed-stride payload) to their owning rank via the NBX engine, with periodic wrap.
- `halo/grid_halo_kokkos.hpp` — `DeviceGridExchangeKokkos<T>`: portable GPU-resident halo (Kokkos;
  CUDA / HIP / OpenMP backends). pack/unpack/self-copy run as `parallel_for` over `Kokkos::View`s; only
  the compact halo buffers are host-staged for MPI by default (the field stays on the device), with an
  opt-in GPU-aware path (`TPX_GPU_AWARE_MPI`). Built from a host `GridHalo`'s `flatten()`. Bit-for-bit
  matches the CPU exchange. (The legacy native-CUDA `grid_halo_cuda.cuh` / `DeviceGridExchange<T>` was
  retired when Kokkos became the canonical device path; see `docs/cuda-aware-mpi.md` for the historical
  host-staging-vs-GPU-aware analysis.)
- `halo/particle_halo_kokkos.hpp` — `DeviceParticleHaloKokkos<Dim>`: the Lagrangian device counterpart
  (forward gather + reverse atomic-accumulate), consumed by dem's distributed step.

## Gotchas

- `GridHalo` caches a distributed-graph `MPI_Comm`. Its destructor guards `MPI_Comm_free` with
  `MPI_Finalized` so an instance that outlives `MPI_Finalize` (e.g. on `main`'s stack) does not abort.
  Don't remove that guard. The class is non-copyable (it owns the comm).
- The halo is owner-based, not adjacency-based: a ghost cell maps to whichever rank owns its wrapped
  global cell, so it is correct for ORB's irregular block neighbours and any ghost width — no
  Cartesian-grid assumption.
- Tests are dependency-free (`tests/test_util.hpp`, non-zero exit on failure). MPI tests run under
  `mpirun` at several rank counts via ctest.

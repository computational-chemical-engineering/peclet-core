# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`transport-core` is the shared infrastructure library for the transport-phenomena simulation suite
(sibling repos under `../`: `cfd-gpu`, `packing-gpu`, `voronoi_dynamics`, `morton_arithmetic`,
`block_decomposer`). The suite-wide design contract lives in `../docs/` — read
`../docs/ARCHITECTURE.md`, `CONVENTIONS.md`, `STYLE.md`, `INTERFACES.md`, `ROADMAP.md` before
cross-cutting changes. Header-only C++20; anything that must compile as CUDA device code stays
C++17-compatible.

## Build / test / benchmark

```bash
# CUDA tests need nvcc on PATH (this box: /usr/local/cuda-13.2/bin); without it the GPU path is
# skipped and the CPU library still builds/tests fine.
export PATH=/usr/local/cuda-13.2/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure   # serial + MPI halo + particle migration + diffusion
                                             # + GPU halo (np=1..8), 16 tests
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300
```

CUDA arch is forced to `native` (the RTX 5080 is sm_120; CMake's `enable_language(CUDA)` otherwise
defaults to an old arch and kernels silently fail to launch). Override with `-DTPX_CUDA_ARCH=120`.

## Architecture

Header-only under `include/tpx/`:

- `common/types.hpp` — `Index` (int64), `Real` (double), `IVec<Dim>`/`Vec<Dim>`, `wrap()`,
  compile-time `forEachInBox`. **Convention: x-fastest linear index** `I = x + y*nx + z*nx*ny`
  (matches cfd-gpu and `../docs/CONVENTIONS.md`). Keep this header C++17-clean (pulled into `.cu`).
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
- `halo/grid_halo_cuda.cuh` — `DeviceGridExchange<T>`: GPU-resident halo. pack/unpack/self-copy run as
  CUDA kernels; only the compact halo buffers are host-staged for MPI (the field stays on the GPU).
  Built from a host `GridHalo`'s `flatten()`. Direct device-pointer MPI is *not* used (CUDA-aware MPI
  segfaults on this box — stock OpenMPI built without CUDA; see `docs/cuda-aware-mpi.md` for the
  diagnosis, the sysadmin ask, and the localized swap-in); host-staging is the portable path.
  Bit-for-bit matches the CPU exchange.

## Gotchas

- `GridHalo` caches a distributed-graph `MPI_Comm`. Its destructor guards `MPI_Comm_free` with
  `MPI_Finalized` so an instance that outlives `MPI_Finalize` (e.g. on `main`'s stack) does not abort.
  Don't remove that guard. The class is non-copyable (it owns the comm).
- The halo is owner-based, not adjacency-based: a ghost cell maps to whichever rank owns its wrapped
  global cell, so it is correct for ORB's irregular block neighbours and any ghost width — no
  Cartesian-grid assumption.
- Tests are dependency-free (`tests/test_util.hpp`, non-zero exit on failure). MPI tests run under
  `mpirun` at several rank counts via ctest.

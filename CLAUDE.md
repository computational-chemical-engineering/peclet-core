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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure        # serial decomposition + MPI halo (np=1,2,4,8)
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300
```

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
  (`MPI_Neighbor_alltoallv`, faster for static grids).

## Gotchas

- `GridHalo` caches a distributed-graph `MPI_Comm`. Its destructor guards `MPI_Comm_free` with
  `MPI_Finalized` so an instance that outlives `MPI_Finalize` (e.g. on `main`'s stack) does not abort.
  Don't remove that guard. The class is non-copyable (it owns the comm).
- The halo is owner-based, not adjacency-based: a ghost cell maps to whichever rank owns its wrapped
  global cell, so it is correct for ORB's irregular block neighbours and any ghost width — no
  Cartesian-grid assumption.
- Tests are dependency-free (`tests/test_util.hpp`, non-zero exit on failure). MPI tests run under
  `mpirun` at several rank counts via ctest.

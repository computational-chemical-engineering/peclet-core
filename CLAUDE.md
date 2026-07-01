# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`core` is the shared infrastructure library for the transport-phenomena simulation suite
(sibling repos under `../`: `flow`, `dem`, `voro`, `morton`). The suite-wide design contract lives in `../docs/` — read
`../docs/ARCHITECTURE.md`, `CONVENTIONS.md`, `STYLE.md`, `INTERFACES.md`, `ROADMAP.md` before
cross-cutting changes. Header-only C++20; the device side is compiled through Kokkos (CUDA / HIP /
OpenMP) and is also C++20 — only the `morton` dependency pins C++17 (see `../docs/STYLE.md`). CUDA is
retired; Kokkos is the canonical device path.

## Build / test / benchmark

```bash
# CPU library + tests (no device dependency):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure   # serial + MPI halo + particle migration + diffusion

# Portable Kokkos device halo (CUDA / HIP / OpenMP) -- opt-in, find_package(Kokkos):
export PATH=/usr/local/cuda-13.2/bin:$PATH    # if the Kokkos install targets the CUDA backend
cmake -S . -B build_kokkos -DPECLET_CORE_ENABLE_KOKKOS=ON \
  -DCMAKE_PREFIX_PATH=../extern/install/nvidia-cuda
cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure  # + GPU halo np=1,2,4
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300
```

The Kokkos halo path is provisioned via `find_package(Kokkos CONFIG)` against a cluster module or the
suite's local install prefix (`../tools/bootstrap_deps.sh`). The legacy native-CUDA halo was retired.

## Architecture

Header-only under `include/peclet/core/`:

- `common/types.hpp` — `Index` (int64), `Real` (double), `IVec<Dim>`/`Vec<Dim>`, `wrap()`,
  compile-time `forEachInBox`. **Convention: x-fastest linear index** `I = x + y*nx + z*nx*ny`
  (matches flow and `../docs/CONVENTIONS.md`). Keep this header C++17-clean (shared with `morton`,
  which pins C++17).
- `decomp/block_decomposer.hpp` — ORB decomposition (ported & modernized from
  `../block_decomposer/src/BlockDecomposer.hpp`). `ownerOf()` walks the implicit binary tree
  (children at `2i+1`/`2i+2`, leaves carry the block index) and is the key primitive for halo
  topology. `linearGlobal`/`multiGlobal` are x-fastest and mutually inverse. `init(numBlocks,
  globalSize, weights)` is the **weighted ORB** for dynamic load balancing: it bisects at the cell
  boundary whose cumulative weight reaches the sub-block target fraction (vs equal cell count);
  equal weights reduce to the unweighted `init()` bit-for-bit.
- `decomp/block_indexer.hpp` — local↔global indexing for an extended (inner+ghost) block.
- `decomp/morton_indexer.hpp` — `MortonIndexer<Dim>`: Z-order (Morton) cell indexing via the `morton`
  primitive (`morton::Morton<Dim,Bits>`), guarded by `PECLET_CORE_HAVE_MORTON`. The cache-friendly alternative
  to the x-fastest order (which stays the convention): `codeOf`/`multiIndex` map global multi-index ↔
  Z-order code, `neighborCode` steps one cell along an axis directly in Morton space. Methods carry
  morton's `MORTON_HD`, so they are device-callable under a Kokkos build (the Kokkos build defines
  `MORTON_ENABLE_KOKKOS` ⇒ `MORTON_HD` is `KOKKOS_FUNCTION`).
- `halo/nbx.hpp` — `NbxEngine`: canonical NBX (Issend + Ibarrier consensus). Reimplements the engine
  from `../block_decomposer/src/MPISync.hpp`. Use for dynamic/sparse exchange.
- `halo/grid_halo_topology.hpp` — `GridHaloTopology<Dim>`: the ghost-layer exchange. **Topology** (who
  owns each ghost cell, established via one NBX round so owners learn what to send) is built once in
  `buildTopology()`; **exchange** runs every step. Field-agnostic: any type with
  `bytesPerElem()`/`pack(localIdx,dst)`/`unpack(localIdx,src)` works (`GridFieldView<T>` is the
  contiguous-array adapter). Two engines give identical results — `exchangeNbx`/`start`+`wait`
  (overlap-capable) and `exchangePersistent` (`MPI_Neighbor_alltoallv`, faster for static grids).
  `flatten()` exposes a device-friendly topology consumed by the device `GridHalo`.
- `halo/particle_migrator.hpp` — `ParticleMigrator<Dim>`: Lagrangian counterpart. Reassigns particles
  (positions + opaque fixed-stride payload) to their owning rank via the NBX engine, with periodic wrap.
  `cellOf()` exposes the global binning cell (`ownerOf == dec.ownerOf(cellOf(x))`).
- `halo/particle_rebalance.hpp` — `rebalanceByParticleCount(dec, mig, pos, payload, …)`: Lagrangian load
  balancing. Bins particles onto the grid, re-inits `dec` in place with the **weighted ORB** (so a
  migrator/halo holding a pointer to it sees the new partition), and migrates. Pure redistribution
  (count/payload preserved). The dem distributed step is the consumer; also bound in `python/tpx_mpi.cpp`.
- `halo/grid_halo.hpp` — `GridHalo<T>`: portable GPU-resident halo (Kokkos; CUDA / HIP / OpenMP
  backends). pack/unpack/self-copy run as `parallel_for` over the device `peclet::core::View<T>` field; only the
  compact halo buffers are host-staged for MPI by default (the field stays on the device), with an
  opt-in GPU-aware path (env `PECLET_CORE_GPU_AWARE_MPI`, legacy `PECLET_CORE_CUDA_AWARE_MPI` still honoured). Built
  from a host `GridHaloTopology<Dim>::flatten()` via `init()`. Bit-for-bit matches the CPU exchange.
  (The legacy native-CUDA `grid_halo_cuda.cuh` / `DeviceGridExchange<T>` was retired when Kokkos became
  the canonical device path; see `docs/cuda-aware-mpi.md` for the historical
  host-staging-vs-GPU-aware analysis.)
- `halo/particle_halo_topology.hpp` — `ParticleHaloTopology<Dim>`: persistent Lagrangian ghost halo
  (host topology + field-agnostic exchange). `build()` establishes the owner↔ghost correspondence from
  particle proximity; `forward` (owner→ghost), `reverse` (ghost→owner, accumulate) and
  `forwardPositions` (periodic image shift) are the cheap per-step exchanges. The standard distributed
  particle schemes (frozen/replicate, Newton-on, force-accumulate) are compositions of these.
- `halo/particle_halo.hpp` — `ParticleHalo<Dim>`: the Kokkos GPU-resident driver for
  `ParticleHaloTopology` (on-device forward gather + reverse atomic-accumulate; host-staged or
  GPU-aware MPI). Built from `ParticleHaloTopology::flatten()`; consumed by dem's distributed step.
- `geom/` — shared SDF solids. `geom/sdf.hpp` is the `Sdf` concept + analytic primitives;
  `geom/grid_sdf.hpp` is the trilinearly-sampled `GridSdf`; `geom/vti_io.hpp` reads/writes scalar &
  vector VTI (`.vti`). The shared geometry representation behind flow's and dem's cut-cell IBM.
- `amr/` — block-local-Morton **AMR octree** flow subsystem (`peclet::core::amr`, guarded by `PECLET_CORE_HAVE_MORTON`).
  `amr/block_octree.hpp` is the per-block octree; `amr/flow.hpp` is the canonical device `AmrFlow`
  (collocated-projection Navier–Stokes with `maskSolid` and a div-free face field), with
  `amr/flow_oracle.hpp` an unexposed serial host reference. Device + distributed multigrid live in
  `amr/pcg.hpp`, `amr/multigrid.hpp`, `amr/velocity_mg.hpp`, `amr/momentum.hpp` and the
  `amr/distributed_*.hpp` set (`distributed_octree.hpp::rebalance` is the Eulerian leaf/field load
  balancer). Cut-cell openness is `amr/cut_cell.hpp`; solution-adaptive refinement is `amr/adapt.hpp` /
  `amr/indicators.hpp` / `amr/refine.hpp`. Design notes: `docs/amr_collocated_projection.md`,
  `docs/amr_device_assembly_plan.md`.
- `python/` + `python/include/peclet/core/python/ndarray_interop.hpp` — **nanobind** Python bindings over a
  shared **zero-copy `peclet::core::View`↔ndarray bridge** (`include/peclet/core/python/ndarray_interop.hpp`).
  `python/tpx_mpi.cpp` is host-only (no Kokkos): exposes `ParticleMigrator` / `ParticleHaloTopology` /
  `rebalanceByParticleCount` for an mpi4py driver. `python/tpx_amr.cpp` exposes the device `AmrFlow`
  (needs the `morton` sibling + a Kokkos backend). Both are built via `include(SuiteNanobind)` +
  `suite_require_nanobind()` from `../cmake/SuiteNanobind.cmake` (suite-root).

## Gotchas

- `GridHalo` caches a distributed-graph `MPI_Comm`. Its destructor guards `MPI_Comm_free` with
  `MPI_Finalized` so an instance that outlives `MPI_Finalize` (e.g. on `main`'s stack) does not abort.
  Don't remove that guard. The class is non-copyable (it owns the comm).
- The halo is owner-based, not adjacency-based: a ghost cell maps to whichever rank owns its wrapped
  global cell, so it is correct for ORB's irregular block neighbours and any ghost width — no
  Cartesian-grid assumption.
- Tests are dependency-free (`tests/test_util.hpp`, non-zero exit on failure). MPI tests run under
  `mpirun` at several rank counts via ctest.
- `../cmake/SuiteNanobind.cmake` MUST be a CMake **macro**, not a `function()`: it sets/propagates
  variables (the located nanobind, the interpreter) into the including scope, which a function's nested
  scope would swallow. Keep `suite_require_nanobind` defined as a macro.

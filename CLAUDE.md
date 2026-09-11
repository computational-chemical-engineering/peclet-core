# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`core` is the shared infrastructure library of the **peclet** suite
(sibling repos under `../`: `flow`, `dem`, `voro`, `morton`). The suite-wide design contract lives in `../docs/` — read
`../docs/ARCHITECTURE.md`, `CONVENTIONS.md`, `STYLE.md`, `INTERFACES.md`, `ROADMAP.md` before
cross-cutting changes. Header-only C++20; the device side is compiled through Kokkos (CUDA / HIP /
OpenMP) and is also C++20 — only the `morton` dependency pins C++17 (see `../docs/STYLE.md`). CUDA is
retired; Kokkos is the canonical device path.

## Settled decisions — do not reverse silently

Chosen *against* the obvious or textbook alternative, on measured evidence. Full entries with
verbatim quotes and provenance in [`../docs/decisions/core.md`](../docs/decisions/core.md); the index is
[`../docs/DECISIONS.md`](../docs/DECISIONS.md). Reversing one takes a new recorded decision, not a
judgement call in the moment.

- **Rebalance is pure migration** — same global mesh, new owners — and must never use
  `transferFields`.
- **AMR PCG must mask solid AND project onto the fluid range** (mask + fluid-only mean), not just
  one of the two.
- **The CUDA-aware MPI device path is gated on an explicit env var**, never on the MPI query API
  alone; auto-detection uses query plus a checksum loopback probe, never blind probing.
- **Anisotropic coarse-grid partitioning requires `cellExtent`**, not raw cell-count `kLargest`.

## Build / test / benchmark

```bash
# CPU library + tests (no device dependency):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure -LE bench   # 52 ctests (53 with `bench`): decomposition, MPI halo, particle migration, diffusion, geometry

# Portable Kokkos device halo (CUDA / HIP / OpenMP) -- opt-in, find_package(Kokkos):
export PATH=/usr/local/cuda-13.2/bin:$PATH    # if the Kokkos install targets the CUDA backend
cmake -S . -B build_kokkos -DPECLET_CORE_ENABLE_KOKKOS=ON \
  -DCMAKE_PREFIX_PATH=../extern/install/nvidia-cuda
cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure -LE bench  # 67 ctests (68 with `bench`): + device halo / geometry / solver, np=1,2,4,8
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300

# Python modules + their ctests (test_mpi.py np=1,2,4,8; state_hash; ndarray interop):
cmake -S python -B build_rel_py -DCMAKE_PREFIX_PATH=../extern/install/host-openmp
cmake --build build_rel_py -j && ctest --test-dir build_rel_py --output-on-failure   # 6 ctests
```

**ctest protocol** (suite/docs/QUALITY_PLAN.md §3.D; helpers in `cmake/PecletCoreTest.cmake`, the ONE
place every test is registered through): a test that cannot run in a configuration — no `morton`
sibling (`PECLET_CORE_MORTON_DIR`), so `PECLET_CORE_HAVE_MORTON` is unset (`morton_indexer`) — exits 77
(`tests/test_util.hpp kSkipExitCode`; MPI tests via `tests/test_skip_mpi.hpp`, which
Init/Finalizes first so the launcher forwards the 77 instead of aborting with 1) and ctest reports it
"Not Run (skipped)", never Passed. Labels: `mpi` (every mpirun test), `np8` (8-rank instances —
a LOCAL gate; CI's 4-core runners run `-LE np8`), `bench` (`bench_halo`, excluded by default), `python`. Thread bounds for batteries on this
host: `OMP_NUM_THREADS=2 OMP_PROC_BIND=false`, np=8 subset last. CI (`.github/workflows/ci.yml`)
runs host+MPI (gcc/clang × Debug/Release), Kokkos-OpenMP + Python, and no-MPI, each with the
`morton` tag checked out as a sibling; the clang-format check (`quality.yml`, clang-format 18.1.8)
is blocking over `include/ tests/ python/ benchmarks/`.

The Kokkos halo path is provisioned via `find_package(Kokkos CONFIG)` against a cluster module or the
suite's local install prefix (`../tools/bootstrap_deps.sh`). The legacy native-CUDA halo was retired.

CMake identifiers: `project(peclet_core VERSION …)` with the version read from `pyproject.toml` (the
one version source); targets `peclet_core` / `peclet::core` (header-only) and `peclet_halo` /
`peclet::halo` (+ MPI, or the single-rank stub); `cmake --install` exports them for
`find_package(peclet-core CONFIG)`. Python modules: `cmake -S python -B build_rel_py -DCMAKE_PREFIX_PATH=../extern/install/host-openmp`
(→ `peclet.core.{mpi,geom}` under `build_rel_py/peclet/core/`, `PYTHONPATH=build_rel_py`); their
stubs are `python/packaging/core_<mod>.pyi` — regenerate with `python -m nanobind.stubgen` after
changing a binding.

## Architecture

Header-only under `include/peclet/core/`:

- `common/types.hpp` — `Index` (int64), `Real` (double), `IVec<Dim>`/`Vec<Dim>`, `wrap()`,
  compile-time `forEachInBox`. **Convention: x-fastest linear index** `I = x + y*nx + z*nx*ny`
  (matches flow and `../docs/CONVENTIONS.md`). Keep this header C++17-clean (shared with `morton`,
  which pins C++17).
- `decomp/block_decomposer.hpp` — ORB decomposition. `ownerOf()` walks the implicit binary tree
  (children at `2i+1`/`2i+2`, leaves carry the block index) and is the key primitive for halo
  topology. `linearGlobal`/`multiGlobal` are x-fastest and mutually inverse. `init(numBlocks,
  globalSize, weights)` is the **weighted ORB** for dynamic load balancing: it bisects at the cell
  boundary whose cumulative weight reaches the sub-block target fraction (vs equal cell count);
  equal weights reduce to the unweighted `init()` bit-for-bit.
  **Multigrid-safe partitions** come in two flavours. `init(…, align)` is the *aligned* ORB: it picks
  a split on the fine grid and snaps it to a multiple of `align[k]`, so `coarsened()` divides
  cleanly. `refined(ratio)` is the inverse of `coarsened()` and enables the stronger *coarse-first*
  route — decompose the grid coarsened `ratio` times, then refine the partition upward, so blocks are
  multiples of `ratio` by construction and the hierarchy nests for the full depth. Coarse-first also
  balances better, because the aligned ORB can round a balanced split into an unbalanced one (96|96
  → 128|64) while on the coarse grid one cell *is* the quantum. When the axes were coarsened by
  DIFFERENT factors, pass `init(…, align, cellExtent)` with `cellExtent = ratio`: the split-axis
  choice then compares physical extents (`size[k]*cellExtent[k]`) rather than cell counts, which is
  what stops the ORB bisecting an axis the fine grid would never have cut. flow drives all of this
  through `CutcellMG::decomposition()` (see `../flow/CLAUDE.md`).
- `decomp/block_indexer.hpp` — local↔global indexing for an extended (inner+ghost) block.
- `decomp/morton_indexer.hpp` — `MortonIndexer<Dim>`: Z-order (Morton) cell indexing via the `morton`
  primitive (`morton::Morton<Dim,Bits>`), guarded by `PECLET_CORE_HAVE_MORTON` — core's only morton
  consumer since the AMR tree left. The cache-friendly alternative
  to the x-fastest order (which stays the convention): `codeOf`/`multiIndex` map global multi-index ↔
  Z-order code, `neighborCode` steps one cell along an axis directly in Morton space. Methods carry
  morton's `MORTON_HD`, so they are device-callable under a Kokkos build (the Kokkos build defines
  `MORTON_ENABLE_KOKKOS` ⇒ `MORTON_HD` is `KOKKOS_FUNCTION`).
- `halo/nbx.hpp` — `NbxEngine`: canonical NBX (Issend + Ibarrier consensus, Hoefler et al.). Use for
  dynamic/sparse exchange.
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
  (count/payload preserved). The dem distributed step is the consumer; also bound in `python/mpi_bindings.cpp`
  (`peclet.core.mpi.ParticleMigrator.rebalance`).
- `halo/grid_halo.hpp` — `GridHalo<T>`: portable GPU-resident halo (Kokkos; CUDA / HIP / OpenMP
  backends). pack/unpack/self-copy run as `parallel_for` over the device `peclet::core::View<T>` field; only the
  compact halo buffers are host-staged for MPI by default (the field stays on the device), with an
  opt-in GPU-aware path (env `PECLET_CORE_GPU_AWARE_MPI`, legacy `PECLET_CORE_CUDA_AWARE_MPI` still honoured). Built
  from a host `GridHaloTopology<Dim>::flatten()` via `init()`. Bit-for-bit matches the CPU exchange.
  (`docs/archive/cuda-aware-mpi.md` holds the historical host-staging-vs-GPU-aware analysis and the
  user-space UCX + OpenMPI recipe that gives this box a GPU-aware MPI.)
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
- `vof/` — **layer L1 of the VoF stack** (`peclet::core::vof`, `../docs/archive/VOF_PLAN.md` §11), promoted
  out of `flow/src/vof/` by WO-W0 (2026-09-02) as a plain file move. Container-free
  `KOKKOS_INLINE_FUNCTION`s of scalars and small local arrays — **no `Kokkos::View`, no grid
  indexing, no halo types in any signature**, which is exactly what lets ONE copy serve all three
  VoF containers (flow's structured colour field, the per-bubble block container of Part III, and
  the AMR path). `vof/plic.hpp` is the PLIC toolbox (SZ2000 / Lehmann–Gekle plane↔volume,
  MYC and Youngs normals on a supplied 3³ array, slab flux volumes); `vof/curvature.hpp` the
  Popinet height-function cascade + the PLIC-volumetric paraboloid fallback on supplied column sums
  and points; `vof/cutcell.hpp` the cut-cell colour-transport rules (`eps_eff`, Weymouth's
  admissible flux interval on a cut donor, the solid-band fill state machine); `vof/wetting.hpp`
  the θ-consistent band-fill math. **Keep them container-free**: the drivers (`WyAdvector`,
  `VofCurvature`, `VofBlockSet`) stay in `flow`, and flow's `src/vof/{plic,curvature,cutcell,
  wetting}.hpp` are thin includes + a using-directive so every `peclet::flow::vof::` spelling still
  resolves. Gate on the move: every flow VoF ctest bit-identical, both backends.
- `solver/` — mesh-agnostic linear-algebra shared by the method codes. `solver/graph_amg.hpp` (+
  `graph_amg_device.hpp`) is the smoothed-aggregation graph AMG (host setup, device apply).
  `solver/face_csr.hpp`, `coloring.hpp`, `csr_operator.hpp`, `csr_bicgstab.hpp` and `vector_ops.hpp`
  are the **assembled face-CSR operator layer**, lifted VERBATIM out of `amr/` on 2026-09-10
  (QUALITY_PLAN G.2): the host+device row kernels (`FaceCsrOpT`, `FvCsrOpT`, `HostArr`), the greedy
  symmetrised graph colouring (`Coloring`, `greedyColoring`), the device operator + smoothers
  (`MomentumOp`, `applyMom`, `residualMom`, `jacobiMom`, `multicolorGSMom`), the preconditioned
  BiCGStab / defect-correction solver (`MomentumSolver` — no template parameter; the AMR `Bits` was
  vestigial) and the `axpy`/`zpby`/`negate`/`dotPlain`/`bicgPUpdate` primitives. The names keep their
  AMR spelling on purpose (a rename is not a structural move): voro's `mesh_optimizer.hpp` /
  `ot_optimizer.hpp` and the AMR package are the consumers. Gate: `tests/test_csr_solver.cpp`
  (Kokkos build) — host row kernel vs device matvec, proper colouring of an asymmetric CSR,
  smoothers, BiCGStab + defect correction.
- **`amr/` is gone (2026-09-10)** — the whole AMR tree (the block-local-Morton octree, the distributed
  octree with leaf halos and weighted-ORB rebalancing, the collocated-projection cut-cell
  Navier–Stokes solver, its tests, studies, docs and campaign logs) is the **`peclet-amr`** package,
  `../amr` (`peclet::amr`, Python `peclet.amr`), relocated with its git history under
  `suite/docs/QUALITY_PLAN.md` D6 / G.2. It depends on core + morton; core depends on nothing of it.
  What it left behind in core is the face-CSR solver layer under `solver/` (above) and
  `decomp/morton_indexer.hpp` as the one remaining morton consumer. Read `../amr/CLAUDE.md` for the
  AMR design, its two projection schemes and its environment-variable table.
- `python/` — **nanobind** Python bindings over a
  shared **zero-copy `peclet::core::View`↔ndarray bridge** (`include/peclet/core/python/ndarray_interop.hpp`).
  `python/mpi_bindings.cpp` (→ `peclet.core.mpi`) is host-only (no Kokkos): exposes `ParticleMigrator`
  (migrate / gather_ghosts / rebalance) and `ParticleHalo` (the persistent `ParticleHaloTopology`) for an
  mpi4py driver, both constructed as `(origin, extent, cells, periodic)`. `python/geom_bindings.cpp`
  (→ `peclet.core.geom`) is the analytic-SDF scene authoring. (`peclet.core.amr` is `peclet.amr` in
  the peclet-amr package since 2026-09-10.) Both are built via `include(SuiteNanobind)` +
  `suite_require_nanobind()` from `../cmake/SuiteNanobind.cmake` (suite-root). `python/state_hash.py`
  is the structural byte gate (QUALITY_PLAN §3.G): fixed-seed runs of every `peclet.core.{geom,mpi}`
  entry path, SHA-256 of the final state, `--check`ed against `python/state_hash_reference.json` by
  the `python_state_hash` ctest (np=1) and by hand under `mpirun -np 2`.

## Gotchas

- `GridHalo` caches a distributed-graph `MPI_Comm`. Its destructor guards `MPI_Comm_free` with
  `MPI_Finalized` so an instance that outlives `MPI_Finalize` (e.g. on `main`'s stack) does not abort.
  Don't remove that guard. The class is non-copyable (it owns the comm).
- **Consecutive NBX rounds on one communicator must use different tags**, and `NbxEngine` does
  that itself: `exchange(…, baseTag)` sends on a wire tag in the RESERVED range [24576, 32768)
  — block `baseTag % 128` (64 tags each), rotated by a round counter kept as an MPI attribute of
  the communicator (`detail::nbxRoundTag`). A rank that has observed the Ibarrier complete starts
  the next round while a neighbour is still probing the old tag and would receive the new message
  as an old one — with one topology build per multigrid level this lost up to all 26 send
  partners of a rank at 1536 ranks on Snellius (2026-09-02), hanging the first exchange or,
  worse, silently corrupting ghost values. Two rules: **direct point-to-point tags stay below
  24576** (the suite's: 0–63, AMR 11/41/45, particle 7502/7503/7603/7604, voro 7601, flow VoF
  4096–20479) and **distinct NBX call sites use baseTags distinct modulo 128** (0, 11, 7301, 7401,
  7402, 7411, 7501 today). Until 2026-09-05 the wire tag was `baseTag + round`, so the particle
  topology's second round (7501 + 1) matched the direct forwardPositions tag 7502 — `particle_halo_np8`
  hung on GitHub's oversubscribed 2-core runners (CI red from 09-04) — and family-0 rounds walked
  over the AMR gather tags: `amr_distributed_{fv,mg,graded_mg,openness,poisson}_np{4,8}` returned
  wrong ghosts (intermittent, 1–3 failures per run on a pristine main). Never call `MPI_Comm_free`
  on a communicator with a round in flight.
  `buildTopology` cross-checks promised against requested cells with one allreduce and throws
  on a mismatch, so a lost consensus message now fails at build time. Gate:
  `tests/test_nbx_rounds.cpp` (fails on a laptop with the rotation ablated).
- The halo is owner-based, not adjacency-based: a ghost cell maps to whichever rank owns its wrapped
  global cell, so it is correct for ORB's irregular block neighbours and any ghost width — no
  Cartesian-grid assumption.
- Tests are dependency-free (`tests/test_util.hpp`, non-zero exit on failure). MPI tests run under
  `mpirun` at several rank counts via ctest. **The launcher is pinned to the MPI we link**
  (`cmake/PecletCorePinMpiexec.cmake`, included by the root AND the python CMake): a foreign
  `mpiexec` on PATH (ParaView's) makes every rank a singleton, and N non-communicating copies
  "pass" — `build_rel_py` carried exactly that until 2026-09-08 (`python_amr_np2` = two singletons
  racing on one VTU). The Python tests now also exit non-zero when `comm.size` differs from the
  `PECLET_CORE_TEST_NP` ctest launched them with.
- `../cmake/SuiteNanobind.cmake` MUST be a CMake **macro**, not a `function()`: it sets/propagates
  variables (the located nanobind, the interpreter) into the including scope, which a function's nested
  scope would swallow. Keep `suite_require_nanobind` defined as a macro.

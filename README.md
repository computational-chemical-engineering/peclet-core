# core — `peclet-halo` (+ the `peclet-core` compatibility shell)

[![PyPI version](https://img.shields.io/pypi/v/peclet-halo.svg)](https://pypi.org/project/peclet-halo/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/peclet-halo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/computational-chemical-engineering/peclet-core/blob/main/LICENSE)
[![CI](https://github.com/computational-chemical-engineering/peclet-core/actions/workflows/ci.yml/badge.svg)](https://github.com/computational-chemical-engineering/peclet-core/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21132435.svg)](https://doi.org/10.5281/zenodo.21132435)

Shared infrastructure for the **peclet** suite (the suite-wide
[architecture](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/ARCHITECTURE.md), [conventions](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/CONVENTIONS.md),
[style](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/STYLE.md), [interfaces](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/INTERFACES.md) and [roadmap](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/ROADMAP.md)
live in the [umbrella repository](https://github.com/computational-chemical-engineering/peclet)).

It provides the pieces every method code (`flow`, `dem`, `voro`, …) should
share: a common MPI **block domain decomposition**, an efficient **asynchronous ghost-layer
exchange** (CPU + portable Kokkos GPU), **particle migration**, **dynamic load balancing**, unified
**SDF geometry** (`peclet::core::geom`), the **face-CSR solver layer** (`peclet::core::solver`),
the **coarse-level multigrid stages** that move a level onto fewer ranks, and **nanobind Python
bindings**. Header-only C++20 (the device side, compiled through Kokkos, is also C++20; only the
`morton` dependency pins C++17 — see [STYLE.md](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/STYLE.md)). The AMR octree and the
Navier–Stokes solver on it are the separate [peclet-amr](https://github.com/computational-chemical-engineering/peclet-amr)
package (`peclet::amr`, `peclet.amr`) since 2026-09-10 — relocated out of this tree with its history.

## Install

The Python surface is the `peclet.halo` module of the **peclet-halo** distribution. It links MPI,
so it ships as an sdist and builds against the MPI on your machine (an MPI C++ compiler wrapper and a
C++20 compiler; Linux):

```bash
pip install peclet-halo          # or: pip install "peclet[mpi]", which also brings peclet-geom
```

`peclet.halo` never initialises MPI itself: import `mpi4py.MPI` (built against the same MPI) first.
The C++ library is header-only — use it from a CMake project with `find_package(peclet-core CONFIG)`
after `cmake --install`, or as a sibling checkout in the suite.

## What works today

- `peclet::core::decomp::BlockDecomposer<Dim>` — orthogonal recursive bisection of a global cell grid into
  rank-owned blocks; `ownerOf()` tree-walk; x-fastest global/local linear indexing. The **weighted
  ORB** (`init(numBlocks, globalSize, weights)`) balances a per-cell work field; the multigrid-safe
  forms keep every block a multiple of an alignment so the grid coarsens cleanly — `init(…, align)`,
  the coarse-first `refined(ratio)`, and the **aligned weighted ORB** `init(…, weights, align)`, with
  `chooseAlignedWeighted` picking the deepest alignment within an imbalance budget
  (`weightImbalance`).
- **Coarse-level multigrid stages** (`decomp/stage_target.hpp`, `stage_comm.hpp`,
  `redistribute_topology.hpp`, `gather_by_global_id.hpp`) — where a multigrid level goes when it can
  no longer coarsen on its own decomposition, and how its fields get there and back.
  `chooseStageTarget` is the replicated policy (in place, a sibling merge of ORB subtrees, a
  repartition onto fewer ranks, or full replication), `makeStageComm` builds the stage's
  communicators, `RedistributeTopology` plans the movement once and replays it every V-cycle
  (gather/scatter, allgather or planned point-to-point), and `gatherByGlobalId` is the id-keyed
  replicated gather. Tested bitwise against flow's `Telescope` and peclet-amr's replicated tail
  stage, and against `redistributeGridFields`, at np 1–8.
  `redistributeGridFields` (`grid_redistribute.hpp`) is the one-shot move of grid fields between two
  decompositions.
- `peclet::core::decomp::BlockIndexer<Dim>` — local↔global indexing for a block with a ghost layer.
- `peclet::core::halo::NbxEngine` — nonblocking-consensus sparse exchange (Issend + Ibarrier), for dynamic
  patterns.
- `peclet::core::halo::GridHaloTopology<Dim>` (`grid_halo_topology.hpp`) — asynchronous ghost-layer exchange
  with **topology separated from exchange** and a **field-agnostic** pack/unpack interface.
  `buildTopology()` runs once; two interchangeable exchange engines give identical results:
  - `exchangeNbx()` / `start()`+`wait()` — NBX, supports compute/comm overlap.
  - `exchangePersistent()` — `MPI_Neighbor_alltoallv` on a cached distributed-graph communicator;
    fastest for the static neighbour pattern of a fixed grid.
- `peclet::core::halo::GridFieldView<T>` — wraps a contiguous local array as an exchangeable field.
- `peclet::core::halo::GridHalo<T>` (`grid_halo.hpp`) — portable **GPU-resident** ghost-layer exchange (Kokkos:
  CUDA / HIP / OpenMP). Built once from a host `GridHaloTopology<Dim>::flatten()`; pack / unpack /
  periodic self-copy run as `Kokkos::parallel_for` over the device `peclet::core::View<T>` field, so the full
  field never crosses the bus — only the compact halo buffers are host-staged for MPI by default, with
  an opt-in GPU-aware path (env `PECLET_CORE_GPU_AWARE_MPI`). Bit-for-bit identical to the CPU exchange.
- `peclet::core::halo::ParticleMigrator<Dim>` — Lagrangian particle migration to owning ranks (NBX), the
  dynamic counterpart to the Eulerian grid halo.
- `peclet::core::halo::ParticleHaloTopology<Dim>` (`particle_halo_topology.hpp`) — persistent Lagrangian ghost
  halo: `forward` (owner→ghost), `reverse` (ghost→owner, accumulate) and `forwardPositions` (periodic
  image shift). `peclet::core::halo::ParticleHalo<Dim>` (`particle_halo.hpp`) is its GPU-resident Kokkos driver
  (on-device gather/scatter, host-staged or GPU-aware MPI), consumed by dem's distributed step.
- `peclet::core::halo::rebalanceByParticleCount(...)` (`particle_rebalance.hpp`) — **dynamic load balancing**
  for the Lagrangian path: re-inits the decomposition in place with the **weighted ORB**
  (`BlockDecomposer::init(numBlocks, globalSize, weights)`) and migrates. The Eulerian/AMR counterpart
  is `peclet::amr::DistributedOctree::rebalance` (peclet-amr).
- `peclet::core::geom` (`sdf.hpp`, `grid_sdf.hpp`, `vti_io.hpp`) — shared SDF solids: analytic primitives +
  trilinear `GridSdf` behind one `Sdf` concept, with VTI (.vti) read/write.
- `peclet::core::vof` (`include/peclet/core/vof/`) — layer L1 of the VoF stack: container-free
  `KOKKOS_INLINE_FUNCTION` kernels (PLIC plane↔volume and normals, the height-function curvature
  cascade, the cut-cell colour-transport rules, wetting). No `Kokkos::View` and no grid indexing in
  any signature, so one copy serves every VoF container; the drivers live in `flow`.
- `peclet::core::solver` (`include/peclet/core/solver/`) — mesh-agnostic linear algebra: the
  smoothed-aggregation graph AMG (`graph_amg.hpp`, host setup; `graph_amg_device.hpp`, device apply)
  and the **assembled face-CSR operator layer** lifted out of the AMR tree on 2026-09-10 —
  `face_csr.hpp` (host+device row kernels), `coloring.hpp` (greedy symmetrised graph colouring),
  `csr_operator.hpp` (the device operator, Jacobi and multicolour Gauss–Seidel sweeps),
  `csr_bicgstab.hpp` (preconditioned BiCGStab / defect correction) and `vector_ops.hpp`. Consumed by
  voro's mesh optimiser and by peclet-amr.
- **Python bindings** (`python/halo_bindings.cpp`) — a host-only **nanobind** module (no Kokkos):
  **`peclet.halo`** exposes the Lagrangian halo (`ParticleMigrator`, `ParticleHalo`: migration /
  ghosts / count-weighted rebalance) to an mpi4py driver. The shared zero-copy Kokkos
  `View`↔ndarray bridge that flow, pnm, dem, coupling and peclet-amr bind through lives here too
  (`include/peclet/core/python/ndarray_interop.hpp`). Type stubs ship beside it (`python/packaging/_halo.pyi`,
  generated with `python -m nanobind.stubgen`). `python/state_hash.py` is the structural byte gate:
  fixed-seed runs of every entry path, SHA-256 of the final state.

  **The scene-authoring bindings left this repository in peclet 1.2.0** for
  [`peclet-geom`](https://pypi.org/project/peclet-geom/) (`peclet.geom`, wheels). They are
  host-only with no MPI in them, and while they shared this distribution they inherited its
  `find_package(MPI REQUIRED)` — so a pure-geometry API could not be installed without an MPI
  toolchain. The C++ headers `include/peclet/core/geom/` **stay here**; peclet-geom vendors them at
  `PECLET_CORE_TAG`. The old spellings keep working through the `peclet-core` compatibility shell
  until 2.0.0 — `peclet.core.geom` is now `peclet.geom`, `peclet.core.mpi` is now `peclet.halo`,
  and each pair is the same object. See
  [docs/CORE_BOUNDARY.md](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/CORE_BOUNDARY.md).

Validated end-to-end by distributed explicit heat-diffusion solvers (plain, and **around an SDF solid
obstacle**) matching a serial reference cell-for-cell across ranks, and consumed by the validated
`flow` and `dem` distributed solvers. 72 ctests in the plain host+MPI build, 87 with Kokkos (`np` 1–8),
plus 6 Python ctests in the `python/` build.

## Build / test / benchmark

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -LE bench  # 72 ctests: serial + MPI (np=1,2,4,8); 87 with -DPECLET_CORE_ENABLE_KOKKOS=ON
ctest --test-dir build -L bench                       # the halo benchmark (label `bench`)

# halo microbenchmark: weak scaling, NBX vs persistent
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300  # cells/rank/axis, ghost, iters
```

ctest labels: `mpi` (every mpirun test), `np8` (the 8-rank instances — a local gate, excluded in CI whose
runners have 4 cores), `bench` (benchmarks/studies, excluded by default). A test that cannot run in a
configuration (no `morton` sibling: `morton_indexer`) exits 77 and ctest reports it **skipped**, never
passed. The Python modules and their tests (`test_mpi.py` np=1,2,4,8, `state_hash.py`, the
ndarray-interop pytest) are a second CMake project: `cmake -S python -B build_py
-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<kokkos prefix> && cmake --build build_py -j && ctest
--test-dir build_py`. Kokkos is needed only for the ndarray-interop test; the build type matters
because the `state_hash` byte gate compares against a reference recorded with a Release build and
reports **skipped** for any other toolchain string.

Requires MPI (OpenMPI/MPICH) and a C++20 compiler (`-DPECLET_CORE_ENABLE_MPI=OFF` builds the
single-rank no-MPI stub). `morton` is picked up automatically if checked out as a sibling directory
(enables `PECLET_CORE_HAVE_MORTON`; `-DPECLET_CORE_MORTON_DIR=<dir>` points elsewhere). CI builds all
three configurations (host+MPI gcc/clang Debug/Release, Kokkos-OpenMP + Python, no-MPI) against the
pinned `morton` tag. The CMake project is
`peclet_core` (its version is read from `pyproject.toml`); it exports the header-only targets
`peclet::core` and `peclet::halo` (`cmake --install` + `find_package(peclet-core CONFIG)`).

## Documentation

The AMR reference notes and campaign records moved to
[peclet-amr](https://github.com/computational-chemical-engineering/peclet-amr) with the code.
What stays here is [docs/archive/](https://github.com/computational-chemical-engineering/peclet-core/blob/main/docs/archive/README.md)
(the GPU-aware-MPI recipe) and the Doxygen API pages (`docs/Doxyfile`, README + `include/`), published to GitHub Pages by
`.github/workflows/docs.yml`.

## Status

Complete and in production. The block decomposition, the async ghost-layer exchange (CPU + portable
Kokkos GPU, host-staged and opt-in GPU-aware), particle migration, dynamic load balancing (weighted
ORB + Lagrangian rebalancing), SDF geometry, the face-CSR solver layer, and the nanobind Python
bindings are all shipped and tested (counts above; `np` 1–8). The aligned weighted ORB and the
coarse-level multigrid stages are the newest pieces: tested here, bitwise against the stages they
generalise, and not yet adopted on the consumers' main branches. `flow` (distributed cut-cell IBM Navier–Stokes), `dem` (distributed XPBD
and Hertz–Mindlin with load rebalancing), `voro` and `peclet-amr` are its consumers. CUDA is retired;
Kokkos (CUDA / HIP / OpenMP) is the canonical device path. Remaining work is at-scale multi-GPU
tuning.

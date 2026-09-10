# core

[![PyPI version](https://img.shields.io/pypi/v/peclet-core.svg)](https://pypi.org/project/peclet-core/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/peclet-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/computational-chemical-engineering/peclet-core/blob/main/LICENSE)
[![CI](https://github.com/computational-chemical-engineering/peclet-core/actions/workflows/ci.yml/badge.svg)](https://github.com/computational-chemical-engineering/peclet-core/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21132435.svg)](https://doi.org/10.5281/zenodo.21132435)

Shared infrastructure for the **peclet** suite (see `../docs/` for the suite-wide
[architecture](../docs/ARCHITECTURE.md), [conventions](../docs/CONVENTIONS.md),
[style](../docs/STYLE.md), [interfaces](../docs/INTERFACES.md) and [roadmap](../docs/ROADMAP.md)).

It provides the pieces every method code (`flow`, `dem`, `voro`, …) should
share: a common MPI **block domain decomposition**, an efficient **asynchronous ghost-layer
exchange** (CPU + portable Kokkos GPU), **particle migration**, **dynamic load balancing**, unified
**SDF geometry** (`peclet::core::geom`), the **face-CSR solver layer** (`peclet::core::solver`) and
**nanobind Python bindings**. Header-only C++20 (the device side, compiled through Kokkos, is also
C++20; only the `morton` dependency pins C++17 — see `../docs/STYLE.md`). The AMR octree and the
Navier–Stokes solver on it are the separate [peclet-amr](https://github.com/computational-chemical-engineering/peclet-amr)
package (`peclet::amr`, `peclet.amr`) since 2026-09-10 — relocated out of this tree with its history.

## What works today

- `peclet::core::decomp::BlockDecomposer<Dim>` — orthogonal recursive bisection of a global cell grid into
  rank-owned blocks; `ownerOf()` tree-walk; x-fastest global/local linear indexing.
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
- **Python bindings** (`python/mpi_bindings.cpp`, `python/geom_bindings.cpp`) —
  **nanobind** modules over the shared zero-copy `View`↔ndarray bridge
  (`include/peclet/core/python/ndarray_interop.hpp`). `peclet.core.mpi` exposes the host Lagrangian halo
  (`ParticleMigrator`, `ParticleHalo`: migration / ghosts / rebalance); `peclet.core.geom` the analytic-SDF
  scene authoring + rigid-body mass properties. Type stubs ship beside the modules
  (`python/packaging/core_*.pyi`, generated with `python -m nanobind.stubgen`). `python/state_hash.py`
  is the structural byte gate: fixed-seed runs of every entry path, SHA-256 of the final state.

Validated end-to-end by distributed explicit heat-diffusion solvers (plain, and **around an SDF solid
obstacle**) matching a serial reference cell-for-cell across ranks, and consumed by the validated
`flow` and `dem` distributed solvers. 53 ctests in the plain host+MPI build, 68 with Kokkos (`np` 1–8), plus 6
Python ctests in the `python/` build.

## Build / test / benchmark

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -LE bench  # 52 ctests: serial + MPI (np=1,2,4,8); 67 with -DPECLET_CORE_ENABLE_KOKKOS=ON
ctest --test-dir build -L bench                       # the halo benchmark (label `bench`)

# halo microbenchmark: weak scaling, NBX vs persistent
mpirun -np 4 ./build/benchmarks/bench_halo 48 1 300  # cells/rank/axis, ghost, iters
```

ctest labels: `mpi` (every mpirun test), `np8` (the 8-rank instances — a local gate, excluded in CI whose
runners have 4 cores), `bench` (benchmarks/studies, excluded by default). A test that cannot run in a
configuration (no `morton` sibling: `morton_indexer`) exits 77 and ctest reports it **skipped**, never
passed. The Python modules and their tests (`test_mpi.py` np=1,2,4,8, `state_hash.py`, the
ndarray-interop pytest) are a second CMake project: `cmake -S python -B build_py
-DCMAKE_PREFIX_PATH=<kokkos prefix> && cmake --build build_py -j && ctest --test-dir build_py`.

Requires MPI (OpenMPI/MPICH) and a C++20 compiler (`-DPECLET_CORE_ENABLE_MPI=OFF` builds the
single-rank no-MPI stub). `morton` is picked up automatically if checked out as a sibling directory
(enables `PECLET_CORE_HAVE_MORTON`; `-DPECLET_CORE_MORTON_DIR=<dir>` points elsewhere). CI builds all
three configurations (host+MPI gcc/clang Debug/Release, Kokkos-OpenMP + Python, no-MPI) against the
pinned `morton` tag. The CMake project is
`peclet_core` (its version is read from `pyproject.toml`); it exports the header-only targets
`peclet::core` and `peclet::halo` (`cmake --install` + `find_package(peclet-core CONFIG)`).

## Documentation

The AMR reference notes and campaign records moved to peclet-amr with the code (`../amr/docs/`).
What stays here is [docs/archive/](docs/archive/README.md) (the GPU-aware-MPI recipe) and the Doxygen
API pages (`docs/Doxyfile`, README + `include/`), published to GitHub Pages by
`.github/workflows/docs.yml`.

## Status

Complete and in production. The block decomposition, the async ghost-layer exchange (CPU + portable
Kokkos GPU, host-staged and opt-in GPU-aware), particle migration, dynamic load balancing (weighted
ORB + Lagrangian rebalancing), SDF geometry, the face-CSR solver layer, and the nanobind Python bindings
are all shipped and tested (53 ctests plain, 68 with Kokkos; `np` 1–8). `flow` (distributed cut-cell
IBM Navier–Stokes), `dem`
(distributed XPBD with load rebalancing) are validated consumers. CUDA is retired; Kokkos
(CUDA / HIP / OpenMP) is the canonical device path. Remaining work is at-scale multi-GPU tuning.

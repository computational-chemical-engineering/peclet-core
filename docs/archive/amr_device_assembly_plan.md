# Plan: move AMR operator/geometry **assembly** onto the device (parallel)

## Why

The host/device kernel consolidation (`face_csr.hpp` + `advect_recon.hpp`, the Phase A/B/C work) made
the *application* of every AMR flow operator a single shared body that runs on host or device. What is
still **host-serial** is the *assembly* — turning the octree + SDF geometry into the per-cell diagonal,
the face CSR (`start/nbr/coef`), the openness weights, the cut-cell ξ-overlay stencils, the implicit-FOU
advection operator, and the face-geometry tables.

For a **static** simulation that is fine: assembly happens once in `setSolid`, then every step only
*applies* the operators (already device-resident on the device path). But for a **dynamic** simulation —
moving immersed boundaries, free surfaces, or solution-adaptive AMR (`adapt`) — the geometry changes and
the operators must be **re-assembled every step (or every adapt)**. Then the serial host assembly becomes
the bottleneck (and, on the device path, forces a host round-trip + re-upload every step). Assembly is
mostly **embarrassingly parallel per cell/face**, so it belongs on the device.

## Current assembly inventory

| Producer (host, serial) | What it builds | When (static) | When (dynamic) | Already on device? |
|---|---|---|---|---|
| `AmrCutCell::build` (`cut_cell.hpp`) | per-cell κ, cut flags, ξ-overlay stencil (`AC_/off_/nb_`), `D_rescale`, openness sampling from the SDF | once | **per geometry change** | no |
| `AmrCutCell::buildAdvectionFou` | implicit-FOU operator (`advDiag_/advStart_/advNbr_/advCoef_`) from the lagged velocity | — | **per step** | **yes** — `deviceBuildFou` (flow.hpp) |
| `AmrCutCell::assembleOperator` | the momentum `diag + face CSR` (folds viscous + cut + FOU) | once (device path) / per call (host) | **per step** | partial — device applies the static CSR + device FOU; full re-assembly is host |
| `AmrPoisson::buildOpenness` / `assembleFv` | pressure FV weight-CSR (`invVol/start/nbr/w/bcDiag`) | once | **per geometry change** | host only (device MG `buildFaceCsr` calls the host walk) |
| `AmrMultigrid` hierarchy build | coarsened octrees + per-level openness/κ + per-level CSRs | once | **per adapt** | no (host coarsening + per-level host assembly) |
| `buildFaceGeom` (flow.hpp) | `DeviceFaceGeom` (area/openness/dist/upstream-probe face tables) | once | **per geometry change** | host-side build, then upload |
| `BlockOctree::adapt` / coarsen | the leaf set + Z-order ordering itself | — | **per adapt** | no (inherently host topology) |

The single most important dynamic-path cost is the **re-assembly of the cut-cell stencils + operator CSR
+ face geometry** whenever the SDF moves, and the **MG hierarchy rebuild** on `adapt`.

## Parallelizability

- **Embarrassingly parallel (per cell or per face)** — ideal for `parallel_for`:
  - SDF sampling at cell/face centres; κ and openness per face; the per-cell ξ-overlay stencil
    (`buildCutStencil` is already a self-contained per-cell function of the 6 neighbour SDFs);
    `D_rescale`; the FOU velocity-out per face; face-geometry tables (area/dist/upstream probe).
  - The FOU build is the proof of concept: `deviceBuildFou` already does this per face on the device.
- **Parallel after a prefix sum (scan)** — the CSR layout:
  - `start[]` (row offsets) is a prefix sum of per-cell face counts. Counting per cell is parallel; the
    offsets need a **device scan** (`Kokkos::parallel_scan`); then the fill is parallel per cell.
  - This is the standard two-pass parallel CSR assembly (count → scan → fill) and applies to the momentum
    CSR, the FV CSR, the FOU CSR, and the face-geometry CSR.
- **Inherently serial / host-resident (leave on host, or a separate effort)**:
  - Octree **topology mutation** (`adapt`/coarsen rebuilds the sorted leaf arrays) — pointer/sort work.
    Keep on host; only the *fields* migrate (the existing distributed `rebalance` already does leaf/field
    migration). Re-deriving the neighbour structure on device is a larger, separate project.
  - The greedy GS **colouring** (host, already) — cheap, rebuilt on adapt; can stay host.

## The enabler already in place

Because Phase A/B/C made the operators **apply** through `FaceCsrOpT` / `FvCsrOpT` / `hoFaceValue`, a
device-assembled CSR feeds the *same* shared kernels with **zero host round-trip**: assemble on device →
apply on device. Today the device flow path still calls host `assembleOperator` (once) + `buildFaceGeom`
(once) and only `deviceBuildFou` is per-step on device. Completing device assembly closes the loop:
fully device-resident dynamic geometry.

## Phased plan

- **D1 — Device CSR-fill primitive.** A small reusable `count → Kokkos::parallel_scan → fill` helper that
  turns a per-cell "emit faces" device functor into a `start/nbr/coef` CSR on the device. This is the
  backbone for every assembler below. Validate against the host `assembleOperator` CSR (bit-exact on
  OpenMP for the same emit order).
- **D2 — Device FV (pressure) assembly.** Port `AmrPoisson::assembleFv` to a device kernel over the
  octree (`DeviceBlockOctree::faceNeighbor` already exists): emit `(nbr, w=α·A/d)`, `invVol`, `bcDiag` per
  cell, then D1. Replaces the host `buildFaceCsr` walk in the device MG build. Lock: device-assembled FV
  CSR == host `assembleFv` (the Phase B anti-drift test, now cross-backend).
- **D3 — Device cut-cell stencil + momentum assembly.** Port `buildCutStencil` (already a pure per-cell
  function of the 6 neighbour SDFs) + the regular-fluid stencil + `assembleOperator` to device kernels:
  per-cell κ/openness/ξ-overlay/`D_rescale` (parallel) → emit faces → D1. This is the big one and the main
  dynamic-sim win. Lock against host `assembleOperatorGeometric`.
- **D4 — Device face-geometry assembly.** Port `buildFaceGeom` (area/openness/dist/`upupI/upupJ` probes)
  to a device kernel — pure per-face geometry. Lock against the host build.
- **D5 — Device MG hierarchy rebuild on adapt.** The per-level operators rebuilt from the (host-mutated)
  coarsened octrees via D2/D3 device assembly; only the topology stays host. Hook into the existing
  `adapt`/`rebalance` path so an adaptive step never round-trips the operators through the host.
- **D6 — Wire the dynamic loop.** `DeviceAmrFlow::setSolid`/`step` call the device assemblers when the
  geometry/topology changed this step (a dirty flag), so a moving-boundary or adaptive run reassembles
  entirely on the device. Benchmark vs the host-assembly round-trip at 64³/128³.

## Risks / gotchas

- **Emit order must match** the host for the bit-exact OpenMP locks (the CSR `nbr` order and the
  summation order in the apply). Keep the device emit in the same axis/dir order as
  `forEachFaceNeighbor` / `forEachFaceFull`.
- **Scan determinism**: `parallel_scan` offsets are deterministic; the *fill* must write each cell's faces
  to its own `[start_i, start_{i+1})` slice (no atomics) to stay reproducible.
- **GPU is not host-bit-exact** (FMA) — keep the assembly locks tolerance-based on CUDA/HIP, bit-exact on
  OpenMP (the existing convention).
- **`adapt` topology** stays host for now; D5 only moves the *operator* assembly, not the leaf-set
  mutation. Full device topology is a separate, larger project (device octree refine/coarsen + neighbour
  rebuild) and should not be bundled here.
- Start with **D1+D2** (smallest, the FV path is the simplest operator) to prove the device CSR-fill
  primitive before the cut-cell stencil port (D3).

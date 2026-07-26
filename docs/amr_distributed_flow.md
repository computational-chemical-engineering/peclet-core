# Distributed AmrFlow — design (D0 audit + plan)

Status 2026-07-26: rungs 1–3(aperture) SHIPPED —
* **Rung 1** `leaf_halo.hpp`: LeafHalo registry (miss-collect coverLevels fixpoint, canonical
  covering-leaf-anchor dedup) + topology-once value exchange (host + device LeafHaloExchange,
  batched 3-component). ctests bit-exact vs the coverValues oracle np=1..8, host/OpenMP/CUDA.
* **Rung 2**: the resolver seam through the EXISTING host builders (AmrPoisson
  setResolver/setGhosts/setFrameShift + probeSlot + ghost-aware levelOf/loOf/periodicNeighbor
  + ghost α rows; AmrCutCell mirrors it and fills ghost sdfC/fluid) — all world-coordinate
  evaluation in the GLOBAL frame (bit-identical geometry samples on every rank).
  MomentumSolver::setDistributed (halo per matvec/sweep, injected dot reduction, extended
  scratch). test_amr_distributed_momentum_mpi: CSR row-for-row BIT-EXACT vs single-rank on a
  graded sphere mesh; np=1 solve bit-exact, np>1 Krylov tolerance.
* **Rung 3 (aperture)** `distributed_flow_mg.hpp`: DistributedFlowMultigrid — coarsened
  COPIES of the flow octree (decomposition preserved), per-level LeafHalo built by the same
  fixpoint (ghosts must be re-installed into ap at the top of EVERY round — same-round
  registry hits read levelOf), level-count padding to the global max, area-averaged openness
  (local rows = exact single-rank arithmetic; GHOST α rows owner-exchanged once per build),
  Allreduce'd removeMeanFvDist. PCG::setDistributed (sync direction per matvec, reduced dots,
  removeMeanVolReduced). test_amr_distributed_flow_mg_mpi: V-cycle WORLD==SELF **bit-exact**
  (mean removal off) np=1..8 OpenMP+CUDA; distributed MG-PCG np=1 bit-exact vs single-rank,
  np>1 tolerance.
* **c2p transfer fix (measured)**: the ancestor(level+1)+find parent map mis-parented
  root-level rows in mixed-depth ladders (49/56 on a graded 16³ probe) and was
  block-alignment-dependent; replaced suite-wide (Multigrid, MomentumMG, AmrMultigrid) by the
  covering construction c.find(f.code(i)). Converged solutions unchanged; graded V-cycle
  rates shift (kappa experiment 0.49→0.65/cyc — the old map accidentally aggregated root
  bricks further; a PRINCIPLED super-root coarsening is the perf lever if bottom-solve
  strength ever limits).
* Remaining: ghost pressure + full step wiring (rung 4, inside AmrFlow via initMpi), NS
  advection halo (rung 5), adapt/rebalance + distributed pocket guard (rung 6), perf (7).

Original plan (2026-07-25, ladder step 0) below. The goal is the distributed (MPI)
`peclet::core::amr::AmrFlow`: the full ghost-projection NS step — momentum, pressure,
overlays, adaptivity — running multi-rank on the ORB block decomposition, validated by the
suite's contract (np=1 bit-exact vs single-rank on OpenMP; WORLD==SELF for order-independent
pieces; Z&H drag at np=2,4 equal to the single-rank values; end gate = the adaptive Z&H
battery distributed).

## 1. Audit: what exists, and what it gives us

* **`DistributedOctree`** (distributed_octree.hpp) — the mesh side is complete: ORB root
  bricks, cross-block 2:1 `balance`, `rebalance` (weighted ORB + leaf/field migration),
  owner-based gathers. Three gather layers matter here:
  - `coverLevels` / `coverValues`: by-coordinate owner request/reply (coords travel every
    call) — the **build-time** query primitive.
  - `FaceGatherPlan` (C1): the ±1 face classification cached once; values-only per call.
  - `GatherHaloTopology` + **`DistributedGatherHalo`** (C2, distributed_view.hpp): topology
    established ONCE (one NBX round does `locateGlobal` on the owner), then each exchange
    moves only compact `double` buffers, device pack/scatter, host-staged MPI (GPU-aware
    opt-in). **`buildGatherHaloTopology` is already generic over any
    (remoteCoords, remoteSlot) list** — it is not limited to the ±1 face plan it is fed
    today. This is ~80 % of the LeafHalo.
* **`DistributedFvOperator`** (distributed_fv.hpp, host) — the architecture template: the
  graded consistent FV Laplacian with openness, where cross-block neighbours become **ghost
  entries in the CSR** (`ref >= nLocal` indexes a deduped ghost-coord list), remote levels
  come from one `coverLevels` round, and openness needs **no exchange at all** because the
  openFn is a symmetric world-coordinate callable (both sides sample the same face centroid —
  the Phase-6k trick). Bit-identical WORLD==SELF because per-row face order is preserved.
* **`GradedDistributedMultigrid`** (host) — the graded+openness distributed MG algorithm:
  per-rank `coarsenIf` hierarchy down to the root brick, **level-count padding to the global
  max** (identity root-brick levels — otherwise the per-level collective gathers deadlock at
  np>1), local transfers (parents never cross root cells), Jacobi smoothing (previous-iterate
  ⇒ order-independent), openness bottom = Jacobi on the coarsest openness-carrying operator.
  Bit-exact WORLD==SELF.
* **`DistributedPoissonView` / `DistributedMultigridView`** (C2, device) — device-resident
  distributed Jacobi/MG, but **plain-Laplacian, uniform (lmax=0) only**. So Phase 3 C2 is
  *not* the pressure story itself; it contributes the device halo pattern and the
  bit-exactness argument. The device distributed **graded openness** MG does not exist yet —
  it is part of this campaign (§4b).
* **`AmrFlow`** (flow.hpp) — everything the time step runs is a **free kernel over CSR data**
  (`FaceGeom`, `MomentumOp`, `GhostOverlayDev`, `GhostGradOverlay`, `CfCsrDev`): divergence,
  grad3, buildFou/deferredSou, buildFaceField, the overlay deltas, the BiCGStab bodies. None
  of them care whether a column index is < nLocal — **they work unchanged on ghost-extended
  arrays**. The distribution problem is therefore (i) the halo, (ii) the *builders'*
  neighbour resolution, (iii) the solvers' reductions/guards.
* **Builder resolution sites are few**: `AmrPoisson::periodicNeighbor` + the
  `t.find(encode(probe))` walks in `poisson.hpp` (forEachFace*), `cut_cell.hpp:~664`,
  `cf_scheme.hpp` (tangential + finer-children probes), `buildGhostOverlay` /
  `buildGhostGradOverlay` (±2 chains via periodicNeighbor), `buildFaceGeom` (upup ±2 probes).
  All funnel through "resolve a global cell coordinate to a leaf".
  The **host** builders are the complete, parity-locked set (the D3–D6 device assemblers are
  a performance mirror); distributed mode can build on host + upload, exactly like the
  single-rank oracle path.

## 2. Architecture (the fork to confirm)

**Ghost-slot extended arrays + a ±2 LeafHalo — the DistributedFvOperator pattern
generalized. NOT a ghost-extended local octree** (block-local Morton codes make a unified
`find` across blocks ugly; multi-hop probes are just more global coordinates, so no ghost
tree is needed).

Every per-leaf field View gets ghost slots appended: locals `[0, nLocal)`, ghosts
`[nLocal, nLocal + nGhost)`. Every CSR the step reads (`MomentumOp.faceNbr`, `FaceGeom.nbr`
+ `upupI/upupJ`, overlay `nbr` chains, cf CSRs) may reference ghost slots. Kernels launch
over `[0, nLocal)` rows only and are otherwise **unchanged**.

### 2a. LeafHalo (the one new primitive)

Built once per mesh (after `setSolid`'s builders have registered every ghost):

1. **Registry**: dedup map global-coord → ghost slot, filled by the builders (§3). Per-ghost
   metadata kept host-side: coordinate, level (from `coverLevels`), cell-center SDF sample
   (evaluated **locally** from the world-coord sdfFn — no exchange), fluid flag.
2. **Topology**: feed (ghostCoords, slot=nLocal+g) through
   `DistributedOctree::buildGatherHaloTopology` (verbatim — it is already coordinate-generic);
   one NBX round resolves owner leaves.
3. **Exchange**: `DistributedGatherHalo`-style device pack → compact MPI (host-staged,
   GPU-aware opt-in) → device scatter into the extended View's ghost tail. Add a **batched
   K-component variant** (u0,u1,u2 in one message) — the projection exchanges 3-vectors at
   every sync point and per-field latency would triple the halo count.

The ±2 depth is **not geometric**: the registry contains exactly the coordinates some local
row's stencil actually touches (closure chains ±2, SOU upup ±2, cf tangential ±1-of-face-
neighbour, ghost gradient ±2), discovered during the builds. Nothing wider is fetched.

### 2b. Where it lives

`AmrFlow` gains an **optional distributed context** (`initMpi(DistributedOctree&)`;
null ⇒ today's single-rank behaviour, bit-identical — the guards compile to no-ops on the
null path). One `step()` driver, one Picard loop, one projection tail, one adapt path —
no second driver to drift. New headers keep flow.hpp from bloating:
`amr/leaf_halo.hpp` (registry + halo), `amr/distributed_mg.hpp` (§4b). The alternative — a
separate `DistributedAmrFlow` — was rejected for driver drift risk (same reason the shared
`demSolveContacts` driver exists in dem).

## 3. The resolve seam in the builders

Single-rank builders resolve probes via the local `BlockOctree::find` /
`AmrPoisson::periodicNeighbor`. Distributed, a probe that leaves the block resolves via the
registry: `resolve(globalCoordProbe) → local leaf | ghost slot`, with `level(slot)` and
`sdfC(slot)` served for ghosts from the registry metadata (builders read neighbour levels
and SDF samples — e.g. the overlay's same-level assertion and float classification).

Mechanically: **one resolver callable threaded through the existing host builders** (an
optional parameter defaulting to the local-only resolver — single-rank call sites unchanged,
np=1 bit-exact by construction). NOT duplicated distributed builders: DistributedFvOperator
already had to replicate AmrPoisson's face enumeration order by hand for one operator; doing
that for the full builder set (cut cells, overlays, 5 cf CSRs, FaceGeom) is the drift risk.

**Build rounds**: `coverLevels` answers are needed *during* enumeration (is the across-face
leaf finer/coarser? does the ±2 chain stay same-level?). Rather than hand-staging two-pass
builds per builder, run the builds to a **miss-collect fixpoint**: attempt the build with the
current registry, record unresolved coords as misses, one collective `coverLevels` round for
the misses, re-run. Reach is bounded (±2) so ≤ 3 rounds; each round ends with an Allreduce of
"any rank still missing" so the collectives stay matched. Build time is once-per-setSolid;
simplicity wins over cleverness here.

## 4. Solvers

### 4a. Momentum (phase 1: correctness, np-invariant)

`MomentumSolver` gets an optional exchange hook + communicator: halo-exchange the iterate
before **every** `applyMom`/`residualMom`/Jacobi sweep; `dotPlain` → Allreduce'd dot.
* **Phase-1 preconditioner = Jacobi sweeps with halo per sweep**: previous-iterate Jacobi is
  order-independent, so the whole preconditioned BiCGStab iterate sequence is np-invariant
  up to dot reduction order — the validation-friendly configuration.
* **Perf option (flagged): rank-local MomentumMG V-cycle** (additive-Schwarz-ish, zero comm
  inside the preconditioner; iterations may grow with np, converged step unchanged).
* **Galerkin RAP across ranks is explicitly NOT phase 1.**

Advection: `uf` is per-LOCAL-face (each rank owns its rows' faces) — no face halo. Exchange
uⁿ (batched 3-vector) before `buildFou`/`deferredSou`; the upup probes read ghost slots.

### 4b. Pressure

* **Aperture path first**: a device distributed graded openness MG — the
  `GradedDistributedMultigrid` structure (per-rank `coarsenIf` levels, level-count padding,
  local transfers, Jacobi smoother) on device data, with each level's operator a ghost-slot
  face CSR + its own LeafHalo(±1), openness from the symmetric world-coord openFn (no
  exchange). Mean removal / maskSolid sums → volume-weighted Allreduce. PCG: Allreduce dots,
  halo before matvecs. V-cycle bit-exact WORLD==SELF (the DistributedMultigridView argument
  carries over level by level).
* **Then ghost**: `ghostMatvec` = binary-L (distributed MG level 0 op) + `ghostApplyDelta`
  (overlay rows are local; chains may read ghosts) + `gpProject` with **global**
  volume-weighted mean over the coupled mask. `solveGhostBiCGStab`: halo before each matvec,
  Allreduce dots; the preconditioner (2 binary V-cycles) rides the distributed MG.

### 4c. Generalization: multigrid levels and colored GS

* **MG levels**: the registry/halo machinery is level-agnostic — every level's operator is a
  ghost-slot CSR with its own LeafHalo built from its own columns. Parents never cross ranks
  (block boundaries are root-cell boundaries) ⇒ transfers stay local at every level; the
  level-count padding keeps collectives matched; Jacobi-smoothed V-cycles stay bit-exact
  WORLD==SELF. **Galerkin RAP distributes exactly**: a coarse row sums only over local
  children, every A[i][j] it needs is a local fine row's entry, and a ghost column's parent
  coordinate is computable locally from registry metadata (→ a coarse-level ghost). The
  distributed coarse operator is arithmetically identical to the serial RAP, so the true
  Galerkin momentum preconditioner is a phase-2 *effort* item (per-level halos on the
  RAP-fill CSRs), not an architectural compromise. Stencil reach stays ~±2 per level under
  child-average R / piecewise-constant P.
* **Colored GS** (opt-in smoother, `setMomentumGS`): **processor-block GS** — rank-local
  coloring, ONE halo exchange per sweep; local edges get true GS ordering, cross-rank edges
  see start-of-sweep values (dem step_mpi precedent). The smoother is then np-dependent ⇒
  WORLD==SELF bit-exactness is lost for GS-smoothed paths — acceptable because GS appears
  only inside preconditioners (converged step unchanged; np=1 stays bit-exact since there
  are no cross-rank edges). True distributed GS (halo per color + a deterministic
  np-invariant coloring from global coords/level) remains possible on this architecture if
  ever needed; not the default (12–16 exchanges per symmetric sweep).

### 4d. Collective agreement (deadlock guards)

* The overlay build's band-margin THROW and the auto-mode aperture fallback: Allreduce the
  violation flag — **all ranks throw or all fall back** (one rank alone deadlocks the MG
  collectives).
* `findPocketCells`: pockets span ranks → distributed label propagation (local CCL, exchange
  boundary labels over the halo, min-label merge, Allreduce changed-flag to fixpoint), then
  global component sizes by Allreduce. Needed before ghost mode meets fragmenting geometry;
  the Z&H battery does not fragment, so this lands with its own ctest, not blocking rung 3.
* Stagnation/early-break decisions inside BiCGStab/PCG are made from Allreduce'd scalars
  only — every rank takes the same branch by construction. Audit each `break`.

## 5. Adapt / rebalance integration

`distributedAdapt` + `DistributedOctree::rebalance` already migrate mesh + field columns.
The flow-side pattern mirrors single-rank `beginAdapt`/`finishAdapt`: snapshot fields (as
rebalance columns), mutate/rebalance externally, conservative `transferField` block-locally
(when ownership kept; rebalance carries columns itself), full distributed `setSolid` rebuild
(new registry, new halos — everything topology-derived is rebuilt). The measured p-remap
lesson stands: at steady dt, zero p after adapt and re-accumulate.

## 6. Ladder (commit per validated rung)

1. **LeafHalo**: registry + topology + device exchange (+ batched variant); ctest bit-exact
   vs a serial gather and WORLD==SELF at np=1,2,4.
2. **Distributed momentum**: resolver through AmrCutCell/FaceGeom builds; BiCGStab with halo
   per matvec; matvec bit-exact vs single-rank at np=1, solves to tolerance np=2,4
   (Poiseuille + sphere).
3. **Distributed pressure — aperture**: device distributed openness MG + PCG + maskSolid
   Allreduce; V-cycle WORLD==SELF bit-exact; then **ghost**: overlay in the matvec +
   collective guards; ghost BiCGStab np=2,4.
4. **Full distributed Stokes step**: Z&H np=1,2,4 == single-rank (−0.056 % ghost N=64 band).
5. **NS**: uⁿ halo before advection builds; upup rides the halo; impulsive-start trajectory
   parity.
6. **Mid-run adapt + rebalance**; end gate = distributed adaptive Z&H battery
   (tests/study/amr_adaptive_zh.py pattern), dense + dilute.
7. **Perf** (after correctness): halo/compute overlap (GridHalo begin/end split pattern),
   batched exchanges, reduction-light Krylov variants.

## 7. Standing pitfalls (inherited, all previously hit)

`Kokkos_ENABLE_CUDA_CONSTEXPR` ON in the CUDA install; ghost slots of φ/p refreshed before
every gradient/faceField read (stale ghosts = flow's free-variable bug in halo form); GPU
tolerance-not-bit-exact (OpenMP carries determinism); `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`;
distinct NBX tag pairs per gather round; `flow` and `peclet.core.amr` never share a process.

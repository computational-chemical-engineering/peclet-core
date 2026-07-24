# AMR collocated projection: solid-pressure handling, divergence-free face field, `uf` advection

Status as of 2026-06-28. All work below is **done, validated, and pushed** (core `main`;
umbrella `peclet` pointer bumped each step). This doc is the resume point — read it before continuing.

## TL;DR

The AMR `AmrFlow` (cell-centred / collocated cut-cell Stokes–NS on a `BlockOctree`) now has the three
pieces flow's collocated solver already had, so it is robust and conservative:

1. **`maskSolid`** in the pressure PCG — projects out the per-solid-region pressure null modes. Fixes
   the cut-cell Stokes blow-up. (commit `3aff1ab`)
2. **ABC/Basilisk divergence-free FACE field** `uf` — built each projection, exactly `∇·uf = 0`,
   correct across 2:1 interfaces. Host oracle `c37c23b`, device `4924129`.
3. **`uf` as the NS advecting velocity** — conservative advection (Bell–Colella–Glaz), consistent across
   implicit-FOU + explicit-FOU + SOU. Host + device `9faf5d8`.

Canonical engine = device `peclet::core::amr::AmrFlow` (`include/peclet/core/amr/flow.hpp`), Python-exposed as
`peclet.core.amr.Flow`. The serial host driver is `peclet::core::amr::oracle::AmrFlow` (`flow_oracle.hpp`) — dev-only
validation oracle, **not** exposed.

## The physics that drove this (Frank's framing — keep)

The cut-cell **openness** `α` (per-face fluid fraction; a face whose centroid is in solid gets `α=0`)
**decouples fluid from solid**. The pressure operator `L = div(α grad)` then splits into:
- the **connected fluid** system — well-posed up to ONE constant (its total pressure drifts → remove the
  mean), and
- **inert solid cells**, each connected solid region carrying **its own constant null mode**.

The solid pressure is **not solved** — it is **pinned to 0 and projected out**. flow does this with
`mg_mask_solid_k` (`maskSolid`): zero every `AC ≤ 1e-30` (solid, all faces closed) cell, applied
**together with `removeMean` after every matvec / seed / residual**, so the Krylov iteration lives
strictly in the fluid range with both null modes deflated. This machinery is **shared by flow
staggered AND collocated** (`buildCutcellOp` + `cutcellSmoothColor` + `maskSolid` + `removeMean`); only
the velocity divergence/correction differs (faces vs cells).

The AMR PCG (`pcg.hpp`) had only `removeMeanVol` (the fluid constant), **no maskSolid** ⇒ CG amplified
the solid / near-disconnected-fluid null modes ⇒ the Stokes pressure → ~1e24 over a few hundred steps.
(The V-cycle/GS path already skipped `diag==0`, which is why advection-via-V-cycle stayed bounded — only
the Krylov path needed the explicit projection.)

## What each piece is

### 1. maskSolid (`include/peclet/core/amr/pcg.hpp`)
`buildFluidMask(op, mask, n)` sets `mask(i)=1` where the operator diagonal `Σ_f w_f + bcDiag > 1e-30`,
else 0. `project(u)` = `maskSolid` (zero solid) + (singular only) fluid-only `removeMeanVol`. The mean is
taken **over fluid cells only** — the old `removeMeanVol` averaged over ALL cells incl. the pinned solid,
diluting the mean and letting the solid drift. `project` is applied to: initial residual, each iter
residual, preconditioned `z`, `Ap`, and the final `x`.

### 2. Divergence-free face field (`flow_oracle.hpp::buildFaceField`, `flow.hpp::deviceBuildFaceField`)
After the pressure solve, before the cell-gradient correction (so `u_` still holds `u*`):
```
uf_f = ½(u*_i + u*_j) − (φ₊ − φ₋)/d_f         (+axis face velocity)
```
over the `forEachFaceFull` (sub)faces. Because `L = D·G_face` on the SAME faces,
`D(uf) = D u* − Lφ = 0` to the φ-solve residual — exactly divergence-free, unlike the cell field's
O(h²) approximate-projection residual.

**2:1 faces (the asked-for hard part):** `uf` lives on the **finest** face touching each interface — a
coarse cell owns its `2^(D-1)` fine sub-faces (fine area, `(φ_fine−φ_coarse)/d`), the fine cell owns its
single face. The **orientation-based (+axis) build keeps the two incident copies identical**, so there
is NO separate coarse-face value and NO face restriction/prolongation; `D`, `G_face`, `L` stay the
conservative consistent triple across the interface. Stored as a CSR (`faceStart_`/`uf_`) parallel to
`forEachFaceFull`; the device reuses `FaceGeom`'s existing CSR.

Validated (`tests/test_amr_face_field.cpp`): uniform N=16 `div(uf)` ≈ 4e-6 (host) / 5e-14 (device — the
MG-PCG solves φ to 1e-10) vs cell ≈ 1.4e-3; graded 2:1 `div(uf)` ≈ 2e-6 vs cell ≈ 2e-2.

### 3. uf advection
Replaced the advecting velocity `velOut = dir·½(u_i+u_j)` with `dir·uf` in ALL of: host `advectHO` +
`AmrCutCell::buildAdvectionFou`; device `deviceBuildFou` / `deviceDeferredSou` / `deviceAdvectExplicit`.
The implicit FOU, explicit FOU and explicit SOU must use the SAME velocity so the FOU cancels at steady
state. A `faceFieldBuilt_` flag falls back to `½(u_i+u_j)` before the first projection (keeps the
advection unit test, which sets `u` directly with no projection, unchanged).

**Important:** at steady state `φ→0 ⇒ uf = ½(u_i+u_j)`, so the **steady solution is unchanged** — the
gain is transient/unsteady conservation. So `uf`/`maskSolid` do NOT move the steady Z&H k; they are
robustness + conservation enablers. (This is why the staircase MG now survives at scale.)

### Rhie–Chow note (Frank corrected an earlier sloppy claim)
There is **no Rhie–Chow update of the cell centres**. Cells get the plain cell pressure gradient
(`gradOf` = ½(g⁻+g⁺), already there). Rhie–Chow is purely the FACE term — the compact `(φ_j−φ_i)/d` vs
the averaged `½(g_i+g_j)` — and is automatic once `uf` is kept separate from `interp(u)`.

## The comparison: flow collocated vs uniform AMR (the "where do we stand" run)

Z&H SC sphere, φ=0.125, K_exact = 4.292, **RTX 5080 (CUDA)**. AMR = device `AmrFlow` with `uf` +
`maskSolid`, staircase velocity-MG + multicolour-GS (Galerkin gives the same converged k).

| N   | flow collocated | AMR (uf, staircase/Galerkin MG) | gap   |
|-----|-------------------|----------------------------------|-------|
| 32  | 4.3313 (+0.92%)   | 4.3048 (+0.30%)                  | 0.62% |
| 64  | 4.3148 (+0.53%)   | 4.2977 (+0.13%)                  | 0.40% |
| 128 | 4.3070 (+0.35%)   | 4.2943 (**+0.05%**)              | 0.30% |

Findings:
- **AMR is closer to Z&H at every N and converges faster** (~order 1.3 vs flow ~0.7). At 128³ AMR is
  essentially exact (+0.05%) vs flow +0.35%.
- The gap **shrinks with N** (0.62→0.40→0.30%) — both → the same continuum; the difference is the
  projection STRUCTURE (AMR genuinely cell-centred FV vs flow's staggered-MAC-heritage ABC
  projection), now seen across resolution. **NOT yet isolated to a single line of code.**
- **Staircase velocity-MG is fixed at scale**: no longer diverges at 64³ (that was the pre-maskSolid
  limit recorded in [[amr-gpu-smoother-flow-port]]); runs at 128³ too, just slowly (bounded to the
  sphere's feature depth → many momentum iterations).
- **Perf caveat:** flow is faster in wall-clock at 128³ (~121 s to converge vs ~704 s for AMR's 150
  steps). The AMR velocity-MG + GS momentum solve is the cost — the obvious thing to profile if making
  AMR competitive at scale matters.

## How to reproduce (GPU) — and the gotchas (these cost time)

Build the CUDA `peclet.core.amr` Python module:
```bash
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd core/python
PYBIND=$(cd ../../flow && .venv/bin/python -m pybind11 --cmakedir)
PYEXE=$(cd ../../flow && .venv/bin/python -c "import sys;print(sys.executable)")
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/../../extern/install/nvidia-cuda;$PYBIND" \
  -DPython_EXECUTABLE="$PYEXE" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF        # ← REQUIRED: LTO+nvcc fatbin clash
cmake --build build_cuda --target tpx_amr -j
```
- **LTO must be OFF for the CUDA build** or the link dies with `symbol 'fatbinData' is already defined`
  / `lto-wrapper failed`. (The OpenMP/Serial builds are fine with LTO on.)
- **Run GPU scripts in the FOREGROUND.** At interpreter exit the CUDA module throws a benign
  `cudaErrorCudartUnloading` ("driver shutting down") → non-zero exit. Harmless for the computation, but
  in a *background* task it loses the buffered stdout (you get empty output + exit 1). Foreground (or
  flush every line + read the file before exit) avoids it.
- **`pkill -9` a CUDA python can wedge nothing** (GPU frees fine — checked with `nvidia-smi`), but don't
  rely on background runs; prefer foreground with line-buffered prints.
- flow and peclet.core.amr **cannot share one process** (flow finalizes Kokkos out from under peclet.core.amr's
  Views → abort). Run each engine in its own process.
- AMR N=128 staircase is genuinely slow (hundreds of seconds); use Galerkin for the converged k (same
  answer) and a short run only to confirm staircase stability.

Driver knobs (Python `peclet.core.amr.Flow`): `set_momentum_mg(True)`, `set_velocity_mg_staircase(True/False)`,
`set_momentum_gs(True)`, `set_momentum_mg_solver`, `set_outer_iterations`. Diagnostics:
`divergence_norm()` (cell), `divergence_norm_face()` (uf), `faceField()`.

flow collocated reference run lived at `/tmp/sdflow_coloc_gpu.py` (single SC sphere, tol 1e-6,
`SolverColocated`, `set_pressure_pcg(True,200,1e-8)`); the field-localisation harness is
`flow/scripts/compare_amr_sdflow_field.py` (subprocess-isolated).

## Open / next (pick up here)

1. **Profile the AMR momentum solve at 128³** — it's the wall-clock gap vs flow. The velocity-MG + GS
   momentum solve dominates; the staircase is bounded to the feature scale (slow), Galerkin scales
   better. See `bench_amr_flow mgstrat` and [[amr-gpu-smoother-flow-port]].
2. **Unsteady NS test** — the ONLY thing that would actually exercise the `uf` conservation benefit (all
   the steady cases here are `uf`-invariant by construction). E.g. a decaying Taylor–Green or a
   vortex-shedding case; check tracer/energy conservation vs the old `½(u_i+u_j)` advection.
3. **Device NS via MG-PCG** — advection currently forces the bounded V-cycle (`if(presPCG_ && !advect_)`
   in `project()`), because the large transient advection divergence excites the near-nullspace. With
   `maskSolid` now in the PCG it may be robust enough to cover advection too — worth retrying.
4. **The ~1% projection-structure difference** is still not isolated to one line. The clean way: expose
   each engine's projection as a standalone `project(u)->u_divfree` callable, feed both the SAME
   synthetic field on the SAME cut geometry, diff cell-by-cell. Neither exposes that yet.

Related memories: [[device-naming-retirement]] (the umbrella record of this whole arc),
[[amr-gpu-smoother-flow-port]], [[flow-collocated-solver]], [[amr-octree-status]].

---

# Addendum (2026-07-24): directional ghost-cell projection — 2nd-order cut cells on the octree

Status: **done, validated, committed** (ladder steps 0–3 of the ghost-projection port; see the
`amr-ghost-collocated-ns-plan` memory + flow/doc/collocated_second_order_open_problem.md §9 for
the uniform-grid original). Everything below is measured on the device engine (RTX 5080), with
the host oracle in lock-step (parity ctests).

## The two measured defects (step 0 — tests/study_amr_ghost_apriori.cpp, 6/6 gates)

The aperture projection above is first-order at cut cells for exactly flow's collocated mode-0
reasons, measured on the real `AmrPoisson` operators:

- `gradOf` (the ABC ½(g⁻+g⁺) cell gradient, feeding the −∇pⁿ predictor and the cell correction)
  reads the DECOUPLED p=0 of solid-centered neighbours through partially-open faces: O(1/h) at
  cut cells (11.3 → 91.1 over N=16..128, order −1.0) and gauge-dependent (+5 on p ⇒ O(5/h)
  gradient error; 13 869 open-solid-centered faces at N=128).
- The α-weighted ½/½ face-average constraint has O(1) local truncation at cut faces (O(h²) bulk).

## The fix, two stages

1. **`setGhostGradient`** (the mode-9 hybrid, cheap win): directional cell gradient on cut cells
   only — central where both axis-neighbour centers are fluid, 2nd-order one-sided toward the
   fluid else; O(h²), gauge-exact. Aperture projection untouched (throat-safe).
2. **`setGhostProjection(on, matrix_order=1, rhs_order=2)`** (the full scheme): the pressure
   system becomes ρ·(L_bin + Δ)φ = ρ·D_g(u*) — the BINARY-openness FV Laplacian (face open iff
   its sample + both adjacent centers are fluid) on the *unchanged* openness-MG rails
   (preconditioner + matvec base) plus the wall-anchored closure overlay
   (`amr/ghost_projection.hpp`, sharing `peclet::core::scheme` closures verbatim with flow),
   solved by MG-preconditioned BiCGStab on the coupled subspace. Implies the ghost gradient.
   **Sign note:** the suite operator is +L (negative-definite), so the matrix delta is the
   NEGATIVE of flow's `gpApplyDelta` expression; the divergence delta keeps flow's orientation.

**The octree enabler:** cut cells live in a uniformly-finest band (the `AmrCutCell` same-level
contract), so no closure ever crosses a 2:1 boundary — `buildGhostOverlay` THROWS if an overlay
row's ±2 reach touches a different-level leaf (widen the `refineToSdf` band, don't weaken this).

## Uniform Z&H (SC sphere, φ=0.125, K=4.292) — the headline

| N   | aperture (baseline) | + ghost gradient | full ghost projection | flow collocated ghost |
|-----|--------------------:|-----------------:|----------------------:|----------------------:|
| 32  | +0.302%             | +0.045%          | **−0.175%**           | −0.175%               |
| 64  | +0.135%             | −0.003%          | **−0.056%**           | −0.056%               |
| 128 | +0.054%             | +0.009%          | **−0.021%**           | −0.018%               |

The unrefined octree reproduces flow's collocated ghost to the 3rd decimal (bit-comparable
scheme, different engine) — the step-2 gate. BiCGStab stays flat at 6/7/8 iterations
(momentum/pressure iteration counts otherwise unchanged). Both ghost variants sit at or below
the Z&H table's own precision from N=32 on.

## Graded meshes (step 3): the machinery works; the C/F flux is now the limiter

Finest band at the sphere + coarse far field (`tests/study/amr_zh_graded.py`, `amr_zh_dilute.py`,
volume-weighted superficial velocity):

- φ=0.125, finest 64, ghost: band 3/5/8 → +8.8/+7.9/+6.4% at 20/29/42% of the cells; uniform
  reproduces −0.056% exactly through the graded machinery. The dense SC array has shear in every
  inter-sphere channel — a poor AMR showcase (as noted above for the aperture path).
- Dilute sphere (same R in a 4× box, φ≈0.002), finest 128: graded band 4 at **0.9% of the
  cells** runs stable with BiCGStab flat at 8 (the aperture MG-PCG caps at 60 on the same mesh),
  but sits +9.2% above the uniform K. **Attribution (measured, not assumed):** the aperture
  scheme shows the SAME +9.2% graded offset (ghost +9.3%) and the two uniform limits agree to 4
  digits — the graded error is entirely the far-field/level-boundary treatment (1st-order C/F
  flux where the 1/r Stokes disturbance still varies), independent of the ghost band.

**Consequence / next lever:** the cut-cell band is no longer the accuracy limiter — the 2:1
level-boundary flux is. The quadratic Martin–Cartwright C/F machinery already exists on the
Poisson rails (`applyLaplacianQuad`/`coarseStar`, P5b) but the projection deliberately uses the
standard consistent operator, and the momentum diffusion C/F is first-order too. Making the
*projection + momentum* C/F treatment 2nd-order (or simply placing level boundaries further into
the smooth field) is the open item for graded accuracy.

Guards: `test_amr_flow_solver::{test_sphere_ghost, test_sphere_ghostproj, test_graded_ghostproj}`
(oracle==device parity CUDA+OpenMP; thin-band throw; graded stability). Deferred, as in flow:
the tight-throat fragmentation guard, NS advection with the ghost uf, MPI (halo ±2 + BiCGStab
reductions).

## Update (2026-07-24, later): pluggable C/F schemes — Martin–Cartwright for the velocities too

`setCfScheme(scheme)` (`amr/cf_scheme.hpp`, oracle + device + Python `set_cf_scheme`) makes the
2:1 interface treatment a pluggable scheme: 0 = the standard two-point flux (default,
bit-identical), 1 = the Martin–Cartwright tangential quadratic, now applied to **everything the
steady solution feels**:

- the **momentum (velocity) diffusion** — the coarseStar substitution CSR ×μ on the α=1 velocity
  geometry, applied as a lagged deferred-correction RHS term (the implicit-FOU/SOU pattern; the
  steady operator carries the quadratic flux exactly);
- the **divergence constraint** — the ½/½ face average at a 2:1 sub-face is *normally* offset
  from the face (an error the tangential fix alone cannot remove), so the scheme's face value is
  the distance-weighted interpolation of {u_fine, u_coarse*}: exactly conservative, both incident
  rows share the value;
- the **pressure gradients** (−∇pⁿ predictor + cell correction) — coarse* substitution in the
  C/F face gradients *plus* side-reweighting of the ½(g⁻+g⁺) recombination so the sampled
  gradient lands on the cell center.

The pressure MATRIX / MG hierarchy / PCG / ghost BiCGStab stay on the standard consistent
operator — at the projection's fixed point φ→0, the matrix C/F order does not move the steady
solution (the (1,2)-mixed philosophy). Adding another scheme later = a new `CfScheme` value + a
branch in `cfAppendStencil`; every operator delta picks it up.

**A-priori (test_amr_cf_vector, graded mesh, C/F rows; the L delta is bit-parity with the
validated P5b quad correction):** divergence order 0 → **1.95** (+ exact conservation), cell
gradient 0 → **1.95**, momentum Helmholtz SOLUTION 0.8 → **2.0** (the flux has O(1) *pointwise*
row truncation at C/F faces by design — it telescopes; pointwise truncation is the wrong metric
for a conservative flux, cf. P5b).

**Graded drag (ghost projection, offsets vs the same-finest uniform):**

| case | band | standard C/F | quadratic C/F | cells |
|------|-----:|-------------:|--------------:|------:|
| SC φ=0.125, finest 64 | 3 | +8.8% | **+4.7%** | 20% |
| | 5 | +7.9% | **+3.5%** | 29% |
| | 8 | +6.4% | **+2.6%** | 42% |
| dilute φ≈0.002, finest 128 | 4 | +9.3% | **+2.9%** | 0.9% |
| | 6 | +8.3% | **+1.7%** | 1.2% |

The qualitative change: under the standard flux the graded error was scheme-floor-limited
(band-widening barely helped); under the quadratic scheme it is genuinely resolution-limited and
falls fast with band. Uniform meshes are an exact no-op (no C/F faces). Oracle==device locked by
`test_graded_cf_quadratic` (rel 2e-11). Residual limiter: at corners/edges of level islands the
tangential stencil gates out (a tangential neighbour is itself finer — the P5b robustness
fallback), leaving locally 1st-order rows; interpolating those from the fine side is the natural
next scheme refinement if the remaining offset matters.

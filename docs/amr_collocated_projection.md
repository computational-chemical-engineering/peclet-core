# AMR collocated projection: solid-pressure handling, divergence-free face field, `uf` advection

Status as of 2026-06-28. All work below is **done, validated, and pushed** (transport-core `main`;
umbrella `peclet` pointer bumped each step). This doc is the resume point — read it before continuing.

## TL;DR

The AMR `AmrFlow` (cell-centred / collocated cut-cell Stokes–NS on a `BlockOctree`) now has the three
pieces sdflow's collocated solver already had, so it is robust and conservative:

1. **`maskSolid`** in the pressure PCG — projects out the per-solid-region pressure null modes. Fixes
   the cut-cell Stokes blow-up. (commit `3aff1ab`)
2. **ABC/Basilisk divergence-free FACE field** `uf` — built each projection, exactly `∇·uf = 0`,
   correct across 2:1 interfaces. Host oracle `c37c23b`, device `4924129`.
3. **`uf` as the NS advecting velocity** — conservative advection (Bell–Colella–Glaz), consistent across
   implicit-FOU + explicit-FOU + SOU. Host + device `9faf5d8`.

Canonical engine = device `tpx::amr::AmrFlow` (`include/tpx/amr/flow.hpp`), Python-exposed as
`tpx_amr.Flow`. The serial host driver is `tpx::amr::oracle::AmrFlow` (`flow_oracle.hpp`) — dev-only
validation oracle, **not** exposed.

## The physics that drove this (Frank's framing — keep)

The cut-cell **openness** `α` (per-face fluid fraction; a face whose centroid is in solid gets `α=0`)
**decouples fluid from solid**. The pressure operator `L = div(α grad)` then splits into:
- the **connected fluid** system — well-posed up to ONE constant (its total pressure drifts → remove the
  mean), and
- **inert solid cells**, each connected solid region carrying **its own constant null mode**.

The solid pressure is **not solved** — it is **pinned to 0 and projected out**. sdflow does this with
`mg_mask_solid_k` (`maskSolid`): zero every `AC ≤ 1e-30` (solid, all faces closed) cell, applied
**together with `removeMean` after every matvec / seed / residual**, so the Krylov iteration lives
strictly in the fluid range with both null modes deflated. This machinery is **shared by sdflow
staggered AND collocated** (`buildCutcellOp` + `cutcellSmoothColor` + `maskSolid` + `removeMean`); only
the velocity divergence/correction differs (faces vs cells).

The AMR PCG (`pcg.hpp`) had only `removeMeanVol` (the fluid constant), **no maskSolid** ⇒ CG amplified
the solid / near-disconnected-fluid null modes ⇒ the Stokes pressure → ~1e24 over a few hundred steps.
(The V-cycle/GS path already skipped `diag==0`, which is why advection-via-V-cycle stayed bounded — only
the Krylov path needed the explicit projection.)

## What each piece is

### 1. maskSolid (`include/tpx/amr/pcg.hpp`)
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

## The comparison: sdflow collocated vs uniform AMR (the "where do we stand" run)

Z&H SC sphere, φ=0.125, K_exact = 4.292, **RTX 5080 (CUDA)**. AMR = device `AmrFlow` with `uf` +
`maskSolid`, staircase velocity-MG + multicolour-GS (Galerkin gives the same converged k).

| N   | sdflow collocated | AMR (uf, staircase/Galerkin MG) | gap   |
|-----|-------------------|----------------------------------|-------|
| 32  | 4.3313 (+0.92%)   | 4.3048 (+0.30%)                  | 0.62% |
| 64  | 4.3148 (+0.53%)   | 4.2977 (+0.13%)                  | 0.40% |
| 128 | 4.3070 (+0.35%)   | 4.2943 (**+0.05%**)              | 0.30% |

Findings:
- **AMR is closer to Z&H at every N and converges faster** (~order 1.3 vs sdflow ~0.7). At 128³ AMR is
  essentially exact (+0.05%) vs sdflow +0.35%.
- The gap **shrinks with N** (0.62→0.40→0.30%) — both → the same continuum; the difference is the
  projection STRUCTURE (AMR genuinely cell-centred FV vs sdflow's staggered-MAC-heritage ABC
  projection), now seen across resolution. **NOT yet isolated to a single line of code.**
- **Staircase velocity-MG is fixed at scale**: no longer diverges at 64³ (that was the pre-maskSolid
  limit recorded in [[amr-gpu-smoother-flow-port]]); runs at 128³ too, just slowly (bounded to the
  sphere's feature depth → many momentum iterations).
- **Perf caveat:** sdflow is faster in wall-clock at 128³ (~121 s to converge vs ~704 s for AMR's 150
  steps). The AMR velocity-MG + GS momentum solve is the cost — the obvious thing to profile if making
  AMR competitive at scale matters.

## How to reproduce (GPU) — and the gotchas (these cost time)

Build the CUDA `tpx_amr` Python module:
```bash
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd transport-core/python
PYBIND=$(cd ../../sdflow && .venv/bin/python -m pybind11 --cmakedir)
PYEXE=$(cd ../../sdflow && .venv/bin/python -c "import sys;print(sys.executable)")
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
- sdflow and tpx_amr **cannot share one process** (sdflow finalizes Kokkos out from under tpx_amr's
  Views → abort). Run each engine in its own process.
- AMR N=128 staircase is genuinely slow (hundreds of seconds); use Galerkin for the converged k (same
  answer) and a short run only to confirm staircase stability.

Driver knobs (Python `tpx_amr.Flow`): `set_momentum_mg(True)`, `set_velocity_mg_staircase(True/False)`,
`set_momentum_gs(True)`, `set_momentum_mg_solver`, `set_outer_iterations`. Diagnostics:
`divergence_norm()` (cell), `divergence_norm_face()` (uf), `faceField()`.

sdflow collocated reference run lived at `/tmp/sdflow_coloc_gpu.py` (single SC sphere, tol 1e-6,
`SolverColocated`, `set_pressure_pcg(True,200,1e-8)`); the field-localisation harness is
`sdflow/scripts/compare_amr_sdflow_field.py` (subprocess-isolated).

## Open / next (pick up here)

1. **Profile the AMR momentum solve at 128³** — it's the wall-clock gap vs sdflow. The velocity-MG + GS
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
[[amr-gpu-smoother-flow-port]], [[sdflow-collocated-solver]], [[amr-octree-status]].

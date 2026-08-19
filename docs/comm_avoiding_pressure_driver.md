# A communication-avoiding pressure driver: once-per-solve RHS projection + Chebyshev V-cycles

*Written 2026-08-19, from the `dev/aperture-compat-rhs` experiment (core `94a069d`). Status:
**PROPOSAL — not implemented.** This note exists so the option is on the table the next time
multi-GPU pressure-solve scaling is the work item. Read
`amr_aperture_advection_plan.md` §RESOLVED first (the diagnosis this builds on) and
`../../docs/DECOMPOSITION_AND_MULTIGRID.md` for the coarse-level context.*

## The opportunity, in one paragraph

The MG-PCG pressure solve costs ~30–45 **blocking all-reduces per solve** (2–3 volume-weighted
dot products per iteration at 15–16 iterations, plus the nullspace-projection mean sums), and the
measured weak-scaling campaigns identify exactly this global synchronization as THE bottleneck at
scale (flow on Snellius: np32 weak efficiency 35 %, recovered to 62 % only by comm-avoiding
event-halving — see the comm-scaling memory/study). A **stationary (or Chebyshev-accelerated)
V-cycle driver needs no dot products at all** — only neighbor halo exchanges, which overlap and
never synchronize the whole machine. What has kept that driver off the table in the AMR aperture
path is that it *stalls*: the aperture RHS carries an incompatible fluid-mean component that only
CG's per-iteration deflation removes. The 2026-08-19 experiment proved the stall has **exactly
one removable cause**, and that it can be removed for ONE all-reduce per solve with numerics
identical to production. That makes an all-reduce-free pressure driver a real option for the
latency-bound regime.

## What the experiment established (measured, Z&H φ=0.125, RTX 5080)

Branch `dev/aperture-compat-rhs` computed the divergence RHS on ALL operator DOF (restoring the
consistent `L = D·G` triple ⇒ RHS exactly compatible by telescoping, fluid-mean 1e-6 → 1e-21):

1. **The V-cycle stall is exactly the incompatible component and nothing else.** With a
   compatible RHS the bounded stationary V-cycle converges cleanly at rate ~0.7/cycle to 7.2e-13
   in 60 cycles, where production (main) floors at 2.1e-4 from the first cycle — and the floor on
   main equals `|fluid-mean(b)|·sqrt(V_fluid)` to <2 %. No second pathology is hiding behind it.
2. **PCG is indifferent to compatibility** (15–16 iters both ways) — its projection already
   deflates the component for free. So compatibility buys nothing on the Krylov path; its value
   is exclusively that it *enables non-Krylov drivers*.
3. **But the discretization route to compatibility is NOT the one to ship**: computing div at
   solid-centered rows makes the projection chase a persistent collocated representation
   artifact (steady |rhs|_D 2.7e-2 instead of 4e-4) and introduces a resolution-independent
   K bias ~−0.09 % (Stokes N=32/64: +0.040/−0.003 % → −0.084/−0.092 %). Rejected for production.

## The design (what to actually build)

**Step 1 — solver-side RHS projection (numerics-preserving, 1 all-reduce/solve).**
At solve start, subtract the volume-weighted fluid mean from `b`:
`b ← b − (Σ V_i m_i b_i / Σ V_i m_i)·m` with `m` the fluid mask (`buildFluidMask`). This is
*exactly* the projection PCG applies to `r₀` (pcg.hpp `removeMeanVolReduced`), so the solution is
identical to production's — the removed component is unsolvable under any driver. Cost: one
fused all-reduce (two scalars) per solve. This alone makes the stationary V-cycle stall-free
**without** the dev branch's discretization change or its bias.

**Step 2 — drop the per-level `removeMean` inside the V-cycle.**
With a mean-free RHS, the iteration's null component only drifts the constant, which no gradient
ever sees (`∇φ` is invariant); remove the mean once at solve end if the raw φ value matters, or
never. This deletes the per-level projection work — and, on the distributed hierarchy, any global
sums hiding in it. (Verify on the distributed `DistributedFlowMultigrid` path: its per-level
mean removal is where residual all-reduces could re-enter.)

**Step 3 — Chebyshev acceleration instead of plain stationary cycling.**
The stationary cycle at rate ~0.7 needs ~60 cycles to 1e-10 vs PCG's 15–16 (each ≈ 1 V-cycle +
1 matvec) — roughly **3–4× more local work and halo traffic**. Chebyshev iteration over the SAME
MG preconditioner needs *no dot products* (only a one-time eigenvalue-interval estimate, which
can be done once per `setSolid` with a few power iterations — off the critical path) and closes
most of that gap: for a preconditioned operator with κ_eff ≈ (1+0.7)/(1−0.7) ≈ 5.7, Chebyshev's
asymptotic factor is ~0.38/step vs stationary 0.7 — ~2× fewer cycles, i.e. ~25–30 cycles to
1e-10, or ~10–15 at a realistic per-step tolerance. Note Chebyshev (like the stationary cycle)
cannot deflate an incompatible RHS — Step 1 is a prerequisite, which is why this family was
never viable before the diagnosis.

**Step 4 — fixed cycle count or infrequent convergence checks.**
A tolerance-based exit needs a residual norm (an all-reduce). Either run a fixed count tuned per
case (the projection tolerates a loose solve; production PCG runs 1e-10, which is far tighter
than the O(h²) projection needs), or check every k cycles. Target: **≤2 all-reduces per pressure
solve, total.**

## When it wins (and when it doesn't)

- **Single GPU / few ranks: it loses.** ~2–4× more local V-cycle work with nothing to save.
  Keep MG-PCG the default there — this is a *scale* driver, selected like flow's CA levers.
- **Latency-bound weak scaling (many ranks, small per-rank grids): it should win.** Trading
  ~40 blocking global syncs for ~2, against 2–4× more overlappable neighbor traffic, in the
  regime where the measured campaigns show global syncs dominating the pressure step. The
  crossover rank count must be MEASURED, not assumed — the honest expectation is "significant
  above some np, useless below it".
- Composes with the existing CA event-halving (they attack different terms: message *events* vs
  global *synchronizations*).

## Scope beyond AMR

flow's `CutcellMG` PCG has the same dot-product structure and the same deflation projection
(`maskSolid` + `removeMean`), so the same driver family applies. **To verify before porting:**
whether flow's collocated aperture RHS carries the same incompatible mean (likely — same
truncated-divergence structure at solid-centered open DOF) and whether the staggered path does
(likely not; its divergence rows and operator DOF coincide). The Chebyshev lever was already on
the Snellius campaign's list; this note adds the missing enabler (Step 1) and the evidence that
nothing else blocks it.

## Implementation entry points

- `core/include/peclet/core/amr/flow.hpp::project()` — the driver branch; Step 1 is ~10 lines
  reusing `buildFluidMask` + the mean kernels from `pcg.hpp` (the debug block added 2026-08-19
  already computes exactly this mean — `PECLET_CORE_AMR_PRES_DEBUG`).
- `core/include/peclet/core/amr/multigrid.hpp` / `distributed_flow_mg.hpp` — Step 2 (per-level
  removeMean) and the Chebyshev wrapper (Step 3) around `vcycle()`.
- `set_pressure_pcg(False)` (Python binding, exists on `dev/aperture-compat-rhs`; cherry-pick) —
  the A/B switch.
- Measurement: count all-reduces per solve (instrument `allred_`/`dotReduce_`), then the
  distributed study battery (`python/test_amr.py` np ladder locally; the Snellius weak-scaling
  harness for the real curve).

## Pitfalls recorded up front

- Do NOT take the dev branch's divergence change as the compatibility mechanism — it biases K
  (−0.09 % floor). Step 1 (project `b`) is the numerics-preserving route; the branch survives
  only as the proof that the stall has a single cause.
- The per-step warm-start structure differs: PCG restarts from x=0 each step (incremental φ);
  a Chebyshev/stationary driver could warm-start from the previous φ — a further saving, but
  measure the transient behavior before relying on it.
- The eigenvalue interval for Chebyshev must be re-estimated when the operator is rebuilt
  (adaptivity, `finishAdapt`) — hook it there, not per step.
- Convergence-check all-reduces sneak back in easily (diagnostics, `divergence_norm()` per step
  in driver scripts). Budget them explicitly when measuring.

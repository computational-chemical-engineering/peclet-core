# The AMR aperture pressure solve under advection — the blocker on retiring the ghost projection

*Written 2026-08-18, as the handoff from the collocated-scheme decision. Read
`amr_collocated_projection.md` §"Measured (tests/study/amr_ns_ghost.py …)" and the 2026-07-25
update first — this note only states the problem, the evidence, and what would settle it.*

## RESOLVED 2026-08-19 — characterisation results and the fix

The stall was characterised before fixing, per the plan. All three candidate mechanisms were
measured on the N=32 Z&H aperture NS case (RTX 5080, `PECLET_CORE_AMR_PRES_DEBUG=1` — the
instrumentation is kept in `flow.hpp::project()`):

1. **SPD — REFUTED as the cause.** The pressure operator is assembled in `setSolid` from geometry
   only (`openFn`); `advect_` never enters the assembly, so the operator under advection is the
   same object the Stokes MG-PCG solves in 17–30 iterations. Measured directly on the assembled
   `FvOp` with advection on: `<y,Lx>_D` vs `<x,Ly>_D` agree to **8.8e-15 relative** and random
   Rayleigh quotients are strictly negative — symmetric negative-definite in the volume-weighted
   inner product, exactly as in Stokes.
2. **Agglomerated coarse level — NOT the mechanism here.** `AmrMultigrid::build` coarsens until a
   **single leaf** remains, so the (single-rank) bottom is exact by construction; there is no
   under-solved coarsest level for flow's agglomeration cure to fix. (The measured stall is also
   level-independent — see below — which an unsolved bottom would not be.)
3. **Compatibility — CONFIRMED, with the projector-side twist.** The V-cycle residual trace shows
   *clean* convergence at rate ≈0.70/cycle at step 1 (rel 8e-10 at the 60-cap), degrading to a
   **hard stall** as the flow develops: at step 50 the residual floors at 1.435e-4, at step 200 it
   floors at 2.108e-4 **from the first cycle**. The floor is exactly the incompatible RHS
   component: the fluid-volume mean of `div_` times `sqrt(V_fluid)` (8.34e-7·169 = 1.41e-4 and
   1.22e-6·169 = 2.07e-4 — match to <2%). The mean is nonzero because the divergence RHS is only
   computed at fluid-*centered* cells while the aperture operator's DOF set also contains
   solid-centered cells with partially-open faces — zeroing those rows' RHS breaks the telescoping
   that would make `Σ V·div = 0` over the DOF set. The defect scales with the developed flow
   (~3e-3 relative at steady state), which is why Stokes-from-rest never hurt and why it looked
   "transient". So "60 bounded V-cycles/step" was a **stagnation cap**, not a convergence count —
   and the solution `x` the stalled cycle produces is fine (the stall lives in the unsolvable
   null component of the residual).

**The fix is the PCG gate, nothing else.** The MG-PCG's per-iteration fluid-range projection
(`maskSolid` + volume-weighted fluid-mean removal, in `pcg.hpp` since the maskSolid work) deflates
exactly the incompatible component, so CG is valid and healthy under advection. The historic
`presPCG_ && !advect_` exclusion predates `maskSolid` and was simply stale. Removing `!advect_`:
flat **15–17 PCG iterations/step (tol 1e-10)** across the whole impulsive N=32 transient, steady
K identical to the V-cycle path to 4+ digits (4.4276 at step 200 on both), RHS compatibility and
solver health traced per step. The bounded V-cycle remains the `setPressurePCG(false)` fallback
(unchanged, still stalls benignly at the incompatible floor).

Acceptance measurements + the ghost-projection retirement that followed: see
`amr_collocated_projection.md` (2026-08-19 update).

## Where this sits

The collocated second-order question is closed: the **gauge-exact** scheme (aperture constraint +
directional cell gradient) is the production collocated path in flow and the default AMR cell
gradient (`setGhostGradient`, core eb73c6c). Refinement ladders on two periodic sphere beds put it
at second order and make it the cheapest scheme measured — 4.6× faster than the staggered cut-cell
reference and 5–6× faster than the directional ghost projection, which it matches in accuracy
(peclet-examples `benchmarks/porous-scaling`, `colcmp*` / `colcmp060*`). The ghost projection is
therefore **quarantined in flow**.

**It cannot yet be retired in AMR**, and the reason is not that ghost is good — it is that the
*aperture* path has a solver defect that only appears under advection.

## The defect

From `amr_collocated_projection.md` (measured, `tests/study/amr_ns_ghost.py`, Z&H at Re≈9.7):

> the aperture path must run the bounded V-cycle (**60 cycles/step — MG-PCG excluded by the
> transient near-nullspace issue**) while the ghost BiCGStab stays at 6–7 iterations: 18 s vs 341 s
> (N=32), 52 s vs 586 s (N=64) to converge

So under advection the aperture pressure solve is 10–19× more expensive **for solver reasons, not
discretization reasons** — MG-PCG is unusable and the fallback is a bounded V-cycle at ~60
cycles/step. Everything else favours the aperture path: accuracy differs by only 0.24 %, the
trajectories track through an impulsive start with no drift, and in Stokes the aperture family is
the one that is both second order and cheap.

Note the contrast that makes this suspicious rather than fundamental: in **Stokes**, on real sphere
beds, the same aperture operator solves in 17–30 MG-PCG iterations and is the fastest scheme
measured. Something specific to the advecting configuration is breaking the Krylov path.

## What to find out

1. **What exactly is the "transient near-nullspace issue"?** It is named in the docs but not
   characterised. Candidate mechanisms, in the order I would test them:
   - the pressure operator under advection is assembled with a component that makes it
     non-symmetric or indefinite, so CG is not merely slow but invalid (check: is the assembled
     operator still SPD when advection is on? compare `A` against `Aᵀ` directly);
   - the near-nullspace is genuinely richer than the constant (fragmented/pocketed fluid regions,
     or the C/F interface rows), so the V-cycle's coarse-grid correction cannot represent it —
     this is the same disease the binary-openness hierarchy has in flow, and the fix there is
     the agglomerated bottom (`set_pressure_bottom("auto")`, flow 3493a89), which AMR may not have;
   - the compatibility projector is wrong under advection (the RHS is not being made compatible
     with the actual left null space), so the residual stalls and the bounded V-cycle count is
     really a stagnation cap.
2. **Does the agglomerated coarse solve fix it?** In flow this exact symptom — a V-cycle that is
   not domain-independent because the coarsest level is not effectively solved — was resolved by
   agglomerating the bottom (`docs/DECOMPOSITION_AND_MULTIGRID.md` §1.1/1.2, §2.7). If AMR's
   coarsest level has the same problem, this is the cheapest possible fix.
3. **Is MG-PCG actually invalid, or just untuned?** If the operator is SPD, PCG should work and
   the exclusion is a bug worth re-testing rather than a constraint.

## Acceptance

The aperture path under advection reaches the ghost BiCGStab's cost envelope (single-digit
iterations, wall time within ~2×) on `tests/study/amr_ns_ghost.py` at N=32 and 64, with
`test_sphere_ghostproj_adv` still green. At that point `setGhostProjection`'s AUTO tri-state can be
retired to plain OFF, `core/include/peclet/core/amr/ghost_projection.hpp` follows flow's overlay
into quarantine, and `peclet::core::scheme::ghost_closure` loses its last production consumer.

## Traps

- `setGhostProjection` is a **tri-state** (`-1` auto / `0` off / `1` on) and auto-arms on
  `advect_` at `setSolid` time — measure with an explicit value, never by relying on the default.
- The oracle (`amr/flow_oracle.hpp`) and the device path (`amr/flow.hpp`) both carry these flags
  and must stay in lockstep; the parity ctests are the guard.
- `setGhostProjection` implies `setGhostGradient` — when comparing projections, hold the gradient
  fixed (it is now ON by default on both paths) or you will be measuring two changes at once.

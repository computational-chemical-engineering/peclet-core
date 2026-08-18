# The AMR aperture pressure solve under advection — the blocker on retiring the ghost projection

*Written 2026-08-18, as the handoff from the collocated-scheme decision. Read
`amr_collocated_projection.md` §"Measured (tests/study/amr_ns_ghost.py …)" and the 2026-07-25
update first — this note only states the problem, the evidence, and what would settle it.*

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

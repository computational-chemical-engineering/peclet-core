# Session prompt — the AMR aperture pressure solve under advection

*Paste the block below into a fresh session started in `~/Codes/suite`. Written 2026-08-18 as the
handoff from the collocated-scheme decision; the technical background is
`core/docs/amr_aperture_advection_plan.md`.*

---

Fix the AMR aperture pressure solve under advection. It is the last thing keeping the directional
ghost projection alive anywhere in the suite.

**Background — read first, in this order:**
1. `core/docs/amr_aperture_advection_plan.md` — the problem statement, evidence, three candidate
   mechanisms in test order, acceptance bar, and the traps. Start here.
2. `core/docs/amr_collocated_projection.md`, the "Measured (tests/study/amr_ns_ghost.py …)" block
   and the 2026-07-25 update — where the numbers come from and how the tri-state default works.
3. `docs/DECOMPOSITION_AND_MULTIGRID.md` §1.1/1.2 and §2.7 — the agglomerated coarse solve, which
   is the top hypothesis (see below).
4. My memory `collocated-second-order-verdict.md` for why this matters now, and
   `snellius-access.md` if you need the cluster (ssh alias `snellius` works non-interactively;
   jobs are billed per ALLOCATED GPU, so right-size `--gpus-per-node`).

**The defect.** Under advection the aperture pressure solve runs ~60 bounded V-cycles per step
with MG-PCG excluded by what the docs call a "transient near-nullspace issue", while the ghost
BiCGStab needs 6–7 iterations — 18 s vs 341 s at N=32, 52 s vs 586 s at N=64. Accuracy differs by
only 0.24 %, so this is purely a solver problem. In **Stokes**, on real sphere beds, the same
aperture operator solves in 17–30 MG-PCG iterations and is the fastest scheme measured
(peclet-examples `benchmarks/porous-scaling`, `colcmp*`/`colcmp060*`). Something specific to the
advecting configuration breaks the Krylov path — that asymmetry is the main clue.

**Top hypothesis, test it first:** flow had the *same* symptom — a V-cycle that is not
domain-independent because the coarsest level is not effectively solved — and it was cured by
agglomerating the bottom level (`set_pressure_bottom("auto")`, flow 3493a89, now the default).
Check whether AMR's hierarchy has an equivalent, and whether its coarsest level is the stall. The
other two candidates (operator not SPD under advection ⇒ CG invalid; compatibility projector wrong
⇒ the "60 cycles" is really a stagnation cap) are in the plan doc with the checks to run.

**Do this first, before any fix:** characterise the stall. Is the assembled operator still SPD when
advection is on (compare `A` against `Aᵀ` directly)? Does the residual stagnate or decrease slowly?
Where does the V-cycle stop gaining — which level? Right now "transient near-nullspace issue" is a
name in a doc, not a diagnosis, and I do not want a fix built on it as an assumption.

**Constraints.**
- `amr/flow.hpp` (device) and `amr/flow_oracle.hpp` (host oracle) both carry these flags and must
  stay in lockstep; the parity ctests are the guard. 89/89 core ctests must stay green.
- `setGhostProjection` is a tri-state (−1 auto / 0 off / 1 on) that auto-arms on `advect_` at
  `setSolid` time — always measure with an explicit value, never the default.
- `setGhostProjection` implies `setGhostGradient`, and the gauge-exact gradient is now ON by
  default on both paths (core eb73c6c). When comparing projections, hold the gradient fixed or you
  are measuring two changes at once.
- Don't change numerics while changing solvers — this is a solver-performance task, and the steady
  answers (aperture vs ghost within 0.24 %) must not move.

**Acceptance.** The aperture path under advection reaches the ghost BiCGStab's cost envelope
(single-digit iterations, wall time within ~2×) on `tests/study/amr_ns_ghost.py` at N=32 and 64,
with `test_sphere_ghostproj_adv` green. Then retire AMR's ghost projection: `setGhostProjection`
AUTO → plain OFF, `core/include/peclet/core/amr/ghost_projection.hpp` follows flow's overlay into
quarantine, and `peclet::core::scheme::ghost_closure` loses its last production consumer.

**If the top hypothesis is wrong**, say so early rather than forcing it — the phase-A ghost work
(`flow/doc/ghost_hardening_findings_A.md`) is a worked example of the plan's lead hypothesis being
refuted by measurement, and that was the useful outcome.

Commit at milestones; push only when asked.

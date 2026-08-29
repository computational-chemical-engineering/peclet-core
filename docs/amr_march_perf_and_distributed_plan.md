# Next phase: march-time economics + the distributed mixed-level band

*Plan, 2026-08-30 (Fable). Follows `amr_setup_parallel_plan.md` (complete, F1 resolved) and
`amr_mixed_level_cut_band_plan.md` (P3c). Ownership per rung is marked **[OPUS]** (execution
against settled decisions) or **[FABLE]** (a decision gate — Opus stops there and hands back,
same escalation contract as the setup-parallel plan §5, which applies verbatim here).*

**Status: PLANNED.** Two phases, independent — M can run before, after, or interleaved with D.

## Why these two, in this order ahead of device assembly

The setup phase is now ~3% of a bed job (2.4 µs/leaf @8t; F1 closed multi-rank too), so the
outstanding performance question is P3c observation 3: **the graded mesh saves 1.62× in cells
but ~0% in step time** (d8: 1.85 vs 1.81 s/step; d7: 0.89 vs 0.92). Until that is understood,
the porous payoff is memory-only and the cell-count headline overstates the runtime win — that
is a diagnosis debt, and it blocks honest Phase-3 claims. Independently, F1's resolution
unblocked the distributed rungs, and the depth-9 uniform control (~104 GB) is exactly the case
that NEEDS multi-GPU — the missing piece is the distributed sample halo. Device-resident
assembly (`amr_device_assembly_plan.md`) stays parked behind both: its payoff concentrates in
adapt-during-run, which neither current study exercises.

## Phase M — why doesn't the graded march get faster?

### M0 [OPUS] — instrument the step

An env-gated per-phase step profiler (`PECLET_CORE_PROFILE_STEP=1`), same pattern as the
setSolid one: fenced timers around the momentum solve, the pressure solve (split matvec /
MG-precond / dots if cheap to separate), the overlay delta kernels, the cf delta kernels, and
the per-step glue; plus counters already available (`last_pres_iters`, momentum iters). Print
per-step averages over a window, µs/leaf AND ms/step. Zero cost when unset. Gate: profiler-off
run bitwise-unchanged (it must not add fences on the hot path when disabled).

### M1 [OPUS] — the measurement matrix

On the RCP bed (the geometry where the anomaly is measured), depth 7 AND 8, uniform vs graded
arms, 200-step windows after warm-up, 3 repeats: per-phase ms/step, pressure iterations per
step, MG level count and per-level cell counts, BiCGStab/PCG iteration counts. Also the Z&H
N=256 pair from P2b (a second geometry family — does the anomaly reproduce off the bed?).
Deliverable: one table attributing the missing 1.62× phase by phase. No fixes — measurement
only, committed with the table in the message.

Hypotheses the matrix must separate (pre-registered so the attribution is honest):
- H-iters: the graded mesh needs more pressure/momentum iterations per step (worse
  conditioning from C/F faces or the coarse cut band).
- H-mg: the MG hierarchy over the graded mesh has more levels engaged / worse per-level
  shapes, so each V-cycle costs relatively more than the leaf count suggests.
- H-launch: many small kernels (per-level, per-overlay) put the graded mesh into
  launch-latency-bound territory the uniform mesh amortizes.
- H-band: the step cost is dominated by a term proportional to CUT-BAND size, not leaf count
  (both arms have identical bands at matched depth — this would explain ~equal step times
  exactly).

### M2 [FABLE] — the verdict + fix selection

Whatever M1 attributes, the fix is a design decision (MG cycle shape, kernel fusion,
level-batched launches, or "accept: the win is memory + capacity, document it honestly").
Opus stops after M1.

## Phase D — the distributed mixed-level cut band

Goal: `setGhostSampled` drops its single-rank guard, so graded beds decompose across ranks —
the depth-9 uniform control (~104 GB) and production beds become reachable. The classic
distributed ghost machinery (miss-collect fixpoint, collective band decisions, LeafHalo ±2
registry) is the template throughout.

### D-decisions (settled here — Opus implements)

**DD1 — Clouds become deterministic probe sets.** The LS cloud collection currently gathers
candidate fluid leaves by spatial BIN search over the local leaf array — a block-local search
would give decomposition-DEPENDENT clouds near rank boundaries, which violates D6 (row state a
pure function of geometry) across decompositions. Redesign: each row's cloud candidates are a
DETERMINISTIC coordinate set (the fixed probe pattern around the row's virtual positions, in a
fixed order, at the row's own level), each resolved through `probeSlot` — exactly how every
other builder already reaches across ranks. Single-rank must reproduce today's clouds
bitwise or the change is wrong (the probe pattern must enumerate the same candidates the bin
search finds — verify on the P2b meshes before touching the distributed path; if the bin
search's candidate set cannot be reproduced by a bounded fixed pattern, STOP — that is a
[FABLE] gate, not a workaround).

**DD2 — Discovery via the existing fixpoint, no new machinery.** `buildGhostOverlaySampled`
(and `buildMomSeamDelta`, and `makeBinaryOpenFnMixed`'s center probes) join
`prepareDistributed`'s prober list, tolerating `kPending` during discovery rounds exactly as
`mom_.build` does (results discarded until the final round). The miss-collect fixpoint already
handles level-dependent reach — a coarse row's ±2-at-its-level probes are just coords; the
registry doesn't care. The F1 mutex makes all of this thread-safe.

**DD3 — Collective agreement on fallbacks.** Any per-row degrade decision that could differ
across ranks near a boundary must be a pure function of (geometry, probe results) — never of
"what this rank happens to own" (the D6 discipline). The collective band-violation pattern
(one Allreduce before anyone commits) is the template if a global decision is ever needed;
prefer designs that need none.

**DD4 — Acceptance ladder for the phase** (the campaign's standing pattern):
np=1 distributed == single-rank BITWISE (the classic contract); np=2,4 == np=1 within the
established ~5e-12 (decomposition-independence); the P2b latitude mesh and the depth-7 graded
bed as the two test geometries; a new ctest pinning np=1 bitwise on a seamed mesh.

### Rungs

- **D0 [OPUS]** — DD1 single-rank: probe-set clouds, bitwise vs today's bin-search clouds on
  the P2b meshes + depth-7 bed (this is a pure refactor gate; any diff ⇒ stop per DD1).
- **D1 [OPUS]** — DD2: wire the sampled builders into the fixpoint; np=1 bitwise ctest.
- **D2 [OPUS]** — drop the single-rank guard; np=2,4 acceptance per DD4; measure distributed
  setSolid scaling (the F1 fix should give near-single-rank per-rank cost).
- **D3 [FABLE]** — review the np>1 numbers, decide whether the ~5e-12 class holds for the
  sampled path or the deviation needs attribution; then the depth-9 TWO-ARM run on 2×H100
  (the uniform control finally fits split) — the full accuracy-matched headline at R/h₀=48.
- **Backlog, explicitly NOT this phase**: sub-face closures (accuracy item, bounded at ≤1.6%
  of rows — needs its own Fable design pass), pocket exclusion in LS clouds (fold into D0's
  cloud redesign ONLY if it falls out naturally; otherwise unchanged), device-resident
  assembly (next campaign after these).

## Standing rules (unchanged)

Escalation contract = setup-parallel plan §5 verbatim. Every rung: 149/149 both nets, the
relevant bitwise gates, measured numbers in the commit message, push submodule-then-umbrella.
The march profiler (M0) must never perturb the march when disabled.

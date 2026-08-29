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

## Findings

### F2 (D0, 2026-08-30) — the LS cloud's periodic period is 4 fine cells SHORT: DD1 cannot be made bitwise without fixing it

**Status: D0 BLOCKED on a [FABLE] decision (DD1's own escalation clause). The probe-set rewrite is
written, gated and green everywhere EXCEPT this; it is parked on branch `dev/d0-probe-clouds`
(core), not on `main`, because landing it would silently change march numerics.**

**The bug.** `buildGhostOverlaySampled` sizes its LS hash bins from the octree:

```cpp
long ext = 0;
for (Index i = 0; i < n; ++i) { auto b = t.bounds(i);
  for (int d = 0; d < 3; ++d) ext = std::max(ext, (long)b[1][d]); }
nbx = std::max<long>(1, ext / 4);              // "ext fine cells / 4 per bin"
const double domain = (double)nbx * hb;        // hb = 4*h0 — "world extent (cubic domains)"
```

`BlockOctree::bounds` returns **inclusive** bounds, so `ext = fineExt − 1`, and for every
power-of-two domain the truncating divide loses exactly one bin:

| N (fine) | ext | nbx | nbx should be | `domain` | true period | error |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 31 | 7 | 8 | 28 | 32 | 4 |
| 64 | 63 | 15 | 16 | 60 | 64 | 4 |
| 128 | 127 | 31 | 32 | 124 | 128 | 4 |
| 256 | 255 | 63 | 64 | 252 | 256 | 4 |
| 512 | 511 | 127 | 128 | 508 | 512 | 4 |

`domain` is the period the cloud's minimum-image wrap uses, both for the `r2 <= rho²` membership
test and for the monomial offsets `d = del/H` that BUILD the least-squares weights. So every cloud
whose ball crosses a periodic domain face gets its across-the-seam candidates placed **4 fine cells
away from where they are**, and its membership decided at those wrong positions.

**Why it went unnoticed.** It is invisible unless the cut band reaches within `rho` of a domain
face. Every calibration geometry in this campaign is a centred sphere or sphere pair (Z&H, M2, P2b
latitude, the two-sphere gap ladder) whose seams sit in the middle of the box, so `del` never
approaches `0.5*domain` and the wrap never fires — the classic uniform-band path has no clouds at
all. The RCP **bed** is the first geometry whose surface crosses the periodic boundary, and it is
the one the P3c numbers come from.

**Measured blast radius.**
- Mini-bed harness (5 periodic spheres, one ON the corner, three-level graded map, N=64):
  **66.8%** of LS clouds (41 009 / 61 362) get both a wrong candidate set and wrong offsets;
  worst offset error exactly 4 fine cells. The bin gather agrees with brute force under its own
  (wrong) metric — the enumeration is faithful, the METRIC is wrong.
- Depth-7 gap-graded RCP bed, `set_ghost_sampled`, cf-quadratic, 20 steps from rest: the census is
  unchanged to the last count (229 485 rows, 31 095 LS2, 0 degraded, 453 closed mixed faces, 410
  pocket cells) and the fluid mask is bitwise identical, but the velocity field moves —
  `max|Δu_x| = 6.9e-3` against a field max of `1.5e-1` (4.5% locally), `Σu_x` by **5.0e-5
  relative**, and the pressure iteration count at step 20 goes 54 → 60. Geometrically ~19% of the
  depth-7 bed's LS clouds sit within `rho ≈ 4.4 h0` of a face.
- So the P3c headline is NOT at risk: a 5e-5 shift in `Σu_x` is four orders below the 0.26% / 1.79%
  policy numbers. It is a correctness fix, not a headline revision.

**Why DD1 cannot route around it.** The bin search's candidate set is not a geometric
neighbourhood: a leaf at `c` enters `p`'s cloud through `del = wrap_{124}(c − p)`, i.e. it is
treated as sitting at `p + del`, which is **not a periodic image of `c` under the true period**.
`probeSlot` — the only route to a leaf that works across ranks — wraps by the TRUE domain, so no
probe pattern can return that leaf at that position. Reproducing the bin search bitwise would mean
probing three shifted boxes at `p + k·124·h0` deliberately, i.e. re-implementing the bug in a more
elaborate form. That is DD1's pre-registered stop condition, so I stopped.

**Why Phase D cannot carry it forward either** (the reason this is a blocker, not a nit): `ext` is a
reduction over **the local leaf array**. On one rank that is the domain; on four ranks each rank
gets its own block's extent, so each rank would use a DIFFERENT period, and the cloud metric itself
would become decomposition-dependent — exactly what D6 forbids and what DD1 exists to remove. The
period has to become a global, geometry-derived quantity before `setGhostSampled` can drop its
single-rank guard. The fix is a prerequisite for D, not an optional cleanup.

**What the parked rung already establishes** (so the decision is cheap to act on):
- The probe descent `detail::forEachCoveringSlot` (probe a region's lo corner; if the covering leaf
  contains the region, take it, else split the longest axis at its coarsest power-of-two boundary)
  enumerates exactly the leaves overlapping a fine-coord box in `O(#leaves)` probes, through
  `probeSlot` alone, and handles a `kPending` region by not descending it — the discovery fixpoint
  then needs one extra round per octree level the box spans (D1 must measure that round count).
- The canonical order `(bin in the old traversal order, then Morton code of the leaf's global lo
  corner)` reproduces today's accumulation order bit-for-bit — the leaf array is Z-order sorted, so
  ascending code IS ascending leaf index — and unlike the leaf index it is decomposition-independent.
- Cost of the route change, measured on the same machine and thread count (8): `setSolid` on the
  depth-7 graded bed 4.35 s → 4.82 s, i.e. **2.4 → 2.7 µs/leaf** (+15%); depth 6 is 2.5 → 2.5,
  unchanged. The descent's `probeSlot` is a binary search over the leaf array where the bin gather
  was O(1), and that is paid back in part by dropping the serial `bins` build (an O(n) pass over
  every leaf). Still inside the setup-parallel plan's ≤2.5 target at depth 6 and just outside at
  depth 7; if that matters, hoisting one descent per ROW (the 15 slots share a centre) is the
  obvious lever and was not attempted.
- Gate evidence: the FULL sampled overlay (sampStart/sampIdx/sampW/sampFluid/rowOf + every base row
  field) is **bitwise identical** old vs new on the P2b latitude two-level meshes at N = 64, 128 and
  256, and the depth-7 bed's census and fluid mask are unchanged. The bed's field difference above
  is entirely the period fix.

**The decision Fable owns.** Take the fix (`nbx = fineExt/4`, `domain = nbx·hb`, and in the
distributed build the GLOBAL fine extent rather than `pres.fineExt()`, which is the block's), which
re-blesses any bed reference that carries sampled clouds near a boundary — or specify a different
period convention. Either way D0 lands the same refactor; only the constant changes.

## Standing rules (unchanged)

Escalation contract = setup-parallel plan §5 verbatim. Every rung: 149/149 both nets, the
relevant bitwise gates, measured numbers in the commit message, push submodule-then-umbrella.
The march profiler (M0) must never perturb the march when disabled.

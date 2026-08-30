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

## M1 — the attribution matrix (measured 2026-08-30, RTX 5080; `tests/study/amr_march_profile.py`)

**Verdict: H-band and H-launch, in that order. H-iters and H-mg are refuted. The graded mesh's
cell saving is real and it is SPENT, not lost — on the least-squares sample clouds that make a
mixed-level band legal, and on a per-step cost floor that does not shrink with the mesh.**

### How the measurement was made honest

Two things had to be controlled before any of this was quotable.

1. *The box is shared.* A neighbouring job on the same GPU moved a FIXED workload from 3540 to
   1210 ms/step across three consecutive 200-step windows while its iteration counts stayed at
   58/59/56 — every phase scaled by the same factor, so it is contention, not physics. So the
   script has a `--structure` mode whose outputs (cells, band, MG shape, iteration counts) are
   wall-clock-free, and a `--time` mode that INTERLEAVES the arms window by window, both Flows
   resident, so a ratio survives what an absolute number does not. Every window's ms/step is
   printed; the numbers below are from windows where the box was quiet (the depth-7 uniform arm
   at 892 ms/step against P3c's 895 on the same mesh).
2. *The depth-7 pair has no lever.* At the shipped gap floor n = 4 the arms differ by 1.07× in
   cells — smaller than the contention noise. The lever comes from the policy dial instead: the
   `g<K>` arms are the same graded policy at gap floor n = K, and **n = 2 reproduces the depth-8
   pair's cell ratio (1.586× vs 1.619×) at a depth where both arms fit in one 16 GB GPU**. That is
   what makes the anomaly reproducible under a profiler.

(Trap found on the way, worth knowing suite-wide: with the OpenMP host backend in the default
prefix, a mesh/`set_solid` path driven by a PYTHON-callback SDF at `OMP_NUM_THREADS=8` becomes a
GIL convoy — the Z&H arm sat at 3% CPU for 20 minutes and finished in ~1 minute at
`OMP_NUM_THREADS=1`. The bed is immune because it uses the native `set_solid_spheres`.)

### The structure table — RCP bed, depth 7, gap-floor ladder (contention-free)

| arm | leaves | cells ÷ u | overlay rows | LS2 slots | overlay CSR | CSR × u | MG levels / Σ cells | pres it | mom it |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| u (uniform band) | 1 923 587 | 1.00 | 232 227 | 0 | 3 483 405 | 1.00 | 8 / 2 214 143 | 49.6 | 20.0 |
| g n=4 (shipped) | 1 794 535 | 1.07 | 229 485 | 31 095 | 8 456 826 | 2.43 | 8 / 2 078 105 | 50.5 | 19.3 |
| g n=3 | 1 592 032 | 1.21 | 212 872 | 119 891 | 20 882 784 | 5.99 | 8 / 1 862 516 | 58.3 | 17.6 |
| g n=2 | 1 212 730 | 1.59 | 163 438 | 223 191 | 32 658 174 | 9.38 | 8 / 1 446 159 | 54.3 | 14.7 |
| g n=1 | 690 488 | 2.79 | 79 647 | 188 458 | 18 735 248 | 5.38 | 8 / 857 403 | 53.8 | 11.8 |

Depth 8, the resolution P3c's headline came from:

| arm | leaves | overlay rows | LS2 slots | overlay CSR | MG levels / Σ cells |
|---|---:|---:|---:|---:|---|
| u | 11 350 032 | 953 419 | 0 | 14 301 285 | — (see the gap below) |
| g n=4 | 7 010 669 | 691 813 | 429 694 | 64 352 523 | 9 / 8 364 736 |

**1.619× fewer cells, 1.378× fewer band rows — and 4.50× MORE overlay CSR.**

### The timed table — depth 7, arms interleaved, quiet window (ms/step)

| phase | u | g n=2 | Δ |
|---|---:|---:|---:|
| momentum solve | 66.7 | 32.2 | −34.5 |
| pressure solve | 821.0 | 842.0 | +21.0 |
| &nbsp;&nbsp;· binary matvec | 33.7 | 22.3 | −11.4 |
| &nbsp;&nbsp;· **overlay matvec** | **17.2** | **227.0** | **+209.8** |
| &nbsp;&nbsp;· projection | 31.2 | 22.8 | −8.4 |
| &nbsp;&nbsp;· MG preconditioner | 718.4 | 554.3 | −164.1 |
| cf delta kernels | 1.1 | 3.9 | +2.8 |
| finish projection | 2.4 | 3.8 | +1.4 |
| **step** | **892.5** | **886.7** | **−5.8 (1.007×)** |

1.586× fewer cells buys 1.007×. The books balance exactly: the coarser mesh saves 164 ms in the MG
preconditioner and 35 ms in the momentum solve, and hands 210 ms of it straight back to the overlay
matvec.

### The second geometry family — Z&H N=128, arms interleaved (ms/step)

| | a (uniform control) | b (two-level latitude) |
|---|---:|---:|
| leaves | 268 752 | 118 812 (2.26× fewer) |
| overlay rows / LS2 / CSR | 16 776 / 0 / 251 640 | 7 284 / 444 / 195 192 |
| momentum solve | 148.8 | 133.8 |
| · overlay matvec | 7.3 | 10.6 |
| · MG preconditioner | 246.3 | 242.8 |
| pressure iters | 13.0 | 14.0 |
| **step** | **449.2** | **438.5 (1.024×)** |

The anomaly reproduces off the bed — **2.26× fewer cells, 1.02× faster** — but by the OTHER
mechanism: this arm has almost no sample clouds (444 LS2 slots, and a SMALLER CSR than the
control), and its MG preconditioner is nonetheless unchanged (246.3 → 242.8 ms) at less than half
the cells. At a few hundred thousand cells the V-cycle is entirely launch-bound.

### The floor, quantified

| mesh | leaves | MG levels | ms/step | pres it |
|---|---:|---:|---:|---:|
| bed depth 6 (levels [262 128, 2] — effectively uniform) | 262 130 | 7 | 275.6 | 59.9 |
| bed depth 7 uniform | 1 923 587 | 8 | 892.5 | 58.4 |

7.34× cells → 3.24× time. A linear fit gives **t ≈ 178 ms + 3.71e-4 ms/cell**: a per-step cost of
~178 ms that does not depend on the mesh at all — 20% of a depth-7 step and 65% of a depth-6 one.
That is the right order for launch latency: a step runs ~60 BiCGStab iterations × 2 preconditioner
calls × 2 V-cycles = ~240 V-cycles, and a 7–8 level V-cycle with a 60-iteration bottom solve is
~110 kernel launches, so 178 ms/240 ≈ 0.74 ms per cycle ≈ 7 µs per launch.

### The hypotheses, judged

- **H-iters — REFUTED.** Pressure iterations move by less than 20% across a ladder spanning 2.79×
  in cells (49.6 → 58.3 → 54.3 → 53.8), both arms sit at the 60-iteration cap for most of the
  march, and momentum iterations FALL on the coarser arms (20.0 → 11.8), which helps the graded
  side. Conditioning is not what eats the saving.
- **H-mg — REFUTED.** Every depth-7 arm has the same 8 MG levels, and the hierarchy's total cell
  count tracks the leaf count closely (Σ levels ÷ leaves: 1.151 for u, 1.192 for g n=2, 1.242 for
  g n=1). The V-cycle's WORK does shrink with the mesh; what does not shrink is its launch count.
- **H-launch — CONFIRMED, and it is the whole story on small meshes.** The ~178 ms/step floor
  above; on the Z&H pair it is the entire explanation (MG preconditioner flat at half the cells).
  On the depth-7 bed it explains the shortfall in the MG saving (1.30× where the cells give 1.586×)
  but not the sign of the result.
- **H-band — CONFIRMED, in a SHARPER form than pre-registered.** The pre-registration said "both
  arms have identical bands at matched depth". They do not: the graded band has FEWER rows
  (163 438 vs 232 227 at depth 7 n=2; 691 813 vs 953 419 at depth 8). The cost is not the row
  count, it is **what the rows contain**. On a uniform band all 15 chain slots of a row are
  IDENTITY slots — one CSR entry each. On a mixed-level band every slot crossing a 2:1 boundary
  becomes a degree-2 least-squares cloud, measured at **95–162 CSR entries per slot** (162.4 at
  n=4, 148.8 at n=3, 136.6 at n=2, 94.5 at n=1). So the overlay CSR GROWS as the mesh coarsens:
  9.38× at depth 7 n=2 for 1.59× fewer cells, and 4.50× at depth 8 for 1.62× fewer cells. Applied
  twice per BiCGStab iteration as a scattered gather, that is +210 ms/step at depth 7 n=2 — more
  than the 164 ms the smaller mesh saves in the preconditioner. Measured 13.2× on a 9.38× CSR
  ratio, so the gather's irregularity costs a further ~1.4× on top of the raw entry count.

**The one cell of the matrix that is missing.** The depth-8 UNIFORM bed cannot be marched here: it
needs ~20 GB and this GPU has 16, and it dies in `set_solid` (the overlay census still prints,
which is where its band and CSR numbers above come from). Its per-phase table needs the H100:
`core/tests/study/amr_march_profile.py --geom bed --depth 8 --arms u,g4 --time --steps 200
--repeats 3`. Nothing in the verdict depends on it — the depth-7 n=2 pair reproduces the same cell
ratio with the same signature, and depth 8's structural numbers (4.50× CSR at 1.62× fewer cells)
point the same way — but the confirmation is worth one job.

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

# Next phase: march-time economics + the distributed mixed-level band

*Plan, 2026-08-30 (Fable). Follows `amr_setup_parallel_plan.md` (complete, F1 resolved) and
`amr_mixed_level_cut_band_plan.md` (P3c). Ownership per rung is marked **[OPUS]** (execution
against settled decisions) or **[FABLE]** (a decision gate — Opus stops there and hands back,
same escalation contract as the setup-parallel plan §5, which applies verbatim here).*

**Status (2026-09-04): BOTH PHASES EXECUTED except two items, one of which needs a decision and
one of which needs cluster time.** Everything below is on `main` and gated; read this box first
and the rung bodies only for detail.

| rung | state |
|---|---|
| M0 profiler | DONE (`bef993b`, `3d7c896`) — `PECLET_CORE_PROFILE_STEP=1`; off = one bool test |
| M1 attribution matrix | DONE (`1c9517d`) — verdict: H-band + H-launch; H-iters, H-mg refuted |
| M2 verdict | DONE (`bc2b117`) — H-launch accepted; H-band → cloud economy |
| M2a cloud-economy table | DONE (`94cdc28`) — §M2a results |
| **M2b pick production rho/N** | **OPEN — [FABLE] decision, the table is waiting** |
| M2c slot caching | DONE (`dd5bc83`) — bitwise; wall-clock unmeasured (box contended) |
| F2 (LS cloud period) | RESOLVED (`dfe8065`) — §Findings |
| D0 probe-set clouds | DONE (`c8c19fc`) |
| D1 fixpoint wiring | DONE (`fe86380`) — ctest count 149 → **153** |
| D2 distributed band | DONE (`96fc78d`) — np=1 bitwise, np=2/4 ~3e-7 |
| D3(a) np>1 class | RULED — accepted; §D3 rulings |
| D3 setup cost | RESOLVED (`5a271a2`) — probe-only discovery + aligned lattice |
| **D3(b) cluster runs** | **OPEN — needs billed Snellius time; user go-ahead required** |

Two phases, independent — M can run before, after, or interleaved with D.

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

#### M2 VERDICT (Fable, 2026-08-30) — accept H-launch; attack H-band through cloud economy

M1's attribution splits into a term worth accepting and a term worth one bounded campaign.

**H-launch: ACCEPTED as-is.** The ~178 ms/step floor is a SMALL-MESH tax: 20% of a depth-7 step,
65% of a depth-6 one — and a vanishing fraction at the production sizes the campaign actually
targets (7M cells at depth 8, ~5× that at depth 9). Fusing V-cycle launches / CUDA graphs is
device-side engineering whose natural home is the device-assembly campaign
(`amr_device_assembly_plan.md`); it is noted there, not here.

**H-band: the clouds are ~10× oversampled, and that is the honest lever.** A degree-2 LS needs
12 points; the shipped rho = 2.2·max(h,H) gathers 95–162. Shrinking the cloud attacks the CSR
directly (the +210 ms/step term) — but it changes the WEIGHTS, i.e. march numerics, so it goes
through a pre-registered a-priori ladder, not a tuning loop ([[first-principles-over-literature]]):

- **M2a [OPUS] — the cloud-economy study. Measurement only, no production change.** Sweep
  rho ∈ {2.2, 1.8, 1.5}·max(h,H) and, orthogonally, a nearest-N candidate cap
  (N ∈ {16, 24, 32}; order by (distance², then global Morton key) — deterministic and
  decomposition-independent; the weights change under any cap, so re-sorting is licensed here,
  unlike in D0). For each variant: (i) the P2b ladder's seam-offset convergence order (the
  campaign's accuracy instrument — must stay in the 1.7–1.9 class), (ii) the depth-7 bed
  permeability offset vs the uniform arm (current class: +0.26%), (iii) overlay CSR size and the
  M0-profiled overlay-matvec ms/step, (iv) the degraded-row count (must stay 0 on the bed —
  a cloud too small to solve falls down the cascade, and M1 retired that risk at rho = 2.2).
  Deliverable: one table. STOP (do not pick a winner) — that is M2b.
- **M2b [FABLE] — pick the production rho/N from M2a's table.** Not before the data exists.
- **M2c [OPUS] — slot-value caching in the sampled matvec kernels. Bitwise, independent of
  M2a/M2b.** `ghostApplyDeltaSampled` / `ghostDivergDeltaSampled` re-gather the same slot's CSR
  sum up to twice per row (the ±faces share interior slots). Evaluate each of the row's 15 slot
  sums ONCE into scratch, in today's per-slot CSR order, then combine in today's order — pure
  read-dedup, bit-identical by construction, gated by the standing bitwise gates (depth-7 bed
  fields, seam ctest) + an M0 before/after on the bed. Expected ~1.3–1.6× on the overlay matvec;
  worth taking regardless of where M2b lands.

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
  *Status 2026-08-30: escalated as F2 (the bin search's period was 4 fine cells short — a
  pre-existing bug, not an enumeration failure), RESOLVED by Fable (true-period fix landed on
  `main` first, separately). The rewrite is parked on `dev/d0-probe-clouds`, already proven
  bitwise against the FIXED bin search on every gate geometry; remaining work = the merge +
  gate rerun per the F2-resolution instruction in §Findings.*
- **D1 [OPUS]** — DD2: wire the sampled builders into the fixpoint; np=1 bitwise ctest.
  *DONE 2026-08-30 (core `fe86380`).* Frames threaded (global fine extent + block frame shift,
  defaulting to the single-rank values); the sampled builders and the mixed openness rule joined
  `prepareDistributed`'s fixpoint; new ctest `tests/test_amr_distributed_seam_mpi.cpp` at
  np=1,2,4,8 (ctest count 149 → **153**). Found and fixed a latent bug the guard had been hiding:
  sampled mode built `presMG_` unconditionally while `gpOp0()` reads `presMGD_` when distributed,
  so the ghost solver pointed at an empty operator.
- **D2 [OPUS]** — drop the single-rank guard; np=2,4 acceptance per DD4; measure distributed
  setSolid scaling (the F1 fix should give near-single-rank per-rank cost).
  *DONE 2026-08-30 — see §D2 results below.*
- **D3 [FABLE]** — review the np>1 numbers, decide whether the ~5e-12 class holds for the
  sampled path or the deviation needs attribution; then the depth-9 TWO-ARM run on 2×H100
  *Status 2026-08-30: the setup-cost item is RESOLVED (see the D3 resolution under §D2 results)
  and the ~3e-7 class is RULED ON (accepted — see below). Remaining: the depth-9 two-arm run,
  now unblocked and queued as D3b [OPUS-executable] below.*
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

## D2 results — the mixed-level cut band is distributed (2026-08-30)

`setGhostSampled` has no single-rank guard left. Test geometry: the two-level LATITUDE map (the
seam family that actually produces sample rows — a perpendicular jump produces none), WORLD
(`initMpi`) vs SELF, leaf-matched by global Morton code.

### Acceptance (DD4)

| np | WORLD vs SELF, 3 steps | gate |
|---:|---|---|
| 1 | `|d|max = 0.000e+00` (scale 1.58e-01) — **BITWISE** | == 0 |
| 2 | rel 3.08e-07 | ≤ 5e-6 |
| 4 | rel 2.63e-07 | ≤ 5e-6 |

The overlay BUILD is exact: at np=2 the two ranks produce 42 rows / 570 identity / 36 LS2 / 6718
CSR entries each — 2× each of those is 84 / 1140 / 72 / 13436, the single-rank build to the last
count — and a per-row fingerprint (CSR length + Σw + Σ|w| keyed by global Morton code) is **bitwise
identical** to the single-rank overlay. So the residual is the solvers' global reductions, not the
clouds.

Its shape, for D3 to judge (the same run at several step counts):

| steps | np=2 | np=4 |
|---:|---|---|
| 1 | 4.73e-09 | 5.25e-09 |
| 2 | 5.85e-07 | 5.07e-07 |
| 3 | 3.08e-07 | 2.63e-07 |
| 6 | 1.40e-07 | 5.64e-08 |

It starts at the reduction floor (~5e-9), peaks mid-transient and **decays** toward the steady
state — it does not accumulate. The identity-only reference (same test, `SEAM_UNIFORM=1`, so the
band is uniform and every chain slot is an identity slot) sits at 1.5e-8 / 2.5e-8 at 3 steps, so
the sample clouds add about an order of magnitude to the transient peak and nothing at the fixed
point. **This is not the ~5e-12 class DD4 names**; that class was measured on operator-level
comparisons, whereas this is three steps of a Krylov march. D3 owns whether it needs attribution
beyond "global reductions reassociate".

### Two bugs the acceptance caught (both invisible single-rank)

1. **`t.level(j)` on an extended slot.** D2 made the covering-leaf probe `j` an EXTENDED slot, and
   the identity-slot test still asked the LOCAL octree for its level — reading past the end of the
   level array for any ghost, reporting a wrong level, and so sending same-level covers down the
   least-squares branch. Signature: the np=2 census summed to 129 LS2 slots where single-rank had
   72. Same bug in `buildMomSeamDelta`'s seam test. Both now go through `pres.levelOf`.
2. **`buildMomSeamDelta` built its stencil in the LOCAL frame.** The row's centre and its virtual
   ±1 probe positions are WORLD coordinates; without the frame shift every rank whose block does
   not start at the origin evaluated the geometry of the wrong place. This one was worth ~1e-2
   relative after one step and diverged the march by ~1e3 per step. Signature: the overlay
   fingerprint matched bitwise while the march still blew up — i.e. the build was right and a
   consumer was wrong.

### Distributed setup cost — the honest number

`setSolid` per rank does NOT stay at the single-rank cost, and the reason is the discovery
fixpoint, not the builders:

| lmax | np=1 | np=2 | np=4 |
|---:|---|---|---|
| 2 | 1 round, 0 ghosts | 5 rounds, 606 ghosts | 7 rounds, 909 ghosts |
| 4 | 1 round, 0 ghosts | 8 rounds, 2860 ghosts | 9 rounds, 4290 ghosts |
| 6 | 1 round, 0 ghosts | 8 rounds, 2020 ghosts | 8 rounds, 3030 ghosts |

(`PECLET_CORE_PROFILE_SETUP=1` now prints the round count and ghost count.) The classic ±2 chain
converges in ≤3 rounds; the sampled path needs **5–9**, because an LS cloud descends a whole box
and D0's descent deliberately does not descend into a `kPending` region — it learns one octree
level per round rather than registering one miss per fine cell. Each round is a full rebuild plus
a collective, so on the small test mesh the wall cost per rank rises with np instead of falling.
The ghost count is also much larger than the classic registry (np=4, lmax=4: 4290 ghosts against
1430 local leaves), because the clouds reach ~2.2·max(h,H) rather than ±2 cells.

Bounded, correct, and clearly improvable — the obvious lever is to let the descent register a
BOUNDED set of probes inside a pending region (e.g. at the coarsest level present in the block)
so it learns several levels per round, trading a larger miss set for fewer collectives. That is a
design choice with a real trade-off, so it is flagged here rather than taken: **a D3 item.**

#### D3 resolution (Fable, 2026-08-30): discovery is probe-only + aligned-lattice pending regions

Taken and landed. Two independent pieces, and two wrong turns worth recording because each was
caught by a measurement, not by review:

1. **Probe-only discovery** (`discovery` flag on `buildGhostOverlaySampled`; only
   `prepareDistributed`'s fixpoint sets it). Control flow — every SDF evaluation, gate, and
   probe — is IDENTICAL to a real build, so the registered miss set is what the real build
   needs; but the least-squares work a discovery round discards is skipped: no candidate
   collection, no sorts, no normal matrices, no solves, no second (degree-1) descent of the same
   box, no pass 2, no census print. If discovery ever under-probed, the real build would hit
   `LeafHalo::resolve: unknown coord after finalize()` — a throw, never silent corruption.
   `mom_.build` also leaves the sampled loop (its ±1 probes are verbatim a subset of the face
   sweep's explicit `periodicNeighbor` calls — checked in `cut_cell.hpp`, whose only build-time
   probe is `neighbor(i,k)`) and runs ONCE after the fixpoint. The classic path's loop is
   untouched.
2. **Aligned-lattice pending regions** (`descendPending` on `forEachCoveringSlot`; discovery
   only). A pending region is no longer abandoned for the round (one octree level learnt per
   round): it is probed on an ALIGNED power-of-two lattice, stride chosen so the lattice is at
   most ~9 points per axis, floored at 4. Every leaf of size ≥ stride is stride-aligned, so it
   resolves this round; finer leaves surface next round as pending sub-regions one stride class
   smaller — the unknown shrinks geometrically, and each round is cheap.

   *Wrong turn #1*: recursive blind splitting to a 4-cell cap. Terminal-region corners inherit
   each cloud box's own offset, so overlapping boxes register DISJOINT coord sets and the miss
   map's dedup is defeated: 560 184 distinct pending coords and an 8.5 s round 1 on the lmax=6
   seam mesh (vs 0.4 s for the whole fixpoint before the "fix"). Alignment is what makes coords
   shared.
   *Wrong turn #2*: a FIXED stride of 4. A coarse background row (level L) has a cloud of radius
   2.2·h(L) — domain-sized at L=6 — and a stride-4 lattice over it IS the remote domain: 263 876
   coords. Single-rank the same box is cheap because the covers-shortcut steps through coarse
   leaves; the stride must scale with the region.

**Measured** (two-level latitude seam mesh, `SEAM_LMAX`, slowest-rank wall for `setSolid`,
`PECLET_CORE_PROFILE_SETUP=1` prints per-round build time + pending count):

| | np=2 | np=4 |
|---|---|---|
| lmax=6 before (D2) | 0.405 s (8 rounds × full builds) | 0.593 s |
| lmax=6 after | **0.104 s** (7 rounds ≈ 30 ms each + one full build) | **0.098 s** |
| lmax=4 before / after | 0.106 / 0.103 s | 0.101 / 0.087 s |

Round count at lmax=6 np=2: 8 full-build rounds → 7 cheap rounds (pending trace
8416 → 23376 → 7056 → 2478 → 542 → 42 → 0), final ghost count **2020 — identical to before**, so
the lattice inflates nothing. The absolute per-leaf numbers on these 1–6k-leaf test meshes are
fixed-cost dominated (collectives + launches ÷ tiny leaf counts); at bed scale the discovery
rounds cost ~µs/leaf against the one full build. Remaining slack if it ever matters: the
openness prober still writes its (discarded) α values each round, and the 542→42→0 tail rounds
could be trimmed by a finer lattice floor — both diminishing returns, neither taken.

Gates: 153/153 both nets (one battery run had `amr_distributed_view_np4/np8` livelock in
`MPI_Waitall` inside `DistributedGatherHalo::gather` — code this change does not touch — while a
SECOND MPI test battery from a concurrent session ran on the same box; both pass standalone,
stacks recorded; the known battery-load flake class, now seen on the view tests); depth-7 bed
set_solid + 20 steps BITWISE vs the D2 baseline; depth-6 mask bitwise; seam acceptance unchanged
(np=1 bitwise, np=2/4 in the 3e-7 class).

## M2a results — the cloud-economy table (Opus, 2026-08-30; `tests/study/amr_cloud_economy.py`)

Measurement only, per the M2 verdict; **M2b (picking the production rho/N) is FABLE's** and is
deliberately not taken here. Knobs (both inert at their defaults, verified bitwise on the depth-7
bed): `PECLET_CORE_GPS_RHO` (radius factor, default 2.2) and `PECLET_CORE_GPS_MAXN` (keep the N
nearest by (distance², global Morton key), default 0 = no cap; the kept set is emitted in the
unchanged canonical (bin, Morton) order, so only the WEIGHTS move, never the accumulation
discipline).

| variant | CSR d7 | ÷base | CSR d8 | ÷base | vs uniform d8 | LS1 d7 | LS1 d8 | bed k (d7) | offset vs uniform | seam gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| rho 2.2, no cap (shipped) | 8 251 091 | 1.00 | 61 887 407 | 1.00 | 4.33× | 0 | 0 | 5.854858e-01 | +0.247% | 4/4 |
| rho 1.8 | 5 980 191 | 0.72 | 38 577 176 | 0.62 | 2.70× | 0 | 5 | 5.853315e-01 | +0.220% | 4/4 |
| rho 1.5 | 5 182 401 | 0.63 | 29 721 806 | 0.48 | 2.08× | 7 | 747 | 5.852880e-01 | +0.213% | 4/4 |
| N ≤ 32 | 4 401 625 | 0.53 | 23 561 296 | 0.38 | 1.65× | 7 | 53 | 5.852828e-01 | +0.212% | 4/4 |
| N ≤ 24 | 4 152 865 | 0.50 | 20 123 997 | 0.33 | 1.41× | 103 | 1 310 | 5.852714e-01 | +0.210% | **3/4** |
| N ≤ 16 | 3 904 105 | 0.47 | 16 686 448 | 0.27 | 1.17× | 2 742 | 49 683 | **DIVERGED** (step 40) | — | **1/4** |

Uniform control (variant-independent — it has no sample slots): k = 5.840443e-01, reproducing
P3c's published value exactly. The shipped graded arm reads +0.247% where P3c published +0.256%;
the 9.4e-5 shift is the F2 period fix, exactly the ~5th-digit move F2's resolution predicted.
Overlay rows, identity slots and degraded-row count are IDENTICAL across every variant (229 485
rows / 0 degraded at d7; 691 813 / 0 at d8) — the cascade floor never fires, so `degraded` is not
the discriminator here. `LS1` is: it counts slots that fell to degree 1, which M2 established is
an O(1) non-decaying perturbation.

**The finding that matters: the bed permeability is NOT a sensitive enough instrument on its
own.** N ≤ 24 halves the d7 CSR and looks *better* than the shipped arm on the bed (+0.210% vs
+0.247%, i.e. closer to the uniform control) — and yet it FAILS the seam gate "LS2 matrix
perturbation decays (agg ord ≥ 0.6)", meaning the degree-2 reconstruction has stopped converging
and the arm is only accidentally close at this one resolution. N ≤ 16 fails three of four gates
and diverges the march outright. Reading the offsets alone would have selected exactly the wrong
variant; this is why the M2 verdict pre-registered a convergence instrument alongside the
permeability.

**Instrument coverage, honestly.** (ii) bed permeability, (iii) CSR, (iv) degraded rows: complete,
all variants, both depths where applicable. For (i) the seam-order instrument I ran
`tests/study_amr_seam_sample_order.cpp` — the study that ESTABLISHED the degree-2 requirement,
now carrying the same two knobs — which measures the LS2 reconstruction's convergence order
against the exact virtual sample across N = 64/128/256 in ~2 s per variant. That is a supplement,
NOT the thing the M2 verdict named: the **P2b march ladder's seam-offset convergence order was
not run**. Its cost is the reason — arm b alone at three resolutions is ~2 h per variant (Z&H
relaxes in ~9000 steps), so six variants is a >12 h unattended run on a box that has been
contended all session. If M2b wants it before committing, the command is
`tests/study/amr_zh_ladder.py --cf 1 --arms ab 64 128 256` per variant with the knobs exported;
the uniform arm a is variant-independent and needs running once.

**Timing (iii, ms/step) is also not reported**: the same contention that blocked M2c's wall-clock
number (a fixed workload wandered 886 → 5886 ms/step across windows today) makes it meaningless.
The CSR column is the contention-free proxy, and it is exact.

## D3 rulings (Fable, 2026-08-30)

**D3(a) — the ~3e-7 np>1 class: ACCEPTED as the sampled path's march-level
decomposition-independence class; no further attribution required.** The evidence Opus assembled
is exactly what an acceptance needs: the overlay BUILD is bitwise identical across decompositions
(per-row fingerprints keyed by global Morton code), the residual enters at the reduction floor
(~5e-9 after one step), peaks mid-transient and CONTRACTS toward the fixed point (1.4e-7 at
6 steps and falling), and the identity-only band shows the same mechanism at 1.5e-8 — the sample
clouds only amplify the transient, never the steady state. DD4's "~5e-12" was an operator-level
number; three steps of MG-preconditioned BiCGStab compose hundreds of reassociated global
reductions, and ~e-7 mid-transient is the expected shape of that. The ctest gate stays 5e-6.
What would REOPEN this ruling: a residual that grows with step count instead of decaying, or any
np-dependence in the steady-state permeability beyond the ~1e-10 class.

**D3(b) [OPUS-executable] — the cluster runs, now unblocked.** Two jobs, one recipe
(`tools/snellius_amr_bed.md`), in this order:
1. **Prefix first**: the Snellius `nvidia-cuda` prefix has NOT had rung 0's
   `-DKokkos_ENABLE_OPENMP=ON` rebuild yet — without it every setup gain of this campaign is
   absent on the machine where it costs money. Same one-liner as local; rebuild the dependent
   trees; rerun the standing fence (flow probe, ctest subset) before any billed arm.
2. **M1's missing cell**: the depth-8 UNIFORM bed per-phase table —
   `tests/study/amr_march_profile.py --geom bed --depth 8 --arms u,g4 --time --steps 200
   --repeats 3` on one H100 (fits in 80 GB). Confirmatory only; the M2 verdict does not wait
   for it.
3. **The depth-9 TWO-ARM run on 2×H100** — the accuracy-matched headline at R/h₀ = 48, the
   uniform control finally fitting split. Check job 26205730 (the pending single-GPU depth-9
   graded arm) before submitting: if it ran on the OLD code its numbers predate the F2 period
   fix and must not be mixed into the two-arm comparison. Right-size `--gpus-per-node`
   ([[snellius-access]]: billed per allocated GPU) and remember `--export=ALL,VAR=…`
   ([[snellius-sbatch-env-vars]]). Report the k pair, the census, and the fixpoint round count
   at np=2 — then stop; reading the headline is FABLE.

## Findings

### F2 (D0, 2026-08-30) — the LS cloud's periodic period is 4 fine cells SHORT: DD1 cannot be made bitwise without fixing it

**Status: RESOLVED 2026-08-30 (Fable) — the true-period fix is TAKEN and is on `main` (see the
resolution at the end of this finding). D0 is unblocked; the parked branch's remaining diff is
enumeration-only.**

*(The finding as escalated, kept verbatim for the record:)* D0 BLOCKED on a [FABLE] decision
(DD1's own escalation clause). The probe-set rewrite is written, gated and green everywhere
EXCEPT this; it is parked on branch `dev/d0-probe-clouds` (core), not on `main`, because landing
it would silently change march numerics.

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

#### F2 RESOLUTION (Fable, 2026-08-30) — take the true period; land it FIRST, separately from D0

There is no alternative convention to weigh. The minimum-image period of a periodic domain of
`fineExt` fine cells is `fineExt·h0` — the value `probeSlot`, `LeafHalo::wrap`, and every other
wrap in the codebase already uses. `((fineExt−1)/4)·4·h0` is an off-by-one (inclusive `bounds()`)
fed through a truncating divide, not a design. Blessed as found.

**Landing shape: two steps, so the numerics change and the refactor never share a diff.**

*Step 1 (done, this commit): fix the period inside the OLD bin-search code on `main`* — replace
the inclusive-bounds `ext` derivation with `nbx = max(fineExt)/4`, byte-for-byte the expression
the parked branch uses, everything else untouched. This makes `main` correct independent of the
D0 refactor, and it shrinks the branch's remaining diff to enumeration only.

*Step 2 (Opus): re-gate and land `dev/d0-probe-clouds`* — see the instruction below.

**Verification of step 1** (all against artifacts produced this session, single process,
`np.array_equal` / `cmp`, never printed hashes):

- *Inert where it must be inert*: the full sampled-overlay dumps on the P2b latitude meshes at
  N = 64/128/256 are BITWISE unchanged by the fix (centred geometry: same bin coordinates, no
  wrap engaged) — so every existing parity ctest and bitwise reference stands un-re-blessed.
- *The strong cross-check*: with the period fixed, `main`'s bin search is **bitwise equal to the
  parked D0 branch** on the mini-bed overlay dumps (N = 64, 128 — the harness where 66.8% of
  clouds engage the wrap) AND on the depth-7 RCP bed march (mask + all three velocity components
  after set_solid + 20 steps; `Σu_x = 9.07850260547263315e+03`, matching the branch's recorded
  value exactly). Two consequences: the bed's field shift under the branch was 100% the period
  fix and 0% the enumeration — and **the branch's probe enumeration is thereby proven faithful
  to the (fixed) bin search on boundary-crossing geometry**, retroactively completing the piece
  of D0's gate the wrong period had made unreachable.
- *Batteries*: 149/149 on `build_komp3` and `build_kcuda2`; depth-6 fluid mask bitwise vs
  `mask6_before.npy` (the mask never depended on clouds).

**Re-blessing consequences, stated honestly**: no stored reference changes (none carries sampled
clouds near a periodic face). The P3c bed permeabilities (d7 +0.256%, d8 −1.79%) were computed
with the short period; the depth-7 20-step probe moves `Σu_x` by 5.0e-5 relative under the fix,
so a rerun will shift k in the ~5th digit — the P3c conclusions are untouched, but bed numbers
from now on are corrected-period ones and must not be compared to the old logs at finer than
~1e-4 relative.

**Instruction for Opus (step 2 + the unblocked D rungs):**

1. **Land D0**: merge `dev/d0-probe-clouds` (core `76144a5`) onto the fixed `main`. The
   `ghost_projection_sampled.hpp` merge must leave exactly the enumeration change (the
   `detail::forEachCoveringSlot` descent + the canonical `(bin, Morton)` re-sort); the period
   expression is now IDENTICAL on both sides — if the merge shows any other semantic diff in
   that region, stop. Gates (all expected bitwise-clean, since fixed-main == branch was already
   measured this session): latitude dumps N = 64/128/256, mini-bed dumps, depth-7 bed march
   (mask + u bitwise vs the step-1 baseline), depth-6 mask, 149/149 both nets. Record in the
   commit message the branch's measured setSolid cost (2.4 → 2.7 µs/leaf @8t on the d7 bed); if
   that regression matters later the noted lever is hoisting one descent per ROW — the 15 slots
   share a centre — but do NOT attempt it inside D0.
2. **D1 as planned (DD2)**, with one addition now settled by F2: `buildGhostOverlaySampled` must
   receive the **GLOBAL fine extent and the block's frame shift** — `pres.fineExt()` is the
   block's on np>1, which would be F2's decomposition-dependence in a new coat. Thread
   `DistributedOctree`'s global fine size / block fine origin through to the builder's
   `gfine`/`shiftG` (the branch already stubs both, with a comment saying exactly this);
   default them to `(pres.fineExt(), {0,0,0})` so np=1 stays bit-identical by construction. The
   `keyOf` Morton sort key must use the GLOBAL lo (the branch already writes `b[0] + shiftG`).
3. **D2 as planned.** Nothing in DD3/DD4 changes.

## Resuming this campaign (written at session close, 2026-09-04)

Two items are open. Neither is blocked on code — one is a decision, one is machine time.

### 1. M2b — pick the production cloud (FABLE decision; the data is in §M2a results)

The table is complete and the shape is clean: **rho 1.8, rho 1.5 and N ≤ 32 all pass every gate**;
N ≤ 24 passes the bed permeability while FAILING the LS2-convergence gate; N ≤ 16 fails three
gates and diverges. N ≤ 32 is the aggressive end of the safe set and takes the depth-8 overlay CSR
from 4.33× the uniform arm's down to 1.65×, which is where M1's +210 ms/step penalty lives.

Before deciding, two things a reviewer should know:
- The **P2b march ladder was not run** (cost: >12 h for six variants). The per-variant seam
  evidence in the table comes from `study_amr_seam_sample_order.cpp`, the study that established
  the degree-2 requirement, now carrying the same knobs at ~2 s per variant. If M2b wants the
  named instrument first: `tests/study/amr_zh_ladder.py --cf 1 --arms ab 64 128 256` per variant
  with `PECLET_CORE_GPS_RHO` / `PECLET_CORE_GPS_MAXN` exported; arm a is variant-independent and
  needs running once.
- Whatever is chosen, changing the DEFAULT is a numerics change: it re-blesses the bed references
  (the shipped arm reads +0.247% today) and needs the standing bitwise gates re-baselined, not
  merely re-run. The knobs themselves are already on `main` and inert at their defaults.

### 2. D3(b) — the cluster runs (needs billed Snellius time; ask the user first)

In this order, one recipe (`tools/snellius_amr_bed.md`):

1. **Prefix rebuild FIRST.** The Snellius `nvidia-cuda` prefix still has never had rung 0's
   `-DKokkos_ENABLE_OPENMP=ON`. Until it does, every setup gain of this campaign (setSolid
   127 → 2.4 µs/leaf, and D3's fixpoint fix) is absent on the machine that bills. Same one-liner
   as local; rebuild the dependent trees; rerun the standing fence before any billed arm.
2. **M1's missing cell** — the depth-8 UNIFORM bed per-phase table, one H100 (it needs ~20 GB and
   dies in `set_solid` on the local 16 GB card):
   `tests/study/amr_march_profile.py --geom bed --depth 8 --arms u,g4 --time --steps 200
   --repeats 3`. Confirmatory only; the M2 verdict does not depend on it.
3. **The depth-9 two-arm run on 2×H100** — the accuracy-matched headline at R/h₀ = 48.

Three cautions carried forward:
- Job **26205730** (the pending single-GPU depth-9 graded arm) predates the F2 period fix. If it
  ran, its k is on the old cloud metric and must NOT be mixed into the two-arm comparison.
- Bed k values from now on are corrected-period ones; do not compare to pre-F2 logs finer than
  ~1e-4 relative (measured: the d7 graded arm moved +0.256% → +0.247%).
- The **NBX per-round tag rotation** (`10294e6`, 2026-09-02, landed after this campaign) fixed an
  at-scale halo hang/corruption that bit at 1536 ranks with one topology build per MG level. Every
  distributed number in this document predates it and was taken at np ≤ 8 locally; the cluster
  runs are the first that exercise both together.

### Things measured but NOT quotable, and why

Two numbers this campaign owes and could not take: **M2c's wall-clock speed-up** and **any
ms/step in M2a**. This box was shared with another agent's GPU work throughout, and a *fixed*
workload wandered 886 → 1887 → 3853 → 5886 ms/step across consecutive windows of one run;
normalising against an untouched kernel did not stabilise it either. The contention-free proxies
(analytic gather counts for M2c, CSR sizes for M2a) are in their commits and are exact. To close
them, wait for a quiet box and run
`tests/study/amr_march_profile.py --geom bed --depth 7 --arms u,g2 --time --steps 60 --repeats 3`,
comparing the overlay-matvec row against M1's recorded quiet values (u 17.2, g2 227.0 ms/step at
892.5 / 886.7 ms/step total).

## Standing rules (unchanged)

Escalation contract = setup-parallel plan §5 verbatim. Every rung: 149/149 both nets, the
relevant bitwise gates, measured numbers in the commit message, push submodule-then-umbrella.
The march profiler (M0) must never perturb the march when disabled.

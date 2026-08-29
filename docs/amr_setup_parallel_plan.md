# Parallel AMR setup: the builder restructure (post-scene-layer)

*Plan, 2026-08-30 (Fable analysis; implementation intended for an Opus session). Companion to
`amr_mixed_level_cut_band_plan.md` (the campaign this serves) and
`../../docs/AMR_GEOMETRY_SETUP_REQUIREMENTS.md` §5–6 (the scene-layer handoff this builds on).*

**Status: PLANNED — decisions D1–D4 below are settled by Fable; the rungs are execution.**

## 1. The measured starting point (the current denominator — do not reuse old numbers)

The scene layer took `AmrFlow::setSolid` from 127 → **13.1 µs/leaf** (depth-7 RCP bed, 1.79M
leaves, RTX 5080, single-threaded; `.sdf-campaign-probes/time_setsolid.py`). The remaining cost
is FIVE serial host builders (`PECLET_CORE_PROFILE_SETUP=1`, 2026-08-30):

| phase | µs/leaf | share | parallel structure (verified in source) |
|---|---:|---:|---|
| `mom_.build` (cut stencils) | 4.2 | 32% | per-leaf loops over `i < n`, per-leaf array writes |
| `presMG_.build` (openness per level) | 2.6 | 20% | per-level, per-cell loops |
| cf overlays (4 builders) | 2.2 | 17% | **already per-leaf staged** (`per[i]` + `compactCsr`) |
| `buildOpenness` (mixed rule) | 2.0 | 15% | per-leaf face loops |
| `buildGhostOverlaySampled` | 1.4 | 11% | **sequential row append** (`slot = ov.base.n`) — needs two-pass |
| pockets, MG builds, device assembly, uploads | ~0.9 | 5% | leave serial (already cheap) |

Target: **13.1 → ≤2.5 µs/leaf at 8–16 threads** ⇒ depth-8 `setSolid` ≈ 15–20 s (from 95.6 s),
depth-9 ≈ 1–2 min — and ~77% of a Snellius bed job's bill stops being serial host time.
Evaluation is NO LONGER the lever (~0.03–0.05 µs/eval); the lever is loop parallelism and
allocation hygiene. Re-measure every rung against the CURRENT number (handoff trap 6.3.4).

## 2. Decisions (settled here — Opus implements, does not revisit)

**D1 — Host parallelism via OpenMP pragmas in the builders, NOT a Kokkos host-space change.**
The python module builds Kokkos SERIAL+CUDA; switching its host execution space would change the
device build globally and every downstream consumer. Instead: `#pragma omp parallel for
schedule(static)` on the builder loops, guarded `#ifdef _OPENMP`, with `-fopenmp` (and `-mfma`,
the §5 note) added to the **amr_bindings target's host flags only** (`python/CMakeLists.txt`;
`-Xcompiler` through nvcc where applicable). Serial build stays byte-identical in behaviour.
*Why not device assembly now:* these builders are host-by-design for oracle parity (weights
host-built, shared verbatim — the parity-by-construction contract); device-resident overlay
assembly is a later campaign (`amr_device_assembly_plan.md`), not this one.

**D2 — Determinism by construction, verified by thread-count invariance.** Every parallelized
builder must produce BITWISE-identical output at OMP_NUM_THREADS=1 and =16. This is achievable
because none of the five builders carries a cross-leaf floating-point reduction — each leaf's
rows/values depend only on (octree, scene). The patterns:
- per-leaf array writes (`mom_.build`, `buildOpenness`, `presMG_.build` per level): disjoint
  writes, `schedule(static)` — nothing else needed.
- per-leaf staged vectors (`cf` builders): parallelize the leaf loop filling `per[i]`;
  `compactCsr` stays serial (cheap, order-preserving).
- sequential append (`buildGhostOverlaySampled`): TWO-PASS — pass 1 (parallel) classifies each
  leaf (clean/row) and computes its slot counts; exclusive scan (serial) assigns row indices
  and CSR offsets in leaf order; pass 2 (parallel) fills each row at its precomputed offsets.
  Row order = leaf order = today's serial order ⇒ the CSR is bitwise-identical, and the
  `test_seam_sampled` parity and mask references hold without re-blessing.
If any builder turns out to carry hidden sequential state, that is an ESCALATION (see §5),
not a workaround.

**D3 — The collect-points → batch-evaluate → build-rows restructure is DEFERRED, not dropped.**
With `SphereBedQuery` inlining into the templated `SdfFn` at ~0.04 µs/eval, batching buys the
HOST path little; its real value is device-resident assembly later. Restructuring row builders
around point batches now would churn the exact code being parallelized. It moves to the
device-assembly campaign; this plan's rungs must not make it harder (keep sampling call sites
localized per leaf).

**D4 — Scope guard.** No numerics change anywhere: same samples, same gates, same row content,
same order. No `geom/` edits (the scene layer is frozen; trap 6.3.1: never touch the
fma-canonical expressions). `findPocketCells`, nullspace, MG hierarchy internals, device
assembly: untouched. Distributed path: untouched (the builders run identically per rank; the
pragmas are rank-local and safe, but DO NOT restructure any distributed logic).

## 3. Rungs (commit per rung; every rung ends green and MEASURED)

Gates for every rung: (a) 148/148 core ctests on `build_komp3` (OpenMP battery:
`OMP_NUM_THREADS=8 OMP_PROC_BIND=false`) and `build_kcuda2`; (b) `time_setsolid.py 7` bitwise
mask vs `.sdf-campaign-probes/mask6_before.npy` at depth 6; (c) thread-count invariance:
`set_solid` + 200 steps, fields bitwise at OMP_NUM_THREADS=1 vs 16; (d) the rung's µs/leaf,
in the commit message, measured at 1 and 8 threads.

- **Rung 0 — build flags.** `-fopenmp -mfma` on amr_bindings host flags (D1). Expect ~15% from
  `-mfma` alone at 1 thread (§5 note: fma is a libm call today). No source change.
- **Rung 1 — cf builders parallel.** The `per[i]` leaf loops in `buildCfLapDelta` /
  `buildCfDivDelta` / `buildCfGradDelta` / `buildCfUfDelta`. Lowest risk (staging already
  per-leaf), a clean template for the rest. 2.2 → ~0.3 µs/leaf expected at 8 threads.
- **Rung 2 — `buildOpenness` + `presMG_.build` parallel.** Per-cell/per-level loops; verify
  per-face slot ownership is cell-major (it is — `forEachFaceFull` is cell-major) before
  writing. 4.6 → ~0.6.
- **Rung 3 — `mom_.build` parallel.** The largest phase; several consecutive per-leaf loops.
  Watch for: the ±probe classification writing neighbour metadata (if any loop writes to
  OTHER leaves' slots, that loop stays serial or becomes owner-computed — check first, escalate
  if ambiguous). 4.2 → ~0.6.
- **Rung 4 — sampled overlay two-pass.** D2's third pattern. The delicate one: row order and
  slot CSR order are CONTRACTUAL (device upload, parity ctest, oracle equivalence). Pass-1
  classification must call the same gates as today's single pass (D6 determinism of the plan:
  row state is a pure function of geometry — so classifying twice is safe by contract).
  1.4 → ~0.3.
- **Rung 5 — re-measure + record.** `time_setsolid.py` at depths 6/7/8, 1/8/16 threads; update
  `AMR_GEOMETRY_SETUP_REQUIREMENTS.md` §1 table and `amr_mixed_level_cut_band_plan.md` P3c
  observation 4; memory hook. If ≤2.5 µs/leaf at 8 threads: DONE. The oracle
  (`flow_oracle.hpp`) shares the cf/overlay builders, so it speeds up for free — verify its
  ctests unchanged, do NOT add oracle-only pragmas beyond what it inherits.

## 4. What Opus must know (traps, verbatim from the handoff + this campaign)

- OpenMP batteries: `OMP_NUM_THREADS=8 OMP_PROC_BIND=false` — 48 unbound threads = an hour.
- Never modify classic-path kernels/builders' semantics; the "call both, empties no-op"
  pattern is contractual.
- `.sdf-campaign-probes/flow_probe.py` runs at 4 threads with a pinned u_sum — do not "fix" its
  thread sensitivity, it is a pre-existing host-reduction property, not a regression.
- Speedups do not multiply across baselines; measure each rung against its predecessor.
- Push after gated rungs: submodule first, umbrella last (dangling-pointer check in
  [[push-directly-to-main]]).

## 5. ESCALATION (stop and hand to Fable — do not improvise)

1. Any builder loop that writes to another leaf's slots, carries state across iterations, or
   needs an atomic/reduction to parallelize — stop, document the loop and the dependency in
   THIS file under a "## Findings" heading, leave that loop serial, continue other rungs.
2. Any bitwise gate failure (thread-count variance, mask diff, parity ctest) that is not an
   obvious ordering bug you can fix by restoring serial order — stop, record the failing gate,
   the diff magnitude, and the rung.
3. Any temptation to change WHAT a builder computes (gates, sample positions, row content) —
   forbidden outright; if a rung seems to require it, that rung is mis-scoped: stop.
4. Anything touching `geom/`, the distributed path's logic, or Kokkos configuration beyond the
   amr_bindings host flags — out of scope; stop.
Escalations go in "## Findings" here + the memory file (`amr-mixed-level-cut-band-plan`), so a
Fable session picks them up with context.

# Mixed-level cut cells: local surface refinement for the AMR ghost projection

*Plan, 2026-08-24 (F. Peters / Claude session). Companion to `amr_collocated_projection.md`
(the AMR ghost port + graded-mesh record) and flow's `collocated_invisible_subspace.md` /
`fluid_only_constraint_plan.md` (the campaign verdicts that constrain any design here).*

**Status (2026-08-27): PHASE 2 COMPLETE — the gate passed and the design is measured.** The
Snellius verdict came in 2026-08-25 and the ghost projection is the production default suite-wide
(flow c672014, core 7472306). Phase 0 (M1–M3) is done (§8); Phase 1 rungs 1–4 are done (momentum
ξ-row seam correction, wall-aware C/F gates, device mirror, graded refinement policy API); Phase 2
is done (§Phase 2: family-free on a seamed mesh at N=128, and the seam offset converges at ~1.7–1.9
so **seams do not set the order**).

**Phase 3 opened, blocked, and UNBLOCKED (P3a/P3b, 2026-08-27).** The two-sphere gap case
calibrated n (the plan's default n = 4 is supported: 4–8 cells across the throat, 0.1–1.3%, at
18–30× fewer cells) and found a hard march blow-up on 2 of 12 throat-graded meshes. P3b bisected
it — NOT rung 2 (first attribution, refuted) but the pre-existing `cfDiv_` C/F divergence delta
firing at CUT rows, where the ghost closure overlay owns the constraint (a C2 violation) — and
fixed it with a one-line row gate (`rowFluid → rowRegular` for `cfDiv_`/`cfGrad_`, matching
`cfMom_`; inert by geometry on finest-band meshes, measured bit-identical).

**Phase 3 bed run DONE at depth 7 (P3c, 2026-08-28): the gap-graded policy reproduces the RCP
bed's permeability to +0.256%**, marching stably with 0 degraded rows. The accuracy-matched
CELL-COUNT headline is not yet quotable: the saving is only 1.07× at depth 7 (the band already
covers 92% of that domain) and 1.62× at depth 8 — but depth 8 exceeds the local RTX 5080 twice
over (the control needs ~21.5 GB of 16.3; the graded arm then hits `cudaErrorLaunchTimeout`).
**The headline needs a compute-mode cluster GPU; everything else is ready.** Still open:
pocket exclusion in LS clouds, **sub-face closures** (now bounded — an accuracy item, not a
stability blocker), and the distributed sample halo.

*Original gating note, kept for the record:* implementation was gated on flow's clean-ladder
R=24/32 order result + the np>=16 instability re-diagnosis; if that ladder had disappointed, the
seam design below would have survived in outline but the closure family it generalizes would have
had to be revisited first. Phase 0 was allowed to run before the gate.

## 1. Why

For porous media / dense packings, the current refinement policy (`refineToSdf`: a uniform
finest-level band covering the entire solid surface) puts fine cells almost everywhere in the
fluid phase — the AMR benefit evaporates exactly where AMR is wanted most. The goal: a level
function L(x) on the surface — fine in narrow throats and near particle contacts, coarse on
smooth wide sphere caps — so **cut cells exist at different refinement levels**, while keeping
(near) second-order spatial accuracy and every stability/uniqueness property the attractor
campaign established.

Measured motivation: the dilute-sphere graded run (amr_collocated_projection.md) showed a
ghost band at 0.9% of the uniform cell count running stable with flat BiCGStab — the entire
graded error was the far-field C/F flux, half-removed by the quadratic C/F scheme. Dense
packings invert the geometry: the fine fraction is ~100% because the *surface* forces it.
Patch-level surface grading is what recovers the AMR win there.

## 2. The reframing (what is already true)

The "uniform finest band" is **policy, not discretization contract**. `buildGhostOverlay`
(`ghost_projection.hpp`) requires only that each non-clean row's ±2 closure reach is
same-level **as the row itself** — a cut cell at level L with a locally uniform level-L
neighborhood already gets a correct (2,2) closure at spacing h_L. The binary openness
operator, the openness-MG hierarchy (octree coarsening, area-averaged alpha), and the uf face
field (finest-sub-face-owned, conservative) all work per-level already.

So the problem decomposes:

- **Patches** (surface regions internally uniform at some level): essentially free.
- **Seams** (the codimension-2 curves where two patches of different level meet on the
  surface): the only place new numerics is needed — there, some cut cell's ±2 chain
  unavoidably crosses a 2:1 boundary and today throws.

The codimension argument (Johansen & Colella 1998: O(h) truncation on a codim-1 set still
yields O(h²) solution error) sets the accuracy budget: the seam set is codim-2 (O(N) cells vs
O(N²) surface cells), so locally first-order-consistent seam rows are asymptotically
invisible in integrated quantities — *provided they are stable and never O(1)-inconsistent*.

## 3. Campaign constraints every design element must satisfy

From flow's attractor campaign (non-negotiable, each has a measured counterexample):

- **C2 / support consistency** (invisible-subspace note, Prop. 2): the constraint must never
  couple pressure DOFs the gradient does not read. Seam fallbacks must stay **fluid-only** —
  never aperture rows with solid-centered DOFs (reopens the attractor family).
- **The pairing principle** (fluid_only_constraint_plan.md): the (cell gradient, constraint)
  pair must be structurally matched near the wall — both sides from the SAME directional
  closure family. Three measured mismatches marched unstable; the ghost's shared closures are
  the stable architecture. Seam machinery must feed matrix delta, divergence delta, and cell
  gradient from the same generalized samples.
- **(2,2) closures only** ((1,2) mixed is march-unstable above ~2000 spheres, flow hardening
  Phase A).
- **Stationarity batteries carry over**: dt=60..1e20 stationarity, dt-cycling reversibility,
  the C2 acceptance battery (tests/study/amr_zh_c2.py) must pass ON SEAMED MESHES — the
  campaign's instabilities (all of lagged-stiff-term shape) are exactly the kind of failure
  that hides in a "small local modification" like a seam row.

## 4. The design: generalized sample slots + fluid-only fallback cascade

### 4.1 Core mechanism (Option 1 of the brainstorm)

The closure (`scheme/ghost_closure.hpp`) is a pure 1-D function of samples along an axis;
today a sample = one same-level leaf value. Generalize each chain entry from a leaf index to
a **precomputed linear functional of nearby leaves** (a small per-slot CSR):

- **same level** → identity (weight 1; today's behavior, zero overhead on the common path);
- **finer region** → volume average of the 2^D covering children (the island-corner sampling
  arithmetic already in `cf_scheme.hpp`);
- **coarser region** → interpolation of the coarse field to a **virtual sample at the fine
  cell's uniform-grid position** (tangentially: the Martin–Cartwright `coarseStar` quadratic;
  normally: interpolation along the axis — order per Phase 0 measurement M2).

Data-structure change: `GhostOverlay.nbr` (`[n*15]` leaf indices) becomes
`nbrStart/nbrIdx/nbrW` CSR; `ghostApplyDeltaHost/ghostApplyDelta` and
`ghostDivergDeltaHost/ghostDivergDelta` change indexed loads into short weighted sums — same
kernel shape, device-trivial, oracle==device parity by construction (weights host-built,
shared verbatim). Rows whose 15 slots are all identity take the exact current fast path.

Literature pattern: least-squares/interpolated ghost fill from the composite valid-cell set
(chombo-discharge 2-ghost-layer LS reconstruction near EB×C/F; Cart3D wall-normal quadratic
reconstruction; Min–Gibou T-junction interpolation). See §9.

### 4.2 Fallback cascade (Option 3)

Where a generalized sample cannot be built robustly (gates of §5-D6 fail), the row degrades
**within the same fluid-only closure family**: QUAD → LIN → BC_ONLY/EXPLICIT — the existing
state cascade, never aperture rows (C2), never a mismatched pair (pairing). By the codim-2
argument a curve of degraded rows costs O(h²·h) in integrated drag; Phase 0 M1 measures how
often the cascade actually fires.

### 4.3 Per-closure-axis relaxation (Option 0′, part of the same change)

The current validity check demands all 3 axes × 5 chain entries same-level, including axes
whose faces are all COUPLED and solid-side entries the closure never reads. Relax to:
same-level (or generalizable-sample) required only along **closure axes** (axes carrying a
non-COUPLED face), fluid-side entries only. Where the level jump crosses the wall roughly
perpendicular to the surface, closure axes (≈ wall-normal) run parallel to the jump and many
seam rows need no interpolation at all. M1 quantifies this.

## 5. Decisions (chosen route + reviewable alternative)

Each decision records the alternative to revisit if the chosen route disappoints.

**D1 — Virtual samples at uniform positions (CHOSEN) vs generalized closure polynomials.**
Keep `gpFillRow` and the float closure polys verbatim; synthesize samples at the uniform
±1h, ±2h positions by interpolation. Preserves the shared flow↔AMR scheme header, float
bit-parity, and the pairing principle by construction.
*Alternative (more principled, more invasive):* re-derive the wall-anchored quadratic for
arbitrary node positions (non-uniform 1-D closure). Revisit if virtual-sample interpolation
error measurably dominates seam truncation (M2 would show quadratic samples still
insufficient), or if a reviewer/paper requires the direct non-uniform derivation.

**D2 — Classification by SDF re-sampling at virtual positions (CHOSEN) vs aggregation.**
The face-state cascade, momentum solid masks, and binary openness must keep agreeing at
seams. Chosen: evaluate the (float) SDF at the virtual uniform positions — the SDF is a
function, so the classification arithmetic (face = ½(center+center)) runs verbatim on
virtual values. One consistency contract: overlay classification == momentum masks ==
`makeBinaryOpenFn` at every row, with the openness probe distance the ROW-LOCAL h (verify:
the current `makeBinaryOpenFn(sdfFn, h0)` probes at the finest ±h0/2 — must become
level-aware alongside this work).
*Alternative:* Chombo-style aggregation of fine-geometry classifications up-level. Revisit
if float re-sampling produces classification flicker at seams (rows changing state between
adapt cycles) or overlay/openness disagreement.

**D3 — Keep per-level SDF openness sampling + hard connectivity floor (CHOSEN) vs
telescoping geometry.** The tagging floor (§7, gap ≥ n·h_L) is made a mesh-generator
INVARIANT so no coarse cut cell can pinch fluid connectivity that is open at fine level (the
AMReX multi-valued-cell rule inverted). Tripwire: a connectivity assertion comparing the
pocket-guard component structure of the leaf-level binary graph against a finest-level
reference graph on the band.
*Alternative (the literature-safe route):* full Chombo/AMReX telescoping — coarse openness =
aggregate of fine openness, exact by construction. Revisit if the tripwire ever fires, or if
MG convergence degrades on mixed-level cut bands (hierarchy/leaf geometry disagreement).

**D4 — Seam-minimizing level policy (CHOSEN shape).** L(x) is quantized with hysteresis and
a minimum patch size (patch diameter >= ~8–16 cells at the patch's level), so the policy
produces few, large, uniform patches — never a pointwise tracking of the gap field (which
would drape seams everywhere and multiply the fallback set). Seams should preferentially
land where the surface is smooth and the jump can cross the wall near-perpendicular
(maximizing the Option-0′ no-interpolation fraction).
*Alternative:* free pointwise L(x) with denser seam handling. Revisit only if the patch
constraint measurably wastes cells on real packings (M1 census reports the waste).

**D5 — Rung-1 scope (CHOSEN).** Stokes-only, single-rank, static mesh (no adapt-during-run),
WITH the momentum-diffusion closure at mixed-level cut cells and the wall-aware CfScheme
gates (§6.1) — both are accuracy-critical at seams, not optional extras. Deferred to later
rungs: advection/uf seam handling beyond what `buildCfUfDelta` already gives, MPI
(collective band-violation machinery already exists; halo must carry the wider sample
support), adaptivity-during-run (overlay/CSR rebuild with seams).
*Alternative:* none needed — this is sequencing, not design.

**D6 — Deterministic fallback gates (CHOSEN).** A row's sample construction and cascade
state must be a pure function of (geometry, octree) — independent of build order and adapt
history. Gates, mirroring cf_scheme's: a virtual sample is valid iff its support cells
exist, are fluid (never a decoupled/held solid value), and its tangential faces are
sufficiently open (openness >= 0.5); otherwise degrade per §4.2. Hysteresis lives ONLY in
the level policy (D4), never in the row classification.
*Alternative:* adaptive/iterative LS sample selection (chombo-discharge style, larger
candidate sets). Revisit if the deterministic gates degrade too many rows on real packings.

## 6. Supporting cast (changes that ride along)

### 6.1 CfScheme wall-aware gates
The quadratic C/F scheme's robustness gates assume level boundaries sit in smooth flow
(`cf_scheme.hpp`); at seams, tangential neighbours are solid and the gated faces revert to
the raw 1st-order flux exactly where accuracy is scarcest. Fix (already flagged in
amr_collocated_projection.md as the "natural next scheme refinement"): one-sided /
fine-side-interpolating tangential stencils drawn from the same directional closure family.
In scope for rung 1 (D5).

### 6.2 Momentum diffusion closure
Shares the 1-D wall-anchored closure — gets the same generalized samples (same overlay CSR
machinery, velocity geometry). In scope for rung 1.

### 6.3 uf / divergence face values at seams
Ghost-closure faces at seam rows must use the SAME face-value convention as
`buildCfUfDelta`'s distance-weighted + coarse* form, so constraint and advecting uf agree
(pairing, again). Rung-1 records the convention; the advection wiring is a later rung.

### 6.4 Pocket guard
`findPocketCells` runs on the actual mixed-level leaf graph (it already does — leaf-indexed);
add the D3 tripwire comparison against the finest-level band graph.

### 6.5 Distributed
`bandViolation` + collective fallback logic already exists. The generalized samples widen
halo support (coarseStar tangential neighbours, ±2 virtual-support cells) — LeafHalo must
carry them. Deferred rung.

## 7. The refinement policy for porous media

L(x) on the surface = union of criteria, then D4-quantized:

1. **Gap-width floor (geometric, HARD invariant per D3):** a cell at level L may be cut only
   if the local gap/pore width >= n·h_L (n from M2/M4; start n=4). Equivalent to the AMReX
   multi-valued rule: coarsening must never merge or disconnect fluid. Computed from the SDF
   pore-radius field (maximal inscribed ball / medial axis) — **or seeded from a coarse
   `peclet.pnm` extraction**, which already delivers throat locations and radii
   (PNM-informed AMR; the cheap and suite-native route).
2. **Curvature:** κ·h <= tol on the surface — contacts (diverging union-curvature) refine
   automatically, smooth caps coarsen (the Cart3D pattern).
3. **Solution adaptivity:** Löhner on velocity/dissipation (`indicators.hpp`) on top of the
   geometric floor; throats carry the dissipation, so equidistribution and the floor agree.
   Goal-oriented (adjoint on permeability) is a recorded stretch option, not planned.
4. **Closure-quality feedback:** refine where the overlay reports pathological
   thetas/sliver-heavy rows (self-diagnosing criterion; cheap since th is already stored).

## 8. Phases

### Phase 0 — a-priori measurements (may run BEFORE the Snellius gate; no solver commitment)

- **M1 — Option-0′ census.** On a real packing SDF with two candidate level maps: fraction of
  seam rows that pass the per-closure-axis relaxation with NO interpolation; fraction needing
  virtual samples; fraction hitting the fallback cascade. Decides whether the sample
  machinery is a rare path or the main path, and reports D4's cell waste.

  **DONE 2026-08-24** (`tests/study_amr_seam_census.cpp`, log `docs/data/amr_seam_census_m1.log`;
  50-sphere contact-rich periodic packing + 3 engineered throat pairs at 2/6/20 h0, finest
  1/256, background L3, gap floor n=4, D2 virtual-position classification, states from the
  real `gpFillRow` (2,2)):

  | map | cells vs uniform band | seam rows | of seam: axis-pass / samples / fallback |
  |---|---:|---:|---|
  | U uniform finest band | 100% | 0% | — (sanity: 100% strict-pass) |
  | G gap ladder {0,1,2,3}, pointwise | **13.9%** | 56.2% | 43.4% / 56.1% / **0.5%** |
  | Q two surface levels {0,2} + dilation | 73.0% | 16.2% | 16.7% / 83.0% / 0.3% |
  | hemisphere sphere, one clean seam | 64.1% | 3.3% | **99.8%** / 0% / 0.2% |

  Verdicts (they revise the plan):
  1. **The sample machinery is the MAIN path, not a rare path** — on the economically
     interesting ladder map, ~31% of all rows need virtual samples. D1 gets first-class
     design care; the Option-0′ relaxation is a worthwhile free fast path (24% of rows, and
     ~100% of a clean isolated seam) but not sufficient alone.
  2. **The fallback cascade is negligible** (0.1–0.5% of rows on every map): sample support
     almost always exists, so almost no rows degrade below the (2,2) closure. The "fallback
     fires too often" risk is retired by measurement.
  3. **D4 revised:** on contact-rich packings the intermediate levels carry the economics
     (most of the surface sits at intermediate gap scales), so their concentric seam rings
     around contacts are NOT removable by quantization — collapsing the ladder to two surface
     levels kept only 27% of the saving. Keep the full ladder, accept seams-as-main-path;
     quantization/hysteresis remains only for adapt-cycle stability, not seam avoidance.
  4. ~~Follow-up: rerun on a REAL dense bed~~ **DONE 2026-08-25** on the dem RCP bed
     (`tests/data/rcp_pack_seed3_unit.txt` = flow's `rcp_pack_seed3.npz`, 180 monodisperse
     spheres, φ=0.63; logs `amr_seam_census_m1_rcp{8,9}.log`):

     | R/h0 | map G cells vs uniform band | seam rows | of seam: axis-pass / samples / fallback |
     |---:|---:|---:|---|
     | 24.1 (depth 8) | 66.2% (1.5×) | 28.6% | 42.1% / 57.3% / 0.6% |
     | 48.2 (depth 9) | **34.5% (2.9×)** | 34.9% | 46.0% / 53.2% / 0.7% |

     **The real-bed economics are RESOLUTION-DEPENDENT:** at φ=0.63 most gaps are only a few
     fine cells at R/h≈24, so little of the surface may coarsen (1.5×); doubling resolution
     doubles every gap in cell units and the saving jumps to 2.9×, i.e. the win grows toward
     production resolutions (R/h≳96 extrapolates to ~5×; the dilute/flocculated φ≈13% case
     reached 7.2×). Verdicts 1–3 are unchanged on the real bed: samples are the main path
     (~19% of all rows), Option-0′ covers ~15% free, fallback stays negligible (0.3%).
     Consequence for Phase 3: quote the headline cell-count reduction AT production
     resolution, not at the smallest bed that fits.
- **M2 — Sample-order requirement.** Two-level hemisphere sphere (L caps / L+1 band, one
  clean seam): row truncation + drag error for linear vs quadratic virtual-sample
  interpolation (normal direction), coarseStar on/off (tangential). Decides the D1
  interpolation order and informs n in §7.

  **DONE 2026-08-25** (`tests/study_amr_seam_sample_order.cpp`, log
  `docs/data/amr_seam_sample_order_m2.log`). Two findings:

  *Geometry lesson first:* the planned hemisphere jump produces almost NO sample-needing rows
  (it is perpendicular to the wall everywhere — the census's 99.8% axis-pass). The study runs
  on a LATITUDE jump plane (z = C0z−0.15, oblique to the wall), which is also the realistic
  seam orientation for gap-graded maps.

  *The verdict (4/4 gates, ladder N=64..512, actual gpFillRow (2,2) weights, degree-d LS
  reconstruction from fluid leaf point values — coarseStar being the aligned degree-2 special
  case):* sample errors behave exactly as theory (e1 ~ H², e2 ~ H³); row-propagated
  perturbations:

  | | max @ N=512 | aggregate order (max / rms) | vs physical scale |
  |---|---:|---:|---|
  | matrix, LS1 | 62.5 | −0.43 / −0.10 (does NOT decay) | **53% of ∇²φ** |
  | matrix, LS2 | 0.51 | **0.83 / 0.97** | 0.43% of ∇²φ |
  | divergence, LS1 | 7.6e−3 | 0.25 / 0.58 | ~1% of row action |
  | divergence, LS2 | 2.4e−4 | 0.88 / **1.45** | 0.034% of row action |

  **D1 is settled: degree-2 (quadratic, cross terms included) LS reconstruction for virtual
  samples, in BOTH the matrix and divergence paths.** Linear samples are an O(1),
  non-decaying operator perturbation at seam rows (the Martin–Colella linear-C/F result
  reproduced on the actual closure); quadratic keeps seam rows consistent (~O(h), sub-percent
  constants) with the codim-2 argument then guaranteeing global accuracy. The degree-2
  fallback gate never fired on this geometry (clouds always sufficient). Drag-error
  confirmation stays a Phase-1/2 item (needs the solver on a seamed mesh).
- **M3 — Seam stability probe.** Minimal march on the seamed mesh, dt=60..1e20 + dt-cycling.
  A lagged-stiff-term instability found HERE changes the design (matrix-side seam
  treatment), not just the test section — this is the cheapest point to find it.

  **DONE 2026-08-25** — via the HOST-ORACLE PROTOTYPE of the D1 machinery
  (`ghost_projection_sampled.hpp` + `oracle::AmrFlow::setGhostSampled`; §8a below), probed by
  `tests/study_amr_seam_march.cpp` (phase-parallel, logs `docs/data/m3_*.log` + v1
  `amr_seam_march_m3_v1.log`). Latitude two-level sphere N=64 (1735 rows, 225 LS2 slots, 0
  degraded), CONTROL = the identical protocol on the uniform finest band:

  | probe | seamed | control (uniform band) | verdict |
  |---|---:|---:|---|
  | uniform-band parity (classic vs sampled) | — | rel **0.000e+00** | identity slots bit-exact |
  | march bounded, dt=60/600/1e20 | all | all | no blow-up at any dt |
  | statRes @ step 300, dt=60 | 2.70e-6 | 2.74e-6 | indistinguishable, same decay |
  | statRes @ step 200, dt=1e20 | 1.01e-5 | 9.15e-6 | indistinguishable, both decaying |
  | K dt-spread 60→1e20 (at this budget) | 9.0e-5 | 1.03e-4 | seam adds NOTHING |
  | dt-cycling 60→1e20→60 (300-step legs) | rel **0.000e+00** | rel **0.000e+00** | NO attractor family |
  | K offset vs uniform band | +0.31% at 43% of cells | — | accuracy plausible pre-Phase-2 |

  **Verdict: no seam-induced instability and no attractor family at prototype level** — the
  seamed march is statistically identical to the uniform-band control under the identical
  protocol (the v1 absolute-gate "failures" were shared convergence-budget floors, present
  bit-comparably in the control). The dt-cycling reversibility — the campaign's family
  discriminator — is exact to all printed digits on the seamed mesh. Residual risk moves to
  Phase 2 scale (bed geometry, per the flow campaign's lesson that families need beds).

### 8a. The host-oracle prototype (built 2026-08-25, pre-gate — doubles as the Phase-1 oracle)

`include/peclet/core/amr/ghost_projection_sampled.hpp` (host-only) + `flow_oracle.hpp`
`setGhostSampled(true)`: chain entries → sample-slot CSR functionals (identity at same level —
bit-exact on uniform bands; degree-2 LS virtual samples across 2:1 boundaries per M2;
fluid-only fallback), face classification FORCED to the canonical actual-center openness
(`makeBinaryOpenFnMixed`, any-sub-face-open for finer-across faces) so overlay-closed <=>
binary-closed (no double-counted flux), and the row functionals feed the directional gradient
(pairing). 51/51 host AMR ctests unchanged (opt-in path). KNOWN PROTOTYPE GAPS (Phase-1
scope): the momentum ξ-overlay still reads covering neighbours with same-level coefficients
at seam cut cells (silent O(1) coefficient error, §6.2); pocket cells are not excluded from
LS clouds; no device mirror; sub-face-resolved closures (the mixed-face Neumann-zero
degeneracy) unimplemented — counted by the builder (`nMixedFace`, 0 on the probe geometries).

### Phase 1 progress (started 2026-08-25 — the gate PASSED: ghost is the production default
### suite-wide, flow c672014 + core 7472306)

**Rung 1 — momentum ξ-row seam correction (§6.2): DONE** (`buildMomSeamDelta` in
`ghost_projection_sampled.hpp`, applied as a lagged deferred-correction CSR in the oracle's
predictor — the cfMom_ pattern; assumes u_bc=0). The raw AmrCutCell seam rows were wrong
TWICE: covering-center classification/anchoring AND the global β = μ/h0² (dimensionally off
4^ΔL at coarser rows). Measured (`study_amr_seam_sample_order` §[B], rescaled-row truncation
on the wall-consistent f = sdf·φ, latitude mesh N=64/128/256):

| N | raw RMS | corrected (exact) RMS | corrected (LS2) RMS |
|---:|---:|---:|---:|
| 64 | 25.2 | 0.346 | 2.52 |
| 128 | 46.2 | 0.402 | 1.41 |
| 256 | 90.9 | **0.464** | **0.846** |

raw DIVERGES O(1/h); corrected is the scheme's normal bounded closure truncation (~200×
below raw at N=256), LS2 samples converge onto the exact floor. March acceptance: parity
still exact, dt-cycling with the new lagged term still rel 0.000e+00, dt=1e20 bounded, same
stationarity. K(seam)−K(ctrl) at N=64 moved +0.31%→+0.51% (cf=0) / −0.72% (cf=1) — single-N
offsets against the (biased) uniform control are a weak instrument; the accuracy verdict is
Phase 2's ladder. Out of rung-1 scope (counted, logged): raw-regular-but-virtually-ghost
rows; u_bc ≠ 0.

**Rung 2 — wall-aware CfScheme gates (§6.1): DONE** (`cfAppendStencil`: when exactly one
tangential side survives the fluid/openness gates — the near-wall seam case — substitute the
one-sided LINEAR interpolation dt·u′ instead of falling back to the raw coarse value: O(H²)
tangential sample error vs the raw fallback's O(H) offset). Two-sided path bit-identical
(51/51 host AMR ctests unchanged, incl. the pinned cf-quadratic gates — the finest-band
contract keeps them inert there).

**Rung 3 — device mirror: DONE 2026-08-26.** `AmrFlow::setGhostSampled` (device, single-rank
— the distributed sample halo is a later rung): sampled overlay host-built + uploaded
(`GhostOverlaySampledDev`, sample-slot CSR kernels `ghostApplyDeltaSampled` /
`ghostDivergDeltaSampled`), level-aware canonical openness (`makeBinaryOpenFnMixed`) on the
unchanged MG rails, CSR directional-gradient overlay (`GhostGradCsrDev` — cascade over sample
functionals on overlay rows, classic same-level fallback on rowless cut cells: the oracle's
exact gradP split), momentum seam CSR re-folded per row to post-rscale (the device cfApply
convention). Every delta site calls both classic and sampled appliers (empties no-op) — the
classic path is untouched. Parity ctest `test_seam_sampled`: (a) uniform band, device sampled
vs device classic **bit-identical** (max diff 0.00e+00, CUDA); (b) two-level latitude band,
device vs oracle rel 2.4e-06 mean / 3e-6 fields. Full flow-solver battery green on CUDA.

**Rung 4 — graded refineToSdf policy API (§7): DONE 2026-08-26** (core `83e9dff`). The
mesh-generator side: `refineToSdfGraded(t, geo, sdf, targetFn, band, balance)` takes a per-point
TARGET LEVEL instead of one global `targetLevel`, with the band margin measured in cells of the
level being CREATED (not in h0 — that is what keeps the margin meaningful where the target is
coarse), and the band predicate evaluated BEFORE `targetFn` (the gap/medial-axis field is the
expensive one). `gapFloorTarget(gapFn, h0, coarsestLevel, n = 4)` packages §7 criterion 1 — the
coarsest level clearing `gap >= n·h_L`. Python: `Octree.refine_to_sdf_graded` /
`refine_to_gap_floor`, plus `Flow.set_ghost_sampled` (the sampled overlay had no binding at all)
and `Flow.set_dt` (whose docstring carries the stale-momentum-operator trap: dt is baked in at
build time, so a dt switch is read → `set_dt` → `set_solid` → `set_velocity`/`set_pressure`).
`test_amr_sdf::runGraded` covers it; 51/51 host AMR ctests green.

Remaining Phase-1 rungs: pocket exclusion in LS clouds, sub-face closures, distributed sample
halo.

### Phase 1 — overlay generalization (rung 1, post-gate)
Sample-slot CSR in `GhostOverlay` + builders (D1/D2/D6), per-closure-axis validity, delta
kernels, momentum closure samples, CfScheme wall-aware gates, level-aware openness probe.
Oracle first, device mirror second (parity by shared host-built weights).
**Acceptance:** uniform meshes bit-identical to today (all-identity fast path); two-level
sphere (M2 geometry) drag within the uniform-ghost band's own spread; 86/86 existing AMR
ctests green CUDA+OpenMP; new seam parity ctest oracle==device.

### Phase 2 — stability + accuracy batteries on seamed meshes
amr_zh_c2 battery + dt=1e20 stationarity + dt-cycling on the seamed sphere; Z&H ladder with
a graded surface (caps L, sector band L+1): clean-ladder convergence and the ~+0.2% ghost
bias must survive seams. **Acceptance:** family-free (stationarity to the C2 thresholds),
seam contribution to K consistent with the codim-2 estimate.

#### P2a — the C2 acceptance battery on a seamed mesh at scale (DONE 2026-08-27)

`tests/study/amr_zh_c2_seam.py` (log `docs/data/amr_zh_c2_seam_n128.log`), CUDA/RTX 5080. The
ladder's arm (b) — the two-level latitude map, whose jump plane is OBLIQUE to the wall and so
actually produces LS2 sample rows (M2's geometry lesson) — at **N = 128**, i.e. at scale rather
than M3's N = 64 prototype, with the uniform-band control run under the identical protocol.
`set_ghost_sampled(True)`, (2,2), cf-quadratic. The mixed-level band adds two lagged terms (the
momentum ξ-row seam correction and the wall-aware C/F tangential fallback), and every instability
the campaign found is of lagged-stiff-term shape, so this is the gate that says the seam
machinery has not re-opened the attractor family.

| probe | arm (b) seamed | arm (a) uniform control |
|---|---:|---:|
| K at dt = 60 (statRes) | 4.271444175 (1.03e-09) | 4.260989442 (9.89e-10) |
| K at dt = 600 (statRes) | 4.271444172 (1.38e-08) | 4.260989445 (8.13e-09) |
| K at dt = 1e20 (statRes) | 4.271444164 (3.04e-08) | 4.260989445 (8.99e-09) |
| **dt-spread 60→1e20** | **2.46e-09 (PASS at 1e-5)** | **7.26e-10 (PASS)** |
| dt-cycling 60→1e20→60, K | 1.42e-08 rel | 1.37e-08 rel |
| dt-cycling, velocity field | 1.57e-08 rel | 1.51e-08 rel |

**Verdict: family-free on a seamed mesh at N = 128.** Every probe on the seamed mesh matches the
control to the same order of magnitude, and the cycling residual is essentially IDENTICAL
(1.42e-08 vs 1.37e-08) — i.e. it is the shared convergence-budget floor of the two marches, not a
seam effect. dt-independence holds at 2.5e-09, three and a half decades inside the 1e-5 gate.

*Protocol note (worth not re-deriving).* Each cycling leg must march to STATIONARITY, not for a
fixed number of steps. On the device, K-stationarity at N = 128 needs ~2800 steps; a first pass
with M3's fixed 100-step legs compared a 100-step transient against a 233-step transient and read
7.8e-02 — a pure protocol artefact, not a family. Also: the device `set_solid` reallocates and
zeroes u and p (the host oracle's does not), so the dt switch has to save and restore the state —
hence the new `set_dt` / `set_velocity` / `set_pressure` round trip.

#### P2b — the graded accuracy ladder on device (DONE 2026-08-27)

Single-N K offsets against the uniform control are a weak instrument (see "Phase 1 progress"):
the control carries its own bias, so an offset of either sign is uninterpretable. Convergence
WITH RESOLUTION is the honest one — a seam treatment that is *stable but inconsistent* shows up
as an offset that stops shrinking.

`tests/study/amr_zh_ladder.py` (CUDA `python/build_cuda2`, RTX 5080; logs
`docs/data/amr_zh_ladder_n{64,128,256}.log`). Z&H simple-cubic sphere, φ = 0.125, K_ZH = 4.292;
`set_ghost_sampled(True)`, (2,2) closures, dt = 60, K-stationarity to 1e-8 relative, K from the
VOLUME-weighted (superficial) mean. Three arms differ ONLY in the surface level map — background
uniform at LFAR = 3 levels above finest and band margin 4 cells of the level being created, so
the mesh family is self-similar in N (every arm refines toward the exact answer rather than to a
graded plateau) and the far field is identical across arms:

- **(a) uniform** — finest band over the whole surface (today's policy; the control).
- **(b) two-level** — latitude jump at z = c − R/2, OBLIQUE to the wall (M2's lesson: a
  hemisphere jump is ⟂ to the wall and produces no sample rows at all), cap above at L1.
- **(c) gap-ordered three-level** — L0/L1/L2 ordered by the §7 gap proxy (two-closest-surfaces
  d1 + d2 over the 27 nearest periodic images): finest at the narrow pole directions, coarsest at
  the wide corner directions, thresholds splitting the geometry's own gap range in three.

| cf | arm | N=64 K (err%) | N=128 K (err%) | N=256 K (err%) | cells% of N³ (64/128/256) |
|---:|---|---:|---:|---:|---|
| 1 | a uniform | 4.250879 (−0.958) | 4.260990 (−0.723) | 4.277542 (−0.337) | 26.2 / 12.8 / 6.4 |
| 1 | b two-level | 4.284915 (−0.165) | 4.271445 (−0.479) | 4.280699 (−0.263) | 11.7 / 5.7 / 2.9 |
| 1 | c gap-ordered | 4.188793 (−2.405) | 4.251682 (−0.939) | 4.273262 (−0.437) | 12.3 / 6.4 / 3.3 |
| 0 | a uniform | 4.533042 (+5.616) | 4.570203 (+6.482) | 4.476080 (+4.289) | 26.2 / 12.8 / 6.4 |
| 0 | b two-level | 4.507181 (+5.014) | 4.547695 (+5.958) | 4.458776 (+3.886) | 11.7 / 5.7 / 2.9 |
| 0 | c gap-ordered | 4.550496 (+6.023) | 4.558979 (+6.220) | 4.454879 (+3.795) | 12.3 / 6.4 / 3.3 |

**The seam instrument** — |K(arm) − K(a)|, which cancels the shared far field:

| | N=64 | N=128 | N=256 | order 64→128 | order 128→256 | over the full range |
|---|---:|---:|---:|---:|---:|---:|
| cf=1, b − a | 3.404e-02 | 1.045e-02 | 3.157e-03 | **1.70** | **1.73** | **1.72** |
| cf=1, c − a | 6.209e-02 | 9.308e-03 | 4.280e-03 | 2.74 | 1.12 | **1.93** |
| cf=0, b − a | 2.586e-02 | 2.251e-02 | 1.730e-02 | 0.20 | 0.38 | 0.29 |
| cf=0, c − a | 1.745e-02 | 1.122e-02 | 2.120e-02 | 0.64 | **−0.92** | −0.14 |

**Verdict: the seam machinery is consistent — seams do not set the order.** With the quadratic
C/F scheme the seam offset decays at 1.72 (arm b, two clean consecutive ratios of 1.70 and 1.73)
and 1.93 (arm c, over the full range), and lands at 0.074% / 0.100% of K at N = 256 — inside the
ghost scheme's own ~0.2–0.3% bias. The reference point that makes this readable: the ARMS
themselves converge at ~1.1 order on the last doubling (arm a: 0.0411 → 0.0310 → 0.0145, orders
0.41 and 1.10), because the graded family's leading error is the far-field C/F flux, not the cut
cells — and arm (a) has ZERO sample slots, so that sub-2 order is demonstrably NOT seam-caused.
The seam offset therefore shrinks at least as fast as the discretization error it rides on, which
is exactly the codim-2 prediction: locally first-order-consistent seam rows on an O(N) set are
asymptotically invisible in an integrated quantity.

**Both C/F schemes had to be run, and the standard one is uninformative here** (as anticipated —
log `docs/data/amr_zh_ladder_n256_cf0.log`): with the 1st-order two-point flux the offsets do not
converge at all — they wander (arm b order 0.20 then 0.38; arm c 0.64 then **−0.92**, i.e. the
offset GROWS on the last doubling) — and the control arm's own error is non-monotone
(+5.62% → +6.48% → +4.29%). That is the level-boundary flux, not seam numerics: arms b and c
introduce more C/F faces near the surface than arm a does, so at cf=0 the "offset" mostly measures
C/F count. Nothing about seams can be read off the cf=0 column; the verdict rides on cf=1. (This
also re-confirms, on a third geometry family, the graded-mesh finding already in
`amr_collocated_projection.md`: the quadratic C/F scheme is not optional on graded meshes.)

**Overlay census across the ladder** (rows | LS2 sample slots | degraded | closed mixed faces):

| arm | N=64 | N=128 | N=256 |
|---|---|---|---|
| a | 4208 \| 0 \| 0 \| 0 | 16776 \| 0 \| 0 \| 0 | 66432 \| 0 \| 0 \| 0 |
| b | 1856 \| 232 \| 0 \| 0 | 7284 \| 444 \| 0 \| 0 | 28776 \| 872 \| 0 \| 0 |
| c | 2304 \| 2136 \| 0 \| 96 | 9288 \| 5664 \| 0 \| 0 | 36608 \| 9072 \| 0 \| 240 |

Three things the census settles. (i) **The codim-2 premise holds on real meshes**: LS2 slots grow
~linearly in N (arm b 232 → 444 → 872) while rows grow ~N² — the seam set really is codimension 2.
(ii) **The fallback cascade never fires** (0 degraded at every N on every arm), confirming M1's
retirement of that risk on solver meshes and not just in the census. (iii) Arm (a) reports
ALL-IDENTITY slots, i.e. the control is the bit-parity path — verified directly at ladder scale
by `tests/study/amr_zh_ladder_parity.py` (N = 128, cf = 1, 200 steps: max|sampled − classic| =
**0.000e+00** in u, v, w AND p; log `docs/data/amr_zh_ladder_parity_n128.log`).

Caveat carried forward: arm (c) still hits the unimplemented **sub-face-resolved closures**
(96 closed mixed faces at N = 64, 240 at N = 256; §8a's known gap) — geometry-dependent, not
resolution-vanishing. It is the natural attribution for arm (c) being the noisier of the two
seam instruments (orders 2.74 then 1.12 vs arm b's 1.70/1.73). Arm (b) has zero mixed faces at
every N, and it is arm (b) that gives the clean pair of ratios.

**Policy measurement, and why the porous payoff is Phase 3.** Run `amr_zh_ladder.py --gapfloor`:
the ABSOLUTE gap floor of §7 criterion 1 (gap ≥ 4·h_L) is **inert on a dilute SC array**. The
surface gap there is 0.38–0.55 of the box at every N, so the floor permits the background level
everywhere and the criterion asks for no surface refinement at all (N = 128 and 256: the mesh
comes back as the plain LFAR background, 4096 and 32768 leaves). The gap floor is a THROAT
criterion and Z&H has no throats — its gap contrast is only 1.44×, which is less than one level.
That is why arm (c) uses the gap ORDERING with thresholds normalised to the geometry's own range:
its job in this ladder is to stress many seams and a two-level jump, not to demonstrate cell
savings. The headline cell-count reduction belongs to Phase 3's two-sphere gap case and the bed
(where M1 measured 2.9× at R/h = 48, growing with resolution). For the record, the AMR economics
that DO show up here: arm (b) runs at 2.24× fewer cells than the uniform band at every N
(6.4% → 2.9% of N³ at N = 256) for a 0.074% shift in K.

### Phase 3 — the porous payoff
Two-sphere gap unit case (throat criterion vs gap resolution n); then a mid-size sphere bed
with pnm-seeded L(x): permeability + cell count vs uniform-finest-band. **The headline
number** = accuracy-matched cell-count reduction.

#### P3a — two-sphere gap unit case: n calibrated, and a BLOCKING divergence (2026-08-27)

`tests/study/amr_two_sphere_gap.py` (log `docs/data/amr_two_sphere_gap_n128.log`), N = 128,
cf-quadratic, `set_ghost_sampled(True)`, (2,2), dt = 60, tol 1e-7 (the control moves by 1.2e-06
between tol 1e-8 and 1e-7, four decades below the ~1e-3 effects measured here). Geometry: a
periodic chain of two equal spheres on z sized so BOTH z-gaps equal the throat width g
(R = (N−2g)/4), Stokes flow along x, so the throat is a channel carrying flux while the open
x/y directions sit at gap ≈ N−2R — several levels of contrast, which is exactly what Z&H lacked.
Instrument: Darcy k = μ⟨u_x⟩/f against the uniform finest band on the same mesh family.

| g=8 (R=28), ctl k = 6.590793e+02, 262480 leaves | | | | | |
|---|---:|---:|---:|---:|---|
| n | cells across throat (level) | k | vs ctl | leaves | vs ctl |
| 1 | 1 (L3) | 6.294569e+02 | −4.494% | 4096 | 64.1× |
| 2 | 2 (L2) | 6.353119e+02 | −3.606% | 5216 | 50.3× |
| **3** | **4 (L1)** | — | **DIVERGED @ step 100** | 9472 | 27.7× |
| **4** | **4 (L1)** | — | **DIVERGED @ step 170** | 14512 | 18.1× |
| 6 | 8 (L0) | 6.567198e+02 | −0.358% | 43632 | 6.02× |
| 8 | 8 (L0) | 6.582720e+02 | −0.123% | 76560 | 3.43× |

| g=16 (R=24), ctl k = 9.313106e+02, 200432 leaves | | | | | |
|---|---:|---:|---:|---:|---|
| 1 | 2 (L3) | 8.462643e+02 | −9.132% | 4096 | 48.9× |
| 2 | 2 (L3, same mesh) | 8.462643e+02 | −9.132% | 4096 | 48.9× |
| 3 | 4 (L2) | 9.192287e+02 | −1.297% | 5216 | 38.4× |
| 4 | 4 (L2) | 9.334591e+02 | +0.231% | 6784 | 29.5× |
| 6 | 8 (L1) | 9.242515e+02 | −0.758% | 17536 | 11.4× |
| 8 | 8 (L1) | 9.247510e+02 | −0.704% | 30192 | 6.64× |

**n calibration (from the stable rows).** Read on the physical axis — cells across the throat,
which the floor pins to [n, 2n) by construction — the policy error is −4.5 to −9% at 1–2 cells,
~1% at 4 cells, and sub-1% at 8. **The plan's default n = 4 is supported**: it buys 4–8 cells
across the throat and 0.1–1.3% at 18–30× fewer cells than the uniform band. Two honesty
caveats: the sub-1% rows are NON-MONOTONE (g=16 n=4 reads +0.23% while the finer n=6/n=8 read
−0.7%), so this single geometry supports "n = 4 is a sound floor" and nothing sharper; and the
`Lthroat` column is the level at the throat CENTRE, which for a wide gap is a bulk fluid cell
rather than a cut cell (that is why the g=16 control reads L1).

**THE BLOCKER: two of the twelve graded meshes march to a hard blow-up** — k reaches ~1e12 by
step 100/170, then NaN. Not a resolution effect: 1, 2 and 8 cells across the throat are all
stable and only 4-cells-at-L1 fails, while 4-cells-at-L2 (g=16, n=3/4) is fine.

**Attribution (`tests/study/amr_two_sphere_diverge_probe.py`, log
`docs/data/amr_two_sphere_diverge_probe.log`)** — a bisection over existing public knobs, no
numerics touched:

| n | cf | dt | momentum MG | outcome |
|---:|---:|---:|---|---|
| 3 | 1 | 60 | on / off | DIVERGED @ step 97 (both) |
| 3 | 1 | 1e20 | on / off | DIVERGED @ step 201 (both) |
| 3 | 0 | 60 / 1e20 | on / off | survived, k = 5.93e+02 / 6.44e+02 |
| 4 | 1 | 60 | on / off | DIVERGED @ step 164 (both) |
| 4 | 1 | 1e20 | on / off | "survived" at k = **−2.03e+03** (negative — diverging, slower) |
| 4 | 0 | 60 / 1e20 | on / off | survived, k = 5.91e+02 / 6.40e+02 |

**Every failure carries cf = 1; every cf = 0 run survives. The momentum MG is irrelevant
(bit-identical outcomes on and off). dt moves the timing (step 97 → 201), the lagged-stiff-term
signature.** The first written attribution here blamed Phase-1 rung 2 (the only NEW cf=1 code);
**P3b below refutes that and finds the real carrier** — the section is kept as the record of the
coarse bisection, its conclusion superseded.

#### P3b — the divergence diagnosed and fixed: cfDiv at cut rows, not rung 2 (2026-08-27)

**Rung 2 is exonerated.** Rebuilding with the one-sided branch (a) fully disabled (raw fallback,
pre-rung-2) and (b) restricted to convex interpolation only: the blow-up is bit-comparably
unchanged (n=3 diverges at step 97/96/99, n=4 at 164/140/163 for current/disabled/convex). The
carrier is in the PRE-EXISTING cf=1 machinery, which on mixed-level bands fires at cut-band C/F
faces for the first time.

**Second bisection — per application path** (diag-mask build, disabling one cf delta at a time
on the g=8, n=3 mesh): disabling `cfMom_` (lagged momentum delta), `cfGrad_` (pressure
gradients), or `cfUf_` (advecting face field) changes nothing — divergence at step 97/98/97.
**Disabling `cfDiv_` — the C/F divergence-constraint delta — alone restores a stable march.**
That also rules out a cfDiv/cfGrad pairing break (removing either member of a broken pair would
misbehave; only cfDiv is toxic). The discriminating variable is WHERE cfDiv fires: its row gate
was `rowFluid` (all fluid rows, cut included), while `cfMom_` was already gated to
`rowRegular` — under the wiring comment "cut rows are finest-band: no C/F faces", an assumption
the mixed-level band killed.

**Mechanism (reasoned, consistent with every measurement, not independently proven).** In ghost
mode a cut row's divergence constraint and pressure gradient are OWNED by the closure overlay
(sampled mode: `ghostDivergDeltaSampled` / `gcS_`), whose virtual samples already handle the 2:1
level crossing inside the closure. `cfDiv_` at such a row adds the smooth-field coarse*
substitution on top: extra velocity reads in the constraint that the row's gradient never sees —
a support-consistency (C2) violation of exactly the campaign's kind. On the Z&H latitude seam
those rows exist too but the march merely tolerated them; in a 4-cell throat at L1 the loop
gains exceed one and the lagged iteration runs away.

**The fix (one line per file, applied):** `cfDiv_` and `cfGrad_` row gates changed
`rowFluid → rowRegular` in `flow.hpp` and `flow_oracle.hpp` — cut rows belong to the closure
family, C/F substitutions to the smooth-field family; the same ownership split `cfMom_` always
had. On finest-band meshes cut rows have no C/F faces, so the gate is **inert by geometry**:
measured bit-identical (old vs fixed module, uniform-band arm, 200 steps: K, Σ|u|, Σ|p| equal
to all 15 printed digits). 51/51 host AMR ctests green. Both blocked meshes (g=8, n=3/4) march
stably under the gate with identical k between gating cfDiv alone and cfDiv+cfGrad (the cfGrad
part of the gate is inert here too — kept for symmetry of ownership).

**Acceptance under the fix** (build_cuda3; the pre-fix module kept in build_cuda2 as the A/B
reference):

- *Formerly blocked rows, marched to tol 1e-7* (g=8, vs control 6.590793e+02): n=3 →
  k = 6.490584e+02 (**−1.52%**), n=4 → k = 6.548991e+02 (**−0.63%**). The neighbours shift only
  slightly where the gate touches their seam-adjacent cut rows (n=2: −3.61% → −3.69%; n=6:
  −0.36% → −0.70%) — the removed cut-row C/F correction costs a little accuracy there, the
  price of the stability. The calibration table is now clean and monotone on the
  cells-across-the-throat axis: 1 cell −4.5%, 2 cells −3.7%, 4 cells −0.6…−1.5%, 8 cells −0.7%.
  **The n = 4 verdict strengthens**: −0.63% at 18× fewer cells on the geometry that used to
  blow up.
- *P2b ladder, seamed arm (b), cf = 1, tol 1e-8*: N=64 K = 4.2857391 (−0.146%), N=128
  K = 4.2715797 (−0.476%), N=256 K = 4.2807445 (−0.262%). Seam offsets against the recorded
  uniform control (valid — the control is bit-identical under the gate): 3.486e-2 → 1.059e-2 →
  3.203e-3, **orders 1.72 and 1.73** (pre-fix: 1.70 / 1.73). The Phase-2 accuracy verdict
  survives the fix unchanged.
- 51/51 host AMR ctests and 86/86 CUDA ctests green (incl. `test_seam_sampled` device==oracle
  parity — both sides carry the gate). Full bisection + acceptance log:
  `docs/data/amr_cfdiv_cutrow_fix.log`.

The bisection toggles were diagnostic-only and are not committed; the committed change is the
two row gates plus this record.

#### P3c — the sphere bed: policy validated end-to-end, headline capped by the local GPU (2026-08-28)

`tests/study/amr_bed_graded.py` on the M1 census bed (`tests/data/rcp_pack_seed3_unit.txt`, 180
monodisperse spheres, φ = 0.63), Stokes along x, Darcy k = μ⟨u_x⟩/f, gap floor at the
P3a-calibrated n = 4, `set_ghost_sampled(True)`, cf-quadratic. Logs
`docs/data/amr_bed_graded_d{7,8}*.log`.

**Mesh economics (`--probe`, CPU only) — M1's resolution dependence confirmed on solver meshes:**

| depth | R/h₀ | uniform finest band | gap-graded (n=4) | saving |
|---:|---:|---:|---:|---:|
| 7 (N=128) | 12.1 | 1 923 587 | 1 794 535 | 1.07× |
| 8 (N=256) | 24.1 | 11 350 032 | 7 010 669 | **1.62×** |

Depth 7 is useless as a headline — the band already covers 92% of the domain, so there is almost
nothing to coarsen. This is exactly M1's "quote the saving at production resolution" warning,
now measured on the meshes the solver actually runs.

**Accuracy (depth 7, both arms marched to tol 1e-7):**

| arm | k | leaves | steps |
|---|---:|---:|---:|
| uniform finest band | 5.840443e-01 | 1 923 587 | 560 |
| gap-graded (n=4) | 5.855410e-01 | 1 794 535 | 570 |

**The gap-graded policy reproduces the bed permeability to +0.256%** — comfortably inside the
ghost scheme's own bias, on a real contact-rich packing rather than a model geometry. Both arms
converge in ~560 steps (dense beds relax much faster than the Z&H sphere's ~9000). The march is
stable, which is the risk register's third witness geometry and the first that is neither
Z&H-like nor two-sphere-like.

**Bed-scale overlay census** — the numbers that settle two open rungs:

| arm | rows | LS2 slots | degraded | closed mixed faces | pockets |
|---|---:|---:|---:|---:|---|
| depth 7, uniform | 232 227 | 0 | 0 | 0 | 410 cells / 376 comps |
| depth 7, graded | 229 485 | 31 095 | **0** | 453 | — |
| depth 8, graded | 691 813 | 429 694 | **0** | 10 846 | 968 cells / 826 comps |

1. **The fallback cascade never fires on a real bed either** — 0 degraded rows at every size,
   up to 692k rows. M1's retirement of that risk now holds on solver meshes at bed scale.
2. **Sub-face closures are real but small**: 453/229 485 = 0.20% of rows at depth 7 and
   10 846/691 813 = 1.57% at depth 8 (vs 0.66% on Z&H's gap-ordered arm). Non-vanishing and
   growing with resolution, but the depth-7 march carrying 453 of them still lands within
   0.26%. So the rung is a real accuracy item, NOT a stability blocker.
3. **Pockets are plentiful on a real bed** (826 components at depth 8), which makes "pocket
   exclusion in LS clouds" a concrete rather than theoretical rung.

**Why the depth-8 headline is not here: two hard limits of the local RTX 5080 (16 GB, display
attached), neither a code defect.**
- *Memory.* The uniform control (11.35M leaves) dies in `setSolid` — `Kokkos ERROR: Cuda memory
  space failed to allocate 531 MiB (label="df_uf")`. The graded arm (7.01M) fits at 13.3 GB of
  16.3 GB, i.e. ~1.9 kB/leaf, which puts the wall at ≈8.5M leaves — the control needs ≈21.5 GB.
- *Kernel watchdog.* The depth-8 graded arm then died mid-march with
  `cudaErrorLaunchTimeout`: at 7M leaves a single kernel exceeds the driver's launch limit.

So **the accuracy-matched headline at production resolution needs a compute-mode GPU** (no
watchdog, ≥40 GB — Snellius). Everything needed to run it is in place; only the hardware is
short. Depth 9 (R/h₀ = 48, where M1 measured 2.9×) is firmly cluster work.

**Study-side lesson worth not re-deriving:** `set_solid` samples the SDF tens of times per leaf
(operator build, overlay classification at virtual positions, openness probe), so a *Python*
SDF callback dominates everything at bed scale — the first depth-8 attempt spent >1h43m inside
numpy before the GPU ran one kernel (confirmed by native backtraces). `Flow.set_solid_spheres`
(core 29ba5b0) evaluates the union natively: 311s → 33s on a 262k-leaf slice, fields
bit-identical to 15 digits.

### Phase 4 — deferred integration
Advection/uf seams, MPI (halo support + collective fallback), adapt-during-run rebuild.

## 9. Literature anchors (full scan in the 2026-08-24 session record)

- Trebotich & Graves 2015 (CAMCoS 10:43) — pore-scale incompressible NS, EB on every level,
  coarse geometry by aggregation, Martin–Colella quadratic C/F ghosts, widened
  redistribution for porous cusps. The closest prior art.
- Johansen & Colella 1998 (JCP 147:60) — the codimension/potential-theory accuracy argument
  (§2).
- Barrio Sanchez et al. 2023 (arXiv:2309.06372, JCP) — AMReX re-redistribution: EB crossing
  level jumps needs explicit inter-level bookkeeping for any neighborhood-based small-cell
  mechanism.
- AMReX EB docs — the multi-valued-cell coarsening constraint = our gap-width floor,
  inverted. MFIX-Exa docs — level-set resolution decoupled from fluid-grid resolution.
- Marskar 2019 (chombo-discharge, JCP 388) — LS valid-cell-only ghost reconstruction where
  C/F stencils touch the EB; 2 ghost layers near EB×C/F. The mechanical template for D1.
- Berger & Aftosmis (Cart3D) — decades of production cut cells at adaptation-chosen levels;
  curvature/adjoint wall-level assignment; wall-normal quadratic reconstruction.
- Min & Gibou 2006 (JCP 219) — 2nd-order solution AND gradients on non-graded trees via
  compact T-junction interpolation (the node-based existence proof).
- Basilisk embed.h (Popinet) — boundary at any level; the documented weak spot is the
  fraction restriction/prolongation near the embed (anti-pattern for D2/D3).
- Losasso, Gibou, Fedkiw 2004 — symmetric-but-O(h) T-junctions: compatibility buys
  robustness when pointwise order is sacrificed (context for the fallback cascade).
- Martin, Colella, Graves 2008 (JCP 227) — quadratic C/F interpolation, refluxing, sync
  projections, freestream preservation (the C/F backbone; cf_scheme.hpp is its collocated
  echo).

## 10. Risk register

| risk | detection | response |
|---|---|---|
| seam-row march instability (lagged-stiff-term shape) | M3, Phase-2 batteries, P3a/P3b | **FIRED once, diagnosed, FIXED 2026-08-27 (P3b).** History: retired on P2a's Z&H evidence; un-retired hours later when P3a found the throat blow-up (first blamed on rung 2 — refuted); P3b bisected it to `cfDiv_` firing at CUT rows (a C2 violation: constraint reads the row's gradient never sees) and fixed it with the `rowRegular` gate. Lesson standing: **Z&H-class geometries cannot retire throat risks** — every future retirement of this row needs a throat battery. The fixed configuration is stable on both formerly-blocked meshes; residual exposure = geometries neither Z&H-like nor two-sphere-like (the Phase-3 bed run is the next witness) |
| classification flicker at seams | D2 flicker check across adapt cycles | D2-alternative (aggregation) |
| coarse openness pinches a throat | D3 tripwire assertion | D3-alternative (telescoping geometry) |
| fallback cascade fires too often (M1 >~ 30% of seam rows) | M1 census | RETIRED 2026-08-24: measured 0.1–0.5% on all maps |
| accuracy loss beyond codim-2 estimate | Phase-2 ladder | RETIRED 2026-08-27 (P2b: seam offset decays at 1.72 / 1.93 over N=64→256, faster than the arms' own ~1.1 order; 0.07–0.10% of K at N=256). Reopen only if a bed ladder disagrees; levers stay raise sample order (M2) / widen seam collars |
| sub-face closures (mixed-face Neumann-zero degeneracy) | overlay `nMixedFace` counter | NOT retired, but BOUNDED (P3c): 0.20% of rows on the depth-7 bed and 1.57% at depth 8 (Z&H gap-ordered arm: 0.66%) — non-vanishing and growing with resolution, yet a bed march carrying 453 of them lands within 0.26% of the uniform control. So it is an ACCURACY item, not a stability blocker; still an unimplemented rung (§8a) and a design fork |
| Snellius ladder disappoints | the gate | plan survives in outline; re-target the closure family per fluid_only_constraint_plan.md decision tree |

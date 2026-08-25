# Mixed-level cut cells: local surface refinement for the AMR ghost projection

*Plan, 2026-08-24 (F. Peters / Claude session). Companion to `amr_collocated_projection.md`
(the AMR ghost port + graded-mesh record) and flow's `collocated_invisible_subspace.md` /
`fluid_only_constraint_plan.md` (the campaign verdicts that constrain any design here).*

**Status: PLANNED — implementation gated on the Snellius verdict** (flow's clean-ladder
R=24/32 order result + the np>=16 instability re-diagnosis). If the ladder confirms the ghost
projection as the production collocated scheme, this plan executes on `dev/ghost-production`;
if it disappoints, the seam design below survives in outline but the closure family it
generalizes must be revisited first. Phase 0 (a-priori measurements, no solver commitment)
may run before the gate.

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

Remaining Phase-1 rungs: device mirror (+ level-aware openness on device, seam parity
ctest), graded refineToSdf policy API, pocket exclusion in LS clouds, sub-face closures.

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

### Phase 3 — the porous payoff
Two-sphere gap unit case (throat criterion vs gap resolution n); then a mid-size sphere bed
with pnm-seeded L(x): permeability + cell count vs uniform-finest-band. **The headline
number** = accuracy-matched cell-count reduction.

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
| seam-row march instability (lagged-stiff-term shape) | M3, Phase-2 batteries | RETIRED at prototype level 2026-08-25 (M3: seam == control, cycling exact); Phase-2 bed battery remains the final check |
| classification flicker at seams | D2 flicker check across adapt cycles | D2-alternative (aggregation) |
| coarse openness pinches a throat | D3 tripwire assertion | D3-alternative (telescoping geometry) |
| fallback cascade fires too often (M1 >~ 30% of seam rows) | M1 census | RETIRED 2026-08-24: measured 0.1–0.5% on all maps |
| accuracy loss beyond codim-2 estimate | Phase-2 ladder | raise sample order (M2); widen seam collars locally |
| Snellius ladder disappoints | the gate | plan survives in outline; re-target the closure family per fluid_only_constraint_plan.md decision tree |

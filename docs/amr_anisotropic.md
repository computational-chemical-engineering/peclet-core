# Anisotropic AMR — per-axis root spacing through the octree, the mixed-level cut band and the sampled builders

*Design note, 2026-09-06, Phase 3 (AMR half) of `suite/docs/PHYSICAL_UNITS_PLAN.md` (§9.5,
§3.4). Status: **IMPLEMENTED AND MEASURED** (2026-09-06) — work orders A0–A5 landed; §4 and the
A3 entry of §9 were REWRITTEN by what the measurement said, which is the one place the design was
too optimistic. The VoF half is `flow/doc/anisotropic_vof.md`; its §1 Rule A/Rule B are the rules
here too and are restated in §1. Plan decisions D1–D5 are not reopened.*

## 0. The result in one paragraph

`AmrGeometry<Dim>::h0` becomes a `Vec<Dim>`: the root brick's cells are boxes `h0_x × h0_y × h0_z`,
a level-`l` leaf is `2^l h0` per axis, and **every level inherits the aspect ratio** because the
octree refines by 2 on all axes (Morton codes, `bounds()`, `level()`, the halo, the ORB, the
rebalance and the adapt/remap are integer fine-unit arithmetic and do not change). Every
downstream use of `h0` falls into one of four kinds — a *position* (per-axis already), a
*face area or a distance along an axis* (per-axis by construction), a *scalar operator constant*
that hides an axis sum (`6*beta`, `F/h0²`, `sqrt(3)/2·w`, `4*h0` bins, a ball radius) which is
rewritten as a per-axis sum or an axis-wise metric, and a *threshold length* (refinement bands,
gap floors, opening criteria) which takes `max_a h0_a`. The 1-D ghost closure and the mixed-level
cut band are expressible per axis without a new scheme: the closure is a function of samples
along ONE axis at `±q h_a`, the openness rule is a sign rule on centre samples, and the
least-squares virtual samples are affine-invariant (the degree-2 polynomial space is preserved
by a per-axis scaling), so the clouds move to an index-space (ellipsoidal) metric with identical
weights up to round-off — and identical bits at equal spacings. The one octree-inherent limit is
multigrid: the hierarchy coarsens all axes together, so an aspect ratio persists on every level
and the point smoother fails — **measured, the standalone V-cycle DIVERGES beyond aspect ~1**, so
on a box mesh it is a preconditioner and not a solver (§4). The operators themselves are exact
there (A2 = 1e-15 on the cut-cell Poiseuille, A4 = -0.76 % on Zick & Homsy); it is the V-cycle's
smoother/transfer pair that is the limit, and the remedies are named and unscheduled. Python: `Octree(..., extent=(Lx, Ly, Lz))` accepts any positive extent; the cubic assert
moves to the scalar `spacing_from_extent` helper, which keeps its contract.

## 1. Rules (shared with the VoF note)

**Rule A — per-axis constants only.** No operator gets a different algorithm for `h0_x != h0_y`.

**Rule B — the isotropic path executes today's floating-point operations**, in one of three
forms, first applicable wins: (1) a ratio that is exactly `1.0` at equal spacings (`x*1.0`,
`x/1.0`, `q/q`); (2) the same value on every axis in the same loop order; (3) a guarded isotropic
branch, only where the operation tree genuinely differs (`sqrt(Dim)*w` vs `|w|`) — every use listed
in its commit message. A former single product that becomes an axis sum (`6*beta`, `F*inv`) is
parenthesised as one sum before anything else is added, so the isotropic value is the single
correctly-rounded result it was.

`AmrGeometry` gains the helpers that make (1) natural: `hMin()`, `hMax()`, `ratio(a) = h0[a]/hMin()`,
`cellVolume(level)`, `isotropic()`. For the AMR the natural reference is `hMin`; nothing below
depends on that choice at equal spacings.

## 2. `AmrGeometry` and its two funnels

    template <int Dim> struct AmrGeometry {
      Vec<Dim> origin{};
      Vec<Dim> h0{1,...};                                   // per-axis finest (level-0) spacing
      Vec<Dim> leafSize(unsigned level) const;              // h0 * 2^level, per axis
      Real     leafSize(unsigned level, int a) const;       // one axis
      Vec<Dim> lowerCorner(lo) const;   Vec<Dim> center(b) const;   // per-axis, as today with h0[d]
      Real hMin() const; Real hMax() const; Real ratio(int a) const; Real cellVolume(unsigned) const;
      bool isotropic() const;
    };

`lowerCorner`/`center` are already per-axis loops with a scalar factor; `origin[d] + lo[d]*h0` →
`origin[d] + lo[d]*h0[d]` is Rule B(2). `leafSize(level)` returning a `Vec` breaks every scalar
caller at compile time, which is the desired inventory mechanism: **there are ~180 `h0` uses in 23
headers plus the bindings, and the compiler finds all of them** (§3 classifies them so no
judgement is made twice).

Construction: `AmrGeometry{origin, h0}` with a `Vec`; a convenience `AmrGeometry::isotropic(origin,
h)` keeps every existing C++ test one-line. `DistributedOctree::h0()` returns the `Vec`; a
`hMin()` accessor is added for the callers that want one number.

## 3. Every `h0` use, classified (the implementer's map)

Kinds: **P** position/coordinate (per-axis, Rule B(2)); **F** face area × distance along an axis
(per-axis products in the same loop order, Rule B(2)); **S** scalar constant hiding an axis sum
(rewrite, Rule B(1) ratio form or a parenthesised sum); **T** threshold length (`hMax`/`hMin`, an
explicit `max`/`min` that is the old value at equal spacings); **M** metric of a neighbourhood
(ball → axis-scaled ball; bins per axis).

| header | uses | kind → treatment |
|---|---|---|
| `leaf_field.hpp` | `leafSize`, `lowerCorner`, `center` | the funnels, §2 |
| `poisson.hpp` (20) | `cellWidth(i)`, `areaOf(s)`, `coeff(si,sj)`, face centroids `plane*h0`, `(lo+0.5s)*h0`, wall term `areaOf/(0.5 s h0)` | **F/P**: `cellWidth(i,a)`, `areaOf(s,axis) = Π_{b≠axis} s h0_b` (same two multiplications, same order), `coeff(si,sj,axis) = areaOf(min,axis)/(0.5(si+sj) h0_axis)`; `cellVolume = Π`; the wall term per axis. The `fn(j, axis, dir, area, dist, open)` callback already passes `axis` |
| `multigrid.hpp` (10) | `build(finest, h0)`, `addCoarseStarStencil(…, h0)` tangential distances | **P**: the tangential offset along axis `tt` uses `h0[tt]`; hierarchy build passes the `Vec` |
| `fv_op.hpp`, `assembly.hpp`, `facegeom_assembly.hpp`, `momentum_assembly.hpp` | `inv = 1/h0²`, `emit.h0`, `areaOf`, `dist` | **S/F**: per-axis `inv[a]`; the device `FvOp` diagonal `F*inv` becomes `((2 inv_x + 2 inv_y) + 2 inv_z)` (Dim-generic loop, parenthesised); `emit.h0` a `Vec` |
| `distributed_poisson.hpp`, `distributed_view.hpp` | `inv = 1/h0²`, `diag = F*inv`, `lg.h0 = h` coarsening `h *= 2` | **S**: as above; coarsen every axis by 2 (`h0 *= 2` per axis) |
| `distributed_fv.hpp` (9) | `cellWidth(level)`, `areaOf`, `dist`, root width `ig.h0 = h0*2^lmax` | **F/P**: per axis |
| `distributed_flow_mg.hpp`, `velocity_mg.hpp` | `ap.init(…, h0)`, `coef = mu/H²` ("isotropic coarse cell"), `Ldom = h0·brick[0]·2^lmax` floor | **S**: `coef_a = mu/H_a²` per face axis, diagonal as a parenthesised sum; `Ldom` keeps its axis-0 expression (`h0[0]·brick[0]·2^lmax` — it is a non-singularity floor, identical today on non-cubic bricks; not a discretisation) |
| `cut_cell.hpp` (10) | `beta = mu/h0²`, `AC0 = idiag + 6 beta`, `off[k] = -beta`, `buildCutStencil(…, beta, …)`, centre/subsample positions `base + (a+0.5)/nsub·w` | **S/P**: `beta[3] = mu/h0_a²`; `AC0 = idiag + ((2β_x + 2β_y) + 2β_z)`; `off[k] = -beta[k/2]`; `buildCutStencil` takes `const double beta[3]` (the ξ overlay is 1-D per axis: `vnb = -beta[axis]`); the 4³ subsample positions per axis (`w_a = s h0_a`) |
| `ghost_projection.hpp` (5) | `makeBinaryOpenFn(sdf, h0)` probes `±h0/2` along the face axis; `invh = 1/cellWidth` per row; delta kernels `ih²·Σ_a(…)_a`, `ih·dd` | **P/S**: probe `±h0[axis]/2`; `invh[3]` per row (three doubles); the Laplacian delta `ih_x²(…)_x + …` spelled as `ih²·(r_x²(…)_x + r_y²(…)_y + r_z²(…)_z)` with `ih = 1/hMin(level)`, `r_a = hMin/h_a` — exactly `1.0` at equal spacings so the bits are today's; the divergence delta per axis `ih_a·dd_a` the same way |
| `ghost_projection_sampled.hpp` (21) | see §5 | **P/M/S** |
| `cf_scheme.hpp` | tangential offset `dt·h0` (`:106`), `H = cellWidth(coarse)`, `h/H` normal weights | **P/F**: `dt` along axis `tt` uses `h0[tt]`; `H`, `h` along the closure axis; `wF = (H/2)/d` etc. per axis — same expressions with the axis's spacing |
| `flow.hpp` (24) | `init(t, h0, origin)`, `beta = mu/h0²`, `faceFrac` order-2 offsets `e = 0.5 h0` and order-1 FD gradient `±h0`, `denom = (|g_t1|+|g_t2|)/gmag·h0`, `pres_.cellWidth(i)` in the ±2 chain and advection | **S/P**: `beta[3]`; `e_t = 0.5 h0[t]` per transverse axis; the FD gradient per axis; `denom = (|g_t1|·r_t1 + |g_t2|·r_t2)/gmag·hMin` (ratio form, Rule B(1)); the chain samples at `±q·cellWidth(i, a)` |
| `flow_oracle.hpp` (14) | the serial mirror of the above | the same edits (oracle == device parity is a gate) |
| `scalar_transport.hpp` (6) | `cellWidth`, areas, distances, face-plane centroids | **F/P** per axis |
| `refine.hpp` (6) | `halfDiag = 0.5√Dim·width`, `band·h0`; `gapFloorTarget(h0·2^(L+1)·n ≤ g)` | **S/T**: `halfDiag = 0.5√Dim·w_min·sqrt(Σ r_a²/Dim)` (`sqrt(1.0) = 1.0` exactly at equal spacings — Rule B(1)); `band·hMax`; the gap floor compares against `hMax·2^(L+1)·n` (the coarsest direction decides — refines no less than today) |
| `barnes_hut.hpp` (3) | binning `floor((x - o)/h0)`, opening `width = h0·2^L` | **P/T**: per-axis binning; `width = hMax·2^L` (the opening criterion stays conservative) |
| `distributed_octree.hpp` (2) | `h0()`, `localGeometry` origin shift | **P** |
| `vtu_io.hpp` | corner positions | **P** |
| `adapt.hpp`, `indicators.hpp`, `advect_recon.hpp`, `distributed_adapt.hpp` | none directly (go through `cellWidth`/positions) | follow their callee |
| `python/amr_bindings.cpp` (39) | constructors, `h0`, `spacing`, `extent`, `sizes`, `spacing_from_extent`, `h0FromExtent`, `Flow` extent | §6 |

## 4. The operators on boxes — DECISION AM1: octree coarsening keeps the aspect ratio

The FV Laplacian on a box mesh is the standard one: face coefficient of axis `a` between leaves
of widths `s_i h0`, `s_j h0`

    coeff_a = (Π_{b≠a} min(s_i,s_j) h0_b) / (0.5 (s_i + s_j) h0_a)      (poisson.hpp::coeff, per axis)

and the momentum ξ-overlay's constant-coefficient fold is `beta_a = mu/h_a²` with diagonal
`idiag + 2 Σ_a beta_a` — the octree twin of flow's U10 (`beta_b = mu'/h'_b²`). Both are exact
per-axis rewrites of what the code does, nothing else changes: the 2:1 sub-face enumeration, the
Martin–Cartwright coarse star (tangential offsets are per-axis lengths), the PCG, the ghost
projection's constraint rows (§5), the pocket guard, the deflation.

**Multigrid.** `AmrMultigrid`/`DistributedMultigrid` coarsen by merging sibling groups — all axes
at once — so a root aspect ratio `(1, 0.5, 2)` is the aspect ratio of every level. With a point
smoother that is the textbook anisotropic-Poisson degradation (the strong-coupling direction is
not smoothed; only semi-coarsening or line relaxation restores the rate) and **semi-coarsening
is not expressible on an octree** (a level whose cells are not the parents of the finer level's
cells is not an octree level; the halo, the ORB and the 2:1 balance all assume the parent
relation). DECISION: keep full coarsening; bound the cost by measurement (gate A3: MG-PCG iteration
count at aspect 4 ≤ 3× the cubic count, reported); state the supported range as aspect ratio
≤ 4 between any two axes, with a one-line stderr notice above it. The remedies if A3 fails, in
order: a per-level Chebyshev smoother with the anisotropic eigenvalue bounds (no structural
change), then a line (axis-wise tridiagonal) smoother on the strong axis. Neither is scheduled.
This is the octree analogue of flow's U11 ⚑ (coarsen the finest axis first) and it is decided
differently *because the octree has no per-axis coarsening to choose from*.

**The bottom solver** (`distributed_fv.hpp`: the uniform root-grid `DistributedMultigrid`
under the graded hierarchy) inherits the per-axis root width `h0·2^lmax` and the same statement.

## 5. The ghost closure and the mixed-level cut band, per axis — DECISIONS AM2–AM4

`scheme/ghost_closure.hpp` is a pure 1-D function of SDF samples along ONE axis (`th =
sdf_near/(sdf_near - sdf_ghost)`, the float closure polynomials of `th`): with the samples taken at
`c ± q h_a e_a` it is the closure of a stretched cell along axis `a` without any change — the
wall crossing along a grid line is an axis ratio (VoF note §2). The row therefore carries three
`invh[a] = 1/cellWidth(i, a)` instead of one, and the delta kernels use them per axis in the
ratio form of §3 (`ghost_projection.hpp` row) so the uniform-band path stays bitwise. That settles
the classic overlay and the `gpFillRow` shared with flow (flow's Phase 2 U12 makes the same
statement from its side; the shared header does not change).

The sampled overlay (`ghost_projection_sampled.hpp`) adds four per-axis facts:

**AM2 — Virtual sample positions and openness probes are per-axis lengths.** `p = c + q h_a e_a`
with `h_a = cellWidth(i, a)`; `makeBinaryOpenFnMixed` probes `±0.25 h0_axis` across a face;
`probeCoord` floors per axis with `h0[d]`; the leaf/ghost centres are §2's `center`. All Rule B(2).
The classification invariant (overlay-closed ⇔ binary-closed, a sign rule on centre samples) is
untouched.

**AM3 — The LS clouds use the index-space metric.** Today a cloud is "every fluid leaf whose
centre lies within the Euclidean ball `rho = 2.2·max(h, H)` of the virtual position", enumerated in
a canonical `(4h0 bin, global Morton key)` order, and the degree-2 fit is done on `d = del/H`. The
per-axis form: the candidate set is the axis-scaled ball `Σ_a (del_a / (2.2·max(h_a, H_a)))² ≤ 1`
(i.e. today's ball in index units of the coarser of the two levels — the same leaves as today on
every axis at equal spacings, and the same *count* of leaves per axis on a stretched grid, which is
what the 12-point degree-2 requirement cares about); the covering box `qlo/qhi` is floored per axis
with `rho_a/h0_a`; the bins are `4 h0_a` per axis (the order key is unchanged in meaning: bin
traversal in `(bx, by, bz)` then Morton); the monomials are built on `d_a = del_a/H_a`. The fit is
**affine-invariant**: the degree-2 space is closed under per-axis scaling, so the LS weights are the
same functional as today up to round-off, and with `H_a == H` on every axis they are bitwise
(`del/H` on each axis is today's operation). The fallback cascade LS2 → LS1 → covering identity,
the `PECLET_CORE_GPS_RHO`/`_MAXN` study knobs (they scale `rho_a` and cap the candidate count),
the deterministic probe set (`forEachCoveringSlot` over the per-axis box) and the accumulation
order are unchanged in structure.

**AM4 — The periodic minimum-image period is per axis.** `domain = nbx·hb` (one number,
"cubic domains") becomes `domain_a = gfine_a·h0_a`. On a cubic periodic brick that is today's
number on every axis. On a **non-cubic brick** today's single period is the longest axis's, which
is wrong for the shorter axes whenever a cloud straddles a periodic boundary there — so a
bit-identity gate CAN move on such a case, and if it does, that is the pre-existing wrong period
surfacing, not the anisotropic change. Pre-announced here so it is reported as such (the plan's
STOP rule applies; the report says which test and shows the two periods).

None of the four needs a new scheme: the closure is 1-D, the openness is a sign rule, the clouds
are affine-invariant and the period is a per-axis wrap. This is the answer to the brief's STOP
condition "the AMR change reaches a scheme the mixed-level cut band cannot express per-axis": it
does not. The known gaps of the band (sub-face closures Neumann-zero, pocket cells in clouds) are
unchanged by anisotropy and stay in the risk register.

## 6. Sampled builders and thresholds

- `cut_cell.hpp` openness/fluid-fraction subsampling (`nsub = 4` per axis): sample positions
  `base_a + (k + 0.5)/nsub · s h0_a` — per axis (Rule B(2)); the 1/64 quantum is unchanged.
- `flow.hpp::faceFrac` order 2 (the default, `apertureOrder_ = 2`): the four corner probes at
  `±0.5 h0_t1`, `±0.5 h0_t2`; `triFrac` unchanged. Order 1 (ablation): per-axis FD gradient and
  the ratio-form `denom` of §3.
- `refine.hpp`: `refineToSdf` band `|phi| ≤ halfDiag + band·hMax` with the ratio-form
  half-diagonal; `refineToSdfGraded`/`gapFloorTarget` compare the gap against `n·hMax·2^(L+1)`
  (coarsest direction decides; the tagging floor of the cut-band plan's D3 stays an invariant on
  the coarsest axis, which is the conservative reading).
- `barnes_hut.hpp`: per-axis binning; opening width `hMax·2^L`.
- `vtu_io.hpp`: corners per axis; ParaView shows boxes.

## 7. Python bindings (`core/python/amr_bindings.cpp`, `packaging/core_amr.pyi`)

- `Octree(brick, lmax, origin=(0,0,0), h0=1.0, extent=None)`: `h0` accepts a float **or a 3-tuple**;
  `extent` derives `h0_a = extent_a/(brick_a·2^lmax)` **per axis, no cubic assert** (the plan's
  "relax `h0FromExtent`" — done by routing the constructor through a new `spacingsFromExtent`
  that returns the triple).
- `spacing_from_extent(extent, root_cells, lmax) -> float` keeps raising on a non-cubic extent
  with the same "CUBES" message (its contract is one number; `test_amr.py` keeps its negative
  check there); `spacings_from_extent(...) -> (dx, dy, dz)` is added.
- `Octree.spacing -> (dx, dy, dz)`; `Octree.extent` per axis; `Octree.h0 -> float` returns the
  spacing when the three are equal and **raises `RuntimeError("anisotropic octree: use
  .spacing")`** otherwise — never silently the x value; `sizes()` (leaf widths) likewise, with a
  new `sizes(axis)` per-axis form; `centers()` unchanged (per-axis already).
- `DistributedOctree(global_root_size, lmax, origin, h0, periodic, extent=None)`: the same
  float-or-triple `h0` plus an `extent` keyword (global box); `.h0` as above; `.spacing` added.
- `Flow(octree, ...)`: `extent_` per axis (it is used for `set_solid_spheres`' min-image box).
- `.pyi` stubs updated; docstrings say "boxes" where they said "cubes".

## 8. Work orders (one commit each; the message names the gate and its numbers)

| WO | what | gates |
|---|---|---|
| **A0** | `AmrGeometry::h0` → `Vec`, helpers of §2, the compile-driven inventory: every **P/F** site per axis, every **S** site in ratio/parenthesised form, every **T** site `hMax`; `buildCutStencil(beta[3])`; oracle mirrored | A1 (the whole battery bitwise) |
| **A1** | ghost overlay `invh[3]`, delta kernels in ratio form (`ghost_projection.hpp` host + device) | A1 |
| **A2** | the sampled band: AM2 positions/probes, AM3 axis-scaled clouds + per-axis bins + monomials, AM4 per-axis period; discovery mode identical control flow | A1, A5 |
| **A3** | Poisson/MG/velocity-MG/FvOp/distributed per-axis constants; the aspect-ratio notice | A1, A3 |
| **A4** | bindings + `.pyi` + `test_amr.py` new cases (anisotropic constructor, Poiseuille ×3 orientations, `h0` raise, `spacing_from_extent` still raises) | A1, A2, A6 |
| **A5** | new C++ gates: stretched `amr_poisson`/`amr_multigrid` (A3), stretched `amr_drag` uniform + graded (A4), stretched `amr_distributed_seam` (A5); umbrella pointer bump last | A2–A6 |

core edits go in the shared `core/` checkout only while `git status` is clean there (untracked
build logs excepted); named paths only; `git pull --rebase origin main && git push origin HEAD:main`.

## 9. Acceptance gates and the numbers they must hit

**A1 — bit-identity (every commit).** The full core ctest suite in the plain Release tree
(**104/104** as recorded in RELEASE_PREP §8.1, incl. `domain`) and in the Kokkos tree (device AMR +
MPI np = 1, 2, 4, 8; the count recorded on the day), `core/python/test_amr.py` at np = 1, 2, 4 — all
green, and a byte dump of `Flow.velocity(0..2)` + `pressure()` after the Poiseuille case and of
the adapt-transport field on the cubic extent, `np.array_equal` IDENTICAL against a dump from the
unmodified tree. `amr_distributed_seam` np = 1 stays `gdmax == 0.0` (bitwise vs single-rank).

**A2 — anisotropic Poiseuille, exact** (`test_amr.py`, plan §5.3 on the AMR side). The existing
16³ cut-cell channel on `extent = (1, 0.5, 2)` (`h0 = (1/16, 1/32, 1/8)`), walls ⊥ x with the force
along y, then walls ⊥ y / force along x, then walls ⊥ z / force along x: `max|u - u_parabola| /
max u_parabola < 1e-6` in all three orientations, cross-flow `≈ 0`, `divergence_norm < 1e-6`,
`|u| < 1e-9` in the solid — the same numbers the cubic case holds today, because the quadratic is
exact per axis.

**A3 — multigrid on aspect ratios 1, 2, 4** (`test_amr_poisson.cpp::test_anisotropic_mg`).
**RESTATED after measurement** (§4): the gate ASSERTS the cubic path (10 V-cycles to `1e-10`, and
`<= 12` is the bound it holds) and REPORTS the anisotropic rows, because the V-cycle diverges there
and asserting a rate the scheme does not have would be fiction. The anisotropic operator's
correctness is gated by A2 and A4 instead, neither of which uses the V-cycle as its outer solver.

**A4 — Zick & Homsy drag on stretched root spacing** (`amr_drag`, new cases; plan §5.4).
Uniform: min-axis count N = 16 → 32 on `h0 = (h, 0.5h, 2h)`: the finest rung within **3 %** of
`k_ZH = 4.292` (the cubic gate at N = 16) and the two-rung error ratio **≥ 2** (order ≥ 1). Graded
(`refine_to_sdf_graded`, `setGhostSampled`): stable (`|u_sup| < 1`) and within **10 %** of Z&H, as
the cubic graded gate.

**A5 — the distributed cut band on a stretched mesh** (`amr_distributed_seam_mpi`, stretched
variant). np = 1: `gdmax == 0.0` versus the single-rank build (bitwise, the D6 contract);
np = 2, 4, 8: `gdmax ≤ 5e-6·gscale` (decomposition independence, unchanged tolerance).

**A6 — the physical-units case** (`test_amr.py`). `Octree(brick=[16]*3, lmax=0, extent=[1, 0.5,
2]).spacing == (1/16, 1/32, 1/8)` exactly; `.extent` round-trips; `.h0` raises with "anisotropic";
`spacing_from_extent([1, 0.3, 1], …)` still raises with "CUBES"; `spacings_from_extent` returns
the triple; lmax = 2 divides each spacing by 4.

## 10. Open, and what would stop the work

- **`Ldom` in `velocity_mg.hpp`** keeps its axis-0 form (a floor). Recorded; not a discretisation.
- **AM4** can move a bit on a non-cubic periodic brick test — pre-announced above; report, do not
  paper over.
- **The V-cycle on a box mesh** is a preconditioner, not a solver (§4, measured). Every operator
  is correct at any aspect ratio (A2 exact to 1e-15, A4 -0.76 %); a caller that drives
  `AmrMultigrid::vcycle` to a tolerance must stay near-cubic. Chebyshev / line smoothing are the
  named remedies and are not scheduled.
- **STOP conditions** (brief + plan §9.6): a bit-identity gate fails and two focused attempts do
  not find it; np = 1 stops being bitwise; or a scheme turns out not to be expressible per axis —
  §5 argues none does; if the implementer finds one, the note is wrong and the finding goes into
  the report with the evidence.

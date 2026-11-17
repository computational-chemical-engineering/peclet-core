# peclet.halo (core python/) playbook: user-facing round (WO-35, 2026-11-17; deltas from dem's, flow's, amr's, pnm's)
- **Scope**: python/ only; never include/ (every consumer builds on it). Gate: config.toml here, `--behaviour-only --base 150f00e`.
- **Checks**: `python/halo_bindings_checks.hpp` (`halo_checks`, alias `chk`), one `chk::` line at the top of each
  constructor/method of the TU's wrapper classes. Array shape stays unconstrained in the nanobind type (a `nb::ndim<2>`
  would change `__nb_signature__`); the rank and N are checked by hand. Message strings are built only on failure.
- **Local, not collective**: no added communication; a bad per-rank array raises on that rank only (documented).
- **Found**: 72 silent + 1 crash of 113 per np. extent/cells <= 0 accepted; NaN positions binned anywhere; rcut < 0 acted
  as |rcut| (`d2 < rcut*rcut`); a cells product wrapping int64 gave rebalance a 0-length weight array (SIGSEGV);
  wrong-shape positions/payload/halo fields read or wrote past the array. Probe table: `probes_wo35.tsv`.
- **Valid input kept**: rcut 0, empty arrays, K = 0, fewer cells than ranks, float32 (converted), huge-but-representable
  cells (rebalance then raises MemoryError), far finite points.
- **Not fixed (header side)**: cellOf casts floor(rel) to int64 unchecked; a point beyond 2**63 cells is UB (clamps wrong).
- **Traps**: the shell test python_shell_deprecation_geom skips without peclet-geom: excluded by name in the gate.
  The checks header is outside quality.yml's clang-format glob (python/*.cpp); the gate's tier adds python/*.hpp.
- **Wall clock (TRS2)**: build 8 s; gate --behaviour-only 44 s; probes 113 cases x np 1, 2: about 1 min; pytest 102 cases 0.4 s.

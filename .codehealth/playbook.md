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

# peclet-core playbook: host-orchestration round (WO-40, 2026-11-18; config.toml now gates include/, not python/)
- **Scope**: include/ only, zone `host` (cpp@1 + zones, tool 0.3.0). Every consumer (flow, amr, dem, pnm, voro,
  coupling) builds on these headers, so the round also ends with consumers' state hashes bitwise. Public names and
  signatures stay exactly as they are (consumers pin core by tag). Gate:
  `PECLET_SUITE=/home/frankp/Codes/suite codehealth gate --ref-base <onboarding sha>`.
- **Reference**: `tests/regression/state_hash_driver.cpp` (raw output bytes of fixed-input runs; not a ctest) and
  `tests/regression/state_hash.py` (SHA-256 per case; halo/migrate at np 1, 2, 4, 8; halo on three transports that
  must agree; the per-test FNV digests of test_anderson and test_anderson_mpi np 1, 2, 4), plus `python/state_hash.py`.
- **Keep-out**: `solver/graph_amg_setup.hpp`, `solver/graph_amg_device.hpp`, and `HostCsrOp` / `AmgParams` in
  `solver/graph_amg.hpp` (session voronoi's unlanded branch adds lines there): no edit, move or rename.
- **Device despite the scan**: `scheme/cut_cell_geometry.hpp` (`PECLET_CORE_CC_HD`) and `geom/quadrature.hpp`
  (`PECLET_HD`) are called per cell from flow's `KOKKOS_LAMBDA` kernels (src/scalar_cutcell_geometry.hpp); the scan
  counts them host because the macros are not Kokkos markers. Leave them alone (cutCellGeometryFanTet skipped).
- **GridHalo**: exchangeBegin runs every time step of every consumer. Helpers are private members (implicitly
  inline), no allocation, no `std::function`, same MPI calls, order and tags. nvcc rejects an extended lambda in a
  private member function, so the gather launches stay in exchangeBegin; only lambda-free phases move.
- **Host flags**: the Kokkos host build compiles with `-ffp-contract=off` and no `-march`, so a function seam cannot
  change FMA contraction on the host (dem's S7 trap needs contraction on).
- **Traps**: `pkill -f <pattern>` from a shell whose own command line contains the pattern kills that shell (144).
- **Halo stress is campaign cadence**: the 15 grid_halo_variants instances (10^4 skewed exchanges each) take 9-25 min
  each alone at load ~110 and time out at 1800 s when co-scheduled; tier halo_stress runs them -j1 after the GridHalo
  steps and at the round's end (`gate --cadence campaign`). Measured base, 2026-11-18: main battery 98 tests 2480 s
  at -j8 (4 np4 variants timed out co-scheduled), np8 subset 22 tests 4253 s at -j1 (all passed).

# GraphAMG parallel setup (C1) and FP32 values (C2) — results

Design: peclet-papers `work/M7/DESIGN_NOTE.md` §14.5 / §14.6 (WO-CORE-AMG). Branch `amgpar`.
Code: `include/peclet/core/solver/graph_amg_setup.hpp` (the setup), `graph_amg_device.hpp`
(`build(DeviceCsr)`, dense coarsest level, FP32), `AmgParams` in `graph_amg.hpp`. Gate test:
`tests/test_graph_amg_parallel.cpp` (ctest `graph_amg_parallel`; `--hash m`, `--bench m`).

## API (opt-in; the defaults are byte-identical to 8bfd73c)
`AmgParams::aggregation` {`Greedy` (host oracle, default) | `Mis2`}, `coarseSolve` {`Cg` | `Dense`},
`precision` {`Fp64` | `Fp32`}. Only `GraphAMGDevice` honours the new options; the host `GraphAMG`
throws on them. `GraphAMGDevice::build(const DeviceCsr&, prm)` builds on the device (requires `Mis2`);
`coarseSolveUsed()` / `denseFallback()` report the coarse solve actually in use.

## Decisions (orchestrator, 2026-11-09)
- **Q1** λ estimates: per level, two ∞-norm power iterations (15 steps, ±1 splitmix64 seed of the DOF
  index): λ(D⁻¹A) for the Chebyshev smoother (which keeps its 1.1), λ(D_F⁻¹A_F) for prolongator
  smoothing with NO safety factor (host semantics). The coarsest level uses λ(D⁻¹A) only. Both run
  in one fused pass over the matrix (bitwise the two separate iterations).
- **Q2** `Dense` stops coarsening at ≤ 200 DOFs (`kDenseCoarseMax`). If the coarsest level stays
  larger (maxLevels, stalled coarsening) the existing coarse CG runs, `denseFallback()` is true and
  `coarseSolveUsed()` is `Cg`; tested with maxLevels = 1. `coarseSweeps > 0` keeps precedence.
- **Q3** gate (d) at 8 threads only when 1-min load + 8 ≤ 36; everything else ≤ 4 threads.
- Implementation readings accepted: leftover rule over strong-adjacent aggregates; Gustavson with a
  dense per-thread accumulator from a `UniqueToken` pool; the host's accumulation order everywhere
  (so steps 5–7 are checked bitwise); fused smoother kernels out of scope.
- **C2 scope (this agent, flagged for confirmation)**: "P, R, A_c on levels ≥ 1" read narrowly — the
  fine↔level-1 transfer P₀/R₀ and the fine residual operator stay FP64; FP32 = A_c, P, R from level 1
  down, all smoother diagonals, and the fine smoother copy. One line in `toFp32()` flips P₀/R₀.

## Gates (host-openmp, shared host at 1-min load 31–34; timings ±30 % run to run)
| gate | result |
|---|---|
| (a) default path unchanged | default-path dump (greedy hierarchy + host and device V-cycles, s = 1, 3, Jacobi and Chebyshev) byte-identical to 8bfd73c, 3.1 MB. Core battery `-LE "bench\|np8"`, 1 thread/rank: Kokkos (host-openmp) 95/95, CPU 62/62 (np8 subset not run: load). flow 0abafe2 `state_hash.py` against core 8bfd73c vs this branch: 13/13 cases identical. |
| steps 5–7 exact | P, R, A_c bitwise equal to a host-order reference recomputed from the mis2 aggregates on every level of all 4 operators |
| (b) iterations | m 12 → 24 (N ×8), PCG 1e-8: poisson7 greedy 8→9, mis2 8→9; voronoiH 7→9 / 7→9; S₀ 16→22 / 16→16; vec3 8→8 / 8→8. ≤ 1.2× oracle and ≤ +2: **pass** (dense bottom identical counts) |
| (c) setup, N 32768, 4 thr | S₀ setup 123 ms = **16.4 V-cycles (gate ≤ 10: miss)**; 1→4 thr speedup 332/123 = **2.7× (gate ≥ 3: miss)**. poisson7 29 V-cycles (35 ms) |
| (d) determinism | hierarchy hash (4 operators × cg/dense, m 16) identical at 1/2/4 threads and on the Serial backend; **8 threads not run** (load 32–34 > 28) |
| C2 iterations | FP32 = FP64 on every operator at m 12/24 (gate ≤ +10 %: pass) |
| C2 V-cycle, N 32768 | S₀ −14.8 %, vec3 −14.2 %, voronoiH −8.6 %, poisson −5.7 % (**gate −25 %: miss**); P₀/R₀ FP32 variant: S₀ −21.4 % |
| dense coarsest | singular Neumann 125 DOFs: ‖A·A_c⁺b − b‖∞/‖b‖∞ = 1.8e-14; Neumann 4096 PCG 9 iterations |

Gate (c) breakdown, S₀ N 32768, ms at 1 / 4 threads: strength 51/16, MIS-2 + aggregates 38/15,
filter + fused λ 105/37, P 16/7, R 5/3, AP 77/26, RAP 30/23. The design's own spec puts a floor near
the gate: two 15-step power iterations per level ≈ 15 fused SpMV passes ≈ 1.7–3 V-cycles; MIS-2 ≈ 4
strong-graph passes per round. The 1→4 speedup of the V-cycle itself is ~2× on this loaded host.
C2's −25 % cannot be reached by values alone: `Index` is int64, so FP32 values save 4 of 16 bytes
per nonzero (≤ 25 % of matrix traffic, less of a V-cycle). Reaching it needs 32-bit column indices.
Both (c) and C2 want a quiet-host re-measurement before any design change.

## Open items
- CUDA: the Gustavson pool is concurrency × n_c (fine on host; on CUDA a hashed per-row
  accumulator is needed). Untested on CUDA (GPU held); `sortPairs`' constexpr gap table and the
  `PRowBuilder` local lambda need a CUDA compile check.
- The `PECLET_CORE_AMGS_PROFILE` macro (compile-time only) prints the phase split in `--bench`.

## C3 (MPI) interface notes — not implemented
- `DeviceCsr` gains a row map: owned rows first, ghosts sorted by (owner, gid); `nbr` holds local
  column ids; a `gid` view per local row. Priorities already hash the gid (`misPriority(gid, level)`),
  and the λ seed hashes the DOF index — switch it to the DOF gid so the seeds are partition-free.
- MIS-2: each round's 4 passes become 4 one-layer halo exchanges of (state, m1/h1, m2/h2, f1); the
  kernels already read neighbours only through the strong-graph CSR (pull form), so ghost entries
  slot in. The leftover and subgraph phases read `root` of neighbours: one more exchange each.
- Aggregate ids: exclusive scan over roots in gid order = local scan + an `MPI_Exscan` of the counts.
- λ: the max reductions become `MPI_Allreduce(MAX)`; values stay partition-independent.
- P rows are row-local given ghost aggregates and ghost dF/keep; R and A_c: rows assembled at the
  root's owner from per-fine-row contributions summed in increasing fine gid (§14.5) — the current
  R row sort by fine index is that order once fine indices are gids.
- Coarsest: replicate (Allgather) and apply the dense A_c⁺ on every rank.

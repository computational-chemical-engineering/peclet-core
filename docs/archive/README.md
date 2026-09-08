# Archive — design notes and campaign records

These are **dated design notes, campaign plans and session handoffs**, kept for the record and moved
out of the live documentation set on 2026-09-08 (`suite/docs/QUALITY_PLAN.md` decision D7: *docs
describe the code that exists*). Each one steered a piece of work that has since landed, been
superseded, or been folded into a reference document; the authority on how `core` behaves today is
the code plus [README.md](../../README.md), [CLAUDE.md](../../CLAUDE.md) and the four reference notes
that stay in the parent directory ([amr_collocated_projection](../amr_collocated_projection.md),
[amr_mixed_level_cut_band_plan](../amr_mixed_level_cut_band_plan.md),
[amr_setup_parallel_plan](../amr_setup_parallel_plan.md),
[amr_anisotropic](../amr_anisotropic.md)).

Nothing here is maintained: `file:line` citations, status lines and "next step" sections are
snapshots of their date — re-read the source before acting on any of them. Nothing here is deleted
either — the AMR notes in particular document work that is still under active development
(QUALITY_PLAN decision D6).

**Source-comment citations.** Header, test and script comments written before this move cite these
notes by their old path (`docs/amr_distributed_flow.md`, `docs/amr_march_perf_and_distributed_plan.md`,
`docs/amr_aperture_advection_plan.md`, `core/docs/cuda-aware-mpi.md`). They all mean
`docs/archive/<same file>`; the prefix is added file by file as those sources are next touched, so
that a documentation pass does not churn code.

| note | date | what it is |
|---|---|---|
| [amr_aperture_advection_plan.md](amr_aperture_advection_plan.md) | 2026-08-18/19 | The AMR aperture pressure solve stalling under advection: the three candidate mechanisms, what was measured, and the **RESOLVED 2026-08-19** verdict (un-deflated RHS mean + a stale PCG gate; deflation wins over a compatible-RHS repair). |
| [amr_device_assembly_plan.md](amr_device_assembly_plan.md) | 2026-06/07 | Plan to move AMR operator/geometry *assembly* (cut stencils, openness, face CSR, FOU) from serial host code onto the device, with the assembly inventory table. Parked: `amr_setup_parallel_plan.md` took the multithreaded-host route instead. |
| [amr_distributed_flow.md](amr_distributed_flow.md) | 2026-07-26/28 | Design + rung-by-rung record of distributing `AmrFlow` (LeafHalo, the resolver seam, the distributed multigrid, `step`/adapt/rebalance). All rungs shipped; the shipped surface is described in `CLAUDE.md` §Architecture. |
| [amr_march_perf_and_distributed_plan.md](amr_march_perf_and_distributed_plan.md) | 2026-08-30 … 09-04 | The march-time economics + distributed mixed-level band campaign (M0–M2c, F2, D0–D3): the step profiler, the attribution matrix, the cloud-economy table and the distributed band. Both phases executed; M2b (pick the production `rho`/`N`) is the one open decision, and the two LS-cloud knobs stay inert at their defaults. |
| [comm_avoiding_pressure_driver.md](comm_avoiding_pressure_driver.md) | 2026-08-19 | Proposal for an all-reduce-free pressure driver (once-per-solve RHS projection + Chebyshev V-cycles), from the `dev/aperture-compat-rhs` experiment. **Never implemented** — kept so the option is on the table when multi-GPU pressure-solve scaling is the work item. |
| [cuda-aware-mpi.md](cuda-aware-mpi.md) | 2026-05/07 | Getting device-pointer MPI working on the workstation (user-space UCX 1.20 + OpenMPI 5.0.7 against CUDA 13.2) and the host-staging-vs-GPU-aware analysis from before Kokkos was the canonical device path. The build recipe is still valid; the integration sections describe the retired native-CUDA halo. |

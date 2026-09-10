# Archive — design notes and campaign records

Dated design notes and campaign records, kept for the record and moved out of the live
documentation set on 2026-09-08 (`suite/docs/QUALITY_PLAN.md` decision D7: *docs describe the code
that exists*). Nothing here is maintained: `file:line` citations, status lines and "next step"
sections are snapshots of their date — re-read the source before acting on any of them.

**The AMR notes moved with the AMR tree** to the `peclet-amr` package on 2026-09-10 (QUALITY_PLAN
D6 / G.2, relocated with their git history): `amr_aperture_advection_plan.md`,
`amr_device_assembly_plan.md`, `amr_distributed_flow.md`, `amr_march_perf_and_distributed_plan.md`
and `comm_avoiding_pressure_driver.md` are `../amr/docs/archive/<same file>` now, indexed by that
directory's README, together with the four reference notes (`amr_collocated_projection.md`,
`amr_mixed_level_cut_band_plan.md`, `amr_setup_parallel_plan.md`, `amr_anisotropic.md`) that were
in `docs/` here, the measurement logs of `docs/data/` and the study drivers of `tests/study/`.

| note | date | what it is |
|---|---|---|
| [cuda-aware-mpi.md](cuda-aware-mpi.md) | 2026-05/07 | Getting device-pointer MPI working on the workstation (user-space UCX 1.20 + OpenMPI 5.0.7 against CUDA 13.2) and the host-staging-vs-GPU-aware analysis from before Kokkos was the canonical device path. The build recipe is still valid; the integration sections describe the retired native-CUDA halo. |

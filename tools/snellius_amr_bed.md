# Running the AMR mixed-level bed study on Snellius

The Phase-3 headline (P3c) needs a compute-mode GPU: the depth-8 uniform control wants ~21.5 GB
and the local RTX 5080 has 16 GB *and* a kernel watchdog that kills 7M-leaf kernels. An H100
(95 GB, no watchdog) clears both. Recipe as run 2026-08-28.

## Layout — an ISOLATED checkout, not the shared suite

`/projects/0/prjs1022/peclet/suite/core` is far behind and `flow` builds against its headers, so
do NOT move it. Instead:

    /projects/0/prjs1022/peclet/amrbed/
      core/          # fresh clone at the AMR commit
      morton -> ../suite/morton      # core needs these as SIBLINGS
      cmake  -> ../suite/cmake       # SuiteNanobind lives here
      extern -> ../suite/extern      # the Kokkos CUDA install

    ln -sfn build_cuda core/python/build_cuda2   # the study hardcodes build_cuda2 in sys.path

## Transfer without pushing to origin

    git bundle create core-amr.bundle <remote-HEAD>..main     # needs a REF, not a bare range
    scp core-amr.bundle gap_unit_128.npy snellius:/projects/0/prjs1022/peclet/
    ssh snellius 'cd amrbed && git clone -q ../suite/core core &&
                  cd core && git fetch -q ../../core-amr.bundle main:bundlemain &&
                  git checkout -q <commit>'

## Build (login node, ~10 min)

    module purge; module load 2024
    module load OpenMPI/5.0.3-GCC-13.3.0     # find_package(MPI) fails without it
    module load CUDA/12.6.0                  # Kokkos install is HOPPER90 / CUDA 12.4
    cmake -S core/python -B core/python/build_cuda -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$P/suite/extern/install/nvidia-cuda \
      -DPython_EXECUTABLE=$P/suite/flow/.venv/bin/python
    cmake --build core/python/build_cuda -j16

Checked before building: the installed Kokkos defines `KOKKOS_ENABLE_CUDA_CONSTEXPR`
(without it morton's device encode silently returns 0 — see the kokkos-cuda-constexpr memory).

## The venv has no scipy

Ship a precomputed unit-box gap table instead and pass `--gap-file`; the gap proxy scales
linearly with N, so ONE table serves every depth, and the KD-tree build stays off billed GPU
time. Verified to reproduce the depth-8 probe exactly.

## Submitting

Billing is per ALLOCATED GPU. This study is single-rank (`set_ghost_sampled` is single-rank
only), so ask for exactly one:

    #SBATCH --account=tes24005 --partition=gpu_h100
    #SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16

sbatch echoes "You will be charged for 1 GPUs" — check that line. Never use the leading
`VAR=x sbatch` form (silently dropped on Snellius); pass modes as positional args.

Measured: smoke (depth-7 graded, 50 steps) = 0.56 s/step at 1.79M leaves, ~1.6x the local
RTX 5080; whole smoke job 7m16s including mesh build and setSolid.

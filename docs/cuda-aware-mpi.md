# CUDA-aware MPI on this workstation — WORKING (local build)

Direct **device-pointer** MPI (CUDA-aware MPI) now works on this box via a **user-space** OpenMPI+UCX
built against CUDA 13.2 — no root, no sysadmin. `tools/cuda_aware_mpi_check.cpp` passes:
`MPI_Send`/`MPI_Recv` on **device pointers** transfer device→device through UCX `cuda_ipc`/`cuda_copy`.

```
rank 0: sent 4096 doubles from device memory
rank 1: device recv CORRECT -- CUDA-aware MPI works
```

## The stack (built 2026-06, ~/opt)

- **UCX 1.20.1** — `~/opt/ucx`, `./configure --with-cuda=/usr/local/cuda-13.2 --enable-mt
  --enable-optimizations --without-rocm` → gives `cuda_copy` + `cuda_ipc` transports
  (`ucx_info -d | grep cuda`). NB: UCX **>= 1.20** is required for CUDA 13 (1.18.x fails to compile —
  the `cuMemGetHandleForAddressRange` typedef was versioned to `_v11070` in CUDA 13 headers).
- **OpenMPI 5.0.7** — `~/opt/openmpi-cuda`,
  `CFLAGS="-O3 -Wno-int-conversion -Wno-implicit-function-declaration -Wno-incompatible-pointer-types"
  ./configure --with-cuda=/usr/local/cuda-13.2 --with-cuda-libdir=/usr/lib/x86_64-linux-gnu
  --with-ucx=~/opt/ucx --without-rocm --disable-mpi-fortran --disable-oshmem`. Notes:
  - `--disable-oshmem` — OpenSHMEM `sshmem` fails to build under gcc 14 (`-Wint-conversion` is now an
    error) and we don't need it; the `-Wno-*` CFLAGS downgrade the other gcc-14 conversion errors.
  - **Do NOT use `--enable-mca-dso`.** With DSO components the prte-launched ranks can't locate the
    component directory and *every* accelerator component fails to open -> `MPI_Init` aborts. The
    default **static** build links the components in and works.

Release tarballs are pre-bootstrapped, so libtool/autoconf are not needed -- just `configure && make`.

## Using it

```bash
source ~/opt/cudampi-env.sh        # sets PATH, LD_LIBRARY_PATH, OPAL_PREFIX, OMPI_MCA_pml=ucx
mpirun -np 2 ./your_cuda_mpi_prog  # device pointers now go through UCX
```

`OMPI_MCA_pml=ucx` is the key: on a single node OpenMPI otherwise picks `ob1`+`sm` (no CUDA), so the
UCX pml -- which carries `cuda_ipc`/`cuda_copy` -- must be forced. Verify with:

```bash
source ~/opt/cudampi-env.sh
mpic++ transport-core/tools/cuda_aware_mpi_check.cpp -I/usr/local/cuda-13.2/include \
       -L/usr/local/cuda-13.2/lib64 -lcudart -o /tmp/check && mpirun -np 2 /tmp/check
```

## Gotcha: `MPIX_Query_cuda_support()` returns 0 (but device MPI still works)

OpenMPI's *accelerator framework* selects the `null` component here (the `cuda` accelerator is built
but isn't auto-selected), so `MPIX_Query_cuda_support()` reports **0** even though device-pointer
transfers succeed via UCX. **Do not gate the device path on `MPIX_Query_cuda_support()`** -- it would
wrongly fall back to host staging. Gate instead on a build/runtime flag (e.g. `TPX_CUDA_AWARE_MPI=1`,
set when this stack is in use) or probe UCX. This matters for the `DeviceGridExchange` swap-in below.

## Swap-in: device-pointer path in `DeviceGridExchange`

`include/tpx/halo/grid_halo_cuda.cuh` currently host-stages (`d_sendBuf_`->host->MPI->host->`d_recvBuf_`).
The swap-in passes `d_sendBuf_`/`d_recvBuf_` directly to `MPI_Isend`/`MPI_Irecv` (dropping the two
`cudaMemcpy`s) when CUDA-aware MPI is in use, falling back to host-staging otherwise -- a runtime-gated
branch in one function. Build the CUDA tests against this MPI (`source ~/opt/cudampi-env.sh`, then
configure with `-DMPIEXEC_EXECUTABLE=$CUDAMPI_HOME/bin/mpirun` and the matching `mpicc`) to exercise
it. On one shared GPU the win is modest (intra-GPU `cuda_ipc`); the payoff is real multi-GPU/node.

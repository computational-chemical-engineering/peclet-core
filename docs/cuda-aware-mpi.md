# CUDA-aware MPI on this workstation

`DeviceGridExchange` (the GPU-resident halo) uses **host-staged** transfers: pack/unpack/self-copy run
on the device, but the compact halo buffers are bounced through host memory for MPI. This is the
portable path and is correct/tested. Direct **device-pointer** MPI (a.k.a. CUDA-aware MPI) would skip
the bounce, but it **segfaults** on this box.

## Why (diagnosed 2026-05)

The stock Debian **OpenMPI 5.0.7 was built without CUDA support**:

- `ompi_info --all | grep cuda_support` → `opal_built_with_cuda_support: false`
- `ompi_info | grep accelerator` → only the `null` accelerator component (no `accelerator_cuda`;
  OpenMPI 5 routes CUDA through the accelerator framework).
- `libucx0 1.18.1` is installed and `pml ucx` exists, but the Debian UCX is built **without** the
  `cuda_copy`/`cuda_ipc` transports.

So MPI does a host `memcpy` on a device pointer → SIGSEGV. This is a single-GPU node, so only
intra-node **CUDA IPC / cuda_copy** is needed — no GPUDirect RDMA / `nvidia_peermem` / special fabric.

## To enable it (ask the sysadmin)

Provide a CUDA-aware MPI built against CUDA 13.2 (`/usr/local/cuda-13.2`), one of:

1. OpenMPI configured `--with-cuda=/usr/local/cuda-13.2 --with-ucx=<prefix>`, where UCX itself is
   built `--with-cuda` (gives `cuda_copy` + `cuda_ipc`); or
2. a prebuilt CUDA-aware stack — NVIDIA **HPC-X** or the **HPC SDK** OpenMPI — added to the module env.

Non-admin alternative (user space): a conda-forge env, `conda install "openmpi=*=*cuda*" ucx`.

Verify:
```bash
ompi_info --all | grep -i cuda_support   # -> "true"
ompi_info | grep -i accelerator          # -> a cuda accelerator component
ucx_info -d | grep -i cuda               # -> cuda_copy, cuda_ipc
```

## Swap-in when available (localized)

In `include/tpx/halo/grid_halo_cuda.cuh`, `DeviceGridExchange::exchange` would, when
`MPIX_Query_cuda_support()` is true, pass `d_sendBuf_`/`d_recvBuf_` directly to `MPI_Isend`/`MPI_Irecv`
and drop the two `cudaMemcpy`s — a runtime-gated branch in one function, falling back to host-staging
otherwise. Worth it mainly for multi-GPU/multi-node scaling; on one shared GPU the win is modest.

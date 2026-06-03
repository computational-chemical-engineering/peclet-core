// Minimal CUDA-aware MPI check: pass DEVICE pointers to MPI_Send/Recv. With a CUDA-aware MPI this
// transfers device->device (cuda_copy/cuda_ipc) and verifies; with a non-CUDA-aware MPI it segfaults
// or corrupts. Build: mpic++ cuda_aware_mpi_check.cpp -I$CUDA/include -L$CUDA/lib64 -lcudart -o check
// Run:   mpirun -np 2 ./check     (single GPU: both ranks use device 0 -> exercises cuda_ipc/cuda_copy)
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <cstdio>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  cudaSetDevice(ndev > 0 ? rank % ndev : 0);

  if (rank == 0) {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("compile-time MPIX_CUDA_AWARE_SUPPORT = 1\n");
#endif
#ifdef OMPI_HAVE_MPI_EXT_CUDA
    printf("runtime  MPIX_Query_cuda_support()  = %d\n", MPIX_Query_cuda_support());
#endif
    fflush(stdout);
  }

  const int N = 4096;
  double* d = nullptr;
  cudaError_t ce = cudaMalloc(&d, N * sizeof(double));
  if (ce != cudaSuccess) { printf("rank %d cudaMalloc failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

  if (size >= 2 && rank == 0) {
    double* h = new double[N];
    for (int i = 0; i < N; ++i) h[i] = i * 1.5;
    cudaMemcpy(d, h, N * sizeof(double), cudaMemcpyHostToDevice);
    MPI_Send(d, N, MPI_DOUBLE, 1, 7, MPI_COMM_WORLD);     // DEVICE pointer
    printf("rank 0: sent %d doubles from device memory\n", N);
    delete[] h;
  } else if (size >= 2 && rank == 1) {
    MPI_Recv(d, N, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // DEVICE pointer
    double* h = new double[N];
    cudaMemcpy(h, d, N * sizeof(double), cudaMemcpyDeviceToHost);
    bool ok = true; int bad = -1;
    for (int i = 0; i < N; ++i) if (h[i] != i * 1.5) { ok = false; bad = i; break; }
    printf("rank 1: device recv %s%s\n", ok ? "CORRECT -- CUDA-aware MPI works" : "WRONG at idx ",
           ok ? "" : "");
    if (!ok) printf("   first mismatch idx=%d got=%g want=%g\n", bad, h[bad], bad * 1.5);
    delete[] h;
  } else if (size == 1) {
    printf("(run with -np 2 to exercise the device<->device path)\n");
  }
  cudaFree(d);
  MPI_Finalize();
  return 0;
}

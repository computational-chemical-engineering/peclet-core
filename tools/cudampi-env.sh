# Source this to use the locally-built CUDA-aware OpenMPI (UCX cuda_copy/cuda_ipc), e.g.
#   source ~/opt/cudampi-env.sh && mpirun -np 2 ./prog
# Built 2026-06 vs CUDA 13.2: OpenMPI 5.0.7 (static components) + UCX 1.20.1, both --with-cuda.
# Device-pointer MPI is verified working with the UCX pml (forced below). NB: MPIX_Query_cuda_support()
# reports 0 because OpenMPI's accelerator framework selects 'null', but UCX does the device transfer.
export CUDAMPI_HOME="$HOME/opt/openmpi-cuda"
export UCXDIR="$HOME/opt/ucx"
export PATH="$CUDAMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDAMPI_HOME/lib:$UCXDIR/lib:/usr/local/cuda-13.2/lib64:$LD_LIBRARY_PATH"
export OPAL_PREFIX="$CUDAMPI_HOME"
export OMPI_MCA_pml=ucx          # force the UCX pml -> CUDA-aware device transfers (else ob1 on 1 node)
export UCX_WARN_UNUSED_ENV_VARS=n

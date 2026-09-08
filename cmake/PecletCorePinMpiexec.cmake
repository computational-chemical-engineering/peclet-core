# Pin the MPI launcher to the MPI compiler's own prefix. Shared by the root CMakeLists.txt and
# python/CMakeLists.txt (both register mpirun ctests).
#
# FindMPI locates the compiler wrapper (mpicxx) deterministically, but searches MPIEXEC_EXECUTABLE
# on PATH — so a foreign launcher earlier on PATH (e.g. ParaView's bundled mpiexec) gets picked up
# while the binary is built against the system MPI. That mismatch is silent and nasty: every rank
# inits as a singleton (MPI_Comm_size==1), so `mpirun -n N` runs N independent serial processes and
# multi-rank tests "pass" without ever communicating (the Python tree carried exactly this until
# 2026-09-08: python_amr_np2 ran two singletons racing on one VTU file). The launcher next to mpicxx
# always belongs to the same MPI, on any system. Scheduler-launcher clusters (srun/aprun) pass
# -DPECLET_CORE_PIN_MPIEXEC=OFF and set MPIEXEC_EXECUTABLE themselves.
option(PECLET_CORE_PIN_MPIEXEC "Pin MPIEXEC_EXECUTABLE to the MPI compiler's prefix" ON)
if(PECLET_CORE_PIN_MPIEXEC AND MPI_CXX_COMPILER)
  get_filename_component(_mpi_bin "${MPI_CXX_COMPILER}" DIRECTORY)
  unset(_mpi_launcher CACHE)
  find_program(_mpi_launcher NAMES mpirun mpiexec HINTS "${_mpi_bin}" NO_DEFAULT_PATH)
  if(_mpi_launcher AND NOT _mpi_launcher STREQUAL "${MPIEXEC_EXECUTABLE}")
    message(STATUS "peclet-core: MPIEXEC_EXECUTABLE was '${MPIEXEC_EXECUTABLE}' — "
                   "pinning to '${_mpi_launcher}' (matches ${MPI_CXX_COMPILER})")
    set(MPIEXEC_EXECUTABLE "${_mpi_launcher}" CACHE FILEPATH "MPI launcher" FORCE)
  endif()
  unset(_mpi_launcher CACHE)
endif()

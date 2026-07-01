cmake_minimum_required(VERSION 3.24)
project(transport_core LANGUAGES CXX)

# --- Standard: C++20 host (see suite/docs/STYLE.md). Device/CUDA stays C++17-compatible. ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(PECLET_CORE_BUILD_TESTS "Build tests" ON)
option(PECLET_CORE_BUILD_BENCHMARKS "Build benchmarks" ON)
option(PECLET_CORE_ENABLE_MPI "Build the halo layer against MPI (OFF = single-rank no-MPI stub)" ON)

# --- Header-only core library ---
add_library(tpx_core INTERFACE)
add_library(tpx::core ALIAS tpx_core)
target_include_directories(tpx_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(tpx_core INTERFACE cxx_std_20)

# Optional: the morton spatial-index primitive if present as a sibling checkout.
set(_morton_inc "${CMAKE_CURRENT_SOURCE_DIR}/../morton/include")
if(EXISTS "${_morton_inc}/morton/morton.hpp")
  target_include_directories(tpx_core INTERFACE $<BUILD_INTERFACE:${_morton_inc}>)
  target_compile_definitions(tpx_core INTERFACE PECLET_CORE_HAVE_MORTON=1)
  message(STATUS "transport-core: morton found at ${_morton_inc}")
endif()

# --- Halo layer: against MPI (default), or a single-rank no-MPI stub (PECLET_CORE_ENABLE_MPI=OFF). One code
#     path either way -- see tpx/common/mpi.hpp + mpi_stub.hpp. PECLET_CORE_HALO_OK marks the layer available
#     (MPI tests/benchmarks still guard on MPI_FOUND; the no-MPI build runs single-rank). ---
set(PECLET_CORE_HALO_OK OFF)
if(PECLET_CORE_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if(MPI_FOUND)
    # Pin the launcher to the MPI compiler's own prefix. FindMPI locates the compiler
    # wrapper (mpicxx) deterministically, but searches MPIEXEC_EXECUTABLE on PATH — so a
    # foreign launcher earlier on PATH (e.g. ParaView's bundled mpiexec) gets picked up
    # while the binary is built against the system MPI. That mismatch is silent and nasty:
    # every rank inits as a singleton (MPI_Comm_size==1), so `mpirun -n N` runs N
    # independent serial processes and multi-rank tests "pass" without ever communicating.
    # The launcher next to mpicxx always belongs to the same MPI, on any system.
    # Scheduler-launcher clusters (srun/aprun) pass -DTPX_PIN_MPIEXEC=OFF and set
    # MPIEXEC_EXECUTABLE themselves.
    option(PECLET_CORE_PIN_MPIEXEC "Pin MPIEXEC_EXECUTABLE to the MPI compiler's prefix" ON)
    if(PECLET_CORE_PIN_MPIEXEC AND MPI_CXX_COMPILER)
      get_filename_component(_mpi_bin "${MPI_CXX_COMPILER}" DIRECTORY)
      unset(_mpi_launcher CACHE)
      find_program(_mpi_launcher NAMES mpirun mpiexec HINTS "${_mpi_bin}" NO_DEFAULT_PATH)
      if(_mpi_launcher AND NOT _mpi_launcher STREQUAL "${MPIEXEC_EXECUTABLE}")
        message(STATUS "transport-core: MPIEXEC_EXECUTABLE was '${MPIEXEC_EXECUTABLE}' — "
                       "pinning to '${_mpi_launcher}' (matches ${MPI_CXX_COMPILER})")
        set(MPIEXEC_EXECUTABLE "${_mpi_launcher}" CACHE FILEPATH "MPI launcher" FORCE)
      endif()
      unset(_mpi_launcher CACHE)
    endif()
    add_library(tpx_halo INTERFACE)
    add_library(tpx::halo ALIAS tpx_halo)
    target_link_libraries(tpx_halo INTERFACE tpx_core MPI::MPI_CXX)
    set(PECLET_CORE_HALO_OK ON)
    message(STATUS "transport-core: MPI found (${MPI_CXX_COMPILER}) — halo layer enabled")
  else()
    message(WARNING "transport-core: MPI not found — halo layer and its tests are disabled")
  endif()
else()
  add_library(tpx_halo INTERFACE)
  add_library(tpx::halo ALIAS tpx_halo)
  target_link_libraries(tpx_halo INTERFACE tpx_core)
  target_compile_definitions(tpx_halo INTERFACE PECLET_CORE_NO_MPI=1)
  set(PECLET_CORE_HALO_OK ON)
  message(STATUS "transport-core: MPI disabled — halo uses the single-rank no-MPI stub")
endif()

# --- Optional Kokkos: portable (CUDA/HIP/OpenMP) GPU-resident halo exchange ---
# (The legacy native-CUDA halo, grid_halo_cuda.cuh, was retired in favour of the Kokkos path below.)
# Consumed via find_package(Kokkos CONFIG); provide it with a cluster module or the suite's local
# install prefix (suite/tools/bootstrap_deps.sh) on CMAKE_PREFIX_PATH. Independent of the legacy
# native-CUDA path above — both can be built at once.
option(PECLET_CORE_ENABLE_KOKKOS "Build the portable Kokkos halo path (find_package(Kokkos))" OFF)
set(PECLET_CORE_HAVE_KOKKOS OFF)
if(PECLET_CORE_ENABLE_KOKKOS AND PECLET_CORE_HALO_OK)
  find_package(Kokkos CONFIG REQUIRED)
  set(PECLET_CORE_HAVE_KOKKOS ON)
  message(STATUS "transport-core: Kokkos ${Kokkos_VERSION} found (${Kokkos_DEVICES}) — "
                 "portable halo path enabled")
  # When morton is present, build it in Kokkos mode so MORTON_HD resolves to
  # KOKKOS_FUNCTION — the MortonIndexer (tpx/decomp/morton_indexer.hpp) is then
  # callable from device kernels on any backend.
  if(TARGET Kokkos::kokkos AND EXISTS "${_morton_inc}/morton/morton.hpp")
    target_link_libraries(tpx_core INTERFACE Kokkos::kokkos)
    target_compile_definitions(tpx_core INTERFACE MORTON_ENABLE_KOKKOS=1)
  endif()
endif()

if(PECLET_CORE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PECLET_CORE_BUILD_BENCHMARKS AND MPI_FOUND)
  add_subdirectory(benchmarks)
endif()

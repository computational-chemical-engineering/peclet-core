cmake_minimum_required(VERSION 3.24)
project(transport_core LANGUAGES CXX)

# --- Standard: C++20 host (see suite/docs/STYLE.md). Device/CUDA stays C++17-compatible. ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(TPX_BUILD_TESTS "Build tests" ON)
option(TPX_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TPX_ENABLE_MPI "Build the halo layer against MPI (OFF = single-rank no-MPI stub)" ON)

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
  target_compile_definitions(tpx_core INTERFACE TPX_HAVE_MORTON=1)
  message(STATUS "transport-core: morton found at ${_morton_inc}")
endif()

# --- Halo layer: against MPI (default), or a single-rank no-MPI stub (TPX_ENABLE_MPI=OFF). One code
#     path either way -- see tpx/common/mpi.hpp + mpi_stub.hpp. TPX_HALO_OK marks the layer available
#     (MPI tests/benchmarks still guard on MPI_FOUND; the no-MPI build runs single-rank). ---
set(TPX_HALO_OK OFF)
if(TPX_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if(MPI_FOUND)
    add_library(tpx_halo INTERFACE)
    add_library(tpx::halo ALIAS tpx_halo)
    target_link_libraries(tpx_halo INTERFACE tpx_core MPI::MPI_CXX)
    set(TPX_HALO_OK ON)
    message(STATUS "transport-core: MPI found (${MPI_CXX_COMPILER}) — halo layer enabled")
  else()
    message(WARNING "transport-core: MPI not found — halo layer and its tests are disabled")
  endif()
else()
  add_library(tpx_halo INTERFACE)
  add_library(tpx::halo ALIAS tpx_halo)
  target_link_libraries(tpx_halo INTERFACE tpx_core)
  target_compile_definitions(tpx_halo INTERFACE TPX_NO_MPI=1)
  set(TPX_HALO_OK ON)
  message(STATUS "transport-core: MPI disabled — halo uses the single-rank no-MPI stub")
endif()

# --- Optional Kokkos: portable (CUDA/HIP/OpenMP) GPU-resident halo exchange ---
# (The legacy native-CUDA halo, grid_halo_cuda.cuh, was retired in favour of the Kokkos path below.)
# Consumed via find_package(Kokkos CONFIG); provide it with a cluster module or the suite's local
# install prefix (suite/tools/bootstrap_deps.sh) on CMAKE_PREFIX_PATH. Independent of the legacy
# native-CUDA path above — both can be built at once.
option(TPX_ENABLE_KOKKOS "Build the portable Kokkos halo path (find_package(Kokkos))" OFF)
set(TPX_HAVE_KOKKOS OFF)
if(TPX_ENABLE_KOKKOS AND TPX_HALO_OK)
  find_package(Kokkos CONFIG REQUIRED)
  set(TPX_HAVE_KOKKOS ON)
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

if(TPX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TPX_BUILD_BENCHMARKS AND MPI_FOUND)
  add_subdirectory(benchmarks)
endif()

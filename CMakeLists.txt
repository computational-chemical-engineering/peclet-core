cmake_minimum_required(VERSION 3.24)
# --- Version: single-sourced from pyproject.toml (suite/docs/QUALITY_PLAN.md D4) ---
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _peclet_core_version_line REGEX "^version = \"")
string(REGEX MATCH "[0-9]+(\\.[0-9]+)*" PECLET_CORE_VERSION "${_peclet_core_version_line}")
if(NOT PECLET_CORE_VERSION)
  message(FATAL_ERROR "peclet-core: could not read `version = \"x.y.z\"` from pyproject.toml")
endif()
project(peclet_core VERSION ${PECLET_CORE_VERSION} LANGUAGES CXX)

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
add_library(peclet_core INTERFACE)
add_library(peclet::core ALIAS peclet_core)
target_include_directories(peclet_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(peclet_core INTERFACE cxx_std_20)

# Optional: the morton spatial-index primitive (github.com/computational-chemical-engineering/
# peclet-morton, header-only), used by decomp/morton_indexer.hpp only (the AMR octree that also
# needed it is the peclet-amr package since 2026-09-10). Auto-detected as the sibling checkout
# ../morton; point PECLET_CORE_MORTON_DIR elsewhere to override (or at a non-existent path to build
# WITHOUT it — the guarded test then reports "Not Run (skipped)", never Passed).
set(PECLET_CORE_MORTON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../morton" CACHE PATH
    "morton checkout (headers under <dir>/include/morton); unset/absent = build without morton")
set(_morton_inc "${PECLET_CORE_MORTON_DIR}/include")
if(EXISTS "${_morton_inc}/morton/morton.hpp")
  target_include_directories(peclet_core INTERFACE $<BUILD_INTERFACE:${_morton_inc}>)
  target_compile_definitions(peclet_core INTERFACE PECLET_CORE_HAVE_MORTON=1)
  message(STATUS "peclet-core: morton found at ${_morton_inc}")
else()
  message(STATUS "peclet-core: morton NOT found at ${_morton_inc} — the morton_indexer test will be "
                 "reported as skipped (set PECLET_CORE_MORTON_DIR)")
endif()

# --- Halo layer: against MPI (default), or a single-rank no-MPI stub (PECLET_CORE_ENABLE_MPI=OFF). One code
#     path either way -- see peclet/core/common/mpi.hpp + mpi_stub.hpp. PECLET_CORE_HALO_OK marks the layer available
#     (MPI tests/benchmarks still guard on MPI_FOUND; the no-MPI build runs single-rank). ---
set(PECLET_CORE_HALO_OK OFF)
if(PECLET_CORE_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if(MPI_FOUND)
    # Pin MPIEXEC_EXECUTABLE to the launcher next to mpicxx (cmake/PecletCorePinMpiexec.cmake):
    # a foreign launcher on PATH silently runs every rank as a singleton.
    include(cmake/PecletCorePinMpiexec.cmake)
    add_library(peclet_halo INTERFACE)
    add_library(peclet::halo ALIAS peclet_halo)
    target_link_libraries(peclet_halo INTERFACE peclet_core MPI::MPI_CXX)
    set(PECLET_CORE_HALO_OK ON)
    message(STATUS "peclet-core: MPI found (${MPI_CXX_COMPILER}) — halo layer enabled")
  else()
    message(WARNING "peclet-core: MPI not found — halo layer and its tests are disabled")
  endif()
else()
  add_library(peclet_halo INTERFACE)
  add_library(peclet::halo ALIAS peclet_halo)
  target_link_libraries(peclet_halo INTERFACE peclet_core)
  target_compile_definitions(peclet_halo INTERFACE PECLET_CORE_NO_MPI=1)
  set(PECLET_CORE_HALO_OK ON)
  message(STATUS "peclet-core: MPI disabled — halo uses the single-rank no-MPI stub")
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
  message(STATUS "peclet-core: Kokkos ${Kokkos_VERSION} found (${Kokkos_DEVICES}) — "
                 "portable halo path enabled")
  # When morton is present, build it in Kokkos mode so MORTON_HD resolves to
  # KOKKOS_FUNCTION — the MortonIndexer (peclet/core/decomp/morton_indexer.hpp) is then
  # callable from device kernels on any backend.
  if(TARGET Kokkos::kokkos AND EXISTS "${_morton_inc}/morton/morton.hpp")
    target_link_libraries(peclet_core INTERFACE Kokkos::kokkos)
    target_compile_definitions(peclet_core INTERFACE MORTON_ENABLE_KOKKOS=1)
  endif()
endif()

# --- Install + package config: `find_package(peclet-core CONFIG)` yields the header-only
#     peclet::core / peclet::halo (suite/docs/STYLE.md). The morton include is BUILD_INTERFACE-only,
#     so an installed consumer puts morton on its own include path. ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set_target_properties(peclet_core PROPERTIES EXPORT_NAME core)
set(_peclet_core_export_targets peclet_core)
if(TARGET peclet_halo)
  set_target_properties(peclet_halo PROPERTIES EXPORT_NAME halo)
  list(APPEND _peclet_core_export_targets peclet_halo)
endif()
set(PECLET_CORE_CONFIG_NEED_MPI OFF)
if(PECLET_CORE_ENABLE_MPI AND MPI_FOUND)
  set(PECLET_CORE_CONFIG_NEED_MPI ON)
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS ${_peclet_core_export_targets} EXPORT peclet-coreTargets)
install(EXPORT peclet-coreTargets NAMESPACE peclet::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/peclet-core)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/peclet-coreConfigVersion.cmake"
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file(cmake/peclet-coreConfig.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/peclet-coreConfig.cmake"
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/peclet-core)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/peclet-coreConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/peclet-coreConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/peclet-core)

# Test registration helpers (SKIP_RETURN_CODE 77 + the mpi / np8 / bench labels) — one place for
# tests/ and benchmarks/. See the module header and suite/docs/QUALITY_PLAN.md §3.D.
include(cmake/PecletCoreTest.cmake)

if(PECLET_CORE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PECLET_CORE_BUILD_BENCHMARKS AND MPI_FOUND)
  enable_testing()  # the benchmarks are registered as ctests labelled `bench` (ctest -L bench to run)
  add_subdirectory(benchmarks)
endif()

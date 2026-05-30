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

# --- Header-only core library ---
add_library(tpx_core INTERFACE)
add_library(tpx::core ALIAS tpx_core)
target_include_directories(tpx_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(tpx_core INTERFACE cxx_std_20)

# Optional: morton_arithmetic (spatial-index primitive) if present as a sibling checkout.
set(_morton_inc "${CMAKE_CURRENT_SOURCE_DIR}/../morton_arithmetic/include")
if(EXISTS "${_morton_inc}/morton/morton.hpp")
  target_include_directories(tpx_core INTERFACE $<BUILD_INTERFACE:${_morton_inc}>)
  target_compile_definitions(tpx_core INTERFACE TPX_HAVE_MORTON=1)
  message(STATUS "transport-core: morton_arithmetic found at ${_morton_inc}")
endif()

# --- MPI (required for the halo layer; tests/benchmarks that need it guard on MPI_FOUND) ---
find_package(MPI COMPONENTS CXX)
if(MPI_FOUND)
  add_library(tpx_halo INTERFACE)
  add_library(tpx::halo ALIAS tpx_halo)
  target_link_libraries(tpx_halo INTERFACE tpx_core MPI::MPI_CXX)
  message(STATUS "transport-core: MPI found (${MPI_CXX_COMPILER}) — halo layer enabled")
else()
  message(WARNING "transport-core: MPI not found — halo layer and its tests are disabled")
endif()

# --- Optional CUDA: GPU-resident (host-staged) halo exchange ---
if(NOT CMAKE_CUDA_COMPILER)
  find_program(_tpx_nvcc nvcc
    PATHS /usr/local/cuda/bin /usr/local/cuda-13.2/bin /usr/local/cuda-13/bin NO_DEFAULT_PATH)
  if(_tpx_nvcc)
    set(CMAKE_CUDA_COMPILER ${_tpx_nvcc} CACHE FILEPATH "Path to nvcc" FORCE)
  endif()
endif()
include(CheckLanguage)
check_language(CUDA)
set(TPX_HAVE_CUDA OFF)
if(CMAKE_CUDA_COMPILER AND MPI_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # enable_language(CUDA) defaults this to an old arch (e.g. 75); force the real device arch so
  # kernels actually launch (the RTX 5080 is sm_120). Override on the command line if needed.
  if(NOT DEFINED TPX_CUDA_ARCH)
    set(TPX_CUDA_ARCH native)
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${TPX_CUDA_ARCH} CACHE STRING "" FORCE)
  set(TPX_HAVE_CUDA ON)
  message(STATUS "transport-core: CUDA enabled (${CMAKE_CUDA_COMPILER}, arch "
                 "${CMAKE_CUDA_ARCHITECTURES}) — GPU halo tests on")
else()
  message(STATUS "transport-core: CUDA not enabled — GPU halo path skipped")
endif()

if(TPX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TPX_BUILD_BENCHMARKS AND MPI_FOUND)
  add_subdirectory(benchmarks)
endif()

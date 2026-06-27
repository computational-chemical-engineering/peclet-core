# Python bindings for the transport-core Lagrangian halo (migration + ghosts).
# Standalone (host C++ + MPI + pybind11 + header-only transport-core); no CUDA.
#   cmake -S python -B python/build && cmake --build python/build -j
#   PYTHONPATH=python/build mpirun -np 4 python3 python/test_tpx_mpi.py
# Run with a Python that has mpi4py built against the same MPI.
cmake_minimum_required(VERSION 3.16)
project(tpx_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(pybind11 REQUIRED)

pybind11_add_module(tpx_mpi tpx_mpi.cpp)
target_include_directories(tpx_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(tpx_mpi PRIVATE MPI::MPI_CXX)

# AMR octree bindings (tpx_amr): requires the morton sibling checkout, since tpx::amr is guarded by
# TPX_HAVE_MORTON. Built only when ../../morton/include/morton/morton.hpp is present (a local/opt-in
# target — CI builds transport-core without morton). C++20 (the AMR headers are C++20 host).
#   PYTHONPATH=python/build mpirun -np 4 python3 python/test_tpx_amr.py
# tpx_amr exposes the device (Kokkos) AMR flow, so it needs both the morton sibling and a Kokkos
# install (point CMAKE_PREFIX_PATH at a backend prefix from ../tools/bootstrap_deps.sh, e.g.
# host-serial / host-openmp / nvidia-cuda). morton is built in Kokkos mode (MORTON_ENABLE_KOKKOS)
# so MORTON_HD resolves to KOKKOS_FUNCTION (device-callable). Skipped if either is missing.
set(_morton_inc "${CMAKE_CURRENT_SOURCE_DIR}/../../morton/include")
find_package(Kokkos CONFIG QUIET)
if(EXISTS "${_morton_inc}/morton/morton.hpp" AND Kokkos_FOUND)
  pybind11_add_module(tpx_amr tpx_amr.cpp)
  target_include_directories(tpx_amr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${_morton_inc})
  target_compile_definitions(tpx_amr PRIVATE TPX_HAVE_MORTON=1 MORTON_ENABLE_KOKKOS=1)
  target_compile_features(tpx_amr PRIVATE cxx_std_20)
  target_link_libraries(tpx_amr PRIVATE MPI::MPI_CXX Kokkos::kokkos)
  message(STATUS "tpx_amr: morton + Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}) — building device AMR bindings")
elseif(NOT Kokkos_FOUND)
  message(STATUS "tpx_amr: Kokkos not found (set CMAKE_PREFIX_PATH to a backend prefix) — skipping AMR bindings")
else()
  message(STATUS "tpx_amr: morton sibling not found — skipping AMR Python bindings")
endif()

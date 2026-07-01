# Python bindings for the transport-core Lagrangian halo (migration + ghosts) + the AMR octree.
# Standalone (host C++ + MPI + nanobind + header-only transport-core); nanobind is found via the
# active interpreter (cmake/SuiteNanobind.cmake at the suite root).
#   cmake -S python -B python/build -DCMAKE_PREFIX_PATH=../extern/install/host-openmp
#   cmake --build python/build -j
#   PYTHONPATH=python/build mpirun -np 4 python3 python/test_tpx_mpi.py
# Run with a Python that has mpi4py built against the same MPI.
cmake_minimum_required(VERSION 3.24)
project(tpx_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(SuiteNanobind)
suite_require_nanobind()

# tpx_mpi: host-only Lagrangian halo (no Kokkos). Exposed as the `peclet.core.mpi` submodule (the
# nanobind module leaf name is `mpi`; NB_MODULE(mpi, ...) in tpx_mpi.cpp), so the CMake target keeps its
# descriptive name but the built extension is `mpi.*.so` installed under peclet/core/.
nanobind_add_module(tpx_mpi NB_STATIC tpx_mpi.cpp)
# -> peclet.core.mpi (NB_MODULE(mpi)). The peclet/core package __init__.py is kept as a plain file under
# packaging/ (OUTSIDE any importable peclet/ dir, so an incomplete source package can never shadow the
# installed one) and staged into <build>/peclet/core so `PYTHONPATH=<build> python …` works in dev too.
set_target_properties(tpx_mpi PROPERTIES OUTPUT_NAME mpi
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/core")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/core_init.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/core/__init__.py COPYONLY)
target_include_directories(tpx_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(tpx_mpi PRIVATE MPI::MPI_CXX)
# `pip install .` (scikit-build-core, via ../pyproject.toml with cmake.source-dir=python) installs the
# built modules into the `peclet.core` package. Inert for a plain `cmake --build`, so the developer
# workflow is unchanged.
if(DEFINED SKBUILD)
  install(TARGETS tpx_mpi LIBRARY DESTINATION peclet/core COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/core_init.py
          DESTINATION peclet/core RENAME __init__.py COMPONENT python)
endif()

# AMR octree bindings (tpx_amr): requires the morton sibling checkout, since peclet::core::amr is guarded by
# PECLET_CORE_HAVE_MORTON. Built only when ../../morton/include/morton/morton.hpp is present (a local/opt-in
# target — CI builds transport-core without morton). C++20 (the AMR headers are C++20 host).
#   PYTHONPATH=python/build mpirun -np 4 python3 python/test_tpx_amr.py
# tpx_amr exposes the device (Kokkos) AMR flow, so it needs both the morton sibling and a Kokkos
# install (point CMAKE_PREFIX_PATH at a backend prefix from ../tools/bootstrap_deps.sh, e.g.
# host-serial / host-openmp / nvidia-cuda). morton is built in Kokkos mode (MORTON_ENABLE_KOKKOS)
# so MORTON_HD resolves to KOKKOS_FUNCTION (device-callable). Skipped if either is missing.
set(_morton_inc "${CMAKE_CURRENT_SOURCE_DIR}/../../morton/include")
find_package(Kokkos CONFIG QUIET)
if(EXISTS "${_morton_inc}/morton/morton.hpp" AND Kokkos_FOUND)
  # NOMINSIZE: nanobind's default -Os is rejected by nvcc ("'s': expected a number") since the Kokkos
  # AMR device sources compile as CXX through the launch compiler.
  nanobind_add_module(tpx_amr NB_STATIC NOMINSIZE tpx_amr.cpp)
  set_target_properties(tpx_amr PROPERTIES OUTPUT_NAME amr   # -> peclet.core.amr (NB_MODULE(amr, ...))
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/core")
  target_include_directories(tpx_amr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${_morton_inc})
  target_compile_definitions(tpx_amr PRIVATE PECLET_CORE_HAVE_MORTON=1 MORTON_ENABLE_KOKKOS=1)
  target_compile_features(tpx_amr PRIVATE cxx_std_20)
  target_link_libraries(tpx_amr PRIVATE MPI::MPI_CXX Kokkos::kokkos)
  if(DEFINED SKBUILD)
    install(TARGETS tpx_amr LIBRARY DESTINATION peclet/core COMPONENT python)
  endif()
  message(STATUS "tpx_amr: morton + Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}) — building device AMR bindings")
elseif(NOT Kokkos_FOUND)
  message(STATUS "tpx_amr: Kokkos not found (set CMAKE_PREFIX_PATH to a backend prefix) — skipping AMR bindings")
else()
  message(STATUS "tpx_amr: morton sibling not found — skipping AMR Python bindings")
endif()

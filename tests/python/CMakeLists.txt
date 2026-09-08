# Test for the Kokkos-View <-> nanobind-ndarray zero-copy bridge
# (include/peclet/core/python/ndarray_interop.hpp). Normally registered by python/CMakeLists.txt
# (add_subdirectory, sharing its Kokkos + nanobind resolution); still configurable standalone
# against a bootstrapped Kokkos prefix:
#
#   cmake -S tests/python -B build_pyinterop \
#     -DCMAKE_PREFIX_PATH=../extern/install/host-openmp \
#     -DPython_EXECUTABLE=$(which python)
#   cmake --build build_pyinterop -j
#   ctest --test-dir build_pyinterop --output-on-failure
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.24)
  project(peclet_core_python_interop_test LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()

  find_package(Kokkos CONFIG REQUIRED)

  # Shared nanobind helper: the umbrella's copy in a suite checkout, core's vendored copy otherwise.
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake"
                                "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
  include(SuiteNanobind)
  suite_require_nanobind()
  enable_testing()
endif()

# NOMINSIZE: nanobind's default -Os is rejected by nvcc on the CUDA backend (Kokkos device sources
# compile as CXX through the launch compiler).
nanobind_add_module(interop_test_module NB_STATIC NOMINSIZE interop_test_module.cpp)
target_include_directories(interop_test_module PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_libraries(interop_test_module PRIVATE Kokkos::kokkos)

add_test(NAME ndarray_interop
         COMMAND "${Python_EXECUTABLE}" -m pytest -q -p no:cacheprovider
                 "${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_interop.py")
set_tests_properties(ndarray_interop PROPERTIES LABELS "python"
  ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:interop_test_module>")

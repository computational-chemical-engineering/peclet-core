cmake_minimum_required(VERSION 3.24)
project(tpx_python_interop_test LANGUAGES CXX)

# Self-contained test for the Kokkos-View <-> nanobind-ndarray zero-copy bridge
# (include/tpx/python/ndarray_interop.hpp). Its own find_package project, like sdflow/tests/kokkos:
# build against a bootstrapped Kokkos prefix on CMAKE_PREFIX_PATH, then run the pytest:
#
#   cmake -S tests/python -B build_pyinterop \
#     -DCMAKE_PREFIX_PATH=../extern/install/host-openmp \
#     -DPython_EXECUTABLE=$(which python)
#   cmake --build build_pyinterop -j
#   ctest --test-dir build_pyinterop --output-on-failure

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)

# Shared nanobind helper (cmake/SuiteNanobind.cmake at the suite root).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
include(SuiteNanobind)
suite_require_nanobind()

nanobind_add_module(interop_test_module NB_STATIC interop_test_module.cpp)
target_include_directories(interop_test_module PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_libraries(interop_test_module PRIVATE Kokkos::kokkos)

enable_testing()
add_test(NAME ndarray_interop
         COMMAND "${Python_EXECUTABLE}" -m pytest -q
                 "${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_interop.py")
set_tests_properties(ndarray_interop PROPERTIES
  ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:interop_test_module>")

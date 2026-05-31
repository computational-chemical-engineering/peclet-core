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

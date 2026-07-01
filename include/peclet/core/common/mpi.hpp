// core -- MPI include shim. Lets the whole stack compile with or without MPI from ONE code
// path. Include this instead of <mpi.h>. With MPI (the default) it is transparent; define PECLET_CORE_NO_MPI
// (CMake: -DPECLET_CORE_ENABLE_MPI=OFF) to build the single-rank, no-MPI variant against peclet/core/common/mpi_stub.hpp.
#pragma once

#if defined(PECLET_CORE_NO_MPI)
#include "peclet/core/common/mpi_stub.hpp"
#else
#include <mpi.h>
#endif

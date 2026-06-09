// transport-core -- MPI include shim. Lets the whole stack compile with or without MPI from ONE code
// path. Include this instead of <mpi.h>. With MPI (the default) it is transparent; define TPX_NO_MPI
// (CMake: -DTPX_ENABLE_MPI=OFF) to build the single-rank, no-MPI variant against tpx/common/mpi_stub.hpp.
#pragma once

#if defined(TPX_NO_MPI)
#include "tpx/common/mpi_stub.hpp"
#else
#include <mpi.h>
#endif

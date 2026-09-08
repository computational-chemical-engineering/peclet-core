// The skip protocol for tests launched under mpirun (see tests/test_util.hpp kSkipExitCode and
// cmake/PecletCoreTest.cmake SKIP_RETURN_CODE 77).
//
// A rank that exits non-zero WITHOUT having initialised MPI is an abnormal termination to the
// launcher: Open MPI 5's prterun then tears the job down and returns 1 — not the rank's 77 — whenever
// the ranks' exits are not simultaneous (reproduced 2 times in 108 launches under `ctest -j8`), and
// ctest reports Failed instead of Skipped. So a skipped MPI test goes through a collective
// MPI_Init / MPI_Finalize first: every rank then terminates normally, together, and the launcher
// forwards the 77.
#ifndef PECLET_CORE_TEST_SKIP_MPI_HPP
#define PECLET_CORE_TEST_SKIP_MPI_HPP

#include <mpi.h>

#include <cstdio>

#include "test_util.hpp"

namespace peclet::core::test {
inline int skipMpiTest(int argc, char** argv, const char* what) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) std::printf("PECLET_CORE_HAVE_MORTON not set — skipping %s\n", what);
  MPI_Finalize();
  return kSkipExitCode;
}
}  // namespace peclet::core::test

#endif  // PECLET_CORE_TEST_SKIP_MPI_HPP

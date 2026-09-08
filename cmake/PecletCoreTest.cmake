# ---- Test registration: ONE place for the skip protocol and the labels (suite/docs/QUALITY_PLAN.md §3.D).
#  * SKIP_RETURN_CODE 77: a binary that cannot run in this configuration (no morton sibling, ...)
#    exits 77 (tests/test_util.hpp kSkipExitCode) and ctest reports it "Not Run (skipped)" — never
#    Passed. `ctest -N` therefore always lists the full battery, whatever the configuration.
#  * LABELS: `mpi` on every mpirun test; `np8` on the 8-rank instances (CI's hosted runners have
#    4 cores — `ctest -LE np8` there, np=8 stays a local gate); `bench` on benchmarks and measurement
#    studies (`ctest -LE bench` by default; they print tables and have no correctness gate).
function(peclet_core_add_test name target)
  cmake_parse_arguments(_t "" "WORKING_DIRECTORY;LABELS" "ARGS" ${ARGN})
  if(_t_WORKING_DIRECTORY)
    add_test(NAME ${name} COMMAND ${target} ${_t_ARGS} WORKING_DIRECTORY ${_t_WORKING_DIRECTORY})
  else()
    add_test(NAME ${name} COMMAND ${target} ${_t_ARGS})
  endif()
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 LABELS "${_t_LABELS}")
endfunction()

function(peclet_core_add_mpi_test name target np)
  cmake_parse_arguments(_t "" "LABELS" "ARGS" ${ARGN})
  add_test(NAME ${name}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS} ${_t_ARGS})
  set(_labels mpi)
  if(np GREATER 4)
    list(APPEND _labels np8)
  endif()
  if(_t_LABELS)
    list(APPEND _labels ${_t_LABELS})
  endif()
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 LABELS "${_labels}")
endfunction()

# Benchmarks are registered as ctests with the label `bench` so `ctest -N` lists them and
# `ctest -L bench` runs them at a small size; they are excluded from the correctness battery
# (`ctest -LE bench`) because they print timings and gate nothing.
add_executable(bench_halo bench_halo.cpp)
target_link_libraries(bench_halo PRIVATE peclet_halo)
peclet_core_add_mpi_test(bench_halo bench_halo 2 LABELS bench ARGS 16 1 20)   # per-rank cells, ghost, iters

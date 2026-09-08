# Benchmarks are registered as ctests with the label `bench` so `ctest -N` lists them and
# `ctest -L bench` runs them at a small size; they are excluded from the correctness battery
# (`ctest -LE bench`) because they print timings and gate nothing.
add_executable(bench_halo bench_halo.cpp)
target_link_libraries(bench_halo PRIVATE peclet_halo)
peclet_core_add_mpi_test(bench_halo bench_halo 2 LABELS bench ARGS 16 1 20)   # per-rank cells, ghost, iters

# AMR flow / Poisson host-vs-device benchmark (Kokkos + morton). Inert without them.
if(PECLET_CORE_HAVE_KOKKOS)
  add_executable(bench_amr_flow bench_amr_flow.cpp)
  target_link_libraries(bench_amr_flow PRIVATE peclet_core MPI::MPI_CXX)
  peclet_core_add_test(bench_amr_flow bench_amr_flow LABELS bench ARGS 4 2 3)   # Lmax_device, steps, Lmax_host
endif()

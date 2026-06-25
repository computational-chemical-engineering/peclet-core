add_executable(bench_halo bench_halo.cpp)
target_link_libraries(bench_halo PRIVATE tpx_halo)

# AMR flow / Poisson host-vs-device benchmark (Kokkos + morton). Inert without them.
if(TPX_HAVE_KOKKOS)
  add_executable(bench_amr_flow bench_amr_flow.cpp)
  target_link_libraries(bench_amr_flow PRIVATE tpx_core)
endif()

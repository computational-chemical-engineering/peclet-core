# Serial tests (no MPI).
add_executable(test_decomposition test_decomposition.cpp)
target_link_libraries(test_decomposition PRIVATE tpx_core)
target_include_directories(test_decomposition PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME decomposition COMMAND test_decomposition)

# Morton/Z-order cell indexer (inert unless TPX_HAVE_MORTON — set on tpx_core when
# the morton sibling checkout is present).
add_executable(test_morton_indexer test_morton_indexer.cpp)
target_link_libraries(test_morton_indexer PRIVATE tpx_core)
target_include_directories(test_morton_indexer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME morton_indexer COMMAND test_morton_indexer)

# Per-block adaptive octree (tpx::amr::BlockOctree). Inert unless TPX_HAVE_MORTON;
# cross-checks against the morton_octree::Octree reference oracle when the sibling
# octree project is present (its header is not on tpx_core's include path).
add_executable(test_block_octree test_block_octree.cpp)
target_link_libraries(test_block_octree PRIVATE tpx_core)
target_include_directories(test_block_octree PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set(_morton_octree_inc "${CMAKE_CURRENT_SOURCE_DIR}/../../morton/octree/include")
if(EXISTS "${_morton_octree_inc}/morton_octree/octree.hpp")
  target_include_directories(test_block_octree PRIVATE ${_morton_octree_inc})
endif()
add_test(NAME block_octree COMMAND test_block_octree)

# AMR leaf fields + VTU (UnstructuredGrid) visualization output.
add_executable(test_amr_vtu test_amr_vtu.cpp)
target_link_libraries(test_amr_vtu PRIVATE tpx_core)
target_include_directories(test_amr_vtu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_vtu COMMAND test_amr_vtu)

# SDF-driven refinement (refineToSdf) over the shared tpx::geom SDF.
add_executable(test_amr_sdf test_amr_sdf.cpp)
target_link_libraries(test_amr_sdf PRIVATE tpx_core)
target_include_directories(test_amr_sdf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_sdf COMMAND test_amr_sdf)

# Cell-centered FV Poisson on the octree + geometric multigrid.
add_executable(test_amr_poisson test_amr_poisson.cpp)
target_link_libraries(test_amr_poisson PRIVATE tpx_core)
target_include_directories(test_amr_poisson PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_poisson COMMAND test_amr_poisson)

# Quadratic coarse-fine flux + refluxing (2nd-order C/F interface).
add_executable(test_amr_cf_quadratic test_amr_cf_quadratic.cpp)
target_link_libraries(test_amr_cf_quadratic PRIVATE tpx_core)
target_include_directories(test_amr_cf_quadratic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_cf_quadratic COMMAND test_amr_cf_quadratic)

# Cut-cell openness folded into operator + MG coarsening (with quadratic C/F flux).
add_executable(test_amr_openness test_amr_openness.cpp)
target_link_libraries(test_amr_openness PRIVATE tpx_core)
target_include_directories(test_amr_openness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_openness COMMAND test_amr_openness)

# Cut-cell Dirichlet sub-cell BC (port of sdflow's xi-polynomial scheme) + kappa.
add_executable(test_amr_cut_cell test_amr_cut_cell.cpp)
target_link_libraries(test_amr_cut_cell PRIVATE tpx_core)
target_include_directories(test_amr_cut_cell PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_cut_cell COMMAND test_amr_cut_cell)

# Collocated Stokes momentum+pressure step (cut-cell IBM momentum + openness projection).
add_executable(test_amr_flow test_amr_flow.cpp)
target_link_libraries(test_amr_flow PRIVATE tpx_core)
target_include_directories(test_amr_flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_flow COMMAND test_amr_flow)

# Stokes sphere-array drag vs Zick & Homsy (sdflow's ground truth): convergence trend.
add_executable(test_amr_drag test_amr_drag.cpp)
target_link_libraries(test_amr_drag PRIVATE tpx_core)
target_include_directories(test_amr_drag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_drag COMMAND test_amr_drag)

# Barnes-Hut particle-tree mode over the octree.
add_executable(test_amr_barnes_hut test_amr_barnes_hut.cpp)
target_link_libraries(test_amr_barnes_hut PRIVATE tpx_core)
target_include_directories(test_amr_barnes_hut PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_barnes_hut COMMAND test_amr_barnes_hut)

# Explicit FV scalar transport (advection-diffusion) on the octree.
add_executable(test_amr_transport test_amr_transport.cpp)
target_link_libraries(test_amr_transport PRIVATE tpx_core)
target_include_directories(test_amr_transport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME amr_transport COMMAND test_amr_transport)

# Device-resident block-octree queries (Kokkos): device locate / face-neighbour
# must match the host BlockOctree. No MPI needed; linking tpx_core pulls in Kokkos
# + the morton include (+ MORTON_ENABLE_KOKKOS) under a Kokkos build.
if(TPX_HAVE_KOKKOS)
  add_executable(test_block_octree_kokkos test_block_octree_kokkos.cpp)
  target_link_libraries(test_block_octree_kokkos PRIVATE tpx_core)
  target_include_directories(test_block_octree_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME block_octree_kokkos COMMAND test_block_octree_kokkos)

  add_executable(test_amr_device_poisson_kokkos test_amr_device_poisson_kokkos.cpp)
  target_link_libraries(test_amr_device_poisson_kokkos PRIVATE tpx_core)
  target_include_directories(test_amr_device_poisson_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME amr_device_poisson_kokkos COMMAND test_amr_device_poisson_kokkos)

  add_executable(test_amr_device_multigrid_kokkos test_amr_device_multigrid_kokkos.cpp)
  target_link_libraries(test_amr_device_multigrid_kokkos PRIVATE tpx_core)
  target_include_directories(test_amr_device_multigrid_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME amr_device_multigrid_kokkos COMMAND test_amr_device_multigrid_kokkos)

  add_executable(test_amr_device_openness_kokkos test_amr_device_openness_kokkos.cpp)
  target_link_libraries(test_amr_device_openness_kokkos PRIVATE tpx_core)
  target_include_directories(test_amr_device_openness_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME amr_device_openness_kokkos COMMAND test_amr_device_openness_kokkos)

  add_executable(test_amr_kappa_restrict_kokkos test_amr_kappa_restrict_kokkos.cpp)
  target_link_libraries(test_amr_kappa_restrict_kokkos PRIVATE tpx_core)
  target_include_directories(test_amr_kappa_restrict_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME amr_kappa_restrict_kokkos COMMAND test_amr_kappa_restrict_kokkos)
endif()

add_executable(test_sdf test_sdf.cpp)
target_link_libraries(test_sdf PRIVATE tpx_core)
target_include_directories(test_sdf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME sdf COMMAND test_sdf)

add_executable(test_vti test_vti.cpp)
target_link_libraries(test_vti PRIVATE tpx_core)
target_include_directories(test_vti PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME vti COMMAND test_vti)

# MPI tests: run across several rank counts.
if(MPI_FOUND)
  add_executable(test_grid_halo test_grid_halo.cpp)
  target_link_libraries(test_grid_halo PRIVATE tpx_halo)
  target_include_directories(test_grid_halo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_particle_migration test_particle_migration.cpp)
  target_link_libraries(test_particle_migration PRIVATE tpx_halo)
  target_include_directories(test_particle_migration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_diffusion test_diffusion.cpp)
  target_link_libraries(test_diffusion PRIVATE tpx_halo)
  target_include_directories(test_diffusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_diffusion_sdf test_diffusion_sdf.cpp)
  target_link_libraries(test_diffusion_sdf PRIVATE tpx_halo)
  target_include_directories(test_diffusion_sdf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_ghost_particles_mpi test_ghost_particles_mpi.cpp)
  target_link_libraries(test_ghost_particles_mpi PRIVATE tpx_halo)
  target_include_directories(test_ghost_particles_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_particle_halo_mpi test_particle_halo_mpi.cpp)
  target_link_libraries(test_particle_halo_mpi PRIVATE tpx_halo)
  target_include_directories(test_particle_halo_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_mpi test_amr_distributed_mpi.cpp)
  target_link_libraries(test_amr_distributed_mpi PRIVATE tpx_halo)
  target_include_directories(test_amr_distributed_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_poisson_mpi test_amr_distributed_poisson_mpi.cpp)
  target_link_libraries(test_amr_distributed_poisson_mpi PRIVATE tpx_halo)
  target_include_directories(test_amr_distributed_poisson_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_mg_mpi test_amr_distributed_mg_mpi.cpp)
  target_link_libraries(test_amr_distributed_mg_mpi PRIVATE tpx_halo)
  target_include_directories(test_amr_distributed_mg_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_fv_mpi test_amr_distributed_fv_mpi.cpp)
  target_link_libraries(test_amr_distributed_fv_mpi PRIVATE tpx_halo)
  target_include_directories(test_amr_distributed_fv_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_graded_mg_mpi test_amr_distributed_graded_mg_mpi.cpp)
  target_link_libraries(test_amr_distributed_graded_mg_mpi PRIVATE tpx_halo)
  target_include_directories(test_amr_distributed_graded_mg_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_openness_mpi test_amr_distributed_openness_mpi.cpp)
  target_link_libraries(test_amr_distributed_openness_mpi PRIVATE tpx_halo)
  target_include_directories(test_amr_distributed_openness_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  foreach(np IN ITEMS 1 2 4 8)
    add_test(NAME grid_halo_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_grid_halo> ${MPIEXEC_POSTFLAGS})
    add_test(NAME particle_migration_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_particle_migration> ${MPIEXEC_POSTFLAGS})
    add_test(NAME diffusion_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_diffusion> ${MPIEXEC_POSTFLAGS})
    add_test(NAME diffusion_sdf_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_diffusion_sdf> ${MPIEXEC_POSTFLAGS})
    add_test(NAME ghost_particles_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_ghost_particles_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME particle_halo_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_particle_halo_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME amr_distributed_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_amr_distributed_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME amr_distributed_poisson_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_amr_distributed_poisson_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME amr_distributed_mg_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_amr_distributed_mg_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME amr_distributed_fv_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_amr_distributed_fv_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME amr_distributed_graded_mg_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_amr_distributed_graded_mg_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME amr_distributed_openness_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_amr_distributed_openness_mpi> ${MPIEXEC_POSTFLAGS})
  endforeach()

  # Portable GPU-resident halo (Kokkos): device result must match the CPU path, on whatever backend
  # Kokkos was built for (CUDA / HIP / OpenMP). Compiled as CXX — Kokkos routes it to the device
  # compiler via its launch compiler (so this is a .cpp, not a .cu).
  if(TPX_HAVE_KOKKOS)
    add_executable(test_grid_halo_kokkos test_grid_halo_kokkos.cpp)
    target_include_directories(test_grid_halo_kokkos PRIVATE
      ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_grid_halo_kokkos PRIVATE Kokkos::kokkos MPI::MPI_CXX)

    add_executable(test_particle_halo_kokkos test_particle_halo_kokkos.cpp)
    target_include_directories(test_particle_halo_kokkos PRIVATE
      ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_particle_halo_kokkos PRIVATE Kokkos::kokkos MPI::MPI_CXX)

    foreach(np IN ITEMS 1 2 4)
      add_test(NAME grid_halo_kokkos_np${np}
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                       ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_grid_halo_kokkos> ${MPIEXEC_POSTFLAGS})
      add_test(NAME particle_halo_kokkos_np${np}
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                       ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_particle_halo_kokkos> ${MPIEXEC_POSTFLAGS})
    endforeach()
  endif()
endif()

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

# Device-resident block-octree queries (Kokkos): device locate / face-neighbour
# must match the host BlockOctree. No MPI needed; linking tpx_core pulls in Kokkos
# + the morton include (+ MORTON_ENABLE_KOKKOS) under a Kokkos build.
if(TPX_HAVE_KOKKOS)
  add_executable(test_block_octree_kokkos test_block_octree_kokkos.cpp)
  target_link_libraries(test_block_octree_kokkos PRIVATE tpx_core)
  target_include_directories(test_block_octree_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME block_octree_kokkos COMMAND test_block_octree_kokkos)
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

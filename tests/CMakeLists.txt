# Serial tests (no MPI).
add_executable(test_decomposition test_decomposition.cpp)
target_link_libraries(test_decomposition PRIVATE tpx_core)
target_include_directories(test_decomposition PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME decomposition COMMAND test_decomposition)

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
  endforeach()
endif()

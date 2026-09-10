# Every test is registered through peclet_core_add_test / peclet_core_add_mpi_test
# (cmake/PecletCoreTest.cmake): SKIP_RETURN_CODE 77 and the mpi / np8 / bench labels live there.

# The physical domain (peclet/core/domain.hpp): Box + UniformGrid, cell units exact, anisotropic
# spacing, MAC face/node positions, index<->physical round trips, periodic wrap + minimum image.
add_executable(test_domain test_domain.cpp)
target_link_libraries(test_domain PRIVATE peclet_core)
target_include_directories(test_domain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(domain test_domain)

# Serial tests (no MPI).
add_executable(test_decomposition test_decomposition.cpp)
target_link_libraries(test_decomposition PRIVATE peclet_core)
target_include_directories(test_decomposition PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(decomposition test_decomposition)

# Smoothed-aggregation graph-AMG (peclet::core::solver) — host oracle, no Kokkos. Proves the
# mesh-independent (O(N)) CG iteration count on a model vector-Laplacian.
add_executable(test_graph_amg test_graph_amg.cpp)
target_link_libraries(test_graph_amg PRIVATE peclet_core)
target_include_directories(test_graph_amg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(graph_amg test_graph_amg)

# Host-parallel shim (common/host_parallel.hpp): serial ≡ parallel BITWISE on a disjoint-write
# fill, an integer exclusive scan, and a two-pass CSR fill at the scanned offsets. Runs the Kokkos
# host space when Kokkos is in the build tree, the serial fallback otherwise — both are gated.
add_executable(test_host_parallel test_host_parallel.cpp)
target_link_libraries(test_host_parallel PRIVATE peclet_core)
target_include_directories(test_host_parallel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(host_parallel test_host_parallel)

# Morton/Z-order cell indexer (inert unless PECLET_CORE_HAVE_MORTON — set on peclet_core when
# the morton sibling checkout is present).
add_executable(test_morton_indexer test_morton_indexer.cpp)
target_link_libraries(test_morton_indexer PRIVATE peclet_core)
target_include_directories(test_morton_indexer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(morton_indexer test_morton_indexer)

# Per-block adaptive octree (peclet::core::amr::BlockOctree). Inert unless PECLET_CORE_HAVE_MORTON;
# cross-checks against the test-local morton_octree::Octree reference oracle
# (tests/oracle/morton_octree.hpp, a std::map linear octree on the morton library).
add_executable(test_block_octree test_block_octree.cpp)
target_link_libraries(test_block_octree PRIVATE peclet_core)
target_include_directories(test_block_octree PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(block_octree test_block_octree)

# AMR leaf fields + VTU (UnstructuredGrid) visualization output.
add_executable(test_amr_vtu test_amr_vtu.cpp)
target_link_libraries(test_amr_vtu PRIVATE peclet_core)
target_include_directories(test_amr_vtu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_vtu test_amr_vtu)

# SDF-driven refinement (refineToSdf) over the shared peclet::core::geom SDF.
add_executable(test_amr_sdf test_amr_sdf.cpp)
target_link_libraries(test_amr_sdf PRIVATE peclet_core)
target_include_directories(test_amr_sdf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_sdf test_amr_sdf)

# Cell-centered FV Poisson on the octree + geometric multigrid.
add_executable(test_amr_poisson test_amr_poisson.cpp)
target_link_libraries(test_amr_poisson PRIVATE peclet_core)
target_include_directories(test_amr_poisson PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_poisson test_amr_poisson)

add_executable(test_amr_transfer test_amr_transfer.cpp)
target_link_libraries(test_amr_transfer PRIVATE peclet_core)
target_include_directories(test_amr_transfer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_transfer test_amr_transfer)

add_executable(test_amr_adapt test_amr_adapt.cpp)
target_link_libraries(test_amr_adapt PRIVATE peclet_core)
target_include_directories(test_amr_adapt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_adapt test_amr_adapt)

add_executable(test_amr_adapt_transport test_amr_adapt_transport.cpp)
target_link_libraries(test_amr_adapt_transport PRIVATE peclet_core)
target_include_directories(test_amr_adapt_transport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_adapt_transport test_amr_adapt_transport)

# Quadratic coarse-fine flux + refluxing (2nd-order C/F interface).
add_executable(test_amr_cf_quadratic test_amr_cf_quadratic.cpp)
target_link_libraries(test_amr_cf_quadratic PRIVATE peclet_core)
target_include_directories(test_amr_cf_quadratic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_cf_quadratic test_amr_cf_quadratic)

# Cut-cell openness folded into operator + MG coarsening (with quadratic C/F flux).
add_executable(test_amr_openness test_amr_openness.cpp)
target_link_libraries(test_amr_openness PRIVATE peclet_core)
target_include_directories(test_amr_openness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_openness test_amr_openness)

# Cut-cell Dirichlet sub-cell BC (port of sdflow's xi-polynomial scheme) + kappa.
add_executable(test_amr_cut_cell test_amr_cut_cell.cpp)
target_link_libraries(test_amr_cut_cell PRIVATE peclet_core)
target_include_directories(test_amr_cut_cell PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_cut_cell test_amr_cut_cell)

# Collocated Stokes momentum+pressure step (cut-cell IBM momentum + openness projection).
add_executable(test_amr_flow test_amr_flow.cpp)
target_link_libraries(test_amr_flow PRIVATE peclet_core)
target_include_directories(test_amr_flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_flow test_amr_flow)

# Stokes sphere-array drag vs Zick & Homsy (sdflow's ground truth): convergence trend.
add_executable(test_amr_drag test_amr_drag.cpp)
target_link_libraries(test_amr_drag PRIVATE peclet_core)
target_include_directories(test_amr_drag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_drag test_amr_drag)

# ABC/Basilisk divergence-free face field (uniform + graded 2:1).
add_executable(test_amr_face_field test_amr_face_field.cpp)
target_link_libraries(test_amr_face_field PRIVATE peclet_core)
target_include_directories(test_amr_face_field PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_face_field test_amr_face_field)

# Barnes-Hut particle-tree mode over the octree.
add_executable(test_amr_barnes_hut test_amr_barnes_hut.cpp)
target_link_libraries(test_amr_barnes_hut PRIVATE peclet_core)
target_include_directories(test_amr_barnes_hut PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_barnes_hut test_amr_barnes_hut)

# Explicit FV scalar transport (advection-diffusion) on the octree.
add_executable(test_amr_transport test_amr_transport.cpp)
target_link_libraries(test_amr_transport PRIVATE peclet_core)
target_include_directories(test_amr_transport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_transport test_amr_transport)

# Block-octree queries : device locate / face-neighbour
# must match the host BlockOctree. No MPI needed; linking peclet_core pulls in Kokkos
# + the morton include (+ MORTON_ENABLE_KOKKOS) under a Kokkos build.
if(PECLET_CORE_HAVE_KOKKOS)
  # Named field registry (multiphysics field container).
  add_executable(test_field_set test_field_set.cpp)
  target_link_libraries(test_field_set PRIVATE peclet_core)
  target_include_directories(test_field_set PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(field_set test_field_set)

  # Trilinear particle<->grid interpolation (CFD-DEM P2G/G2P primitive).
  add_executable(test_particle_grid test_particle_grid.cpp)
  target_link_libraries(test_particle_grid PRIVATE peclet_core)
  target_include_directories(test_particle_grid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(particle_grid test_particle_grid)

  add_executable(test_block_octree_view test_block_octree_view.cpp)
  target_link_libraries(test_block_octree_view PRIVATE peclet_core)
  target_include_directories(test_block_octree_view PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(block_octree_view test_block_octree_view)

  add_executable(test_amr_fv_op test_amr_fv_op.cpp)
  target_link_libraries(test_amr_fv_op PRIVATE peclet_core)
  target_include_directories(test_amr_fv_op PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_fv_op test_amr_fv_op)

  add_executable(test_amr_assembly test_amr_assembly.cpp)
  target_link_libraries(test_amr_assembly PRIVATE peclet_core)
  target_include_directories(test_amr_assembly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_assembly test_amr_assembly)

  add_executable(test_amr_momentum_assembly test_amr_momentum_assembly.cpp)
  target_link_libraries(test_amr_momentum_assembly PRIVATE peclet_core)
  target_include_directories(test_amr_momentum_assembly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_momentum_assembly test_amr_momentum_assembly)

  add_executable(test_amr_facegeom test_amr_facegeom.cpp)
  target_link_libraries(test_amr_facegeom PRIVATE peclet_core MPI::MPI_CXX)
  target_include_directories(test_amr_facegeom PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_facegeom test_amr_facegeom)

  add_executable(test_amr_multigrid test_amr_multigrid.cpp)
  target_link_libraries(test_amr_multigrid PRIVATE peclet_core)
  target_include_directories(test_amr_multigrid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_multigrid test_amr_multigrid)

  add_executable(test_amr_fv_openness test_amr_fv_openness.cpp)
  target_link_libraries(test_amr_fv_openness PRIVATE peclet_core)
  target_include_directories(test_amr_fv_openness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_fv_openness test_amr_fv_openness)

  add_executable(test_amr_kappa_restrict test_amr_kappa_restrict.cpp)
  target_link_libraries(test_amr_kappa_restrict PRIVATE peclet_core)
  target_include_directories(test_amr_kappa_restrict PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_kappa_restrict test_amr_kappa_restrict)

  add_executable(test_amr_kappa_dirichlet test_amr_kappa_dirichlet.cpp)
  target_link_libraries(test_amr_kappa_dirichlet PRIVATE peclet_core)
  target_include_directories(test_amr_kappa_dirichlet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_kappa_dirichlet test_amr_kappa_dirichlet)

  # MG-preconditioned CG (PCG): Krylov acceleration over the device V-cycle.
  add_executable(test_amr_pcg test_amr_pcg.cpp)
  target_link_libraries(test_amr_pcg PRIVATE peclet_core)
  target_include_directories(test_amr_pcg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_pcg test_amr_pcg)

  # Device momentum operator + solver (MomentumOp / MomentumSolver).
  add_executable(test_amr_momentum test_amr_momentum.cpp)
  target_link_libraries(test_amr_momentum PRIVATE peclet_core)
  target_include_directories(test_amr_momentum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_momentum test_amr_momentum)

  # Collocated Stokes flow step (AmrFlow) — device vs host AmrFlow + analytic.
  add_executable(test_amr_flow_solver test_amr_flow_solver.cpp)
  # flow.hpp carries the distributed context (LeafHalo/DistributedOctree) => MPI headers.
  target_link_libraries(test_amr_flow_solver PRIVATE peclet_core MPI::MPI_CXX)
  target_include_directories(test_amr_flow_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  peclet_core_add_test(amr_flow_solver test_amr_flow_solver)
endif()

# Layer 0 rung 0: device-callable SDF leaf primitives (peclet/core/geom/primitives.hpp) are
# bit-exact to their dem/voro origins. Host-only TU -- also proves PECLET_HD degrades to `inline`
# when Kokkos is absent. See suite/docs/ANALYTIC_SDF_GEOMETRY.md.
add_executable(test_geom_primitives test_geom_primitives.cpp)
target_link_libraries(test_geom_primitives PRIVATE peclet_core)
target_include_directories(test_geom_primitives PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(geom_primitives test_geom_primitives)

# Layer 0 rung 1: runtime node layer -- transforms, iterative CSG, grid leaf in both off-grid
# extension policies -- over the rung-0 leaves. Host-only TU.
add_executable(test_geom_scene test_geom_scene.cpp)
target_link_libraries(test_geom_scene PRIVATE peclet_core)
target_include_directories(test_geom_scene PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(geom_scene test_geom_scene)

# Layer 0 rung 2: a-priori property tests over the leaf vocabulary -- sign vs an independent
# oracle, eikonal |grad|=1, closed-form grad vs FD, 1-Lipschitz, and a brute-force metric check
# that MEASURES the bound direction of the approximate leaves.
add_executable(test_geom_properties test_geom_properties.cpp)
target_link_libraries(test_geom_properties PRIVATE peclet_core)
target_include_directories(test_geom_properties PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(geom_properties test_geom_properties)

# Layer 0 rung 5: host scene assembly + the flat encoding the Python bindings speak. The gate is
# a round trip -- compose, encode to flat int/Real arrays, decode, re-evaluate bit-identically.
add_executable(test_geom_scene_builder test_geom_scene_builder.cpp)
target_link_libraries(test_geom_scene_builder PRIVATE peclet_core)
target_include_directories(test_geom_scene_builder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(geom_scene_builder test_geom_scene_builder)

# Layer 2-for-core: accelerated sphere-union scene queries (candidate grids, equal-R fast path,
# min-image periodicity) must be BIT-IDENTICAL to the brute-force scan -- gated against an
# independent transcription of the AMR stopgap. Needs tests/data/rcp_pack_seed3_unit.txt (run
# from the tests/ directory: ctest sets WORKING_DIRECTORY).
add_executable(test_geom_scene_query test_geom_scene_query.cpp)
target_link_libraries(test_geom_scene_query PRIVATE peclet_core)
target_include_directories(test_geom_scene_query PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(geom_scene_query test_geom_scene_query WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Layer 3 rung 1: cut OWNERSHIP (SceneQueryView::owner) -- the argmin behind eval's min, with a
# scan-order-independent lowest-index tie-break, so moving geometry reads its wall velocity (and
# resolved CFD-DEM posts its force) off the right body no matter which query path answered.
add_executable(test_geom_owner test_geom_owner.cpp)
target_link_libraries(test_geom_owner PRIVATE peclet_core)
target_include_directories(test_geom_owner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(geom_owner test_geom_owner WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Layer 2: implicit-quadrature face apertures / cell volume fractions (geom/quadrature.hpp),
# gated against the closed-form disk-rectangle overlap and for the order they actually deliver.
add_executable(test_geom_quadrature test_geom_quadrature.cpp)
target_link_libraries(test_geom_quadrature PRIVATE peclet_core)
target_include_directories(test_geom_quadrature PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(geom_quadrature test_geom_quadrature)

# Rigid-body mass properties by implicit quadrature (geom/body_properties.hpp): closed-form gates
# incl. the bound-leaf ellipsoid (sign-exact bracketing beats the voxel integrator's bias) and the
# addReframed principal-frame round trip.
add_executable(test_geom_body test_geom_body.cpp)
target_link_libraries(test_geom_body PRIVATE peclet_core)
target_include_directories(test_geom_body PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(geom_body test_geom_body)

# Analytic tree gradients (evalTreeGrad): value bit-parity with evalTree, chain-rule exactness,
# O(h^2) FD agreement on smooth regions, and the RIDGE demonstration (exact face normal where a
# central difference smears by tens of degrees).
add_executable(test_geom_tree_grad test_geom_tree_grad.cpp)
target_link_libraries(test_geom_tree_grad PRIVATE peclet_core)
target_include_directories(test_geom_tree_grad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(geom_tree_grad test_geom_tree_grad)

add_executable(test_sdf test_sdf.cpp)
target_link_libraries(test_sdf PRIVATE peclet_core)
target_include_directories(test_sdf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(sdf test_sdf)

add_executable(test_vti test_vti.cpp)
target_link_libraries(test_vti PRIVATE peclet_core)
target_include_directories(test_vti PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(vti test_vti)

# MPI tests: run across several rank counts.
if(MPI_FOUND)
  add_executable(test_grid_halo test_grid_halo.cpp)
  target_link_libraries(test_grid_halo PRIVATE peclet_halo)
  target_include_directories(test_grid_halo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_nbx_rounds test_nbx_rounds.cpp)
  target_link_libraries(test_nbx_rounds PRIVATE peclet_halo)
  target_include_directories(test_nbx_rounds PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_particle_migration test_particle_migration.cpp)
  target_link_libraries(test_particle_migration PRIVATE peclet_halo)
  target_include_directories(test_particle_migration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_particle_rebalance test_particle_rebalance.cpp)
  target_link_libraries(test_particle_rebalance PRIVATE peclet_halo)
  target_include_directories(test_particle_rebalance PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  # Eulerian load balancing: redistribute grid fields between two ORB decompositions.
  add_executable(test_grid_redistribute_mpi test_grid_redistribute_mpi.cpp)
  target_link_libraries(test_grid_redistribute_mpi PRIVATE peclet_halo)
  target_include_directories(test_grid_redistribute_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_diffusion test_diffusion.cpp)
  target_link_libraries(test_diffusion PRIVATE peclet_halo)
  target_include_directories(test_diffusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_diffusion_sdf test_diffusion_sdf.cpp)
  target_link_libraries(test_diffusion_sdf PRIVATE peclet_halo)
  target_include_directories(test_diffusion_sdf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_ghost_particles_mpi test_ghost_particles_mpi.cpp)
  target_link_libraries(test_ghost_particles_mpi PRIVATE peclet_halo)
  target_include_directories(test_ghost_particles_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_particle_halo_mpi test_particle_halo_mpi.cpp)
  target_link_libraries(test_particle_halo_mpi PRIVATE peclet_halo)
  target_include_directories(test_particle_halo_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_mpi test_amr_distributed_mpi.cpp)
  target_link_libraries(test_amr_distributed_mpi PRIVATE peclet_halo)
  target_include_directories(test_amr_distributed_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_poisson_mpi test_amr_distributed_poisson_mpi.cpp)
  target_link_libraries(test_amr_distributed_poisson_mpi PRIVATE peclet_halo)
  target_include_directories(test_amr_distributed_poisson_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_mg_mpi test_amr_distributed_mg_mpi.cpp)
  target_link_libraries(test_amr_distributed_mg_mpi PRIVATE peclet_halo)
  target_include_directories(test_amr_distributed_mg_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_fv_mpi test_amr_distributed_fv_mpi.cpp)
  target_link_libraries(test_amr_distributed_fv_mpi PRIVATE peclet_halo)
  target_include_directories(test_amr_distributed_fv_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_graded_mg_mpi test_amr_distributed_graded_mg_mpi.cpp)
  target_link_libraries(test_amr_distributed_graded_mg_mpi PRIVATE peclet_halo)
  target_include_directories(test_amr_distributed_graded_mg_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_openness_mpi test_amr_distributed_openness_mpi.cpp)
  target_link_libraries(test_amr_distributed_openness_mpi PRIVATE peclet_halo)
  target_include_directories(test_amr_distributed_openness_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_adapt_mpi test_amr_distributed_adapt_mpi.cpp)
  target_link_libraries(test_amr_distributed_adapt_mpi PRIVATE peclet_halo)
  target_include_directories(test_amr_distributed_adapt_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_distributed_rebalance_mpi test_amr_distributed_rebalance_mpi.cpp)
  target_link_libraries(test_amr_distributed_rebalance_mpi PRIVATE peclet_halo)
  target_include_directories(test_amr_distributed_rebalance_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_amr_leaf_halo_mpi test_amr_leaf_halo_mpi.cpp)
  target_link_libraries(test_amr_leaf_halo_mpi PRIVATE peclet_halo)
  target_include_directories(test_amr_leaf_halo_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  # Device-resident distributed AMR multigrid (C2): needs Kokkos + MPI. peclet_halo carries both when
  # PECLET_CORE_ENABLE_KOKKOS is on (same target the Kokkos grid-halo test links).
  if(PECLET_CORE_ENABLE_KOKKOS)
    add_executable(test_amr_distributed_view_mpi test_amr_distributed_view_mpi.cpp)
    target_link_libraries(test_amr_distributed_view_mpi PRIVATE peclet_halo)
    target_include_directories(test_amr_distributed_view_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_particle_migrator_view_mpi test_particle_migrator_view_mpi.cpp)
    target_link_libraries(test_particle_migrator_view_mpi PRIVATE peclet_halo)
    target_include_directories(test_particle_migrator_view_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_amr_leaf_halo_view_mpi test_amr_leaf_halo_view_mpi.cpp)
    target_link_libraries(test_amr_leaf_halo_view_mpi PRIVATE peclet_halo)
    target_include_directories(test_amr_leaf_halo_view_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_amr_distributed_momentum_mpi test_amr_distributed_momentum_mpi.cpp)
    target_link_libraries(test_amr_distributed_momentum_mpi PRIVATE peclet_halo)
    target_include_directories(test_amr_distributed_momentum_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_amr_distributed_flow_mg_mpi test_amr_distributed_flow_mg_mpi.cpp)
    target_link_libraries(test_amr_distributed_flow_mg_mpi PRIVATE peclet_halo)
    target_include_directories(test_amr_distributed_flow_mg_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_amr_distributed_flow_mpi test_amr_distributed_flow_mpi.cpp)
    target_link_libraries(test_amr_distributed_flow_mpi PRIVATE peclet_halo)
    target_include_directories(test_amr_distributed_flow_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_amr_distributed_adaptflow_mpi test_amr_distributed_adaptflow_mpi.cpp)
    target_link_libraries(test_amr_distributed_adaptflow_mpi PRIVATE peclet_halo)
    target_include_directories(test_amr_distributed_adaptflow_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_amr_distributed_seam_mpi test_amr_distributed_seam_mpi.cpp)
    target_link_libraries(test_amr_distributed_seam_mpi PRIVATE peclet_halo)
    target_include_directories(test_amr_distributed_seam_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  foreach(np IN ITEMS 1 2 4 8)
    if(TARGET test_amr_distributed_view_mpi)
      peclet_core_add_mpi_test(amr_distributed_view_np${np} test_amr_distributed_view_mpi ${np})
    endif()
    if(TARGET test_particle_migrator_view_mpi)
      peclet_core_add_mpi_test(particle_migrator_view_np${np} test_particle_migrator_view_mpi ${np})
    endif()
    peclet_core_add_mpi_test(amr_leaf_halo_np${np} test_amr_leaf_halo_mpi ${np})
    if(TARGET test_amr_leaf_halo_view_mpi)
      peclet_core_add_mpi_test(amr_leaf_halo_view_np${np} test_amr_leaf_halo_view_mpi ${np})
    endif()
    if(TARGET test_amr_distributed_momentum_mpi)
      peclet_core_add_mpi_test(amr_distributed_momentum_np${np} test_amr_distributed_momentum_mpi ${np})
    endif()
    if(TARGET test_amr_distributed_flow_mg_mpi)
      peclet_core_add_mpi_test(amr_distributed_flow_mg_np${np} test_amr_distributed_flow_mg_mpi ${np})
    endif()
    if(TARGET test_amr_distributed_flow_mpi)
      peclet_core_add_mpi_test(amr_distributed_flow_np${np} test_amr_distributed_flow_mpi ${np})
    endif()
    if(TARGET test_amr_distributed_adaptflow_mpi)
      peclet_core_add_mpi_test(amr_distributed_adaptflow_np${np} test_amr_distributed_adaptflow_mpi ${np})
    endif()
    if(TARGET test_amr_distributed_seam_mpi)
      peclet_core_add_mpi_test(amr_distributed_seam_np${np} test_amr_distributed_seam_mpi ${np})
    endif()
    peclet_core_add_mpi_test(grid_redistribute_np${np} test_grid_redistribute_mpi ${np})
    peclet_core_add_mpi_test(grid_halo_np${np} test_grid_halo ${np})
    peclet_core_add_mpi_test(nbx_rounds_np${np} test_nbx_rounds ${np})
    peclet_core_add_mpi_test(particle_migration_np${np} test_particle_migration ${np})
    peclet_core_add_mpi_test(particle_rebalance_np${np} test_particle_rebalance ${np})
    peclet_core_add_mpi_test(diffusion_np${np} test_diffusion ${np})
    peclet_core_add_mpi_test(diffusion_sdf_np${np} test_diffusion_sdf ${np})
    peclet_core_add_mpi_test(ghost_particles_np${np} test_ghost_particles_mpi ${np})
    peclet_core_add_mpi_test(particle_halo_np${np} test_particle_halo_mpi ${np})
    peclet_core_add_mpi_test(amr_distributed_np${np} test_amr_distributed_mpi ${np})
    peclet_core_add_mpi_test(amr_distributed_poisson_np${np} test_amr_distributed_poisson_mpi ${np})
    peclet_core_add_mpi_test(amr_distributed_mg_np${np} test_amr_distributed_mg_mpi ${np})
    peclet_core_add_mpi_test(amr_distributed_fv_np${np} test_amr_distributed_fv_mpi ${np})
    peclet_core_add_mpi_test(amr_distributed_graded_mg_np${np} test_amr_distributed_graded_mg_mpi ${np})
    peclet_core_add_mpi_test(amr_distributed_openness_np${np} test_amr_distributed_openness_mpi ${np})
    peclet_core_add_mpi_test(amr_distributed_adapt_np${np} test_amr_distributed_adapt_mpi ${np})
    peclet_core_add_mpi_test(amr_distributed_rebalance_np${np} test_amr_distributed_rebalance_mpi ${np})
  endforeach()

  # Portable GPU-resident halo : device result must match the CPU path, on whatever backend
  # Kokkos was built for (CUDA / HIP / OpenMP). Compiled as CXX — Kokkos routes it to the device
  # compiler via its launch compiler (so this is a .cpp, not a .cu).
  if(PECLET_CORE_HAVE_KOKKOS)
    # Layer 2-for-core: batched scene evaluation on device vs the host query — the sphere-union
    # path (fma-canonical) must be BITWISE identical, the general CSG tree within sign + ULP band.
    add_executable(test_geom_batch_device test_geom_batch_device.cpp)
    target_include_directories(test_geom_batch_device PRIVATE
      ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_geom_batch_device PRIVATE Kokkos::kokkos)
    peclet_core_add_test(geom_batch_device test_geom_batch_device WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    # The face-CSR solver layer (peclet::core::solver: face_csr / coloring / csr_operator /
    # csr_bicgstab / vector_ops, lifted out of the AMR tree 2026-09-10): host row kernels vs the
    # device matvec, a proper symmetrised colouring, Jacobi / multicolour-SGS smoothing, BiCGStab
    # and defect correction on a 7-point reaction-diffusion CSR. No MPI, no octree.
    add_executable(test_csr_solver test_csr_solver.cpp)
    target_include_directories(test_csr_solver PRIVATE
      ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_csr_solver PRIVATE Kokkos::kokkos)
    peclet_core_add_test(csr_solver test_csr_solver)

    # Device GraphAMG apply vs the host oracle (no MPI; serial + device).
    add_executable(test_graph_amg_device test_graph_amg_device.cpp)
    target_include_directories(test_graph_amg_device PRIVATE
      ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_graph_amg_device PRIVATE Kokkos::kokkos)
    peclet_core_add_test(graph_amg_device test_graph_amg_device)

    add_executable(test_grid_halo_exchange test_grid_halo_exchange.cpp)
    target_include_directories(test_grid_halo_exchange PRIVATE
      ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_grid_halo_exchange PRIVATE Kokkos::kokkos MPI::MPI_CXX)

    add_executable(test_particle_halo_exchange test_particle_halo_exchange.cpp)
    target_include_directories(test_particle_halo_exchange PRIVATE
      ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_particle_halo_exchange PRIVATE Kokkos::kokkos MPI::MPI_CXX)

    foreach(np IN ITEMS 1 2 4)
      peclet_core_add_mpi_test(grid_halo_exchange_np${np} test_grid_halo_exchange ${np})
      peclet_core_add_mpi_test(particle_halo_exchange_np${np} test_particle_halo_exchange ${np})
    endforeach()
  endif()
endif()

# A-priori defect study for the AMR collocated projection (ghost-projection port, step 0).
# A measurement study (printed tables + gates documenting the mode-0 defects); ctest label `bench`.
add_executable(study_amr_ghost_apriori study_amr_ghost_apriori.cpp)
target_link_libraries(study_amr_ghost_apriori PRIVATE peclet_core)
target_include_directories(study_amr_ghost_apriori PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(study_amr_ghost_apriori study_amr_ghost_apriori LABELS bench)

# C/F interface schemes (cf_scheme.hpp): Martin–Cartwright quadratic deltas for the collocated
# flow operators (divergence / cell gradient / momentum Laplacian) — a-priori truncation on a
# graded mesh + bit-parity of the L delta vs applyLaplacianQuad.
add_executable(test_amr_cf_vector test_amr_cf_vector.cpp)
target_link_libraries(test_amr_cf_vector PRIVATE peclet_core)
target_include_directories(test_amr_cf_vector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(amr_cf_vector test_amr_cf_vector)

# M1 seam census for the mixed-level cut-band plan (docs/amr_mixed_level_cut_band_plan.md §8).
# A measurement study (printed tables, no gates); ctest label `bench`.
add_executable(study_amr_seam_census study_amr_seam_census.cpp)
target_link_libraries(study_amr_seam_census PRIVATE peclet_core)
target_include_directories(study_amr_seam_census PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(study_amr_seam_census study_amr_seam_census LABELS bench)

# M2 virtual-sample order study for the mixed-level cut-band plan (§8 M2). A measurement study.
add_executable(study_amr_seam_sample_order study_amr_seam_sample_order.cpp)
target_link_libraries(study_amr_seam_sample_order PRIVATE peclet_core)
target_include_directories(study_amr_seam_sample_order PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(study_amr_seam_sample_order study_amr_seam_sample_order LABELS bench)

# M3 seam stability probe for the mixed-level cut-band plan (§8 M3): sampled ghost projection
# march on a two-level cut band (host oracle). A measurement study.
add_executable(study_amr_seam_march study_amr_seam_march.cpp)
target_link_libraries(study_amr_seam_march PRIVATE peclet_core)
target_include_directories(study_amr_seam_march PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
peclet_core_add_test(study_amr_seam_march study_amr_seam_march LABELS bench)

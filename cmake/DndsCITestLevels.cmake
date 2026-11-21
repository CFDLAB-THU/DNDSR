# Each registration has one minimum level. The build targets are cumulative.
# Level 0 exercises correctness on small fixtures, normally at one rank count
# per contract. Four ranks are intentional where a subset or empty owner matters.
set(_critical_tests
    # Array layout, ownership, algebra, MPI payloads and collective failures.
    dnds_array dnds_runtime_array dnds_runtime_row_builder dnds_runtime_element_schema
    dnds_runtime_engine dnds_access_tracking dnds_native_dof_algebra_np2
    dnds_runtime_row_transfer_np4 dnds_sparse_batch_exchange_np4
    dnds_id_row_transfer_np2 dnds_runtime_native_io_np2 dnds_runtime_transient_np2
    dnds_distributed_failure_factory_np2 dnds_distributed_failure_transfer_np2
    dnds_distributed_failure_reducer-completion_np2 dnds_distributed_failure_solver-worker_np2
    dnds_distributed_failure_solver-checked-preflight_np2 dnds_distributed_failure_solver-unchecked-preflight_np2

    # Geometry, independent connectivity oracles, transitions and matching.
    geom_elements geom_quadrature geom_mesh_connectivity_ghost_np2
    geom_native_mesh_periodic_import_np2 geom_native_mesh_transition_np2 geom_mesh_preflight_np2
    geom_native_connectivity_np2 geom_native_mesh_wall_dist_np2 geom_mesh_locality_ordering_np2
    geom_distributed_boundary_interface_np2 geom_mesh_cgns_multizone

    # Reconstruction, limiter and interpolation mathematics. The 3D contract
    # uses 64 cells; the full 32^3 golden/convergence suite stays at level 2.
    cfv_gradient_plan_np2 cfv_fv_facility_np2 cfv_distributed_interface_sampling_np2
    cfv_reconstruction_overlap_np2 cfv_reconstruction_overlap_omp2_np2
    cfv_reconstruction3d_contract_np2 cfv_limiters cfv_limited_vr_np2 cfv_reconstruction_math

    # Physical models, temporal algebra, linear solves and the FAS cycle.
    euler_gas_thermo euler_riemann_solvers euler_rans euler_physics_properties
    euler_rotating_frame euler_source_chemical euler_evaluator_small_np2
    solver_field_time_stepper solver_field_gmres solver_direct solver_fas solver_aggregation_amg

    # Solver composition, pressure/thermodynamic fluxes, implicit operators,
    # convergence, subcommunicators and restart across MPI partitions.
    dnds_cauchy_config dnds_cauchy_config_shared dnds_cauchy_state_input
    dnds_cauchy_equation_dofs_np4 dnds_cauchy_riemann dnds_cauchy_riemann_legacy
    dnds_cauchy_boundary_inputs_np2 dnds_cauchy_finite_volume_physics_np2
    dnds_cauchy_viscous_flux_np2 dnds_cauchy_rans_fv_np2 dnds_cauchy_equation_source_np2
    dnds_cauchy_jacobian_np2 dnds_cauchy_linear_solver_np2 dnds_cauchy_sgs_subpartitions_omp_np2
    dnds_cauchy_implicit_domain_np2 dnds_cauchy_polynomial_multigrid_np2
    dnds_cauchy_multidomain_np4 dnds_cauchy_checkpoint_np4 cauchy_frontend_np2

    # Device-enabled builds add one representative run per execution/transport path.
    dnds_runtime_cuda_engine dnds_device_packing_np2 dnds_nccl_transfer_np2
    cfv_reconstruction_cuda_np2 cfv_reconstruction_nccl_np2 dnds_cauchy_domain_cuda_np2)

# End-to-end method/hardware matrices compile only for qualification.
set(_qualification_targets
    dnds_test_cauchy_polynomial_multigrid dnds_test_cauchy_multidomain
    euler_test_evaluator_pipeline cfv_test_reconstruction3d
    dnds_test_amgx_preconditioner dnds_test_cauchy_domain_cuda
    dnds_test_cauchy_spatial_schedule_cuda dnds_test_cauchy_time_fas_cuda
    cfv_test_reconstruction_cuda cfv_test_limited_vr_cuda
    cauchy_test_interface_cuda cfv_test_distributed_interface_sampling_cuda)

foreach(_level RANGE 0 2)
    add_custom_target(ci_cpp_level${_level})
endforeach()
add_dependencies(ci_cpp_level1 ci_cpp_level0)
add_dependencies(ci_cpp_level2 ci_cpp_level1)

set(_catalog "")
get_property(_tests DIRECTORY PROPERTY TESTS)
foreach(_test IN LISTS _tests)
    get_property(_labels TEST ${_test} PROPERTY LABELS)
    get_property(_owner TEST ${_test} PROPERTY DNDS_TEST_TARGET)
    if(NOT _owner)
        message(FATAL_ERROR "CTest registration ${_test} has no executable owner")
    endif()
    if(_test IN_LIST _critical_tests)
        set(_level 0)
    elseif(_test STREQUAL "dnds_cauchy_limited_vr_checkpoint_np4")
        # The fourth rank introduces an empty owner.
        set(_level 1)
    elseif("integration" IN_LIST _labels OR "extended" IN_LIST _labels OR
           "mpi-high" IN_LIST _labels OR "sycl" IN_LIST _labels OR
           _owner IN_LIST _qualification_targets)
        set(_level 2)
    else()
        set(_level 1)
    endif()
    set_property(TEST ${_test} APPEND PROPERTY LABELS level${_level})
    add_dependencies(ci_cpp_level${_level} ${_owner})
    string(APPEND _catalog "${_test}\t${_level}\t${_owner}\t$<TARGET_PROPERTY:${_owner},SOURCES>\n")
endforeach()

file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/dnds-test-catalog.tsv" CONTENT "${_catalog}")

# Compatibility name for existing local build scripts.
add_custom_target(ci_critical_tests DEPENDS ci_cpp_level0)

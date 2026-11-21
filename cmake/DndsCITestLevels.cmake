# Each registration has one minimum level. The build targets are cumulative.
# Only the smoke catalog is enumerated. Routine local/two-rank tests default
# to level 1; large fixtures and higher-rank qualification belong to level 2.
set(_smoke_tests
    dnds_runtime_array
    dnds_runtime_engine
    dnds_id_row_transfer_np2
    geom_elements
    geom_native_mesh_periodic_import_np2
    cfv_gradient_plan_np2
    dnds_cauchy_finite_volume_physics_np2
    solver_field_time_stepper
    dnds_runtime_cuda_engine
    dnds_device_packing_np2)

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
    if(_test IN_LIST _smoke_tests)
        set(_level 0)
    elseif(_test MATCHES "^dnds_cauchy_(checkpoint|equation_dofs|limited_vr_checkpoint)_np4$")
        # The fourth rank introduces a subset reader/writer or an empty owner.
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

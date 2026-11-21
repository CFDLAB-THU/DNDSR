# Python test catalog

Each row is a collected pytest case or parameterization. The columns show its minimum level and the CI lanes that select it. Runtime skips still depend on hardware, optional packages, and rank count.

## [test/CFV/test_cfv_dissdisp.py](../../test/CFV/test_cfv_dissdisp.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_native_affine_shear_geometry` |

## [test/CFV/test_fv_correctness.py](../../test/CFV/test_fv_correctness.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `TestArrayBytes::test_array_bytes_positive` |
| 1 | 1 | `TestCellBarycenters::test_wall_mesh_barycenters_are_half_integers` |
| 1 | 1 | `TestCellBarycenters::test_wall_mesh_barycenters_in_domain` |
| 1 | 1 | `TestCellVolumes::test_periodic_mesh_cell_volumes` |
| 1 | 1 | `TestCellVolumes::test_wall_mesh_cell_volumes` |
| 1 | 1 | `TestDOFArrays::test_build_udof_dynamic` |
| 1 | 1 | `TestDOFArrays::test_build_udof_fixed_5` |
| 1 | 1 | `TestDOFArrays::test_build_ugrad_3xD` |
| 1 | 1 | `TestDOFArrays::test_udof_write_read` |
| 1 | 1 | `TestFaceAreas::test_periodic_mesh_face_areas` |
| 1 | 1 | `TestFaceAreas::test_wall_mesh_face_areas` |
| 1 | 1 | `TestGlobalVolume::test_periodic_mesh_global_volume` |
| 1 | 1 | `TestGlobalVolume::test_wall_mesh_global_volume` |
| 1 | 1 | `TestJacobiDet::test_cell_jacobi_det_positive` |
| 1 | 1 | `TestJacobiDet::test_face_jacobi_det_positive` |
| 1 | 1 | `TestSmoothScale::test_smooth_scale_ratio` |

## [test/CFV/test_fv_facility.py](../../test/CFV/test_fv_facility.py)

| Level | MPI ranks | Case |
|---|---|---|
| 0 | 2 | `test_reconstruction_shared_buffers_explicit_halo_and_point_sampling[True]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FP64-mpi]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FP64-nccl]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FP64MatrixShared-mpi]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FP64MatrixShared-nccl]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FP64Pipelined-mpi]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FP64Pipelined-nccl]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FP64TensorCoreAsync-mpi]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FP64TensorCoreAsync-nccl]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FloatFloatPipelined-mpi]` |
| 1 | 1 | `test_device_reconstruction_publication_layouts_and_python_shared_buffers[FloatFloatPipelined-nccl]` |
| 1 | 1 | `test_distributed_interface_owner_sampling_and_query_order` |
| 1 | 1 | `test_metric_options_match_keyword_adapter` |
| 1 | 1 | `test_prepared_capabilities_preserve_sweep_and_freeze_operation_contract` |
| 1 | 1, 2 | `test_reconstruction_shared_buffers_explicit_halo_and_point_sampling[False]` |
| 1 | 1 | `test_standalone_fv_gradients_and_vr_conversion` |
| 1 | 1 | `test_stride_aware_reconstruction_uses_shared_runtime_physical_fields[col]` |
| 1 | 1 | `test_stride_aware_reconstruction_uses_shared_runtime_physical_fields[tiled]` |
| 1 | 1 | `test_two_domains_interface_sampling_and_transform_publication` |

## [test/CFV/test_vr_correctness.py](../../test/CFV/test_vr_correctness.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `TestBoundaryCallback::test_get_fboundary` |
| 1 | 1 | `TestConstantExactness::test_constant_field_zero_urec` |
| 1 | 1 | `TestLinearFieldReconstruction::test_linear_x_reconstruction` |
| 1 | 1 | `TestMatrixAccess::test_matrixAAInvB_accessible` |
| 1 | 1 | `TestMatrixAccess::test_vectorAInvB_accessible` |
| 1 | 1 | `TestModelEvaluatorRHS::test_rhs_nonzero_for_nonuniform_field` |
| 1 | 1 | `TestModelEvaluatorRHS::test_rhs_zero_for_constant_field` |
| 1 | 1 | `TestReconstructionConvergence::test_iterative_vr_convergence` |
| 1 | 1 | `TestVRConstruction::test_cell_bary_in_range` |

## [test/Cauchy/test_ditr_equations.py](../../test/Cauchy/test_ditr_equations.py)

| Level | MPI ranks | Case |
|---|---|---|
| 0 | 1 | `test_queued_acm_stage_sources_and_stopping_residual` |

## [test/Cauchy/test_domain.py](../../test/Cauchy/test_domain.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_configured_implicit_method_and_retained_shared_state[bdf2-1]` |
| 1 | 1 | `test_configured_implicit_method_and_retained_shared_state[ditr-1]` |
| 1 | 1 | `test_configured_implicit_method_and_retained_shared_state[esdirk4-5]` |
| 1 | 1 | `test_configured_mixture_fields_implicit_step_and_native_worker_refresh[KOmegaSST]` |
| 1 | 1 | `test_configured_mixture_fields_implicit_step_and_native_worker_refresh[SA]` |
| 1 | 1 | `test_configured_mixture_fields_implicit_step_and_native_worker_refresh[none]` |
| 1 | 1 | `test_configured_rans_fields_wall_setup_implicit_restart_and_output[KOmegaSST]` |
| 1 | 1 | `test_configured_rans_fields_wall_setup_implicit_restart_and_output[SA]` |
| 1 | 1 | `test_coordinated_simulation_python_owners_and_reversed_subgroups[False]` |
| 1 | 1 | `test_coordinated_simulation_python_owners_and_reversed_subgroups[True]` |
| 1 | 1 | `test_explicit_derived_refresh_shared_storage_and_freshness` |
| 1 | 1 | `test_explicit_numerical_rejection_returns_a_status` |
| 1 | 1 | `test_native_lvr_config_and_warm_restart_through_shared_python_arrays[v2-soa]` |
| 1 | 1 | `test_native_lvr_config_and_warm_restart_through_shared_python_arrays[v3-aosoa32]` |
| 1 | 1 | `test_physical_step_selection_retains_state_and_restores_growth_history[bdf2]` |
| 1 | 1 | `test_physical_step_selection_retains_state_and_restores_growth_history[ssprk3]` |
| 1 | 1 | `test_python_coupled_iteration_matches_native_step_and_restart[cpu]` |
| 1 | 1 | `test_python_coupled_iteration_matches_native_step_and_restart[cuda]` |
| 1 | 1 | `test_queued_device_matrix_action_and_jacobi_fixed_point` |
| 1 | 1 | `test_rank_local_invalid_config_is_collectively_rejected` |
| 1 | 1 | `test_registered_python_policy_matches_native_connected_domains[bdf1-False-pythonSingleStageImplicit]` |
| 1 | 1 | `test_registered_python_policy_matches_native_connected_domains[ditr-True-pythonDITRSequential2Stage]` |
| 1 | 1 | `test_shared_state_real_residual_ssprk3_and_restart[aos-Roe-0]` |
| 1 | 1 | `test_shared_state_real_residual_ssprk3_and_restart[aosoa32-HLLEP-2]` |
| 1 | 1 | `test_shared_state_real_residual_ssprk3_and_restart[soa-AUSMPlusUp-2]` |
| 1 | 1 | `test_spatial_expression_uses_physical_state_units_and_conservative_output` |
| 1 | 1 | `test_steady_bdf1_override_keeps_the_physical_clock_frozen` |
| 1 | 1 | `test_tagged_physical_inputs_match_code_unit_frontend` |

## [test/Cauchy/test_operators.py](../../test/Cauchy/test_operators.py)

| Level | MPI ranks | Case |
|---|---|---|
| 0 | 2 | `test_finite_volume_operator_frontend_composes_native_residual_and_linear_solve[False]` |
| 0 | 2 | `test_finite_volume_operator_frontend_composes_native_residual_and_linear_solve[True]` |

## [test/DNDS/test_arraydof.py](../../test/DNDS/test_arraydof.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test` |

## [test/DNDS/test_basic.py](../../test/DNDS/test_basic.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_ArrayEigenMatrix` |
| 1 | 1 | `test_ArrayEigenMatrixBatch` |
| 1 | 1 | `test_ArrayEigenUniMatrixBatch` |
| 1 | 1 | `test_ArrayEigenVector` |
| 1 | 1 | `test_ParArrayPair` |
| 1 | 1 | `test_adj` |
| 1 | 1 | `test_all_reduce_scalar` |
| 1 | 1 | `test_arrayRU` |
| 1 | 1 | `test_array_trans` |

## [test/DNDS/test_native_dof_algebra.py](../../test/DNDS/test_native_dof_algebra.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_cuda_global_reductions_wait_for_local_device_results_before_mpi` |
| 1 | 1 | `test_eager_algebra_cross_stream_results_retain_native_operands[device0]` |
| 1 | 1 | `test_eager_algebra_cross_stream_results_retain_native_operands[device1]` |
| 1 | 1 | `test_field_import_rejects_wrong_dtype_variable_layout_and_space` |
| 1 | 1 | `test_global_reductions_count_uneven_owned_rows_and_reject_other_spaces_collectively` |
| 1 | 1 | `test_native_owned_operations_share_python_values_and_leave_ghosts[114-dtype0]` |
| 1 | 1 | `test_native_owned_operations_share_python_values_and_leave_ghosts[114-dtype1]` |
| 1 | 1 | `test_native_owned_operations_share_python_values_and_leave_ghosts[5-dtype0]` |
| 1 | 1 | `test_native_owned_operations_share_python_values_and_leave_ghosts[5-dtype1]` |
| 1 | 1 | `test_native_owned_operations_share_python_values_and_leave_ghosts[6-dtype0]` |
| 1 | 1 | `test_native_owned_operations_share_python_values_and_leave_ghosts[6-dtype1]` |
| 1 | 1 | `test_native_owned_operations_share_python_values_and_leave_ghosts[7-dtype0]` |
| 1 | 1 | `test_native_owned_operations_share_python_values_and_leave_ghosts[7-dtype1]` |
| 1 | 1 | `test_physical_tables_share_native_fields_and_async_mixed_layout_algebra[dtype0-device0]` |
| 1 | 1 | `test_physical_tables_share_native_fields_and_async_mixed_layout_algebra[dtype0-device1]` |
| 1 | 1 | `test_physical_tables_share_native_fields_and_async_mixed_layout_algebra[dtype1-device0]` |
| 1 | 1 | `test_physical_tables_share_native_fields_and_async_mixed_layout_algebra[dtype1-device1]` |

## [test/DNDS/test_runtime_array.py](../../test/DNDS/test_runtime_array.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1, 2 | `test_allocation_names_share_storage_without_registry_retention` |
| 1 | 1, 2 | `test_collective_checkpoint_rejects_one_rank_shape_edit_and_recovers` |
| 1 | 1, 2 | `test_payload_rejects_same_size_different_element_schema[float64-int64]` |
| 1 | 1, 2 | `test_payload_rejects_same_size_different_element_schema[source_type1-bound_type1]` |
| 1 | 1, 2 | `test_physical_layout_debug_views_and_element_updates[col]` |
| 1 | 1, 2 | `test_physical_layout_debug_views_and_element_updates[tiled]` |
| 1 | 1, 2 | `test_runtime_builtin_reverse_reductions` |
| 1 | 1, 2 | `test_runtime_direct_indexing_and_live_debug_views[device0]` |
| 1 | 1, 2 | `test_runtime_direct_indexing_and_live_debug_views[device1]` |
| 1 | 1, 2 | `test_runtime_failed_write_and_pair_overlap` |
| 1 | 1, 2 | `test_runtime_leases_and_snapshots[dtype5]` |
| 1 | 1, 2 | `test_runtime_leases_and_snapshots[float32]` |
| 1 | 1, 2 | `test_runtime_leases_and_snapshots[float64]` |
| 1 | 1, 2 | `test_runtime_leases_and_snapshots[int32]` |
| 1 | 1, 2 | `test_runtime_leases_and_snapshots[int64]` |
| 1 | 1, 2 | `test_runtime_leases_and_snapshots[uint8]` |
| 1 | 1, 2 | `test_runtime_parallel_checkpoint[device0]` |
| 1 | 1, 2 | `test_runtime_parallel_checkpoint[device1]` |
| 1 | 1, 2 | `test_runtime_pod_typemap_and_custom_reducer` |
| 1 | 1, 2 | `test_runtime_ragged_debug_row_views` |
| 1 | 1, 2 | `test_runtime_ragged_mpi[device0-transport0]` |
| 1 | 1, 2 | `test_runtime_ragged_mpi[device0-transport1]` |
| 1 | 1, 2 | `test_runtime_ragged_mpi[device1-transport0]` |
| 1 | 1, 2 | `test_runtime_ragged_mpi[device1-transport1]` |

## [test/DNDS/test_runtime_array_bridge.py](../../test/DNDS/test_runtime_array_bridge.py)

| Level | MPI ranks | Case |
|---|---|---|
| 0 | 2 | `test_one_pod_schema_spans_native_python_mpi_and_direct_io[True-h5]` |
| 1 | 1, 2 | `test_bound_pod_class_snapshots_and_tracked_live_fields[device0]` |
| 1 | 1, 2 | `test_bound_pod_class_snapshots_and_tracked_live_fields[device1]` |
| 1 | 1, 2 | `test_bound_pod_codec_is_reused_by_row_construction_and_editing` |
| 1 | 1, 2 | `test_class_pod_transfer_and_checkpoint[device0]` |
| 1 | 1, 2 | `test_class_pod_transfer_and_checkpoint[device1]` |
| 1 | 1, 2 | `test_exports_retain_allocations_and_metadata_after_cpp_reset_or_destruction[device0]` |
| 1 | 1, 2 | `test_exports_retain_allocations_and_metadata_after_cpp_reset_or_destruction[device1]` |
| 1 | 1, 2 | `test_foreign_owner_survives_python_slices_and_cpp_aliases` |
| 1 | 1, 2 | `test_native_owner_and_python_share_values_layout_and_controller[device0]` |
| 1 | 1, 2 | `test_native_owner_and_python_share_values_layout_and_controller[device1]` |
| 1 | 1, 2 | `test_nested_slice_and_access_retain_storage_until_last_owner[device0]` |
| 1 | 1, 2 | `test_nested_slice_and_access_retain_storage_until_last_owner[device1]` |
| 1 | 1, 2 | `test_one_pod_schema_spans_native_python_mpi_and_direct_io[False-h5]` |
| 1 | 1, 2 | `test_one_pod_schema_spans_native_python_mpi_and_direct_io[False-json]` |
| 1 | 1, 2 | `test_one_pod_schema_spans_native_python_mpi_and_direct_io[True-json]` |
| 1 | 1, 2 | `test_payload_and_reducer_cross_native_python_boundary[False-False]` |
| 1 | 1, 2 | `test_payload_and_reducer_cross_native_python_boundary[False-None]` |
| 1 | 1, 2 | `test_payload_and_reducer_cross_native_python_boundary[False-True]` |
| 1 | 1, 2 | `test_payload_and_reducer_cross_native_python_boundary[True-False]` |
| 1 | 1, 2 | `test_payload_and_reducer_cross_native_python_boundary[True-None]` |
| 1 | 1, 2 | `test_payload_and_reducer_cross_native_python_boundary[True-True]` |
| 1 | 1, 2 | `test_pod_edit_retains_original_storage_if_callback_rebinds_handle[device0]` |
| 1 | 1, 2 | `test_pod_edit_retains_original_storage_if_callback_rebinds_handle[device1]` |
| 1 | 1, 2 | `test_pod_view_proxy_and_saved_method_ownership[device0]` |
| 1 | 1, 2 | `test_pod_view_proxy_and_saved_method_ownership[device1]` |
| 1 | 1, 2 | `test_python_constructed_class_pod_and_type_rejection[device0]` |
| 1 | 1, 2 | `test_python_constructed_class_pod_and_type_rejection[device1]` |
| 1 | 1, 2 | `test_python_created_array_can_be_used_and_retained_by_native_owner[device0]` |
| 1 | 1, 2 | `test_python_created_array_can_be_used_and_retained_by_native_owner[device1]` |
| 1 | 1, 2 | `test_ragged_integer_and_custom_pod_cross_language[device0]` |
| 1 | 1, 2 | `test_ragged_integer_and_custom_pod_cross_language[device1]` |
| 1 | 1, 2 | `test_registered_numpy_record_retains_native_alignment_and_schema_across_io` |
| 1 | 1, 2 | `test_registered_pod_byte_fields_keep_byte_semantics_in_dtype_and_io[h5]` |
| 1 | 1, 2 | `test_registered_pod_byte_fields_keep_byte_semantics_in_dtype_and_io[json]` |
| 1 | 1, 2 | `test_registered_schema_custom_reducer_uses_declared_numpy_representation` |
| 1 | 1, 2 | `test_registered_schema_rejects_untyped_same_size_storage_collectively[False]` |
| 1 | 1, 2 | `test_registered_schema_rejects_untyped_same_size_storage_collectively[True]` |
| 1 | 1, 2 | `test_static_width_conversion_needs_no_python_class_instantiation` |

## [test/DNDS/test_runtime_dlpack.py](../../test/DNDS/test_runtime_dlpack.py)

| Level | MPI ranks | Case |
|---|---|---|
| 2 | 1, 2 | `test_capsule_single_consumption_and_runtime_roundtrip` |
| 2 | 1, 2 | `test_cuda_nondefault_producer_consumer_and_explicit_join` |
| 2 | 1, 2 | `test_empty_vector_offset_import_and_roundtrip` |
| 2 | 1, 2 | `test_engine_destructor_drains_foreign_owner_without_holding_gil` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-float32]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-float64]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-int16]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-int32]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-int64]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-int8]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-uint16]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-uint32]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-uint64]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[False-uint8]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-float32]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-float64]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-int16]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-int32]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-int64]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-int8]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-uint16]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-uint32]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-uint64]` |
| 2 | 1, 2 | `test_export_pointer_mutation_alias_lifetime_and_generation[True-uint8]` |
| 2 | 1, 2 | `test_export_survives_reset_and_native_owner[False]` |
| 2 | 1, 2 | `test_export_survives_reset_and_native_owner[True]` |
| 2 | 1, 2 | `test_import_owner_survives_queued_kernel_copy_and_release[False]` |
| 2 | 1, 2 | `test_import_owner_survives_queued_kernel_copy_and_release[True]` |
| 2 | 1, 2 | `test_import_retains_producer_and_shares_with_cpp[False]` |
| 2 | 1, 2 | `test_import_retains_producer_and_shares_with_cpp[True]` |
| 2 | 1, 2 | `test_numpy_export_capsule_once_and_failure_cleanup` |
| 2 | 1, 2 | `test_physical_table_exports_are_truthful_and_retain_lease[False-col]` |
| 2 | 1, 2 | `test_physical_table_exports_are_truthful_and_retain_lease[False-tiled]` |
| 2 | 1, 2 | `test_physical_table_exports_are_truthful_and_retain_lease[True-col]` |
| 2 | 1, 2 | `test_physical_table_exports_are_truthful_and_retain_lease[True-tiled]` |
| 2 | 1, 2 | `test_reject_layouts_readonly_and_unsupported_dtype` |

## [test/DNDS/test_runtime_element_schema.py](../../test/DNDS/test_runtime_element_schema.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1, 2 | `test_flattened_names_distinguish_literal_delimiters_from_paths[dtype0]` |
| 1 | 1, 2 | `test_flattened_names_distinguish_literal_delimiters_from_paths[dtype1]` |
| 1 | 1, 2 | `test_flattened_names_distinguish_literal_delimiters_from_paths[dtype2]` |
| 1 | 1, 2 | `test_non_native_scalar_and_nested_byte_orders_still_reject_input` |
| 1 | 1, 2 | `test_opaque_void_dtype_declares_bytes_and_preserves_simple_native_field_names` |
| 1 | 1, 2 | `test_packed_record_keeps_unaligned_scalar_word_offsets` |
| 1 | 1, 2 | `test_structured_schema_rejects_same_size_different_words_collectively[False]` |
| 1 | 1, 2 | `test_structured_schema_rejects_same_size_different_words_collectively[True]` |
| 1 | 1, 2 | `test_subarray_words_use_base_offsets_and_stable_little_endian_identity` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[False-dtype0]` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[False-dtype1]` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[False-dtype2]` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[False-dtype3]` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[False-dtype4]` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[True-dtype0]` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[True-dtype1]` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[True-dtype2]` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[True-dtype3]` |
| 1 | 1, 2 | `test_unsupported_word_dtypes_are_rejected_before_transfer_or_io[True-dtype4]` |

## [test/DNDS/test_runtime_engine.py](../../test/DNDS/test_runtime_engine.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution0-False]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution0-True]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution1-False]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution1-True]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution2-False]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution2-True]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution3-False]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution3-True]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution4-False]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution4-True]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution5-False]` |
| 1 | 1, 2 | `test_async_pod_values_share_prepared_variable_metadata[execution5-True]` |
| 1 | 1, 2 | `test_checkpoint_requires_finished_stage_then_restores_shared_debug_snapshot[execution0]` |
| 1 | 1, 2 | `test_checkpoint_requires_finished_stage_then_restores_shared_debug_snapshot[execution1]` |
| 1 | 1, 2 | `test_checkpoint_requires_finished_stage_then_restores_shared_debug_snapshot[execution2]` |
| 1 | 1, 2 | `test_checkpoint_requires_finished_stage_then_restores_shared_debug_snapshot[execution3]` |
| 1 | 1, 2 | `test_checkpoint_requires_finished_stage_then_restores_shared_debug_snapshot[execution4]` |
| 1 | 1, 2 | `test_checkpoint_requires_finished_stage_then_restores_shared_debug_snapshot[execution5]` |
| 1 | 1, 2 | `test_close_drains_work_and_rejects_old_streams[execution0]` |
| 1 | 1, 2 | `test_close_drains_work_and_rejects_old_streams[execution1]` |
| 1 | 1, 2 | `test_close_drains_work_and_rejects_old_streams[execution2]` |
| 1 | 1, 2 | `test_close_drains_work_and_rejects_old_streams[execution3]` |
| 1 | 1, 2 | `test_close_drains_work_and_rejects_old_streams[execution4]` |
| 1 | 1, 2 | `test_close_drains_work_and_rejects_old_streams[execution5]` |
| 1 | 1, 2 | `test_dynamic_table_allocation_and_rejected_release_preserve_ownership[execution0]` |
| 1 | 1, 2 | `test_dynamic_table_allocation_and_rejected_release_preserve_ownership[execution1]` |
| 1 | 1, 2 | `test_dynamic_table_allocation_and_rejected_release_preserve_ownership[execution2]` |
| 1 | 1, 2 | `test_dynamic_table_allocation_and_rejected_release_preserve_ownership[execution3]` |
| 1 | 1, 2 | `test_dynamic_table_allocation_and_rejected_release_preserve_ownership[execution4]` |
| 1 | 1, 2 | `test_dynamic_table_allocation_and_rejected_release_preserve_ownership[execution5]` |
| 1 | 1, 2 | `test_managed_mpi_handle_can_be_dropped_before_engine_close[False]` |
| 1 | 1, 2 | `test_managed_mpi_handle_can_be_dropped_before_engine_close[True]` |
| 1 | 1, 2 | `test_python_orchestrates_native_operators_and_retains_debug_views[execution0]` |
| 1 | 1, 2 | `test_python_orchestrates_native_operators_and_retains_debug_views[execution1]` |
| 1 | 1, 2 | `test_python_orchestrates_native_operators_and_retains_debug_views[execution2]` |
| 1 | 1, 2 | `test_python_orchestrates_native_operators_and_retains_debug_views[execution3]` |
| 1 | 1, 2 | `test_python_orchestrates_native_operators_and_retains_debug_views[execution4]` |
| 1 | 1, 2 | `test_python_orchestrates_native_operators_and_retains_debug_views[execution5]` |
| 1 | 1, 2 | `test_recorded_completion_can_outlive_python_engine[cpu-0]` |
| 1 | 1, 2 | `test_recorded_completion_can_outlive_python_engine[cpu-2]` |
| 1 | 1, 2 | `test_recorded_completion_can_outlive_python_engine[cuda-2]` |
| 1 | 1, 2 | `test_shared_checker_rejects_unordered_alias_access_and_accepts_an_explicit_join` |

## [test/DNDS/test_runtime_nccl.py](../../test/DNDS/test_runtime_nccl.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1, 2 | `test_nccl_capability_is_explicit` |
| 1 | 1, 2 | `test_nccl_queued_copies_and_retained_shared_arrays[False]` |
| 1 | 1, 2 | `test_nccl_queued_copies_and_retained_shared_arrays[True]` |
| 1 | 1, 2 | `test_streamed_global_sum_retains_python_operands` |

## [test/DNDS/test_runtime_row_builder.py](../../test/DNDS/test_runtime_row_builder.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1, 2 | `test_detached_edit_keeps_old_view_and_exclusive_edit_refuses_aliases` |
| 1 | 1, 2 | `test_exclusive_commit_and_abandonment_are_explicit` |
| 1 | 1, 2 | `test_gpu_padded_metadata_and_detached_cpu_edit` |
| 1 | 1, 2 | `test_incremental_rows_commit_once_and_keep_draft_snapshots_independent[kind0]` |
| 1 | 1, 2 | `test_incremental_rows_commit_once_and_keep_draft_snapshots_independent[kind1]` |
| 1 | 1, 2 | `test_metadata_snapshots_and_padded_debug_indexing` |

## [test/DNDS/test_runtime_transient.py](../../test/DNDS/test_runtime_transient.py)

| Level | MPI ranks | Case |
|---|---|---|
| 0 | 2 | `test_shared_context_known_route_and_tag_reuse[transport0-mode0]` |
| 1 | 1, 2 | `test_completed_callback_does_not_retain_its_frame_and_transfer` |
| 1 | 1, 2 | `test_preserved_transfer_layout_defaults_and_explicit_compaction[csr]` |
| 1 | 1, 2 | `test_preserved_transfer_layout_defaults_and_explicit_compaction[padded]` |
| 1 | 1, 2 | `test_preserved_transfer_layout_defaults_and_explicit_compaction[table]` |
| 1 | 1, 2 | `test_private_dynamic_storage_migration_and_retained_views[False]` |
| 1 | 1, 2 | `test_private_dynamic_storage_migration_and_retained_views[True]` |
| 1 | 1, 2 | `test_python_reducer_lifetime_through_eager_mpi_progress` |
| 1 | 1, 2 | `test_reducer_failure_after_posting_aborts_in_child[persistent]` |
| 1 | 1, 2 | `test_reducer_failure_after_posting_aborts_in_child[transient]` |
| 1 | 1, 2 | `test_retained_binding_and_slice_follow_old_mesh_snapshot_after_reset[False]` |
| 1 | 1, 2 | `test_retained_binding_and_slice_follow_old_mesh_snapshot_after_reset[True]` |
| 1 | 1, 2 | `test_sender_ordering_returns_route_and_explicit_array_permutation` |
| 1 | 1, 2 | `test_shared_context_known_route_and_tag_reuse[transport0-mode1]` |
| 1 | 1, 2 | `test_shared_context_known_route_and_tag_reuse[transport1-mode0]` |
| 1 | 1, 2 | `test_shared_context_known_route_and_tag_reuse[transport1-mode1]` |
| 1 | 1, 2 | `test_streamed_host_allreduce_orders_consumer_after_binding_release[False]` |
| 1 | 1, 2 | `test_streamed_host_allreduce_orders_consumer_after_binding_release[True]` |

## [test/DNDS/test_view_ownership.py](../../test/DNDS/test_view_ownership.py)

| Level | MPI ranks | Case |
|---|---|---|
| 0 | 1 | `test_flat_view_retains_allocation_not_array` |
| 1 | 1 | `test_csr_row_views_survive_resize_and_compression[False]` |
| 1 | 1 | `test_csr_row_views_survive_resize_and_compression[True]` |
| 1 | 1 | `test_derived_views_own_each_exported_buffer[False-batch]` |
| 1 | 1 | `test_derived_views_own_each_exported_buffer[False-matrix]` |
| 1 | 1 | `test_derived_views_own_each_exported_buffer[False-uniform_batch]` |
| 1 | 1 | `test_derived_views_own_each_exported_buffer[False-vector]` |
| 1 | 1 | `test_derived_views_own_each_exported_buffer[True-batch]` |
| 1 | 1 | `test_derived_views_own_each_exported_buffer[True-matrix]` |
| 1 | 1 | `test_derived_views_own_each_exported_buffer[True-uniform_batch]` |
| 1 | 1 | `test_derived_views_own_each_exported_buffer[True-vector]` |
| 1 | 1 | `test_empty_export_has_own_metadata` |
| 1 | 1 | `test_row_starts_survive_structure_replacement` |

## [test/Euler/test_restart_redistribute.py](../../test/Euler/test_restart_redistribute.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_restart_redistribute_same_np` |
| 2 | 1 | `test_restart_redistribute_different_np` |
| 2 | 1 | `test_restart_redistribute_large_mesh_multi_np` |

## [test/EulerP/test_basic_eulerP.py](../../test/EulerP/test_basic_eulerP.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_basic_eulerP` |

## [test/EulerP/test_native_solver.py](../../test/EulerP/test_native_solver.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_native_cuda_evaluator_read_facade_matches_cpu_rhs` |
| 1 | 1 | `test_native_solver_setup_evaluator_rhs_matches_primary_wire_restart` |

## [test/Geom/test_basic_geom.py](../../test/Geom/test_basic_geom.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_mesh0` |
| 1 | 1 | `test_multi_layer_ghost` |

## [test/Geom/test_mesh_demos.py](../../test/Geom/test_mesh_demos.py)

| Level | MPI ranks | Case |
|---|---|---|
| 2 | 1 | `test_cart_demo_native_setup_has_exact_geometry` |
| 2 | 1 | `test_draw_demo_retains_owned_and_ghost_polygons` |

## [test/Geom/test_mesh_edit.py](../../test/Geom/test_mesh_edit.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_current_authority_geometry_and_escaped_copies` |
| 1 | 1 | `test_explicit_commit_metadata_validation_and_abort` |
| 1 | 1 | `test_neighbor_order_validation_and_reader_conflict` |

## [test/Geom/test_mesh_facade.py](../../test/Geom/test_mesh_facade.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_coordinate_callback_failure_preserves_mesh_for_retry` |
| 1 | 1 | `test_coordinate_callback_shape_and_finite_checks[bad_point0]` |
| 1 | 1 | `test_coordinate_callback_shape_and_finite_checks[bad_point1]` |
| 1 | 1 | `test_detached_cuda_snapshot_debug_copy` |
| 1 | 1 | `test_detached_snapshot_and_escaped_rows` |
| 1 | 1 | `test_entity_distribution_queries_and_stale_descriptor` |
| 1 | 1 | `test_escaped_numpy_slices_retain_scope` |
| 1 | 1 | `test_exception_closes_context_and_snapshots_do_not_pin_mesh` |
| 1 | 1 | `test_managed_coordinate_transform_rejects_escaped_rows` |
| 1 | 1 | `test_native_publication_retains_readonly_aliases_and_detached_snapshot` |
| 1 | 1 | `test_parent_deletion_keeps_row_and_metadata` |
| 1 | 1 | `test_periodic_geometry_transform_updates_physical_coords` |
| 1 | 1 | `test_periodic_incidence_queries` |
| 1 | 1 | `test_readonly_queries_and_checked_indices` |

## [test/Geom/test_native_affine_geometry.py](../../test/Geom/test_native_affine_geometry.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_affine_python_shape_errors_and_retained_read_rollback` |

## [test/Geom/test_native_mesh_builder.py](../../test/Geom/test_native_mesh_builder.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_explicit_setup_audits_and_checkpoint` |
| 1 | 1 | `test_independent_boundaries_supplied_ids_and_pod_metadata` |
| 1 | 1 | `test_native_archive_json_roundtrip_and_escaped_rows` |
| 1 | 1 | `test_native_builder_participates_on_empty_ranks` |
| 1 | 1 | `test_one_rank_conversion_failure_is_collective_and_retry_succeeds[coordinate_csr_width]` |
| 1 | 1 | `test_one_rank_conversion_failure_is_collective_and_retry_succeeds[coordinate_width]` |
| 1 | 1 | `test_one_rank_conversion_failure_is_collective_and_retry_succeeds[index_dtype]` |
| 1 | 1 | `test_one_rank_conversion_failure_is_collective_and_retry_succeeds[non_array]` |
| 1 | 1 | `test_one_rank_conversion_failure_is_collective_and_retry_succeeds[opaque_metadata]` |
| 1 | 1 | `test_remote_node_star_and_owned_global_csr[False]` |
| 1 | 1 | `test_remote_node_star_and_owned_global_csr[True]` |
| 1 | 1 | `test_shared_input_buffers_and_escaped_rows_retain_protection` |
| 1 | 1 | `test_uniform_csr_and_padded_companions_share_original_values[False]` |
| 1 | 1 | `test_uniform_csr_and_padded_companions_share_original_values[True]` |

## [test/Geom/test_native_mesh_cgns.py](../../test/Geom/test_native_mesh_cgns.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_native_cgns_disabled_preflight_reaches_topology` |
| 1 | 1 | `test_native_cgns_topology_archive_and_retained_source` |

## [test/Geom/test_native_mesh_extraction.py](../../test/Geom/test_native_mesh_extraction.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_extract_native_boundary_keeps_source_reads_and_durable_ancestry[False-False]` |
| 1 | 1 | `test_extract_native_boundary_keeps_source_reads_and_durable_ancestry[False-True]` |
| 1 | 1 | `test_extract_native_boundary_keeps_source_reads_and_durable_ancestry[True-False]` |
| 1 | 1 | `test_extract_native_boundary_keeps_source_reads_and_durable_ancestry[True-True]` |

## [test/Geom/test_native_mesh_helpers.py](../../test/Geom/test_native_mesh_helpers.py)

| Level | MPI ranks | Case |
|---|---|---|
| 0 | 2 | `test_default_native_archive_file_faults_reject_collectively_and_retry` |
| 1 | 1 | `test_default_native_preparation_preserves_mesh_and_serial_adapter_contract` |
| 1 | 1 | `test_h5_subset_writer_world_reader_changed_size_and_two_layer_closure` |
| 1 | 1 | `test_native_bisection_and_default_frames_preserve_geometry` |
| 1 | 1 | `test_native_helpers_names_prepare_extract_and_restart` |
| 1 | 1 | `test_native_helpers_reject_mixed_authority_before_dispatch` |
| 1 | 1 | `test_native_helpers_reject_unsupported_options` |
| 1 | 1 | `test_native_partition_prepare_matches_cfv_metrics_and_preserves_source` |
| 1 | 1 | `test_native_periodic_helper_import_and_cfv_geometry` |
| 1 | 1 | `test_native_preparation_wrong_reader_rejects_on_all_ranks_and_retries` |
| 1 | 1 | `test_native_reader_decodes_legacy_primary_archive_without_legacy_state` |
| 1 | 1 | `test_native_weighted_recursive_reader_and_cfv_metrics` |
| 1 | 1 | `test_retired_mesh_backend_rejects_before_file_io` |

## [test/Geom/test_native_mesh_periodic_import.py](../../test/Geom/test_native_mesh_periodic_import.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_native_periodic_reader_exact_physical_cells_and_collective_retry` |

## [test/Geom/test_native_mesh_refinement.py](../../test/Geom/test_native_mesh_refinement.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_native_elevation_bisection_exact_cones_ancestry_and_restart[False-False]` |
| 1 | 1 | `test_native_elevation_bisection_exact_cones_ancestry_and_restart[False-True]` |
| 1 | 1 | `test_native_elevation_bisection_exact_cones_ancestry_and_restart[True-False]` |
| 1 | 1 | `test_provenance_array_independently_retains_shared_owner_after_mesh_release` |

## [test/Geom/test_native_pair_views.py](../../test/Geom/test_native_pair_views.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_all_incidence_aliases_and_owned_ghost_boundaries[2]` |
| 1 | 1 | `test_all_incidence_aliases_and_owned_ghost_boundaries[3]` |
| 1 | 1 | `test_coordinate_and_id_buffers_match_current_numbering` |
| 1 | 1 | `test_empty_parts_ragged_buffers_and_metadata_after_mesh_deletion` |
| 1 | 1 | `test_escaped_rows_and_whole_buffers_block_mutation_and_survive_mesh_owner` |
| 1 | 1 | `test_metadata_is_readonly_retained_and_edit_uses_detached_values` |
| 1 | 1 | `test_periodic_image_aliases_and_retained_wall_vectors` |
| 1 | 1 | `test_rank_local_escaped_id_rejects_collective_edit_and_allows_retry` |

## [test/Geom/test_overset_facade.py](../../test/Geom/test_overset_facade.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_overset_travelling_geometry_uses_current_query_authority[False]` |
| 1 | 1 | `test_overset_travelling_geometry_uses_current_query_authority[True]` |

## [test/Geom/test_owned_mesh_setup.py](../../test/Geom/test_owned_mesh_setup.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_combined_setup_rejects_rank_mismatches_before_publication` |
| 1 | 1 | `test_owned_preparation_detaches_or_rejects_in_place_active_reads` |
| 1 | 1 | `test_python_setup_partition_precedes_refinement_and_retains_original_labels` |

## [test/Geom/test_surface_point_map.py](../../test/Geom/test_surface_point_map.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1 | `test_shared_surface_snapshot_and_warped_quad_projection[backend0]` |
| 1 | 1 | `test_shared_surface_snapshot_and_warped_quad_projection[backend1]` |

## [test/demos/test_runtime_solver.py](../../test/demos/test_runtime_solver.py)

| Level | MPI ranks | Case |
|---|---|---|
| 1 | 1, 2 | `test_direct_ragged_permutation_and_collision_reduction[cuda-mode0]` |
| 1 | 1, 2 | `test_direct_ragged_permutation_and_collision_reduction[cuda-mode1]` |
| 1 | 1, 2 | `test_direct_ragged_permutation_and_collision_reduction[inline-mode0]` |
| 1 | 1, 2 | `test_direct_ragged_permutation_and_collision_reduction[inline-mode1]` |
| 1 | 1, 2 | `test_direct_ragged_permutation_and_collision_reduction[workers-mode0]` |
| 1 | 1, 2 | `test_direct_ragged_permutation_and_collision_reduction[workers-mode1]` |
| 1 | 1, 2 | `test_iterate_permute_restart[cuda-mode0]` |
| 1 | 1, 2 | `test_iterate_permute_restart[cuda-mode1]` |
| 1 | 1, 2 | `test_iterate_permute_restart[inline-mode0]` |
| 1 | 1, 2 | `test_iterate_permute_restart[inline-mode1]` |
| 1 | 1, 2 | `test_iterate_permute_restart[workers-mode0]` |
| 1 | 1, 2 | `test_iterate_permute_restart[workers-mode1]` |

## Standalone tooling

CTest runs these unittest methods in `harness_tooling` at level 1.

### [test_allocation_explorer.py](../../scripts/tests/test_allocation_explorer.py)

- `AllocationExplorerTest.test_identity_aliases_and_totals`
- `AllocationExplorerTest.test_snapshots_keep_process_identity_separate`
- `AllocationExplorerTest.test_safe_fragment_and_standalone_boundary`
- `AllocationExplorerTest.test_gzip_archive_cli_roundtrip`

### [test_compare_frozen_euler.py](../../scripts/tests/test_compare_frozen_euler.py)

- `FrozenProbeComparison.test_exact_id_join_and_visible_numeric_mismatch`
- `FrozenProbeComparison.test_missing_ids_and_nonfinite_payload_are_rejected`

### [test_cuda_assertions.py](../../scripts/tests/test_cuda_assertions.py)

- `CUDAAssertionScopeTests.test_source_definitions_are_local_and_default_checked`

### [test_generate_cauchy_rhs.py](../../scripts/tests/test_generate_cauchy_rhs.py)

- `RHSCodeGenerationTests.test_catalog_and_unchanged_output`
- `RHSCodeGenerationTests.test_cpu_catalog_distinguishes_physics_form_and_width`
- `RHSCodeGenerationTests.test_cmake_tracks_inputs_and_recovers_every_output`

### [test_harness.py](../../scripts/tests/test_harness.py)

- `HarnessTests.test_hook_rejects_partial_file_before_any_formatting`
- `HarnessTests.test_hook_preserves_paths_and_restages_formatted_files`
- `HarnessTests.test_each_formatter_failure_is_nonzero_and_not_restaged`
- `HarnessTests.test_missing_optional_formatter_does_not_block`
- `HarnessTests.test_summary_process_failure_is_not_success`
- `HarnessTests.test_summary_does_not_count_failed_output_replay`
- `HarnessTests.test_summary_counts_identical_rank_summaries`
- `HarnessTests.test_summary_failed_counts_override_zero_exit`
- `HarnessTests.test_summary_ignores_summary_text_in_command_description`
- `HarnessTests.test_summary_failure_without_any_doctest_output`
- `HarnessTests.test_summary_empty_selection_is_not_success`
- `HarnessTests.test_default_timeout_respects_ctest_and_explicit_override`
- `HarnessTests.test_explicit_zero_timeout_remains_supported`
- `HarnessTests.test_mpi_release_guard_and_reconfigure`
- `HarnessTests.test_all_presets_on_selected_cmake`
- `HarnessTests.test_schema_two_cannot_preserve_generator_discovery`
- `HarnessTests.test_packaging_respects_parallel_environment`
- `HarnessTests.test_shared_skill_aliases_resolve_to_single_source`

### [test_test_targets.py](../../scripts/tests/test_test_targets.py)

- `TestAggregateCoverage.test_transitive_aggregate_and_later_frontend`
- `TestAggregateCoverage.test_registered_orphan_is_rejected`

set(CPPS
    Core/DomainPlacement.cpp
    Core/DomainExecutionContext.cpp
    Core/DomainComponents.cpp
    Core/Domain_Operations.cpp
    Core/DomainStateSnapshot.cpp
    Core/Domain.cpp
    Core/Domain_ExplicitStage.cpp
    Core/CPU/TimeStageKernels.cpp
    Core/TimeStageOperators.cpp
    Core/Domain_ImplicitStep.cpp
    Core/Domain_TimeStep.cpp
    Core/Simulation.cpp
    Core/Simulation_Advance.cpp

    Config/SimulationConfig.cpp
    Config/SimulationConfig_Validation.cpp
    Config/SimulationConfig_Schema.cpp
    Config/CUDAKernelDefaults.cpp
    Config/CUDARHSKernelSelector.cpp

    Physics/EquationSystem.cpp
    Physics/PhysicsResources.cpp
    Physics/PointProperties.cpp
    Physics/EquationSystem_PointProperties.cpp
    Physics/ThermalAdmissibility.cpp
    Operators/Source/ChemicalSource.cpp
    Operators/Source/CPU/ChemicalSourceKernels.cpp
    State/RANSState.cpp
    Discretization/RANSReconstruction.cpp
    Boundary/CPU/RANSWallKernels.cpp
    Input/StateInput.cpp
    Input/InitialField.cpp

    State/EquationDofs.cpp
    State/EquationDofAlgebra.cpp
    State/MeanStateCheck.cpp
    State/StateDiagnostics.cpp
    Discretization/BoundaryForceReduction.cpp
    Discretization/FlowErrorOperator.cpp
    State/EquationDofCommunication.cpp
    State/DerivedFields.cpp
    State/CPU/StatePropertiesKernels.cpp
    Motion/RigidMotion.cpp

    Discretization/FiniteVolume_InterfaceInputs.cpp
    Boundary/BoundarySystem.cpp
    Boundary/BoundarySystem_Operators.cpp
    Boundary/BoundaryPointOperator.cpp
    Interface/InterfaceCoupling.cpp
    Interface/CPU/InterfacePacketKernels.cpp
    Interface/InterfacePacketOperator.cpp
    Interface/CPU/InterfaceSamplingKernels.cpp

    Discretization/FiniteVolume.cpp
    Discretization/FiniteVolume_Placement.cpp
    Discretization/FiniteVolumeInternal_SourceQuadrature.cpp
    Discretization/FiniteVolumeInternal_Stencil.cpp
    Discretization/FiniteVolumeInternal_PositivityQuadrature.cpp
    Discretization/FiniteVolumeViews.cpp
    Discretization/FiniteVolume_RHSOperator.cpp
    Discretization/FiniteVolume_RHS.cpp
    Discretization/FiniteVolume_RHSQueued.cpp
    Discretization/FiniteVolume_RHSHostQueued.cpp
    Discretization/FiniteVolume_Gradient.cpp
    Discretization/FiniteVolume_GradientQueued.cpp
    Discretization/FiniteVolume_CharacteristicsQueued.cpp
    Discretization/FiniteVolume_ReconstructionNorm.cpp
    Discretization/FiniteVolume_GradientHostQueued.cpp
    Discretization/FiniteVolume_DirectGradientQueued.cpp
    Discretization/FiniteVolume_SourcePreparation.cpp
    Discretization/FiniteVolume_SourceQueued.cpp
    Discretization/FiniteVolume_SourceHostQueued.cpp
    Discretization/FiniteVolume_Positivity.cpp
    Discretization/FiniteVolume_PositivityQueued.cpp
    Discretization/FiniteVolume_AuxiliaryPreparation.cpp
    Discretization/FiniteVolume_AuxiliaryQueued.cpp
    Discretization/FiniteVolume_Workspace.cpp
    Discretization/FiniteVolume_WorkspaceFactory.cpp
    Discretization/FiniteVolume_Evaluation.cpp
    Discretization/FiniteVolume_Profiling.cpp
    Discretization/FiniteVolume_ReconstructionState.cpp
    Discretization/FiniteVolume_ReconstructionRestore.cpp
    Discretization/FiniteVolume_BoundaryInputs.cpp
    Discretization/FiniteVolume_BoundaryQueued.cpp
    Discretization/FiniteVolume_BoundaryHostQueued.cpp
    Discretization/FiniteVolume_Flux.cpp
    Discretization/FiniteVolume_Source.cpp
    Discretization/FiniteVolume_Diagnostics.cpp
    Discretization/CPU/FiniteVolumeGradientKernels.cpp
    Discretization/CPU/FiniteVolumeReconstructionKernels.cpp
    Discretization/CPU/FiniteVolumeStateKernels.cpp
    Discretization/CPU/FiniteVolumeBoundaryKernels.cpp
    Discretization/CPU/FiniteVolumeLimitedVRKernels.cpp
    Discretization/CPU/FiniteVolumeAuxiliaryKernels.cpp
    Discretization/CPU/ModalPositivityKernels.cpp
    Discretization/CPU/FiniteVolumePositivityKernels.cpp
    Discretization/CPU/FiniteVolumeRHSKernels_Flow5.cpp
    Discretization/EquationPhysicsProfile.cpp
    Discretization/CPU/FiniteVolumeRHSKernels_Equation.cpp
    Discretization/CPU/FiniteVolumeSourceKernels.cpp
    Discretization/FiniteVolume_AuxiliaryValues.cpp
    Discretization/FiniteVolume_MeanGradientQueued.cpp
    Discretization/FiniteVolume_LimitedVR.cpp
    Discretization/FiniteVolume_LimitedVRExecution.cpp
    Discretization/FiniteVolume_PrepareAuxValues.cpp
    Discretization/FiniteVolume_TimeStep.cpp
    Discretization/FiniteVolume_InitializeField.cpp
    Discretization/WallDistanceSamples.cpp
    Discretization/PerfectGasKernelPoint.cpp

    Implicit/LinearPattern.cpp
    Implicit/LinearProblem.cpp
    Implicit/LinearProblem_Queued.cpp
    Implicit/LinearProblem_Inverse.cpp
    Implicit/LinearProblem_ActionStatus.cpp
    Implicit/FrozenJacobianOperator.cpp
    Implicit/CPU/LocalAggregationAMG.cpp
    Implicit/CPU/LinearBlockKernels_Array.cpp
    Implicit/CPU/LinearBlockKernels_Dof.cpp
    Implicit/CPU/LinearBlockKernels_ILU.cpp
    Implicit/CPU/LinearProblemKernels_Diagonal.cpp
    Implicit/LinearProblem_Dof.cpp
    Implicit/LinearProblem_SGS.cpp
    Implicit/LinearProblem_DofSGS.cpp
    Implicit/LinearProblem_ILU.cpp
    Implicit/LinearSolver.cpp
    Implicit/LocalAMGPreconditioner.cpp
    Implicit/NonlinearSolve.cpp
    Implicit/NonlinearSolver.cpp
    Implicit/NonlinearSolver_State.cpp
    Implicit/NonlinearOperators.cpp
    Implicit/NonlinearOperators_Positivity.cpp
    Implicit/CPU/NonlinearKernels.cpp
    Implicit/UpdatePolicy.cpp
    Implicit/FiniteVolume_PrepareJacobian.cpp
    Implicit/FiniteVolume_Jacobian.cpp
    Implicit/FiniteVolume_DeviceJacobian.cpp
    Implicit/FiniteVolumeDeviceJacobian.cpp
    Implicit/JacobianGeometry.cpp
    Implicit/CPU/JacobianKernels.cpp
    Implicit/CPU/FiniteVolumeJacobianKernels_Equation.cpp
    Implicit/CPU/FiniteVolumeJacobianKernels_Flow.cpp
    Implicit/CPU/FiniteVolumeJacobianKernels_Preparation.cpp
    Implicit/CPU/ACMJacobianKernels.cpp
    Multigrid/PolynomialMultigrid.cpp
    Multigrid/PolynomialMultigridOperations.cpp
    Multigrid/CPU/PolynomialMultigridKernels.cpp

    IO/Domain_Checkpoint.cpp
    IO/Domain_CheckpointWrite.cpp
    IO/Domain_CheckpointRead.cpp
    IO/Domain_CheckpointReadInternal.cpp
    IO/CheckpointMetadata.cpp
    IO/StateArchive.cpp
    IO/FiniteVolume_FieldOutput.cpp
    IO/OutputFields.cpp
    IO/PreparedCheckpoint.cpp
    IO/OutputMeasurement.cpp
    IO/BoundaryOutputPlan.cpp
    IO/PreparedVisualization.cpp
    IO/OutputQueue.cpp
    IO/AsyncOutput.cpp
    IO/CPU/OutputFieldsKernels.cpp
    IO/Simulation_Checkpoint.cpp
    IO/ProgressOutput.cpp
    IO/OutputPlan.cpp
    IO/OutputCoordinator.cpp

    Riemann/RiemannFlux.cpp)
include("${PROJECT_SOURCE_DIR}/cmake/DndsCauchyRHS.cmake")
dnds_generate_cauchy_rhs(_cpu_flux_sources _cpu_flux_include _cpu_flux_codegen cauchy_cpu_flux)
list(APPEND CPPS ${_cpu_flux_sources})
if(DNDS_USE_CUDA)
    list(APPEND CPPS Interface/CUDA/InterfacePacketKernels.cu)
    list(APPEND CPPS Implicit/CUDA/FrozenJacobianKernels.cu)
    list(APPEND CPPS Implicit/CUDA/FiniteVolumeJacobianKernels.cu)
    list(APPEND CPPS Implicit/CUDA/FiniteVolumeJacobianKernels_Faces.cu)
    list(APPEND CPPS Implicit/CUDA/ACMJacobianKernels.cu)
    list(APPEND CPPS Implicit/CUDA/LinearBlockKernels.cu)
    list(APPEND CPPS Implicit/CUDA/AMGGeometry.cu)
    list(APPEND CPPS Core/CUDA/TimeStageKernels.cu)
    list(APPEND CPPS State/CUDA/FieldKernels.cu)
    list(APPEND CPPS State/CUDA/StateDiagnosticsKernels.cu)
    list(APPEND CPPS Discretization/CUDA/BoundaryForceKernels.cu)
    list(APPEND CPPS IO/CUDA/OutputFieldsKernels.cu IO/CUDA/BoundaryOutputGeometry.cu)
    list(APPEND CPPS Implicit/CUDA/LinearDiagonalKernels.cu)
    list(APPEND CPPS Implicit/CUDA/NonlinearKernels.cu Implicit/CUDA/NonlinearKernels_Positivity.cu)
    list(APPEND CPPS Multigrid/CUDA/PolynomialMultigridKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeAuxiliaryKernels.cu Discretization/CUDA/FiniteVolumeBoundaryKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeRHSKernels.cu Discretization/CUDA/FiniteVolumeRHSKernels_Placement.cpp)
    # One manifest owns the valid C/M/D catalog. Generate one specialization
    # per CUDA TU in the build tree, shared by static and Python libraries.
    dnds_generate_cauchy_rhs(_rhs_sources _rhs_include _rhs_codegen cauchy_rhs)
    list(APPEND CPPS ${_rhs_sources})
    # Runtime-shaped ACM leaves provide both mappings for direct gradients,
    # first-order convection and polynomial shapes outside the static catalog.
    foreach(acm_width IN ITEMS 4 5 6)
        set(acm_source "${_rhs_include}/FiniteVolumeACMFluxKernels_C${acm_width}.cu")
        configure_file(Discretization/CUDA/FiniteVolumeACMFluxKernels.cu.in "${acm_source}" @ONLY)
        list(APPEND CPPS "${acm_source}")
    endforeach()
    # Keep reduced-precision face kernels in bounded instantiation groups.
    # Their host dispatcher does not include the numerical kernel definitions.
    list(APPEND CPPS
        Discretization/CUDA/FiniteVolumeRHSKernels_Compute.cpp
        Discretization/CUDA/FiniteVolumeRHSKernels_Compute_FP32_Polynomial.cu
        Discretization/CUDA/FiniteVolumeRHSKernels_Compute_FP32_Dissipation.cu
        Discretization/CUDA/FiniteVolumeRHSKernels_Compute_FP32_Flux.cu
        Discretization/CUDA/FiniteVolumeRHSKernels_Compute_FloatFloat_Polynomial.cu
        Discretization/CUDA/FiniteVolumeRHSKernels_Compute_FloatFloat_Dissipation.cu
        Discretization/CUDA/FiniteVolumeRHSKernels_Compute_FloatFloat_Flux.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeRHSFacePackages.cpp Discretization/CUDA/FiniteVolumeRHSKernels_Segments.cu)
    # A 512-thread segment CTA has at most 128 registers per lane on the
    # qualified SM80/SM86 devices. Apply the same bound to its out-of-line
    # numerical callees; launch_bounds alone constrains only the entry kernel.
    # Block size is runtime, not another template axis. This bound also keeps
    # the shared numerical callee valid for the supported 512-thread launch.
    set_property(SOURCE ${_rhs_sources}
        APPEND PROPERTY COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=128>")
    # Bound each source C/D/M specialization to one TU. Both FV fluxes and
    # sources take their P1--P3 modal widths from the same manifest.
    set(_source_shape_file "${PROJECT_SOURCE_DIR}/scripts/manifests/cauchy_rhs_shapes.json")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_source_shape_file}")
    file(READ "${_source_shape_file}" _source_shapes)
    foreach(source_model IN ITEMS ACM PerfectGas)
        if(source_model STREQUAL ACM)
            set(_rans_widths 5 6)
        else()
            set(_rans_widths 6 7)
        endif()
        foreach(source_width IN LISTS _rans_widths)
            foreach(source_dim IN ITEMS 2 3)
                string(JSON _mode_count LENGTH "${_source_shapes}" modes_by_dimension ${source_dim})
                math(EXPR _last_mode "${_mode_count} - 1")
                foreach(_mode_index RANGE 0 ${_last_mode})
                    string(JSON source_modes GET "${_source_shapes}" modes_by_dimension ${source_dim} ${_mode_index})
                    set(_source_tu "${_rhs_include}/FiniteVolumeSourceKernels_${source_model}_C${source_width}_D${source_dim}_M${source_modes}.cu")
                    configure_file(Discretization/CUDA/FiniteVolumeSourceKernels_Shape.cu.in "${_source_tu}" @ONLY)
                    list(APPEND CPPS "${_source_tu}")
                endforeach()
            endforeach()
        endforeach()
    endforeach()
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeSourceKernels.cu)
    list(APPEND CPPS Discretization/CUDA/ModalPositivityKernels.cu Discretization/CUDA/FiniteVolumePositivityKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeGradientKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeLimitedVRKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeReconstructionKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeCharacteristicKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeStateKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeStateKernels_Statistics.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeStateKernels_Profile.cu)
    # CPPS is the explicit production source catalog, including generated RHS
    # wrappers. Source properties apply to both static/shared objects, without
    # propagating to host code, headers' consumers or standalone CUDA tests.
    set(_cauchy_checked_kernel_sources ${CPPS})
    list(FILTER _cauchy_checked_kernel_sources INCLUDE REGEX "\\.cu$")
    dnds_cuda_kernel_assertions(${_cauchy_checked_kernel_sources})
endif()
dnds_add_lib(cauchy "${CPPS}" "physics;cfv;geom;dnds;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
dnds_add_lib(cauchy_shared "${CPPS}" "physics_shared;cfv_shared;geom_shared;dnds_shared;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
foreach(_consumer cauchy cauchy_shared)
    target_include_directories(${_consumer} PRIVATE "${_cpu_flux_include}")
    add_dependencies(${_consumer} ${_cpu_flux_codegen})
endforeach()
if(DNDS_USE_CUDA)
    foreach(_rhs_consumer cauchy cauchy_shared)
        target_include_directories(${_rhs_consumer} PRIVATE "${_rhs_include}")
        add_dependencies(${_rhs_consumer} ${_rhs_codegen})
    endforeach()
endif()
target_link_libraries(cauchy PUBLIC DNDS::ComputeScalar)
# Public state, FV and policy headers use these modules' types and inline
# functions. Static consumers already receive their counterparts via dnds_add_lib.
# Expose the native interfaces without exporting the optional provider list.
target_link_libraries(cauchy_shared PUBLIC DNDS::ComputeScalar
                      INTERFACE physics_shared cfv_shared geom_shared dnds_shared MPI::MPI_CXX)
install(TARGETS cauchy_shared LIBRARY DESTINATION "DNDSR/lib" COMPONENT py OPTIONAL)
install(TARGETS cauchy_shared LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib" COMPONENT py OPTIONAL)
dnds_add_py_module(cauchy_pybind11 "Bindings/cauchy_pybind11.cpp;Bindings/OperatorsBindInternal.cpp;Bindings/Operators_bind.cpp;Bindings/Operators_bind_UpdatePolicy.cpp;Bindings/Operators_bind_State.cpp;Bindings/Operators_bind_Linear.cpp;Bindings/Operators_bind_Nonlinear.cpp;Bindings/Operators_bind_FiniteVolume.cpp" "cauchy_shared" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
install(TARGETS cauchy_pybind11 DESTINATION "DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)
install(TARGETS cauchy_pybind11 DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)

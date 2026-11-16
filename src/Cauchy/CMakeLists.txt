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
    IO/FiniteVolume_SurfaceOutput.cpp
    IO/OutputFields.cpp
    IO/PreparedCheckpoint.cpp
    IO/PreparedVisualization.cpp
    IO/OutputQueue.cpp
    IO/AsyncOutput.cpp
    IO/CPU/OutputFieldsKernels.cpp
    IO/Simulation_Checkpoint.cpp
    IO/ProgressOutput.cpp
    IO/OutputPlan.cpp
    IO/OutputCoordinator.cpp

    Riemann/RiemannFlux.cpp)
if(DNDS_USE_CUDA)
    list(APPEND CPPS Interface/CUDA/InterfacePacketKernels.cu)
    list(APPEND CPPS Implicit/CUDA/FrozenJacobianKernels.cu)
    list(APPEND CPPS Implicit/CUDA/FiniteVolumeJacobianKernels.cu)
    list(APPEND CPPS Implicit/CUDA/FiniteVolumeJacobianKernels_Faces.cu)
    list(APPEND CPPS Implicit/CUDA/LinearBlockKernels.cu)
    list(APPEND CPPS Core/CUDA/TimeStageKernels.cu)
    list(APPEND CPPS State/CUDA/FieldKernels.cu)
    list(APPEND CPPS State/CUDA/StateDiagnosticsKernels.cu)
    list(APPEND CPPS Discretization/CUDA/BoundaryForceKernels.cu)
    list(APPEND CPPS IO/CUDA/OutputFieldsKernels.cu)
    list(APPEND CPPS Implicit/CUDA/LinearDiagonalKernels.cu)
    list(APPEND CPPS Implicit/CUDA/NonlinearKernels.cu Implicit/CUDA/NonlinearKernels_Positivity.cu)
    list(APPEND CPPS Multigrid/CUDA/PolynomialMultigridKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeAuxiliaryKernels.cu Discretization/CUDA/FiniteVolumeBoundaryKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeRHSKernels.cu Discretization/CUDA/FiniteVolumeRHSKernels_Placement.cpp)
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
    # Keep this local to the experimental segments TU, never all RHS policies.
    set_property(SOURCE Discretization/CUDA/FiniteVolumeRHSKernels_Segments.cu
        APPEND PROPERTY COMPILE_OPTIONS "--maxrregcount=128")
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeSourceKernels.cu)
    list(APPEND CPPS Discretization/CUDA/ModalPositivityKernels.cu Discretization/CUDA/FiniteVolumePositivityKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeGradientKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeLimitedVRKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeReconstructionKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeStateKernels.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeStateKernels_Statistics.cu)
    list(APPEND CPPS Discretization/CUDA/FiniteVolumeStateKernels_Profile.cu)
endif()
dnds_add_lib(cauchy "${CPPS}" "physics;cfv;geom;dnds;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
dnds_add_lib(cauchy_shared "${CPPS}" "physics_shared;cfv_shared;geom_shared;dnds_shared;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
target_link_libraries(cauchy PUBLIC DNDS::ComputeScalar)
target_link_libraries(cauchy_shared PUBLIC DNDS::ComputeScalar)
install(TARGETS cauchy_shared LIBRARY DESTINATION "DNDSR/lib" COMPONENT py OPTIONAL)
install(TARGETS cauchy_shared LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib" COMPONENT py OPTIONAL)
dnds_add_py_module(cauchy_pybind11 "Bindings/cauchy_pybind11.cpp;Bindings/OperatorsBindInternal.cpp;Bindings/Operators_bind.cpp;Bindings/Operators_bind_State.cpp;Bindings/Operators_bind_Linear.cpp;Bindings/Operators_bind_Nonlinear.cpp;Bindings/Operators_bind_FiniteVolume.cpp" "cauchy_shared;cfv_shared;geom_shared;dnds_shared;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
install(TARGETS cauchy_pybind11 DESTINATION "DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)
install(TARGETS cauchy_pybind11 DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)

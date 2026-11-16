# message("source dir is ${PROJECT_SOURCE_DIR}")

# project(CFV LANGUAGES CXX)

set(LIBNAME cfv)

set(LINKS
# geom
# dnds
# ${DNDS_EXTERNAL_LIBS}
)

set(LINKS_SHARED
geom_shared
dnds_shared
${DNDS_EXTERNAL_LIBS}
)

set(CPPS
Legacy/FiniteVolume.cpp
Legacy/VRDefines.cpp
Legacy/VariationalReconstruction.cpp
Legacy/ModelEvaluator.cpp
Prepare/FiniteVolumePlan.cpp
Prepare/FiniteVolumePlan_Reconstruction.cpp
Detail/ReconstructionData_FaceBasis.cpp
Prepare/FiniteVolumePlan_Placement.cpp
Prepare/GradientPlan.cpp
Prepare/GradientPlan_Placement.cpp
Prepare/LimitedVRPlan.cpp
Prepare/LimitedVRPlan_Placement.cpp
Operators/GradientPlan_Operators.cpp
Operators/GradientPlan_Queued.cpp
Operators/GradientPlan_Conversion.cpp
Operators/GradientPlan_Limiter.cpp
CPU/GradientKernels.cpp
CPU/LimitedVRKernels.cpp
Operators/PolynomialSamplingPlan.cpp
CPU/PolynomialSamplingKernels.cpp
CPU/PolynomialSamplingKernels_Values.cpp
CPU/PolynomialSamplingKernels_ValueGradient.cpp
Operators/PolynomialSamplingPlan_Values.cpp
Operators/PolynomialSamplingPlan_ValueGradient.cpp
Operators/LimitedVRPlan_Operators.cpp
Operators/LimitedVRPlan_Queued.cpp
Prepare/PolynomialSamplingPlan.cpp
Prepare/PolynomialSamplingPlan_Sharing.cpp
Plans/DistributedInterfaceSamplingPlan.cpp
CPU/DistributedInterfaceSamplingKernels.cpp
CPU/ReconstructionBasisKernels.cpp
CPU/ReconstructionKernels.cpp
Operators/FiniteVolumePlan_Reconstruction.cpp
Operators/FiniteVolumePlan_Linear.cpp
)
if (DNDS_USE_CUDA)
list(APPEND CPPS Operators/CUDA/CharacteristicKernels_CUDA.cu Operators/CUDA/GradientKernels_CUDA.cu Operators/CUDA/LimitedVRKernels_CUDA.cu Operators/CUDA/PolynomialSamplingKernels_CUDA.cu Legacy/VRDefines.cu Operators/CUDA/ReconstructionDevice.cu Operators/CUDA/ReconstructionTensorCore.cu Operators/CUDA/ReconstructionTensorShared.cu)
list(APPEND CPPS Operators/CUDA/ReconstructionDeviceKernels_TF32.cu Operators/CUDA/ReconstructionDeviceKernels_Raw.cu Operators/CUDA/ReconstructionDeviceKernels_RawTensor.cu)
list(APPEND CPPS Operators/CUDA/DistributedInterfaceSamplingKernels.cu)
endif()

file(GLOB EXPLICIT_INST_CPP_FILES CONFIGURE_DEPENDS "Legacy/ExplicitInstantiations/*.cpp")

list(APPEND CPPS ${EXPLICIT_INST_CPP_FILES})

dnds_add_lib(${LIBNAME}        "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
            OFF ON ON) # SHARED, FAST, EXCLUDE

dnds_add_lib(${LIBNAME}_shared "${CPPS}" "${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
            ON ON ON) # SHARED, FAST, EXCLUDE

target_link_libraries(${LIBNAME} PUBLIC DNDS::ComputeScalar)
target_link_libraries(${LIBNAME}_shared PUBLIC DNDS::ComputeScalar)

install(TARGETS ${LIBNAME}_shared
    LIBRARY DESTINATION "DNDSR/lib"
    ARCHIVE DESTINATION "DNDSR/lib"
    COMPONENT py
)

# Also install into the python package tree for editable installs
install(TARGETS ${LIBNAME}_shared
    LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib"
    COMPONENT py
)

set(CPPS_BIND
Legacy/Benchmarks/BenchmarkFiniteVolume.cpp
Plans/FiniteVolumePlan_bind.cpp
Bindings/cfv_pybind11.cpp
Legacy/FiniteVolume_bind.cpp
Legacy/ModelEvaluator_bind.cpp
Legacy/VariationalReconstruction_bind.cpp
)

if(DNDS_USE_CUDA)
    list(APPEND CPPS_BIND Legacy/Benchmarks/BenchmarkFiniteVolume.cu)
endif()

dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
    TARGETS ${LIBNAME}_pybind11
    DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/CFV/_ext"
    RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
    COMPONENT py
)

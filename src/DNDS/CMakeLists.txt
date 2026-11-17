# message("source dir is ${PROJECT_SOURCE_DIR}")

# project(DNDS LANGUAGES CXX)

set(DNDS_PCH_CPPS
Support/PCH/EigenPCH.cpp
Support/PCH/ExprtkPCH.cpp)
set(DNDS_PCH_HPPS
Support/PCH/EigenPCH.hpp
Support/PCH/ExprtkPCH.hpp)

dnds_add_lib_pch(dnds_PCHs "${DNDS_PCH_CPPS}" "${DNDS_PCH_HPPS}" "" 
    OFF OFF) # SHARED,FAST
set(DNDS_PCH_TARGETS dnds_PCHs)
dnds_variable_to_parent_scope(DNDS_PCH_TARGETS)

dnds_add_lib_pch(dnds_PCHs_fast "${DNDS_PCH_CPPS}" "${DNDS_PCH_HPPS}" ""
    OFF ON) # SHARED,FAST
set(DNDS_PCH_TARGETS_FAST dnds_PCHs_fast)
dnds_variable_to_parent_scope(DNDS_PCH_TARGETS_FAST)

set(LIBNAME dnds)
find_package(Threads REQUIRED)
if(DNDS_USE_CALIPER)
    find_package(caliper CONFIG REQUIRED)
endif()

set(LINKS
${DNDS_EXTERNAL_LIBS}
Threads::Threads
)
if(DNDS_USE_NCCL)
    find_path(DNDS_NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_ROOT} ENV NCCL_ROOT PATH_SUFFIXES include)
    find_library(DNDS_NCCL_LIBRARY NAMES nccl libnccl.so.2 HINTS ${NCCL_ROOT} ENV NCCL_ROOT PATH_SUFFIXES lib lib64)
    if(NOT EXISTS "${DNDS_NCCL_INCLUDE_DIR}/nccl.h" OR NOT EXISTS "${DNDS_NCCL_LIBRARY}")
        message(FATAL_ERROR "NCCL is enabled (the CUDA default), but headers/runtime were not found. Set NCCL_ROOT or DNDS_NCCL_INCLUDE_DIR/DNDS_NCCL_LIBRARY, or explicitly opt out with -DDNDS_USE_NCCL=OFF.")
    endif()
    add_library(DNDS_NCCL UNKNOWN IMPORTED GLOBAL)
    set_target_properties(DNDS_NCCL PROPERTIES IMPORTED_LOCATION "${DNDS_NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${DNDS_NCCL_INCLUDE_DIR}")
endif()

set(CPPS
MPI/MPI.cpp
Core/Defines.cpp
Config/EnvReader.cpp
Device/DeviceStorage.cpp
Runtime/Storage/Buffer.cpp
Runtime/Storage/AllocationDiagnostics.cpp
Runtime/Execution/EagerEngine.cpp
Runtime/Execution/AccessTracking.cpp
Runtime/Execution/CUDAExecution.cpp
Runtime/Communication/Exchange.cpp
Runtime/Storage/ElementSchema.cpp
Runtime/Communication/RowTransfer.cpp
Runtime/Communication/NCCLTransfer.cpp
Algebra/DofAlgebra.cpp
Algebra/AmgXPreconditioner.cpp
Algebra/AmgXDiagnostics.cpp
Algebra/BatchedSVD.cpp
Algebra/LocalProducts.cpp
Diagnostics/Profiling.cpp
Diagnostics/Timeline.cpp
Math/HardEigen.cpp
Serializer/SerializerBase.cpp
Serializer/SerializerJSON.cpp
Serializer/SerializerH5.cpp
Config/ExprtkWrapper.cpp
Config/JsonSchema.cpp
Legacy/Vector.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_1.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_2.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_3.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_4.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_5.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_6.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_7.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_8.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_D.cpp
Legacy/ArrayDOF_inst/ArrayDOF_op_I.cpp
)

if (DNDS_USE_CUDA)
    list(APPEND CPPS Algebra/DofAlgebra_CUDA.cu)
    list(APPEND CPPS Algebra/LocalProducts_CUDA.cu)
    list(APPEND CPPS Runtime/Communication/RowTransfer_CUDA.cu)
    list(APPEND CPPS Device/DeviceStorage_CUDA.cu)
    list(APPEND CPPS
        Legacy/ArrayDOF_inst/ArrayDOF_op_1.cu
        Legacy/ArrayDOF_inst/ArrayDOF_op_2.cu
        Legacy/ArrayDOF_inst/ArrayDOF_op_3.cu
        Legacy/ArrayDOF_inst/ArrayDOF_op_4.cu
        Legacy/ArrayDOF_inst/ArrayDOF_op_5.cu
        Legacy/ArrayDOF_inst/ArrayDOF_op_6.cu
        Legacy/ArrayDOF_inst/ArrayDOF_op_7.cu
        Legacy/ArrayDOF_inst/ArrayDOF_op_8.cu
        Legacy/ArrayDOF_inst/ArrayDOF_op_D.cu
        Legacy/ArrayDOF_inst/ArrayDOF_op_I.cu)
endif()

message(STATUS "DNDS_PCH_TARGETS ${DNDS_PCH_TARGETS}")
message(STATUS "DNDS_PCH_TARGETS_FAST ${DNDS_PCH_TARGETS_FAST}")
message(STATUS "CMAKE_DISABLE_PRECOMPILE_HEADERS ${CMAKE_DISABLE_PRECOMPILE_HEADERS}")

dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON ON)  # SHARED,FAST, EXCLUDE

dnds_add_lib(${LIBNAME}_shared "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE
if(DNDS_USE_CALIPER)
    foreach(_target ${LIBNAME} ${LIBNAME}_shared)
        target_link_libraries(${_target} PRIVATE caliper)
        target_compile_definitions(${_target} PRIVATE DNDS_USE_CALIPER)
    endforeach()
endif()
if(DNDS_USE_CUDA)
    target_link_libraries(${LIBNAME} PRIVATE CUDA::cusolver)
    target_link_libraries(${LIBNAME}_shared PRIVATE CUDA::cusolver)
    # Installed Python libraries must keep the toolkit selected at configure
    # time, including transitive AmgX libraries. The host may have several
    # same-soname CUDA releases in ldconfig; none is an implicit substitute.
    get_target_property(_DNDS_CUDART_LIBRARY CUDA::cudart IMPORTED_LOCATION)
    get_filename_component(_DNDS_CUDA_RUNTIME_DIR "${_DNDS_CUDART_LIBRARY}" DIRECTORY)
    set_property(TARGET ${LIBNAME}_shared APPEND PROPERTY INSTALL_RPATH "${_DNDS_CUDA_RUNTIME_DIR}")
endif()
if(DNDS_USE_AMGX)
    target_link_libraries(${LIBNAME} PRIVATE DNDS::AmgX)
    target_link_libraries(${LIBNAME}_shared PRIVATE DNDS::AmgX)
    get_filename_component(_DNDS_AMGX_LIBRARY_DIR "${DNDS_AMGX_LIBRARY}" DIRECTORY)
    set_property(TARGET ${LIBNAME}_shared APPEND PROPERTY INSTALL_RPATH "${_DNDS_AMGX_LIBRARY_DIR}")
endif()
if(DNDS_USE_NCCL)
    # The static-library helper links dependencies INTERFACE-only. This source
    # also needs the provider headers while compiling the library itself.
    target_link_libraries(${LIBNAME} PRIVATE DNDS_NCCL)
    target_link_libraries(${LIBNAME}_shared PRIVATE DNDS_NCCL)
    target_compile_definitions(${LIBNAME} PRIVATE DNDS_USE_NCCL)
    target_compile_definitions(${LIBNAME}_shared PRIVATE DNDS_USE_NCCL)
    # NCCL is an explicit external dependency, not a copied Torch payload.
    # Preserve the configured runtime location for the locally installed Python
    # package too; installing strips the ordinary build-tree RPATH.
    get_filename_component(_DNDS_NCCL_LIBRARY_DIR "${DNDS_NCCL_LIBRARY}" DIRECTORY)
    set_property(TARGET ${LIBNAME}_shared APPEND PROPERTY INSTALL_RPATH "${_DNDS_NCCL_LIBRARY_DIR}")
endif()

# install(TARGETS ${LIBNAME}
#     LIBRARY DESTINATION "DNDSR/lib"
#     ARCHIVE DESTINATION "DNDSR/lib"
#     COMPONENT libs
# )

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
Bindings/dnds_pybind11.cpp
Runtime/Interop/Runtime_bind.cpp
Algebra/Algebra_bind.cpp
Runtime/Execution/Eager_bind.cpp
Runtime/Interop/MPI_bind.cpp
Runtime/Interop/DLPack_bind.cpp
Bindings/Defines_bind.cpp
Bindings/MPI_bind.cpp
Legacy/Array_bind.cpp
Legacy/ArrayDerived/ArrayAdjacency_bind.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind.cpp
Legacy/ArrayDerived/ArrayEigenVector_bind.cpp
Legacy/ArrayDOF_bind.cpp
Legacy/Array_bind_offset/Array_bind_1.cpp
Legacy/Array_bind_offset/Array_bind_2.cpp
Legacy/Array_bind_offset/Array_bind_3.cpp
Legacy/Array_bind_offset/Array_bind_4.cpp
Legacy/Array_bind_offset/Array_bind_5.cpp
Legacy/Array_bind_offset/Array_bind_6.cpp
Legacy/Array_bind_offset/Array_bind_7.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_n1.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_n2.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_n3.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_n4.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_n5.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_n6.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_n7.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_n8.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_nD.cpp
Legacy/ArrayDerived/ArrayEigenMatrix_bind_cols/ArrayEigenMatrix_bind_nI.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/ArrayEigenUniMatrixBatch_bind_n1.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/ArrayEigenUniMatrixBatch_bind_n2.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/ArrayEigenUniMatrixBatch_bind_n3.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/ArrayEigenUniMatrixBatch_bind_n4.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/ArrayEigenUniMatrixBatch_bind_n5.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/ArrayEigenUniMatrixBatch_bind_n6.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/ArrayEigenUniMatrixBatch_bind_n7.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/ArrayEigenUniMatrixBatch_bind_n8.cpp
Legacy/ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/ArrayEigenUniMatrixBatch_bind_nD.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_n1.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_n2.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_n3.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_n4.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_n5.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_n6.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_n7.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_n8.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_nD.cpp
Legacy/ArrayDOF_bind_cols/ArrayDOF_bind_nI.cpp
Serializer/Serializer_bind.cpp
)


dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE


install(
  TARGETS ${LIBNAME}_pybind11
  DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/DNDS/_ext"
  RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
  COMPONENT py
)

# install(FILES ${DNDS_EXTERNAL_LIBS_REAL}
#     DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/_internal"
# )


# # this might not be a good idea.
# install(
#     RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
#     DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/_internal
#     # POST_EXCLUDE_REGEXES ${DNDS_EXTERNAL_LIBS_DIRS_REGEX}
#     )



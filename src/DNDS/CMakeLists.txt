cmake_minimum_required(VERSION 3.1)

# message("source dir is ${PROJECT_SOURCE_DIR}")
include("${PROJECT_SOURCE_DIR}/cmakeCommonUtils.cmake") # from father

# project(DNDS LANGUAGES CXX)

set(DNDS_PCH_CPPS
EigenPCH.cpp
ExprtkPCH.cpp)
set(DNDS_PCH_HPPS
EigenPCH.hpp
ExprtkPCH.hpp)

dnds_add_lib_pch(dnds_PCHs "${DNDS_PCH_CPPS}" "${DNDS_PCH_HPPS}" "" 
    OFF OFF) # SHARED,FAST
set(DNDS_PCH_TARGETS dnds_PCHs)
dnds_variable_to_parent_scope(DNDS_PCH_TARGETS)

dnds_add_lib_pch(dnds_PCHs_fast "${DNDS_PCH_CPPS}" "${DNDS_PCH_HPPS}" ""
    OFF ON) # SHARED,FAST
set(DNDS_PCH_TARGETS_FAST dnds_PCHs_fast)
dnds_variable_to_parent_scope(DNDS_PCH_TARGETS_FAST)

set(LIBNAME dnds)

set(LINKS
${DNDS_EXTERNAL_LIBS}
)

set(CPPS
MPI.cpp
Defines.cpp
DeviceStorage.cpp
Profiling.cpp
HardEigen.cpp
SerializerBase.cpp
SerializerJSON.cpp
SerializerH5.cpp
ExprtkWrapper.cpp
)

file(GLOB ARRAYDOF_OP_SOURCES "ArrayDOF_inst/*.cpp")
list(APPEND CPPS ${ARRAYDOF_OP_SOURCES})

if (DNDS_USE_CUDA)
    list(APPEND CPPS DeviceStorage_CUDA.cu)
    file(GLOB ARRAYDOF_OP_SOURCES_CUDA "ArrayDOF_inst/*.cu")
    list(APPEND CPPS ${ARRAYDOF_OP_SOURCES_CUDA})
endif()

message(STATUS "DNDS_PCH_TARGETS ${DNDS_PCH_TARGETS}")
message(STATUS "DNDS_PCH_TARGETS_FAST ${DNDS_PCH_TARGETS_FAST}")
message(STATUS "CMAKE_DISABLE_PRECOMPILE_HEADERS ${CMAKE_DISABLE_PRECOMPILE_HEADERS}")

dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON ON)  # SHARED,FAST, EXCLUDE

dnds_add_lib(${LIBNAME}_shared "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

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


set(CPPS_BIND
dnds_pybind11.cpp
Defines_bind.cpp
MPI_bind.cpp
Array_bind.cpp
ArrayDerived/ArrayAdjacency_bind.cpp
ArrayDerived/ArrayEigenMatrix_bind.cpp
ArrayDerived/ArrayEigenUniMatrixBatch_bind.cpp
ArrayDerived/ArrayEigenVector_bind.cpp
ArrayDOF_bind.cpp
Serializer_bind.cpp
)

file(GLOB ARRAY_BIND_OFFSET_SOURCES "Array_bind_offset/*.cpp")
list(APPEND CPPS_BIND ${ARRAY_BIND_OFFSET_SOURCES})
file(GLOB ARRAYEIGENMATRIX_COLS_SOURCES "ArrayDerived/ArrayEigenMatrix_bind_cols/*.cpp")
list(APPEND CPPS_BIND ${ARRAYEIGENMATRIX_COLS_SOURCES})
file(GLOB ARRAYEIGENUNIMATRIXBATCH_COLS_SOURCES "ArrayDerived/ArrayEigenUniMatrixBatch_bind_cols/*.cpp")
list(APPEND CPPS_BIND ${ARRAYEIGENUNIMATRIXBATCH_COLS_SOURCES})

file(GLOB ARRAYDOF_COLS_SOURCES "ArrayDOF_bind_cols/*.cpp")
list(APPEND CPPS_BIND ${ARRAYDOF_COLS_SOURCES})


dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE


install(
  TARGETS ${LIBNAME}_pybind11
  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/_internal"
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



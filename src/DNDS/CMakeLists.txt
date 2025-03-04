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
Profiling.cpp
HardEigen.cpp
SerializerJSON.cpp
SerializerH5.cpp
ExprtkWrapper.cpp
)
message(STATUS "DNDS_PCH_TARGETS ${DNDS_PCH_TARGETS}")
message(STATUS "DNDS_PCH_TARGETS_FAST ${DNDS_PCH_TARGETS_FAST}")
message(STATUS "CMAKE_DISABLE_PRECOMPILE_HEADERS ${CMAKE_DISABLE_PRECOMPILE_HEADERS}")

dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON)  # SHARED,FAST

dnds_add_lib(${LIBNAME}_shared "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON)  # SHARED,FAST

install(TARGETS ${LIBNAME}_shared;${LIBNAME}
    LIBRARY DESTINATION "DNDSR/bin"
    ARCHIVE DESTINATION "DNDSR/lib"
)

set(CPPS_BIND
MPI_bind.cpp
)

dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON)  # SHARED,FAST


install(
  TARGETS ${LIBNAME}_pybind11
  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/_internal"
  RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
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



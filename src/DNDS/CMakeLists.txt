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

dnds_add_lib_pch(dnds_PCHs "${DNDS_PCH_CPPS}" "${DNDS_PCH_HPPS}" "")
set(DNDS_PCH_TARGETS dnds_PCHs)
dnds_variable_to_parent_scope(DNDS_PCH_TARGETS)

dnds_add_lib_pch_fast(dnds_PCHs_fast "${DNDS_PCH_CPPS}" "${DNDS_PCH_HPPS}" "")
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

dnds_add_lib_fast(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}")


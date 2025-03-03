cmake_minimum_required(VERSION 3.1)

# message("source dir is ${PROJECT_SOURCE_DIR}")
include("${PROJECT_SOURCE_DIR}/cmakeCommonUtils.cmake") # from father

# project(GEOM LANGUAGES CXX)

set(LIBNAME geom)

set(LINKS
# dnds
# ${DNDS_EXTERNAL_LIBS}
)

message(STATUS "DNDS_PCH_TARGETS: ${DNDS_PCH_TARGETS}")

set(CPPS
Elements.cpp
Mesh.cpp
Mesh_Plts.cpp
Mesh_Serial_BuildCell2Cell.cpp
Mesh_Serial_ReadFromCGNS.cpp
Mesh_Serial_Partition.cpp
Mesh_Elevation.cpp
Mesh_Elevation_SmoothSolver.cpp
)

dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS}"
    OFF ON)  # SHARED,FAST

dnds_add_lib(${LIBNAME}_shared "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS}"
    ON ON)  # SHARED,FAST


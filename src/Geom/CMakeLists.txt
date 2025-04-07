cmake_minimum_required(VERSION 3.1)

# message("source dir is ${PROJECT_SOURCE_DIR}")
include("${PROJECT_SOURCE_DIR}/cmakeCommonUtils.cmake") # from father

# project(GEOM LANGUAGES CXX)

set(LIBNAME geom)

set(LINKS
# dnds
# ${DNDS_EXTERNAL_LIBS}
)

set(LINKS_SHARED
dnds_shared
${DNDS_EXTERNAL_LIBS}
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
    OFF ON ON)  # SHARED,FAST, EXCLUDE

dnds_add_lib(${LIBNAME}_shared "${CPPS}" "${LINKS_SHARED}" "${DNDS_PCH_TARGETS}"
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
geom_pybind11.cpp
)

dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
  TARGETS ${LIBNAME}_pybind11
  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/_internal"
  RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
  COMPONENT py
)


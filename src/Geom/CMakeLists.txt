# message("source dir is ${PROJECT_SOURCE_DIR}")

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
PeriodicInfo.cpp
Mesh/MeshConnectivity.cpp
Mesh/MeshConnectivity_Interpolate.cpp
Mesh/MeshConnectivity_Ghost.cpp
Mesh/Mesh.cpp
Mesh/Mesh_Legacy.cpp
Mesh/Mesh_Plts.cpp
Mesh/Mesh_Serial_BuildCell2Cell.cpp
Mesh/Mesh_Serial_ReadFromCGNS.cpp
Mesh/Mesh_Serial_Partition.cpp
Mesh/Mesh_ReadSerializeDistributed.cpp
Mesh/Mesh_Elevation.cpp
Mesh/Mesh_Elevation_SmoothSolver.cpp
Mesh/Mesh_Elevation_SmoothV2.cpp
Mesh/Mesh_WallDist.cpp
Mesh/SerialAdjReordering.cpp
)

if (DNDS_USE_CUDA)
list(APPEND CPPS Mesh/Mesh.cu PeriodicInfo.cu)
endif()

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

# Also install into the python package tree for editable installs
install(TARGETS ${LIBNAME}_shared
    LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib"
    COMPONENT py
)

set(CPPS_BIND
geom_pybind11.cpp
Elements_bind.cpp
Mesh/Mesh_bind.cpp
)

dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
  TARGETS ${LIBNAME}_pybind11
  DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/Geom/_ext"
  RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
  COMPONENT py
)


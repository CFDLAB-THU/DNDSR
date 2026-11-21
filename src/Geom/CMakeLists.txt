# message("source dir is ${PROJECT_SOURCE_DIR}")

# project(GEOM LANGUAGES CXX)

set(LIBNAME geom)

set(LINKS dnds)

set(LINKS_SHARED
dnds_shared
${DNDS_EXTERNAL_LIBS}
)

message(STATUS "DNDS_PCH_TARGETS: ${DNDS_PCH_TARGETS}")

set(CPPS
Elements/Elements.cpp
Periodic/PeriodicInfo.cpp
Interface/BoundaryInterface.cpp
Spatial/ElementGeometry.cpp
Spatial/BoundingBoxTree.cpp
Spatial/SurfacePointMap.cpp
Connectivity/MeshConnectivity.cpp
Access/RelationAccess_Operations.cpp
Connectivity/AdjacencyRegistry_Operations.cpp
Connectivity/AdjacencyRegistry.cpp
Connectivity/MeshConnectivity_Ghost.cpp
Unstructured/UnstructuredMesh.cpp
Access/MeshAccess.cpp
Access/CellFaceIncidence.cpp
Access/UnstructuredMesh_Access.cpp
Geometry/UnstructuredMesh_Geometry.cpp
Geometry/MeshEdit.cpp
Unstructured/UnstructuredMeshSnapshot.cpp
Unstructured/UnstructuredMeshState.cpp
Unstructured/NativeMeshBuilder.cpp
Transition/NativeMeshTransition.cpp
Unstructured/NativeMeshPreparation.cpp
Unstructured/NativeMeshBuilder_Construct.cpp
Connectivity/NativeMeshTopology_Prepare.cpp
Unstructured/UnstructuredMesh_Topology.cpp
Transition/NativeMeshPartitionGraph_Owned.cpp
Transition/NativeMeshAssembler.cpp
IO/NativeMeshArchive.cpp
Compat/NativeMeshLegacyArchive.cpp
IO/NativeMeshCGNS.cpp
IO/NativeMeshOpenFOAM.cpp
IO/NativeMeshPeriodicImport.cpp
Connectivity/NativeMeshTopology.cpp
Connectivity/NativeMeshClosure.cpp
Refinement/NativeMeshRefinement.cpp
Refinement/NativeMeshRefinement_Elevation.cpp
Refinement/NativeMeshExtraction.cpp
Geometry/NativeMeshSmoothing.cpp
IO/NativeMeshOutput.cpp
IO/NativeMeshSerialOutput.cpp
Transition/MeshTransition.cpp
Access/EntitySpace.cpp
IO/UnstructuredMesh_Output.cpp
Compat/UnstructuredMesh_SerialPartition.cpp
Refinement/UnstructuredMesh_Elevation.cpp
Refinement/UnstructuredMesh_ElevationSmooth.cpp
Refinement/UnstructuredMesh_ElevationElasticity.cpp
Geometry/UnstructuredMesh_WallDistance.cpp
Transition/ReorderPlan.cpp
Transition/SerialAdjReordering.cpp
)

if (DNDS_USE_CUDA)
list(APPEND CPPS Periodic/PeriodicInfo.cu)
endif()

dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS}"
    OFF ON ON)  # SHARED,FAST, EXCLUDE

dnds_add_lib(${LIBNAME}_shared "${CPPS}" "${LINKS_SHARED}" "${DNDS_PCH_TARGETS}"
    ON ON ON)  # SHARED,FAST, EXCLUDE
# Public headers and inline accessors expose the underlying native module.
target_link_libraries(${LIBNAME}_shared INTERFACE dnds_shared)

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
Interface/BoundaryInterface_bind.cpp
Bindings/geom_pybind11.cpp
Elements/Elements_bind.cpp
Unstructured/UnstructuredMesh_bind.cpp
)

list(APPEND CPPS_BIND
    Compat/AdjPairTracked_bind_inst/AdjPairTracked_bind_1_1_N.cpp
    Compat/AdjPairTracked_bind_inst/AdjPairTracked_bind_2_2_N.cpp
    Compat/AdjPairTracked_bind_inst/AdjPairTracked_bind_I_I_N.cpp)

dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
  TARGETS ${LIBNAME}_pybind11
  DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/Geom/_ext"
  RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
  COMPONENT py
)


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
FiniteVolume.cpp
VRDefines.cpp
VariationalReconstruction.cpp
ModelEvaluator.cpp
test_FiniteVolume.cpp
)
if (DNDS_USE_CUDA)
list(APPEND CPPS VRDefines.cu test_FiniteVolume.cu)
endif()

file(GLOB EXPLICIT_INST_CPP_FILES CONFIGURE_DEPENDS "_explicit_instantiation/*.cpp")

list(APPEND CPPS ${EXPLICIT_INST_CPP_FILES})

dnds_add_lib(${LIBNAME}        "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
            OFF ON ON) # SHARED, FAST, EXCLUDE

dnds_add_lib(${LIBNAME}_shared "${CPPS}" "${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
            ON ON ON) # SHARED, FAST, EXCLUDE

install(TARGETS ${LIBNAME}_shared
    LIBRARY DESTINATION "DNDSR/lib"
    ARCHIVE DESTINATION "DNDSR/lib"
    COMPONENT py
)

set(CPPS_BIND
cfv_pybind11.cpp
FiniteVolume_bind.cpp
ModelEvaluator_bind.cpp
VariationalReconstruction_bind.cpp
)

dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
    TARGETS ${LIBNAME}_pybind11
    DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/_internal"
    RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
    COMPONENT py
)

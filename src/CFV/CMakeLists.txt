cmake_minimum_required(VERSION 3.1)

# message("source dir is ${PROJECT_SOURCE_DIR}")
include("${PROJECT_SOURCE_DIR}/cmakeCommonUtils.cmake") # from father

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
VariationalReconstruction.cpp
_explicit_instantiation/VariationalReconstruction_Reconstruction_2_4.cpp
_explicit_instantiation/VariationalReconstruction_Reconstruction_2_5.cpp
_explicit_instantiation/VariationalReconstruction_Reconstruction_2_6.cpp
_explicit_instantiation/VariationalReconstruction_Reconstruction_2_7.cpp
_explicit_instantiation/VariationalReconstruction_Reconstruction_2_D.cpp
_explicit_instantiation/VariationalReconstruction_Reconstruction_3_5.cpp
_explicit_instantiation/VariationalReconstruction_Reconstruction_3_6.cpp
_explicit_instantiation/VariationalReconstruction_Reconstruction_3_7.cpp
_explicit_instantiation/VariationalReconstruction_Reconstruction_3_D.cpp
_explicit_instantiation/VariationalReconstruction_LimiterProcedure_2_4.cpp
_explicit_instantiation/VariationalReconstruction_LimiterProcedure_2_5.cpp
_explicit_instantiation/VariationalReconstruction_LimiterProcedure_2_6.cpp
_explicit_instantiation/VariationalReconstruction_LimiterProcedure_2_7.cpp
_explicit_instantiation/VariationalReconstruction_LimiterProcedure_2_D.cpp
_explicit_instantiation/VariationalReconstruction_LimiterProcedure_3_5.cpp
_explicit_instantiation/VariationalReconstruction_LimiterProcedure_3_6.cpp
_explicit_instantiation/VariationalReconstruction_LimiterProcedure_3_7.cpp
_explicit_instantiation/VariationalReconstruction_LimiterProcedure_3_D.cpp
)

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
)

dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
    TARGETS ${LIBNAME}_pybind11
    DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/_internal"
    RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
    COMPONENT py
)

cmake_minimum_required(VERSION 3.1)

# message("source dir is ${PROJECT_SOURCE_DIR}")
include("${PROJECT_SOURCE_DIR}/cmakeCommonUtils.cmake") # from father

set(LIBNAME euler_library)

set(LINKS
cfv
geom
dnds
${DNDS_EXTERNAL_LIBS}
)

foreach(item IN LISTS DNDS_Euler_Models_List)
    string(REPLACE "=" ";" keyval ${item})
    list(GET keyval 0 key)
    list(GET keyval 1 value)
    set(CPPS_FAST
    _explicit_instantiation/EulerEvaluator_EvaluateDt_${key}.cpp
    _explicit_instantiation/EulerEvaluator_EvaluateRHS_${key}.cpp
    _explicit_instantiation/EulerEvaluator_${key}.cpp
    _explicit_instantiation/EulerSolver_PrintData_${key}.cpp
    _explicit_instantiation/EulerSolver_Init_${key}.cpp
    )
    set(CPPS
    _explicit_instantiation/EulerSolver_${key}.cpp
    )
    dnds_add_lib_fast(${LIBNAME}_fast_${key} "${CPPS_FAST}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}")
    dnds_add_lib(${LIBNAME}_${key} "${CPPS}" "${LINKS};${LIBNAME}_fast_${key}" "${DNDS_PCH_TARGETS}")
endforeach()

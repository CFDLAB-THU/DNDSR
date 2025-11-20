cmake_minimum_required(VERSION 3.1)

# message("source dir is ${PROJECT_SOURCE_DIR}")
include("${PROJECT_SOURCE_DIR}/cmakeCommonUtils.cmake")

set(LIBNAME euler_library)

set(LINKS
# cfv
# geom
# dnds
# ${DNDS_EXTERNAL_LIBS}
)

set(LINKS_SHARED
cfv_shared
geom_shared
dnds_shared
${DNDS_EXTERNAL_LIBS}
)

set(CPPS_BIND
eulerP_pybind11.cpp
)

dnds_add_py_module(eulerP_pybind11 "${CPPS_BIND}" "${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
  TARGETS eulerP_pybind11
  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/_internal"
  RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
  COMPONENT py
)
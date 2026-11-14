# Optional header-only compute arithmetic. Numerical targets opt in explicitly;
# an enabled build must use one coherent CCCL header family in every consumer.
option(DNDS_USE_CCCL_COMPUTE "Enable experimental CCCL float-float kernel arithmetic" OFF)
set(DNDS_CCCL_SOURCE_ROOT "${PROJECT_SOURCE_DIR}/external/cccl-source" CACHE PATH
    "Pinned CCCL source checkout for experimental compute arithmetic")
add_library(dnds_compute_scalar INTERFACE)
add_library(DNDS::ComputeScalar ALIAS dnds_compute_scalar)
if(DNDS_USE_CCCL_COMPUTE)
    foreach(_part libcudacxx/include/cuda/fpmp cub/cub/cub.cuh thrust/thrust/version.h)
        if(NOT EXISTS "${DNDS_CCCL_SOURCE_ROOT}/${_part}")
            message(FATAL_ERROR "CCCL compute arithmetic needs a coherent source checkout: missing ${_part}")
        endif()
    endforeach()
    # Use ordinary -I, not SYSTEM: nvcc's implicit toolkit headers otherwise
    # take precedence. All three components must originate in this same tree.
    set(_dnds_cccl_headers
        "${DNDS_CCCL_SOURCE_ROOT}/libcudacxx/include"
        "${DNDS_CCCL_SOURCE_ROOT}/thrust"
        "${DNDS_CCCL_SOURCE_ROOT}/cub")
    target_include_directories(dnds_compute_scalar BEFORE INTERFACE ${_dnds_cccl_headers})
    # DNDS libraries also receive explicit toolkit include paths from the
    # external-dependency list. A transitive INTERFACE BEFORE does not precede
    # those direct paths: mixing the toolkit's old __config guard with fpmp's
    # new headers breaks both GCC and nvcc preprocessing. Keep the same pinned
    # family first in direct and transitive include lists. This affects only
    # builds explicitly enabling experimental CCCL, not installed toolkits.
    list(PREPEND DNDS_EXTERNAL_INCLUDES ${_dnds_cccl_headers})
    target_include_directories(dnds_external_deps BEFORE INTERFACE ${_dnds_cccl_headers})
    target_compile_definitions(dnds_compute_scalar INTERFACE DNDS_USE_CCCL_COMPUTE)
    target_compile_features(dnds_compute_scalar INTERFACE cxx_std_17)
    message(STATUS "Experimental compute arithmetic: CCCL headers at ${DNDS_CCCL_SOURCE_ROOT}")
endif()

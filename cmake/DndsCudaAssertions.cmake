include_guard(GLOBAL)

# This option never adds a directory/target/global definition. Only numerical
# CUDA sources explicitly registered below opt in; host and test TUs keep their
# checks. Changing it therefore invalidates just those source objects.
option(DNDS_CUDA_KERNEL_ASSERTS "Keep device assertions in registered CFV/Cauchy CUDA kernel sources" ON)

function(dnds_cuda_kernel_assertions)
    if(DNDS_CUDA_KERNEL_ASSERTS)
        return()
    endif()
    foreach(_source IN LISTS ARGN)
        if(NOT _source MATCHES "\\.cu$")
            message(FATAL_ERROR "Device assertion opt-out requires a CUDA source: ${_source}")
        endif()
        # DNDS_HD_assert*, including the device Eigen assertion hook, uses this
        # existing device-only gate. The host pass of this same TU stays checked.
        set_property(SOURCE "${_source}" APPEND PROPERTY COMPILE_DEFINITIONS DNDS_NDEBUG_DEVICE=1)
    endforeach()
endfunction()

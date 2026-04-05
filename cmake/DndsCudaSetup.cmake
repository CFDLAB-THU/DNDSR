# cmake/DndsCudaSetup.cmake
# Enable CUDA language support and find the CUDA toolkit.
# Only included when DNDS_USE_CUDA is ON.

if(NOT DNDS_USE_CUDA)
    return()
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
# ! disable --options-file for nvcc
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
enable_language(CUDA)
add_compile_definitions(DNDS_USE_CUDA)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA toolkit: ${CUDAToolkit_VERSION}")
else()
    message(FATAL_ERROR "CUDA toolkit required but not found")
endif()
# !
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=-O3")

# CUDA 13.x (CCCL 3.x) moved thrust/cub/libcudacxx headers into a
# cccl/ subdirectory.  nvcc adds this path automatically, but the host
# C++ compiler (used for .cpp files that #include thrust headers via
# DNDS_USE_CUDA guards) does not know about it.  Add the path so that
# #include <thrust/...> works from both .cu and .cpp translation units.
if(CUDAToolkit_INCLUDE_DIRS AND EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cccl")
    set(DNDS_CUDA_CCCL_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}/cccl")
    message(STATUS "Found CCCL include directory: ${DNDS_CUDA_CCCL_INCLUDE_DIR}")
endif()

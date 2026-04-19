# cmake/DndsOptions.cmake
# All user-facing cache options for the DNDSR build.

# need CUDA project support ?
# manual perferences
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

#! why?
# find_package(MPI REQUIRED)

set(DNDS_SUPRESS_INSERT_CHECK ON CACHE BOOL "close the inserted outputs")
set(DNDS_NODEBUG_MODULES "" CACHE STRING "List of modules to exclude from debugging, semicolon-separated")
set(DNDS_USE_DNDS_NDEBUG_MACRO OFF CACHE BOOL "use DNDS_NDEBUG macro")
set(DNDS_USE_NDEBUG_MACRO OFF CACHE BOOL "use NDEBUG macro for eigen output and cassert etc") # performance impact is small so off by default(without -DNDEBUG)
set(DNDS_USE_PARALLEL_MACRO OFF CACHE BOOL "for auto multi-thread?")
set(DNDS_USE_OMP ON CACHE BOOL "enables openmp functionality")
set(DNDS_USE_CUDA OFF CACHE BOOL "enables cuda functionality")
set(DNDS_DIST_MT_USE_OMP OFF CACHE BOOL "use openmp in purely distributed code for multi-treading")
set(DNDS_FAST_BUILD_FAST ON CACHE BOOL "don't -g on basic modules")
set(DNDS_BUILD_TESTS OFF CACHE BOOL "build C++ unit tests (doctest)")
set(DNDS_USE_NO_OMIT_FRAME_POINTER OFf CACHE BOOL "in fast build fast part, use -fno-omit-frame-pointer to get correct call graph in perf's fp mode")
set(DNDS_UNSAFE_MATH_OPT OFF CACHE BOOL "use -funsafe-math-optimizations")
set(DNDS_NATIVE_ARCH     OFF CACHE BOOL "use -march=native")
set(DNDS_LTO OFF CACHE BOOL "use -flto")
set(DNDS_LTO_THIN OFF CACHE BOOL "use -flto=thin")
set(DNDS_PYBIND11_NO_LTO OFF CACHE BOOL "disable lto in pybind11")
cmake_host_system_information(RESULT NPROC
                              QUERY NUMBER_OF_PHYSICAL_CORES)
set(DNDS_LTO_N ${NPROC} CACHE STRING "n in -flto=<n>")
set(DNDS_USE_FULL_TEMPLATE_TRACE OFF CACHE BOOL "use -ftemplate-backtrace-limit=0")
set(DNDS_USE_RDYNAMIC ON CACHE BOOL "use -rdynamic on posix")
set(DNDS_USE_PRECOMPILED_HEADER OFF CACHE BOOL "use precompiled header")

find_program(DNDS_CCACHE_EXEC ccache)
if(DNDS_CCACHE_EXEC AND NOT SKBUILD_PROJECT_NAME) # do not use ccache for pip build
    set(DNDS_USE_CCACHE ON CACHE BOOL "use ccache as compiler launcher")
else()
    set(DNDS_USE_CCACHE OFF CACHE BOOL "use ccache as compiler launcher")
endif()

set(DNDS_EIGEN_USE_BLAS OFF CACHE BOOL "use blas in eigen")
set(DNDS_EIGEN_USE_LAPACK OFF CACHE BOOL "use lapack in eigen")

set(DNDS_SUPERLU_DIST_USE_BLAS OFF CACHE BOOL "use blas in superlu_dist")

set(DNDS_RECORD_COMMIT ON CACHE BOOL "record commit id for each cmake configure")
set(DNDS_VERBOSE_BUILDING OFF CACHE BOOL "make the makefile verbose, overrides CMAKE_VERBOSE_MAKEFILE")
set(DNDS_USE_CLANG_TIDY OFF CACHE BOOL "use clang-tidy")

set(DNDS_GENERATE_COMPILE_COMMANDS OFF CACHE BOOL "generate proper compile_commands.json and post-process for clangd")

# for we need fmtlib to give a shared library
set(BUILD_SHARED_LIBS ON CACHE BOOL "set by DNDS, not used for now")

# -------------------------------------------------------------------
# Record commit hash (uses DNDS_VERSION_COMMIT from DndsVersion.cmake)
# -------------------------------------------------------------------
if(DNDS_RECORD_COMMIT)
    # Full hash for backward compat (some targets compile-define it).
    # DndsVersion.cmake provides the short hash; get the full one too.
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            OUTPUT_VARIABLE DNDS_RECORDED_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _ret
        )
        if(NOT _ret EQUAL 0)
            set(DNDS_RECORDED_COMMIT_HASH "UNKNOWN")
        endif()
    else()
        set(DNDS_RECORDED_COMMIT_HASH "UNKNOWN")
    endif()
    set(DNDS_RECORDED_COMMIT_HASH "${DNDS_RECORDED_COMMIT_HASH}" CACHE INTERNAL
        "Project Public Current Commit Hash")
    message(STATUS "current commit id: ${DNDS_RECORDED_COMMIT_HASH}")
endif()

# -------------------------------------------------------------------
# Verbose / clang-tidy
# -------------------------------------------------------------------
if(DNDS_VERBOSE_BUILDING)
    set(CMAKE_VERBOSE_MAKEFILE ON)
else()
    set(CMAKE_VERBOSE_MAKEFILE OFF)
endif()

if(DNDS_USE_CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-format-style='file';")
else()
    set(CMAKE_CXX_CLANG_TIDY "")
endif()

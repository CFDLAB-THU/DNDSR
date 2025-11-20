## For configuration on Windows, use clang/gcc + makefile
## cmake .. -G "Unix Makefiles" -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang

## cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx ..


# Prevent in-source builds
get_filename_component(_src "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_bin "${CMAKE_BINARY_DIR}" REALPATH)
if(_src STREQUAL _bin)
  message(FATAL_ERROR
    "In-source builds are not allowed.\n"
    "Please create a separate build directory, for example:\n"
    "  cmake -S . -B build\n"
    "Then build with:\n"
    "  cmake --build build")
endif()

if(UNIX OR MINGW)
    cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
else()
    cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
endif()

cmake_policy(SET CMP0082 NEW) # ensures install exec order == declare order



set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE FILEPATH "install place")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Release or Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Release" "Debug" "RelWithDebInfo" "MinSizeRel")

if(DNDS_GENERATE_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()


# ##########################################
project(DNDSR LANGUAGES CXX) #! need to be in front of find_package(MPI)
enable_language(C) #! building C sub-projects

message(STATUS "CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")
set(DNDS_USING_CLANG_WIN OFF)
if(WIN32)
    if(${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang)
        set(DNDS_USING_CLANG_WIN ON)
    endif()
endif()
if(UNIX)
#! we are packaging a most likely correct libstdc++.so 
#! now should do well without explicit importing a libstdc++.so at runtime
#! as fmt is static and libdnds.so is the first to require any libstdc++.so
#! keeping this incase we really need a bundled libstdc++.so
    if(NOT DEFINED LIBSTDCXX_PATH)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
            OUTPUT_VARIABLE LIBSTDCXX_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()

    if(EXISTS "${LIBSTDCXX_PATH}")
        file(INSTALL "${LIBSTDCXX_PATH}" DESTINATION ${CMAKE_INSTALL_PREFIX}/DNDSR/lib/dndsr_external FOLLOW_SYMLINK_CHAIN)
    else()
        message(FATAL_ERROR "libstdc++.so [${LIBSTDCXX_PATH}] told by compiler not found!")
    endif()
endif()

set(DNDS_CFD_EXTERNALS_INSTALL  ${CMAKE_SOURCE_DIR}/external/cfd_externals/install CACHE PATH "path to cfd_externals's install")

set(DNDS_CFD_EXTERNALS_LIB ${DNDS_CFD_EXTERNALS_INSTALL}/lib)
set(DNDS_CFD_EXTERNALS_INC ${DNDS_CFD_EXTERNALS_INSTALL}/include)

if(EXISTS ${DNDS_CFD_EXTERNALS_INSTALL} AND EXISTS ${DNDS_CFD_EXTERNALS_LIB} AND EXISTS ${DNDS_CFD_EXTERNALS_INC})
    set(CMAKE_PREFIX_PATH ${DNDS_CFD_EXTERNALS_INSTALL}) # to override system ones
    message(STATUS "DNDS_CFD_EXTERNALS_INSTALL ${DNDS_CFD_EXTERNALS_INSTALL}")
else()
    message(FATAL_ERROR "cfd_externals install or install/lib or install/include not existent")
endif()

# message("external install hard guess at: external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
# set(HARD_GUESS_PATH external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

# # First Hard Guess
# set(CMAKE_PREFIX_PATH ${HARD_GUESS_PATH})

# need CUDA project support ?
# manual perferences
# set(CMAKE_CXX_STANDARD_REQUIRED 17)
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

if(DNDS_RECORD_COMMIT)
    execute_process(COMMAND git rev-parse HEAD OUTPUT_FILE ${CMAKE_BINARY_DIR}/commitID.txt WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} TIMEOUT 120 RESULT_VARIABLE RET)
    message(STATUS "git rev-parse returned ${RET}")
    if(NOT RET EQUAL 0)
        message(WARNING "git rev-parse failed")
        set(DNDS_RECORDED_COMMIT_HASH "UNKNOWN" CACHE INTERNAL  "Project Public Current Commit Hash")
    else()
        # message(STATUS "current commit id: ")
        # execute_process(COMMAND cat commitID.txt WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TIMEOUT 10)
        file(READ "${CMAKE_BINARY_DIR}/commitID.txt" DNDS_RECORDED_COMMIT_HASH)
        string(STRIP ${DNDS_RECORDED_COMMIT_HASH} DNDS_RECORDED_COMMIT_HASH)
        set(DNDS_RECORDED_COMMIT_HASH ${DNDS_RECORDED_COMMIT_HASH} CACHE INTERNAL  "Project Public Current Commit Hash")
        message(STATUS "current commit id: ${DNDS_RECORDED_COMMIT_HASH}")
    endif()
endif()

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

if(DNDS_LTO)
    if(DNDS_LTO_THIN)
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -flto=thin")
    else()
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -flto=${DNDS_LTO_N}")
    endif()
    # ! todo: add changing to gcc-ar gcc-ranlib gcc-nm
endif()

if(DNDS_USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17) 
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    # set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    enable_language(CUDA)
    add_compile_definitions(DNDS_USE_CUDA)

    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDA toolkit: ${CUDAToolkit_VERSION}")
    else()
        message(FATAL_ERROR "CUDA toolkit required but not found")
    endif()
    # !
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=-O3" )
endif()

if(DNDS_USE_FULL_TEMPLATE_TRACE)
    add_compile_options(-ftemplate-backtrace-limit=0)
endif()


if(UNIX)
    if(DNDS_USE_RDYNAMIC)
        add_compile_options(-rdynamic)
        add_link_options(-rdynamic)
    endif()
endif()


find_package(MPI REQUIRED)
if(UNIX)
    # set(CMAKE_CXX_COMPILER mpicxx CACHE FILEPATH "compiler with your MPI wrapping")
    # set(CMAKE_FIND_LIBRARY_SUFFIXES ".a") # ! using static libs
elseif(MSVC OR WIN32 OR MINGW)
    # nothing
    
    if(MPI_CXX_FOUND)
        set (CMAKE_EXE_LINKER_FLAGS "${MPI_C_LINK_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
    else()
        message((FATAL_ERROR "MPI NOT FOUND"))
    endif()
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()





set(CMAKE_USE_RELATIVE_PATHS ON)




# ##########################################
find_library(DNDS_EXTERNAL_LIB_BACKTRACE NAMES backtrace)
# need zlibstatic first to find the static version on windows
find_library(DNDS_EXTERNAL_LIB_ZLIB NAMES libz.so  zlibstatic z z_D
    PATHS "${DNDS_CFD_EXTERNALS_LIB}" 
    REQUIRED)
find_library(DNDS_EXTERNAL_LIB_HDF5 NAMES libhdf5.so libhdf5.lib hdf5 
    PATHS "${DNDS_CFD_EXTERNALS_LIB}" 
    REQUIRED)
# find_library(DNDS_EXTERNAL_LIB_HDF5_HL NAMES libhdf5_hl.a hdf5_hl PATHS 
#     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
#     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
#      REQUIRED)
find_library(DNDS_EXTERNAL_LIB_CGNS NAMES libcgns.so cgns.lib cgns 
    PATHS "${DNDS_CFD_EXTERNALS_LIB}"
    REQUIRED)
find_library(DNDS_EXTERNAL_LIB_METIS NAMES libmetis.so metis.lib metis 
    PATHS "${DNDS_CFD_EXTERNALS_LIB}"
    REQUIRED)
find_library(DNDS_EXTERNAL_LIB_PARMETIS NAMES libparmetis.so parmetis.lib parmetis 
    PATHS "${DNDS_CFD_EXTERNALS_LIB}"
    REQUIRED)
# find_library(DNDS_EXTERNAL_LIB_TECIO tecio PATHS 
    # "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
    # "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
    # NO_DEFAULT_PATH) #not needed for now
find_library(DNDS_EXTERNAL_LIB_BLAS NAMES openblas blas )
find_library(DNDS_EXTERNAL_LIB_LAPACK NAMES lapack )
find_library(DNDS_EXTERNAL_LIB_LAPACKE NAMES lapacke)

find_path(DNDS_EXTERNAL_INCLUDE_ZLIB zlib.h 
    PATHS "${DNDS_CFD_EXTERNALS_INC}" 
    REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_HDF5 hdf5.h 
    PATHS "${DNDS_CFD_EXTERNALS_INC}"
    REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_CGNS cgnslib.h 
    PATHS "${DNDS_CFD_EXTERNALS_INC}"
    REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_METIS metis.h 
    PATHS "${DNDS_CFD_EXTERNALS_INC}"
    REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_PARMETIS parmetis.h
    PATHS "${DNDS_CFD_EXTERNALS_INC}"
    REQUIRED)
# find_path(DNDS_EXTERNAL_INCLUDE_TECIO TECIO.h PATHS 
#     "${PROJECT_SOURCE_DIR}/external/tecio/include" 
#     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
#     NO_DEFAULT_PATH) #!not needed for now
find_path(DNDS_EXTERNAL_INCLUDE_EIGEN eigen3.pc.in PATHS 
    "${PROJECT_SOURCE_DIR}/external/eigen" 
    "${PROJECT_SOURCE_DIR}/external/eigen-3.4.0" 
    NO_DEFAULT_PATH REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_BOOST boost PATHS 
    "${PROJECT_SOURCE_DIR}/external/boost" 
    "${PROJECT_SOURCE_DIR}/external/boost_1_82_0" 
    NO_DEFAULT_PATH REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_CGAL CGAL PATHS 
    "${PROJECT_SOURCE_DIR}/external/CGAL/include"
    "${PROJECT_SOURCE_DIR}/external/CGAL-5.6/include" 
    "${PROJECT_SOURCE_DIR}/external/CGAL-5.6" 
    NO_DEFAULT_PATH REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_JSON nlohmann PATHS 
    "${PROJECT_SOURCE_DIR}/external/nlohmann"
    "${PROJECT_SOURCE_DIR}/external" 
    NO_DEFAULT_PATH REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_ARGPARSE argparse.hpp PATHS
    "${PROJECT_SOURCE_DIR}/external/argparse/include/argparse"
    "${PROJECT_SOURCE_DIR}/external/argparse-3.0/include/argparse" 
    NO_DEFAULT_PATH REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_EXPRTK exprtk.hpp PATHS 
    "${PROJECT_SOURCE_DIR}/external/exprtk"
    NO_DEFAULT_PATH REQUIRED)
# find_path(DNDS_EXTERNAL_INCLUDE_CPPTRACE cpptrace.hpp PATHS 
#     "${PROJECT_SOURCE_DIR}/external/cpptrace-0.5.1/include/cpptrace" 
#     NO_DEFAULT_PATH REQUIRED)
find_path(DNDS_EXTERNAL_INCLUDE_CPPCODEC base32_hex.hpp PATHS
    "${PROJECT_SOURCE_DIR}/external/cppcodec" 
    NO_DEFAULT_PATH REQUIRED)
# find_path(DNDS_EXTERNAL_INCLUDE_RAPIDJSON rapidjson 
#     PATHS 
#     "${PROJECT_SOURCE_DIR}/external/rapidjson/include" 
#     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
#     NO_DEFAULT_PATH)
find_path(DNDS_EXTERNAL_INCLUDE_NANOFLANN nanoflann.hpp PATHS
    "${PROJECT_SOURCE_DIR}/external/nanoflann"
    "${PROJECT_SOURCE_DIR}/external/nanoflann-1.4.3" 
    NO_DEFAULT_PATH REQUIRED)

find_path(DNDS_EXTERNAL_INCLUDE_PYBIND11 pybind11 PATHS
    "${PROJECT_SOURCE_DIR}/external/pybind11/include"
    "${PROJECT_SOURCE_DIR}/external/pybind11-2.11.1/include" 
    NO_DEFAULT_PATH REQUIRED)

find_path(DNDS_EXTERNAL_INCLUDE_PYBIND11_JSON pybind11_json PATHS
    "${PROJECT_SOURCE_DIR}/external/pybind11_json/include"
    NO_DEFAULT_PATH REQUIRED)

find_path(DNDS_EXTERNAL_INCLUDE_FMT fmt PATHS
    "${PROJECT_SOURCE_DIR}/external/fmt/include"
    "${PROJECT_SOURCE_DIR}/external/fmt-10.1.1/include" 
    NO_DEFAULT_PATH REQUIRED)

find_path(DNDS_EXTERNAL_INCLUDE_SUPERLU superlu_dist.pc.in PATHS
    "${PROJECT_SOURCE_DIR}/external/superlu_dist-8.2.1" 
    NO_DEFAULT_PATH)

find_path(EXTERNAL_DOXYGEN_AWESOME doxygen-awesome.css PATHS 
    "${PROJECT_SOURCE_DIR}/external/doxygen-awesome-css"
    "${PROJECT_SOURCE_DIR}/external/doxygen-awesome-css-2.2.1" 
    NO_DEFAULT_PATH)

set(PYBIND11_FINDPYTHON ON)
add_subdirectory("${DNDS_EXTERNAL_INCLUDE_PYBIND11}/.." "${CMAKE_BINARY_DIR}/pybind11")
set(FMT_LIB_DIR "DNDSR/lib/dndsr_external")
set(FMT_INC_DIR "DNDSR/include")
set(FMT_CMAKE_DIR "DNDSR/lib/cmake/fmt")
set(FMT_PKGCONFIG_DIR "DNDSR/lib/pkgconfig")
set(DNDS_TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "FOR FMT")
add_subdirectory("${DNDS_EXTERNAL_INCLUDE_FMT}/.."      "${CMAKE_BINARY_DIR}/fmt")
# set(BUILD_SHARED_LIBS ${DNDS_TEMP_BUILD_SHARED_LIBS} CACHE BOOL "set by DNDSR") #! you cannot move this out of internal
set_target_properties(fmt PROPERTIES EXCLUDE_FROM_ALL OFF)
if (UNIX OR DNDS_USING_CLANG_WIN)
    target_compile_options(fmt PRIVATE -fPIC)
endif()

# install(TARGETS fmt 
#     LIBRARY DESTINATION "DNDSR/lib"
#     ARCHIVE DESTINATION "DNDSR/lib"
#     COMPONENT external)

# add_subdirectory("${DNDS_EXTERNAL_INCLUDE_CPPTRACE}/../..")

if(DNDS_EXTERNAL_INCLUDE_SUPERLU)
    set(USE_XSDK_DEFAULTS ON CACHE BOOL "Init by DNDS") # for superlu
    if (DNDS_SUPERLU_DIST_USE_BLAS)
        message("Setting DNDS_SUPERLU_DIST_USE_BLAS")
        set(TPL_ENABLE_INTERNAL_BLASLIB OFF CACHE BOOL "Init by DNDS")
        set(TPL_BLAS_LIBRARIES ${DNDS_EXTERNAL_LIB_BLAS} CACHE FILEPATH "Init by DNDS")
    else()
        set(TPL_ENABLE_INTERNAL_BLASLIB ON CACHE BOOL "Init by DNDS")
    endif()
    set(XSDK_ENABLE_Fortran OFF CACHE BOOL "Init by DNDS")
    set(TPL_PARMETIS_INCLUDE_DIRS ${DNDS_EXTERNAL_INCLUDE_PARMETIS}  CACHE PATH "Init by DNDS")
    set(TPL_PARMETIS_LIBRARIES ${DNDS_EXTERNAL_LIB_PARMETIS}  CACHE FILEPATH "Init by DNDS")
    set(XSDK_INDEX_SIZE 64 CACHE STRING "Init by DNDS")
    if(WIN32 OR MSVC)
        if(MSVC_VERSION) #! is using mingw/clang+msvc toolchain, inform superlu that do not link -lm
            set(MSVC 1 CACHE INTERNAL "Init by DNDS")
        endif()
    endif()
    add_subdirectory("${DNDS_EXTERNAL_INCLUDE_SUPERLU}")
    #! for superlu, remember to comment the #include<unistd.h>, not needed
endif()


set (DNDS_EXTERNAL_LIBS "" CACHE INTERNAL
    "Project Public External Lib Dependencies")

if (MPI_CXX_FOUND)
    list(APPEND DNDS_EXTERNAL_LIBS ${MPI_CXX_LIBRARIES})
    message(STATUS "MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES}")
endif()

if(DNDS_EXTERNAL_LIB_BACKTRACE)
    list(APPEND DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIB_BACKTRACE})
    add_compile_definitions(BOOST_STACKTRACE_USE_BACKTRACE)
endif()

# set(DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIBS}
#     fmt::fmt
#     CACHE INTERNAL
#     "Project Public External Lib Dependencies"
#     )

set(DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIBS}
    ${DNDS_EXTERNAL_LIB_CGNS}
    # ${DNDS_EXTERNAL_LIB_HDF5_HL} # HDF5_HL relies on HDF5
    ${DNDS_EXTERNAL_LIB_HDF5}
    ${DNDS_EXTERNAL_LIB_PARMETIS}
    ${DNDS_EXTERNAL_LIB_METIS}
    ${DNDS_EXTERNAL_LIB_ZLIB}
    # ${DNDS_EXTERNAL_LIB_TECIO}
    )
if (DNDS_EIGEN_USE_BLAS)
    list(APPEND DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIB_BLAS})
endif()
if (DNDS_EIGEN_USE_LAPACK)
    list(APPEND DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIB_LAPACK})
    list(APPEND DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIB_LAPACKE})
endif()
message(DEBUG "DNDS_EXTERNAL_LIBS external files (to be installed):  ${DNDS_EXTERNAL_LIBS}")

set(DNDS_EXTERNAL_LIBS_REAL "" CACHE INTERNAL
    "Project Public External Lib Dependencies' Directories")
set(DNDS_EXTERNAL_LIBS_DIRS "" CACHE INTERNAL
    "Project Public External Lib Dependencies' Directories")
set(DNDS_EXTERNAL_LIBS_DIRS_REGEX "" CACHE INTERNAL
    "Project Public External Lib Dependencies' Directories regex")
foreach(LIB ${DNDS_EXTERNAL_LIBS})
    get_filename_component(LIB_DIR ${LIB} DIRECTORY)
    list(APPEND DNDS_EXTERNAL_LIBS_DIRS       ${LIB_DIR})
    list(APPEND DNDS_EXTERNAL_LIBS_DIRS_REGEX "${LIB_DIR}.*")
    get_filename_component(LIB_REAL ${LIB} REALPATH)
    list(APPEND DNDS_EXTERNAL_LIBS_REAL ${LIB_REAL})
endforeach()

# now DNDS_EXTERNAL_LIBS contains externally built libraries
foreach(LIB ${DNDS_EXTERNAL_LIBS})
    file(INSTALL ${LIB} DESTINATION ${CMAKE_INSTALL_PREFIX}/DNDSR/lib/dndsr_external FOLLOW_SYMLINK_CHAIN)
endforeach()

list(REMOVE_DUPLICATES DNDS_EXTERNAL_LIBS_DIRS)
list(REMOVE_DUPLICATES DNDS_EXTERNAL_LIBS_DIRS_REGEX)

if (UNIX)
    # Join the directories into a single string separated by colon (:)
    string(REPLACE ";" ":" DNDS_EXTERNAL_LIBS_DIRS_SHELL_LIST "${DNDS_EXTERNAL_LIBS_DIRS}")
    string(APPEND DNDS_EXTERNAL_LIBS_DIRS_SHELL_LIST ":${CMAKE_INSTALL_PREFIX}/DNDSR/bin")
    # Generate the shell script
    set(SCRIPT_NAME "${CMAKE_INSTALL_PREFIX}/DNDSR/set_library_path.sh")
    file(WRITE ${SCRIPT_NAME} "#!/bin/bash\n")
    file(APPEND ${SCRIPT_NAME} "export LD_LIBRARY_PATH=${DNDS_EXTERNAL_LIBS_DIRS_SHELL_LIST}:\$LD_LIBRARY_PATH\n")
    message(STATUS "Generated LIBRARY script: ${SCRIPT_NAME}")
elseif(WIN32)
    string(REPLACE ";" ";" DNDS_EXTERNAL_LIBS_DIRS_SHELL_LIST "${DNDS_EXTERNAL_LIBS_DIRS}")
    string(APPEND DNDS_EXTERNAL_LIBS_DIRS_SHELL_LIST ";${CMAKE_INSTALL_PREFIX}/DNDSR/bin")
    # Generate the shell script
    set(SCRIPT_NAME "${CMAKE_INSTALL_PREFIX}/DNDSR/set_library_path.bat")
    file(WRITE ${SCRIPT_NAME} "@echo off\n")
    file(APPEND ${SCRIPT_NAME} "set PATH=%PATH%;${LIBRARY_PATH}\n")
    message(STATUS "Generated LIBRARY script: ${SCRIPT_NAME}")
else()
    message(WARNING "The list of libraries might need be aded to dynamic linking: \n"
        "${DNDS_EXTERNAL_LIBS_DIRS}"
    )

endif()

message(DEBUG "DNDS_EXTERNAL_LIBS_DIRS_REGEX:  ${DNDS_EXTERNAL_LIBS_DIRS_REGEX}")
message(DEBUG "DNDS_EXTERNAL_LIBS_REAL:  ${DNDS_EXTERNAL_LIBS_REAL}")
# add internal built libraries

list(APPEND DNDS_EXTERNAL_LIBS fmt::fmt)
if (DNDS_USE_CUDA)
    #! don't do this, could cause linking both shared/static cudart
    # list(APPEND DNDS_EXTERNAL_LIBS CUDA::cudart)
endif()

if(DNDS_EXTERNAL_INCLUDE_SUPERLU)
    list(APPEND DNDS_EXTERNAL_LIBS superlu_dist)
endif()

set(DNDS_EXTERNAL_INCLUDES 
    ${DNDS_EXTERNAL_INCLUDE_ZLIB}
    ${DNDS_EXTERNAL_INCLUDE_HDF5}
    ${DNDS_EXTERNAL_INCLUDE_CGNS}
    ${DNDS_EXTERNAL_INCLUDE_METIS}
    ${DNDS_EXTERNAL_INCLUDE_PARMETIS}
    # ${DNDS_EXTERNAL_INCLUDE_TECIO}
    ${DNDS_EXTERNAL_INCLUDE_EIGEN}
    ${DNDS_EXTERNAL_INCLUDE_JSON}
    ${DNDS_EXTERNAL_INCLUDE_ARGPARSE}
    ${DNDS_EXTERNAL_INCLUDE_EXPRTK}
    # ${DNDS_EXTERNAL_INCLUDE_CPPTRACE}
    # ${DNDS_EXTERNAL_INCLUDE_RAPIDJSON}
    ${DNDS_EXTERNAL_INCLUDE_CPPCODEC}
    ${DNDS_EXTERNAL_INCLUDE_NANOFLANN}
    ${DNDS_EXTERNAL_INCLUDE_BOOST}
    ${DNDS_EXTERNAL_INCLUDE_CGAL}
    ${DNDS_EXTERNAL_INCLUDE_PYBIND11}
    ${DNDS_EXTERNAL_INCLUDE_PYBIND11_JSON}
    ${DNDS_EXTERNAL_INCLUDE_FMT}
    CACHE INTERNAL "Project External Includes")

if(DNDS_EXTERNAL_INCLUDE_SUPERLU)
    set(DNDS_EXTERNAL_INCLUDES ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_EXTERNAL_INCLUDE_SUPERLU}/SRC
        CACHE INTERNAL "Project External Includes")
endif()

if(DNDS_USE_CUDA)
    list(APPEND DNDS_EXTERNAL_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
endif()

set(DNDS_PCH_TARGETS "sss")
set(DNDS_PCH_TARGETS_FAST "sss")

add_compile_definitions(__DNDS_REALLY_COMPILING__)

if(DNDS_DIST_MT_USE_OMP)
    if(NOT DNDS_USE_OMP)
        message(FATAL_ERROR "must use DNDS_USE_OMP if you need DNDS_DIST_MT_USE_OMP")
    endif()
    add_compile_definitions(DNDS_DIST_MT_USE_OMP)
endif()

if(DNDS_EIGEN_USE_BLAS)
    add_compile_definitions(EIGEN_USE_BLAS)
endif()

if(DNDS_EIGEN_USE_LAPACK)
    add_compile_definitions(EIGEN_USE_LAPACKE_STRICT)
endif()

############################
# Platform specific settings
############################

if(UNIX OR MINGW)
    message(STATUS "UNIX OR MINGW")
    set(DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIBS}
    stdc++fs)
    if(UNIX)
        set(DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIBS}
        dl) # add dl
    endif()
    if(MINGW)
        add_compile_options(-Wa,-mbig-obj) # too many sections
    endif()
    message(${CMAKE_CXX_COMPILER_ID})
    add_compile_options(-std=c++17)
    if(DNDS_NATIVE_ARCH)
        add_compile_options(-march=native)
        add_compile_options(-mtune=native)
        message(WARNING "using -march=native")
    endif()
    if(DNDS_UNSAFE_MATH_OPT)
        add_compile_options(-funsafe-math-optimizations)
        message(WARNING "using -funsafe-math-optimization")
    endif()
    add_compile_options(-fdiagnostics-color=always)
    
    ### set warnings
    # -Wconversion
    add_compile_options(
        -Wall -Wno-unused-but-set-variable -Wno-unused-variable 
        $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror=return-type>
        )
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=128>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177>) # declared but never referenced
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=2417>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    # add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe>)
    # add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--diag_error=host_call_from_device>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Werror>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:cross-execution-space-call>)

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        if (DNDS_USE_OMP)
            add_compile_options(-fopenmp)
            add_link_options(-fopenmp)
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            # doesn't seem available
            message(WARNING "${CMAKE_CXX_COMPILER_ID} compliler not using DNDS_USE_PARALLEL_MACRO")
        endif()
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if (DNDS_USE_OMP)
            add_compile_options(-fopenmp)
            add_link_options(-fopenmp)
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            add_definitions(-D_GLIBCXX_PARALLEL)
            message(WARNING "${CMAKE_CXX_COMPILER_ID} compliler not using DNDS_USE_PARALLEL_MACRO")
        endif()
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # add_compile_options(-fopenmp)
        if (DNDS_USE_OMP)
            add_compile_options(-fiopenmp)
            add_link_options(-fiopenmp)
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            # add_definitions(-D_GLIBCXX_PARALLEL) #! not yet found 
            message(WARNING "${CMAKE_CXX_COMPILER_ID} compliler not using DNDS_USE_PARALLEL_MACRO")
        endif()
        add_compile_options(-diag-disable=1011) # branch no return 
        add_compile_options(-diag-disable=2196) # both inline and no inline
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
        if (DNDS_USE_OMP)
            add_compile_options(-fiopenmp)
            add_link_options(-fiopenmp)
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            # add_definitions(-D_GLIBCXX_PARALLEL) #! not yet found 
            message(WARNING "${CMAKE_CXX_COMPILER_ID} compliler not using DNDS_USE_PARALLEL_MACRO")
        endif()
        add_compile_options(-Wno-tautological-constant-compare) # ! is this risky or not
        message(WARNING "${CMAKE_CXX_COMPILER_ID} using -Wno-tautological-constant-compare")
    endif()
    

elseif(WIN32 OR MSVC)
    message(STATUS "WIN32 OR MSVC")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)
        add_compile_options(-std=c++17 )
        add_link_options(-lshlwapi) # when hdf is compiled with msvc, needs this
        # add_compile_options(--target=x86_64-pc-windows-gnu)
        # add_link_options(--target=x86_64-pc-windows-gnu)
        add_compile_options(-D_CRT_SECURE_NO_WARNINGS -DCOMPILER_MSC)
        if(DNDS_UNSAFE_MATH_OPT)
            add_compile_options(-funsafe-math-optimizations)
            message(WARNING "using -funsafe-math-optimization")
        endif()
        if(DNDS_NATIVE_ARCH)
            add_compile_options(-march=native)
            message(WARNING "using -march=native")
        endif()
        add_compile_options(-fdiagnostics-color=always)
        message(STATUS "MSVC ${MSVC}")
        message(STATUS "MSVC_VERSION ${MSVC_VERSION}")
    endif()
    set(DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIBS} # no dl
    )
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # add_compile_options(-Wall)
        add_compile_options(/std:c++17 /IGNORE:C2124) # cl is buggy in c++17 mode ...
        add_compile_options(-Wall)
        add_compile_options(-Wno-unused-variable -Wno-sign-compare -Wno-unused-but-set-variable -Wno-unused-parameter) # for clang-cl!
        add_compile_options(-Wno-reserved-identifier -Wno-documentation  -Wno-old-style-cast)
        add_compile_options(-Wno-old-style-cast -Wno-exit-time-destructors -Wno-global-constructors -Wno-zero-as-null-pointer-constant)
        add_compile_options(-Wno-sign-conversion -Wno-unused-template)
        add_compile_options(-Wno-documentation-unknown-command)
        add_compile_options(-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-c++11-compat -Wno-c++11-compat-pedantic -Wno-c++14-compat -Wno-c++14-compat-pedantic)
        add_compile_options(-Wno-extra-semi-stmt -Wno-comma -Wno-float-equal -Wno-missing-noreturn)
        add_compile_options(-Wno-undef -Wno-implicit-fallthrough -Wno-redundant-parens)
        # add_compile_options(-Wextra-semi-stmt  -Wshorten-64-to-32 -Wnewline-eof)
    endif()

    if(WIN32 AND("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM"))
         #! we use msvc-style abi in library (which is default in clang but not for icx)
        string(APPEND CMAKE_CXX_FLAGS " -fms-compatibility -fms-extensions")
        # add_compile_options(-fms-compatibility -fms-extensions)
        # add_link_options(-fms-compatibility -fms-extensions)
        
        add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)
        add_compile_options(-D_CRT_SECURE_NO_WARNINGS)

        if(DNDS_UNSAFE_MATH_OPT)
            add_compile_options(-ffast-math)
            message(WARNING "using -ffast-math")
        endif()

        link_libraries(shlwapi) # when hdf is compiled with msvc, needs this
    endif()

    set(DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIBS} # no dl
    )
    
    
    if (DNDS_USE_OMP)
        #
        if(MSVC AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")) #! force use /openmp
            # set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "/openmp")
            add_compile_options(-openmp)
        elseif(WIN32 AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))#! force use -fopenmp
            add_compile_options(-fopenmp)
            add_link_options(-fopenmp)
        elseif(WIN32 AND("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM"))
            add_compile_options(-openmp)
            add_link_options(-openmp)
        else()
            find_package(OpenMP REQUIRED)
            if(OpenMP_CXX_FLAGS)
                set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
            else()
                message(FATAL_ERROR "OMP not found")
            endif()
        endif()

        

    endif()
    if (DNDS_USE_PARALLEL_MACRO)
        # ! to be tested
    endif()
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()


if (MPI_CXX_FOUND)
    set(DNDS_EXTERNAL_INCLUDES ${MPI_CXX_INCLUDE_DIRS} ${DNDS_EXTERNAL_INCLUDES})
    message(STATUS "MPI_CXX_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS}")
endif()

set(DNDS_INCLUDES ${PROJECT_SOURCE_DIR}/src  CACHE INTERNAL "Project Public Includes")

if (UNIX)
add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)
endif()

if (DNDS_SUPRESS_INSERT_CHECK)
    # add_compile_definitions(NINSERT)
    add_definitions(-DNINSERT)
endif()
if (DNDS_USE_NDEBUG_MACRO)
    # add_compile_definitions(NDEBUG)
    add_definitions(-DNDEBUG)
else()
    add_compile_options(-UNDEBUG)
endif()
if (DNDS_USE_DNDS_NDEBUG_MACRO)
    add_definitions(-DDNDS_NDEBUG)
endif()
if (DNDS_USE_OMP)
    # add_compile_definitions(DNDS_USE_OMP)
    add_definitions(-DDNDS_USE_OMP)
endif()

# if (DNDS_USE_PARALLEL_MACRO)
#     add_compile_definitions(PARALLEL)
# endif()

message(STATUS "DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIBS}")
message(STATUS "DNDS_EXTERNAL_INCLUDES ${DNDS_EXTERNAL_INCLUDES}")

#######################################
#       START OF CORE TARGETS         #
#######################################



set(DNDS_Euler_Models_List
NS=
NS_2D=2D 
NS_3D=3D
NS_SA=SA
NS_SA_3D=SA3D
NS_2EQ=2EQ
NS_2EQ_3D=2EQ3D
NS_EX=EX
NS_EX_3D=EX3D
CACHE INTERNAL "the Euler Model list to distinguish their explicit instantiation libraries"
)



## app exes

set(DNDS_APPS_EXTERNAL
cgns_APITest
eigen_Test
STL_Test
json_Test
cgal_AABBTest
mpi_test
)

set(DNDS_APPS_EXTERNAL_CU
cuda_test
)

set(DNDS_APPS_DNDS
array_Test
arrayTrans_test
arrayDerived_test
arrayDOF_test
serializerJSON_Test
serializerH5_Test
stdPowerTest
objectPool_test
)

set(DNDS_APPS_DNDS_CU
array_cuda_Test
array_cuda_Bench
arrayDOF_test_cuda
)

set(DNDS_APPS_Geom
elements_Test
meshSerial_Test
ofReader_Test
partitionMeshSerial
)

set(DNDS_APPS_CFV
vrStatic_Test
vrBasic_Test
diffTensors_Test
)

set(DNDS_APPS_Solver
krylovTest
)



set(DNDS_APPS_Euler
gasTest
jacobiLUTest
oneDimProfileTest
)

# libraries

add_subdirectory(${CMAKE_SOURCE_DIR}/src/DNDS)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/Geom)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/CFV)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/Euler)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/EulerP)

if (NOT SKBUILD_PROJECT_NAME) # do not create links for SCIKIT-BUILD build
    install(CODE "file(CREATE_LINK ${CMAKE_INSTALL_PREFIX}/DNDSR/bin ${CMAKE_SOURCE_DIR}/src/bin SYMBOLIC)" COMPONENT py)
    install(CODE "file(CREATE_LINK ${CMAKE_INSTALL_PREFIX}/DNDSR/lib ${CMAKE_SOURCE_DIR}/src/lib SYMBOLIC)" COMPONENT py)
endif()
# if (SKBUILD_PROJECT_NAME)
# # __init__.py actively linking libdnds_shared.so, 
# # cannot link correct libstdc++.so (links to anaconda's)
# # why?
# message(WARNING "skipping .pyi generation")
# else()
install(CODE "execute_process(COMMAND bash ${CMAKE_SOURCE_DIR}/src/run-pybind11-stubgen.sh  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src)" 
    COMPONENT py)
# endif()

set(EXE_SUFFIX "")
if (UNIX)
    set(EXE_SUFFIX ".exe")
endif()


function(ADD_EXE_APP EXES MAIN_DIR LIBS USE_EXCLUDE_FROM_ALL SRC_SUFFIX)
    message(STATUS "To add exes: ${EXES} with libs: ${LIBS}")
    foreach(EXE ${EXES})
        # add_executable(${EXE} app/external/${EXE}.cpp $<TARGET_OBJECTS:${OBJS}>)
        add_executable(${EXE} ${MAIN_DIR}/${EXE}.${SRC_SUFFIX})
        if(USE_EXCLUDE_FROM_ALL)
            set_target_properties(${EXE} PROPERTIES EXCLUDE_FROM_ALL ON)
        endif()
        target_link_libraries(${EXE} PUBLIC ${LIBS})
        target_link_libraries(${EXE} PUBLIC ${DNDS_EXTERNAL_LIBS})
        target_include_directories(${EXE} PUBLIC ${DNDS_EXTERNAL_INCLUDES} PUBLIC ${DNDS_INCLUDES})
        set_target_properties(${EXE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/app RUNTIME_OUTPUT_NAME "${EXE}${EXE_SUFFIX}")
        if(DNDS_RECORD_COMMIT)
            target_compile_definitions(${EXE} PUBLIC DNDS_CURRENT_COMMIT_HASH=${DNDS_RECORDED_COMMIT_HASH})
        endif()
        if( DNDS_USE_CCACHE )
            set_property(TARGET ${EXE} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
        endif()
    endforeach()
endfunction(ADD_EXE_APP)

# In topolocical order of the dependency graph
# euler_library
# euler_library_fast
# cfv
# geom
# dnds

## Mind That the TOPOLOGICAL ORDER should be obeyed!
ADD_EXE_APP("${DNDS_APPS_EXTERNAL}" "app/external" ";" ON cpp)
ADD_EXE_APP("${DNDS_APPS_DNDS}" "app/DNDS" "dnds;" ON cpp)
if(DNDS_USE_CUDA)
    ADD_EXE_APP("${DNDS_APPS_EXTERNAL_CU}" "app/external" ";" ON cu)
    ADD_EXE_APP("${DNDS_APPS_DNDS_CU}" "app/DNDS" "dnds;" ON cu)
endif()
ADD_EXE_APP("${DNDS_APPS_Solver}" "app/Solver" "dnds;" ON cpp)
ADD_EXE_APP("${DNDS_APPS_Geom}" "app/Geom" "geom;dnds;" ON cpp)
ADD_EXE_APP("${DNDS_APPS_CFV}" "app/CFV" "cfv;geom;dnds;" ON cpp)
ADD_EXE_APP("${DNDS_APPS_Euler}" "app/Euler" "cfv;geom;dnds;" ON cpp)



set(DNDS_APPS_Euler_Models)

foreach(item IN LISTS DNDS_Euler_Models_List)
    string(REPLACE "=" ";" keyval ${item})
    list(GET keyval 0 key)
    list(GET keyval 1 value)
    set(EXE_NAME "euler${value}")
    list(APPEND DNDS_APPS_Euler_Models "${EXE_NAME}")
    message(DEBUG "${keyval} --- ${value} --- ${EXE_NAME}")
    ## Mind That the TOPOLOGICAL ORDER should be obeyed!
    ADD_EXE_APP("${EXE_NAME}" "app/Euler" "euler_library_${key};euler_library_fast_${key};cfv;geom;dnds;" ON cpp)
    ## This works because cmake detects dependency and reorders the libraries
    # ADD_EXE_APP("${EXE_NAME}" "app/Euler" "dnds;geom;cfv;euler_library_fast_${key};euler_library_${key};")
endforeach()






#######################################################
## doxygen
#######################################################

find_package(Doxygen)
if (DOXYGEN_FOUND)
    message(STATUS "Doxygen Found") 
    #! TODO: use automatic config Doxygen.in file
    ##### using custom DOXYGEN
    set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(docs ALL 
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT Doing Doxygen
        VERBATIM
    )
    execute_process(COMMAND python ${CMAKE_SOURCE_DIR}/docs/getAllAttachForDox.py 
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/docs")

    # find_file(DOXYGEN_PDF_FILES docs/*.pdf NO_DEFAULT_PATH)
    # message(STATUS ${DOXYGEN_PDF_FILES})
    ##### using CMAKE convenient DOXYGEN!only in 3.9 +!!!
    
else(DOXYGEN_FOUND)
    message("Doxygen Not Found")
endif(DOXYGEN_FOUND)

if(DNDS_GENERATE_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    find_program(COMpdb_EXECUTABLE compdb)
    if(NOT COMpdb_EXECUTABLE)
        message(WARNING "compdb not found. Install it with: pip install compdb")
    endif()
    if(COMpdb_EXECUTABLE AND CMAKE_EXPORT_COMPILE_COMMANDS)
        add_custom_target(process-compile-commands
            COMMAND ${COMpdb_EXECUTABLE} -p ${CMAKE_BINARY_DIR} list > ${CMAKE_BINARY_DIR}/compile_commands_processed.json
            DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
            COMMENT "Processing compile_commands.json with compdb"
            VERBATIM
        )
        
        # Optional: Create a symlink in source directory for IDE convenience
        if(CMAKE_HOST_UNIX)
            add_custom_command(TARGET process-compile-commands POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E create_symlink 
                    ${CMAKE_BINARY_DIR}/compile_commands_processed.json 
                    ${CMAKE_SOURCE_DIR}/compile_commands.json
                COMMENT "Creating symlink to processed compile_commands.json in source directory"
            )
        endif()
    endif()
endif()




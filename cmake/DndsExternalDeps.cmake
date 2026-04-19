# cmake/DndsExternalDeps.cmake
# Locate all external libraries and headers, set up pybind11/fmt/superlu
# subdirectories, assemble DNDS_EXTERNAL_LIBS / DNDS_EXTERNAL_INCLUDES,
# and create the dnds_external_deps INTERFACE target.
#
# Must be included after DndsOptions.cmake, DndsCudaSetup.cmake, and
# DndsCompilerFlags.cmake (which sets MPI variables and DNDS_EXTERNAL_LIBS
# platform entries like stdc++fs / dl).

# -------------------------------------------------------------------
# External install paths
# -------------------------------------------------------------------
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

# ##########################################
# find_library calls — NOT changed, just moved here
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

# ##########################################
# find_path calls — NOT changed, just moved here
# ##########################################
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

# -------------------------------------------------------------------
# Subdirectory targets: pybind11, fmt, superlu_dist
# -------------------------------------------------------------------
set(PYBIND11_FINDPYTHON ON)
add_subdirectory("${DNDS_EXTERNAL_INCLUDE_PYBIND11}/.." "${CMAKE_BINARY_DIR}/pybind11")
set(FMT_LIB_DIR "DNDSR/lib/dndsr_external")
set(FMT_INC_DIR "DNDSR/include")
set(FMT_CMAKE_DIR "DNDSR/lib/cmake/fmt")
set(FMT_PKGCONFIG_DIR "DNDSR/lib/pkgconfig")
set(DNDS_TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "FOR FMT")
add_subdirectory("${DNDS_EXTERNAL_INCLUDE_FMT}/.."      "${CMAKE_BINARY_DIR}/fmt")
set(BUILD_SHARED_LIBS ${DNDS_TEMP_BUILD_SHARED_LIBS} CACHE BOOL "set by DNDSR, restored after fmt" FORCE)
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

# -------------------------------------------------------------------
# Assemble DNDS_EXTERNAL_LIBS list
# -------------------------------------------------------------------
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

# On macOS/Apple platforms, _Unwind_Backtrace is available without _GNU_SOURCE
if(APPLE)
    add_compile_definitions(BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
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

# -------------------------------------------------------------------
# Resolve real paths and directories for external libs
# -------------------------------------------------------------------
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
# Also copy external libs into python package tree for editable installs
foreach(LIB ${DNDS_EXTERNAL_LIBS})
    file(INSTALL ${LIB} DESTINATION ${PROJECT_SOURCE_DIR}/python/DNDSR/_lib/dndsr_external FOLLOW_SYMLINK_CHAIN)
endforeach()

list(REMOVE_DUPLICATES DNDS_EXTERNAL_LIBS_DIRS)
list(REMOVE_DUPLICATES DNDS_EXTERNAL_LIBS_DIRS_REGEX)

# -------------------------------------------------------------------
# Generate LD_LIBRARY_PATH / PATH helper scripts
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Platform-specific runtime libraries (stdc++fs, dl)
# -------------------------------------------------------------------
if(UNIX OR MINGW)
    # stdc++fs is only needed for older GCC with libstdc++, not for libc++ (macOS/Clang)
    if(NOT DNDS_USING_LIBCXX)
        set(DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIBS}
        stdc++fs)
    endif()
    if(UNIX)
        set(DNDS_EXTERNAL_LIBS ${DNDS_EXTERNAL_LIBS}
        dl) # add dl
    endif()
endif()

# -------------------------------------------------------------------
# Append internally-built library targets
# -------------------------------------------------------------------
# add internal built libraries

list(APPEND DNDS_EXTERNAL_LIBS fmt::fmt)
if (DNDS_USE_CUDA)
    #! don't do this, could cause linking both shared/static cudart
    list(APPEND DNDS_EXTERNAL_LIBS CUDA::cuda_driver)
endif()

if(DNDS_EXTERNAL_INCLUDE_SUPERLU)
    list(APPEND DNDS_EXTERNAL_LIBS superlu_dist)
endif()

# -------------------------------------------------------------------
# Assemble DNDS_EXTERNAL_INCLUDES list
# -------------------------------------------------------------------
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
    # CUDA 13.x: thrust/cub/libcudacxx are under cccl/ (see DndsCudaSetup.cmake)
    if(DNDS_CUDA_CCCL_INCLUDE_DIR)
        list(APPEND DNDS_EXTERNAL_INCLUDES ${DNDS_CUDA_CCCL_INCLUDE_DIR})
    endif()
endif()

# -------------------------------------------------------------------
# MPI include directories (must be first in include path)
# -------------------------------------------------------------------
if (MPI_CXX_FOUND)
    set(DNDS_EXTERNAL_INCLUDES ${MPI_CXX_INCLUDE_DIRS} ${DNDS_EXTERNAL_INCLUDES})
    message(STATUS "MPI_CXX_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS}")
endif()

set(DNDS_INCLUDES ${PROJECT_SOURCE_DIR}/src  CACHE INTERNAL "Project Public Includes")

# -------------------------------------------------------------------
# Compile definitions for external-dep-related features
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# Compile definitions (NINSERT, NDEBUG, DNDS_NDEBUG, OMP, etc.)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# INTERFACE target: dnds_external_deps
# -------------------------------------------------------------------
# Create an INTERFACE library to bundle all external dependencies.
# Targets can link against dnds_external_deps instead of manually passing
# DNDS_EXTERNAL_LIBS and DNDS_EXTERNAL_INCLUDES.
add_library(dnds_external_deps INTERFACE)
target_include_directories(dnds_external_deps INTERFACE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(dnds_external_deps INTERFACE ${DNDS_EXTERNAL_LIBS})

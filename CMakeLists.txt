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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

cmake_policy(SET CMP0082 NEW) # ensures install exec order == declare order

set(CMAKE_BUILD_TYPE Release CACHE STRING "Release or Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Release" "Debug" "RelWithDebInfo" "MinSizeRel")

if(DNDS_GENERATE_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()


# ##########################################
# Read the base version before project() so we can pass it.
# DndsVersion.cmake needs PROJECT_SOURCE_DIR which is not set yet,
# so we use CMAKE_CURRENT_SOURCE_DIR instead.
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")  # temporary, project() overrides
include(cmake/DndsVersion.cmake)

project(DNDSR
    VERSION ${DNDS_VERSION_BASE}
    LANGUAGES CXX) #! need to be in front of find_package(MPI)
enable_language(C) #! building C sub-projects

# Only set a default CMAKE_INSTALL_PREFIX if the user has not specified one.
# Must be after project() for CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT to work.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH
        "Default install location (override with -DCMAKE_INSTALL_PREFIX=...)" FORCE)
endif()

message(STATUS "CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")
set(DNDS_USING_CLANG_WIN OFF)
if(WIN32)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(DNDS_USING_CLANG_WIN ON)
    endif()
endif()

# ===================================================================
# Modular build configuration — each include() pulls in one logical
# section.  The order matters: later modules depend on variables set
# by earlier ones.
# ===================================================================

# 1. Detect and bundle the C++ standard library (libstdc++ or libc++)
include(cmake/DndsStdlibSetup.cmake)

# 2. User-facing cache options, commit recording, verbose/tidy toggles
include(cmake/DndsOptions.cmake)

# 3. CUDA language enable + toolkit discovery (reads DNDS_USE_CUDA)
include(cmake/DndsCudaSetup.cmake)

# 4. LTO, MPI discovery, platform-specific compiler/linker flags, OpenMP
#    (reads options, CUDA state; sets MPI_CXX_* variables for later use)
include(cmake/DndsCompilerFlags.cmake)

# 5. External library / header discovery, pybind11/fmt/superlu subdirs,
#    DNDS_EXTERNAL_LIBS / DNDS_EXTERNAL_INCLUDES assembly, compile defs,
#    and the dnds_external_deps INTERFACE target
#    (reads MPI, options, CUDA toolkit results)
include(cmake/DndsExternalDeps.cmake)

# 6. Helper functions (add_fast_flags, dnds_add_lib, dnds_add_py_module)
include("${PROJECT_SOURCE_DIR}/cmakeCommonUtils.cmake")

# ===================================================================
#       CORE LIBRARY TARGETS
# ===================================================================

# Euler model list (used by both src/Euler/ and the solver executables)
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

add_subdirectory(${CMAKE_SOURCE_DIR}/src/DNDS)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/Geom)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/CFV)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/Euler)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/EulerP)

# ===================================================================
#       TESTS, APPS, DOCS, TOOLING
# ===================================================================

# 7. C++ unit tests (doctest) + Python pytest registration
include(cmake/DndsTests.cmake)

# 8. Application executables (euler, test apps, etc.)
include(cmake/DndsApps.cmake)

# 9. Doxygen documentation target
include(cmake/DndsDocs.cmake)

# 10. Compilable documentation examples
add_subdirectory(${CMAKE_SOURCE_DIR}/examples)

# 11. compile_commands.json post-processing + stub generation
include(cmake/DndsTooling.cmake)

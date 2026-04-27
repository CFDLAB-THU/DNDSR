# cmake/DndsStdlibSetup.cmake
# Detect the C++ standard library (libstdc++ or libc++).
#
# Detection is still performed so that downstream code can check
# DNDS_STDLIB_FOUND and DNDS_USING_LIBCXX.  However, the runtime
# library is NO LONGER bundled into dndsr_external/:
#
# - libstdc++/libc++ are system-provided and must not be duplicated.
#   Bundling them with RTLD_DEEPBIND caused dual-allocator double-free
#   crashes when system libraries (e.g. libmpi) used the system copy
#   while DNDSR used the bundled copy.
#
# For conda/anaconda environments with an older libstdc++, users
# should set LD_LIBRARY_PATH or DNDSR_USE_DEEPBIND=1 instead.

if(NOT UNIX)
    return()
endif()

set(DNDS_STDLIB_FOUND OFF)

# -------------------------------------------------------------------
# Detect if using libc++ by checking compiler flags or compiler default
# -------------------------------------------------------------------
set(DNDS_USING_LIBCXX OFF)
# Check if -stdlib=libc++ is in the flags
string(FIND "${CMAKE_CXX_FLAGS}" "-stdlib=libc++" _libcxx_flag_pos)
if(NOT _libcxx_flag_pos EQUAL -1)
    set(DNDS_USING_LIBCXX ON)
endif()
# Also check if Clang defaults to libc++ (e.g., on macOS or some configurations)
if(NOT DNDS_USING_LIBCXX AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # Try to compile a test to check which stdlib is used
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_QUIET ON)
    check_cxx_source_compiles("
        #include <ciso646>
        #if defined(_LIBCPP_VERSION)
        int main() { return 0; }
        #else
        #error not libc++
        #endif
    " DNDS_COMPILER_USES_LIBCXX)
    if(DNDS_COMPILER_USES_LIBCXX)
        set(DNDS_USING_LIBCXX ON)
    endif()
endif()

# -------------------------------------------------------------------
# libc++ path
# -------------------------------------------------------------------
if(DNDS_USING_LIBCXX)
    message(STATUS "Detected libc++ as the C++ standard library")
    # Find libc++.so using find_library
    if(NOT DEFINED LIBCXX_PATH)
        find_library(LIBCXX_PATH NAMES c++ libc++ NAMES_PER_DIR)
    endif()

    if(LIBCXX_PATH)
        message(STATUS "Found libc++: ${LIBCXX_PATH}")
        # Not bundled — see header comment for rationale.
        # file(INSTALL "${LIBCXX_PATH}" DESTINATION ${CMAKE_INSTALL_PREFIX}/DNDSR/lib/dndsr_external FOLLOW_SYMLINK_CHAIN)
        set(DNDS_STDLIB_FOUND ON)
        # Also try to find libc++abi.so which is often needed alongside libc++.so
        if(NOT DEFINED LIBCXXABI_PATH)
            find_library(LIBCXXABI_PATH NAMES c++abi libc++abi NAMES_PER_DIR)
        endif()
        if(LIBCXXABI_PATH)
            message(STATUS "Found libc++abi: ${LIBCXXABI_PATH}")
            # file(INSTALL "${LIBCXXABI_PATH}" DESTINATION ${CMAKE_INSTALL_PREFIX}/DNDSR/lib/dndsr_external FOLLOW_SYMLINK_CHAIN)
        endif()
    else()
        message(STATUS "libc++.so not found via find_library")
    endif()

# -------------------------------------------------------------------
# libstdc++ path
# -------------------------------------------------------------------
else()
    message(STATUS "Detected libstdc++ as the C++ standard library")
    # Find libstdc++.so using GCC's -print-file-name (works reliably for GCC)
    if(NOT DEFINED LIBSTDCXX_PATH)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
            OUTPUT_VARIABLE LIBSTDCXX_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()

    if(EXISTS "${LIBSTDCXX_PATH}")
        message(STATUS "Found libstdc++.so: ${LIBSTDCXX_PATH}")
        # Not bundled — see header comment for rationale.
        # file(INSTALL "${LIBSTDCXX_PATH}" DESTINATION ${CMAKE_INSTALL_PREFIX}/DNDSR/lib/dndsr_external FOLLOW_SYMLINK_CHAIN)
        # file(INSTALL "${LIBSTDCXX_PATH}" DESTINATION ${PROJECT_SOURCE_DIR}/python/DNDSR/_lib/dndsr_external FOLLOW_SYMLINK_CHAIN)
        set(DNDS_STDLIB_FOUND ON)
    else()
        message(STATUS "libstdc++.so not found at: ${LIBSTDCXX_PATH}")
    endif()
endif()

if(NOT DNDS_STDLIB_FOUND)
    message(FATAL_ERROR "C++ standard library (libstdc++.so or libc++.so) not found!")
endif()

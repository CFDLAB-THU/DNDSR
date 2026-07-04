# cmake/DndsCheckVersion.cmake
# Runs at build time (via dnds_version_check target) to detect git HEAD
# changes.  When HEAD differs from the stamp written at configure time,
# the stamp is updated.  Because the stamp is listed in
# CMAKE_CONFIGURE_DEPENDS, the next build invocation triggers an automatic
# reconfigure with the correct version information.
#
# Invoked: cmake -DPROJECT_SOURCE_DIR=<src> -DCMAKE_BINARY_DIR=<build> -P ...

find_package(Git QUIET)
if(NOT GIT_FOUND)
    return()
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE _head
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _ret
)
if(NOT _ret EQUAL 0)
    return()
endif()

set(_stamp "${CMAKE_BINARY_DIR}/dnds_git_head.txt")
if(EXISTS "${_stamp}")
    file(READ "${_stamp}" _prev)
    string(STRIP "${_prev}" _prev)
    if(_head STREQUAL _prev)
        return()
    endif()
endif()

file(WRITE "${_stamp}" "${_head}")
message(STATUS "DNDSR: git HEAD changed (${_prev} -> ${_head}), next build will reconfigure")

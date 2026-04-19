# cmake/DndsVersion.cmake
# Centralized version management for DNDSR.
#
# Reads the base version from the VERSION file, then uses git describe
# to determine if the current commit is a tagged release or a dev build.
#
# Variables set (CACHE INTERNAL):
#   DNDS_VERSION_MAJOR      e.g. 0
#   DNDS_VERSION_MINOR      e.g. 0
#   DNDS_VERSION_PATCH      e.g. 3
#   DNDS_VERSION_BASE       e.g. "0.0.3"          (from VERSION file)
#   DNDS_VERSION_IS_RELEASE TRUE if HEAD is tagged exactly as v<base>
#   DNDS_VERSION_COMMIT     short commit hash      e.g. "be407e3"
#   DNDS_VERSION_DISTANCE   commits since last tag  e.g. "235"
#   DNDS_VERSION_FULL       display string:
#                             release:  "0.0.3"
#                             dev:      "0.0.3.dev235+gbe407e3"
#   DNDS_VERSION_PEP440     PEP 440 string (same as FULL, used by Python)

# -------------------------------------------------------------------
# 1. Read base version from VERSION file
# -------------------------------------------------------------------
set(_version_file "${PROJECT_SOURCE_DIR}/VERSION")
if(NOT EXISTS "${_version_file}")
    message(FATAL_ERROR "VERSION file not found at ${_version_file}")
endif()
file(READ "${_version_file}" _version_raw)
string(STRIP "${_version_raw}" DNDS_VERSION_BASE)

# Parse major.minor.patch
string(REPLACE "." ";" _version_parts "${DNDS_VERSION_BASE}")
list(LENGTH _version_parts _version_len)
if(_version_len LESS 3)
    message(FATAL_ERROR "VERSION file must contain MAJOR.MINOR.PATCH, got: ${DNDS_VERSION_BASE}")
endif()
list(GET _version_parts 0 DNDS_VERSION_MAJOR)
list(GET _version_parts 1 DNDS_VERSION_MINOR)
list(GET _version_parts 2 DNDS_VERSION_PATCH)

# -------------------------------------------------------------------
# 2. Query git describe for commit distance and hash
# -------------------------------------------------------------------
set(DNDS_VERSION_IS_RELEASE FALSE)
set(DNDS_VERSION_COMMIT "unknown")
set(DNDS_VERSION_DISTANCE "0")

find_package(Git QUIET)
if(GIT_FOUND)
    # Try git describe with the expected tag for this version
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --match "v*"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE _git_describe
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _git_ret
    )
    if(_git_ret EQUAL 0)
        # Parse: v0.0.2-235-gbe407e3
        string(REGEX MATCH "^v([0-9]+\\.[0-9]+\\.[0-9]+)-([0-9]+)-g([0-9a-f]+)$"
               _match "${_git_describe}")
        if(_match)
            set(_tag_version "${CMAKE_MATCH_1}")
            set(DNDS_VERSION_DISTANCE "${CMAKE_MATCH_2}")
            set(DNDS_VERSION_COMMIT "${CMAKE_MATCH_3}")

            if(DNDS_VERSION_DISTANCE STREQUAL "0")
                set(DNDS_VERSION_IS_RELEASE TRUE)
            endif()
        endif()
    endif()

    # Fallback: get commit hash directly if describe failed
    if(DNDS_VERSION_COMMIT STREQUAL "unknown")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short=7 HEAD
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            OUTPUT_VARIABLE DNDS_VERSION_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
endif()

# -------------------------------------------------------------------
# 3. Compose the full version string
# -------------------------------------------------------------------
if(DNDS_VERSION_IS_RELEASE)
    set(DNDS_VERSION_FULL "${DNDS_VERSION_BASE}")
else()
    # PEP 440: 0.0.3.dev235+gbe407e3
    set(DNDS_VERSION_FULL "${DNDS_VERSION_BASE}.dev${DNDS_VERSION_DISTANCE}+g${DNDS_VERSION_COMMIT}")
endif()

set(DNDS_VERSION_PEP440 "${DNDS_VERSION_FULL}")

# Cache everything
set(DNDS_VERSION_MAJOR    "${DNDS_VERSION_MAJOR}"    CACHE INTERNAL "")
set(DNDS_VERSION_MINOR    "${DNDS_VERSION_MINOR}"    CACHE INTERNAL "")
set(DNDS_VERSION_PATCH    "${DNDS_VERSION_PATCH}"    CACHE INTERNAL "")
set(DNDS_VERSION_BASE     "${DNDS_VERSION_BASE}"     CACHE INTERNAL "")
set(DNDS_VERSION_FULL     "${DNDS_VERSION_FULL}"     CACHE INTERNAL "")
set(DNDS_VERSION_PEP440   "${DNDS_VERSION_PEP440}"   CACHE INTERNAL "")
set(DNDS_VERSION_COMMIT   "${DNDS_VERSION_COMMIT}"   CACHE INTERNAL "")
set(DNDS_VERSION_DISTANCE "${DNDS_VERSION_DISTANCE}" CACHE INTERNAL "")
set(DNDS_VERSION_IS_RELEASE "${DNDS_VERSION_IS_RELEASE}" CACHE INTERNAL "")

message(STATUS "DNDSR version: ${DNDS_VERSION_FULL}")
message(STATUS "  base=${DNDS_VERSION_BASE} distance=${DNDS_VERSION_DISTANCE} commit=${DNDS_VERSION_COMMIT} release=${DNDS_VERSION_IS_RELEASE}")

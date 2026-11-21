# cmake/DndsTests.cmake
# C++ unit tests (doctest) and Python test (pytest) registration in CTest.

if(NOT DNDS_BUILD_TESTS)
    return()
endif()

set(DNDS_TEST_LEVEL "2" CACHE STRING "Highest test level built by ci_tests (0, 1 or 2)")
set_property(CACHE DNDS_TEST_LEVEL PROPERTY STRINGS 0 1 2)
if(NOT DNDS_TEST_LEVEL MATCHES "^[012]$")
    message(FATAL_ERROR "DNDS_TEST_LEVEL must be 0, 1 or 2")
endif()
enable_testing()

include(${CMAKE_CURRENT_LIST_DIR}/DndsTestTimeout.cmake)

add_subdirectory(${CMAKE_SOURCE_DIR}/test/cpp)

if(Python_EXECUTABLE)
    # Standalone tooling regressions need no extension modules or solver build.
    add_test(NAME harness_tooling
        COMMAND ${Python_EXECUTABLE} -m unittest discover -s ${CMAKE_SOURCE_DIR}/scripts/tests -v)
    set_tests_properties(harness_tooling PROPERTIES
        LABELS "tooling;level1"
        ENVIRONMENT "DNDS_HARNESS_CMAKE=${CMAKE_COMMAND}")
    _dnds_maybe_set_timeout(harness_tooling ${DNDS_TEST_TIMEOUT})

    # Importing DNDSR loads the four core extension packages. Cauchy bindings
    # and the Euler restart executable first enter the Python catalog at level 1.
    add_custom_target(ci_python_level0 DEPENDS dnds_pybind11 geom_pybind11 cfv_pybind11
        eulerP_pybind11 runtime_array_bridge_test)
    add_custom_target(ci_python_level1 DEPENDS ci_python_level0 cauchy_pybind11 euler)
    add_custom_target(ci_python_level2 DEPENDS ci_python_level1)
    add_custom_target(ci_python_tests DEPENDS ci_python_level2)
    foreach(_level RANGE 0 2)
        add_custom_target(ci_tests_level${_level} DEPENDS ci_cpp_level${_level} ci_python_level${_level})
    endforeach()
    add_custom_target(ci_tests DEPENDS ci_tests_level${DNDS_TEST_LEVEL})

    # Exact tiers here keep unfiltered CTest from running lower-level cases twice.
    foreach(_level RANGE 0 2)
        add_test(NAME pytest_level${_level}
            COMMAND ${Python_EXECUTABLE} -m pytest --with-mpi --dnds-ci-lane=serial --dnds-test-level=${_level} -m level${_level}
                ${CMAKE_SOURCE_DIR}/test --timeout=${DNDS_TEST_TIMEOUT}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        _dnds_maybe_set_timeout(pytest_level${_level} ${DNDS_TEST_TIMEOUT_PIPELINE})
        get_filename_component(_python_bin "${Python_EXECUTABLE}" DIRECTORY)
        set_tests_properties(pytest_level${_level} PROPERTIES LABELS "python;level${_level}"
            ENVIRONMENT "DNDS_EXTERNAL_LIB_CGNS=${DNDS_EXTERNAL_LIB_CGNS};DNDS_BUILD_DIR=${CMAKE_BINARY_DIR};PYTHONPATH=${CMAKE_SOURCE_DIR}/python:${CMAKE_BINARY_DIR}/test/cpp;PATH=${_python_bin}:$ENV{PATH}")
    endforeach()
endif()

# cmake/DndsTests.cmake
# C++ unit tests (doctest) and Python test (pytest) registration in CTest.

if(NOT DNDS_BUILD_TESTS)
    return()
endif()

enable_testing()

include(${CMAKE_CURRENT_LIST_DIR}/DndsTestTimeout.cmake)

add_subdirectory(${CMAKE_SOURCE_DIR}/test/cpp)

# Register pytest suites in CTest (serial only; use mpirun manually for MPI tests).
# Use the venv Python (Python_EXECUTABLE) with -m pytest so the correct
# interpreter, PATH, and linked libraries are always used -- even when CTest
# runs outside an activated venv.
# For manual pytest, build runtime_array_bridge_test and include both
# ${CMAKE_SOURCE_DIR}/python and <build>/test/cpp in PYTHONPATH.
get_filename_component(_DNDS_PYTHON_BIN_DIR "${Python_EXECUTABLE}" DIRECTORY)
if(Python_EXECUTABLE)
    # Standalone tooling regressions need no extension modules or solver build.
    add_test(NAME harness_tooling
        COMMAND ${Python_EXECUTABLE} -m unittest discover -s ${CMAKE_SOURCE_DIR}/scripts/tests -v)
    set_tests_properties(harness_tooling PROPERTIES
        LABELS "tooling"
        ENVIRONMENT "DNDS_HARNESS_CMAKE=${CMAKE_COMMAND}")
    _dnds_maybe_set_timeout(harness_tooling ${DNDS_TEST_TIMEOUT})

    add_test(NAME pytest_DNDS
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/test/DNDS/ -x --timeout=${DNDS_TEST_TIMEOUT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    _dnds_maybe_set_timeout(pytest_DNDS ${DNDS_TEST_TIMEOUT_PIPELINE})
    set_tests_properties(pytest_DNDS PROPERTIES
        LABELS "python"
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python:$<TARGET_FILE_DIR:runtime_array_bridge_test>;PATH=${_DNDS_PYTHON_BIN_DIR}:$ENV{PATH}")
    add_test(NAME pytest_CFV
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/test/CFV/ -x --timeout=${DNDS_TEST_TIMEOUT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    _dnds_maybe_set_timeout(pytest_CFV ${DNDS_TEST_TIMEOUT_PIPELINE})
    set_tests_properties(pytest_CFV PROPERTIES
        LABELS "python"
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python:$<TARGET_FILE_DIR:runtime_array_bridge_test>;PATH=${_DNDS_PYTHON_BIN_DIR}:$ENV{PATH}")
endif()

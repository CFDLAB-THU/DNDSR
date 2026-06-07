# cmake/DndsTests.cmake
# C++ unit tests (doctest) and Python test (pytest) registration in CTest.

if(NOT DNDS_BUILD_TESTS)
    return()
endif()

enable_testing()

# --- Read test timeout from env, default to 120 ---
if(DEFINED ENV{DNDS_TEST_TIMEOUT})
    set(DNDS_TEST_TIMEOUT "$ENV{DNDS_TEST_TIMEOUT}")
    set(DNDS_TEST_SET_TIMEOUT ON)
else()
    set(DNDS_TEST_TIMEOUT "120")
    set(DNDS_TEST_SET_TIMEOUT OFF)
endif()
message(STATUS "DNDS_TEST_TIMEOUT: ${DNDS_TEST_TIMEOUT} (set-timeout: ${DNDS_TEST_SET_TIMEOUT})")

math(EXPR DNDS_TEST_TIMEOUT_SOLVER "${DNDS_TEST_TIMEOUT} / 2")
math(EXPR DNDS_TEST_TIMEOUT_PIPELINE "${DNDS_TEST_TIMEOUT} * 5")

add_subdirectory(${CMAKE_SOURCE_DIR}/test/cpp)

# Register pytest suites in CTest (serial only; use mpirun manually for MPI tests).
# Use the venv Python (Python_EXECUTABLE) with -m pytest so the correct
# interpreter, PATH, and linked libraries are always used -- even when CTest
# runs outside an activated venv.
get_filename_component(_DNDS_PYTHON_BIN_DIR "${Python_EXECUTABLE}" DIRECTORY)
if(Python_EXECUTABLE)
    add_test(NAME pytest_DNDS
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/test/DNDS/ -x --timeout=${DNDS_TEST_TIMEOUT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    _dnds_maybe_set_timeout(pytest_DNDS ${DNDS_TEST_TIMEOUT_PIPELINE})
    set_tests_properties(pytest_DNDS PROPERTIES
        LABELS "python"
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python;PATH=${_DNDS_PYTHON_BIN_DIR}:$ENV{PATH}")
    add_test(NAME pytest_CFV
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/test/CFV/ -x --timeout=${DNDS_TEST_TIMEOUT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    _dnds_maybe_set_timeout(pytest_CFV ${DNDS_TEST_TIMEOUT_PIPELINE})
    set_tests_properties(pytest_CFV PROPERTIES
        LABELS "python"
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python;PATH=${_DNDS_PYTHON_BIN_DIR}:$ENV{PATH}")
endif()

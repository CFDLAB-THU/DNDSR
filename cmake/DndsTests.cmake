# cmake/DndsTests.cmake
# C++ unit tests (doctest) and Python test (pytest) registration in CTest.

if(NOT DNDS_BUILD_TESTS)
    return()
endif()

enable_testing()
add_subdirectory(${CMAKE_SOURCE_DIR}/test/cpp)

# Register pytest suites in CTest (serial only; use mpirun manually for MPI tests).
# Use the venv Python (Python_EXECUTABLE) with -m pytest so the correct
# interpreter, PATH, and linked libraries are always used -- even when CTest
# runs outside an activated venv.
get_filename_component(_DNDS_PYTHON_BIN_DIR "${Python_EXECUTABLE}" DIRECTORY)
if(Python_EXECUTABLE)
    add_test(NAME pytest_DNDS
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/test/DNDS/ -x --timeout=120
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(pytest_DNDS PROPERTIES
        TIMEOUT 180 LABELS "python"
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python;PATH=${_DNDS_PYTHON_BIN_DIR}:$ENV{PATH}")
    add_test(NAME pytest_CFV
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/test/CFV/ -x --timeout=120
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(pytest_CFV PROPERTIES
        TIMEOUT 180 LABELS "python"
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python;PATH=${_DNDS_PYTHON_BIN_DIR}:$ENV{PATH}")
endif()

# cmake/DndsTests.cmake
# C++ unit tests (doctest) and Python test (pytest) registration in CTest.

if(NOT DNDS_BUILD_TESTS)
    return()
endif()

enable_testing()
add_subdirectory(${CMAKE_SOURCE_DIR}/test/cpp)

# Register pytest suites in CTest (serial only; use mpirun manually for MPI tests).
find_program(DNDS_PYTEST_EXEC pytest)
if(DNDS_PYTEST_EXEC)
    add_test(NAME pytest_DNDS
        COMMAND ${DNDS_PYTEST_EXEC} ${CMAKE_SOURCE_DIR}/test/DNDS/ -x --timeout=120
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(pytest_DNDS PROPERTIES
        TIMEOUT 180 LABELS "python"
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python")
    add_test(NAME pytest_CFV
        COMMAND ${DNDS_PYTEST_EXEC} ${CMAKE_SOURCE_DIR}/test/CFV/ -x --timeout=120
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(pytest_CFV PROPERTIES
        TIMEOUT 180 LABELS "python"
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python")
endif()

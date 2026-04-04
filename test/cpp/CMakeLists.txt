# CMakeLists.txt for C++ unit tests (doctest)
#
# Usage:
#   cmake -DDNDS_BUILD_TESTS=ON ..
#   cmake --build . -t dnds_unit_tests -j8
#   ctest --test-dir . -R dnds_ --output-on-failure
#
# MPI tests are registered with mpirun -np 2 and -np 4.

find_path(DNDS_EXTERNAL_INCLUDE_DOCTEST doctest.h PATHS
    "${PROJECT_SOURCE_DIR}/external/doctest"
    NO_DEFAULT_PATH REQUIRED)

# --- helper: add a serial unit test executable ---
function(dnds_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "dnds_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    # serial test
    add_test(NAME dnds_${TEST_NAME} COMMAND ${TARGET_NAME})
    set_tests_properties(dnds_${TEST_NAME} PROPERTIES TIMEOUT 120)
endfunction()

# --- helper: add an MPI unit test executable ---
function(dnds_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "dnds_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    # np=1 (serial MPI)
    add_test(NAME dnds_${TEST_NAME}_np1
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_${TEST_NAME}_np1 PROPERTIES TIMEOUT 120)
    # np=2
    add_test(NAME dnds_${TEST_NAME}_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_${TEST_NAME}_np2 PROPERTIES TIMEOUT 120)
    # np=4
    add_test(NAME dnds_${TEST_NAME}_np4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_${TEST_NAME}_np4 PROPERTIES TIMEOUT 180)
endfunction()

# Aggregate target: build all unit tests
add_custom_target(dnds_unit_tests)

# --- DNDS core serial tests ---
dnds_add_unit_test(array "DNDS/test_Array.cpp" "dnds")

# --- DNDS core MPI tests ---
dnds_add_mpi_test(serializer "DNDS/test_Serializer.cpp" "dnds")
dnds_add_mpi_test(mpi "DNDS/test_MPI.cpp" "dnds")
dnds_add_mpi_test(array_transformer "DNDS/test_ArrayTransformer.cpp" "dnds")
dnds_add_mpi_test(array_derived "DNDS/test_ArrayDerived.cpp" "dnds")
dnds_add_mpi_test(array_dof "DNDS/test_ArrayDOF.cpp" "dnds")
dnds_add_mpi_test(index_mapping "DNDS/test_IndexMapping.cpp" "dnds")

# Add all test executables to the aggregate target
set(DNDS_ALL_TEST_TARGETS
    dnds_test_array
    dnds_test_serializer
    dnds_test_mpi
    dnds_test_array_transformer
    dnds_test_array_derived
    dnds_test_array_dof
    dnds_test_index_mapping
)
foreach(T ${DNDS_ALL_TEST_TARGETS})
    add_dependencies(dnds_unit_tests ${T})
endforeach()

# CMakeLists.txt for C++ unit tests (doctest)
#
# Usage:
#   cmake -DDNDS_BUILD_TESTS=ON ..
#   cmake --build . -t all_unit_tests -j8
#   ctest --test-dir . --output-on-failure
#
# Aggregate targets:
#   dnds_unit_tests  — only DNDS/ tests
#   geom_unit_tests  — only Geom/ tests
#   all_unit_tests   — everything
#
# MPI tests are registered with mpirun -np 1, 2, 4, and 8.

find_path(DNDS_EXTERNAL_INCLUDE_DOCTEST doctest.h PATHS
    "${PROJECT_SOURCE_DIR}/external/doctest"
    NO_DEFAULT_PATH REQUIRED)

# --- helper: add a serial unit test executable (DNDS) ---
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

# --- helper: add an MPI unit test executable (DNDS) ---
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
    set_tests_properties(dnds_${TEST_NAME}_np1 PROPERTIES TIMEOUT 120 PROCESSORS 1)
    # np=2
    add_test(NAME dnds_${TEST_NAME}_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_${TEST_NAME}_np2 PROPERTIES TIMEOUT 120 PROCESSORS 2)
    # np=4
    add_test(NAME dnds_${TEST_NAME}_np4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_${TEST_NAME}_np4 PROPERTIES TIMEOUT 180 PROCESSORS 4)
    # np=8
    add_test(NAME dnds_${TEST_NAME}_np8
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_${TEST_NAME}_np8 PROPERTIES TIMEOUT 240 PROCESSORS 8)
endfunction()

# --- helper: add a serial unit test executable (Geom) ---
function(geom_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "geom_test_${TEST_NAME}")
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
    add_test(NAME geom_${TEST_NAME} COMMAND ${TARGET_NAME})
    set_tests_properties(geom_${TEST_NAME} PROPERTIES TIMEOUT 120)
endfunction()

# --- helper: add an MPI unit test executable (Geom) ---
function(geom_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "geom_test_${TEST_NAME}")
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
    # np=1
    add_test(NAME geom_${TEST_NAME}_np1
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(geom_${TEST_NAME}_np1 PROPERTIES TIMEOUT 120 PROCESSORS 1)
    # np=2
    add_test(NAME geom_${TEST_NAME}_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(geom_${TEST_NAME}_np2 PROPERTIES TIMEOUT 120 PROCESSORS 2)
    # np=4
    add_test(NAME geom_${TEST_NAME}_np4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(geom_${TEST_NAME}_np4 PROPERTIES TIMEOUT 180 PROCESSORS 4)
    # np=8
    add_test(NAME geom_${TEST_NAME}_np8
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(geom_${TEST_NAME}_np8 PROPERTIES TIMEOUT 240 PROCESSORS 8)
endfunction()

# ===================================================================
# Aggregate targets
# ===================================================================
add_custom_target(dnds_unit_tests)   # DNDS/ tests only
add_custom_target(geom_unit_tests)   # Geom/ tests only
add_custom_target(all_unit_tests)    # everything
add_dependencies(all_unit_tests dnds_unit_tests geom_unit_tests)

# ===================================================================
# DNDS core tests
# ===================================================================
dnds_add_unit_test(array "DNDS/test_Array.cpp" "dnds")

dnds_add_mpi_test(serializer "DNDS/test_Serializer.cpp" "dnds")
dnds_add_mpi_test(mpi "DNDS/test_MPI.cpp" "dnds")
dnds_add_mpi_test(array_transformer "DNDS/test_ArrayTransformer.cpp" "dnds")
dnds_add_mpi_test(array_derived "DNDS/test_ArrayDerived.cpp" "dnds")
dnds_add_mpi_test(array_dof "DNDS/test_ArrayDOF.cpp" "dnds")
dnds_add_mpi_test(index_mapping "DNDS/test_IndexMapping.cpp" "dnds")

set(DNDS_TEST_TARGETS
    dnds_test_array
    dnds_test_serializer
    dnds_test_mpi
    dnds_test_array_transformer
    dnds_test_array_derived
    dnds_test_array_dof
    dnds_test_index_mapping
)
foreach(T ${DNDS_TEST_TARGETS})
    add_dependencies(dnds_unit_tests ${T})
endforeach()

# ===================================================================
# Geom tests
# ===================================================================
geom_add_unit_test(elements "Geom/test_Elements.cpp" "geom;dnds")
geom_add_unit_test(quadrature "Geom/test_Quadrature.cpp" "geom;dnds")
geom_add_mpi_test(mesh_index_conversion "Geom/test_MeshIndexConversion.cpp" "geom;dnds")
geom_add_mpi_test(mesh_pipeline "Geom/test_MeshPipeline.cpp" "geom;dnds")
geom_add_mpi_test(mesh_distributed_read "Geom/test_MeshDistributedRead.cpp" "geom;dnds")

set(GEOM_TEST_TARGETS
    geom_test_elements
    geom_test_quadrature
    geom_test_mesh_index_conversion
    geom_test_mesh_pipeline
    geom_test_mesh_distributed_read
)
foreach(T ${GEOM_TEST_TARGETS})
    add_dependencies(geom_unit_tests ${T})
endforeach()

# ===================================================================
# CFV tests
# ===================================================================

# --- helper: add a serial unit test executable (CFV) ---
function(cfv_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "cfv_test_${TEST_NAME}")
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
    add_test(NAME cfv_${TEST_NAME} COMMAND ${TARGET_NAME})
    set_tests_properties(cfv_${TEST_NAME} PROPERTIES TIMEOUT 120)
endfunction()

# --- helper: add an MPI unit test executable (CFV) ---
function(cfv_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "cfv_test_${TEST_NAME}")
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
    add_test(NAME cfv_${TEST_NAME}_np1
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cfv_${TEST_NAME}_np1 PROPERTIES TIMEOUT 120 PROCESSORS 1)
    add_test(NAME cfv_${TEST_NAME}_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cfv_${TEST_NAME}_np2 PROPERTIES TIMEOUT 120 PROCESSORS 2)
    add_test(NAME cfv_${TEST_NAME}_np4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cfv_${TEST_NAME}_np4 PROPERTIES TIMEOUT 180 PROCESSORS 4)
endfunction()

add_custom_target(cfv_unit_tests)
add_dependencies(all_unit_tests cfv_unit_tests)

cfv_add_mpi_test(reconstruction "CFV/test_Reconstruction.cpp" "cfv;geom;dnds")

set(CFV_TEST_TARGETS
    cfv_test_reconstruction
)

# --- CUDA CFV tests ---
if(DNDS_USE_CUDA)
    # helper: add a CUDA MPI test executable for CFV
    function(cfv_add_cuda_mpi_test TEST_NAME SRC_FILE LIBS)
        set(TARGET_NAME "cfv_test_${TEST_NAME}")
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
        # np=1 only for CUDA tests (single GPU)
        add_test(NAME cfv_${TEST_NAME}_np1
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(cfv_${TEST_NAME}_np1 PROPERTIES TIMEOUT 120 PROCESSORS 1)
    endfunction()

    cfv_add_cuda_mpi_test(device_transferable "CFV/test_DeviceTransferable.cu" "cfv;geom;dnds")
    list(APPEND CFV_TEST_TARGETS cfv_test_device_transferable)
endif()

foreach(T ${CFV_TEST_TARGETS})
    add_dependencies(cfv_unit_tests ${T})
endforeach()

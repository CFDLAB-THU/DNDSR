# CMakeLists.txt for C++ unit tests (doctest)
#
# Usage:
#   cmake -DDNDS_BUILD_TESTS=ON ..
#   cmake --build . -t all_unit_tests -j8
#   ctest --test-dir . --output-on-failure
#
# For aggregated doctest summary (test cases + assertions):
#   python scripts/ctest_summary.py [ctest args...]
#
# Aggregate targets:
#   dnds_unit_tests  — only DNDS/ tests
#   geom_unit_tests  — only Geom/ tests
#   all_unit_tests   — everything
#
# MPI tests are registered for each np in DNDS_TEST_NP_LIST.
# Set via environment variable: DNDS_TEST_NP_LIST="1;2;4" cmake ..
# Default: 1;2;4;8
#
# OpenMP threads can be configured via DNDS_TEST_OMP_THREADS (default: 2).

# --- Read NP list from env, default to 1;2;4;8 ---
if(DEFINED ENV{DNDS_TEST_NP_LIST})
    set(DNDS_TEST_NP_LIST "$ENV{DNDS_TEST_NP_LIST}")
else()
    set(DNDS_TEST_NP_LIST "1;2;4;8")
endif()
message(STATUS "DNDS_TEST_NP_LIST: ${DNDS_TEST_NP_LIST}")

# --- Read OMP threads from env, default to 2 ---
if(DEFINED ENV{DNDS_TEST_OMP_THREADS})
    set(DNDS_TEST_OMP_THREADS "$ENV{DNDS_TEST_OMP_THREADS}")
else()
    set(DNDS_TEST_OMP_THREADS "2")
endif()
message(STATUS "DNDS_TEST_OMP_THREADS: ${DNDS_TEST_OMP_THREADS}")

# --- Detect OpenMPI and add --oversubscribe if needed ---
# OpenMPI requires --oversubscribe when np > physical cores (common in CI/testing).
# MPICH/Intel MPI do not need (or recognize) this flag.
#
# Also support DNDS_TEST_MPIEXTRA: space-separated flags appended to every
# add_test MPI command (e.g. "--oversubscribe").  On GitHub runners set
# OMPI_MCA_rmaps_base_oversubscribe=1 as a belt-and-suspenders fallback.
if(NOT "$ENV{DNDS_TEST_MPIEXTRA}" STREQUAL "")
    separate_arguments(_DNDS_MPIEXTRA UNIX_COMMAND "$ENV{DNDS_TEST_MPIEXTRA}")
    list(APPEND MPIEXEC_PREFLAGS ${_DNDS_MPIEXTRA})
    message(STATUS "DNDS_TEST_MPIEXTRA: appending ${_DNDS_MPIEXTRA} to MPIEXEC_PREFLAGS")
elseif(MPIEXEC_EXECUTABLE)
    execute_process(
        COMMAND ${MPIEXEC_EXECUTABLE} --version
        OUTPUT_VARIABLE _MPIEXEC_VERSION_OUTPUT
        ERROR_VARIABLE _MPIEXEC_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_MPIEXEC_VERSION_OUTPUT MATCHES "Open MPI")
        message(STATUS "Detected OpenMPI: adding --oversubscribe to MPIEXEC_PREFLAGS for tests")
        list(APPEND MPIEXEC_PREFLAGS "--oversubscribe")
    endif()
endif()

# --- Timeout per np ---
function(_dnds_np_timeout NP OUT_VAR)
    if(NP LESS_EQUAL 2)
        set(${OUT_VAR} ${DNDS_TEST_TIMEOUT} PARENT_SCOPE)
    elseif(NP LESS_EQUAL 4)
        math(EXPR _t "${DNDS_TEST_TIMEOUT} * 3 / 2")
        set(${OUT_VAR} ${_t} PARENT_SCOPE)
    else()
        math(EXPR _t "${DNDS_TEST_TIMEOUT} * 2")
        set(${OUT_VAR} ${_t} PARENT_SCOPE)
    endif()
endfunction()

find_path(DNDS_EXTERNAL_INCLUDE_DOCTEST doctest.h PATHS
    "${PROJECT_SOURCE_DIR}/external/doctest"
    NO_DEFAULT_PATH REQUIRED)

# --- helper: add a serial unit test executable (DNDS) ---
function(dnds_add_unit_test TEST_NAME SRC_FILE LIBS)
    cmake_parse_arguments(PARSE_ARGV 3 test "TARGET_LINK_ONLY" "" "")
    set(TARGET_NAME "dnds_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS})
    if(NOT test_TARGET_LINK_ONLY)
        target_link_libraries(${TARGET_NAME} PRIVATE ${DNDS_EXTERNAL_LIBS})
    endif()
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    add_test(NAME dnds_${TEST_NAME} COMMAND ${TARGET_NAME})
    _dnds_maybe_set_timeout(dnds_${TEST_NAME} ${DNDS_TEST_TIMEOUT})
    set_tests_properties(dnds_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
endfunction()

# --- helper: add an MPI unit test executable (DNDS) ---
# MPI tests run all TEST_CASEs together per NP value (avoids mpirun overhead per test case).
function(dnds_add_mpi_test TEST_NAME SRC_FILE LIBS)
    cmake_parse_arguments(PARSE_ARGV 3 _test "LOCAL_CASES_ONCE" "" "CRITICAL_NP4")
    set(TARGET_NAME "dnds_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
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
    # Keep rank-local cases at np1, or at the first configured rank count
    # when np1 is absent. MPI-dependent cases still run at every count.
    if(_test_LOCAL_CASES_ONCE AND DNDS_TEST_NP_LIST)
        if("1" IN_LIST DNDS_TEST_NP_LIST)
            set(_local_case_np 1)
        else()
            list(GET DNDS_TEST_NP_LIST 0 _local_case_np)
        endif()
    endif()
    foreach(NP ${DNDS_TEST_NP_LIST})
        set(_case_filter)
        if(DEFINED _local_case_np AND NOT NP EQUAL _local_case_np)
            set(_case_filter --test-suite-exclude=rank-local)
        endif()
        _dnds_np_timeout(${NP} _TIMEOUT)
        if(NP EQUAL 4 AND _test_CRITICAL_NP4)
            list(JOIN _test_CRITICAL_NP4 "," _critical_cases)
            set(_critical_name dnds_${TEST_NAME}_np4_critical)
            add_test(NAME ${_critical_name}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS}
                        "--test-case=${_critical_cases}")
            _dnds_maybe_set_timeout(${_critical_name} ${_TIMEOUT})
            set_tests_properties(${_critical_name} PROPERTIES PROCESSORS 4
                ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}" LABELS "mpi-high")
            set_property(GLOBAL APPEND PROPERTY DNDS_CRITICAL_MPI_TESTS "${_critical_name}:${TARGET_NAME}")
            list(APPEND _case_filter "--test-case-exclude=${_critical_cases}")
        endif()
        add_test(NAME dnds_${TEST_NAME}_np${NP}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS} ${_case_filter})
        _dnds_maybe_set_timeout(dnds_${TEST_NAME}_np${NP} ${_TIMEOUT})
        set_tests_properties(dnds_${TEST_NAME}_np${NP} PROPERTIES
            PROCESSORS ${NP}
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    endforeach()
endfunction()

# --- helper: add a serial unit test executable (Geom) ---
function(geom_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "geom_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
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
    _dnds_maybe_set_timeout(geom_${TEST_NAME} ${DNDS_TEST_TIMEOUT})
    set_tests_properties(geom_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
endfunction()

# --- helper: add an MPI unit test executable (Geom) ---
# MPI tests run all TEST_CASEs together per NP value (avoids mpirun overhead per test case).
function(geom_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "geom_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
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
    foreach(NP ${DNDS_TEST_NP_LIST})
        _dnds_np_timeout(${NP} _TIMEOUT)
        add_test(NAME geom_${TEST_NAME}_np${NP}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
        _dnds_maybe_set_timeout(geom_${TEST_NAME}_np${NP} ${_TIMEOUT})
        set_tests_properties(geom_${TEST_NAME}_np${NP} PROPERTIES
            PROCESSORS ${NP}
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    endforeach()
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
dnds_add_unit_test(runtime "DNDS/test_Runtime.cpp" "dnds")
dnds_add_unit_test(runtime_array "DNDS/test_RuntimeArray.cpp" "dnds")
dnds_add_unit_test(source_organization "DNDS/test_SourceOrganization.cpp" "dnds")
target_link_libraries(dnds_test_source_organization PRIVATE dnds_compat_headers)
add_dependencies(dnds_unit_tests dnds_test_source_organization)
dnds_add_unit_test(source_organization_native "DNDS/test_SourceOrganizationNative.cpp" "geom;dnds")
add_dependencies(dnds_unit_tests dnds_test_source_organization_native)
dnds_add_unit_test(runtime_element_schema "DNDS/test_RuntimeElementSchema.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_runtime_element_schema)
dnds_add_unit_test(legacy_array_bridge "DNDS/test_LegacyArrayBridge.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_legacy_array_bridge)
dnds_add_unit_test(runtime_row_builder "DNDS/test_RuntimeRowBuilder.cpp" "dnds")
dnds_add_unit_test(runtime_engine "DNDS/test_EagerEngine.cpp" "dnds")
dnds_add_unit_test(access_tracking "DNDS/test_AccessTracking.cpp" "dnds")
dnds_add_unit_test(cauchy_config "Cauchy/test_Config.cpp" "cauchy;cfv;geom;dnds")
dnds_add_unit_test(cauchy_progress_output "Cauchy/test_ProgressOutput.cpp" "cauchy;cfv;geom;dnds")
dnds_add_unit_test(cauchy_config_shared "Cauchy/test_ConfigShared.cpp" "cauchy_shared" TARGET_LINK_ONLY)
# Reproduce a real pybind-style consumer of exported native configuration APIs.
set_target_properties(dnds_test_cauchy_config_shared PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
add_dependencies(dnds_unit_tests dnds_test_cauchy_config_shared)
dnds_add_unit_test(cauchy_physics "Cauchy/test_PerfectGas.cpp" "cauchy;cfv;geom;dnds")
dnds_add_unit_test(cauchy_state_input "Cauchy/test_StateInput.cpp" "cauchy;physics;cfv;geom;dnds")
dnds_add_unit_test(cauchy_rans_state "Cauchy/test_EquationRANS.cpp" "cauchy;physics;cfv;geom;dnds")
# Header-policy-only executable: the feature switch must not mix with a Cauchy
# library compiled without it. No Cauchy implementation is linked here.
dnds_add_unit_test(cauchy_rans_reconstruction_reduced "Cauchy/test_RANSReconstructionReduced.cpp" "physics;dnds")
target_compile_definitions(dnds_test_cauchy_rans_reconstruction_reduced PRIVATE USE_NS_SA_NUT_REDUCED_ORDER)
dnds_add_unit_test(cauchy_derived_fields "Cauchy/test_DerivedFields.cpp" "cauchy;cfv;geom;dnds")
if(DNDS_USE_CANTERA)
    set_property(TEST dnds_cauchy_derived_fields APPEND PROPERTY ENVIRONMENT
        "DNDS_MECH_PATH=${DNDS_CANTERA_DATA_DIR}" "CANTERA_DATA=${DNDS_CANTERA_DATA_DIR}")
    dnds_add_mpi_test(cauchy_equation_source "Cauchy/test_EquationSource.cpp" "cauchy;physics;cfv;geom;dnds")
    foreach(NP ${DNDS_TEST_NP_LIST})
        set_property(TEST dnds_cauchy_equation_source_np${NP} APPEND PROPERTY ENVIRONMENT
            "DNDS_MECH_PATH=${DNDS_CANTERA_DATA_DIR}" "CANTERA_DATA=${DNDS_CANTERA_DATA_DIR}")
    endforeach()
endif()
dnds_add_unit_test(cauchy_rigid_motion "Cauchy/test_RigidMotion.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_boundary_inputs "Cauchy/test_BoundaryInputs.cpp" "cauchy;cfv;geom;dnds" LOCAL_CASES_ONCE)
dnds_add_mpi_test(cauchy_wall_distance "Cauchy/test_WallDistanceSamples.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_rans_fv "Cauchy/test_RANSFiniteVolume.cpp" "cauchy;physics;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_multidomain_mesh "Cauchy/test_MultidomainMesh.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_multidomain "Cauchy/test_Multidomain.cpp" "cauchy;cfv;geom;dnds"
    CRITICAL_NP4 "Multi-domain BDF2 restart reserves every domain and redistributes by persistent IDs")
# Fixture-only exact predicates use CGAL's header-only Boost backend. Disable
# automatic GMP selection too: the CORE setting alone leaves that enabled when
# GMP/MPFR headers happen to exist, introducing an undeclared link dependency.
target_compile_definitions(dnds_test_cauchy_multidomain_mesh PRIVATE CGAL_CORE_USE_BOOST_BACKEND=1 CGAL_DISABLE_GMP=1)
target_compile_definitions(dnds_test_cauchy_multidomain PRIVATE CGAL_CORE_USE_BOOST_BACKEND=1 CGAL_DISABLE_GMP=1)
dnds_add_mpi_test(cauchy_equation_dofs "Cauchy/test_EquationDofs.cpp" "cauchy;cfv;geom;dnds")
dnds_add_unit_test(cauchy_riemann "Cauchy/test_Riemann.cpp" "cauchy;cfv;geom;dnds")
dnds_add_unit_test(cauchy_riemann_legacy "Cauchy/test_RiemannLegacy.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_finite_volume "Cauchy/test_FiniteVolume.cpp" "cauchy;cfv;geom;dnds")
if(TEST dnds_cauchy_finite_volume_np2)
    # Exercise opt-in communication execution without weakening the remaining
    # FUNNELED test placements or relying on an MPI-vendor environment override.
    set_property(TEST dnds_cauchy_finite_volume_np2 APPEND PROPERTY ENVIRONMENT "DNDS_TEST_COMMUNICATION_WORKER=1")
endif()
dnds_add_mpi_test(cauchy_finite_volume_physics "Cauchy/test_FiniteVolumePhysics.cpp" "cauchy;cfv;geom;dnds")
target_compile_definitions(dnds_test_cauchy_finite_volume_physics PRIVATE DNDS_TEST_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
if(DNDS_USE_CANTERA)
    foreach(NP ${DNDS_TEST_NP_LIST})
        set_property(TEST dnds_cauchy_finite_volume_physics_np${NP} APPEND PROPERTY ENVIRONMENT
            "DNDS_MECH_PATH=${DNDS_CANTERA_DATA_DIR}" "CANTERA_DATA=${DNDS_CANTERA_DATA_DIR}")
    endforeach()
endif()
dnds_add_mpi_test(cauchy_limited_vr "Cauchy/test_LimitedVR.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_limited_vr_checkpoint "Cauchy/test_LimitedVRCheckpoint.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_foundation "Cauchy/test_Foundation.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_interface_iteration "Cauchy/test_InterfaceIteration.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_linear "Cauchy/test_LinearProblem.cpp" "cauchy;cfv;geom;dnds")
if(DNDS_USE_OMP AND DNDS_DIST_MT_USE_OMP)
    add_test(NAME dnds_cauchy_sgs_subpartitions_omp_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:dnds_test_cauchy_linear> "--test-case=SGS mesh subpartitions*" ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_cauchy_sgs_subpartitions_omp_np2 PROPERTIES
        PROCESSORS 4 TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=2;DNDS_DIST_OMP_NUM_THREADS=2")
endif()
dnds_add_mpi_test(cauchy_linear_solver "Cauchy/test_LinearSolver.cpp" "cauchy;cfv;geom;dnds")
add_test(NAME dnds_cauchy_linear_solver_abort_np2
    COMMAND ${CMAKE_COMMAND}
        "-DTEST_COMMAND=${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};2;${MPIEXEC_PREFLAGS};$<TARGET_FILE:dnds_test_cauchy_linear_solver>;--test-case=Fixed-count final finite check*;${MPIEXEC_POSTFLAGS}"
        "-DEXPECTED_DIAGNOSTIC=nonfinite fixed-count linear correction"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/ExpectMPIFailure.cmake)
set_tests_properties(dnds_cauchy_linear_solver_abort_np2 PROPERTIES
    TIMEOUT 30 PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1;DNDS_TEST_LINEAR_FATAL_LOCAL=1")
dnds_add_mpi_test(cauchy_nonlinear "Cauchy/test_NonlinearSolver.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_polynomial_multigrid "Cauchy/test_PolynomialMultigrid.cpp" "cauchy;cfv;geom;dnds")
add_executable(dnds_test_distributed_failure Cauchy/test_DistributedFailure.cpp)
target_include_directories(dnds_test_distributed_failure PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(dnds_test_distributed_failure PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
set_target_properties(dnds_test_distributed_failure PROPERTIES EXCLUDE_FROM_ALL ON)
foreach(NP ${DNDS_TEST_NP_LIST})
    if(NP GREATER 1)
        foreach(MODE factory transfer reducer-completion solver-checked-preflight solver-unchecked-preflight solver-worker)
            add_test(NAME dnds_distributed_failure_${MODE}_np${NP}
                COMMAND ${CMAKE_COMMAND} "-DMPIEXEC=${MPIEXEC_EXECUTABLE}" "-DNP_FLAG=${MPIEXEC_NUMPROC_FLAG}"
                    "-DNP=${NP}" "-DMPI_FLAGS=${MPIEXEC_PREFLAGS}" "-DFIXTURE=$<TARGET_FILE:dnds_test_distributed_failure>"
                    "-DMODE=${MODE}" -P ${CMAKE_CURRENT_SOURCE_DIR}/Cauchy/ExpectDistributedFailure.cmake)
            set_tests_properties(dnds_distributed_failure_${MODE}_np${NP} PROPERTIES
                TIMEOUT 30 PROCESSORS ${NP} ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endif()
endforeach()
dnds_add_mpi_test(cauchy_implicit_domain "Cauchy/test_ImplicitDomain.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_checkpoint "Cauchy/test_Checkpoint.cpp" "cauchy;cfv;geom;dnds"
    CRITICAL_NP4 "Checkpoint changes MPI partition and polynomial degree using persistent cell IDs with cold reconstruction")
if(DNDS_USE_CANTERA)
    foreach(NP ${DNDS_TEST_NP_LIST})
        set_property(TEST dnds_cauchy_checkpoint_np${NP} APPEND PROPERTY ENVIRONMENT
            "DNDS_MECH_PATH=${DNDS_CANTERA_DATA_DIR}" "CANTERA_DATA=${DNDS_CANTERA_DATA_DIR}")
    endforeach()
endif()
dnds_add_mpi_test(cauchy_physical_timestep "Cauchy/test_PhysicalTimeStep.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_viscous_flux "Cauchy/test_ViscousFlux.cpp" "cauchy;cfv;geom;dnds")
dnds_add_mpi_test(cauchy_auxiliary_preparation "Cauchy/test_AuxiliaryPreparation.cpp" "cauchy;cfv;geom;dnds")
add_executable(cauchy_benchmark_implicit EXCLUDE_FROM_ALL Cauchy/benchmark_Implicit.cpp)
add_executable(cauchy_benchmark_block_lu_comparison EXCLUDE_FROM_ALL Cauchy/benchmark_BlockLUComparison.cpp)
target_include_directories(cauchy_benchmark_block_lu_comparison PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_benchmark_block_lu_comparison PRIVATE cfv geom dnds ${DNDS_EXTERNAL_LIBS})
add_executable(cauchy_benchmark_finite_volume_dispatch EXCLUDE_FROM_ALL Cauchy/benchmark_FiniteVolumeDispatch.cpp)
target_include_directories(cauchy_benchmark_finite_volume_dispatch PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_benchmark_finite_volume_dispatch PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
target_include_directories(cauchy_benchmark_implicit PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_benchmark_implicit PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
add_executable(cauchy_benchmark_geom_acquisition EXCLUDE_FROM_ALL Cauchy/benchmark_GeomAcquisition.cpp)
target_include_directories(cauchy_benchmark_geom_acquisition PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_benchmark_geom_acquisition PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
add_executable(cauchy_benchmark_jacobian EXCLUDE_FROM_ALL Cauchy/benchmark_Jacobian.cpp)
target_include_directories(cauchy_benchmark_jacobian PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_benchmark_jacobian PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
add_executable(cauchy_benchmark_nonlinear_stage EXCLUDE_FROM_ALL Cauchy/benchmark_NonlinearStage.cpp)
target_include_directories(cauchy_benchmark_nonlinear_stage PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_benchmark_nonlinear_stage PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
if(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(cauchy_benchmark_rhs_execution EXCLUDE_FROM_ALL Cauchy/benchmark_RHSExecution.cpp Cauchy/RHSExecutionBenchmark.cpp)
    target_include_directories(cauchy_benchmark_rhs_execution PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cauchy_benchmark_rhs_execution PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    target_link_options(cauchy_benchmark_rhs_execution PRIVATE
        "LINKER:--wrap=_ZNK4DNDS3CFV16FiniteVolumePlan8BindHaloERNS_7Algebra8DofFieldIdLm18446744073709551615EEENS_7Runtime15TransferOptionsE"
        "LINKER:--wrap=_ZNK4DNDS3CFV16FiniteVolumePlan15BindTrackedHaloERNS_7Algebra8DofFieldIdLm18446744073709551615EEENS_7Runtime15TransferOptionsE"
        "LINKER:--wrap=_ZN4DNDS7Runtime6Detail9CopyBytesEPKvNS0_6DeviceEPvS4_mNS0_16ExecutionContextE"
        "LINKER:--wrap=_ZNK4DNDS3CFV29VariationalReconstructionPlan22DoReconstructionIterToERKNS_7Algebra8DofFieldIdLm18446744073709551615EEES6_RNS0_23ReconstructionWorkspaceERKNS0_28ReconstructionBoundaryValuesENS0_20ReconstructionRegionE")
    add_executable(cauchy_benchmark_coefficient_rhs EXCLUDE_FROM_ALL Cauchy/benchmark_CoefficientRHSCost.cpp)
    target_include_directories(cauchy_benchmark_coefficient_rhs PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cauchy_benchmark_coefficient_rhs PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    target_link_options(cauchy_benchmark_coefficient_rhs PRIVATE
        "LINKER:--wrap=_ZNK4DNDS3CFV29VariationalReconstructionPlan20CommitReconstructionERNS_7Algebra8DofFieldIdLm18446744073709551615EEERNS0_23ReconstructionWorkspaceE")
endif()
add_executable(cauchy_experiment_multidomain_iv EXCLUDE_FROM_ALL Cauchy/experiment_MultidomainIV.cpp)
add_executable(cauchy_experiment_acm_vortex_projection EXCLUDE_FROM_ALL Cauchy/experiment_ACMVortexProjection.cpp)
target_include_directories(cauchy_experiment_acm_vortex_projection PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_experiment_acm_vortex_projection PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
target_compile_definitions(cauchy_experiment_acm_vortex_projection PRIVATE CGAL_CORE_USE_BOOST_BACKEND=1 CGAL_DISABLE_GMP=1)
target_include_directories(cauchy_experiment_multidomain_iv PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_experiment_multidomain_iv PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
target_compile_definitions(cauchy_experiment_multidomain_iv PRIVATE CGAL_CORE_USE_BOOST_BACKEND=1 CGAL_DISABLE_GMP=1)
target_compile_definitions(dnds_test_cauchy_checkpoint PRIVATE DNDS_TEST_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
dnds_add_mpi_test(cauchy_jacobian "Cauchy/test_Jacobian.cpp" "cauchy;cfv;geom;dnds" LOCAL_CASES_ONCE)
add_custom_target(cauchy_unit_tests)
if(Python_EXECUTABLE)
    add_dependencies(cauchy_unit_tests cauchy_solver)
    foreach(NP 1 2)
        add_test(NAME cauchy_frontend_np${NP}
            COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/Cauchy/frontend_smoke.py
                --exe $<TARGET_FILE:cauchy_solver> --source ${PROJECT_SOURCE_DIR}
                --mpi ${MPIEXEC_EXECUTABLE} --np ${NP})
        set_tests_properties(cauchy_frontend_np${NP} PROPERTIES TIMEOUT 240
            ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;OMPI_MCA_rmaps_base_oversubscribe=1")
    endforeach()
endif()
if(DNDS_USE_CANTERA)
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_equation_source)
endif()
add_dependencies(cauchy_unit_tests dnds_test_cauchy_equation_dofs dnds_test_cauchy_finite_volume_physics)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_wall_distance)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_rans_state dnds_test_cauchy_state_input dnds_test_cauchy_rans_reconstruction_reduced)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_rans_fv)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_interface_iteration)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_limited_vr dnds_test_cauchy_limited_vr_checkpoint)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_config dnds_test_cauchy_config_shared dnds_test_cauchy_progress_output dnds_test_cauchy_physics dnds_test_cauchy_derived_fields dnds_test_cauchy_riemann dnds_test_cauchy_riemann_legacy dnds_test_cauchy_finite_volume dnds_test_cauchy_foundation dnds_test_cauchy_linear dnds_test_cauchy_linear_solver dnds_test_cauchy_jacobian)
add_dependencies(all_unit_tests cauchy_unit_tests)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_nonlinear)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_polynomial_multigrid)
add_dependencies(cauchy_unit_tests dnds_test_distributed_failure)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_implicit_domain)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_checkpoint)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_physical_timestep)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_rigid_motion)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_boundary_inputs)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_multidomain_mesh)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_multidomain)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_viscous_flux)
add_dependencies(cauchy_unit_tests dnds_test_cauchy_auxiliary_preparation)
add_dependencies(dnds_unit_tests dnds_test_access_tracking)
add_dependencies(dnds_unit_tests dnds_test_runtime_engine)
add_dependencies(dnds_unit_tests dnds_test_runtime_row_builder)
add_dependencies(dnds_unit_tests dnds_test_runtime_array)
# Separate extension exercises sharing across pybind modules, not test-only
# methods embedded in the production Runtime binding. Build/install the normal
# Python modules first, then set PYTHONPATH to include this target's directory.
# Do not apply pybind11's host LTO/strip extras to the CUDA device-link object.
pybind11_add_module(runtime_array_bridge_test MODULE EXCLUDE_FROM_ALL NO_EXTRAS DNDS/RuntimeArrayBridge_bind.cpp)
target_include_directories(runtime_array_bridge_test PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(runtime_array_bridge_test PRIVATE dnds_shared ${DNDS_EXTERNAL_LIBS})
if(DNDS_USE_CUDA)
    add_executable(cfv_test_distributed_interface_sampling_cuda EXCLUDE_FROM_ALL CFV/test_DistributedInterfaceSamplingCUDA.cu)
    target_include_directories(cfv_test_distributed_interface_sampling_cuda PRIVATE ${DNDS_EXTERNAL_INCLUDE_DOCTEST} ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cfv_test_distributed_interface_sampling_cuda PRIVATE cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    add_test(NAME cfv_distributed_interface_sampling_cuda_np1 COMMAND $<TARGET_FILE:cfv_test_distributed_interface_sampling_cuda>)
    add_test(NAME cfv_distributed_interface_sampling_cuda_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py $<TARGET_FILE:cfv_test_distributed_interface_sampling_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cfv_distributed_interface_sampling_cuda_np1 cfv_distributed_interface_sampling_cuda_np2 PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1")
    add_executable(cauchy_test_interface_cuda EXCLUDE_FROM_ALL Cauchy/test_InterfaceCUDA.cu)
    target_include_directories(cauchy_test_interface_cuda PRIVATE ${DNDS_EXTERNAL_INCLUDE_DOCTEST} ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cauchy_test_interface_cuda PRIVATE cauchy physics cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    add_test(NAME cauchy_interface_cuda_np1 COMMAND $<TARGET_FILE:cauchy_test_interface_cuda>)
    add_test(NAME cauchy_interface_cuda_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py $<TARGET_FILE:cauchy_test_interface_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cauchy_interface_cuda_np1 cauchy_interface_cuda_np2 PROPERTIES TIMEOUT 240 ENVIRONMENT "OMP_NUM_THREADS=1")
    add_dependencies(cauchy_unit_tests cauchy_test_interface_cuda)
    target_sources(runtime_array_bridge_test PRIVATE DNDS/RuntimeArrayBridge_CUDA.cu ../demos/MockStencil.cu)
endif()
set_target_properties(runtime_array_bridge_test PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
# The registered Python tests exercise this extension too. Building the full
# test target must make it available even in a fresh separate build directory.
add_dependencies(all_unit_tests runtime_array_bridge_test)
dnds_add_unit_test(config "DNDS/test_Config.cpp" "dnds")
dnds_add_unit_test(csv_log "DNDS/test_CsvLog.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_csv_log)
dnds_add_unit_test(hard_eigen "DNDS/test_HardEigen.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_hard_eigen)
dnds_add_mpi_test(object_contracts "DNDS/test_ObjectContracts.cpp" "dnds" LOCAL_CASES_ONCE)
target_sources(dnds_test_object_contracts PRIVATE DNDS/test_LocalChecks.cpp)

dnds_add_mpi_test(serializer "DNDS/test_Serializer.cpp" "dnds")
dnds_add_mpi_test(mpi "DNDS/test_MPI.cpp" "dnds")
dnds_add_mpi_test(runtime_exchange "DNDS/test_RuntimeExchange.cpp" "dnds")
dnds_add_mpi_test(runtime_row_transfer "DNDS/test_RuntimeRowTransfer.cpp" "dnds"
    CRITICAL_NP4 "Prepared CPU transfer groups retain separate payloads and perform one optional agreement"
                 "Communication worker preserves repeated packed and direct sparse transfers")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(dnds_benchmark_transfer_scheduling EXCLUDE_FROM_ALL DNDS/benchmark_TransferScheduling.cpp)
    target_include_directories(dnds_benchmark_transfer_scheduling PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(dnds_benchmark_transfer_scheduling PRIVATE dnds ${DNDS_EXTERNAL_LIBS})
endif()
if(Python_EXECUTABLE)
    add_test(NAME dnds_runtime_preflight_off_failure_np2
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/check_preflight_failfast.py
                --mpi ${MPIEXEC_EXECUTABLE} --exe $<TARGET_FILE:dnds_test_runtime_row_transfer>)
    set_tests_properties(dnds_runtime_preflight_off_failure_np2 PROPERTIES TIMEOUT 25
        PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_MCA_rmaps_base_oversubscribe=1")
endif()
add_test(NAME dnds_runtime_row_transfer_abort_np2
    COMMAND ${CMAKE_COMMAND}
        "-DTEST_COMMAND=${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};2;${MPIEXEC_PREFLAGS};$<TARGET_FILE:dnds_test_runtime_row_transfer>;--test-abort-transfer-conflict;${MPIEXEC_POSTFLAGS}"
        "-DEXPECTED_DIAGNOSTIC=transfer access/stream preflight failed on a rank"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/ExpectMPIFailure.cmake)
set_tests_properties(dnds_runtime_row_transfer_abort_np2 PROPERTIES
    TIMEOUT 30 PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1")
dnds_add_mpi_test(native_dof_algebra "DNDS/test_NativeDofAlgebra.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_native_dof_algebra)
dnds_add_mpi_test(runtime_transient "DNDS/test_RuntimeTransient.cpp" "dnds")
dnds_add_mpi_test(sparse_batch_exchange "DNDS/test_SparseBatchExchange.cpp" "dnds")
dnds_add_mpi_test(id_row_transfer "DNDS/test_IdRowTransfer.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_id_row_transfer)
add_dependencies(dnds_unit_tests dnds_test_sparse_batch_exchange)
add_dependencies(dnds_unit_tests dnds_test_runtime_transient)
dnds_add_mpi_test(runtime_array_io "DNDS/test_RuntimeArrayIO.cpp" "dnds")
dnds_add_mpi_test(runtime_native_io "DNDS/test_RuntimeNativeIO.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_runtime_native_io)
add_dependencies(dnds_unit_tests dnds_test_runtime_row_transfer dnds_test_runtime_array_io)
dnds_add_mpi_test(array_transformer "DNDS/test_ArrayTransformer.cpp" "dnds")
dnds_add_mpi_test(array_derived "DNDS/test_ArrayDerived.cpp" "dnds")
dnds_add_mpi_test(array_dof "DNDS/test_ArrayDOF.cpp" "dnds")
dnds_add_mpi_test(index_mapping "DNDS/test_IndexMapping.cpp" "dnds")
dnds_add_mpi_test(permutation_transfer "DNDS/test_PermutationTransfer.cpp" "dnds")

if(DNDS_USE_CUDA)
    add_executable(dnds_benchmark_fp64_matrix EXCLUDE_FROM_ALL DNDS/benchmark_FP64Matrix.cu)
    target_include_directories(dnds_benchmark_fp64_matrix PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(dnds_benchmark_fp64_matrix PRIVATE dnds ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_NCCL)
        add_executable(dnds_test_nccl_transfer EXCLUDE_FROM_ALL DNDS/test_NCCLTransfer.cu)
        target_include_directories(dnds_test_nccl_transfer PRIVATE ${DNDS_EXTERNAL_INCLUDE_DOCTEST} ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
        target_link_libraries(dnds_test_nccl_transfer PRIVATE dnds ${DNDS_EXTERNAL_LIBS})
        add_test(NAME dnds_nccl_transfer_np1 COMMAND $<TARGET_FILE:dnds_test_nccl_transfer>)
        add_test(NAME dnds_nccl_transfer_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py $<TARGET_FILE:dnds_test_nccl_transfer> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(dnds_nccl_transfer_np1 dnds_nccl_transfer_np2 PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1")
        add_dependencies(dnds_unit_tests dnds_test_nccl_transfer)
    endif()
    add_executable(dnds_test_device_packing EXCLUDE_FROM_ALL DNDS/test_DevicePacking.cu)
    target_include_directories(dnds_test_device_packing PRIVATE ${DNDS_EXTERNAL_INCLUDE_DOCTEST} ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(dnds_test_device_packing PRIVATE dnds ${DNDS_EXTERNAL_LIBS})
    add_test(NAME dnds_device_packing_np1 COMMAND $<TARGET_FILE:dnds_test_device_packing>)
    add_test(NAME dnds_device_packing_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py $<TARGET_FILE:dnds_test_device_packing> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_device_packing_np1 dnds_device_packing_np2 PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    add_dependencies(dnds_unit_tests dnds_test_device_packing)
    dnds_add_unit_test(runtime_cuda "DNDS/test_Runtime_CUDA.cu" "dnds")
    dnds_add_unit_test(native_dof_algebra_cuda "DNDS/test_NativeDofAlgebra_CUDA.cu" "dnds")
    add_dependencies(dnds_unit_tests dnds_test_native_dof_algebra_cuda)
    dnds_add_unit_test(runtime_cuda_engine "DNDS/test_CUDAExecution.cu" "dnds")
    dnds_add_unit_test(cauchy_point_physics_cuda "Cauchy/test_PointPhysics_CUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_point_physics_cuda)
    dnds_add_unit_test(cauchy_auxiliary_cuda "Cauchy/test_AuxiliaryCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_auxiliary_cuda)
    dnds_add_unit_test(cauchy_rhs_cuda "Cauchy/test_RHSCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_rhs_cuda)
    dnds_add_unit_test(cauchy_source_cuda "Cauchy/test_SourceCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_source_cuda)
    dnds_add_unit_test(cauchy_positivity_cuda "Cauchy/test_PositivityCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_positivity_cuda)
    dnds_add_unit_test(cauchy_gradient_cuda "Cauchy/test_GradientCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    dnds_add_unit_test(cauchy_linear_block_cuda "Cauchy/test_LinearBlockCUDA.cpp" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_linear_block_cuda)
    dnds_add_unit_test(cauchy_linear_inverse_cuda "Cauchy/test_LinearInverseCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    target_sources(dnds_test_cauchy_linear_inverse_cuda PRIVATE Cauchy/LinearInverseReference.cpp)
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_linear_inverse_cuda)
    add_executable(cauchy_benchmark_small_inverse_cuda EXCLUDE_FROM_ALL Cauchy/benchmark_SmallInverseCUDA.cu Cauchy/LinearInverseReference.cpp)
    target_include_directories(cauchy_benchmark_small_inverse_cuda PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cauchy_benchmark_small_inverse_cuda PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    add_executable(cauchy_benchmark_fv_jacobi_cuda EXCLUDE_FROM_ALL Cauchy/benchmark_FVJacobiCUDA.cu)
    target_include_directories(cauchy_benchmark_fv_jacobi_cuda PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cauchy_benchmark_fv_jacobi_cuda PRIVATE cauchy cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    dnds_add_unit_test(cauchy_linear_solver_cuda "Cauchy/test_LinearSolverCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_linear_solver_cuda)
    dnds_add_unit_test(cauchy_time_fas_cuda "Cauchy/test_TimeFASCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_time_fas_cuda)
    add_test(NAME dnds_cauchy_acm_interface_cuda
        COMMAND $<TARGET_FILE:dnds_test_cauchy_multidomain> "--test-case=Moving ACM disk*")
    set_tests_properties(dnds_cauchy_acm_interface_cuda PROPERTIES TIMEOUT 120
        ENVIRONMENT "DNDS_TEST_ACM_INTERFACE_CUDA=1;OMP_NUM_THREADS=1;DNDS_DIST_OMP_NUM_THREADS=1" LABELS "cuda;cauchy")
    dnds_add_unit_test(cauchy_domain_cuda "Cauchy/test_DomainCUDA.cpp" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_domain_cuda)
    add_test(NAME dnds_cauchy_domain_cuda_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py
            $<TARGET_FILE:dnds_test_cauchy_domain_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_cauchy_domain_cuda dnds_cauchy_domain_cuda_np2 PROPERTIES
        TIMEOUT 600 ENVIRONMENT "OMP_NUM_THREADS=1;DNDS_DIST_OMP_NUM_THREADS=1" LABELS "cuda;mpi")
    set_tests_properties(dnds_cauchy_time_fas_cuda PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1;DNDS_DIST_OMP_NUM_THREADS=1" LABELS "cuda")
    add_test(NAME dnds_cauchy_linear_solver_cuda_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:dnds_test_cauchy_linear_solver_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_cauchy_linear_solver_cuda dnds_cauchy_linear_solver_cuda_np2 PROPERTIES
        TIMEOUT 300 ENVIRONMENT "OMP_NUM_THREADS=1;DNDS_DIST_OMP_NUM_THREADS=1" LABELS "cuda;mpi")
    dnds_add_unit_test(cauchy_frozen_jacobian_cuda "Cauchy/test_FrozenJacobianCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_frozen_jacobian_cuda)
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_gradient_cuda)
    dnds_add_unit_test(cauchy_limited_vr_cuda "Cauchy/test_LimitedVRCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_limited_vr_cuda)
    dnds_add_unit_test(cauchy_rhs_face_packages_cuda "Cauchy/test_RHSFacePackagesCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_rhs_face_packages_cuda)
    dnds_add_unit_test(cauchy_characteristics_cuda "Cauchy/test_CharacteristicsCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_characteristics_cuda)
    dnds_add_unit_test(cauchy_state_cuda "Cauchy/test_StateCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_state_cuda)
    dnds_add_unit_test(cauchy_spatial_schedule_cuda "Cauchy/test_SpatialScheduleCUDA.cu" "cauchy;physics;cfv;geom;dnds")
    add_dependencies(cauchy_unit_tests dnds_test_cauchy_spatial_schedule_cuda)
    add_test(NAME dnds_cauchy_spatial_schedule_cuda_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py
        $<TARGET_FILE:dnds_test_cauchy_spatial_schedule_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dnds_cauchy_spatial_schedule_cuda_np2 PROPERTIES
        TIMEOUT 300 ENVIRONMENT "OMP_NUM_THREADS=1;DNDS_DIST_OMP_NUM_THREADS=1" LABELS "cuda;mpi")
    set_tests_properties(dnds_runtime_cuda_engine PROPERTIES TIMEOUT 60)
    add_dependencies(dnds_unit_tests dnds_test_runtime_cuda_engine)
    add_dependencies(dnds_unit_tests dnds_test_runtime_cuda)
    if(DNDS_USE_AMGX)
        dnds_add_unit_test(amgx_preconditioner "DNDS/test_AmgXPreconditioner.cpp" "dnds")
        add_dependencies(dnds_unit_tests dnds_test_amgx_preconditioner)
        add_executable(dnds_benchmark_amgx EXCLUDE_FROM_ALL DNDS/benchmark_AmgX.cpp)
        target_include_directories(dnds_benchmark_amgx PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
        target_link_libraries(dnds_benchmark_amgx PRIVATE dnds ${DNDS_EXTERNAL_LIBS})
        add_executable(dnds_benchmark_aggregation_amg EXCLUDE_FROM_ALL DNDS/benchmark_AggregationAMG.cpp)
        target_include_directories(dnds_benchmark_aggregation_amg PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
        target_link_libraries(dnds_benchmark_aggregation_amg PRIVATE dnds ${DNDS_EXTERNAL_LIBS})
    endif()
    dnds_add_mpi_test(array_dof_cuda "DNDS/test_ArrayDOF_CUDA.cu" "dnds")
    add_dependencies(dnds_unit_tests dnds_test_array_dof_cuda)
endif()

set(DNDS_TEST_TARGETS
    dnds_test_runtime
    dnds_test_runtime_exchange
    dnds_test_array
    dnds_test_config
    dnds_test_object_contracts
    dnds_test_serializer
    dnds_test_mpi
    dnds_test_array_transformer
    dnds_test_array_derived
    dnds_test_array_dof
    dnds_test_index_mapping
    dnds_test_permutation_transfer
)
foreach(T ${DNDS_TEST_TARGETS})
    add_dependencies(dnds_unit_tests ${T})
endforeach()

# ===================================================================
# Geom tests
# ===================================================================
geom_add_unit_test(elements "Geom/test_Elements.cpp" "geom;dnds")
geom_add_unit_test(bounding_box_tree "Geom/test_BoundingBoxTree.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_bounding_box_tree)
geom_add_unit_test(quadrature "Geom/test_Quadrature.cpp" "geom;dnds")
target_sources(geom_test_quadrature PRIVATE Geom/test_QuadratureCachePeer.cpp Geom/test_LocalPerformance.cpp)
geom_add_mpi_test(mesh_index_conversion "Geom/test_MeshIndexConversion.cpp" "geom;dnds")
geom_add_mpi_test(mesh_pipeline "Geom/test_MeshPipeline.cpp" "geom;dnds")
geom_add_mpi_test(boundary_interface "Geom/test_BoundaryInterface.cpp" "geom;dnds")
geom_add_mpi_test(surface_point_map "Geom/test_SurfacePointMap.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_surface_point_map)
add_executable(geom_benchmark_bounding_box_tree EXCLUDE_FROM_ALL Geom/benchmark_BoundingBoxTree.cpp)
target_include_directories(geom_benchmark_bounding_box_tree PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(geom_benchmark_bounding_box_tree PRIVATE geom dnds ${DNDS_EXTERNAL_LIBS})
geom_add_mpi_test(entity_work_partition "Geom/test_EntityWorkPartition.cpp" "geom;dnds")
geom_add_mpi_test(mesh_locality_ordering "Geom/test_MeshLocalityOrdering.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_locality_ordering)
geom_add_mpi_test(distributed_boundary_interface "Geom/test_DistributedBoundaryInterface.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_distributed_boundary_interface)
add_dependencies(geom_unit_tests geom_test_entity_work_partition)
add_dependencies(geom_unit_tests geom_test_boundary_interface)
geom_add_mpi_test(mesh_distributed_read "Geom/test_MeshDistributedRead.cpp" "geom;dnds")
geom_add_mpi_test(mesh_connectivity "Geom/test_MeshConnectivity.cpp" "geom;dnds")
geom_add_mpi_test(mesh_connectivity_ghost "Geom/test_MeshConnectivity_Ghost.cpp" "geom;dnds")
geom_add_mpi_test(mesh_connectivity_interpolate "Geom/test_MeshConnectivity_Interpolate.cpp" "geom;dnds")
geom_add_mpi_test(mesh_reorder "Geom/test_MeshReorder.cpp" "geom;dnds")
add_custom_target(geom_benchmark_topology_np2
    COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
            DNDS_TEST_TOPOLOGY_REPETITIONS=3
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:geom_test_mesh_pipeline> ${MPIEXEC_POSTFLAGS}
            "--test-case=Native inverse compose and face topology match independent incidences"
    DEPENDS geom_test_mesh_pipeline
    USES_TERMINAL)
geom_add_unit_test(mesh_cgns_multizone "Geom/test_MeshCGNSMultiZone.cpp" "geom;dnds")
set_property(TEST geom_mesh_cgns_multizone APPEND PROPERTY ENVIRONMENT "DNDS_EXTERNAL_LIB_CGNS=${DNDS_EXTERNAL_LIB_CGNS}")
geom_add_mpi_test(mesh_audit "Geom/test_MeshAudit.cpp" "geom;dnds")
geom_add_mpi_test(mesh_facade "Geom/test_MeshFacade.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_facade)
geom_add_mpi_test(mesh_facade_pipeline "Geom/test_MeshFacadePipeline.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_facade_pipeline)
geom_add_mpi_test(mesh_facade_overlap "Geom/test_MeshFacadeOverlap.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_facade_overlap)
geom_add_mpi_test(relation_access "Geom/test_RelationAccess.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_relation_access)
geom_add_mpi_test(native_relation_conversion "Geom/test_NativeRelationConversion.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_relation_conversion)
geom_add_mpi_test(native_connectivity "Geom/test_NativeConnectivity.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_connectivity)
geom_add_mpi_test(native_mesh_scenarios "Geom/test_NativeMeshScenarios.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_scenarios)
geom_add_mpi_test(native_mesh_snapshot "Geom/test_NativeMeshSnapshot.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_snapshot)
geom_add_mpi_test(native_mesh_publication "Geom/test_NativeMeshPublication.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_publication)
geom_add_mpi_test(native_mesh_construction "Geom/test_NativeMeshConstruction.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_construction)
geom_add_mpi_test(native_mesh_setup "Geom/test_NativeMeshSetup.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_setup)
geom_add_mpi_test(native_mesh_builder "Geom/test_NativeMeshBuilder.cpp" "geom;dnds")
geom_add_mpi_test(mesh_preflight "Geom/test_MeshPreflight.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_preflight)
add_dependencies(geom_unit_tests geom_test_native_mesh_builder)
geom_add_mpi_test(native_relation_registry "Geom/test_NativeRelationRegistry.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_relation_registry)

geom_add_mpi_test(native_mesh_transition "Geom/test_NativeMeshTransition.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_transition)
geom_add_mpi_test(native_transition_rebuild "Geom/test_NativeTransitionRebuild.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_transition_rebuild)
geom_add_mpi_test(native_mesh_archive "Geom/test_NativeMeshArchive.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_archive)
geom_add_mpi_test(native_mesh_cgns "Geom/test_NativeMeshCGNS.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_cgns)
geom_add_mpi_test(native_mesh_topology "Geom/test_NativeMeshTopology.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_topology)
geom_add_mpi_test(native_mesh_openfoam "Geom/test_NativeMeshOpenFOAM.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_openfoam)
geom_add_mpi_test(native_mesh_legacy_archive "Geom/test_NativeMeshLegacyArchive.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_legacy_archive)
geom_add_mpi_test(native_mesh_preparation "Geom/test_NativeMeshPreparation.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_preparation)
geom_add_mpi_test(native_mesh_partition_graph "Geom/test_NativeMeshPartitionGraph.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_partition_graph)
geom_add_mpi_test(native_mesh_wall_dist "Geom/test_NativeMeshWallDist.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_wall_dist)
geom_add_mpi_test(native_mesh_caches "Geom/test_NativeMeshCaches.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_caches)
geom_add_mpi_test(native_mesh_local_reorder "Geom/test_NativeMeshLocalReorder.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_local_reorder)
geom_add_mpi_test(native_mesh_serial_output "Geom/test_NativeMeshSerialOutput.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_serial_output)
geom_add_mpi_test(native_mesh_refinement "Geom/test_NativeMeshRefinement.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_refinement)
geom_add_mpi_test(native_mesh_extraction "Geom/test_NativeMeshExtraction.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_extraction)
geom_add_mpi_test(native_mesh_smoothing "Geom/test_NativeMeshSmoothing.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_smoothing)
geom_add_mpi_test(native_mesh_periodic "Geom/test_NativeMeshPeriodic.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_periodic)
geom_add_mpi_test(native_mesh_periodic_import "Geom/test_NativeMeshPeriodicImport.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_periodic_import)
geom_add_mpi_test(mesh_edit "Geom/test_MeshEdit.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_edit)
geom_add_mpi_test(mesh_transition "Geom/test_MeshTransition.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_transition)
geom_add_mpi_test(mesh_mutation "Geom/test_MeshMutation.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_mutation)
geom_add_mpi_test(mesh_geometry_edit "Geom/test_MeshGeometryEdit.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_geometry_edit)
geom_add_mpi_test(mesh_current_output "Geom/test_MeshCurrentOutput.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_current_output)
geom_add_mpi_test(mesh_field_checkpoint "Geom/test_MeshFieldCheckpoint.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_field_checkpoint)
geom_add_mpi_test(mesh_entity_field "Geom/test_MeshEntityField.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_entity_field)
if(DNDS_USE_CUDA)
    geom_add_unit_test(mesh_facade_cuda "Geom/test_MeshFacade_CUDA.cu" "geom;dnds")
    target_sources(geom_test_mesh_facade_cuda PRIVATE Geom/MeshViewKernel.cu)
    add_dependencies(geom_unit_tests geom_test_mesh_facade_cuda)
    add_executable(geom_test_mesh_facade_overlap_cuda EXCLUDE_FROM_ALL Geom/test_MeshFacadeOverlap_CUDA.cu)
    target_include_directories(geom_test_mesh_facade_overlap_cuda PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST} ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(geom_test_mesh_facade_overlap_cuda PRIVATE geom dnds ${DNDS_EXTERNAL_LIBS})
    target_sources(geom_test_mesh_facade_overlap_cuda PRIVATE Geom/MeshFacadeOverlapKernel.cu)
    add_test(NAME geom_mesh_facade_overlap_cuda_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py
            $<TARGET_FILE:geom_test_mesh_facade_overlap_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(geom_mesh_facade_overlap_cuda_np2 PROPERTIES
        PROCESSORS 2 TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    add_dependencies(geom_unit_tests geom_test_mesh_facade_overlap_cuda)
    add_executable(geom_test_mesh_entity_field_cuda EXCLUDE_FROM_ALL Geom/test_MeshEntityField_CUDA.cu)
    target_include_directories(geom_test_mesh_entity_field_cuda PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST} ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(geom_test_mesh_entity_field_cuda PRIVATE geom dnds ${DNDS_EXTERNAL_LIBS})
    add_test(NAME geom_mesh_entity_field_cuda_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py
            $<TARGET_FILE:geom_test_mesh_entity_field_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(geom_mesh_entity_field_cuda_np2 PROPERTIES
        PROCESSORS 2 TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    add_dependencies(geom_unit_tests geom_test_mesh_entity_field_cuda)
endif()

set(GEOM_TEST_TARGETS
    geom_test_elements
    geom_test_quadrature
    geom_test_mesh_index_conversion
    geom_test_mesh_pipeline
    geom_test_mesh_distributed_read
    geom_test_mesh_connectivity
    geom_test_mesh_connectivity_ghost
    geom_test_mesh_connectivity_interpolate
    geom_test_mesh_reorder
    geom_test_mesh_cgns_multizone
    geom_test_mesh_audit
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
    set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
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
    _dnds_maybe_set_timeout(cfv_${TEST_NAME} ${DNDS_TEST_TIMEOUT})
    set_tests_properties(cfv_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
endfunction()

# --- helper: add an MPI unit test executable (CFV) ---
# MPI tests run all TEST_CASEs together per NP value (avoids mpirun overhead per test case).
function(cfv_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "cfv_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
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
    foreach(NP ${DNDS_TEST_NP_LIST})
        _dnds_np_timeout(${NP} _TIMEOUT)
        add_test(NAME cfv_${TEST_NAME}_np${NP}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS} ${ARGN})
        _dnds_maybe_set_timeout(cfv_${TEST_NAME}_np${NP} ${_TIMEOUT})
        set_tests_properties(cfv_${TEST_NAME}_np${NP} PROPERTIES
            PROCESSORS ${NP}
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    endforeach()
endfunction()

add_custom_target(cfv_unit_tests)
if(DNDS_USE_CUDA)
    add_dependencies(cfv_unit_tests cfv_test_distributed_interface_sampling_cuda)
endif()
add_dependencies(all_unit_tests cfv_unit_tests)

cfv_add_mpi_test(reconstruction "CFV/test_Reconstruction.cpp" "cfv;geom;dnds")
cfv_add_mpi_test(fv_facility "CFV/test_FVFacility.cpp" "cfv;geom;dnds")
cfv_add_mpi_test(distributed_interface_sampling "CFV/test_DistributedInterfaceSampling.cpp" "cfv;geom;dnds")
cfv_add_mpi_test(reconstruction_overlap "CFV/test_ReconstructionOverlap.cpp" "cfv;geom;dnds")
# Keep the rank-coverage lane single-threaded even when the runner uses a
# distributed OpenMP team for the rest of the suite.
foreach(NP ${DNDS_TEST_NP_LIST})
    set_property(TEST cfv_reconstruction_overlap_np${NP} APPEND PROPERTY
        ENVIRONMENT "DNDS_DIST_OMP_NUM_THREADS=1")
endforeach()
if(DNDS_USE_OMP AND DNDS_DIST_MT_USE_OMP)
    add_test(NAME cfv_reconstruction_overlap_omp2_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:cfv_test_reconstruction_overlap> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cfv_reconstruction_overlap_omp2_np2 PROPERTIES
        TIMEOUT 180 PROCESSORS 4
        ENVIRONMENT "OMP_NUM_THREADS=2;DNDS_DIST_OMP_NUM_THREADS=2")
endif()
cfv_add_mpi_test(gradient_plan "CFV/test_GradientPlan.cpp" "cfv;geom;dnds")
cfv_add_mpi_test(limited_vr "CFV/test_LimitedVR.cpp" "cfv;geom;dnds")
cfv_add_mpi_test(fv_completion "CFV/test_FVCompletion.cpp" "cfv;geom;dnds")
cfv_add_mpi_test(reconstruction_capabilities "CFV/test_ReconstructionCapabilities.cpp" "cfv;geom;dnds")
add_dependencies(cfv_unit_tests cfv_test_reconstruction_capabilities)
add_dependencies(cfv_unit_tests cfv_test_fv_completion)
add_dependencies(cfv_unit_tests cfv_test_gradient_plan)
add_dependencies(cfv_unit_tests cfv_test_limited_vr)
add_dependencies(cfv_unit_tests cfv_test_reconstruction_overlap)
add_dependencies(cfv_unit_tests cfv_test_distributed_interface_sampling)
add_executable(cfv_benchmark_fv_facility EXCLUDE_FROM_ALL CFV/benchmark_FVFacility.cpp)
target_include_directories(cfv_benchmark_fv_facility PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cfv_benchmark_fv_facility PRIVATE cfv geom dnds ${DNDS_EXTERNAL_LIBS})
add_executable(cfv_benchmark_reconstruction_storage EXCLUDE_FROM_ALL CFV/benchmark_ReconstructionStorage.cpp)
target_include_directories(cfv_benchmark_reconstruction_storage PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cfv_benchmark_reconstruction_storage PRIVATE cfv geom dnds ${DNDS_EXTERNAL_LIBS})
add_executable(cfv_benchmark_coefficient_publication EXCLUDE_FROM_ALL CFV/benchmark_CoefficientPublication.cpp)
target_include_directories(cfv_benchmark_coefficient_publication PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cfv_benchmark_coefficient_publication PRIVATE cfv geom dnds ${DNDS_EXTERNAL_LIBS})
cfv_add_unit_test(limiters "CFV/test_Limiters.cpp" "cfv;geom;dnds")
cfv_add_unit_test(reconstruction_math "CFV/test_ReconstructionMath.cpp" "dnds")
add_dependencies(cfv_unit_tests cfv_test_reconstruction_math)
cfv_add_mpi_test(reconstruction3d "CFV/test_Reconstruction3D.cpp" "cfv;geom;dnds" "--test-suite=integration")
# The same binary has a generated 64-cell contract fixture. Keep this routine
# registration separate from the unchanged 32^3 numerical integration fixture.
foreach(NP ${DNDS_TEST_NP_LIST})
    add_test(NAME cfv_reconstruction3d_contract_np${NP}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cfv_test_reconstruction3d>
                ${MPIEXEC_POSTFLAGS} --test-suite-exclude=integration)
    _dnds_np_timeout(${NP} _TIMEOUT)
    _dnds_maybe_set_timeout(cfv_reconstruction3d_contract_np${NP} ${_TIMEOUT})
    set_tests_properties(cfv_reconstruction3d_contract_np${NP} PROPERTIES
        PROCESSORS ${NP} ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
endforeach()


set(CFV_TEST_TARGETS
    cfv_test_fv_facility
    cfv_test_reconstruction
    cfv_test_limiters
    cfv_test_reconstruction3d
)

# --- CUDA CFV tests ---
if(DNDS_USE_CUDA)
    # helper: add a CUDA MPI test executable for CFV
    function(cfv_add_cuda_mpi_test TEST_NAME SRC_FILE LIBS)
        set(TARGET_NAME "cfv_test_${TEST_NAME}")
        add_executable(${TARGET_NAME} ${SRC_FILE})
        set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
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
        _dnds_maybe_set_timeout(cfv_${TEST_NAME}_np1 ${DNDS_TEST_TIMEOUT})
        set_tests_properties(cfv_${TEST_NAME}_np1 PROPERTIES
            PROCESSORS 1
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    endfunction()

    add_executable(cfv_benchmark_reconstruction_cuda EXCLUDE_FROM_ALL CFV/benchmark_ReconstructionCUDA.cpp)
    add_executable(cfv_benchmark_reconstruction_sweep EXCLUDE_FROM_ALL CFV/benchmark_ReconstructionSweep.cpp)
    target_include_directories(cfv_benchmark_reconstruction_sweep PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(cfv_benchmark_reconstruction_sweep PRIVATE cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_AMGX)
        add_executable(cfv_benchmark_reconstruction_amg EXCLUDE_FROM_ALL CFV/benchmark_ReconstructionAMG.cpp)
        target_include_directories(cfv_benchmark_reconstruction_amg PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
        target_link_libraries(cfv_benchmark_reconstruction_amg PRIVATE cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    endif()
    target_include_directories(cfv_benchmark_reconstruction_cuda PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cfv_benchmark_reconstruction_cuda PRIVATE cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    cfv_add_cuda_mpi_test(reconstruction_cuda "CFV/test_ReconstructionCUDA.cu" "cfv;geom;dnds")
    if(DNDS_USE_AMGX)
        # The provider is an implementation-private DNDS dependency. Its macro
        # is not transitive; explicitly enable the downstream VR AMG test body.
        target_compile_definitions(cfv_test_reconstruction_cuda PRIVATE DNDS_USE_AMGX)
    endif()
    cfv_add_cuda_mpi_test(limited_vr_cuda "CFV/test_LimitedVRCUDA.cu" "cfv;geom;dnds")
    cfv_add_cuda_mpi_test(characteristic_cuda "CFV/test_CharacteristicCUDA.cu" "cfv;geom;dnds")
    add_test(NAME cfv_limited_vr_cuda_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py
            $<TARGET_FILE:cfv_test_limited_vr_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cfv_limited_vr_cuda_np2 PROPERTIES PROCESSORS 2 TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1")
    cfv_add_cuda_mpi_test(device_transferable "CFV/test_DeviceTransferable.cu" "cfv;geom;dnds")
    add_test(NAME cfv_reconstruction_cuda_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py
            $<TARGET_FILE:cfv_test_reconstruction_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cfv_reconstruction_cuda_np2 PROPERTIES PROCESSORS 2 TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1")
    # The default executable already compares every supported sweep kernel.
    # Repeat only provider-sensitive cases for NCCL.
    if(DNDS_USE_NCCL)
        foreach(_np 1 2)
            add_test(NAME cfv_reconstruction_nccl_np${_np}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
                    ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py
                    $<TARGET_FILE:cfv_test_reconstruction_cuda> ${MPIEXEC_POSTFLAGS} --test-suite=halo)
            set_tests_properties(cfv_reconstruction_nccl_np${_np} PROPERTIES PROCESSORS ${_np}
                TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1;DNDS_TEST_TRANSFER_PROVIDER=nccl")
        endforeach()
    endif()
    list(APPEND CFV_TEST_TARGETS cfv_test_device_transferable cfv_test_reconstruction_cuda
        cfv_test_limited_vr_cuda cfv_test_characteristic_cuda)
endif()

foreach(T ${CFV_TEST_TARGETS})
    add_dependencies(cfv_unit_tests ${T})
endforeach()

# ===================================================================
# Euler tests
# ===================================================================

# --- helper: add a serial unit test executable (Euler) ---
# Gas.hpp functions are header-only; link cfv;geom;dnds for types.
function(euler_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "euler_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
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
    add_test(NAME euler_${TEST_NAME} COMMAND ${TARGET_NAME})
    _dnds_maybe_set_timeout(euler_${TEST_NAME} ${DNDS_TEST_TIMEOUT})
    set_tests_properties(euler_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS};DNDS_MECH_PATH=${DNDS_CANTERA_DATA_DIR};CANTERA_DATA=${DNDS_CANTERA_DATA_DIR}")
endfunction()

# --- helper: add an MPI unit test executable (Euler) ---
# MPI tests run all TEST_CASEs together per NP value (avoids mpirun overhead per test case).
function(euler_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "euler_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
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
    foreach(NP ${DNDS_TEST_NP_LIST})
        _dnds_np_timeout(${NP} _TIMEOUT)
        add_test(NAME euler_${TEST_NAME}_np${NP}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
        _dnds_maybe_set_timeout(euler_${TEST_NAME}_np${NP} ${_TIMEOUT})
        set_tests_properties(euler_${TEST_NAME}_np${NP} PROPERTIES
            PROCESSORS ${NP}
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS};DNDS_MECH_PATH=${DNDS_CANTERA_DATA_DIR};CANTERA_DATA=${DNDS_CANTERA_DATA_DIR}")
    endforeach()
endfunction()

add_custom_target(euler_unit_tests)
add_dependencies(all_unit_tests euler_unit_tests)

euler_add_unit_test(gas_thermo "Euler/test_GasThermo.cpp" "cfv;geom;dnds")
euler_add_unit_test(riemann_solvers "Euler/test_RiemannSolvers.cpp" "cfv;geom;dnds")
euler_add_unit_test(rans "Euler/test_RANS.cpp" "cfv;geom;dnds")
euler_add_unit_test(rans_model "Physics/test_RANSModel.cpp" "physics;dnds")
add_dependencies(euler_unit_tests euler_test_rans_model)
euler_add_unit_test(rotating_frame "Euler/test_RotatingFrame.cpp" "cfv;geom;dnds")
euler_add_unit_test(limited_vr "Euler/test_LimitedVR.cpp" "cfv;geom;dnds")
if(DNDS_USE_CANTERA)
    euler_add_unit_test(shared_physics "Physics/test_Properties.cpp" "physics;cfv;geom;dnds")
    # Keep the historical target/CTest name for existing test runners; the
    # unchanged physical backend fixtures now use the canonical shared module.
    euler_add_unit_test(source_chemical "Physics/test_ChemicalSource.cpp" "physics;cfv;geom;dnds")
    euler_add_unit_test(reactive_split_indicator "Euler/test_ReactiveSplitIndicator.cpp" "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds")
    euler_add_unit_test(uv "Euler/test_UV.cpp" "euler_library_chemical;cfv;geom;dnds")
    euler_add_unit_test(physics_properties "Euler/test_PhysicsProperties.cpp" "euler_library_chemical;cfv;geom;dnds")
    euler_add_unit_test(chem_ode "Euler/test_ChemODE.cpp" "euler_library_chemical;cfv;geom;dnds")
    target_compile_definitions(euler_test_chem_ode PRIVATE CT_USE_SYSTEM_FMT=1)
    euler_add_mpi_test(evaluator_reactive "Euler/test_EulerEvaluatorReactive.cpp"
        "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds")
endif()

# Common standalone migration probe, also built against pinned legacy sources.
# It is opt-in: regular unit tests do not require a second solver source tree.
add_executable(euler_migration_probe EXCLUDE_FROM_ALL Euler/MigrationParityProbe.cpp)
add_executable(cauchy_benchmark_rhs_comparison EXCLUDE_FROM_ALL Cauchy/benchmark_RHSComparison.cpp)
target_include_directories(cauchy_benchmark_rhs_comparison PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_benchmark_rhs_comparison PRIVATE cauchy euler_library_fast_NS cfv geom dnds ${DNDS_EXTERNAL_LIBS})
add_executable(cauchy_benchmark_spatial_sa EXCLUDE_FROM_ALL Cauchy/benchmark_SpatialSA.cpp)
target_include_directories(cauchy_benchmark_spatial_sa PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(cauchy_benchmark_spatial_sa PRIVATE cauchy euler_library_fast_NS_SA cfv geom dnds ${DNDS_EXTERNAL_LIBS})
if(Python_EXECUTABLE)
    # Opt-in extracted operators, not another production implementation. The
    # source generator fails if the kernel anchors change and records its hash.
    set(_operator_diagnostic_header "${CMAKE_CURRENT_BINARY_DIR}/operator-diagnostic/ExtractedEulerSA.hpp")
    add_custom_command(OUTPUT "${_operator_diagnostic_header}"
        COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/Cauchy/extract_operator_diagnostic.py"
            "${PROJECT_SOURCE_DIR}/src/Euler/EulerEvaluator_EvaluateRHS.hxx" "${_operator_diagnostic_header}"
        DEPENDS Cauchy/extract_operator_diagnostic.py "${PROJECT_SOURCE_DIR}/src/Euler/EulerEvaluator_EvaluateRHS.hxx"
        VERBATIM)
    add_executable(cauchy_diagnostic_operator_phases EXCLUDE_FROM_ALL
        Cauchy/diagnostic_OperatorPhases.cpp "${_operator_diagnostic_header}")
    target_include_directories(cauchy_diagnostic_operator_phases PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/operator-diagnostic" ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cauchy_diagnostic_operator_phases PRIVATE
        cauchy euler_library_fast_NS_SA cfv geom dnds ${DNDS_EXTERNAL_LIBS})
    set(_driver_diagnostic_header "${CMAKE_CURRENT_BINARY_DIR}/operator-diagnostic/ExtractedEulerDriver.hpp")
    add_custom_command(OUTPUT "${_driver_diagnostic_header}"
        COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/Cauchy/extract_driver_diagnostic.py"
            "${PROJECT_SOURCE_DIR}/src/Euler/EulerSolver.hxx" "${_driver_diagnostic_header}"
        DEPENDS Cauchy/extract_driver_diagnostic.py "${PROJECT_SOURCE_DIR}/src/Euler/EulerSolver.hxx"
        VERBATIM)
    add_executable(cauchy_diagnostic_driver_phases EXCLUDE_FROM_ALL
        Cauchy/diagnostic_DriverPhases.cpp "${_driver_diagnostic_header}")
    set(_nonlinear_diagnostic_header "${CMAKE_CURRENT_BINARY_DIR}/operator-diagnostic/ExtractedEquationPipeline.cpp")
    add_custom_command(OUTPUT "${_nonlinear_diagnostic_header}"
        COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/Cauchy/extract_pipeline_diagnostic.py"
            "${PROJECT_SOURCE_DIR}/src/Cauchy/Discretization/FiniteVolume_Evaluation.cpp" "${_nonlinear_diagnostic_header}"
            --extra-source "${PROJECT_SOURCE_DIR}/src/Cauchy/Discretization/FiniteVolume_RHSHostQueued.cpp"
            --extra-source "${PROJECT_SOURCE_DIR}/src/Cauchy/Discretization/FiniteVolume_SourceQueued.cpp"
            --extra-source "${PROJECT_SOURCE_DIR}/src/Cauchy/Discretization/FiniteVolume_SourceHostQueued.cpp"
        DEPENDS Cauchy/extract_pipeline_diagnostic.py
            "${PROJECT_SOURCE_DIR}/src/Cauchy/Discretization/FiniteVolume_Evaluation.cpp"
            "${PROJECT_SOURCE_DIR}/src/Cauchy/Discretization/FiniteVolume_RHSHostQueued.cpp"
            "${PROJECT_SOURCE_DIR}/src/Cauchy/Discretization/FiniteVolume_SourceQueued.cpp"
            "${PROJECT_SOURCE_DIR}/src/Cauchy/Discretization/FiniteVolume_SourceHostQueued.cpp"
        VERBATIM)
    target_sources(cauchy_diagnostic_driver_phases PRIVATE "${_nonlinear_diagnostic_header}")
    set_source_files_properties("${_nonlinear_diagnostic_header}" PROPERTIES HEADER_FILE_ONLY TRUE)
    set(_nonlinear_diagnostic "${CMAKE_CURRENT_BINARY_DIR}/operator-diagnostic/ExtractedNonlinear.cpp")
    add_custom_command(OUTPUT "${_nonlinear_diagnostic}"
        COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/Cauchy/extract_nonlinear_diagnostic.py"
            "${PROJECT_SOURCE_DIR}/src/Cauchy/Implicit/NonlinearSolver.cpp" "${_nonlinear_diagnostic}"
        DEPENDS Cauchy/extract_nonlinear_diagnostic.py "${PROJECT_SOURCE_DIR}/src/Cauchy/Implicit/NonlinearSolver.cpp"
        VERBATIM)
    target_sources(cauchy_diagnostic_driver_phases PRIVATE "${_nonlinear_diagnostic}")
    set_source_files_properties("${_nonlinear_diagnostic}" PROPERTIES HEADER_FILE_ONLY FALSE)
    set(_jacobian_diagnostic "${CMAKE_CURRENT_BINARY_DIR}/operator-diagnostic/ExtractedJacobian.cpp")
    add_custom_command(OUTPUT "${_jacobian_diagnostic}"
        COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/Cauchy/extract_jacobian_diagnostic.py"
            "${PROJECT_SOURCE_DIR}/src/Cauchy/Implicit/FiniteVolume_Jacobian.cpp" "${_jacobian_diagnostic}"
        DEPENDS Cauchy/extract_jacobian_diagnostic.py "${PROJECT_SOURCE_DIR}/src/Cauchy/Implicit/FiniteVolume_Jacobian.cpp"
        VERBATIM)
    target_sources(cauchy_diagnostic_driver_phases PRIVATE "${_jacobian_diagnostic}")
    target_include_directories(cauchy_diagnostic_driver_phases PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/operator-diagnostic" "${CMAKE_CURRENT_SOURCE_DIR}/Cauchy" ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cauchy_diagnostic_driver_phases PRIVATE
        cauchy euler_library_fast_NS_SA cfv geom dnds ${DNDS_EXTERNAL_LIBS})
endif()
target_include_directories(euler_migration_probe PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(euler_migration_probe PRIVATE
    euler_library_NS euler_library_fast_NS euler_library_NS_SA euler_library_fast_NS_SA
    euler_library_NS_3D euler_library_fast_NS_3D cfv geom dnds ${DNDS_EXTERNAL_LIBS})

# Opt-in independent old/new probes; no second source tree in ordinary tests.
add_executable(euler_frozen_transport_probe EXCLUDE_FROM_ALL Euler/FrozenTransportProbe.cpp)
target_include_directories(euler_frozen_transport_probe PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(euler_frozen_transport_probe PRIVATE dnds ${DNDS_EXTERNAL_LIBS})
if(DNDS_USE_CANTERA)
    add_executable(euler_frozen_physics_probe EXCLUDE_FROM_ALL Euler/FrozenPhysicsProbe.cpp)
    target_include_directories(euler_frozen_physics_probe PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(euler_frozen_physics_probe PRIVATE euler_library_NS_EX euler_library_fast_NS_EX
        euler_library_NS_SA euler_library_fast_NS_SA cfv geom dnds ${DNDS_EXTERNAL_LIBS})
endif()

# EulerEvaluator pipeline test: full NS/SA/3D libraries plus the public
# two-equation face-flux probe.
euler_add_mpi_test(evaluator_pipeline "Euler/test_EulerEvaluator.cpp"
    "euler_library_NS;euler_library_fast_NS;euler_library_NS_SA;euler_library_fast_NS_SA;euler_library_NS_3D;euler_library_fast_NS_3D;euler_library_fast_NS_2EQ;cfv;geom;dnds")
foreach(NP ${DNDS_TEST_NP_LIST})
    if(TEST euler_evaluator_pipeline_np${NP})
        _dnds_maybe_set_timeout(euler_evaluator_pipeline_np${NP} ${DNDS_TEST_TIMEOUT_PIPELINE})
    endif()
endforeach()

set(EULER_TEST_TARGETS
    euler_test_gas_thermo
    euler_test_riemann_solvers
    euler_test_rans
    euler_test_rotating_frame
    euler_test_limited_vr
    euler_test_evaluator_pipeline
)
if(DNDS_USE_CANTERA)
    list(APPEND EULER_TEST_TARGETS
        euler_test_shared_physics
        euler_test_source_chemical
        euler_test_reactive_split_indicator
        euler_test_uv
        euler_test_physics_properties
        euler_test_chem_ode
        euler_test_evaluator_reactive)
endif()
foreach(T ${EULER_TEST_TARGETS})
    add_dependencies(euler_unit_tests ${T})
endforeach()

# ===================================================================
# Solver tests (ODE, Linear, Direct, Scalar)
# ===================================================================

add_custom_target(solver_unit_tests)
add_dependencies(all_unit_tests solver_unit_tests)

# Solver tests run in one process. Individual kernels may use OpenMP.
function(solver_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "solver_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS ${TARGET_NAME})
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
    add_test(NAME solver_${TEST_NAME} COMMAND ${TARGET_NAME})
    _dnds_maybe_set_timeout(solver_${TEST_NAME} ${DNDS_TEST_TIMEOUT_SOLVER})
    set_tests_properties(solver_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
endfunction()

solver_add_unit_test(aggregation_amg "Solver/test_AggregationAMG.cpp" "dnds")
set_property(TEST solver_aggregation_amg APPEND PROPERTY ENVIRONMENT
    "DNDS_DIST_OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
solver_add_unit_test(ode "Solver/test_ODE.cpp" "dnds")
solver_add_unit_test(linear "Solver/test_Linear.cpp" "dnds")
solver_add_unit_test(field_gmres "Solver/test_FieldGMRES.cpp" "dnds")
solver_add_unit_test(field_time_stepper "Solver/test_FieldTimeStepper.cpp" "dnds")
solver_add_unit_test(fas "Solver/test_FAS.cpp" "dnds")
solver_add_unit_test(direct "Solver/test_Direct.cpp" "dnds")
solver_add_unit_test(scalar "Solver/test_Scalar.cpp" "dnds")

set(SOLVER_TEST_TARGETS
    solver_test_aggregation_amg
    solver_test_ode
    solver_test_linear
    solver_test_field_gmres
    solver_test_field_time_stepper
    solver_test_fas
    solver_test_direct
    solver_test_scalar
)
foreach(T ${SOLVER_TEST_TARGETS})
    add_dependencies(solver_unit_tests ${T})
endforeach()

geom_add_mpi_test(native_mesh_assertions "Geom/test_NativeMeshAssertions.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_assertions)

# A fresh aggregate build must cover every registered helper-created executable.
include(${PROJECT_SOURCE_DIR}/cmake/DndsTestTargets.cmake)
dnds_check_registered_test_targets(all_unit_tests)

# Higher-rank repetitions of these application/operator suites remain runnable
# in the extended tier. Their routine np1/2 entries cover local and distributed
# semantics; distinct empty-rank/rank-subset/multidomain contracts keep np4/8.
# See docs/dev/plan53_validation/test-tiers.md for the per-suite rationale.
set(_dnds_extended_rank_suites
    cauchy_boundary_inputs cauchy_wall_distance cauchy_rans_fv
    cauchy_equation_source cauchy_viscous_flux cauchy_auxiliary_preparation
    cauchy_nonlinear cauchy_physical_timestep cauchy_implicit_domain
    cauchy_foundation cauchy_polynomial_multigrid)

# Labels separate module ownership from execution cost. No registration/rank is
# dropped here: large legacy mesh/golden fixtures remain explicitly runnable.
get_property(_dnds_cpp_tests DIRECTORY PROPERTY TESTS)
foreach(_dnds_test IN LISTS _dnds_cpp_tests)
    set(_dnds_module dnds)
    if(_dnds_test MATCHES "^(dnds_cauchy_|cauchy_)")
        set(_dnds_module cauchy)
    elseif(_dnds_test MATCHES "^(geom|cfv|euler|solver)_")
        set(_dnds_module ${CMAKE_MATCH_1})
    endif()
    set(_dnds_tier routine)
    if(_dnds_test MATCHES "^(cfv_reconstruction(3d)?|geom_mesh_pipeline|geom_mesh_index_conversion)_np")
        set(_dnds_tier integration)
    endif()
    if(_dnds_test MATCHES "^dnds_(cauchy_[a-z_]+)_np(4|8)$" AND CMAKE_MATCH_1 IN_LIST _dnds_extended_rank_suites)
        set(_dnds_tier extended)
    endif()
    set_property(TEST ${_dnds_test} APPEND PROPERTY LABELS "cpp;${_dnds_module};${_dnds_tier}")
    if(_dnds_test MATCHES "_np(4|8)$")
        set_property(TEST ${_dnds_test} APPEND PROPERTY LABELS "mpi-high")
    endif()
endforeach()

include(${PROJECT_SOURCE_DIR}/cmake/DndsCITestLevels.cmake)

if(DNDS_USE_CUDA)
    add_executable(cauchy_benchmark_rhs_face_mappings_cuda EXCLUDE_FROM_ALL Cauchy/benchmark_RHSFaceMappingsCUDA.cpp)
    target_include_directories(cauchy_benchmark_rhs_face_mappings_cuda PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(cauchy_benchmark_rhs_face_mappings_cuda PRIVATE cauchy physics cfv geom dnds CUDA::cudart ${DNDS_EXTERNAL_LIBS})
endif()

if(DNDS_USE_SYCL)
    # Keep offload flags local to the probe. The Runtime provider uses host
    # SYCL APIs, while only this test compiles kernels for both selected devices.
    set(DNDS_SYCL_TEST_FLAGS "-fsycl-targets=spir64,nvptx64-nvidia-cuda" CACHE STRING "Compile and link flags for the local SYCL migration probes")
    add_executable(geom_test_sycl_migration EXCLUDE_FROM_ALL Geom/test_SYCLMigration.cpp)
    target_compile_options(geom_test_sycl_migration PRIVATE -fsycl ${DNDS_SYCL_TEST_FLAGS})
    target_link_options(geom_test_sycl_migration PRIVATE -fsycl ${DNDS_SYCL_TEST_FLAGS})
    target_include_directories(geom_test_sycl_migration PRIVATE ${DNDS_EXTERNAL_INCLUDE_DOCTEST} ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(geom_test_sycl_migration PRIVATE geom dnds ${DNDS_EXTERNAL_LIBS})
    add_dependencies(geom_unit_tests geom_test_sycl_migration)
    foreach(backend cpu cuda)
        add_test(NAME geom_sycl_${backend}_migration_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                 $<TARGET_FILE:geom_test_sycl_migration> ${MPIEXEC_POSTFLAGS} ${backend})
        set_tests_properties(geom_sycl_${backend}_migration_np2 PROPERTIES LABELS "cpp;geom;sycl;level2" PROCESSORS 2 TIMEOUT 120)
    endforeach()
endif()

# Optional single-device linear preconditioner. DNDS owns all MPI exchange;
# this target supplies only the local AmgX provider and its CUDA dependencies.
option(DNDS_USE_AMGX "Enable the single-device AmgX preconditioner" OFF)
set(DNDS_AMGX_ROOT "${DNDS_CFD_EXTERNALS_INSTALL}" CACHE PATH "Install prefix from cfd_externals --amgx")

if(DNDS_USE_AMGX)
    if(NOT DNDS_USE_CUDA)
        message(FATAL_ERROR "DNDS_USE_AMGX requires DNDS_USE_CUDA")
    endif()
    find_path(DNDS_AMGX_INCLUDE_DIR amgx_c.h
        HINTS "${DNDS_AMGX_ROOT}" PATH_SUFFIXES include REQUIRED)
    find_library(DNDS_AMGX_LIBRARY NAMES amgxsh amgx
        HINTS "${DNDS_AMGX_ROOT}" PATH_SUFFIXES lib lib64 REQUIRED)
    # The pinned provider requires the modal/native-storage patch. Reject an old
    # install during configuration rather than falling back to scalar CSR.
    get_filename_component(_dnds_amgx_prefix "${DNDS_AMGX_INCLUDE_DIR}" DIRECTORY)
    set(_dnds_amgx_manifest "${_dnds_amgx_prefix}/share/cfd_externals/amgx-build.json")
    if(NOT EXISTS "${_dnds_amgx_manifest}")
        message(FATAL_ERROR "AmgX requires the pinned cfd_externals --amgx package with geometric aggregation support")
    endif()
    file(READ "${_dnds_amgx_manifest}" _dnds_amgx_provenance)
    file(READ "${PROJECT_SOURCE_DIR}/external/cfd_externals/dependency_versions.json" _dnds_provider_pins)
    string(JSON _dnds_amgx_expected_revision GET "${_dnds_provider_pins}" amgx revision)
    foreach(_api IN ITEMS allocation_diagnostics_api geometric_aggregation_api)
        string(JSON _value ERROR_VARIABLE _error GET "${_dnds_amgx_provenance}" "${_api}")
        if(_error OR NOT _value EQUAL 1)
            message(FATAL_ERROR "Packaged AmgX requires ${_api} version 1")
        endif()
    endforeach()
    string(JSON _dnds_amgx_revision ERROR_VARIABLE _dnds_amgx_error GET "${_dnds_amgx_provenance}" revision)
    if(_dnds_amgx_error OR NOT _dnds_amgx_revision STREQUAL _dnds_amgx_expected_revision)
        message(FATAL_ERROR "AmgX provider revision does not match the audited pin")
    endif()
    string(JSON _dnds_amgx_mpi ERROR_VARIABLE _dnds_amgx_error GET "${_dnds_amgx_provenance}" mpi)
    if(_dnds_amgx_error OR _dnds_amgx_mpi)
        message(FATAL_ERROR "AmgX must be built as a single-device provider without MPI")
    endif()
    string(JSON _dnds_amgx_native_api ERROR_VARIABLE _dnds_amgx_error GET "${_dnds_amgx_provenance}" native_storage_api)
    if(_dnds_amgx_error OR NOT _dnds_amgx_native_api EQUAL 1)
        message(FATAL_ERROR "AmgX native-storage API is absent; rebuild the Plan 61-6 provider")
    endif()
    get_filename_component(_dnds_amgx_library_name "${DNDS_AMGX_LIBRARY}" NAME)
    string(JSON _dnds_amgx_library_hash ERROR_VARIABLE _dnds_amgx_error GET
        "${_dnds_amgx_provenance}" libraries_sha256 "${_dnds_amgx_library_name}")
    file(SHA256 "${DNDS_AMGX_LIBRARY}" _dnds_amgx_actual_library_hash)
    if(_dnds_amgx_error OR NOT _dnds_amgx_library_hash STREQUAL _dnds_amgx_actual_library_hash)
        message(FATAL_ERROR "Selected AmgX library does not match its native-block build manifest")
    endif()
    find_package(Threads REQUIRED)
    add_library(DNDS::AmgX UNKNOWN IMPORTED)
    set_target_properties(DNDS::AmgX PROPERTIES
        IMPORTED_LOCATION "${DNDS_AMGX_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${DNDS_AMGX_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS DNDS_USE_AMGX
        INTERFACE_LINK_LIBRARIES "CUDA::cudart;CUDA::cusparse;CUDA::cublas;CUDA::cusolver;Threads::Threads;${CMAKE_DL_LIBS}")
    message(STATUS "Single-device AmgX: ${DNDS_AMGX_LIBRARY}")
endif()

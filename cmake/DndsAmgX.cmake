# Optional single-device linear preconditioner. DNDS owns all MPI exchange;
# this target supplies only the local AmgX provider and its CUDA dependencies.
option(DNDS_USE_AMGX "Enable the single-device AmgX preconditioner" OFF)
set(DNDS_AMGX_ROOT "" CACHE PATH "Install prefix from scripts/build_amgx.py")

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
    set(_dnds_amgx_manifest "${_dnds_amgx_prefix}/dndsr-amgx-build.json")
    if(NOT EXISTS "${_dnds_amgx_manifest}")
        message(FATAL_ERROR "AmgX requires the native-block build from scripts/build_amgx.py")
    endif()
    file(READ "${_dnds_amgx_manifest}" _dnds_amgx_provenance)
    string(JSON _dnds_amgx_patch ERROR_VARIABLE _dnds_amgx_error GET "${_dnds_amgx_provenance}" patch_sha256)
    file(SHA256 "${PROJECT_SOURCE_DIR}/scripts/patches/amgx-v2.5.0-native-blocks.patch" _dnds_amgx_expected_patch)
    if(_dnds_amgx_error OR NOT _dnds_amgx_patch STREQUAL _dnds_amgx_expected_patch)
        message(FATAL_ERROR "AmgX native-block patch differs or is absent; rebuild with scripts/build_amgx.py")
    endif()
    string(JSON _dnds_amgx_revision ERROR_VARIABLE _dnds_amgx_error GET "${_dnds_amgx_provenance}" revision)
    if(_dnds_amgx_error OR NOT _dnds_amgx_revision STREQUAL "cc1cebdbb32b14d33762d4ddabcb2e23c1669f47")
        message(FATAL_ERROR "AmgX provider revision does not match the audited pin")
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

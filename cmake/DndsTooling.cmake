# cmake/DndsTooling.cmake
# Developer tooling: compile_commands.json post-processing and stub generation.

# -------------------------------------------------------------------
# compile_commands.json (compdb post-processing)
# -------------------------------------------------------------------
if(DNDS_GENERATE_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Prefer the shipped (modified) compdb under scripts/compdb/.
    # Fall back to a system-installed compdb executable.
    set(DNDS_COMPDB_SCRIPTS_DIR "${PROJECT_SOURCE_DIR}/scripts")
    set(DNDS_COMPDB_FOUND OFF)

    if(EXISTS "${DNDS_COMPDB_SCRIPTS_DIR}/compdb/__main__.py")
        set(DNDS_COMPDB_FOUND ON)
        set(DNDS_COMPDB_COMMAND
            ${CMAKE_COMMAND} -E env "PYTHONPATH=${DNDS_COMPDB_SCRIPTS_DIR}"
            ${Python_EXECUTABLE} -m compdb)
        message(STATUS "Using shipped compdb: PYTHONPATH=${DNDS_COMPDB_SCRIPTS_DIR} python -m compdb")
    else()
        find_program(COMpdb_EXECUTABLE compdb)
        if(COMpdb_EXECUTABLE)
            set(DNDS_COMPDB_FOUND ON)
            set(DNDS_COMPDB_COMMAND ${COMpdb_EXECUTABLE})
            message(STATUS "Using system compdb: ${COMpdb_EXECUTABLE}")
        else()
            message(WARNING "compdb not found (neither scripts/compdb/ nor system compdb).")
        endif()
    endif()

    if(DNDS_COMPDB_FOUND AND CMAKE_EXPORT_COMPILE_COMMANDS)
        add_custom_target(process-compile-commands
            COMMAND ${DNDS_COMPDB_COMMAND} -p ${CMAKE_BINARY_DIR} list > ${CMAKE_BINARY_DIR}/compile_commands_processed.json
            DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
            COMMENT "Processing compile_commands.json with compdb"
            VERBATIM
        )
        
        # Optional: Create a symlink in source directory for IDE convenience
        if(CMAKE_HOST_UNIX)
            add_custom_command(TARGET process-compile-commands POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E create_symlink 
                    ${CMAKE_BINARY_DIR}/compile_commands_processed.json 
                    ${CMAKE_SOURCE_DIR}/compile_commands.json
                COMMENT "Creating symlink to processed compile_commands.json in source directory"
            )
        endif()
    endif()
endif()

# -------------------------------------------------------------------
# Stub generation — automatic post-install step
# -------------------------------------------------------------------
# Generates .pyi type stubs from the built pybind11 modules as the
# final step of `cmake --install <build> --component py`.
#
# Runs immediately after the .so files are placed into python/DNDSR/,
# so there is no separate manual step required.
#
# Requires: pybind11-stubgen installed in the Python environment.
# The DNDSR package is found via PYTHONPATH=python (no pip install).
#
# Workflow:
#   cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
#   cmake --install build --component py   # installs .so AND regenerates stubs
#
# To regenerate stubs manually (e.g. after editing only Python code):
#   PYTHONPATH=python ./scripts/generate-stubs.sh

get_filename_component(_PYTHON_BIN_DIR "${Python_EXECUTABLE}" DIRECTORY)

install(CODE "
    message(STATUS \"Generating .pyi type stubs from pybind11 modules...\")
    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E env
            \"PYTHONPATH=${PROJECT_SOURCE_DIR}/python\"
            \"PATH=${_PYTHON_BIN_DIR}:\$ENV{PATH}\"
            bash \"${PROJECT_SOURCE_DIR}/scripts/generate-stubs.sh\"
        WORKING_DIRECTORY \"${PROJECT_SOURCE_DIR}\"
        RESULT_VARIABLE _stubgen_result
    )
    if(NOT _stubgen_result EQUAL 0)
        message(WARNING \"Stub generation failed (exit \${_stubgen_result}). \"
            \"Stubs may be stale. Run manually: PYTHONPATH=python ./scripts/generate-stubs.sh\")
    else()
        message(STATUS \"Stub generation complete.\")
    endif()
" COMPONENT py)

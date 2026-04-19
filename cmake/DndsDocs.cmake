# cmake/DndsDocs.cmake
# Documentation targets: Doxygen (C++ HTML + XML) and Sphinx (user-facing site).
#
# Dependency graph (ninja tracks file timestamps):
#
#   src/**/*.hpp ──► doxygen_stamp ──► sphinx_stamp ──► [serve-docs]
#   docs/**/*.md ─────────────────────┘
#
# - Doxygen re-runs only when C++ source files change.
# - Sphinx re-runs only when docs or doxygen XML change.
# - After Sphinx builds, Doxygen HTML is copied into the Sphinx output
#   at doxygen/ so the entire site is served from one root.
# - "doxygen" and "sphinx" convenience targets always run (for manual use).

# ===================================================================
#  1. Doxygen — C++ HTML docs + XML for Breathe
# ===================================================================

find_package(Doxygen)
if(DOXYGEN_FOUND)
    message(STATUS "Doxygen found: ${DOXYGEN_EXECUTABLE}")

    set(DOXYGEN_IN  "${PROJECT_SOURCE_DIR}/docs/doxygen/Doxyfile")
    set(DOXYGEN_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
    set(DOXYGEN_XML_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs/xml")
    set(DOXYGEN_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs/html")
    set(DOXYGEN_STAMP "${CMAKE_CURRENT_BINARY_DIR}/docs/.doxygen_stamp")

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # Copy PDF/PNG/JPG attachments into the HTML output tree (at configure time).
    execute_process(
        COMMAND python "${PROJECT_SOURCE_DIR}/docs/getAllAttachForDox.py"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/docs")

    # Collect C++ source files as dependencies for doxygen.
    # CONFIGURE_DEPENDS re-globs at build time so new files are picked up.
    file(GLOB_RECURSE _DOXYGEN_DEPS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.hxx"
        "${PROJECT_SOURCE_DIR}/src/*.cxx"
    )

    # --- Stamp-based doxygen command (incremental) ---
    # Ninja re-runs this only when a C++ source file changes.
    add_custom_command(
        OUTPUT  "${DOXYGEN_STAMP}"
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMAND python3 "${PROJECT_SOURCE_DIR}/docs/clean_doxygen_xml.py"
                "${DOXYGEN_XML_DIR}"
        COMMAND ${CMAKE_COMMAND} -E touch "${DOXYGEN_STAMP}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${DOXYGEN_OUT} ${_DOXYGEN_DEPS}
        COMMENT "Doxygen: building HTML + XML (incremental)"
        VERBATIM
    )

    # --- "doxygen" convenience target (always runs) ---
    add_custom_target(doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMAND python3 "${PROJECT_SOURCE_DIR}/docs/clean_doxygen_xml.py"
                "${DOXYGEN_XML_DIR}"
        COMMAND ${CMAKE_COMMAND} -E touch "${DOXYGEN_STAMP}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Doxygen: building HTML + XML (forced)"
        VERBATIM
    )

    # --- serve-doxygen (standalone, without Sphinx) ---
    add_custom_target(serve-doxygen
        DEPENDS doxygen
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "  Doxygen docs built. To serve standalone, run:"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "    cd ${DOXYGEN_HTML_DIR} && python3 -m http.server 8001"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "  Or build the full site (includes Doxygen at /doxygen/):"
        COMMAND ${CMAKE_COMMAND} -E echo "    cmake --build build -t serve-docs"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMENT "Building Doxygen HTML"
        VERBATIM
    )
else()
    message(STATUS "Doxygen NOT found — doxygen/serve-doxygen targets disabled")
endif()

# ===================================================================
#  2. Sphinx — user-facing documentation (Breathe + autodoc)
# ===================================================================

find_program(SPHINX_BUILD sphinx-build
    HINTS "$ENV{VIRTUAL_ENV}/bin" "${Python_ROOT_DIR}/bin"
    DOC "Path to sphinx-build executable")

if(SPHINX_BUILD)
    message(STATUS "sphinx-build found: ${SPHINX_BUILD}")

    set(SPHINX_CONFDIR "${PROJECT_SOURCE_DIR}/docs/sphinx")
    set(SPHINX_SOURCE  "${PROJECT_SOURCE_DIR}/docs")
    set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx")
    set(SPHINX_STAMP "${CMAKE_CURRENT_BINARY_DIR}/docs/.sphinx_stamp")

    # Collect doc source files as dependencies for sphinx.
    file(GLOB_RECURSE _SPHINX_DEPS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/docs/*.md"
        "${PROJECT_SOURCE_DIR}/docs/*.rst"
        "${PROJECT_SOURCE_DIR}/docs/sphinx/conf.py"
        "${PROJECT_SOURCE_DIR}/docs/sphinx/_ext/*.py"
    )

    # --- Stamp-based sphinx command (incremental) ---
    # Depends on the doxygen stamp (if doxygen is available) so XML is
    # built first, but only when C++ sources actually change.
    #
    # NOTE: Do NOT use -j auto.  Breathe is not parallel-safe: concurrent
    # workers race over Doxygen-XML parse state, causing non-deterministic
    # hangs — especially on machines with many cores.  Use -j 1; it is
    # actually faster than -j auto for incremental builds.
    #
    # After Sphinx finishes, copy the Doxygen HTML tree into the Sphinx
    # output at doxygen/ so the entire site is served from one root.
    set(_SPHINX_XML_DEP "")
    if(DOXYGEN_FOUND)
        set(_SPHINX_XML_DEP "${DOXYGEN_STAMP}")
    endif()

    add_custom_command(
        OUTPUT  "${SPHINX_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E env
            "DOXYGEN_XML_DIR=${DOXYGEN_XML_DIR}"
            "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "DNDS_VERSION_FULL=${DNDS_VERSION_FULL}"
            ${SPHINX_BUILD}
                -b html
                -c "${SPHINX_CONFDIR}"
                "${SPHINX_SOURCE}"
                "${SPHINX_BUILD_DIR}"
                -j 1
        # Copy Doxygen HTML into the Sphinx output tree.
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${DOXYGEN_HTML_DIR}" "${SPHINX_BUILD_DIR}/doxygen"
        COMMAND ${CMAKE_COMMAND} -E touch "${SPHINX_STAMP}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        DEPENDS ${_SPHINX_XML_DEP} ${_SPHINX_DEPS}
        COMMENT "Sphinx: building HTML documentation (incremental)"
        VERBATIM
    )

    # --- "sphinx" convenience target (always runs) ---
    add_custom_target(sphinx
        COMMAND ${CMAKE_COMMAND} -E env
            "DOXYGEN_XML_DIR=${DOXYGEN_XML_DIR}"
            "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "DNDS_VERSION_FULL=${DNDS_VERSION_FULL}"
            ${SPHINX_BUILD}
                -b html
                -c "${SPHINX_CONFDIR}"
                "${SPHINX_SOURCE}"
                "${SPHINX_BUILD_DIR}"
                -j 1
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${DOXYGEN_HTML_DIR}" "${SPHINX_BUILD_DIR}/doxygen"
        COMMAND ${CMAKE_COMMAND} -E touch "${SPHINX_STAMP}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Sphinx: building HTML documentation (forced)"
        VERBATIM
    )
    if(DOXYGEN_FOUND)
        add_dependencies(sphinx doxygen)
    endif()

    # --- "docs" target: stamp-based (incremental by default) ---
    add_custom_target(docs DEPENDS "${SPHINX_STAMP}")

    # --- serve-docs (serves the unified site) ---
    add_custom_target(serve-docs
        DEPENDS "${SPHINX_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "  Documentation built. To serve, run:"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "    cd ${SPHINX_BUILD_DIR} && python3 -m http.server 8000"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "  Sphinx:  http://localhost:8000/"
        COMMAND ${CMAKE_COMMAND} -E echo "  Doxygen: http://localhost:8000/doxygen/"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMENT "Building unified documentation site"
        VERBATIM
    )
else()
    message(STATUS "sphinx-build NOT found — sphinx/docs/serve-docs targets disabled")
    message(STATUS "  Install with: pip install -r docs/sphinx/requirements.txt")

    # Fall back: if doxygen is available, make `docs` an alias for `doxygen`
    if(DOXYGEN_FOUND)
        add_custom_target(docs DEPENDS doxygen)
        add_custom_target(serve-docs
            DEPENDS doxygen
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "  Sphinx not found. Doxygen docs built. To serve, run:"
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "    cd ${DOXYGEN_HTML_DIR} && python3 -m http.server 8000"
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMENT "Sphinx not found — falling back to Doxygen HTML"
            VERBATIM
        )
    endif()
endif()

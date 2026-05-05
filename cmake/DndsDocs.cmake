# cmake/DndsDocs.cmake
# Documentation targets: Doxygen (C++ HTML + XML) and Sphinx (user-facing site).
#
# Dependency graph (ninja tracks file timestamps):
#
#   tools/elements/*.py ──► docs-assets ──┐
#   src/**/*.hpp ──► doxygen_stamp ──────►│── sphinx_stamp ──► [serve-docs]
#   docs/**/*.md ────────────────────────►│
#   docs/presentations/... ──► marp-slides ┘
#
# - Element diagrams are generated into ${CMAKE_BINARY_DIR}/docs_generated/elements/
#   and copied into both the Doxygen and Sphinx HTML output trees.
# - Doxygen re-runs only when C++ source files change.
# - Sphinx re-runs only when docs or doxygen XML change.
# - After Sphinx builds, Doxygen HTML is copied into the Sphinx output
#   at doxygen/ so the entire site is served from one root.
# - Marp slideshows are rendered into ${CMAKE_BINARY_DIR}/docs_generated/
#   presentations/ and staged into the Sphinx site via html_extra_path.

set(DOCS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs_generated")
set(DOCS_ELEMENTS_DIR  "${DOCS_GENERATED_DIR}/elements")
set(DOCS_PRESENTATIONS_DIR "${DOCS_GENERATED_DIR}/presentations")
set(DOCS_ASSETS_STAMP  "${CMAKE_CURRENT_BINARY_DIR}/docs/.docs_assets_stamp")
set(DOCS_ELEMENTS_STAMP "${CMAKE_CURRENT_BINARY_DIR}/docs/.docs_elements_stamp")
set(DOCS_SLIDES_STAMP  "${CMAKE_CURRENT_BINARY_DIR}/docs/.docs_slides_stamp")

# ===================================================================
#  0. Element diagrams — regenerate from tools/elements/gen_diagrams.py
# ===================================================================
#
# Outputs land in ${DOCS_ELEMENTS_DIR} (NOT in the source tree). Both
# Doxygen and Sphinx mount this directory via the asset-copy step.

find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_FOUND)
    file(GLOB_RECURSE _ELEMENT_TOOL_DEPS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/tools/elements/*.py"
    )

    add_custom_command(
        OUTPUT "${DOCS_ELEMENTS_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DOCS_ELEMENTS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${PROJECT_SOURCE_DIR}"
            ${Python3_EXECUTABLE} -m tools.elements.gen_diagrams
                --outdir "${DOCS_ELEMENTS_DIR}"
                --format png
        COMMAND ${CMAKE_COMMAND} -E touch "${DOCS_ELEMENTS_STAMP}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        DEPENDS ${_ELEMENT_TOOL_DEPS}
        COMMENT "Docs: generating element topology diagrams"
        VERBATIM
    )

    add_custom_target(docs-elements DEPENDS "${DOCS_ELEMENTS_STAMP}")
else()
    message(STATUS "Python3 NOT found — element-diagram generation disabled")
    add_custom_target(docs-elements
        COMMAND ${CMAKE_COMMAND} -E echo
            "Python3 not found; element diagrams will not be regenerated"
    )
    set(DOCS_ELEMENTS_STAMP "")
endif()

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

    # Collect C++ source files as dependencies for doxygen.
    # CONFIGURE_DEPENDS re-globs at build time so new files are picked up.
    file(GLOB_RECURSE _DOXYGEN_DEPS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.hxx"
        "${PROJECT_SOURCE_DIR}/src/*.cxx"
    )

    # Shared asset copier (structure-preserving). Populates the Doxygen
    # HTML output so co-located markdown image references resolve.
    set(_COPY_ASSETS "${PROJECT_SOURCE_DIR}/docs/_scripts/copy_assets.py")

    set(_DOX_ASSET_EXTRA_DEP "")
    if(DOCS_ELEMENTS_STAMP)
        list(APPEND _DOX_ASSET_EXTRA_DEP "${DOCS_ELEMENTS_STAMP}")
    endif()
    if(DOCS_SLIDES_STAMP)
        list(APPEND _DOX_ASSET_EXTRA_DEP "${DOCS_SLIDES_STAMP}")
    endif()

    # --- Stamp-based doxygen command (incremental) ---
    # Ninja re-runs this only when a C++ source file changes.
    add_custom_command(
        OUTPUT  "${DOXYGEN_STAMP}"
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMAND python3 "${PROJECT_SOURCE_DIR}/docs/clean_doxygen_xml.py"
                "${DOXYGEN_XML_DIR}"
        # Copy co-located PNG/PDF/etc. assets into the Doxygen HTML tree,
        # preserving subdirectory structure (e.g. docs/theory/*.png →
        # build/docs/html/theory/*.png). Then overlay generated element
        # diagrams at the same /elements/ subtree.
        COMMAND python3 "${_COPY_ASSETS}"
                --dest "${DOXYGEN_HTML_DIR}"
                --root "${PROJECT_SOURCE_DIR}/docs"
                --rel-to "${PROJECT_SOURCE_DIR}/docs"
        COMMAND python3 "${_COPY_ASSETS}"
                --dest "${DOXYGEN_HTML_DIR}/elements"
                --root "${DOCS_ELEMENTS_DIR}"
                --rel-to "${DOCS_ELEMENTS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E touch "${DOXYGEN_STAMP}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${DOXYGEN_OUT} ${_DOXYGEN_DEPS} ${_DOX_ASSET_EXTRA_DEP}
        COMMENT "Doxygen: building HTML + XML + copying assets (incremental)"
        VERBATIM
    )

    # --- "doxygen" convenience target (always runs) ---
    add_custom_target(doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMAND python3 "${PROJECT_SOURCE_DIR}/docs/clean_doxygen_xml.py"
                "${DOXYGEN_XML_DIR}"
        COMMAND python3 "${_COPY_ASSETS}"
                --dest "${DOXYGEN_HTML_DIR}"
                --root "${PROJECT_SOURCE_DIR}/docs"
                --rel-to "${PROJECT_SOURCE_DIR}/docs"
        COMMAND python3 "${_COPY_ASSETS}"
                --dest "${DOXYGEN_HTML_DIR}/elements"
                --root "${DOCS_ELEMENTS_DIR}"
                --rel-to "${DOCS_ELEMENTS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E touch "${DOXYGEN_STAMP}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Doxygen: building HTML + XML + copying assets (forced)"
        VERBATIM
    )
    if(DOCS_ELEMENTS_STAMP)
        add_dependencies(doxygen docs-elements)
    endif()

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
#  2. Marp slideshows — render any presentation decks into docs_generated/
# ===================================================================
#
# Each deck is expected to ship its own build.sh that produces the
# combined Markdown, HTML and optionally PDF in docs/presentations/.
# We invoke those scripts so they work identically inside or outside
# the CMake pipeline, then stage the outputs into DOCS_PRESENTATIONS_DIR
# where Sphinx picks them up via html_extra_path.

set(DOCS_DECKS
    "DNDSR_overview"
)

find_program(MARP_EXECUTABLE marp)
if(MARP_EXECUTABLE)
    set(_SLIDE_OUTPUTS "")
    foreach(_deck ${DOCS_DECKS})
        set(_deck_build "${PROJECT_SOURCE_DIR}/docs/presentations/${_deck}/build.sh")
        set(_deck_out_html "${PROJECT_SOURCE_DIR}/docs/presentations/${_deck}.html")
        set(_deck_out_pdf  "${PROJECT_SOURCE_DIR}/docs/presentations/${_deck}.pdf")

        file(GLOB_RECURSE _deck_deps CONFIGURE_DEPENDS
            "${PROJECT_SOURCE_DIR}/docs/presentations/${_deck}/*.md"
            "${PROJECT_SOURCE_DIR}/docs/presentations/${_deck}/*.sh"
            "${PROJECT_SOURCE_DIR}/docs/presentations/${_deck}/*.txt"
        )

        add_custom_command(
            OUTPUT "${_deck_out_html}" "${_deck_out_pdf}"
            COMMAND bash "${_deck_build}" --html --pdf
            DEPENDS ${_deck_deps}
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            COMMENT "Marp: rendering ${_deck} (HTML + PDF)"
            VERBATIM
        )
        list(APPEND _SLIDE_OUTPUTS "${_deck_out_html}" "${_deck_out_pdf}")
    endforeach()

    # After all decks are rendered, stage them into DOCS_PRESENTATIONS_DIR
    # (the single Sphinx html_extra_path entry). Build the per-deck copy
    # commands first so they go into a single add_custom_command call.
    set(_SLIDE_COPY_COMMANDS "")
    foreach(_deck ${DOCS_DECKS})
        list(APPEND _SLIDE_COPY_COMMANDS COMMAND ${CMAKE_COMMAND} -E copy
            "${PROJECT_SOURCE_DIR}/docs/presentations/${_deck}.html"
            "${DOCS_PRESENTATIONS_DIR}/${_deck}.html")
        list(APPEND _SLIDE_COPY_COMMANDS COMMAND ${CMAKE_COMMAND} -E copy
            "${PROJECT_SOURCE_DIR}/docs/presentations/${_deck}.pdf"
            "${DOCS_PRESENTATIONS_DIR}/${_deck}.pdf")
    endforeach()

    add_custom_command(
        OUTPUT "${DOCS_SLIDES_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DOCS_PRESENTATIONS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJECT_SOURCE_DIR}/docs/presentations/res"
            "${DOCS_PRESENTATIONS_DIR}/res"
        ${_SLIDE_COPY_COMMANDS}
        COMMAND ${CMAKE_COMMAND} -E touch "${DOCS_SLIDES_STAMP}"
        DEPENDS ${_SLIDE_OUTPUTS}
        COMMENT "Marp: staging slideshows into docs_generated/presentations/"
        VERBATIM
    )

    add_custom_target(docs-slides DEPENDS "${DOCS_SLIDES_STAMP}")
else()
    message(STATUS "marp-cli NOT found — slideshow rendering disabled")
    message(STATUS "  Install with: npm install -g @marp-team/marp-cli")
    set(DOCS_SLIDES_STAMP "")

    # Still expose docs-slides as a no-op so downstream targets can
    # unconditionally depend on it.
    add_custom_target(docs-slides
        COMMAND ${CMAKE_COMMAND} -E echo
            "marp-cli not found; slideshows will not be regenerated"
    )
endif()

# ===================================================================
#  3. Sphinx — user-facing documentation (Breathe + autodoc)
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
    set(_SPHINX_SLIDES_DEP "")
    if(DOCS_SLIDES_STAMP)
        set(_SPHINX_SLIDES_DEP "${DOCS_SLIDES_STAMP}")
    endif()
    set(_SPHINX_ELEMENTS_DEP "")
    if(DOCS_ELEMENTS_STAMP)
        set(_SPHINX_ELEMENTS_DEP "${DOCS_ELEMENTS_STAMP}")
    endif()

    # Ensure the generated asset dirs exist even when their generators
    # are disabled, so Sphinx's html_extra_path resolution doesn't fail.
    file(MAKE_DIRECTORY "${DOCS_PRESENTATIONS_DIR}")
    file(MAKE_DIRECTORY "${DOCS_ELEMENTS_DIR}")

    add_custom_command(
        OUTPUT  "${SPHINX_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E env
            "DOXYGEN_XML_DIR=${DOXYGEN_XML_DIR}"
            "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "DNDS_VERSION_FULL=${DNDS_VERSION_FULL}"
            "DNDS_DOCS_GENERATED_DIR=${DOCS_GENERATED_DIR}"
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
                ${_SPHINX_SLIDES_DEP} ${_SPHINX_ELEMENTS_DEP}
        COMMENT "Sphinx: building HTML documentation (incremental)"
        VERBATIM
    )

    # --- "sphinx" convenience target (always runs) ---
    add_custom_target(sphinx
        COMMAND ${CMAKE_COMMAND} -E env
            "DOXYGEN_XML_DIR=${DOXYGEN_XML_DIR}"
            "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "DNDS_VERSION_FULL=${DNDS_VERSION_FULL}"
            "DNDS_DOCS_GENERATED_DIR=${DOCS_GENERATED_DIR}"
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
    if(MARP_EXECUTABLE)
        add_dependencies(sphinx docs-slides)
    endif()
    if(DOCS_ELEMENTS_STAMP)
        add_dependencies(sphinx docs-elements)
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
        COMMAND ${CMAKE_COMMAND} -E echo "  Slides:  http://localhost:8000/presentations/"
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

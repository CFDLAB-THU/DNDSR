# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""Sphinx conf.py for DNDSR — Breathe (C++ API) + autodoc (Python API) + MyST."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

_project_root = Path(os.environ.get("PROJECT_SOURCE_DIR", "../../")).resolve()

# Make the Python package importable for autodoc.
sys.path.insert(0, str(_project_root / "python"))

# Local extensions live in docs/sphinx/_ext/
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# -- Project information -----------------------------------------------------

project = "DNDSR"
author = "DNDSR Team"
copyright = "2024, DNDSR Team"
version = os.environ.get("DNDS_VERSION_FULL", "dev")
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    # Markdown support (MyST)
    "myst_parser",
    # C++ API via Doxygen XML
    "breathe",
    # Python API from docstrings / .pyi stubs
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    # Math rendering
    "sphinx.ext.mathjax",
    # Graphviz dot graphs (for Breathe class/include diagrams)
    "sphinx.ext.graphviz",
    # Nice extras
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    # Local: @ref → {ref}, @tableofcontents → TOC, etc.
    "doxygen_compat",
]

# Fail on broken references only for explicit :ref: / :doc: — not for
# Doxygen-originated references that Breathe might not resolve.
nitpicky = False

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",        # $...$ and $$...$$ math
    "colon_fence",       # ::: directive syntax
    "deflist",           # definition lists
    "fieldlist",         # field lists
    "tasklist",          # - [x] checkboxes
    "attrs_inline",      # inline attributes
    "attrs_block",       # block attributes
]
myst_heading_anchors = 4  # auto-generate anchors for h1..h4

# Recognise the Doxygen-style {#anchor} as the heading slug.
# MyST parses `# Title {#my_id}` as header-id = "my_id" natively.

# Source files: .md handled by MyST, .rst by Sphinx core.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document (relative to source root = docs/).
root_doc = "index"

# Directories / patterns to skip.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Doxygen-only files
    "doxygen",
    # Legacy reports (now in docs/reports/)
    "reports",
    # Build scripts (not documentation)
    "getAllAttachForDox.py",
    "clean_doxygen_xml.py",
    "serve_docs.sh",
]

# -- Breathe configuration (C++ API from Doxygen XML) -----------------------

# DOXYGEN_XML_DIR is passed as an env var from CMake so the path is
# always correct regardless of build directory layout.
_doxygen_xml = os.environ.get(
    "DOXYGEN_XML_DIR",
    str(_project_root / "build" / "docs" / "xml"),
)

breathe_projects = {"DNDSR": _doxygen_xml}
breathe_default_project = "DNDSR"

# -- autodoc configuration (Python API) -------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = True

# -- Intersphinx (external project links) -----------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinxawesome_theme"
html_theme_options = {
    "show_breadcrumbs": True,
    "awesome_external_links": True,
}
html_static_path = []
html_title = f"DNDSR {version}"

# Show version in the top bar.
html_context = {
    "display_version": True,
}

# -- MathJax -----------------------------------------------------------------
# Use Sphinx's default MathJax v3 (handles \(...\) and \[...\] natively).
# The Doxygen build uses MathJax v2 separately; do not share the CDN URL.

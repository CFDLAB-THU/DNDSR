"""Sphinx extension that translates Doxygen commands in MyST markdown.

Handles the subset of Doxygen inline commands used in the DNDSR docs so
that the *same* markdown files can be consumed by both Doxygen and Sphinx
without manual edits.

Why a custom extension?
    There is no standard Sphinx extension for this.  Breathe handles
    Doxygen commands inside C++ *docstrings* (via XML), but does NOT
    process them in standalone .md content files.  MyST-Parser natively
    understands ``{#anchor}`` header IDs and ``[text](#anchor)`` links,
    but knows nothing about ``@ref``, ``@tableofcontents``, or ``@see``.
    sphinxcontrib-doxylink / exhale also don't cover this gap.

Supported translations
----------------------
- ``@tableofcontents``     →  ``{contents}`` directive (local page TOC)
- ``@ref anchor``          →  ``[](#anchor)``  (MyST native cross-ref)
- ``@ref anchor "text"``   →  ``[text](#anchor)``
- ``@see target``          →  **See also:** *target*
- ``@code{.lang}`` … ``@endcode``  →  fenced code block
- ``@section id Title``    →  markdown heading (##)
- ``@subsection id Title`` →  markdown heading (###)
- ``@note``                →  admonition
- ``@warning``             →  admonition
- ``@subpage anchor``      →  ``[](#anchor)`` (same as @ref for Sphinx)
- ``# Title {#id}``        →  ``(id)=`` label + ``# Title`` (clean sidebar titles)

The extension works as a *source-read* event hook: it rewrites the raw
source string before MyST parses it, so no custom docutils nodes are needed.
"""

from __future__ import annotations

import re
from typing import Any

from sphinx.application import Sphinx

# -- regex patterns for Doxygen commands -------------------------------------

# Heading with Doxygen-style {#anchor}: # Title {#some_id}
# Convert to MyST label syntax so the title stays clean in toctree/sidebar.
# Matches all heading levels (# through ######).
_RE_HEADING_ID = re.compile(
    r"^(#{1,6})\s+(.+?)\s+\{#(\w+)\}\s*$", re.MULTILINE
)

# @tableofcontents  (standalone line)
_RE_TOC = re.compile(r"^\s*@tableofcontents\s*$", re.MULTILINE)

# @ref anchor "optional text"   (inline)
_RE_REF_WITH_TEXT = re.compile(r"@ref\s+(\w+)\s+\"([^\"]+)\"")
_RE_REF_PLAIN = re.compile(r"@ref\s+(\w+)")

# @see target  (rest of line becomes the "See also" text)
_RE_SEE = re.compile(r"^\s*@see\s+(.+)$", re.MULTILINE)

# @code{.lang} ... @endcode  (block)
_RE_CODE_BLOCK = re.compile(
    r"@code\{\.(\w+)\}\s*\n(.*?)@endcode", re.DOTALL
)

# @section id Title  /  @subsection id Title
_RE_SECTION = re.compile(r"^\s*@section\s+\w+\s+(.+)$", re.MULTILINE)
_RE_SUBSECTION = re.compile(r"^\s*@subsection\s+\w+\s+(.+)$", re.MULTILINE)

# @note / @warning  (consume the rest of the paragraph)
_RE_NOTE = re.compile(r"^\s*@note\s+(.+)$", re.MULTILINE)
_RE_WARNING = re.compile(r"^\s*@warning\s+(.+)$", re.MULTILINE)

# @subpage anchor  (rare in .md files; treat like @ref)
_RE_SUBPAGE = re.compile(r"@subpage\s+(\w+)")


def _translate(source: str) -> str:
    """Apply all Doxygen-to-MyST translations to *source*."""

    # # Title {#anchor} → (anchor)=\n# Title
    # Must run first so later transforms see clean headings.
    source = _RE_HEADING_ID.sub(r"(\3)=\n\1 \2", source)

    # @tableofcontents → MyST {contents} directive
    source = _RE_TOC.sub(
        "```{contents}\n:local:\n:depth: 3\n```",
        source,
    )

    # @code{.lang} ... @endcode → fenced code block
    def _code_repl(m: re.Match) -> str:
        lang = m.group(1)
        body = m.group(2)
        return f"```{lang}\n{body}```"

    source = _RE_CODE_BLOCK.sub(_code_repl, source)

    # @ref anchor "text" → [text](#anchor)   (MyST native cross-ref)
    source = _RE_REF_WITH_TEXT.sub(r"[\2](#\1)", source)
    # @ref anchor → [anchor](#anchor)         (non-empty text avoids Sphinx crash)
    source = _RE_REF_PLAIN.sub(r"[\1](#\1)", source)

    # @subpage anchor → [anchor](#anchor)
    source = _RE_SUBPAGE.sub(r"[\1](#\1)", source)

    # @see target → bold "See also:" line
    source = _RE_SEE.sub(r"**See also:** *\1*", source)

    # @section / @subsection → markdown headings
    source = _RE_SECTION.sub(r"## \1", source)
    source = _RE_SUBSECTION.sub(r"### \1", source)

    # @note / @warning → admonitions
    source = _RE_NOTE.sub(r"```{note}\n\1\n```", source)
    source = _RE_WARNING.sub(r"```{warning}\n\1\n```", source)

    return source


def _on_source_read(
    app: Sphinx, docname: str, source: list[str]
) -> None:
    """Event handler: rewrite Doxygen commands before MyST parsing."""
    # source is a single-element list; mutate in-place.
    source[0] = _translate(source[0])


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("source-read", _on_source_read)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

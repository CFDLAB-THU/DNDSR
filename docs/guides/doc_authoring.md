# Documentation Authoring Guide {#doc_authoring}

How to add, edit, and organize documentation pages in DNDSR.
The project uses a **dual doc system**: content lives in shared Markdown
files consumed by both Doxygen and Sphinx.

@tableofcontents

## Architecture overview

```
docs/
├── index.md                 # Sphinx root page (NOT seen by Doxygen)
├── doxygen/
│   ├── Doxyfile             # Doxygen config (configure_file'd by CMake)
│   └── main.dox             # Doxygen sidebar tree (@mainpage + @subpage)
├── sphinx/
│   ├── conf.py              # Sphinx config (Breathe, MyST, theme)
│   ├── _ext/doxygen_compat.py  # Translates @ref/@tableofcontents for MyST
│   ├── guides.md            # Sphinx toctree stub for Guides section
│   ├── architecture.md      # Sphinx toctree stub for Architecture
│   ├── theory.md            # ...
│   ├── tests.md
│   ├── dev.md
│   ├── api_cpp.md           # C++ API via Breathe
│   └── api_python.md        # Python API via autodoc
├── guides/                  # Shared content (both systems)
│   ├── building.md
│   ├── style_guide.md
│   └── ...
├── architecture/            # Shared content
├── theory/                  # Shared content
├── tests/                   # Shared content
├── dev/                     # Shared content
├── reports/                 # Legacy reports (excluded from both)
├── clean_doxygen_xml.py     # Fixes malformed Doxygen XML
├── getAllAttachForDox.py     # Copies image attachments for Doxygen
└── serve_docs.sh            # Standalone doc server script
```

Every `.md` file under `guides/`, `architecture/`, `theory/`, `tests/`,
`dev/` is consumed by **both** Doxygen and Sphinx. Doxygen reads them
natively; Sphinx reads them via MyST-Parser, and the `doxygen_compat.py`
extension translates Doxygen commands (`@ref`, `@tableofcontents`, etc.)
into MyST equivalents.

## How content flows through both systems

```
  Markdown file (docs/guides/my_page.md)
       │
       ├──► Doxygen reads it directly
       │    - {#anchor} becomes a page ID
       │    - @ref, @subpage create hyperlinks
       │    - Output: build/docs/html/  (embedded at /doxygen/)
       │
       └──► Sphinx reads it via MyST-Parser
            - doxygen_compat.py translates @ref → [anchor](#anchor)
            - {#anchor} → (anchor)= label (clean sidebar titles)
            - Output: build/docs/sphinx/
```

## Adding a new documentation page

### Step 1: Create the Markdown file

Place it in the appropriate directory under `docs/`:

| Section | Directory | Example |
|---------|-----------|---------|
| Guides | `docs/guides/` | `docs/guides/my_guide.md` |
| Architecture | `docs/architecture/` | `docs/architecture/my_design.md` |
| Theory | `docs/theory/` | `docs/theory/my_method.md` |
| Tests | `docs/tests/` | `docs/tests/my_tests.md` |
| Dev notes | `docs/dev/` | `docs/dev/my_notes.md` |

The file **must** start with a heading that includes a `{#page_id}` anchor:

```markdown
# My New Guide {#my_guide}

Content goes here...
```

The `{#my_guide}` anchor serves both systems:
- **Doxygen** uses it as the page identifier for `@subpage` / `@ref`.
- **Sphinx** uses it as a cross-reference label (translated to `(my_guide)=`
  by `doxygen_compat.py`).

### Step 2: Register in the Doxygen sidebar

Edit `docs/doxygen/main.dox`. Find the parent section page and add a
`@subpage` entry:

```c
/**
 * @page page_guides Guides
 *
 * - @subpage building
 * - @subpage style_guide
 * - @subpage my_guide          ← add this line
 */
```

The `@subpage my_guide` creates a **parent → child** relationship in the
Doxygen sidebar tree. Without it, the page becomes an orphan at the top
level.

### Step 3: Register in the Sphinx toctree

Edit the corresponding Sphinx toctree stub in `docs/sphinx/`. For a
guide, edit `docs/sphinx/guides.md`:

```markdown
\`\`\`{toctree}
:maxdepth: 2

/guides/building
/guides/style_guide
/guides/my_guide
\`\`\`
```

Paths are **absolute from the Sphinx source root** (`docs/`), so
`/guides/my_guide` resolves to `docs/guides/my_guide.md`.

### Step 4: Build and verify

```sh
# Incremental build (only rebuilds changed files)
cmake --build build -t docs

# Serve and inspect
cmake --build build -t serve-docs
# Then open http://localhost:8000/ (Sphinx) and
# http://localhost:8000/doxygen/ (Doxygen) to verify both sidebars.
```

## Doxygen sidebar tree (`main.dox`)

The file `docs/doxygen/main.dox` defines the **entire** Doxygen sidebar
hierarchy using `@mainpage`, `@page`, and `@subpage`. The structure is:

```
@mainpage DNDSR Documentation
├── @subpage page_guides           (Guides)
│   ├── @subpage building
│   ├── @subpage style_guide
│   ├── @subpage my_guide          ← new pages go here
│   └── ...
├── @subpage page_architecture     (Architecture)
│   ├── @subpage paradigm
│   └── ...
├── @subpage page_theory           (Theory)
│   └── ...
├── @subpage page_tests            (Unit Tests)
│   └── ...
└── @subpage page_dev              (Development Notes)
    └── ...
```

**Rules:**

- `@mainpage` can only appear once. It defines the root page.
- Each `@page page_id Title` creates a named page.
- `@subpage child_id` inside a `@page` block creates a parent-child edge
  in the sidebar tree.
- A page that is NOT referenced by any `@subpage` becomes an **orphan** —
  it appears at the top level alongside Namespaces/Classes/Files.
- The `child_id` must match the `{#child_id}` anchor in the `.md` file.

## Sphinx toctree stubs (`docs/sphinx/`)

Each section has a stub file that defines the Sphinx sidebar tree using
MyST `toctree` directives:

| Stub file | Section |
|-----------|---------|
| `docs/sphinx/guides.md` | Guides |
| `docs/sphinx/architecture.md` | Architecture |
| `docs/sphinx/theory.md` | Theory |
| `docs/sphinx/tests.md` | Unit Tests |
| `docs/sphinx/dev.md` | Development Notes |

The stubs are referenced from `docs/index.md` (the Sphinx root):

```markdown
\`\`\`{toctree}
:maxdepth: 2
:caption: Contents

/sphinx/guides
/sphinx/architecture
/sphinx/theory
/sphinx/tests
/sphinx/dev
/sphinx/api_cpp
/sphinx/api_python
\`\`\`
```

**Rules:**

- Toctree paths are absolute from the source root (`docs/`).
- The path `/guides/building` resolves to `docs/guides/building.md`.
- Files not listed in any toctree generate an "orphan" warning.
- The file must exist or sphinx-build will fail.

## Cross-referencing between pages

### In Markdown files (shared content)

Use Doxygen's `@ref` syntax. The `doxygen_compat.py` extension translates
these for Sphinx automatically.

```markdown
See @ref array_infrastructure for the Array class hierarchy.
```

- **Doxygen** creates a hyperlink to the page with `{#array_infrastructure}`.
- **Sphinx** translates to `[array_infrastructure](#array_infrastructure)`,
  which MyST resolves to the page with `(array_infrastructure)=` label.

For references with custom display text:

```markdown
See @ref array_infrastructure "the Array docs" for details.
```

### In C++ docstrings

See the [C++ Docstrings](#c-docstrings-doxygen) section of the
@ref style_guide for `@ref` usage in source comments.

## The `doxygen_compat.py` extension

This Sphinx extension (`docs/sphinx/_ext/doxygen_compat.py`) runs as a
`source-read` hook, rewriting Doxygen commands before MyST parses the
Markdown. Translations:

| Doxygen command | Sphinx/MyST equivalent |
|----------------|----------------------|
| `# Title {#id}` | `(id)=` label + `# Title` |
| `@ref anchor` | `[anchor](#anchor)` |
| `@ref anchor "text"` | `[text](#anchor)` |
| `@subpage anchor` | `[anchor](#anchor)` |
| `@tableofcontents` | `` ```{contents} `` `` directive |
| `@section id Title` | `## Title` |
| `@subsection id Title` | `### Title` |
| `@see target` | **See also:** *target* |
| `@note text` | `` ```{note} `` `` admonition |
| `@warning text` | `` ```{warning} `` `` admonition |
| `@code{.lang}` ... `@endcode` | fenced code block |

## Build system targets

| Target | Behavior | What it builds |
|--------|----------|---------------|
| `docs` | Incremental (stamp-based) | Doxygen XML → Sphinx HTML + embedded Doxygen HTML |
| `sphinx` | Always runs | Sphinx HTML (forces rebuild) |
| `doxygen` | Always runs | Doxygen HTML + XML (forces rebuild) |
| `serve-docs` | Incremental + prints serve command | Full site |
| `serve-doxygen` | Forces doxygen + prints serve command | Doxygen standalone |

The `docs` target uses ninja dependency tracking:
- C++ source changes → re-run doxygen → re-run sphinx
- Markdown changes → re-run sphinx only (doxygen skipped)
- No changes → `ninja: no work to do` (instant)

## Checklist for adding a page

- [ ] Create `docs/<section>/my_page.md` with `# Title {#page_id}` heading
- [ ] Add `@subpage page_id` to the correct `@page` block in `docs/doxygen/main.dox`
- [ ] Add `/section/my_page` to the correct toctree in `docs/sphinx/<section>.md`
- [ ] Run `cmake --build build -t docs` and verify both sidebars
- [ ] Check for orphan warnings in the build output

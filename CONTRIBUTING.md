# Contributing to DNDSR

Thank you for your interest in contributing to DNDSR. This project is
primarily a research code developed by the DNDSR team, but external
contributions — bug reports, documentation improvements, and well-scoped
features — are welcome.

---

## Getting Started

1. **Build the project** — see `docs/guides/building.md` for full instructions.
2. **Run the tests** — see `docs/tests/overview.md` for C++ and Python test commands.
3. **Read the style guide** — see `docs/guides/style_guide.md` for C++ and Python conventions.
4. **Understand the architecture** — see `docs/architecture/` for design documents.

---

## How to Contribute

### Reporting Bugs

- Use GitHub Issues
- Include: build configuration (compiler, MPI implementation, CMake preset),
  steps to reproduce, expected vs. actual behavior, and relevant logs

### Documentation Improvements

- Documentation lives in `docs/` as Markdown files consumed by both Sphinx and Doxygen
- See `docs/guides/doc_authoring.md` for how to add or edit pages
- When editing, keep the dual-doc system in mind (Doxygen + Sphinx)

### Code Contributions

> **TODO:** Add process details:
> - Branch naming conventions
> - Commit message style
> - Pull request template
> - Required reviews and CI checks
> - clang-format and clang-tidy requirements

---

## Development Workflow

### Adding a New Test

> **TODO:** Link to detailed test guide (C++ doctest and Python pytest)

### Adding a New JSON Config Parameter

> **TODO:** Describe the config registration system and where to document

### Adding a New Documentation Page

See `docs/guides/doc_authoring.md` for the full checklist.

---

## Code of Conduct

Be respectful and constructive. DNDSR is a research code — design decisions
often reflect specific numerical-method requirements rather than general
software-engineering preferences. Discussion is welcome.

---

## See Also

- `docs/guides/building.md` — build instructions
- `docs/guides/style_guide.md` — coding conventions
- `docs/guides/doc_authoring.md` — how to write documentation
- `docs/tests/overview.md` — test suite documentation
- `docs/dev/TODO.md` — active development tasks

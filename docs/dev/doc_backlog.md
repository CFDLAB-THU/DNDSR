# Documentation Improvement Backlog

> **Status:** Tracking items from the 2026-05-05 documentation review.
> This is a workspace scratchpad, not published documentation.

---

## Problem 8: Architecture Docs Are Overwhelmingly Long

Dense design documents lack entry points for new readers.

- [ ] `MeshConnectivity.md` (674 lines) — add TL;DR, use collapsible sections for proofs
- [ ] `MeshDAGDesign.md` (765 lines) — clarify proposal vs. implemented status in each section
- [ ] `distributed_reorder_design.md` (1400+ lines) — excellent but exhausting; add section summaries
- [ ] `mesh_connectivity_impl_plan.md` (1200+ lines) — mix of design, implementation, tests, proofs

**Partially addressed:** TL;DR paragraphs added on 2026-05-05.

---

## Problem 9: Dev Notes Are Unstructured

`dev/Notes.md` is 175 lines of raw observations with no structure.

- [ ] Split into focused docs when content stabilizes:
  - `performance_notes.md` — timing numbers, hardware comparisons
  - `build_troubleshooting.md` — platform-specific build snippets
  - `platform_notes.md` — oneAPI, Windows, supercomputer quirks

**Status:** Kept as-is for now; marked as personal notes.

---

## Problem 10: Theory Docs Lack Context

`theory/Variational_Reconstruction.md` and `Shape_Functions.md` are mostly LaTeX with minimal prose.

- [ ] Add introductory paragraphs explaining:
  - What problem VR solves (reconstruction from cell averages)
  - When to use which functional (Normal, X-Y, Pre-Isotropic)
  - How the math maps to code classes
- [ ] For shape functions: explain element hierarchy, node ordering conventions

---

## Problem 11: Missing Cross-References

Many docs exist in isolation.

- [ ] `python_geom_guide.md` → link to `MeshConnectivity.md` for ghost/ adjacency details
- [ ] `array_infrastructure.md` → link to `Serialization.md`
- [ ] `tests/*.md` → cross-link to source files tested and to each other
- [ ] `guides/building.md` → link to `project_structure.md` for directory layout
- [ ] All architecture docs → add "See also" sections at bottom

---

## Problem 12: "What / When / Why" Guidance Missing

Docs explain *how* but not *when* to use APIs.

- [ ] `array_infrastructure.md`: when to use `ArrayDof` vs. `ArrayEigenMatrixPair`
- [ ] `Serialization.md`: when to use `ReadSerializeRedistributed` vs. `ReadSerialize`
- [ ] `python_geom_guide.md`: trade-offs (CGNS vs. H5, elevation vs. bisection)
- [ ] `building.md`: when to use each CMake preset

---

## Problem 13: No Decision Logs

Major design decisions are scattered.

- [ ] Create `docs/architecture/decisions.md` or add section to `Paradigm.md`:
  - Father/son terminology vs. local/ghost
  - Explicit arrays vs. DMPlex
  - Inheritance for `AdjPairTracked` vs. composition
  - JSON vs. other config formats

---

## Problem 14: Style Guide Is Incomplete

- [ ] Python conventions (beyond the brief mention)
- [ ] Docstring conventions for pybind11-exposed methods
- [ ] JSON schema conventions
- [ ] Test naming for new modules
- [ ] Documentation authoring conventions (when to use `@ref`, `{#anchors}`, etc.)

---

## Other Items

- [ ] Add "See also" / "Related pages" footers to all major docs
- [ ] Consider adding a glossary page (terms like father/son, cone/support, VR, VFV)
- [ ] Ensure all new docs are registered in both `main.dox` and `sphinx/*.md`

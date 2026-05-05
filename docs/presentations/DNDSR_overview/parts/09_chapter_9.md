---
<!-- _class: chapter -->
<!-- _paginate: false -->

<div class="ch-num">CHAPTER 9</div>

# Roadmap & Close

## What's next · pointers · acknowledgements

---
<!-- _footer: "docs/architecture/MeshConnectivity.md:416-458 · MeshDAGDesign.md" -->
<!-- _class: denser -->

## Roadmap — mesh & topology

<div class="cols">
<div>

### Near-term: configurable ghost

```cpp
struct GhostRequirement {
    int  cellRings       = 1;     // # of cell2cell rings
    bool nodeNeighbor    = true;  // cell2cell by vertex share vs face share
    bool complementNodes = true;  // ghost cells keep all their nodes
    bool complementBnds  = true;  // owned nodes keep all their bnds
};
```

- FEM needs a **smaller** ghost set than compact FV.
- Wide-stencil FV (2+ layers) needs a **larger** set.
- Current default ghost is hard-coded — `nGhostLayers` only adjusts depth,
  not the *kind* of neighbor.

</div>
<div>

### Medium-term: edge entities

- Add `edge2node`, `cell2edge`, `node2edge` with the same
  `AdjPairTracked` discipline.
- Extract the face-interpolation algorithm into a **generic** codim-K
  template.
- Needed for **node-based FV** and **FEM** workflows.

### Long-term: DMPlex-style DAG

- Unified point numbering across nodes / edges / faces / cells.
- Parameterized adjacency — `useCone` × `useClosure` Boolean matrix.
- PetscSF-style communication structure.
- PetscSection-style decoupling of topology from discretization.
- Proposal in `docs/architecture/MeshDAGDesign.md` (765 lines).

</div>
</div>

---
<!-- _footer: "RELEASE_NOTES.md:20 · src/EulerP/ · docs/dev/ideas.md" -->
<!-- _class: denser -->

## Roadmap — solvers, parallelism, V&V

<div class="cols">
<div>

### Solvers

- **Reactive / multi-species** (`NS_EX`) — maturity pass, published
  validation cases.
- **Overset grids** — 2D demo exists (hole creation, distance map,
  cell-cell connectivity); extend to 3D and add full transfer operators.
- **Full (assembled) Jacobian** — currently matrix-free + LU-SGS only.
  Assembling a block CSR Jacobian opens up AMG + SuperLU_dist + PETSc
  as preconditioners.

### Parallelism

- Extend CUDA coverage from `EulerP` to the full `Euler` evaluator.
- More SoA kernels; GPU-aware MPI over pinned device memory.
- Broader device coverage for VR reconstruction.
- Task-based execution experiments.

</div>
<div>

### V&V workflow

- Standardized **verification & validation** harness running a fixed case
  set on every release, publishing convergence plots to the doc site.
- More turbomachinery cases: NASA Rotor 67, low-speed fan stage.
- Noise-source diagnostics for aeroacoustics.

### Documentation & dev-UX

- **Expand Python examples** to match current C++ coverage.
- **Clang-tidy sanitation** for remaining modules (Solver, Geom, CFV,
  Euler, EulerP) — same recipe as DNDS.
- **Contributor on-boarding** guide, already outlined in `docs/dev/`.

</div>
</div>

---
<!-- _footer: "docs/architecture/ · docs/theory/ · docs/guides/" -->
<!-- _class: denser -->

## Where to read next

<div class="cols">
<div>

### Architecture

- **`array_infrastructure.md`** — bottom-up tour of `Array` →
  `ArrayTransformer` → `ArrayPair` → `ArrayDof`.
- **`MeshConnectivity.md`** — the AdjPairTracked state machine, the
  ghost-spec DSL, the DAG roadmap.
- **`Serialization.md`** — layer-cake I/O, cross-np restart, offset modes.
- **`Paradigm.md`** — the delayed-abstraction philosophy contrasted with
  OpenFOAM / SU2.

### Theory

- **`Variational_Reconstruction.md`** / `.pdf` — full derivation of the
  facial functional, inner-product choices, and local system.
- **`Shape_Functions.md`** — per-element shape functions and quadrature.

</div>
<div>

### Guides

- **`building.md`** — externals, headers, CMake presets.
- **`array_usage.md`** — how to write code with `Array` / `ArrayDof`.
- **`geom_usage.md`** — mesh construction and VR pipeline.
- **`python_geom_guide.md`** — full Python Geom API reference.
- **`serialization_usage.md`** — HDF5 checkpoints, redistribution.
- **`style_guide.md`** — C++ and Python conventions.
- **`examples.md`** — runnable `examples/ex_*.cpp` programs.

### Tests

- **`docs/tests/overview.md`** — golden values, determinism, suite totals.
- Per-module test pages under `docs/tests/{dnds,geom,cfv,euler,solver}_unit_tests.md`.

</div>
</div>

---
<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _footer: "github.com/CFDLAB-THU/DNDSR · cfdlab-thu.github.io/DNDSR" -->

# Thank you

**Try it in three commands**

```bash
cmake --preset release-test
cmake --build build -t euler -j32
mpirun -np 4 ./build/app/euler.exe cases/euler_config_IV.json
```

<br>

**Code** · [github.com/CFDLAB-THU/DNDSR](https://github.com/CFDLAB-THU/DNDSR)
**Docs** · [cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR)
**Release notes** · `RELEASE_NOTES.md` (v0.1.0)

<br>

*CFD Lab, Tsinghua University*

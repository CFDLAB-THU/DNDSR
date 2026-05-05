# 🚀 DNDSR v0.1.0

Major milestone release — 701 files changed, ~110k insertions across 241 commits.

📖 **Documentation**: [cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR/)

---

## ✈️ New Solver Features

- 🔄 **p-Multigrid (pMG)**: HM3 revamp with MG support, tpMG, incFScale for MG lower levels, positivity-preserving limiters in LimiterUGrad
- 🌀 **SA-IDDES**: DES → DDES → IDDES progression, psi term fixes, rotation correction variants, ft2 toggle
- 🧮 **k-omega RANS**: two-equation model with `euler2EQ` / `euler2EQ3D` executables
- ⏱️ **ODE integrators**: ESDIRK2, ESDIRK3, Trapezoidal; HM3 U2R2/U2R1/U3R1 modes
- 🔁 **Axisymmetric flow** with wedge-shaped metric in reconstruction
- 🧊 **Isothermal wall BC** (`BCWallIsothermal`)
- 📈 **incFScale** (incremental flux scaling) integrated into entropy fix
- ⚡ **Roe_M8 flux**; HLLE+ (experimental)
- 🎛️ `source2nd`, `mergeMultiResidual`, `normOrd`, `restartOutAtInit`, `resBaseType` options
- 🕸️ **Overset grid exploration**: hole creation, distance map, cell-cell connectivity (2D demo)

## 🐍 Python Modules (new)

- 📦 pybind11 bindings for **DNDS**, **Geom**, **CFV**, **EulerP** modules
- 📊 Python `ModelEvaluator` for CFV — Fourier dissipation-dispersion analysis
- 🗺️ `create_mesh_from_CGNS`, boundary mesh extraction, VTK output
- 🔗 ArrayPair, ArrayEigenMatrix/Vector/Batch exports
- 🛡️ `py::classh` holder for safe Python↔C++ ownership
- 🧩 Auto stubgen on install; `DNDSR.XXX` import structure

## 🏗️ Build & Packaging

- 🔧 Modular CMake with presets (`release-test`, `debug`, `cuda`)
- 🏷️ Centralized version from `VERSION` file + `git describe` (PEP 440)
- 📦 Python package via scikit-build-core with editable installs
- 📋 JSON Schema generation and config validation (`--emit-schema` CLI)
- ⚙️ `DNDS_DECLARE_CONFIG` registry for typed config sections
- 🧵 OMP support in ILU; `DNDS_PYBIND11_NO_LTO` option
- 🍎 macOS compilation fixes (fmtlib workaround)

## 🔬 CFD Module Refactoring

- **Euler**: `RANSModelTraits`, BC strategy pattern, shared gas physics, centralized Roe preamble / magic numbers / model constants
- **CFV**: 8-phase refactoring — safety, dedup, boundaries, functionals, data decomposition, DeviceTransferable, DOF factory, cleanup
- **Geom**: modular element traits with codegen shape functions, distributed mesh read with ParMetis repartition, extracted helpers (CGNS read, face interpolation, cell reorder, VTK output)

## ⚙️ Core Infrastructure

- 💾 Redistributable Array serialization with HDF5 checkpoint/restart
- 🧰 `ObjectNaming`, `make_ssp`, `ArrayPair` helpers
- 🎮 CUDA: device storage, device views for all derived array types, block-sparse MatVec benchmark, SoA layout tests
- 🚄 **EulerP** module with CUDA-optimized evaluator
- 🗂️ Mesh: `cell2cellOrig` / `node2nodeOrig` / `bnd2bndOrig` index tracking, node wall distance, boundary ghost via `node2bnd`

## ✅ Testing (new)

- 🧪 **C++ (doctest)**: 29 executables, 600 test cases across DNDS (249), Geom (193), CFV (67), Euler (62), Solver (29)
- 🐍 **Python (pytest)**: 58 test functions across 10 files — CFV FV/VR (43), DNDS MPI (9), Euler restart (3), Geom CGNS (2), EulerP CUDA (1)
- 🔁 CTest registration at np=1, 2, 4, 8 for all MPI-aware tests (82 registrations total)

## 📖 Documentation System (new)

- 🌐 **Sphinx + Breathe** site with Doxygen HTML embedded at `/doxygen/`
- 🔀 Shared Markdown for both Doxygen and Sphinx (`doxygen_compat.py` translator)
- ⚡ Incremental builds — no-op <1s, md-only ~10s, full ~2.5min
- 📚 Guides: building, style, arrays, serialization, geometry, examples, doc authoring
- 🏛️ Architecture pages, theory pages, test documentation
- 📐 C++ API with class diagrams (Breathe + Graphviz); Python API via autodoc
- 🤖 GitHub Actions workflow for manual Pages deployment with 3-layer caching
- 🎨 sphinxawesome-theme with colorful syntax highlighting

## 🐛 Notable Bug Fixes

- 🔒 MPI deadlock from collective `globalSize()` — made non-collective
- 🔗 Serializer false shared-ptr dedup from address reuse
- 🔄 `InterpolateFace` wrong face identity on double-periodic BCs
- 💥 Inner partitioning segfault in Metis reordering
- 📋 `ArrayTransformer` clone error
- 🔁 Periodic boundary `RecoverCell2CellAndBnd2Cell` on arbitrary partitions
- 💾 CSR `pRowStart` not written in serialization
- ⏱️ Multigrid `fdtau`/`frhs` ordering corrupting lambda estimates
- 🎮 EulerP accidental `to_device` in face buffer creation
- 🔧 CUDA thrust call fixes with `CMAKE_CUDA_ARCHITECTURE=native`

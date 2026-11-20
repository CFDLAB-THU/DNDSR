(index_page)=
# DNDSR Documentation

DNDSR is a C++17 / Python CFD (Computational Fluid Dynamics) research code
implementing Compact Finite Volume methods with MPI parallelism and optional
CUDA GPU support.

## Start here

| Task | Entry point |
| --- | --- |
| Install dependencies and build | [Build guide](guides/building.md) |
| Run a first case | [Worked examples](guides/examples.md) |
| Write or run tests | [Test guide](sphinx/tests.md) |
| Find past results and numerical corrections | [Validation records](dev/validation_records.md) |

## Architecture entry points

The [source-scoped CUDA assertion control](dev/cuda_assertions.md) describes
device bounds checks and the optional per-source production-kernel opt-out.

The [stream resource ownership contract](architecture/StreamResourceOwnership.md)
describes shared CPU/CUDA contexts and safe lease retirement after engine close.

The [Plan 61-2 profiling guide](dev/plan612_cuda_optimization.md) describes
opt-in CUDA timeline ranges and the fixed uncached TGV optimization baseline.

The [automatic A100 kernel defaults](dev/cauchy_a100_kernel_defaults.md) describe
setup-time device/shape selection and explicit reproducibility overrides.

The [experimental FV and VR matrix storage guide](dev/plan613_matrix_storage.md)
describes independent layouts, raw VR sweeps, and allocation controls.

Cauchy development starts at the [solver contracts and ledger](architecture/Cauchy.md).
The [finite-volume dispatch guide](architecture/CauchyDispatch.md) maps the
CPU and CUDA flux, source and Jacobian routes.
The [constant-density ACM guide](architecture/CauchyACM.md) covers mechanical
state input, pressure constraints, turbulence transport and qualification cases.
The [multidomain input and restart guide](guides/cauchy_multidomain.md) gives the
configuration shape, CGNS/HDF loader boundaries and simulation archive layout.
The [equation-state contract](architecture/CauchyState.md) documents semantic
segments, shared storage, leases and explicit physical/backend dispatch.
The [physical state input guide](architecture/CauchyStateInput.md) specifies
Euler-compatible tagged inputs, unit scales and physical reference states.
The [native LVR guide](architecture/CauchyLimitedVR.md) documents penalty
ownership, sensor ordering, halo phases and warm-state restart semantics.
Native linear operators, Jacobian assembly, staged SGS and local ILU are described
in [Cauchy implicit operator contracts](architecture/CauchyImplicit.md).
The [implicit benchmark guide](guides/cauchy_implicit_benchmarks.md) gives command-line inputs and timing definitions.
The [boundary convention guide](architecture/CauchyBoundary.md) distinguishes
Euler exterior targets, conservative gradient traces, viscous wall laws,
farfield switching and approximate implicit boundary derivatives.
The [compact geometry and cell-local PP contract](architecture/CauchyCompactGeometry.md)
describes canonical face incidence, center consumers, immutable placement and fused positivity.
The [Cauchy positivity policy](architecture/CauchyPositivity.md) specifies
thermal constraints, component scopes and preserved Euler repair ordering.
The [Cauchy output contract](architecture/CauchyOutput.md) defines centralized
history, console, manifest, checkpoint/visualization naming and exact-time
snapping and nonlinear checkpoints.
| Topic | Guide |
| --- | --- |
| Configuration publication, threading and native/Python library boundaries | [Configuration registration](architecture/Configuration.md) |
| Shared thermodynamics, chemistry workers, transport and RANS | [Shared Physics](architecture/Physics.md) |
| Mesh ownership, relation semantics, construction closure and ghost assembly | [Mesh connectivity and construction](architecture/MeshConnectivity.md) |
| Explicit communication policy in owned mesh preparation | [Mesh preparation policy](architecture/MeshPreparationPolicy.md) |
| Owned construction API and mesh access examples | [Native mesh construction](architecture/NativeMesh.md) |
| Shared Runtime arrays, placement, access and transfers | [Runtime Array](architecture/ArrayRuntime.md) |
| Fixed-record setup request/reply routing | [Sparse batch exchange](architecture/SparseBatchExchange.md) |
| Historical Array/ParArray interface | [Legacy Array infrastructure](architecture/array_infrastructure.md) |
| Runtime ordering and lifetime responsibilities | [Runtime caller contract](architecture/RuntimeCallerContract.md) |
| Non-owning allocation names and memory snapshots | [Allocation diagnostics](architecture/AllocationDiagnostics.md) |
| Optional single-device AMG provider | [AmgX source build](guides/amgx.md) |
| Optional submission-time dependency checking | [Checked eager execution](architecture/CheckedExecution.md) |
| Native field algebra | [Native DOF algebra](architecture/NativeDofAlgebra.md) |
| Native multigrid cycles | [FAS ownership, transfer and callback contracts](architecture/FAS.md) |
| Coupled time equations | [Native DITR equations and callback contracts](architecture/DITR.md) |
| Native block matrices and local factors | [Native block algebra](architecture/NativeBlockAlgebra.md) |
| Explicit device pack/MPI/unpack and overlap | [Device-packed transfer](architecture/DevicePackedTransfer.md) |
| Optional stream-ordered NCCL payloads and caller ordering | [NCCL transfers](architecture/NCCLTransfer.md) |
| Resident CUDA Jacobi/WBAP, placements and caller ordering | [Device reconstruction](architecture/DeviceReconstruction.md) |
| Native CPU FV/VR, explicit WBAP and boundary point interfaces | [FV and interface facilities](architecture/FVFacility.md) |
| Shared PP/RHS quadrature, cache placement and direct evaluation | [Quadrature and basis ownership](architecture/QuadratureBasis.md) |
| CPU reconstruction operation profiles and storage accounting | [Reconstruction capabilities](architecture/ReconstructionCapabilities.md) |
| Reusable element projection and distributed surface location | [Surface geometry and location](architecture/SurfaceLocation.md) |
| Completed CFV/Geom work orders and validation limits | [Facility completion ledger](dev/fv_completion_2026-10-31.md) |
| Entity-associated fields and migration layouts | [EntityField](architecture/EntityField.md) |
| Physical table formats, strides and local dense adapters | [Physical Array layouts](architecture/PhysicalArrayLayouts.md) |
| Python Cauchy logical-state, FV and linear operator composition | [Python Cauchy operators](guides/python_cauchy_operators.md) |
| Canonical and compatibility source paths | [Source organization](architecture/source_organization.md) |
| Cauchy CPU kernel ownership, layout and read/write effects | [Kernel I/O contracts](architecture/CauchyKernelContracts.md) |
| OpenMP policy, worker ownership and mesh preprocessing | [CPU threading](architecture/CPUThreading.md) |
| Mesh Python access and operations | [Python mesh guide](guides/python_geom_guide.md) |

## Core Modules

| Module     | Directory    | Description                                              |
|------------|-------------|----------------------------------------------------------|
| **DNDS**   | `src/DNDS`  | MPI arrays, serialization (JSON, HDF5), profiling, CUDA  |
| **Geom**   | `src/Geom`  | Unstructured mesh, CGNS I/O, partitioning (Metis/ParMetis) |
| **CFV**    | `src/CFV`   | Compact Finite Volume, variational reconstruction        |
| **Physics** | `src/Physics` | Shared thermodynamics, transport, chemistry and RANS kernels |
| **Cauchy** | `src/Cauchy` | Native FV solver, domain orchestration and implicit operators |
| **Euler**  | `src/Euler` | Compressible Navier-Stokes solvers (2D/3D, SA, k-omega)  |
| **EulerP** | `src/EulerP`| Alternative evaluator with CUDA GPU support              |
| **Solver** | `src/Solver`| ODE integrators and Krylov solvers (GMRES, PCG)          |

## Quick Start

```sh
# 1. Build external dependencies
git submodule update --init --recursive --depth=1
python3.12 -m venv venv && source venv/bin/activate
pip install -r external/cfd_externals/requirements.txt
cd external/cfd_externals && CC=mpicc CXX=mpicxx python cfd_externals_build.py && cd ../..
bash scripts/install_python_deps.sh

# 2. Build C++ solvers
cmake --preset release-test
cmake --build build -t euler -j32

# 3. Install Python package (editable)
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 pip install -e .

# 4. Run tests
ctest --test-dir build -R dnds_ --output-on-failure   # C++
pytest test/DNDS/test_basic.py -v                      # Python

# 5. Build and serve documentation
cmake --build build -t serve-docs
```

> **See also:** @ref building for the complete build guide (CMake presets,
> Python editable installs, external dependencies, and common issues).

## C++ API (Doxygen)

The Sphinx C++ API pages provide namespace outlines and key class documentation
via Breathe. For the full Doxygen HTML documentation with interactive
**call graphs**, **class inheritance diagrams**, and **include dependency graphs**,
build and serve the Doxygen output separately:

```sh
cmake --build build -t serve-doxygen
```

Once built, the full Doxygen reference is at
<a href="doxygen/index.html">Doxygen C++ API Reference</a> within the same site.

```{toctree}
:maxdepth: 2
:caption: Contents

/sphinx/guides
/sphinx/solver-guide
/sphinx/architecture
/sphinx/theory
/sphinx/tests
/sphinx/dev
/sphinx/api_cpp
/sphinx/api_python
/sphinx/presentations
```

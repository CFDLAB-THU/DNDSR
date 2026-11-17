# Plan 61-6 CUDA consolidation

The production CUDA RHS uses one CTA-segment implementation specialized by
component count C=5/6/7, modal count M and geometry dimension D. The shape
manifest supplies the valid 24 tuples, including non-VR M=0. Runtime block
width and optional viscous roles do not add instantiations. See
[kernel contracts](../architecture/CauchyKernelContracts.md#cuda-rhs-shape-catalog)
for shapes, generated-source dependencies and ownership. Generated outputs
live in the ignored build directory; only the generator, manifest, CMake
helper and private numerical body are versioned.

The reconstruction sweep has one general runtime-shaped FP64 body and one
optimized body. Production `fp64Pipelined` includes C=5/6/7 and a real dynamic-C
fallback, modal sizes 2/3/5/9/19, raw/precomputed matrices, full/triangular local
inverses and LVR. Nine historical arithmetic/mapping selectors retain their
single C5/M19 precomputed comparison fixture. Unsupported explicit combinations
fail before launch. Matrix storage and input field storage are separate choices;
optimized fields require compact AoS. See
[device reconstruction](../architecture/DeviceReconstruction.md).

Shared helpers centralize whole-array projection, dense block addressing,
zero-mean Taylor basis evaluation and integer diagnostic reduction. Numerical
choices, periodic vector transformations and queued operand lifetime contracts
are retained. Characteristic production is separate from its diagnostic norm.

AmgX remains an explicit single-device provider. The patch qualifies native
block widths 2/3/4/5/6/7/8/9/19 and exposes allocator/hierarchy/copy attribution.
Public upload and managed native-storage attachment use the same operator;
native attachment borrows finest numeric storage but does not eliminate
provider vector adaptation, coarse hierarchy ownership or provider host
synchronization. The CFV prototype assembles the projected modal VR system,
reuses one modal hierarchy across physical components, and rejects rotated
periodicity/LVR. See [AmgX contracts](../guides/amgx.md).

## Qualification recorded during implementation

The grouped CUDA13.2 build covers sm80/sm86, static/shared libraries, solver,
focused tests and Python bindings. Make/Ninja tests pass no-op generation,
script/manifest invalidation and missing-wrapper/header/index recovery.
Representative built objects have exactly one polynomial flux entry per
architecture, with no block-size or viscous-role multiplier.

Focused C++ checks pass Runtime Array/configuration, RHS against independent
generic CPU arithmetic for 2D/3D flow5/SA/2EQ, primitive-gradient RHS, domain and
queued spatial scheduling, characteristics, linear solves, reconstruction
math, CUDA reconstruction/LVR and native AmgX. Independent provider Galerkin
and borrowed-storage tests pass, including all required block widths. A
targeted AmgX Compute Sanitizer memcheck reports zero errors. Failed intermediate
fixture/build logs remain in the separate Plan 61-6 ledger; numerical tolerances
were not relaxed to recover these checks.

An inherited experiment Python 3.11 environment had a noncanonical h5py wheel.
Import-order-assisted tests are not final dependency qualification. The parent
DNDSR venv has MPI-enabled h5py/mpi4py using the supplied HDF5 1.14.3 and
OpenMPI 5.0.9. Binding targets are being rebuilt for its Python 3.12 ABI, with
the active checkout selected by `PYTHONPATH`; DNDSR is not pip-installed into
that environment. An unrelated legacy CFV Python test has a missing `os`
import; this work does not rewrite that test.

Matched unprofiled and per-kernel before/after device-0 TGV measurements, plus
flow/VR AMG memory and fixed-work comparisons, remain pending at this source
checkpoint. Passing unit tests or linking does not establish a performance
gain or long-run CFD convergence. Detailed inputs, failed/partial results and
timing provenance belong in the standalone plan/evidence repositories.

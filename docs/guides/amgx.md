# Single-device AmgX dependency

The current direct adapter uses one rank and one device, with the native
int32 pattern and row-major compact blocks supplied directly to AmgX. The
longer-term rank-local preconditioner vision requires native principal-matrix
partitions; it does not authorize hidden off-rank filtering or repacking.
There is no delegated multi-rank provider communication. This document records
the build recipe; an enabled dependency is not evidence of a completed solver
adapter or numerical validation.

AmgX is not in the current cfd_externals package. Build its pinned v2.5.0 source
(`cc1cebdbb32b14d33762d4ddabcb2e23c1669f47`) with the independent recipe:

```sh
python3 scripts/build_amgx.py --root workspace/dependencies/amgx \
    --architectures '80;86' --jobs 16
# Use --host-compiler /path/to/g++ and --cuda-compiler /path/to/nvcc
# when the system defaults are not compatible with the selected CUDA toolkit.
cmake -S . -B build-cuda -DDNDS_USE_CUDA=ON -DDNDS_USE_AMGX=ON \
    -DDNDS_AMGX_ROOT="$PWD/workspace/dependencies/amgx/install"
```

For auwd's RTX3060, use architecture `86` and the explicitly selected checkout
`~/projects/DNDSR`. The recipe forces `CMAKE_NO_MPI=ON`, records its configure
command/revision, bundled patch hash and installed library hashes. It accepts
only the exact clean pin or the pin with its own exact patch, refusing unrelated
local changes. CMake checks the actual selected library against the manifest. It installs only into its dedicated prefix and does not
install OS packages or change cfd_externals. Do not bypass CUDA host-compiler
compatibility checks as a default workaround.

`DNDS_USE_AMGX=OFF` remains valid for CPU/native-Jacobi builds. Explicitly
requesting AmgX requires the imported dependency; no fallback is implicit.
Runtime adapter calls must order input readiness and output completion according
to AmgX's actual API, without assuming it uses the application's stream.

`DNDS::Algebra::AmgXPreconditioner` accepts native block CSR for
native block widths 1/2/3/4/5/6/7/8/9/19 (and the retained width-10 reference); Cauchy flow5/SA/2EQ remain 5/6/7 and never expand into scalar rows.
The bundled [provider patch](../../scripts/patches/README.amgx.md) enables the modal widths 9/19 throughout the selected AMG path and retains the corrected 6/7 large-block kernels.
Genuinely scalar DNDS matrices may still use width one. `SetPattern` shares the
caller's compact host `Runtime::Array<int>` offsets/columns and retains read
leases, so those symbolic arrays cannot mutate until replacement/destruction.
Unique unsorted columns retain their input block order. Numeric values and optional separate diagonal use native row-major block
order. Although public upload supports `COL_MAJOR` for some other algorithms,
the selected `BLOCK_JACOBI` hierarchy explicitly requires `ROW_MAJOR`; this
adapter rejects another format before entering the provider. The public provider upload performs its own internal copies.

The Cauchy adapter currently supports **one rank** with a primary native int32
linear pattern, native row-major full compact AoS diagonal/off-diagonal arrays and compact AoS
vectors. Cauchy constructs its primary blocks in row-major order when AmgX is selected,
using `ROW_MAJOR` with the existing separate diagonal. The same native blocks
are interpreted by DNDS matrix actions/inversion; no second format is kept. It performs
no DNDS packing, transposition, sorting, off-rank filtering, padding removal or
int64-to-int32 conversion. Incompatible representation fails explicitly before
provider execution. Select `implicit.linear.matrixStorage.indices="auto"` (the
default, choosing native int32 for AmgX), or `"int32"`. Diagonal and off-diagonal
matrix layouts must be `"aos"`; the inverse matrix layout is independent. Future
multi-rank support requires a genuinely native principal-matrix partition, not
a hidden copied/filtered matrix. Numeric block views currently retain their
separate int64 block-row offsets; those are distinct from the primary provider
int32 CSR and are not converted or uploaded.

`UpdateValuesBorrowed` takes FP64 device values and optional separate diagonal;
`ApplyBorrowed` takes device residual/correction views. Both wrappers check
placement and synchronize explicitly. The submitting operator must drain
host-progressed communication before this provider boundary, retain scopes,
and abort on an exception in a collective solve. No MPI communicator is passed
to AmgX. Unsupported builds reject the provider instead of falling back.

The default is aggregation AMG with fixed V cycles, zero starting correction,
Jacobi smoothing and fixed coarsest Jacobi sweeps (`coarse_solver=NOSOLVER`).
Residual monitoring is disabled inside the provider, preserving a fixed linear
action. Cauchy `linear.method="amgx"`, `stopping="fixedCount"`, `iterations=2`
performs exactly two stationary corrections with no Krylov method. An explicitly
selected `gmres` method can also use this preconditioner. Pattern/value updates
belong between linear solves. The historical packed-adapter tests and case comparisons remain evidence of
the provider mathematics, not validation of this newer copy-free boundary.
Focused tests exercise native unsorted int32 patterns, separate row-major
nonsymmetric 5/6/7 blocks and fixed-count solver semantics. No arbitrary
nonsymmetric-system or all-component RANS convergence guarantee is implied.

## Caller-stream extension investigation

The pinned provider's internal `memory::setStream` is not a safe asynchronous
API. Selected restriction/prolongation launches still use stream zero;
`BlockJacobiSolver::smooth_BxB` lazily resizes scratch and uses an unqualified
Thrust copy; library handles are process-global; `AMG_Solver::solve` calls
`cudaFreeWait()`, which synchronizes stream zero. C API vector transfers use
synchronous copies. DNDS therefore retains the explicit synchronization above.

A future restricted extension can keep hierarchy setup synchronous and enqueue
a fixed, preallocated V-cycle on a caller stream. Every selected launch, library
handle and Thrust policy must be routed explicitly, with no allocation or host
convergence decision during apply. The new entry must preserve the existing
synchronous API. Initially permit only one outstanding apply per mutable
workspace, retaining the hierarchy, buffers and provider through a recorded
completion. Separate contexts cannot assume independent handles or pools merely
because they are separate C++ objects. Cross-stream reuse needs event ordering.

Extracting the arithmetic kernels alone is insufficient: an independent DNDS
V-cycle also needs managed hierarchy export, transfer maps, inverse blocks and
scratch ownership, followed by equivalence validation. A narrow provider apply
extension is the smaller first step. Neither asynchronous support nor CUDA graph
capture is implemented by this investigation. Capture would additionally require
stable addresses, no first-use allocation, and nondefault-stream/capture tests;
setup, adaptive host decisions and synchronizing diagnostics remain excluded.

The adapter explicitly nests Jacobi's `relaxation_factor` in the smoother's
configuration object. The earlier string-valued smoother selected AmgX's
default scope, so the requested 0.75 was ignored and 0.9 was actually used.
The corrected default is the requested 0.75; one undamped sweep with requested
factor 1 is independently checked against an exact nonsymmetric block solve.
Historical results therefore also differ in effective smoother damping.

## Native attached storage and diagnostics (Plan 61-6)

The ordinary `upload` path passes native numeric pointers directly to public
AmgX upload, which makes its own device copy. Explicit
`implicit.linear.matrixStorage.amgxStorage="native"` instead allocates the
canonical matrix once as `[off-diagonal blocks][diagonal blocks]`.
`LinearProblemContext` exposes the usual diagonal and off-diagonal Arrays as
disjoint shared-allocation subranges. There is no second DNDS numeric matrix.
The fork's restricted attachment API gives AmgX a non-owning numeric view of
that allocation; CSR indices, coarse operators and smoother inverses are still
provider-owned. Matrix-vector action uses the same native blocks.

`AmgXPreconditioner::AttachValues(context, array)` is the reusable managed
boundary. For inline diagonals the allocation includes all CSR blocks plus
one zero sentinel block; for separate diagonal it contains E off-diagonal
blocks followed by N diagonal blocks. Storage is compact FP64 row-major on the
selected CUDA device. The adapter retains the Array owner and acquires read
leases during each synchronous setup/apply. Between solves native mutation is
allowed, but applying before another `AttachValues` refresh rejects stale
matrix epochs. These semantics do not promise asynchronous provider execution.
Native attachment freezes the symbolic pattern. `SetPattern` then rejects
without changing state; create another provider for a different pattern.

The provider refuses resizing, filling, swapping, coloring, reordering or
scaling attached matrix payload. Upload/replace APIs cannot overwrite it.
Public and native attachment modes cannot be mixed on one preconditioner.
Native storage requires the exact patched fork; build in a new dependency root
so existing reference libraries remain available:

```sh
python3 scripts/build_amgx.py --root workspace/dependencies/amgx-plan616 \
  --source-tree /path/to/AMGX-native-blocks --architectures '80;86' \
  --cuda-compiler /usr/local/cuda-13.2/bin/nvcc --host-compiler /usr/bin/g++-12 --jobs 16
```

The external source is verified against the exact compiled-source patch and is
never reset or rewritten by the recipe. The manifest records its checkout
revision, immutable upstream pin, patch and installed library hashes.

`MemorySnapshot()` is an explicit diagnostic synchronization boundary. It
reports per-level numeric-owned/numeric-borrowed, pattern, inverse, transfer and
workspace capacities, plus boundary H2D/D2D byte counts. Set
`AMGX_TRACK_ALLOCATIONS=1` before the first provider creation to record logical
allocator live/peak requests. The CUDA default-pool high-water/reserve figures
cover the process/device pool, potentially shared with DNDS; they are not
additional AmgX-owned buffers and must not be added to level capacities.
Provider library-private allocations and allocator rounding are distinct from
array payload capacity. Record initial upload, setup, first/warm apply, refresh
and owner retirement separately. Production numerical kernels contain no
allocation-statistics atomics.


The maintained `dnds_benchmark_amgx` target emits one JSON record at each
matrix/setup/apply/refresh/retirement boundary for a reproducible block-CSR
microbenchmark. For example, separately run `65536 5 upload 8` and
`65536 5 native 8` on the same selected device. Repeat without
`AMGX_TRACK_ALLOCATIONS` for timing; allocation-traced setup is diagnostic work.
These synthetic operators qualify ownership/copy cost and do not replace CFD
convergence evidence. `LocalAMGPreconditioner::MemorySnapshot` forwards the same
explicit diagnostics for a prepared flow operator after the caller drains its
engine; do not query during host-progressed communication.

For a bounded real flow run, `DNDS_AMGX_FLOW_DIAGNOSTICS=1` prints
`AMGX_FLOW_MEMORY` JSON records at provider creation, initial setup, value
refresh and apply. The switch is off by default. Snapshot joins and printing
belong to diagnostic runs; use a separate run without this switch or allocator
tracking for timing. Native matrix bytes are reported separately from provider
capacity and must not be added again to the borrowed finest-level capacity.

## Experimental VR operator

`CFV::ReconstructionAMGPlan` prepares the **projected fixed-point operator**
`I - A^+ B` from a host `VariationalReconstructionPlan`. Here `A^+` is the
already prepared CFV filtered inverse. The new provider does not reinterpret
singular unprojected reconstruction matrices using its own diagonal inverse.
Repeated neighbor columns, including periodic self columns, are accumulated
into unique block-CSR columns. Raw full/triangular input is normalized during
CPU preparation; precomputed input supplies the same projected maps directly.

The current profile is one rank/device, non-LVR and no rotated periodicity.
It preserves the existing mean and boundary forcing. Translation periodicity
is supported; missing local columns fail explicitly. One modal-block hierarchy
is shared across physical components, with fixed zero-start defect corrections.
Combined coefficient rows remain `[component][mode]`; component gather/scatter
is explicit device work and provider RHS/result D2D copies remain. This is an
exploration facility, not a hidden replacement for the ordinary Jacobi sweep.

`Prepare` accepts `PublicUpload` or `NativeStorage`. In both cases the plan
owns one directly assembled native modal matrix. Native storage additionally
borrows it into the provider, without a provider finest numeric copy. The plan
retains immutable arrays and their leases; submitted operations retain the
plan and mutable workspace operands. A workspace is serially reused and its
output is distinct from input. Source mesh generations are still validated.

`SubmitDefectCorrection` currently drains its engine before the synchronous
provider call for each component. It is not stream-ordered or graph-capturable.
Setup/apply timing, contraction, vector adaptation and provider memory must be
reported separately. Requested widths 2/3/4/5/6/7/8/9/19 qualify the selected
provider kernels; they do not guarantee that AMG will converge for every
nonsymmetric VR operator. Incompatible block layouts and provider numeric
extents beyond its int32 arithmetic are rejected rather than repacked.

# Single-device AmgX dependency

Plan61 uses one single-device AmgX instance per rank/device. DNDS retains the
global operator and halo exchanges; AmgX approximates the owned-owned block.
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
5/6/7-component cell Jacobians; Cauchy never expands them into scalar rows.
The bundled [provider patch](../../scripts/patches/README.amgx.md) enables
6/7 throughout the selected AMG path and repairs its large-block kernels.
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

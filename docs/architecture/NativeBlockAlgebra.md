# Native block storage and local factors

`DNDS/Algebra/BlockMatrixStorage.hpp` owns uniform dense-block values in a native
Runtime Array and shares immutable sparse block-row offsets. It contains no mesh,
equation or symbolic-factorization policy. Copies share values. Read/write access
scopes retain allocation and row metadata; borrowed Eigen block views must not
outlive them. Lease ownership and layout setup occur outside block arithmetic.
These Eigen views are CPU-only.

`Solver/NativeBlockLU.hpp` is a rank-local native numeric matrix/factor adapter.
It uses generic `Direct::LocalLUBase` logic and a published `SerialSymLUStructure`;
it does not include EulerJacobian.hpp or legacy ArrayDOF. Sparse rows store
diagonal, lower, then upper blocks. Matrix and factors use separate allocations.
Fill matrix values before factoring. Diagonal inversion follows the active Euler
policy: row-diagonal scaling, partial-pivot LU with a1e-7 reciprocal-condition
guard, then filtered-SVD fallback. The storage-independent implementation is
`Solver/BlockInverse.hpp`; native LU and Cauchy diagonal preparation share it.
Fallback counts are safe across local parts.

Multiple native local parts require independent factor graphs. The native
constructor rejects cross-part factor edges. Generic symbolic construction
preserves supplied edges; callers provide induced part graphs before fill.
Passing part boundaries alone does not remove those couplings. This guard
prevents independent numeric/triangular scans from silently using cross-part
dependencies. It does not alter the legacy generic symbolic algorithm.

Rebuilding a generic symbolic object clears its previous incidence rows while
reusing row capacities. A new generation replaces the old pattern rather than
appending duplicate entries. Optional `report=false` suppresses the generic
communicator-synchronized diagnostic output, keeping the construction local;
the default retains legacy diagnostics. Native Cauchy uses the local mode.

The caller serializes mutation and execution. A new symbolic generation needs a
new native storage owner. Matrix writes invalidate factors. Solve pins matrix
and factor reads before checking versions, rejecting an active unpublished
writer. RHS, forward scratch and output must have separate allocations; each
can be an AoS, SoA or AoSoA uniform DOF table. Output acquisition failure cancels
an unused scratch write. No communication/error agreement is hidden here;
distributed callers coordinate local failures before subsequent MPI operations.

`SetCoreFirstOrdering` stably partitions the requested ordering within each local
part before symbolic fill. Factor positions remain distinct from cell indices.
`ReadySolveRows` propagates available RHS rows through actual lower/upper edges.
Short factor rows do not imply short solve-dependency chains. The ready core can
be empty; complete LU has no bounded-stencil core guarantee.

Generic numeric ranges are consecutive pivots. Lower solves advance in factor
order and upper solves reverse it. Lower-core completion produces forward scratch,
not final core solution values. Partial-range callers provide completed
predecessors and own phase/readiness validation. Native full Solve runs both
scans without MPI. An exposed middle transfer is a separate SGS orchestration
option. Legacy full-factor/full-solve entry points preserve their arithmetic.

`NativeBlockLU::BeginSolve(rhs, forward, result)` adds a guarded native
`TriangularSolve` scope. `Lower(begin,end)` advances factor-position ranges
within each independent symbolic local part; `Upper(begin,end)` reverses them
after that part's forward scan is complete. Progress uses two counters per part,
not a row bitmap. Independent local parts may advance in a different order.
These entry points require a fully host-visible RHS; they do not promise
concurrent writes to its other owned rows. Symbolic readiness metadata and kernel
ranges remain available for a future independently proven partial-input adapter.

The scope retains matrix/factor/RHS reads and exclusive forward/result writes.
No forward or final result is published before `Finish`, which requires both
complete scans. Bad/repeated ranges, premature publication and changed symbolic
generations throw before arithmetic. Abandoning started work invalidates its
partial outputs; an unused scope cancels unused writes. Numeric allocations and
symbolic metadata survive destruction of original native owners. Callers must
serialize symbolic mutation; direct edits to public generic symbolic vectors
without its generation protocol are unsupported. The full Solve wrapper uses
this same guarded numerical path and retains exact staged/full agreement.

Focused Solver tests cover independent dense solves, incomplete/full factors,
full/ranged agreement, core permutations/local parts, readiness counterexamples,
all three DOF layouts, empty ownership, inverse fallback, retained lifetimes,
aliases and stale/active-writer rejection, staged publication, generation changes,
independent local parts and owner expiration. Cauchy's nonlinear/physical
integration remains a separate consumer and needs its own distributed gates.

## Native field Krylov iteration

`Solver/FieldGMRES.hpp` provides restarted left-GMRES with native DOF workspaces.
It uses explicit operator, fixed-preconditioner and batched-inner-product
callbacks, without owning mesh or communication policy. Preparation visits the
stable operator-input fields so a caller can bind persistent halos once. Their
owned and ghost physical layouts are cloned separately from the prototype.
Callbacks complete synchronously and coordinate distributed errors; callers
serialize workspace access and release conflicting views before reuse. This is
the initial CPU profile; device vectors reject rather than fall back to host.

Two-pass orthogonalization batches projections, and Givens rotations track the
small Hessenberg residual. A small pivoted QR computes the candidate update,
including first-vector closure; rank-zero projections report breakdown. True
residual evaluation decides convergence. Extreme fixed preconditioner scaling
does not make a small left-scaled residual a false success. Usable results also
pass a finite-iterate check; finite iteration-limited corrections are distinct
from converged solves. Other failure reasons are not usable nonlinear updates.
NaN diagnostics mean that quantity was not evaluated before stopping.

Focused tests cover first-step closure, warm guesses, extreme preconditioner
scaling, restarted nonnormal dense oracles, all three layouts with empty local
ranks, zero operators/null preconditioners, nonfinite data and explicit iteration
limits. The legacy `GMRES_LeftPreconditioned` class remains unchanged.
## Device placement and block-row views

`BlockMatrixStorage` owns values and block-row offsets as Runtime arrays. Its
small immutable descriptor contains only row count and block dimension. Copies
share these allocations; a read/write access scope retains both value and offset
leases. `Placed(device, context)` makes a completed setup-time placement copy;
the result does not keep a host mirror of either payload.

`Read(context).ArrayView()` and `Write(context).ArrayView()` expose a borrowed
`BlockMatrixArrayView`: whole-array descriptors for block values and offsets,
plus the block dimension. A row is a flattened column-major square block;
`Element(iRow, iBlock, i, j)` indexes its `(i,j)` entry. The view is usable on
CPU/CUDA and checks no ownership or residency per entry. Host launchers validate
residency once with `ValidateDevice`. The caller retains the access scope until
the asynchronous operation completes, or submits the underlying arrays as
managed Runtime operands. Merely copying the view does not retain allocations.

The existing `View()` remains the CPU Eigen adapter and rejects device storage.
`Offsets()` is an explicit host diagnostic snapshot, not a numerical-loop API.
Device placement alone does not make the Cauchy linear solver CUDA-capable:
queued operator submission and inverse/preconditioner preparation are separate
integration steps.

## Rank-local CPU aggregation AMG

`DNDS::Solver::AggregationAMG` in `Solver/AggregationAMG.hpp` is a
synchronous OpenMP preconditioner for a local block system. The caller owns
the outer linear iteration, communication, and choice of geometric strength.
For example, the caller can supply face area divided by cell-center distance.

1. Construct the hierarchy from host Runtime tables of int32 CSR offsets
   `[N+1,1]`, local off-diagonal column IDs `[E,1]`, and FP64 edge strengths
   `[E,1]`. Each row has unique columns. In ascending seed order, an aggregate
   grows along its strongest positive outward edge to an unassigned row.
   Equal strengths select the smaller destination ID. Across two aggregates,
   directed edge strengths add. Coarsening stops at the configured size or
   depth limit, or when a pass cannot reduce the row count.
2. Call `Refresh(blocks, diagonal)` with FP64 host tables `[E,B*B]` and
   `[N,B*B]`. Each block is flattened in **row-major** order. Table rows may
   have tail padding. Supported block widths are 1 through 9 and 19.
   The solver retains managed handles to the finest blocks and rebuilds the
   numeric operators using the fixed aggregate membership and scatter slots.
   For piecewise-constant prolongation, coarse blocks are
   `A_c = Pᵀ A P`, including intra-aggregate edges in the diagonal. Each
   level uses a row-scaled, condition-guarded partial-pivot LU inverse with
   a filtered SVD fallback. A zero diagonal entry with a nonzero row uses
   that row's largest absolute entry for scaling. A structural zero row is
   retained in the scaled system.
3. Call `Apply(rhs, correction)` on distinct host tables `[N,B]` for one
   fixed V-cycle from zero. Weighted block Jacobi smooths before and after
   coarse correction. The coarsest level also takes a fixed number of Jacobi
   sweeps. Restriction sums the residuals in each aggregate and prolongation
   adds its correction to every member. The finest level reads and writes the
   supplied vectors directly. The caller serializes calls on an instance and
   refreshes after mutating borrowed coefficients. Graph changes require a new
   hierarchy.

Persistent solver allocations are named under
`Solver/aggregationAMG/level<depth>/`. The original matrix and finest vectors
keep their caller-provided names. Symbolic aggregation is deterministic and
serial. Numeric assembly, diagonal inversion, residuals, smoothing and level
transfers use independent rows with the distributed OpenMP thread policy.
The implementation makes no MPI calls. In a distributed solve the caller
provides the rank-local operator and handles inter-rank coupling.

`solver_aggregation_amg` checks a two-level V-cycle and numeric refresh against
independently assembled dense operators, including nonsymmetric blocks, padded
rows and singular diagonal handling. CUDA builds with AmgX also provide the
manual benchmark target `dnds_benchmark_aggregation_amg`. Its arguments are
grid side length, repeat count and CUDA ordinal. It reports relative residual
and timing for each implementation. Multi-level results may differ because the
aggregate selectors are independent.

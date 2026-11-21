# Runtime Array caller contract

For external numeric tensors, also follow [the DLPack contract](ArrayDLPack.md):
export holds a write lease through the last foreign alias, CUDA release fences
the consumer stream, and the caller joins other streams. Import retains foreign
ownership but cannot discover repeated imports or intercept later external writes.

The Runtime checks access conflicts, shapes, typed payload compatibility and
declared versions at API boundaries. [Optional checked eager execution](CheckedExecution.md)
also validates declared allocation dependencies at submission. Unchecked eager
execution leaves ordering proof with the caller. A successful submission or `IsValid()` result is not proof
that an operator read the intended time level. These obligations apply equally
to inline CPU, worker CPU and CUDA execution.

## Distributed exceptions

MPI `TransferOptions::preflight` defaults to `Inherit`, which takes the policy
from its communication context. A default context uses `Collective`: each
launch agrees local access/route/stream validity before posting requests. The explicit
`AbortOnFailure` option retains these local checks but removes their global
agreement; any local rejection calls `MPI::Abort` immediately. It cannot be
caught for recovery or retried. Binding construction still collectively agrees
this policy, layouts and routing. All ranks must follow the same transfer order,
complete producers and join consumers, including ranks with empty payloads.
Checked streams retain dependency validation; portable MPI still observes the
appropriate device producer stream. Stream-native providers keep their own
protocol. This option neither suppresses numerical reductions nor makes MPI
stream-ordered or graph-capturable.

```cpp
Runtime::TransferOptions options;
options.preflight = Runtime::TransferPreflight::AbortOnFailure;
auto binding = payload.Bind(source, destination,
                            Runtime::TransferTransport::Direct, options);
// All participating ranks build the binding and launch in matching order.
auto operation = binding.Begin();
// Independent owned-only work may execute before Wait().
operation.Wait();
```

Python exposes the same choice with `options = TransferOptions()` followed by
`options.preflight = TransferPreflight.AbortOnFailure`. Neither API
relaxes producer readiness or lease conflicts. A conflicting lease in this
mode terminates MPI rather than returning an exception to the caller.

`MPI::LocalStage` is a preflight around **local work only**. Every rank reaches
its single agreement after that work returns or throws. Its action must not
contain communication, collective I/O, a domain solve or an MPI-containing
callback. Local communicator queries and datatype construction are safe.
Wrapping a complete solver/transfer in a catch followed by `Allreduce` cannot
recover it: a peer may still be in an earlier collective on another communicator.

Local APIs may reject invalid input and release local resources. The buffer
lease check itself throws even when called by a solver. A coordinated UT may
catch an agreed preflight rejection before posting and verify that unused
reservations were canceled.

The Cauchy executable opts its controller into `MPI::SolverFailureScope`.
Preflight and collective operator catches then abort at the failure boundary,
with a local diagnostic, in either agreement mode. The optional checks switch
still controls agreements on success. **In a production solver path, never
catch and recover from an exception if the try contains any collectively
ordered work.** This includes point-to-point MPI exchanges whose calls must
match across ranks, as well as MPI collectives, collective I/O and callbacks
that can enter them. Coordinated UTs may catch and recover when no peer can be
stranded. A production failure must not trigger another error
agreement, MPI request drain, communicator release, worker join or
`MPI_Finalize`. Active transfers also fail fast on unwinding without a solver
scope when a peer could still be waiting. Normal retirement still waits.
The CPU executor captures the solver policy when created. A compute worker
failure terminates the process without calling MPI from the worker.

Outside production solver scope, worker execution retains exceptions until its event/synchronization is observed.
That is a transport of the error to the controller, not numerical recovery.
Observe worker completion before entering dependent MPI. For recoverable numerical
outcomes, agree an explicit status after the local calculation and before the next
communication phase. Cauchy's rejected `StepResult` is such a domain-wide status; arbitrary
`std::exception` is not. Python controllers have the same duty and must abort the
MPI run on unexpected exceptions rather than locally catching and continuing.

## Execution, access and final release

`Buffer::Overlaps`, `SameAddress`, `SameRange` and `IsAligned` inspect address,
extent and device metadata only. They acquire no access lease, read no payload
and do not establish producer readiness. They remain usable while an allocation
has an active writer or pending device work. Overlap also detects independently
imported foreign aliases, but does **not** merge their access controllers:
import each allocation once and derive shared ranges from that allocation.
Only typed access leases may authorize reading or writing the values.

1. Declare every accessed allocation and its intent in `SubmitArrays`. Use
   `Read`, `Write` or `ReadWrite` consistently with the kernel. `Write` does not
   zero storage or guarantee that the kernel initializes every element. A raw
   `Submit` callback must implement equivalent lifetime/access discipline itself.
2. Order conflicting uses on different streams explicitly: record the producer's
   prefix and enqueue `WaitFor` on each consumer. `Record()` covers only work
   already submitted. Scheduled reservations retain/protect storage against
   immediate access. Unchecked reservations do not prove scheduled ordering;
   checked engines reject missing declared dependencies without inferring a join.
   Missing joins in unchecked mode can produce silently stale values or data
   races. A race is outside the API contract, not an expected recovery path.
3. Kernel `ArrayView` values are borrowed pointer/offset descriptors. Keep their
   access lease or submitted operation alive through the actual use, including
   device completion. Do not retain a raw pointer for later independent work.
   Bounds, stencil validity and matching device placement are caller obligations;
   there is no per-element lease check. Default kernel views omit bounds checks;
   `View<true>()` or `View<FixedRows<N>, true>()` enables host/device assertions
   for diagnostic kernels. Bounds checks do not validate stencil semantics or
   cross-stream ordering. Select a typed
   layout when acquiring the view (`View<FixedRows<N>>()`, `TableRows`,
   `PaddedRows`, `CSRRows` or the runtime `GenericRowView`). The policy is checked
   once when the view is created; kernels must retain the declared row assumptions.
   `RowCount()` is the canonical row-count spelling; the compatibility `Rows()`
   alias carries the same value.
   Whole-array views carry `Device` (space and ordinal). Host kernel wrappers
   call `ValidateDevice` / `ValidateViewDevices` once before entering numerical
   loops or launching device work. Row views and spans carry no new tag/check.
   Managed factories establish the payload/offset/size placement invariant;
   projections preserve it. An unbound view is allowed only as an empty optional
   operand. A tag does not prove lifetime, stream readiness or a foreign pointer's
   capacity. Nonempty CUDA allocation imports check pointer memory type/ordinal
   once; other provider obligations remain explicit.
4. Launch device work on the supplied `ExecutionContext` stream and check launch
   errors. Work submitted to another native stream must be joined before the
   enclosing operation publishes completion or relinquishes ownership. Keep
   explicitly borrowed foreign-ABI streams alive through all uses and ordered
   frees. Native execution contexts retain a shared abstract stream resource;
   see [stream ownership](StreamResourceOwnership.md). Healthy engine closure
   retires that resource, so a late lease release performs no native fence.
5. Join **all** independent uses before final release. `ReleaseOn(std::move(a), s)`
   consumes one shared handle; it is not a request to free an allocation with
   other live owners. The final decrement determines the actual release. Releasing
   aliases on different streams is safe only if whichever can be final is ordered
   after all uses. Reference counting supplies ownership, not this ordering proof.
   A moved-from or released Array handle has `IsValid() == false` and null
   `SharedStorage()`. Storage queries, reads/writes, copies and placement
   conversions reject that handle; reset or assign it before reuse. Repeated
   `ReleaseOn` is harmless. Already retained access scopes/aliases keep their
   allocation and shape independently. A moved access scope rejects layout/view
   queries; its destination owns the lease. These checks occur at the host
   access boundary, not inside borrowed CPU/GPU kernel views.
6. Observe errors with `WaitHost`, operation `Wait`/`Test`, or engine `Synchronize`.
   Destructor cleanup is a safety fallback and cannot reliably report errors.
   Check failure before publishing results. Exception recovery is only for
   coordinated tests; collective solver paths abort as described above.
   Do not concurrently mutate the same C++/Python
   Array handle; independently owned aliases do not make value races safe.

```cpp
// Both lanes read temp. Neither read is ordered after the other.
R::SubmitArrays(engine, compute, computeKernel, R::Read(temp), R::Write(result));
R::SubmitArrays(engine, comm, deviceCommKernel, R::Read(temp), R::Write(packet));
comm.WaitFor(compute.Record()); // comm already orders its own use
engine.ReleaseOn(std::move(temp), comm); // final owner may retire on comm
engine.Synchronize();
```

CPU workers obey the same rule: callbacks retain submitted owners through actual
execution; explicit CUDA ordered release submits deallocation in stream order.
A CPU lane is not a dedicated thread. Do not block a bounded worker pool waiting
for callbacks that need its occupied workers. Use stream waits for dependencies.
With zero workers, ready CPU work executes inline without changing the contract.

Normal `SubmitArrays` retains allocation and metadata owners through actual
completion, so dropping a temporary input handle cannot enter a synchronous
CUDA free before the controller has progressed its dependent communication.
Retirement uses the known device stream for an ordered final decrement when
supported. The advanced `WithOrderedHandoff(Read/Write/ReadWrite(array))` operand
opts into dropping those owners at launch instead. Use it only when all remaining
aliases have explicit, correctly joined `ReleaseOn` calls. It preserves early
device free submission without a host completion wait; ordinary RAII of the
last remaining alias is not that protocol. CPU operands keep their normal
execution lifetime in both modes.
`CopyAsyncOptions` offers the same explicit source/destination opt-in for byte
copies; default DMA operands stay retained. Async allocation initialization also
owns its buffer until completion, even when the caller drops an unused buffer.

The engine owns its stream launchers. `ExecutionStream` weakly references that
runtime and cannot keep a closed context usable. `Close()` runs on the creating
controller, stops submissions, polls registered progress and stream prefixes
together, drains final tails, and joins workers. It is idempotent; normal engine
destruction invokes it. Existing completion events remain observable after
close, but recording events, allocating or submitting through its streams fails.
`Synchronize()` uses the same progress loop and leaves the context open. Serialize
both operations with submission; neither invents missing cross-stream edges.

For host resources whose final destructor must run on the controller (for
example a mesh publication retaining an MPI communicator), register an owner
with `engine.RetainUntilSynchronize(owner)` **before** submitting work that
captures it. The engine releases that reference after all accepted work drains,
on the creating controller, at the next `Synchronize()` or `Close()`. This
retention does not submit a kernel or record a CUDA event, and is deliberately
coarser than each operation's numerical lifetime. Rejected submissions can
therefore leave the resource pinned until the next drain. It does not make
resource destruction collective or permit shutdown after `MPI_Finalize`.

This is distinct from scheduled Array access: arrays still use tracked operands
and cross-stream event ordering. Retaining an owner neither grants a read/write
lease nor proves producer completion. Do not retain the engine itself, and do
not release the last unrelated communicator owner on a CPU worker. The queued
CFV polynomial sampler uses this retention for its mesh read guard while its
Array buffers use normal `SubmitArrays` access and retirement.

Backend adapters register `ExecutionProgress` before initiating external work.
The registry retains actual resource owners through nonblocking `Poll()` and
releases its reference on the controller. A callback returns true only after
retirement, or throws only for an already-retired failure; outstanding MPI work
must not be hidden by throwing. Callbacks must not retain the runtime or re-enter
synchronization. `BeginChecked(stream)` and `BeginScheduled(stream)` register
MPI work before preparing/posting, in checked and unchecked execution alike.
Their operation handles share the registered state; dropping one does not wait
or lose the requests. `MPIProgress.Import` exposes that state's completion event
directly, so worker consumers can observe engine-driven completion; its explicit
`Poll`/`Drain` calls still work. The explicit `ExecutionContext` overload retains
its shared abstract resource but does not register request progress with an
engine. Its caller must finish requests and supply dependency ordering. Only
explicit foreign-ABI stream loans retain the separate external-lifetime duty.
Registration itself does not permit buffer access or supply dependency ordering.

Queued CPU waits are notification-driven: a blocked stream consumes no worker,
and workers do not use timed event polling. Backend event providers publish
completion only after retiring the covered accesses, then call
`StreamEvent::State::NotifyCompletion()`. Notification callbacks only wake
executors and must not throw or call MPI/CUDA. A late subscriber is notified
immediately. Providers without native push notification still need controller
progress through `Ready`/`WaitHost`/`Synchronize`; those observations notify the
consumer. This distinction preserves FUNNELED MPI and native CUDA stream
ordering without placing a solver-side polling loop in the compute schedule.

## Halo overlap and MPI

Owned and ghost Arrays have separate allocations. A core stencil may read only
owned values while a halo transfer writes ghost values. Classify core/ring cells
from actual adjacency during setup, and rebuild that classification after changes
to connectivity, partition or local numbering. A combined ArrayPair view does not
prove that a core stencil stays within owned rows.

The main thread can begin a transfer, submit core work, progress MPI, and then
submit ring work after the transfer event. Keep output separate from pending
communication writes. `BeginScheduled` permits later scheduled consumers but does
not finish an unfinished CPU producer. Complete that producer before beginning
MPI. For portable CUDA/MPI, progress the host MPI operation to completion before
asking a CUDA stream to wait on its portable event; an unfinished portable event
is rejected. CUDA-aware MPI is not assumed stream ordered or graph capturable.

Call route construction, payload setup, binding and `Begin` in matching collective
order on all participating ranks, including ranks with zero rows. Use MPI only on
threads permitted by the initialized MPI thread level. `MPIProgress` is explicitly
driven on its creating thread; drain it before `MPI_Finalize`. Do not wait on a
worker dependent on MPI completion while withholding the required host progress.
For a checked engine use `BeginChecked(bridgeStream)` and import the operation
through `MPIProgress`; its portable event carries the access frontier. Full
operation completion retires the conservative checked footprint. Do not mix a
pending conflicting checked reservation with an unchecked `BeginScheduled`.
Use distinct matched tags for concurrently active bindings in a shared
`CommunicationContext`. Transient requests change setup cost, not these rules.

MPI completion polling keeps device unpack/copy as a separate phase and observes
its exact event. It does not fence later work on the stream. Direct CUDA-aware
MPI uses `Completion::FromExternalCompletion(context)` after the corresponding
MPI request completes. This low-level backend proof is not a way for callers to
bypass producer synchronization: the foreign backend must actually establish
completion and memory visibility. Source-send retirement and destination-ready
publication remain distinct.

`CommunicationContext(comm, collectivePreflight=true)` agrees an immutable
safety policy once at construction. Existing callers retain enabled behavior.
With `false`, prepared MPI `Begin`, `BeginScheduled` and `BeginChecked` omit
their safety-only reductions. Local lease, validity, readiness and checked
ordering validation still run. A rejected local launch calls `MPI_Abort`:
rank-asymmetric exception recovery without agreement could strand peers or
publish incomplete state. Do not catch-and-continue such a distributed failure.
Required routing/count/schema/binding-control and numerical collectives remain.
Policy replacement requires a coordinated, drained preparation boundary.

Communicator lifetime is independent of that policy. `Adopt(comm)` consumes a
new split/duplicated communicator and clears the input handle; it rejects
predefined WORLD/SELF. `Duplicate(comm)` explicitly creates an isolated MPI
namespace (the raw constructor retains this compatibility behavior).
`WithPreflight(enabled)` is a collective setup operation which shares the
existing communicator **and tag reservations**, with a separately immutable
policy. It does not create an independent message namespace. Retained contexts
must be released on an MPI-permitted thread after requests drain and before
MPI finalization. Entity descriptors attached with `WithCommunication` use
that retained communicator for later collectives; raw descriptors still require
the caller to keep their communicator alive.

`TransferGroup(context, bindings)` is an opt-in CPU persistent-MPI assessment
path. It retains the exact member bindings and separate messages/tags; it does
not coalesce physical fields. All ranks pass the same ordered member tags at
construction. Members must use that exact shared context and host Direct or
HostPacked transport. Device/provider and scheduled/checked-stream launches
are rejected rather than silently falling back. Ordinary separate starts remain
the production default.

Group `Begin()` acquires every input/output before one optional agreement, then
posts each member's prepared requests. If enabled validation rejects, every
successful unmodified reservation is cancelled before any payload is posted.
Disabled preflight retains local validation and the normal fail-fast contract.
Keep producer/consumer accesses complete and serialize the group and all member
bindings. Returned operations preserve per-field destination readiness versus
full send completion; drain all of them before starting the group again.

Pass the same policy context to `EntitySpace::BuildRowTransferPlan(context)`
or a Runtime route constructor. Contexts selected during binding must have the
same resolved mode on every rank; the existing binding-control agreement
checks this before posting. CFV exposes `ConstructMetrics(...,
collectivePreflight)` and shares that context through preparation and halos.
Python uses `collective_preflight` in both constructors. The lightweight
`ExchangePlan` accepts a context too, but still prepares payload protocol
metadata per call; zero launch reductions does not mean zero setup collectives.

Contexts support `MPI_COMM_SELF` and reordered split communicators. When a
communicator does not carry `MPI_TAG_UB`, tag reservation uses the portable
range 0–32767; it does not assume that every communicator inherits world-model
environment attributes. This is the minimum range required by the
[MPI standard](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node245.htm).

`WaitDestinationReady()` makes the destination readable; the source can still be
in use until full completion. `Wait()` or successful `Test()` completes both.
Retain the operation or import it into a live `MPIProgress` owner. Check errors
explicitly, then coordinate recovery across ranks.

For the optional [NCCL payload provider](NCCLTransfer.md), setup remains collective
but hot `Enqueue(engine, stream)` uses CUDA stream order through pack/NCCL/unpack.
Record the transfer stream after unpack and wait on it before dependent kernels.
No host completion wait is needed between these phases. The caller must order
producer readiness, consumer readiness, private scratch reuse and per-peer call
order across bindings sharing a provider. Scheduled reservations are not a
dependency proof. Immediate access still needs observed completion. Observe
provider asynchronous errors for manual operations. Managed Enqueue registers a
controller-held provider owner and polls errors during runtime progress; a
transport failure aborts without draining unmatched work. A CUDA prefix alone
cannot diagnose a missing remote submission. Default MPI behavior is unchanged.

## Schemas, reductions and redistribution

`BindMPIAllreduce(context, source, destination, op)` binds two compact host
tables or a single in-place table to a matched MPI sum, min or max. The
binding shares the supplied communication context. The caller chooses the
ordering namespace during setup and supplies one CPU communication lane per
rank. A different host collective on that communicator must follow completion
of queued reductions. Setup checks the shape, scalar, operation and alias
mode across ranks. The bound count must fit the MPI integer count.

On submission, the engine reserves the Arrays, observes the producer prefix
and posts `MPI_Iallreduce` from the binding's MPI controller. The lane waits for
request completion before publishing its destination write. The returned
prefix can order another CPU lane or be observed by its controller. Dropping
the binding or Array handles after submission leaves the accepted operation
intact. Portable progress requires the controller to enter event observation
or engine progress. For a CUDA caller with host staging, observe the MPI
completion on the controller before importing it and queuing the return copy.
Use NCCL for a native device collective. Open MPI's CUDA-awareness query
alone does not establish support for nonblocking device-buffer reductions.
[Open MPI 5.0.x CUDA operation support](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html)

An `ElementSchema` describes MPI fields with an extent equal to the POD element
stride. Its typemap must be valid and nonoverlapping within an element, with the
same signature and semantics on communicating ranks. Arithmetic and Python dtype
schemas carry checked type tags. Float32/64 and signed/unsigned 8/16/32/64-bit
integer schemas use the same names in C++ and Python, so their payloads and
reducers interoperate. Explicit untagged MPI schemas, raw Buffer binds and native
type-erased ArrayStorage imports remain caller promises: equal byte
size cannot establish a semantic type. Pointer/owning-object bytes are not
portable payloads. Import foreign storage once with a genuine shared owner; derive
aliases/ranges from that Allocation so they share one access controller.

Collision-free receives use copy. Collisions require an explicit reducer, using
receiver staging; unordered multiple writes are rejected. Custom reducers must
have the same named meaning and initialization policy across ranks, read only
transmitted fields, and obey the supplied execution context. Do not retain callback
pointers or call unmatched MPI collectives. Floating-point reductions need not be
bitwise invariant under repartitioning. `RowTransferPlan::Known` requires matching
send/receive row semantics: counts alone cannot detect a wrong permutation.

A route is immutable correspondence, a payload holds layouts, and a binding pins
specific allocations. Resetting an Array or replacing a mesh does not retarget
them. Retained views/aliases continue to refer to the old storage. After migration,
publish returned Arrays, rebuild layout-dependent payloads/bindings, and refresh
adjacency and core/ring classifications. Attach relevant mesh version stamps and
commit mutations; version checking detects stale declared dependencies but is not
a lock. Drain old operations before mutating their dependencies.

`DynamicRowTransfer` pins its source and privately builds a destination preserving
the local table width/stride, padded capacity or CSR family by default;
construction and transport calls must follow matched participant order. Commit publishes
only after successful payload completion. Multiple fields are separate operations:
the application must commit and validate the complete field bundle before
publishing a new mesh epoch. There is no atomic multi-field transaction.
Incoming rows that do not fit the receiving rank's declared shape reject
collectively. Request `TransferLayout::Compact()` explicitly for compact scratch.

## Shape editing, Python and restart

Published row metadata is shared and immutable. `RowBuilder` permits independent
workers to edit distinct rows; callers join them before `Commit`. Resizing a row
invalidates its borrowed pointer. `ExclusiveRows` requires unique storage and no
active access; it blocks sharing until publication. Abandoning a destructive edit
invalidates the Array rather than rolling back its values. Do not publish a draft
or keep row pointers through resize/commit.

Python slices and element proxies retain ArrayStorage (values, metadata and access
controller), including through parent reset/destruction. They do not hold a
permanent lease. Each access acquires a lease; pending asynchronous writes must be
completed before synchronous indexing or NumPy snapshots. NumPy results and POD
class-value reads are copies. Mutate through a write lease, tracked slice assignment
or element proxy; changing a returned snapshot does not change the Array.

The native Array I/O adapter uses the existing Serializer and transfers logical
rows directly from/to the retained host allocation without a payload-sized
packing buffer. Device buffers are rejected until a direct I/O backend exists;
an application can explicitly make a host checkpoint copy outside the adapter.
Complete asynchronous producers first.
Parallel serializers require matching dataset/schema call order; per-rank
serializers are independent. Register POD scalar fields once with ElementSchema,
which also supplies MPI and Python's element description. I/O schema strings may
only confirm that identity. The file-wide byte-order flag controls in-place
scalar-word reversal; registered native layout/extent must still agree. Padding
inside objects is native ABI data and is not byte-swapped. Dynamic file row sizes
must satisfy the requested table/padded/CSR policy; no static type-name matching
is used. Reopen v2 readers for another slice of cached row-start metadata.
Checkpoint the mesh IDs, state time level and all related fields together
at the application level; individual Array writes are not an atomic restart.

## Compound-operator entry validation

`Buffer::ValidateReservation(mode, device, checked)` performs the admission
checks of a future scheduled access without acquiring it, changing a value or
mutation generation, or waiting for existing work. It rejects conflicting
immediate host leases and mixed checked/unchecked access policies; pending
operations under the same policy remain valid. It is not a stream dependency
proof, a readiness check, or permission to dereference a pointer.

A compound collective operator can include this check in its existing entry
preflight before launching halos or writing its first output field. Cauchy FV
does this for every physical residual field. This prevents a rank-local output
lease exception from escaping late in submission while peers reach a different
collective. Actual operators still reserve their operands and validate dependency
ordering. Callers must serialize control-thread mutation/submission. Checked UT
callers can reject coherently; production solver exception scopes still abort
without attempting distributed recovery. No additional MPI agreement is added.

## What tests establish

Tests exercise immediate conflict errors, stale version/schema rejection,
cross-rank validation, failed-write invalidation, actual overlap, and independent
full-value oracles after joins. Missing-join tests deliberately finish a read
before opening a gated writer, demonstrating silent stale data **without** invoking
a data race. Bounds violations, invalid external pointers and genuinely racing
kernels remain caller errors; tests must not rely on undefined behavior to pass.

See [Array runtime](ArrayRuntime.md), [eager execution](EagerExecution.md),
[allocation lifetime](ArrayAllocationLifetime.md), [MPI transfers](ArrayTransientMPI.md)
and [row editing](ArrayRowEditing.md) for interfaces and ownership diagrams.

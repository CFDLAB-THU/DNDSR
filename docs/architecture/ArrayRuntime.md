# Runtime Array and gradual Geom migration

`DNDS::Runtime::Array<T>` is the runtime storage handle. `DNDS::Array`
continues to provide its existing interface and deep-copy semantics; its host
allocation uses the common Buffer facility. Euler, CFV, ArrayPair and
ArrayTransformer can migrate incrementally. This runtime adds no PETSc, hypre,
LibTorch, Legion or mesh-library dependency.

## Responsibilities

The [caller contract](RuntimeCallerContract.md) specifies required stream joins,
raw-view lifetime, MPI participation, migration rebinding and checkpoint discipline.

| Component | Owns |
|---|---|
| Allocation | Shared native/foreign owner, device, identity, value generation and access controller |
| Buffer | A bounded byte range retaining its Allocation; overlapping ranges share exclusion |
| Allocator | Optional allocation/deallocation policy, including pools or pinned host storage |
| AllocationResource | Backend payload with explicit ordered-release capability and conservative RAII fallback |
| RowLayout | Immutable table, variable-length padded, or CSR row shape in element units |
| MetadataBuffer | Shared Buffer containing published host offsets, exposed only for reading |
| ArrayStorage | Shared record with a guarded publication boundary for values and immutable row metadata |
| Runtime::Array | Typed storage handle carrying only element type `T` |
| PythonArray | Runtime dtype/POD codec and the same ArrayStorage as C++ |
| ArrayAccess | Move-only read/write lease retaining allocation and layout; typed views are acquired from it |
| ArrayView / GenericRowView | Borrowed typed or fully dynamic row view; no tracking per element |
| RowBuilder / ExclusiveRows | Unpublished CPU row construction or unique shape edit; Commit publishes metadata |
| ArrayPair | Separate owned and ghost Arrays; optional combined read indexing |
| RowTransferPlan | Immutable distributed row relation, reversible and reusable across fields |
| ElementSchema | Shared immutable scalar-word record, stable identity, native layout and lazy MPI datatype; used by I/O and Python too |
| PayloadPlan | Row lengths, relative MPI datatypes, coalesced segments and contribution groups |
| CommunicationContext | Shared communicator and explicit matched-tag reservations |
| TransferBinding | Allocation identities, transport/reducer/workspace and transient or persistent request policy |
| TransferOperation | Active requests and leases; separate destination-ready and all-complete boundaries |
| DynamicRowTransfer | Pinned source and private storage-preserving destination; collective publication after payload success |
| EagerEngine / StreamLauncher | Common ordered submission contract with inline/worker CPU and native CUDA providers |
| MPIProgress | Main-thread MPI retirement and explicit events observed by eager streams |
| ArrayIO | Schema-tagged row checkpoint adapter over the existing parallel serializer |
| VersionTracker | Selective mesh dependency generations and explicit mutation scopes |
| Geom::NativeRowTransfer / RegisterArray | Bridge from existing mesh permutation correspondence to native Arrays |

`Field` and `ExchangePlan` remain compatibility helpers for the initial prototype.
ExchangePlan now delegates to the same row-transfer implementation. New code
should retain route, payload and binding separately to reuse setup.

The detailed contracts and ownership graphs are in
[row editing](ArrayRowEditing.md), [eager execution](EagerExecution.md),
[allocation lifetime](ArrayAllocationLifetime.md), and
[transient MPI](ArrayTransientMPI.md). The unstructured mesh now owns Runtime
arrays; legacy solver fields continue to use their historical containers.
Runtime Array itself does not depend on Geom or the solver.
The [2026-10-22 audit](../dev/array_runtime_audit_2026-10-22.md) summarizes
coverage, reviewed contracts and the retained-transfer-error correction.

## Element types and device dispatch

The [Euler workload and overlap audit](../dev/array_runtime_scenarios_2026-10-22.md)
maps production data patterns to CPU/MPI/CUDA tests, including detected type
mismatches and deliberately unchecked eager ordering mistakes.

The handle is templated only on element type. A compile-time layout belongs to
the borrowed view acquired from its lease, so table, padded, CSR and
fixed-size policies share one storage type. For example:

```cpp
using namespace DNDS::Runtime;
struct CellInfo { int64_t globalId; int32_t flags; double volume; };
Array<double> state(RowLayout::Dense(100, 5));
Array<int64_t> ids(RowLayout::Dense(100, 1));
Array<CellInfo> metadata(RowLayout::Dense(100, 1));
```

Storage allocation, identity, access tracking and byte copying live in the
non-templated Buffer layer. Host/CUDA dispatch uses Device and ExecutionContext
at operation boundaries. Element loads through a kernel view are ordinary
typed pointer arithmetic; they do not branch on a runtime dtype or device.
Numerical kernels and reducer implementations may separately specialize on T
and backend, without templating the entire communication engine.

T must be trivially copyable and standard layout, with alignment at most 64
bytes. Allocation initializes bytes; it does not run per-element constructors
or destructors. Custom data structs are supported, while owning objects such
as std::string/std::vector are not. Pointer values also do not describe portable
MPI, checkpoint or device payloads. A custom MPI payload supplies ElementSchema
with a matching extent; a custom reduction supplies the appropriate callback.

Python uses a runtime dtype/POD codec over the exact native ArrayStorage record.
The public `Runtime/Array_bind.hpp` converter shares values, row metadata and
the controller with any bound `Array<T>`; no byte-Array intermediary or
width-specific Python class is needed. Numeric and structured POD dtypes are
supported. Bound POD classes can register a class-value codec once per type,
independent of Array widths. Object-reference NumPy dtypes remain rejected.
See [Python ownership and binding](ArrayPythonOwnership.md) for class/object
diagrams, the generic converter and tracked POD field editing.

`FixedRows<N>` adds a C++ view-width constraint; it does not change the handle
type and does not require a new Python
Array class. `Dense(rows,width,stride)` is a uniform-width strided table, not
the legacy padded layout with variable row lengths and uniform capacity.
`Padded(lengths,capacity)` now provides that variable-size representation;
CSR remains available for packed variable rows. See [row construction and
exclusive edits](ArrayRowEditing.md) for Commit, retained snapshots and Python.
Existing legacy Array layouts remain available.

The access policy appears only at view acquisition, for example
`read.View<TableRows>()`, `read.View<FixedRows<5>>()` or
`read.View<CSRRows>()`. It propagates through the returned
`ArrayView<T,...,Policy>`, pair view and `RowSpan<T,...,Width,Capacity>`.
Unchecked views select indexing with `if constexpr`, without per-element
layout detection or ownership operations:

| Policy | Row pointer | Logical row size |
|---|---|---|
| `TableRows` | row times runtime stride | Runtime uniform width |
| `FixedRows<N>` / `FixedRows<N,S>` | row times runtime / static stride | Compile-time N |
| `PaddedRows` | row times runtime capacity | Per-row size buffer |
| `StaticMaxRows<C>` | row times compile-time C | Per-row size, static capacity C |
| `CSRRows` | Per-row offset buffer | Offset difference |

`GenericRowView<T>` is the identity alias of `ArrayView<T, false, RowLayout>`.
It accepts table, padded and CSR storage through one runtime layout and is the
fallback for setup, serialization, Python/debug adapters and generic
algorithms. `DynamicView()` converts a typed view without copying payloads.
Static assumptions are checked once when a typed view is acquired, including
empty arrays; raw row views retain no ownership and need the enclosing lease.
Python remains one dynamic class and needs no width/capacity instantiation
list. There are no Eigen maps or multidimensional numerical views in this API.

Every layout supports `view[iRow]`, returning a device-callable `RowSpan`, and
the development-checked `view(iRow, iCol)`. `RowCount()` is the canonical row
count method; `Rows()` remains temporarily as a compatibility spelling.

`RowSpan` with fully static width and component stride contains only the
borrowed pointer. Other rows retain their general pointer/size/stride descriptor:
shrinking a partial-static descriptor also changes its calling convention and
can hurt generic kernels. Capacity remains a compile-time trait. This does not
change payload layout, ownership or leases.
The descriptors remain standard-layout, trivially copyable and device-callable.
Their internal C++ ABI changes when this representation changes: rebuild all
dependent objects together rather than mixing old and new libraries. Default
null rows still report their static width/stride, as before; null does not
provide a valid payload to dereference.

## Ownership, access and failure

Array copy/alias shares allocation and access state. `Clone()` makes independent
values; `To(device)` makes an explicit placement copy. `CopyFrom` returns an
operation for asynchronous copies. `Reset` replaces the handle's allocation and
layout; existing aliases, bindings and leases retain the previous ones.
`FromStorage` can attach a typed row layout to an owned Buffer, retaining its
controller; it does not wrap unmanaged pointers.

A read shares access with other reads. A write excludes both reads and writes
to the same allocation, including differently shaped aliases. A successful
write publishes one generation on completion. A failing write scope or
explicit `Fail()` marks the allocation unreadable; replace it with `Reset`
to recover. An unissued collective-transfer preflight is cancelled without
publishing a value change. There are no row-mask checks in production views.

```cpp
using namespace DNDS::Runtime;
Array<double> u(RowLayout::Dense(nCells, 5));
{
    auto write = u.Write();
    auto view = write.View<FixedRows<5>>(); // unchecked kernel view
    for (DNDS::index iRow = 0; iRow < view.RowCount(); ++iRow)
        view(iRow, 0) = 1;
}
auto independent = u.Clone();
auto gpu = u.To(Device::CUDA(0));
auto context = ExecutionContext::CUDA(0, stream);
auto read = gpu.Read(context);
// launch<<<..., stream>>>(read.View());
read.Complete(Completion::Record(context));
read.Wait();
```

`At` checks host placement and bounds; `View<true>()` selects a separate
development checking policy. Raw views and `unsafe_address` are borrowed:
a caller must not retain/use them beyond the lease or after submission.
C++ cannot revoke a copied raw pointer.

CUDA leases retain storage until the recorded event completes. Without an
event, release fences that lease's stream. Streams are borrowed and must remain
alive. Allocations are initialized before publication. Host/CUDA copies are
explicit; cross-device ordinal copies are currently rejected. Python debugging
indexing is a deliberate convenience exception: it synchronizes selected
device values with host memory.

The device descriptor and execution context are distinct from Array shape,
routing and physics. New accelerators require implementing allocation,
completion/fence and copy dispatch in the backend layer and supplying their
kernels; no per-element virtual dispatch or mandatory host mirror is required.
Currently executable backends are Host and CUDA. Allocator policies must return
zero-initialized correctly placed storage and preserve ownership until the
last lease ends.

Queued operators instead declare arrays with `Runtime::ReadAs`/`WriteAs` and
`SubmitArrays` (or `SubmitArrayBatch`). Include index, stencil and metric arrays,
not only solution fields. A captured immediate read lease still has the safe
destructor fence above and can serialize submission unnecessarily. Scheduled
operands retain allocation and row metadata through the launcher's completion,
exclude conflicting immediate mutation and publish the same terminal error.
Access retirement itself adds no per-operand event or host fence; final resource
release still follows the allocator/provider contract.

`MetadataBuffer` is immutable after its completed construction/placement.
`MetadataArrayView(metadata, device)` supplies a residency-checked borrowed view
without an immediate lease; capture the metadata owner until completion. This
does not permit untracked reads of ordinary mutable Arrays. Immediate copy
`Completion` aliases share a cached terminal success/error, so repeated `Wait`
and `Ready` observers do not repeat the native synchronization after completion.

## Owned/ghost overlap

Owned and ghost remain separate allocations. A halo operation takes an owned
read lease and a ghost write lease. Core work receives only the owned view and
writes to an independent residual Array. `WaitDestinationReady()` drains
receives, combines contributions if needed and releases the ghost write lease.
Ring work can then read the ghost Array. `Wait()` also drains outgoing requests
and releases the owned send lease. An independent residual can retain one
write lease across core and ring work.

The application must classify core rows from its complete input stencil before
launch and invalidate that classification after connectivity/stencil changes.
A core view has no ghost pointer. This phase does not introduce a solver kernel
registry or claim automatic proof of arbitrary kernel addressing. Raw adjacency
mutation still requires explicit invalidation. General suballocation regional
leases are unnecessary for this owned/ghost pattern and remain deferred.

## MPI routing and payloads

All plan setup, binding and `Begin` calls are collective on the route
communicator and must be called in the same order. Structural orchestration is
serialized; this is not a multi-threaded distributed lock manager. By default
a binding has its own communicator and persistent requests. TransferOptions
can select transient requests and a shared context with an explicit matched
tag; see [the transfer policy guide](ArrayTransientMPI.md). Only one operation may be
active on a binding. Multiple bindings can share the same immutable route and
payload while retaining independent allocations and requests.

The native-array communication inventory is:

| Work | Construction | Payload transport |
|---|---|---|
| generic owner-request discovery | `Alltoall` counts and `Alltoallv` source-row IDs | Direct P2P after route construction |
| known halo/ghost route | precomputed peer row lists; Alltoall validates peer counts, no row-ID discovery | Direct P2P |
| transient field/relation transfer | retained `RowTransferPlan` and `PayloadPlan` | `Isend`/`Irecv`, one h-indexed datatype message per peer |
| persistent solver ring | requests initialized once per peer | `Startall` with one direct h-indexed message per peer |
| explicit host packing | host staging buffers | packed P2P through the same binding state machine |
| ragged row lengths | `Alltoallv` of row lengths | metadata only |
| alias agreement | compact `Allreduce(MPI_BOR)` or scalar min/max for a large descriptor registry | metadata only |
| reduction destination | contribution workspace and host/device/custom reducer | direct P2P contribution transfer plus local combine |

Direct datatypes merge adjacent physical ranges. Sparse rows remain multiple
h-indexed segments within one message; they do not create one MPI message per
segment. The payload path never switches to all-to-all. Metadata discovery,
schema agreement and reductions are separate collective mechanisms.

`Communication/IdRowTransfer.hpp::BuildRowTransferById` provides cold persistent-
ID lookup for restart/migration. It builds a distributed hash directory using
fixed-record `SparseBatchExchange` messages, compares full IDs, and returns an
ordinary `RowTransferPlan`. Source IDs must be globally unique. Gathering may
repeat target IDs; optional bijection mode requires exact one-to-one coverage.
Missing IDs, duplicates and rank-inconsistent policies reject collectively.
There is no allocation indexed by the global ID range or a global row bitmap;
directory memory follows received records. Numerical payloads subsequently use
the normal direct/transient or persistent binding path, without an extra lookup
packing layer. Keep ID producers host-visible and retain the communicator.
The plan snapshots row correspondence: callers rebuild after identity/order
changes, or supply the owning publication's version dependencies. Tests include
arbitrary rank distribution/local order, empty owners, large sparse IDs,
SoA/AoSoA payloads and negative correspondence cases.

```cpp
RowTransferPlan route(mpi.comm, owned.RowCount(), requests.size(), requests);
PayloadPlan payload(route, owned.Layout(), ElementSchema::Arithmetic<double>());
Array<double> ghost(payload.DestinationLayout()); // Preserve family and stride/capacity.
auto binding = payload.Bind(owned, ghost); // Direct is the default
auto transfer = binding.Begin();
// owned.Read().View() can feed a core kernel here
transfer.WaitDestinationReady();
// ghost.Read() is now permitted
transfer.Wait();
```

Each RowRequest names a source rank/row and a local destination row.
`Reverse()` transposes that relation. Ragged widths are exchanged at setup,
including empty rows and empty ranks. Copy destinations must be disjoint;
collisions are detected before starting requests. Shape replacement requires
a new payload/binding; an old binding intentionally continues to target its
retained allocations.

Inferred destinations preserve the receiving rank's local source layout family:
table logical width and physical stride, padded capacity with received logical
sizes, or compact CSR. Empty ranks retain their declared table/padded shape.
Serial mesh ingress communicates the reader's storage prototype to empty ranks;
zero owned rows never implies zero table width. A default zero-width handle is
not a substitute for an empty field with a known common width.
This is the default for generic entity halos, named/canonical adjacency ghost
publication, registered field migration and `DynamicRowTransfer`. Fixed-width
payloads construct tables directly without temporary CSR offsets or compaction.
Only logical values travel through MPI; stride/capacity padding is untouched.

Pass `TransferLayout::Compact()` as the inferred payload's fourth argument (or
the route shape method's second argument) to request compact setup scratch.
Rank-local capacities/strides can differ; each incoming row must fit. Default
preservation never silently changes family when it cannot fit: setup rejects
collectively before binding. For sparse routes, unrequested table slots keep
their declared width and their values are not written; unrequested padded/CSR
rows have zero logical length. Callers initialize any holes they intend to read.

The supplied MPI datatype's **extent** is the element stride, and must match
the stored POD size. Its typemap must fit inside that extent. Schema identity
is an application assertion of matching type signatures across ranks, not
runtime reflection of arbitrary C++ structs. The caller supplies a valid
non-overlapping element typemap. Retained datatype ownership survives freeing
the caller's MPI handle.

Direct transport sends/receives derived datatypes against actual storage.
Adjacent segments merge; physically reordering sender rows outside MPI can
therefore produce a single segment. This is distinct from packaging.
`TransferTransport::HostPacked` explicitly uses MPI_Pack/Unpack and host
workspace, with host/device staging when necessary. It is a portable fallback,
not an automatic policy switch. Its host workspace is currently pageable;
pinned/device packing optimizations are not claimed. Peer packed messages are
limited by MPI's int packing API and rejected when too large.

Direct CUDA communication requires the MPI CUDA capability query to succeed.
Portable producer synchronization happens before MPI starts; receiver
completion happens before consuming kernels launch. This is not stream-native
MPI, GPUDirect performance evidence, or multi-node validation.

## Reductions

Default Copy rejects fan-in. Explicit reducers receive disjoint typed
contributions into operation workspace and combine at the receiver. Sum, min,
max and integer OR/XOR/AND have CPU and CUDA implementations. Custom reducers
can supply CPU and device callbacks; applications own their semantic identity.
Payload and binding own communication details; callbacks own no MPI requests.

`DestinationInit::Preserve` accumulates into existing values.
`Identity` initializes each touched destination once; unrequested rows remain
unchanged. Zero-width rows contribute no operations. Contributor order is
stable for a fixed route (source rank, then edge order). Floating min/max use
number-preferring fmin/fmax behavior. Integer sum wraps at its type width.
No partition-independent floating-point bitwise reproducibility is promised.

A failing custom reducer drains the requests, poisons partial output and
releases access. Explicit Wait reports the error. Destruction is a blocking
fallback, not an error-reporting API. MPI process/transport failure recovery
is outside this phase.

## Checkpoints and shared element schemas

`WriteArray` / `ReadArray<T>` use the existing Serializer backend through
native direct-memory row selections. No legacy Array or whole-payload packing
allocation is made. H5 uses memory hyperslabs over the source or final output
allocation; JSON uses its required document representation. Metadata scratch
is proportional to rows/ranks. Device buffers are rejected before payload I/O
until a direct device-capable backend is supplied: no hidden CUDA staging.
Applications may explicitly copy a checkpoint to host outside this adapter.

One `ElementSchema` record describes scalar fields, word sizes/counts, offsets,
object extent/alignment and stable identity. Its lazy MPI datatype is derived
from those fields. `RegisterArrayElement<T>` attaches Python class load/store
functions to the same registered record and derives its structured dtype.
C++ registration works before MPI initialization and does not require importing
Python. Arbitrary custom PODs need registration or an `ElementSchemaProvider<T>`;
free I/O strings cannot relabel their bytes. An optional `cpp_type_name` is
saved for diagnostics, never compatibility matching. Legacy explicit MPI
constructors remain available with their caller-supplied type/meaning contract.

```cpp
struct Tag { int64_t id; double weight; };
auto schema = ElementSchema::Register<Tag>(ElementSchema::Record(
    "example.Tag.v1", sizeof(Tag),
    {{"id", offsetof(Tag, id), ElementScalar::I64},
     {"weight", offsetof(Tag, weight), ElementScalar::F64}},
    "example::Tag", alignof(Tag)));
// schema.Type(), WriteArray(tags,...), and RegisterArrayElement(pyClass,schema)
// consume this same immutable descriptor. No width-specific binding is needed.
```

Version 3 saves dynamic logical shape independently of the C++ policy:

```text
/ @runtime_array_byte_order=0                 (0 little, 1 big; file-wide)
  field/ @runtime_array_format=3, @element_bytes, @element_schema
         @element_definition, @cpp_type_name, @writer_ranks
         @row_encoding=0, @row_width=W        (dense; even when N=0)
         @rows_global=N, @elements_global=N*W
    row_counts        int64[writer ranks]     (one count in each JSON file)
    values            uint8[logical payload]  (native object bytes, rows packed on disk)
    values::rank_offsets int64[P+1]           (H5 backend metadata)
```

Variable rows use `row_encoding=1`, `row_width=-1` and
`row_offsets:int64[N+1]` in **element units**. There are no row offsets for dense
records. Physical padding between rows is omitted. A generic dense read returns
a table; a generic variable-row read returns CSR. Requested table/padded/CSR
policies reconstruct their chosen allocation directly after dynamic row-size
validation. `FixedRows<N>` checks actual logical width, including empty dense
files; it never compares C++ template signatures. `StaticMaxRows<C>` checks
that each logical row fits C and reads directly into that padded destination.

All arrays in a file use its one native-byte-order flag. If it differs from
the reader, each declared scalar word is reversed in place in the final host
allocation, including repeated POD fields; object padding is untouched. The
schema definition must agree on native word layout/extent. This handles endian
changes for that representation, not unrelated C++ ABIs, float representations,
pointer members or schema evolution. Opaque legacy MPI layouts reject endian
conversion rather than treating an entire struct as one word. Writers reject
appending native bytes with a different file flag.

Version-2 byte-CSR Runtime archives remain readable through the native row
adapter, including old JSON base64/zlib payloads streamed into the final rows.
They lack writer byte-order/static-width metadata and retain their documented
same-representation/empty-shape limits. An explicit old schema string is accepted
only by a v2 reader for historical application labels; new writes and v3 reads
require the registered element identity. Field association/history metadata is
stored separately from the element schema. Legacy `array_sig`, `array_type` and
`row_size_dynamic` belong to that old nested byte-CSR codec; they do not decide
new typed Array compatibility. Unrelated legacy typed Array archives use their
existing dedicated adapters, not automatic Runtime schema guessing.

Parallel H5 supports matching writer ranks, EvenSplit with a different MPI
size, and explicit contiguous global row windows. JSON reads its saved local
partition. Match collective names/schema/call order and finish producers first;
source leases remain held through writes, and reads publish only complete owned
storage. Dynamic partitions and policy failures are agreed before payload calls.
Version-2 shared row-start caches still require reopening for another slice;
version-3 metadata uses ordinary independent direct windows. Checkpoint mesh IDs,
field epochs and all related arrays coherently at the application level;
individual Array writes are not an atomic restart.

## Python debugging and production access

```python
import numpy as np
from DNDSR.DNDS import Runtime as R

a = R.Array(R.RowLayout.dense(4, 3), np.dtype("float64"))
a[:] = np.arange(12).reshape(4, 3)
column = a[:, 1]             # live controlled view of the same allocation
column[1:3] = [-5, -6]
assert a[1, 1] == -5
snapshot = column.numpy()   # independent NumPy copy

gpu = a.to(R.Device.cuda(0))
gpu[2, 1] = 17              # synchronized host-to-device element write
value = gpu[2, 1]           # synchronized device-to-host read
with gpu.write() as edit:   # batching control for debugging is also available
    edit[0, 0] = 3
```

Basic integer, negative, slice and ellipsis indexing is supported. Row/column
slices are live ArrayView objects, so `a[0][1] = value` updates the Array.
Views retain the original shared ArrayStorage after a parent Reset or
destruction, including logical metadata, CUDA CSR offsets and the controller.
Nested slices retain their own flat handle to that record. An idle view holds
no lease; each read/write acquires one. Fancy indexing, boolean masks and new
axes are omitted.
Ragged indexing supports a single row or a rectangle of equal-width rows.
These debugging views are not row-transfer payloads; SoA remains separate
Arrays sharing a route.

GPU indexing synchronizes and copies the selected elements; it is intended
for debugging, not performance kernels. Explicit live leases and transfer
operations still enforce conflicts, including on GPU. A NumPy snapshot cannot
mutate the Array. There is no untracked writable buffer export or implicit
DLPack bridge. Kernel integrations use a retained lease and explicit
`unsafe_address()`/completion with the usual borrowed-pointer responsibility.

MPIInfo, RowRequest, RowTransferPlan, ElementSchema (including mpi4py
`datatype.py2f()`), PayloadPlan, TransferBinding/Operation, reducers, device
copies and serializer operations are exposed. Python custom reducers receive
independent NumPy contribution/current rows and return the combined row.
ArrayPair preserves separate owned/ghost controllers.

## Geom and legacy compatibility

Geom continues to execute its existing local/distributed permutation machinery.
`NativeRowTransfer` extracts that exact correspondence for reuse.
`RegisterArray` migrates an owned dense/fixed/ragged native Array and publishes
the replacement only after its transfer finishes. Ghosts and connectivity
references are rebuilt/remapped by the appropriate Geom caller. Stable IDs and
histories are ordinary registered fields; opaque field values are not
automatically interpreted as indices.

Mesh mutation/version scopes remain at the guarded reorder boundaries.
Other legacy mesh mutations and raw legacy Array writes are not automatically
tracked. Preserve existing interfaces during migration and explicitly guard
new callers. In-place mesh mutation is allowed under its version scope;
field migration retains old allocations through the transition.

Validation commands and actual results belong in
[the phase report](../dev/array_geom_native_phase_2026-10-21.md).

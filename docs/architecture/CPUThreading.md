# CPU threading and setup preprocessing

DNDS uses two explicit OpenMP controls. `OMP_NUM_THREADS` is the ordinary
OpenMP runtime team size, including reader-owned and pre-distribution work.
`DNDS_DIST_OMP_NUM_THREADS` is the per-rank team requested by native DNDS
row kernels after distributed ownership exists. Native DNDS algebra, Geom,
CFV and Cauchy share the latter policy.
Enable **both** `DNDS_USE_OMP=ON` and `DNDS_DIST_MT_USE_OMP=ON` when configuring.
The first switch alone does not enable native row helpers. Run, for example,
with `OMP_NUM_THREADS=16 DNDS_DIST_OMP_NUM_THREADS=16`, BLAS/MKL/OpenBLAS
threads set to one, sufficient allocated CPU cores per rank and no MPI CPU
binding. Ordinary pure-MPI runs continue to use one thread per rank.

CUDA qualification runs use 16 setup threads, including pre-distribution work,
unless reproducing a historical comparison that explicitly fixes OMP1. Do not
mix those reorder/setup policies in a matched GPU performance comparison.

Both values are read once per process; launchers must set each intended policy
explicitly. Cauchy starts in the ordinary `OMP_NUM_THREADS` context. For each
file-backed domain it retains that context through reader-owned import, serial
partitioning and distribution, then switches to `DNDS_DIST_OMP_NUM_THREADS`
before topology, reorder, CFV setup and state initialization. A later colocated
domain restores the reader context for its own ingress. Factory-injected meshes
enter the distributed context directly. Eigen remains single-threaded in both
contexts.

CPU-inline and CPU-worker launchers use the same explicit native-helper policy.
Helpers called inside an active team run their inner loops serially. Direct
OpenMP regions without an explicit `num_threads` clause follow the current
process OpenMP control variable; native row helpers request the distributed
value explicitly.

## Ownership and numerical boundaries

Operators acquire Array/mesh access and prepare shapes and private scratch
before numerical work. Kernels receive retained views and small metadata;
their workers write disjoint outputs. Workers do not acquire leases, call MPI,
publish shapes or submit CUDA work. The operation retains buffers and access
through the team join.

`OMP::ForRows` runs independent nonthrowing arithmetic. `ForRowsStatus` merges
explicit numerical failures in logical-row order. `ForSetupRows` may allocate
or validate setup inputs: it retains worker exceptions, joins all workers,
then rethrows the lowest-row failure on the controller. This is not numerical
recovery and preserves the [distributed failure contract](RuntimeCallerContract.md#distributed-exceptions).
No exception may escape an OpenMP worker region.

Floating sums use private contiguous ranges and an ordered merge, or parallel
independent components/probes that each retain their original row order. There
are no unordered floating atomic sums. OMP=1 retains canonical arithmetic.
Changing partitions may change floating association; numerical agreement then
uses a declared tolerance rather than a bitwise claim.

Thermochemical workers own exclusive cloned contexts. `AcquireUpTo` atomically
reserves an available bounded team before a row loop, returns its actual
capacity and fails immediately on exhaustion. It never waits while holding a
partial team. Cauchy provisions one context per configured OMP worker for a
mixture domain; kernels neither clone Cantera nor acquire workers. An API
explicitly accepting one mutable evaluator/lease requests one serial context;
pool/team overloads enable independent point work.

## Ordered work and setup

Jacobi, WBAP, point sampling, independent geometry and most FV/VR setup fills
use independent rows. SOR and SGS retain ordered scans inside mesh subpartitions
and frozen input across partitions. OMP=1 global GS and multiblock GS are
different valid iteration orderings. ILU factors and triangular solves retain
their dependencies. MPI, METIS, file-library calls, publication, ownership/ID
decisions and shared-destination scatter stay controller work unless explicitly
decomposed. Arbitrary user callbacks retain serial invocation.

Ghost-tree count/prefix/fill stays inside adjacency evaluation, with scratch
scaling with local visited rows/incidences. Frontiers, collection markers,
descriptors and canonical ghost mappings retain their semantics. Read-only
spatial queries use teams only after BVH and distance-hint initialization.
Cauchy's local reorder subpartition count follows its configured OMP team;
compare fields by persistent IDs when that ordering changes.

## Cauchy input preprocessing and console output

`defaults.mesh.scale` is finite and positive (default1), and
`defaults.mesh.rotationZDegrees` is a finite Z rotation (default0). File-loaded
meshes apply rotation then scale exactly once, after native preparation and
before FV metrics, initialization, wall distance and placement. Geom publishes
the change normally. `periodicLengths` are in the original file frame:
matching happens there, then coordinates and periodic frames transform together.
Source CGNS files need no modification or scaled duplicate.

Factory-injected meshes, including archive-backed factories, already own their
geometry and require identity preprocessing settings. Checkpoint reads replace
fields/history on the prepared mesh without another transform. Rigid motion
remains a separate mesh-bound frame/pose operation.

Console step/stage/iteration lines use compact aligned `fmt` formatting;
CSV/JSON/checkpoint precision remains independently configured. Setup lines
report domain-root controller elapsed seconds for read/partition, preparation
and FV/VR/state construction. These are not maximum-rank or GPU-kernel times.
Timers add no barriers, reductions or device synchronization. Normal progress
already reports wall time at observation points.

Plan61-5 entry ledgers and execution evidence distinguish source coverage from
runtime qualification. Flags or an environment string alone do not prove that
a helper actually ran on several workers.

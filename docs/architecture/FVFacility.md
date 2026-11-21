# CPU finite-volume reconstruction and boundary point interfaces

The new `CFV::FiniteVolumePlan` and `CFV::VariationalReconstructionPlan` compose
native Runtime storage, Geom entity identity, prepared metric/matrix buffers,
and explicit numerical operators. They do not wrap the legacy VR owner. The
existing `FiniteVolume` and `VariationalReconstruction<dim>` remain available.
This page describes the CPU envelope. The explicit [CUDA publication and Jacobi/WBAP baseline](DeviceReconstruction.md) uses these same prepared numerical buffers. Neither facility is a complete flow solver.
The [2026-10-31 next-step decisions](https://github.com/harryzhou2000/DNDSR/blob/cfab6865f30aa8ec21a31cbbdeada70e12d9d15b/docs/dev/facilities_next_steps_2026-10-31.md)
record planned extensions; those features are not implied by this implemented API.

## Ownership and responsibilities

```mermaid
flowchart TD
    mesh[UnstructuredMesh] --> metric[FiniteVolumePlan: shared immutable metrics]
    metric --> plan[VariationalReconstructionPlan: moments and local systems]
    metric --> field[DNDS Algebra DofField: owned and ghost Runtime arrays]
    plan --> coefficients[Caller-owned durable coefficients]
    plan --> workspace[ReconstructionWorkspace: temporary scratch only]
    boundary[Explicit boundary quadrature values] --> sweep[Reconstruction sweep]
    field --> sweep
    coefficients --> sweep
    workspace --> sweep
    plan --> sweep
    transforms[Per-face characteristic matrices] --> limiter[WBAP order pass]
    workspace --> limiter
    python[Python facades] --> field
    python --> workspace
```

Copies of plans retain immutable publications and mesh identity. Copies of
fields/Runtime handles share storage. `BuildCoefficients` allocates durable
state independently of `BuildWorkspace`; a serialized workspace can therefore
serve different solver-owned coefficient publications without retaining them.
Python `FVField` is an alias for the
existing DNDS `Runtime.Algebra.DoubleDofField`, not a second algebra implementation.
Its `owned`/`ghost` properties and boundary/transform arrays share the C++
allocations. Retained arrays remain alive after deleting their Python parent.
Array `numpy()` is a snapshot; `assign`, indexing and `read`/`write` obey the
existing Runtime lease contract. Release live views before invoking operators.

Geom owns connectivity, entity identity and halo row mappings; DNDS owns storage,
access and transfer execution; CFV owns reconstruction mathematics; the solver
owns field histories, equations, stage scheduling, stopping and domain coupling.
No solver or CFV dependency is introduced into Geom.

Preparation can select supported operations using an immutable
[`ReconstructionCapabilities`](ReconstructionCapabilities.md) value. The default
retains the complete standalone API; reduced profiles omit unused matrix tables
and reject unavailable operations before communication or output mutation.
Basis caching remains independent of this selection.

## Maintained face geometry

`FiniteVolumePlan` always stores one face-centered record alongside its RHS
quadrature records. `AcquireRead().HostView()` exposes `GetFaceCent(iFace)`,
`GetFaceMeanNorm(iFace)` and `GetFaceArea(iFace)`. The center is the mapped
reference-element center; the mean normal is the normalized surface integral
of the normal; area is the scalar surface measure. On curved faces, area is
not the magnitude of the integrated normal vector. Axisymmetric measures
include radius; a zero-weight face on the axis retains its geometric normal.

Construction integrates geometry once and publishes both tables. With
`faceIntOrder=1`, the single RHS point uses this maintained center, mean normal
and integrated area. Higher-order RHS uses its own point normals and weights;
it does not change the maintained face record. Approximate FV Jacobians can
therefore use one face record at every polynomial degree, without traversing
RHS quadrature. Both tables share the metric publication's mesh version
validation, read leases, and explicit device residency lifetime.

## Supported mathematics

* Cell-centered planar 2D and volumetric 3D unstructured cells, scalar or multiple
  components, polynomial degree 1–3 (formal second through fourth order).
* Legacy monomial ordering, scalar aligned-box scaling, zero-cell-mean basis,
  isotropic variational functional; independent factorial/HQM_OPT/TEST_OPT/manual
  directional weights, none/HQM_SD/SD_Power geometric weights and length controls.
* Explicit relaxed Jacobi and ordered SOR sweeps; order-wise 3WBAP limiting;
  arbitrary physical-point polynomial evaluation.
* Rigid periodic frames, including derivative-tensor and declared vector-block
  rotations; retained original face/cell incidence order.
* Radius-weighted axisymmetric metrics, moments, reconstruction and gradients
  on 2D geometry, with radius=y and no 2*pi factor.

The basis is `phi_j(x)=monomial_j((x-center)/scale)-cell_mean(monomial_j)`.
Stored coefficients have row width `NRecDOF()*components` in **column-major
mode-by-component order**: `row[component*NRecDOF()+mode]`. Cell means are never
modified by reconstruction or limiting. The functional includes derivative
tensor multiplicities and the selected per-order weights. Preparation uses SVD
on every local system and discards modes at or below
`max(svdAbsoluteTolerance, svdTolerance*sigma_max)`. Relative tolerance defaults
to `1e-13`, absolute tolerance to zero. Deficient systems have a minimum-norm
truncated solution; no automatic polynomial-degree reduction is performed.
`GetInverseDiagnostics()` reports global owned-cell filtered/rank-zero counts,
discarded modes, effective-rank histogram and maximum raw condition with its
persistent cell ID (ties choose the smallest ID). `GetCellInverseDiagnostics(iCell)`
reports local effective rank and raw/retained spectral conditions from setup,
without repeating the factorization. Exact zero matrices report rank zero and
infinite conditions. These methods and tolerance controls are exposed in Python.

Anisotropic functionals/bases, LVR penalties, variable per-cell degree and
alternative legacy limiter variants remain deferred. Standalone FV gradients,
explicit conversion, caches and linear operators are described below; they do
not import every legacy `subs2ndOrder` convention automatically. There is no
automatic legacy-config translation or new flow solver in this facility.

### Boundary quadrature protocol

`BuildBoundaryValues(components)` allocates `[NumBoundaryQuadrature(),components]`.
`GetBoundaryQuadrature(i)` gives the owning cell, face, face quadrature index,
physical point and normal. Fill the values before a sweep, refreshing them when
the BC depends on the current nonlinear/reconstruction stage. The numerical
operator reads these arrays and invokes no boundary callback.

The default `boundaryDerivativeWeights={1,1,1,1}` matches legacy's all-order
penalty. Nonzero higher derivative weights impose **homogeneous derivative
targets**; only order zero has nonhomogeneous target values in this API.
`boundaryZoneWeights[zone]` replaces the unsquared per-order multipliers for
one BC zone. `bcWeight` multiplies them before squaring in the functional.
Choose `{1,0,0,0}` when a manufactured polynomial supplies physical values
without homogeneous derivative targets. Tests and benchmarks state that choice
explicitly; the all-order/zone-weight tests compare with the matched legacy
callback. Separate nonhomogeneous-derivative quadrature controls remain deferred.

### Characteristic WBAP protocol

`BuildFaceTransform(first,count)` creates identity forward/inverse matrices for
each local face. Rows contain column-major `count*count` matrices acting on
column vectors in `[first,first+count)`. For a six-component Euler-SA field,
`first=0,count=5` transforms the gas block; SA is limited componentwise.
`count=0` is the explicit disabled descriptor and owns no matrix payload. The
typed optional-reference operator argument may also be absent; both forms run
the same scalar-only limiter path. Raw nullable pointers are confined to
language-binding or immediate kernel-launch frames.

The equation adapter fills the pair from **cell averages** and calls
`FinalizeFaceTransform`. This checks finite mutual inverses once outside hot
loops. The limiter checks storage identity and version in its prologue. Writing
either matrix, even with identical numbers, requires re-finalization. The
adapter is responsible for refreshing the matrices when its cell-mean state
changes: the limiter does not infer thermodynamics or state freshness.

Legacy `DoLimiterWBAP_3` and `_C` also pass cell averages to their characteristic
callbacks. Euler computes the eigenvectors from the arithmetic mean of those
states, with face-frame momentum rotation. The native ranged transform needs no
identity extension for passive components. Reconstructed states enter the separate smoothness indicator, not
that characteristic mean. The new operator limits all selected cell modes; it
accepts a supplied scalar activation indicator through the out-of-place pass;
constructing a physical smoothness indicator remains the equation adapter's job.

Call the WBAP API in **descending degree**, exchanging the coefficient halo
after **each** order. This is required by the legacy order-wise secondary-map
algorithm; one exchange after all orders changes its distributed result.

## Basis, functional, frame and cache contracts

`Math::TaylorBasis` defines mode/derivative ordering, derivative degree and
rigid-frame pullback in one place. Setup, limiting and sampling use that
definition. With scalar length scaling, the functional contracts each symmetric
derivative tensor with multiplicity `order!/alpha!`. This is its Frobenius norm;
`functional.dirWeightCombPowV=1` is rotationally invariant. Compatibility powers
0/2 are available but are rejected for rotated periodic reconstruction.
Anisotropic tensor metrics remain a separate future policy, not another switch
inside the isotropic basis.

`functional` configures the four directional schemes, three geometric schemes,
their coefficients, scale multiplier and length policy. `HQMWeights` remains a
compatibility shorthand for HQM_OPT + HQM_SD with bias one. `BaryDiff` uses the
mapped barycenter separation (reflected normal separation on an external face).
`MeanAACBB` uses the RMS of the arithmetic mean aligned half-box vectors.
`CellMax` uses the symmetric maximum aligned half-box length; it deliberately
does not import the old inertia-oriented/asymmetric parent-slot formula.
TEST_OPT reuses the geometry-only face-span helper. No anisotropic basis or LVR
penalty is implied by these options.

`cacheDiffBase` enables optional face-quadrature basis storage for owned
stencils. `cacheDiffBaseOrder=1` retains values and first derivatives by default;
orders 0..3 select other coverage, capped by polynomial degree. Setup uses
higher derivatives transiently without retaining them. `cacheDiffBase=false`
keeps no basis cache; shared point evaluation uses bounded local scratch.
`GetCachedFaceBasis` copies only retained derivative rows;
unprepared exterior or ghost-only face stencils fail explicitly. The cache is
in Runtime buffers, with basis/config/metric identity retained by the immutable
plan. `AcquireBasisRead().HostView()` exposes borrowed pointer/shape views for
hot loops; retain the scope and use only real parents in owned-cell stencils.
Its `derivatives` count is the matrix leading dimension. The read scope's
`FaceBasis(face,side,globalPoint,order,scratch)` borrows covered cached rows or
evaluates missing rows in caller-owned bounded scratch; it works with caching
disabled. Cauchy's RHS, polynomial BC refresh and surface output use this same
scope and keep no separate per-incidence basis-value allocation.
`PolynomialSamplingPlan(plan,cells,points)` snapshots arbitrary query
cell IDs and coordinates, and follows the same optional derivative-cache policy.
Uncached/higher-order requests use the common zero-mean Taylor evaluator.
Both interface samplers use this facility. Query-array edits do not silently change its meaning; construct a
new cache for new locations. Geometry/order mutations invalidate it through
the metric stamp. `Evaluate(...,derivative)` samples the central derivative-mode
ordinal (0=value, 1..dimension=gradient, higher modes include Hessians). Its
input ghost reads are omitted when all selected cells are owned.

`EvaluateValueGradient(mean,coefficients,outputs)` accepts a
`PolynomialSampleOutputs` bundle with an optional `values` array and three
optional Cartesian `gradients` arrays. Allocate each requested output with
`sampler.Size()` rows and the field's component count. All outputs are in the
donor reference frame. No halo exchange, output allocation or implicit layout
conversion occurs. At least one output is required; gradient slots outside
the mesh dimension and derivatives of a value-only plan are rejected.
Input/output aliases and duplicate/overlapping outputs are rejected before
writes, including shared empty arrays on ranks with no owned query points.
Outputs must have independent payload access identities, even when separate
metadata handles describe the same buffer.
If evaluation fails, discard the bundle rather than publishing partial data.

The CPU implementation combines values, dx and dy for cached owned 2D P3
six-component AoS queries. Fixed row access retains each parent's physical
row pitch, including padding; each coefficient feeds the three sums in the
reference evaluator's mode order. Other layouts, widths, ghost queries,
cache policies and partial requests retain independent `Evaluate` calls.
The public checks and leases stay in the prologue, outside the point loop.
Cauchy uses the bundle for variational RANS source sampling; source physics,
communication, version publication and source Jacobian preparation remain
Cauchy responsibilities. This is a CPU facility; device-resident fields fail
explicitly instead of staging implicitly.

Rigid periodicity transforms both sample locations and derivative tensors during
setup. `BuildWorkspace(components,periodicVectorStarts)` declares sorted disjoint
three-component polar-vector blocks, e.g. `{1}` for `[rho,rhou,rhov,rhow,E]`.
Unlisted components are scalars. Neighbor components are rotated into the output
cell's frame before reconstruction/WBAP; characteristic transformation is a
separate subsequent operation. Three-component vectors work on 2D geometry.
Sampling returns donor-reference components; it does not apply a domain pose or
moving-frame velocity correction. Taylor neighbor coefficient changes are
analytic changes of center/scale/frame, with no SVD truncation in that coordinate
map. Every numerical local reconstruction inverse still uses full SVD.

`ConstructMetrics(mesh,intOrder,true)` selects axisymmetric integration with
`r=y>=0`, `r*dA` volume and `r*ds` face measure, excluding `2*pi`. Moments and
cell averages use the same convention. The integrated unweighted radial moment
supplies the weighted Green-Gauss correction, including cells touching the axis.
No pointwise `1/r` is needed. Negative radii, zero radial cell volume, 3D geometry
or periodic transforms that change radius are errors. Equation source terms and
cylindrical component connections belong to the equation adapter.

## Reconstruction linear-system operators

`ApplyLinearOperator`, `BuildRightHandSide`, `ComputeResidual` and
`ApplyLocalPreconditioner` expose the unscaled system separately from relaxation.
They do not perform halo communication or own a Krylov solver. Refresh input
halos before the stencil operator and supply distinct output arrays. Homogeneous
perturbations use the operator without boundary forcing. Raw A/B/forcing blocks
are retained separately from `Aplus*B`, so choosing a strong SVD cutoff does not
silently replace the original operator with its projected version. The local
preconditioner is the truncated SVD inverse and may be semidefinite when modes
were removed. Its null modes are not restored by iteration.

These interfaces can feed existing GMRES/FGMRES drivers with DNDS field algebra.
Flexible PCG additionally requires a symmetric positive system/subspace and an
admissible positive preconditioner. The default isotropic raw operator passes
bilinear symmetry and positive-form tests, but that is not a universal SPD
certificate for arbitrary weights, boundary penalties, cutoffs or null spaces.
The caller must freeze the plan/linearization and handle compatible nullspace
projection. Ordered SOR and left preconditioning must not be advertised as
symmetric merely because the unscaled functional is symmetric.

## Count, allocate, fill and execute

### Split reconstruction/limiting and scalar activation

`DoReconstructionIterTo(mean,input,workspace,boundary,region)` writes selected
rows of `workspace.temporary`. Jacobi leaves input unchanged. Use `Core` while
input ghosts are being written, then `HaloDependent` after completion, and call
`CommitReconstruction` after both phases. The core entry acquires no input ghost
leases; its work set uses conservative cell–node–cell adjacency. Commit is an
explicit copy into owned coefficients and must wait for old sends to finish.
No regional lease or per-access ownership test is introduced inside cell loops.

For SOR, the explicit caller-owned coefficient field is the working iterate;
`input.Ghost()` is the frozen remote iterate. Initialize the working iterate
from the desired old iterate yourself. If sends overlap, exchange a **separate
snapshot** so SOR can write the working iterate without modifying live send buffers. `coreFirst`
selects stable core-first SOR order. Split versus unsplit results must use that
same logical order; changing SOR order can change the result.

`DoLimiterWBAP_3OrderTo` reads coefficients and writes selected temporary rows.
Complete both regions against the same input, commit, and exchange before the
next descending polynomial order. Its optional owned scalar indicator activates
cells when `indicator >= threshold`; inactive rows copy input unchanged.
Nonfinite thresholds/indicators are errors. Inactive neighbors still participate
as stencil inputs. Characteristic transforms retain their existing data protocol.

Direct MPI/core overlap tests pass np=1/2/4/8 for Jacobi, each WBAP order and
core-first SOR with a separate snapshot. They also verify early commit rejects
writing an active send input. These are correctness/protocol checks, not measured
speedup evidence; MPI may finish payloads early while leases remain active.

### Standalone second-order FV gradients

`GradientPlan::Prepare(mesh,metrics,method)` prepares Green–Gauss, weighted
least squares or GGMP without constructing a legacy VR object. It counts and
allocates Runtime incidence weights and boundary-row IDs, one entry per owned
cell-face incidence. Canonical topology supplies face/neighbor IDs; there are
no duplicated topology or per-incidence rotation payloads. The default sample
is the mapped reference face center with normalized integrated normal and
separately integrated area. The boundary buffer contains **physical boundary
values**, one per boundary face, at the center points exposed
by `GetBoundaryQuadrature`; it never expects reflected ghost states. Output
holds `[component*dimension+direction]` in an ordinary DNDS numerical field.

Least squares uses inverse-distance-squared residual weighting with face
face-area measure weighting; it factors the weighted design matrix using SVD
rather than forming normal equations. GGMP eliminates the face unknowns from
the corrected Green–Gauss system before SVD inversion. Uncorrected Green–Gauss
does not promise linear exactness on general skew cells. Current tests establish
regular-grid linear exactness for all three methods and sheared-grid exactness
for LS/GGMP. Periodic geometry is mapped into the output cell's frame during
preparation. `Evaluate(...,periodicVectorStarts)` declares nonoverlapping
three-component polar-vector blocks; other components are scalars.
Normal distances determine Green-Gauss face interpolation. Axisymmetric GG/GGMP
also include the integrated radial divergence correction, using integral moments
rather than division by point radius; zero-measure axis faces are valid.

`ConvertToCoefficients` explicitly overwrites a VR field with scaled linear
Taylor modes and zeros the higher modes. Its zero-mean basis preserves the cell
average. The gradient facility remains useful independently of VR. Normal
Runtime read/write leases reject aliases; callers refresh halo/BC producers
before evaluation. Preparation remains on CPU; independent cell setup uses the
configured DNDS distributed OMP policy after prefix allocation. The same prepared
incidence order is used with one or multiple workers.

`GradientPlan::Placed(device,context)` copies the immutable tables explicitly;
the overload accepting placed FV metrics shares those tables with the caller.
The placed gradient plan retains no CPU boundary-coordinate vector. Obtain
boundary locations before placement, and supply boundary values on the selected
backend. The CUDA profile currently permits combined AoS flow5/SA/2EQ fields.

Evaluation, linear coefficient conversion/extraction and direct-gradient WBAP
have engine/stream overloads. Their prologues validate association, shape,
residency and aliases once; free CPU/CUDA leaves consume views and numeric
metadata. Core evaluation/limiting declares no input ghost read. The caller
orders halo completion before dependent work and serializes writes to an output.
Queued captures retain Array ownership and geometry guards until completion;
no numerical leaf performs MPI, acquires leases or waits on the host.

Build a `GradientWorkspace` once for each component/vector layout. Periodic
component metadata is uploaded once, version-checked on reuse, and shared by
gradient evaluation and limiting. `withLimiter=true` reserves CPU worker
scratch; CUDA limiting uses bounded thread-local scratch instead. Scratch
contains no durable solution state. The synchronous convenience limiter builds
temporary workspace; repeated solver calls should reuse the queued workspace.
Optional activation values must be finite. The synchronous CPU entry diagnoses
them before execution; queued leaves assert the finite-data precondition without
a host copy or synchronization. Transform count zero never accesses matrices;
nonzero transforms must be finalized, immutable publications. The current
placement API copies a finalized CPU transform. Device-produced characteristic
publication and Cauchy production wiring remain Plan61 phase B work.

The Python bindings expose `EvaluateQueued`, `ConvertToCoefficientsQueued`,
`ExtractLinearGradientsQueued` and `LimitWBAP3Queued` with the same engine/stream
and workspace ownership contract. Finish the stream before inspecting results
through NumPy or other synchronous aliases. Added CUDA/worker tests for these
paths await the grouped Plan61 B build; source availability is not validation.

Preparation first counts face quadrature rows, cell-face incidences and boundary
quadrature rows, allocates dense Runtime tables, then fills them. Face-local
functional blocks are evaluated once and reused by their incident cells. Setup
scratch is proportional to local/ghost cells and faces; no global-entity-sized
scratch is allocated. Bounded small matrices hold polynomial setup algebra.

The operator prologue validates plan/field association, placement, shape,
generations and aliases, then acquires leases and projects table/static row
views. Numerical loops use bare span pointers and Eigen maps; no per-cell
ownership or lease checks occur. Framework workspaces are allocated in advance;
Eigen remains the CPU linear-algebra implementation. This is not a guarantee
about Eigen's internal scratch allocation for arbitrarily wide component blocks.

Future flux kernels can use `auto read=metrics.AcquireRead();` followed by
`auto view=read.HostView();`. The borrowed view exposes cell volumes/centers/
barycenters and face quadrature ranges, points, normals and weighted measures.
Keep `read` alive through the loop. It pins the buffers and blocks mesh mutation;
view access does not reacquire ownership or check generations per element. The
`HostView()` rejects device placement. Explicitly placed metrics expose `View()`
with device-tagged whole-array descriptors for C++ launcher validation; the
retained read scope must still outlive asynchronous use.

The synchronous CPU operators retain their leases until return. The same public
plan/workspace API dispatches prepared CUDA Jacobi and WBAP kernels through the
eager engine when the selected placement supports them; unsupported SOR/backend
combinations fail explicitly. Do not call a CPU operator before its eager-engine
producer or MPI halo is complete. Every workspace is private to one concurrent
invocation on either backend.

```cpp
auto metrics = CFV::FiniteVolumePlan::ConstructMetrics(mesh, 5); // collective
CFV::ReconstructionSettings settings;
auto rec = CFV::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
auto mean = metrics.BuildField(6);
auto coefficients = rec.BuildCoefficients(6); // durable solver state
auto work = rec.BuildWorkspace(6);
auto boundary = rec.BuildBoundaryValues(6);
// Fill mean.Owned() and boundary.values through scoped Runtime write leases.
auto meanHalo = metrics.BindHalo(mean);                 // collective, persistent
auto coefficientHalo = metrics.BindHalo(coefficients);
meanHalo.Begin().Wait();
for (int iteration = 0; iteration < 20; ++iteration)
{
    coefficientHalo.Begin().Wait();
    rec.DoReconstructionIter(mean, coefficients, work, boundary);
}
coefficientHalo.Begin().Wait();
for (int order = settings.maxOrder; order >= 1; --order)
{
    rec.DoLimiterWBAP_3Order(coefficients, work, order); // optional finalized transform argument
    coefficientHalo.Begin().Wait();
}
```

Halo bindings reuse the mesh EntitySpace's canonical row transfer mapping and
native direct MPI datatypes. There is no hidden communication in sweeps,
limiting or sampling. Match collective call order/settings across ranks, drain
operations before MPI finalization, and explicitly refresh halos after writes.
Local setup failures agree across ranks before subsequent communication.
Sampling ghost cells requires already complete geometric closure and fresh
coefficient/mean ghosts. Mesh geometry/topology/distribution/order changes
invalidate plans; rebuild rather than matching fields by equal row counts.

## Geom matching and CFV sampling

The geometry-only projection, immutable surface storage and distributed search
are described in [Surface geometry and location](SurfaceLocation.md). Boundary
interfaces are adapters over that reusable facility; they attach volume-cell
links after geometric donor selection.

```mermaid
flowchart LR
    mesh[Donor mesh] --> patch[BoundaryPatch: selected surface coordinates and volume slots]
    pose[RigidPose] --> patch
    patch --> tree[BoundingBoxTree]
    query[Receiver points and stable query IDs] --> map[InterfacePointMap]
    tree --> map
    map --> sample[CFV InterfaceSamplingPlan]
    plan[Reconstruction plan] --> sample
    field[Mean and coefficients] --> sample
    sample --> output[Values in query order]
```

`BoundaryPatch::Create` reads selected **owned** boundary elements directly from
the volume mesh. It stores only their coordinates, boundary/face/adjacent-cell
IDs, cell-face slot and persistent boundary ID. No boundary volume copy is made.
`BoundaryStencilPlan::Prepare` accepts a caller `GhostSpec`, evaluates it through
the existing registry/compiled tree, and builds a detached adjacency/ghost
publication. It neither changes the mesh's compute halo nor bypasses the DSL.

The search accepts Line2/Tri3/bilinear Quad4 surfaces, including warped quads. A BVH finds
candidates; element-coordinate projection, reference containment and explicit
gap tolerances decide acceptance. Query order and IDs survive search ordering.
Shared seams use the smallest persistent donor boundary ID. Coincident accepted
interiors are rejected as ambiguous. Uncovered queries, nonfinite coordinates,
duplicate query IDs, degenerate surfaces and nonrigid poses produce errors.
High-order surface geometry and periodic image selection fail explicitly.

`WithPose` returns a new pose sharing selected reference coordinates and their
immutable reference-space BVH. Queries transform into that frame, so rigid
motion needs no local BVH rebuild. Old maps keep their old pose.
A fixed receiver query can change donor element/cell after sliding. For separate
domains, retain each patch independently and construct each directional map
separately; a reverse map is not an inverse or conservative transpose.

`InterfaceSamplingPlan` checks donor entity/order identity and stores donor-local
sample points **after projection onto the donor boundary**, so an allowed
normal offset in the receiver query does not become a polynomial extrapolation
off that boundary. Its output is in query order and **donor physical frame**. The
equation adapter handles vector rotation and moving-frame velocity conversion.
Choosing limited/unlimited values means passing the corresponding coefficient
field. No conservative flux, ALE/GCL, FSI force transfer or flow coupling iteration
is implied by point interpolation.

`InterfacePointMap` remains the local API. `DistributedInterfacePointMap::Match`
is the collective distributed API, with all ranks participating even when a
rank has no queries or selected boundaries. `CoarseBoxes` all-gathers
`coarseBoxesPerRank` (1..10, default one) conservative world-space boxes per
rank, sends queries to candidate geometry
ranks, evaluates exact projections there, returns all proposals and selects on
the query origin. It rejects ambiguous accepted interiors and uses persistent
boundary IDs at seams. Bounds include the reference-tolerance extrapolation
envelope as well as the physical gap, so accepted endpoint projections cannot
be lost before exact matching. Multiple boxes never duplicate a query to the
same rank. `AllGatherBoundary` replicates the selected surface
geometry as an explicit correctness baseline; it does not replicate volume
mesh data or solution fields. Both use fixed integer/floating tables and the
same exact projection kernel, so the baseline independently checks routing,
not the projection mathematics.

Matches distinguish the boundary-geometry rank from the donor-cell rank and
owned donor row. `geometry.donorCell` is in the geometry rank's mesh order;
distributed consumers must use `donorRank`/`donorOwnedCell`. Query IDs are unique
per origin and output order follows the original local query order. Poses do
not change internal mesh coordinates. `ValidateFor(activePatch)` rejects a
retained map when the active immutable patch/pose object changes. Collective
motion/deformation updates require new maps on every rank: rank-local validation
cannot independently discover an unannounced update on another rank.

`DistributedInterfaceSamplingPlan` routes selected query points to the actual
donor-cell owners once, then evaluates their reference-frame polynomials and
returns values to the receivers through the reversed canonical row route.
Allocate a `DistributedSamplingWorkspace` with `BuildWorkspace(components)`.
`Evaluate` provides synchronous collective evaluation/return; alternatively
`EvaluateDonors` and `BindResult` separate compute from a reusable persistent
result binding. Sampling construction, result-binding checks and the collective
evaluation wrapper inherit that map context's preflight policy. Disabled mode
omits successful local validation agreements and aborts on local rejection;
counts, schemas, routing and payload transfers remain required. The rank-local
`EvaluateDonors` entry still throws locally; its caller must coordinate errors
before starting any subsequent communication, or use the collective wrapper.
Concurrent bindings sharing a route context use distinct `tag`
arguments; the default tag zero is for non-overlapping bindings. The workspace
is private to one active evaluation. Callers
finish producers, release conflicting Python views, retain all handles through
completion and arrange halo readiness under the polynomial evaluator's contract.
Drain/rebuild result bindings after changing output storage or query/donor order.
Field halos and interface sample routes are separate mappings. There is still
no physical interface-domain orchestrator or conservative flux coupling here.
Cauchy's separate rigid-frame operators do not change this numerical sampler's
reference-frame contract.

The coupling-context overload accepts an optional local donor patch. Donor-domain
nonmembers pass no patch and hold no dummy volume mesh. Domain-communicator cell
owners are translated explicitly into coupling ranks, including noncontiguous
or reversed groups. `HasLocalDonor` distinguishes absent domain membership from
an empty local boundary selection. The optional-reconstruction sampling constructor
uses the same participation contract; its pointer-form `EvaluateDonors` and
`EvaluateCoupled` take null inputs on nonmembers. All ranks still participate in
setup/return transfers. Python exposes `MatchCoupled` and `EvaluateCoupled`, using
`None` for the same absence. Shared contexts require distinct explicit tags for
concurrent persistent result bindings. No solution gather is introduced.

## Validation and reproducible performance

`cfv_test_fv_facility` compares each sweep with legacy VR, independently recovers
polynomials from quadrature cell means, compares six-component characteristic
WBAP, exercises periodic triangles/quads and SOR/Jacobi, empty ranks, stale
plans, invalid shapes and SVD minimum-norm solutions for deficient systems.
The coupling sampling fixture additionally checks a reversed donor subgroup,
query-only nonmembers, distinct geometry/cell owners, sparse/dense matching and
both transient and persistent returns. `dnds_test_cauchy_multidomain_mesh`
constructs complementary P1 Delaunay disk/periodic box-hole meshes on one reader,
then uses native distribution/partition/assembly. Polygon area and moving-circle
matching are independent geometry gates, not a physical Cauchy coupling test.
Fixture generation selects CGAL's header-only Boost exact-predicate backend;
it adds no GMP/MPFR requirement to the solver.
`geom_test_boundary_interface` checks geometric positions, seams, gap failures,
3D triangles/quads, relative sliding, ambiguous interiors, immutable old maps
and explicit DSL stencil publication. `geom_test_distributed_boundary_interface`
uses generated Cartesian boundary meshes and independent ownership/ID/point
oracles at np=1/2/4/8, including empty ranks, seams, sliding across rank ownership,
different geometry/cell owners and agreement with replicated-boundary matching.
`cfv_test_distributed_interface_sampling` checks two manufactured polynomials,
owner-dependent values, moving poses, query ordering and persistent result
transfer updates at np=1/2/4/8. Python tests exercise shared ownership,
DNDS algebra reuse, two domain identities and the sampling chain.

`cfv_benchmark_fv_facility` takes `mesh dim degree sweeps trials periodicLength`.
Both methods run on the **same mesh instance** with the same five-component
means, BC functional and halo schedule. It reports max-rank metric setup,
coefficient setup, per-sweep time including halos, one WBAP application, and
checks coefficient agreement. Alternate order per trial; use OMP=1 and a single
exclusive node for comparisons. The evidence ledger records the actual runs.

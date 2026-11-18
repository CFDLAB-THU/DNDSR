# Cauchy CPU kernel boundaries and I/O contracts

This document describes the Plan60 free-kernel extraction. It is a source
contract, not a CUDA implementation or a numerical validation receipt.

## CUDA RHS shape catalog

The perfect-gas, non-species FP64 RHS uses CTA segments. Its shared private
numerical implementation specializes component count C=5/6/7, geometry
dimension D=2/3 and modal count M. Valid VR shapes are M=2/5/9 in 2D and
M=3/9/19 in 3D. M=0 is the non-VR constant/direct-gradient path; implicit P1
VR still uses M=D and retains its VR allocations. Block size, optional input
roles and face reduction choice do not multiply the kernel catalog.

Mean/interface rows have width C, polynomial rows C*M, conservative physical
gradients C*D and flow5 primitive gradients 5*D. Optional viscous polynomial
coefficients also have width C*M. The host wrapper validates residency, compact
AoS row shape and selected roles once; static typed row access is used within
the specialized numerical body. No per-element device/residency validation or
hidden input transfer is added. Thread-mapped reduced-precision flow5 kernels
remain separate comparison leaves; discarded FP64 mapping experiments are not
fallbacks. Gradient construction and gradient-specific limiter kernels keep
their existing runtime-dimension interfaces.

`scripts/manifests/cauchy_rhs_shapes.json` is the only instantiation registry.
The stdlib Python generator writes 24 one-kernel CUDA translation units plus a
host switch/declarations/source index into the build tree. CMake tracks script
and manifest changes, shares one generation target between static/shared
libraries, and restores missing outputs. Numerical implementation changes are
ordinary compiler include dependencies. Generated wrappers are not tracked in
Git. This bounded catalog reduces compiler tail size; performance claims still
refer to measured shapes/devices rather than every specialization.
Validation state is tracked separately in the Plan60 execution ledger.

## Ownership and execution boundary

An operator facade validates schema, row space, placement, optional inputs and
aliasing; reserves Arrays/geometry; creates the small descriptors; selects a
backend entry; then publishes outputs and merges returned diagnostics.
CPU kernels neither acquire access nor communicate, submit work or publish
generations. Physics workers are explicit CPU arguments. They may mutate their
own point-property scratch, never hidden solution fields.

Kernel views borrow payload, shape and metadata. The facade or queued
operation must retain **all three** through execution. A Runtime Array read
scope retains its allocation and layout; a MetadataBuffer retains descriptor
storage but does not retain the Arrays pointed to by nested views. A copied
request alone therefore cannot own an asynchronous launch.

Current Cauchy worker callbacks execute complete CPU facades and finish before
the enclosing public call returns. Their scopes are acquired inside that
callback. `OperatorSchedule::Submit` rejects a device lane for these callbacks.
Prepared launchers instead use `SubmitArrays` for declared Array operands or
`SubmitPrepared` for a shared record with `Accesses`, `Launch(context)` and
nonthrowing `Retire(error)`. The record must own all nested descriptor/scopes;
the runtime retains it until actual execution completes, including a device
launch's pending work. This infrastructure does not implement the deferred
Cauchy CUDA leaves. Merely swapping a CPU facade for a CUDA launch is incorrect.

Transfer dependencies belong to `TransferBinding::SubmitReady/SubmitAfter`
and `TransferOperation::EnqueueWait`, not a provider switch in Cauchy. Preserve
the operation until its consumer edge has been registered: extracting only
its event loses the portable-MPI versus native-stream completion contract.
NCCL uses native ordering; portable MPI explicitly progresses on its allowed
controller before a CUDA consumer can use the received data. That bridge is
non-capturable. DofHalo records source generations and publishes freshness on
retirement. `RequireCurrent` checks CPU-visible completed state; it cannot
validate a future queued publication at CUDA launch submission time.

Fixed-count VR queues boundary refresh, reconstruction and commit in order;
it does not drain the full engine after each sweep. A requested reconstruction
residual observation remains an explicit host boundary. Before exchanging a
new coefficient publication, `OperatorSchedule::BeginAfterCompute` supplies
the producer prefix to the transport: portable MPI observes that prefix on
its controller, while a stream-native provider imports it on the communication
lane without host synchronization. This does not claim portable MPI is
captureable. Current CPU limiter/PP facades still have an explicit input
publication boundary; removing their controller-only preparation is separate
from the fixed-count VR sweep contract.

The LVR compound operator likewise queues target limiting, target sampling,
gate construction and fixed neighborhood-expansion layers. Producer-prefix
exchanges order each new layer; the compound operator joins once at return.
Mean-gradient auxiliary preparation publishes its provenance and hands off
the chemistry worker on the compute lane before the gradient exchange, rather
than draining the engine for that metadata update. These remain synchronous
public CPU operators; their return boundaries and dynamic PP/convergence
observations are not removed by the fixed-count scheduling contract.

## Storage classes

1. **Mesh-sized fields:** stencil samples, offsets, face/region selections,
   boundary layouts, source quadrature values/weights, sparse symbolic maps
   and frozen Jacobian values are Runtime Arrays. This includes immutable,
   rank-local data with no communication.
2. **Small immutable numeric metadata:** block offsets, component maps,
   vector starts, active component flags and species reference energies use
   MetadataBuffer publications and encapsulated const spans.
3. **Point/row scratch:** local dense matrices, primitive/transport work,
   integrated flux vectors and local reduction totals are per-call CPU scratch.
   Their extent depends on equation/basis size, not the number of mesh cells.
   A caller-supplied small Runtime scratch array is also valid.
4. **Control/owner containers:** vectors of Arrays, leases, workers, boundary
   policies or configuration recipes remain ordinary C++ containers. Nontrivial
   services must not be copied as device metadata bytes.

Setup vectors may count/order/fill final arrays but must not remain a second
numeric authority after publication. Diagnostic list APIs may copy to vectors;
hot paths must use retained row views instead.

## Large-kernel effects

| CPU package | Read inputs | Written data and update mode | Scratch / ordering |
|---|---|---|---|
| Flow5 / equation RHS | Means, finalized reconstruction or physical gradients, optional distinct viscous reconstruction, geometry/basis, frozen interface rows, auxiliary fields, physics | Selected cell residuals **overwrite**; optional low residuals overwrite. Selected face flux/wall-force records overwrite. No means/coefficients are changed. | Equation-sized point/flux/transport scratch. Generic RHS computes face values then gathers cells. Core/dependent face sets may overlap; execute these phases in order on one compute lane. |
| Auxiliary preparation | Owned means for core; complete means for dependent; optional mean gradients, derived values, interface trace, geometry | Enabled mean/face outputs overwrite; cell reductions overwrite. Face scratch is explicit when only reduced cell outputs are requested. | Means -> face values -> cell reduction within each phase. One retained write reservation spans core/dependent phases; no separate reader may consume the incomplete publication. |
| Boundary trace | Means, optional reconstruction/basis, trace geometry, frozen interface packet, physics worker | Reconstruction/gradient boundary rows overwrite | Physical-state and per-equation scratch only. Interface traces are copied from the prepared packet; reconstruction kernels do not evaluate BC branches. |
| Direct gradient / PP | Means, input gradients, periodic geometry and metadata, admissibility policies | Barth destination is distinct from its source; PP **updates gradients in place** under exclusive access. | Extrema/sample scratch is explicit. No extra storage alias or lease is created for the read portion of an in-place update. |
| Polynomial PP | Means, sampled candidate/reference values, degrees, thermal/RANS policy | Reference degree and beta/recovery fields overwrite; coefficients are explicitly **in-out** during truncation/compression/recovery | CFV sampling remains a separate operator between PP leaves. Read-only recovery flags/scales and statistics inputs use const views even when borrowed from a held write scope. |
| Source / source Jacobian | Means or prepared quadrature samples, gradients, weights/offsets, wall distance, physics worker | Residual source entries and source Jacobian are **additive**; optional mean thermodynamic snapshot **overwrites** | Property borrows are consumed before the next worker evaluation. Mean EOS snapshots and quadrature source properties have distinct meanings. |
| FV Jacobian preparation | Frozen means, face auxiliary signals, geometry, BC/interface inputs, selection and moving frame | Initialize diagonal; add source/face terms; off-diagonal blocks overwrite. Evaluated storage freezes mean/EOS/radius arrays. | Two-level CSR retains cell/block/face incidence order. Periodic transformations are explicit. No per-block lease acquisition or live-state read. |
| Apply / Jacobi / SGS / ILU fill | Retained symbolic pattern, diagonal/inverse, assembled blocks or immutable evaluator, local/ghost vectors | Apply/Jacobi outputs overwrite selected rows; lower/upper SGS writes distinct forward/output buffers; ILU fills the complete factor template | CPU TUs select static-size/layout kernels. Core must not read ghost input. SGS new-local-value dependencies are intentional; row scans are not freely parallel. Generic Solver owns factorization/triangular arithmetic. |
| Interface packet | Donor values, receiver/component maps and pose matrices | Selected receiver quadrature/face-center rows overwrite, then declared vector triples rotate in place | Search, routing, transfer and packet publication stay outside the kernel. Return only after all required fields are available. |
| Nonlinear / stage / pMG | Logical DOF views, time coefficients, selections, policies, geometry | Nonlinear candidate output is distinct. Explicit stage state and pMG correction are deliberately **in-out**; local norms returned to controller. | No MPI reduction or stopping decision in the leaf. Explicit RANS reads the already updated flow through a const projection of the same enclosing write reservation. Existing normalization, accumulation order and PP stage policy are preserved. |

A const request object may contain mutable **output views**. Constness of the
descriptor is not constness of the referenced output. Effect comments and
element constness identify access mode; users must not infer permission from
the top-level const reference alone. Output-only views cannot alias read
inputs except for explicitly documented in-out operations.

## Layout and pointer rules

Equation inputs/outputs retain physical row/element stride in EquationLayoutView,
DofComponentView or Runtime table views. Full-state packing is not required for
split allocations. Native geometry and Jacobian snapshots may deliberately use
dense internal tables; that is an explicit internal storage choice.

Raw data access is confined to local physics scratch, fixed POD matrices,
centralized Eigen adapters and the proven dense six-component Jacobian leaf.
The latter accepts a strided input direction and uses unit-stride only for
its explicitly dense snapshots. General equation rows must not be interpreted
as contiguous pointers.

Read scopes remain outside point loops. Bounds assertions may be disabled by
the existing optimized view setting; shape/backend/schema validation belongs
at the facade boundary. No per-element shared-owner lookup is introduced.

## Source audit, 2026-11-12

### CPU failure paths

Numerical CPU entries report `NumericalStatus` / `NumericalResult<T>`. The
converted generic and flow5 entries are `noexcept`: they contain no explicit
throw, exception handler, or exception-driven recovery. A failed status carries
an error category, provider/check detail, and optional row/point index. Outputs
may be partial on failure and must not be consumed or published as valid results.
Normal admissibility decisions (such as rejecting an explicit step) remain
ordinary numerical results, not exceptions.

`Cauchy/Core/NumericalFailure.hpp` is the **operator-side** decision point.
`RequireNumericalSuccess` aborts under `MPI::SolverFailureScope`; outside a
production scope its checked-library contract throws so a coordinated UT can
inspect the failure. Tests of a numerical leaf should normally inspect its
status directly. There is no test/production mode in the numerical leaf itself.
The RHS facade also retains explicit checked/fatal entry points covering request
preparation, with its choice outside `CPU/`.

Solver scopes are controller-thread-local. Construct the production engine/CPU
executor while the solver scope is active: the CPU executor records that mode
at construction and enforces abort on its worker independently of thread-local
scope inheritance. A checked-test executor must not be reused as a production
executor merely by entering a solver scope later. Solver
catches around collective/matched communication abort using rank-local work;
they cannot recover, drain distributed requests or enter agreement collectives.
Setup/lease validation is still allowed to throw. This policy does not turn
local validation into unconditional process termination for coordinated UTs.

External EOS/chemistry exceptions are intercepted by the point-property or
reactor adapter. `TryTemperatureFromUV` distinguishes failed UV solving from
composition/setup, allocation and unexpected backend errors. Only explicit
thermodynamic failure statuses permit the configured point-to-mean retry.
Transport/source failures and invalid requests must not silently become a mean
fallback. The retry reloads the mean **and reapplies the periodic component
rotation**, including a periodic self-neighbor.

The current source pass covers visible Cauchy CPU checks and property calls.
It is not a recursive proof that every reused legacy algebra, RANS, geometric,
expression or allocation path is nonthrowing. An unexpected throw through a
`noexcept` numerical entry terminates; it never drives a fallback. Point-sized
CPU scratch still uses ordinary allocations. Recoverable resource exhaustion,
allocation-free kernels, and conversion of all transitive legacy validations
are not claimed by this pass. Those limits must not be confused with an EOS
nonconvergence status.

The closure audit also follows analytic profiles and perfect-gas boundary
helpers reached by RHS, primitive gradients and Jacobian preparation. Their
`TryProfileValue`, `TryFaceValue` and `TryExterior` entries propagate numerical
status, including total-inlet conversion failures; checked convenience wrappers
remain outside numerical kernels. The same status contract covers profile
initialization and the analytic error oracle. Programmable expression input is
different: expression validation and checked physical-state conversion are
setup work in `FiniteVolume_InitializeField.cpp`, outside any `noexcept` CPU
kernel. They reject invalid user input without substituting fallback states.

### Findings and corrections

1. Free-kernel packages contain no owner-qualified FiniteVolume/Domain/
   LinearProblem member definitions, direct access acquisition, MPI or
   scheduling calls in the audited source. Existing lower-level CFV/DNDS/Solver
   operators are intentionally reused.
2. The evaluated-Jacobian publication comparison now traverses retained CSR
   rows instead of allocating two complete nested host-vector copies per
   nonlinear update. Diagnostic Columns() remains an explicit copy.
3. The gradient closure in EvaluateImpl is now declared before OperatorSchedule,
   so coordinated-test exception cleanup drains queued work before that
   reference-captured closure is destroyed. Solver collective failures still
   take the independent abort path.
4. Read-only PP recovery/statistic arguments now use const views. Auxiliary
   block descriptors are published directly into MetadataBuffer rather than
   retaining a plain vector-only descriptor table.
5. Large requests still carry both read and write fields, and auxiliary
   selection uses a fixed seven-slot internal table. This is explicit in the
   effects table above, but named sub-bundles would make future device entry
   design clearer. It does not require a new per-row runtime mechanism.
6. The former generic RHS EOS-exception fallback reloaded a neighbor mean after that
   point had already been rotated into the face frame, omitting the periodic
   component transform. This predated the extraction and is now corrected:
   the fallback applies the same vector-component rotation after the reload.
   An explicit side flag also handles periodic self-neighbors; comparing cell
   indices would be insufficient. Ordinary successful property evaluation and
   the earlier admissibility fallback are unchanged. No forced-EOS-failure
   regression test was added; existing checks do not prove branch coverage.
7. LVR face-to-cell sensor accumulation is serial scatter; SGS is an ordered
   scan. The explicit views do not make these algorithms safe for arbitrary
   concurrent row/face execution. GPU/OMP kernels need their own validated
   decomposition/reduction, as already deferred.
8. Extraction initially moved rigid-frame preparation before the fatal RHS
   policy guard. An invalid time could consequently throw from the production
   fatal entry, and the fork-based death test re-entered MPI finalization.
   The facade now applies the selected failure policy to descriptor preparation
   and leaf invocation together. Fatal catches perform only rank-local printing
   and abort. The child test exits unsuccessfully if an exception escapes, so a
   regression fails the assertion instead of hanging. Valid-input arithmetic
   and the checked throwing entry are unchanged.
9. Existing DNDS/Solver algebra and point-sized thermodynamic/BC/Riemann
   arithmetic are not mesh orchestration. They remain explicit callees of the
   free row kernels; they do not acquire field leases. A future device leaf
   must resolve device-compatible physics and boundary policies explicitly.
   Host boundary objects and chemistry workers cannot simply be uploaded with
   the numeric descriptor table.
10. Small descriptor construction still allocates MetadataBuffers per call in
    the generic RHS/source paths. This is equation-sized, outside hot loops,
    and introduces no field packing. Reusing those publications is a possible
    optimization, but must key all payload/layout identities and generations;
    caching only raw addresses would reintroduce stale-view risk.


11. The source status pass removed direct throw/check-throw and exception-handler
    tokens from the Cauchy CPU packages. Property, boundary, source, derived,
    PP, stage, Jacobian and nonlinear callers propagate statuses before accepting
    outputs. Physics adapters alone intercept external exceptions. This is a
    source audit result; grouped build and runtime receipts are recorded in
    the execution ledger.
12. The existing zero-diagonal rejection test exposed a checked block-inverse
    helper beneath a `noexcept` leaf. `TryInvertDiagonalBlock` now returns status
    for its visible input/inverse checks, while the historical checked wrapper
    is retained. Cauchy propagates that status to its operator boundary. The
    negative test keeps its original `CHECK_THROWS` contract at the operator;
    it is not changed to accept process termination.
13. Python's SST physical-timestep forecast exposed a pre-existing mean-only
    workspace gap: P0 face reconstruction was disabled together with the
    physical gradients still needed by the RANS source. Both allocation and
    the gradient evaluation pass now retain those derivatives and their frozen
    BC rows. The existing SA/SST domain test exercises physical-step limiting,
    residual evaluation, implicit advance and restart, including CPU workers.
14. Python Array replacement no longer copies the old storage handle before
    allocating its replacement. This preserves `reset()` after an abandoned
    exclusive shape edit; existing retained views still own the old allocation.
15. Python reducer tests still expected recovery after a callback failed during
    posted MPI completion. That expectation predated the existing fail-fast
    policy and disagreed with the C++ distributed-failure fixture. The Python
    callback failure is now checked in a bounded, independent MPI subprocess;
    ordinary successful callback lifetime and repeated waits remain tested.
    No production transfer error policy was changed for these tests.

### Face-flux physics and row shapes

The C++ wrapper resolves the equation model before looking up a row shape.
Five components can mean compressible laminar flow or ACM with SA. Wilcox
and SST share the two-equation shape and retain the selected model at each
point. CPU dispatch selects the compact catalog only for compatible whole
AoS means, outputs and reconstruction operands. A segmented equation,
mixed physical layout, mixture, or unlisted shape uses the generic instantiation.
The generated CPU and CUDA catalogs share the component counts and modal counts
in `scripts/manifests/cauchy_rhs_shapes.json`. Quadrature order remains an
independent property of the metric.

Mean-only flux samples the cell average. Direct-gradient flux evaluates the
physical gradient against the displacement. Modal flux evaluates the selected
basis and can consume a separate set of viscous coefficients. Each produces
the same point state for the subsequent boundary, inviscid and viscous flux
operations. The cell gather follows the face pass for every mapping. The
explicit CPU `Generic` and historical `FixedFlow5` workspace policies
provide numerical and timing comparison routes.

The CUDA wrapper selects face threads for direct gradients and the pMG
mean-only flux. The face quadrature loop uses the prepared rule in all cases.
Modal flux may select either face threads or prepared CTA segments. ACM and
perfect gas have distinct device bodies. The measured perfect-gas launch
presets and user overrides apply within the eligible modal family.

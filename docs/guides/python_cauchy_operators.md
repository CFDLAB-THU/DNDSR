# Python Cauchy finite-volume operators

Python and the C++ frontend use the same native operators. `Domain` and
`Simulation` provide the production time controllers. `advance_domain` lets
Python direct a standalone implicit step on CPU or on a supported CUDA
configuration. `FiniteVolumeOperators` provides lower-level FV and linear
entrypoints for custom orchestration. The examples below use a CPU stream.

## Driving a single implicit stage in Python

`Cauchy.solve_single_stage(stage_solver, res_other, inverse_dt, rhs_scale, time=..., stage=...)` runs the single-domain nonlinear iteration loop in Python. The prepared native session owns the working state, reconstruction, CFL schedule, selected-component convergence test and linear correction. Python begins each iteration, observes the globally reduced domain residual, advances coupled stage zero if another correction is needed, and publishes the terminal decision. The returned report has the same fields as `stage_solver.solve`. The caller supplies the time-method coefficients and the frozen source term for this stage.

Python can also order a bound halo transfer with `binding.submit_after(engine, comm_stream, producer.record())` and `operation.enqueue_wait(consumer)`. The ordinary MPI-to-CUDA edge currently requires controller progress. A prepared NCCL global reduction returns a CUDA event from `binding.submit(engine, comm_stream)`. Use `consumer.wait_for(event)` before reading its result on that stream. All ranks participate in matching collective order; a local Python reference may be released after submission.

For host Arrays, bind the reduction on an explicitly selected MPI
communication context during collective setup. A compact one-row table can
hold several residual norms. The communication stream below is a CPU lane
even when a CUDA producer stages its result into `local`.

```python
from DNDSR.DNDS import Runtime
context = domain.prepare_reduction_context()
binding = context.bind_allreduce(local, total, Runtime.Reduction.Sum)
communication.wait_for(producer.record())
complete = binding.submit(engine, communication)
consumer.wait_for(complete)  # CPU consumer
```

Submit all bindings sharing that MPI communicator on one lane per rank in
matching order. At collective setup, each participating domain duplicates its
communicator once. Later calls return the same owner. Halo transfers and host
solver collectives keep their existing communicators. All ranks must still
reach the communication needed by a dependent host wait. Separate contexts
isolate MPI matching and cannot resolve cyclic waits in the caller. The submitting MPI controller
progresses the request when it observes an event or progresses the engine.
For CUDA staging, allocate `local` and `total` once with
`device_result.empty_pinned_host_like()` and bind them before entering the loop.
They are independent page-locked host Arrays with the device Array's layout and
element schema. Order D2H before the reduction. Call
`complete.wait()` before importing it on the GPU lane and submitting H2D.
The return-copy event orders later consumers. NCCL provides a native CUDA
dependency when the reduction operands reside on the GPU.

## Step acceptance and failures

`Domain.advance(dt)` and `Simulation.advance(dt)` return a `StepResult` with
`accepted` and `reason`. Always inspect it. A rejected numerical candidate leaves
accepted state, time and method history unchanged; the controller may retry the
whole physical step at a smaller dt. The C++ frontend applies its configured retry
limit; the Python API does not retry automatically.

```python
result = simulation.advance(dt)
if not result.accepted:
    print(result.reason)
    result = simulation.advance(dt * 0.5)
    if not result.accepted:
        raise RuntimeError("Numerical candidate still rejected: " + result.reason)
```

This result represents only a collectively agreed numerical decision. Backend,
lease, MPI, allocation and invalid-input failures are not timestep rejection.
Production error handlers terminate the run independently, without a later
barrier/error reduction or exception-based recovery. An application-level Python
exception also needs a fatal MPI boundary when peers could still be communicating.
The C++ `PrepareStep` result must similarly be checked before `CommitStep`.

## Ownership and access

```text
Python Simulation -> Domain -> accepted EquationDofs
                         ^                 |
                 FiniteVolumeOperators         +-> shared schema and row space
                         |                 +-> physical fields
                         +-> shared FV plans       +-> owned Runtime Array
                         +-> worker capacity       +-> ghost Runtime Array
                         +-> native workspaces          +-> shared allocation
```

`domain.dofs` returns a shallow logical handle. Its `fields` mapping contains
native `DoubleDofField` handles, sharing both owned and ghost allocations.
`clone()` copies owned values into independent storage; it does not refresh
ghosts. `zeros_like()` creates independent residual/update scratch with the
same grouping and physical layouts. Zero-filled scratch need not be an
admissible physical state.

`blocks` describes logical component names. `locate_block(name)` returns
`(physical_field_name, first_component, component_number)`. Physical AoS/SoA/
AoSoA layouts and full parent pitches stay intact. `EquationDofs.from_fields`
explicitly binds ordered native fields and `(block, field, offset)` tuples,
using a prototype's immutable equation schema and row space. It shares storage;
it does not copy values or infer a physical profile from component counts.

The Python operator owner retains its originating Domain. A Domain borrowed
from Simulation retains that Simulation, including its domain communicators.
Workspaces and linear solvers retain the operator owner. Copied Runtime arrays
retain their allocations and metadata independently. Bare native views still
obey the ordinary lease lifetime contract; a NumPy snapshot is a copy.

`reconstruction_guess()`, `flux_reconstruction()` and
`restore_reconstruction()` refer to **modal VR coefficients**. Direct P0/P1
has empty modal slots and rejects those calls; it never returns cell means as
fake coefficients. Direct P1 physical gradients are separate temporary
workspace fields. Use `domain.dofs` or an explicit cloned state to inspect or
change means, and run the normal reconstruction/evaluation operation to prepare
gradients. These are the current Cauchy contracts, not compatibility aliases
for the former split flow5/equation APIs.

```text
NonlinearStageSolver -> FiniteVolumeOperators -> Domain -> optional Simulation
                   -> original state + workspace + engine
                   -> copied native stream
PolynomialMultigrid -> finest NonlinearStageSolver + original mesh
```

Release debug views before conflicting native work. Serialize workspace use,
matrix preparation, worker acquisition and Domain/Simulation mutation. Do not
replace the bound field allocations/shapes; build a new workspace after such
a change. For live Simulation state, advance through Simulation so that
accepted clocks and interface traces remain coordinated.

## Native residual and auxiliary preparation

```python
from DNDSR.Cauchy import FiniteVolumeOperators, AuxiliaryRequest
from DNDSR.DNDS import Runtime as R

operators = FiniteVolumeOperators(domain)
state = domain.dofs.clone()
residual = state.zeros_like()
engine = R.EagerEngine(host_threads=2, tracking=R.AccessTrackingContext())
stream = engine.cpu_stream()
workspace = operators.build_workspace(state)
operators.evaluate(state, residual, workspace, engine, stream, time=domain.time)

request = AuxiliaryRequest()
request.mean_thermodynamics = True
request.face_spectral_radius = True
request.cell_spectral_sum = True
auxiliary = operators.build_auxiliary_values(request)
operators.prepare_correction_aux(state, workspace, auxiliary, engine, stream,
                                 time=domain.time)
```

`evaluate` accepts stale input halos, performs canonical exchanges and
configured core/halo overlap internally, and returns CPU-visible owned output.
It updates input ghosts and reconstruction scratch, never owned input means.
The output halo can remain stale. All domain ranks call collectively in the
same order from an MPI-authorized controller thread. Python releases the GIL
during native work; inline and worker execution share the same C++ sequence.
Return joins the relevant stream. An external pending producer must already
be joined; the wrapper does not infer arbitrary cross-stream dependencies.

For independent flux/source work, prepare without a dummy residual:

```python
operators.prepare_reconstruction(state, workspace, engine, stream, time=domain.time)
face_flux = operators.build_face_flux_buffer(state)
operators.evaluate_flux(state, residual, workspace, face_flux=face_flux, time=domain.time)
operators.add_source(state, residual, workspace, time=domain.time)
```

`evaluate_flux` overwrites owned residual rows; `add_source` adds to initialized
rows. Both are local, synchronous calls on the prepared state/time and perform
no reconstruction or communication. Serialize their workspace use and coordinate
rank-local failures. Changing a prepared mean or coefficient invalidates the
publication. The optional caller-owned face array holds integrated negative
outward flux in the local anchor cell's axes, in the C++ `AuxiliaryFaces()` order;
it contains no source contribution or implicit halo. Release debug views before
the write. Direct P1 consumes physical gradients; constant/P0 face reconstruction
keeps gradient-dependent RANS sources independently active.

For a complete local residual on a CPU or CUDA stream, reuse a completed
preparation and submit flux followed by volume sources together.

```python
operators.prepare_reconstruction(state, workspace, engine, producer, time=domain.time)
ready = operators.submit_prepared_residual(state, residual, workspace, engine, producer,
                                           time=domain.time, source_derivative=derivative)
consumer.wait_for(ready)
# A dependent kernel on consumer can read the owned residual here.
ready.wait()  # Observe numerical status before using the same workspace again.
```

Here `derivative` is the optional array from `build_source_derivative(state)`.
Omit it for a residual-only evaluation. If preparation used external auxiliary
values, supply the same bundle as `auxiliary`. Replay validates owned and ghost
means, reconstruction and the receiver packet at this time. After replacing any
of these inputs, prepare again. The submission computes all owned rows, leaves
output halos alone and retains its operands until retirement. A CUDA replay
copies the two numerical status records to reusable pinned host slots on the
producer stream. Host observation reports execution and numerical errors.

The `Workspace` object is the sole native finite-volume workspace and supports
the equation-state profile, including
P3 VR, component WBAP, Euler reconstruction PP and supported RANS/mixture
transport and sources. Unsupported characteristic/LVR/flux/interface profiles
still fail explicitly until a corresponding prepared operator is available.
For Strang domains, these operators expose the resolved
source-free transport profile, as used inside the Domain.

`build_mean_workspace` / `evaluate_mean` use separate zero-reconstruction
physical-step scratch. `reconstruction_guess` and `flux_reconstruction` return
shared unlimited/filtered coefficient fields. `reset_reconstruction` is local;
`restore_reconstruction` validates and publishes an explicit snapshot
collectively. Retain a clone when preserving values across another evaluation.

`build_mean_gradients` and `prepare_aux_values` expose explicit gradient and
auxiliary preparation independently of reconstruction. Two-equation RANS
uses current mean gradients. `prepare_correction_aux` also refreshes the
equation workspace's frozen source-quadrature publication for a Jacobian,
without advancing its polynomial guess. `validate_aux_values` checks the
current state, time and output versions. A debug write invalidates provenance.
Optional outputs are absent rather than silently allocated by consumers.
When `derived_quantities` is nonempty, a thermodynamics request adds the
required p/a/diffusivity columns to that derived bundle and omits the separate
`mean_thermodynamics` table. Inspect `derived_owned.quantities` for its column
order; `validate_aux_values` accepts either prepared representation.

## Approximate matrices and linear solves

Build a `LinearProblem` from the exact state schema/row space, request block or
entry diagonal and assembled or semi-matrix-free off-diagonals, then call
`prepare_jacobian`. The supplied inverse pseudo-time is an owned scalar Runtime
array. `JacobianPreparation` describes physical mass, RHS scaling and optional
named-selection-resolved component indices. Nonconstant chemical reconstruction
also needs the equation workspace as `source_workspace`.

With `implicit.sourceJacobianMode = "rhsQuadrature"`, create one derivative
buffer per retained rate. Keep each pair together until its last matrix
assembly. Shape-compatible buffers can be reused after that point.

```python
derivative = operators.build_source_derivative(state)
operators.evaluate(state, rate, work, engine, stream, time, source_derivative=derivative)
operators.prepare_correction_aux(state, work, auxiliary, engine, stream, time)
operators.prepare_jacobian(state, auxiliary, inverse_dt, problem, controls, time,
                           source_workspace=work, source_derivative=derivative)
```

For centered linearization, `build_source_derivative` returns `None`.

Preparation is synchronous and local. It consumes ready halos and explicit
auxiliary data and performs no hidden EOS policy selection or halo exchange.
After changing coefficients, explicitly invert diagonals or factor ILU before
their low-level uses. The numerical snapshot follows the existing Euler-style
approximate operator; it is not the complete differentiated nonlinear residual.

For a CUDA matrix, pass `source_workspace=work, stream=producer` to
`prepare_jacobian`. Supply the inverse pseudo-time and auxiliary arrays on the
same device. `problem.invert_diagonal(stream=producer)` returns after checking
the numerical status. Both preparation calls are host observation points.

Python can submit the matrix action and individual Jacobi sweeps on CPU or CUDA.
Each returns a Runtime event for its output rows. For example, after preparing
the input halo and inverse, a caller can enqueue a fixed-point check:

```python
ready = problem.submit_apply(producer, x, rhs)
consumer.wait_for(ready)
done = problem.submit_jacobi(consumer, rhs, corrected, input=x)
done.wait()  # host observation; corrected now contains x to roundoff
```

Omitting `input` selects the zero-start sweep and reads only the owned RHS.
All other sweeps and matrix actions require ready input halos for `All` or
`HaloDependent`. A `Core` submission may overlap input halo transfer. Use
distinct input and output storage and keep the matrix coefficients frozen
through the batch. For a CUDA evaluated off-diagonal operator, enqueue
`reset_action_status(stream)` before the first action and call
`check_action_status(stream)` after the last one. Order both with the actions
using stream events when switching lanes. The check observes the status
reduction on that lane. Assembled matrices and host operators need neither call.

`build_linear_solver` constructs the existing native SGS, left-GMRES or
standalone ILU controller using the canonical halo binder. `solve` is collective
and handles its own input exchanges. Always inspect `report.usable`,
`report.converged` and residuals. `solution` shares reusable scratch; clone it
before another solve if the old values are needed.

Low-level `apply`, `begin_sgs`, `apply_sgs` and `apply_ilu` perform no MPI.
Their caller is responsible for current input ghosts and matching matrix
generations. Staged SGS permits real input exchange during lower Core work:

```python
from DNDSR.Cauchy import LinearRegion

bindings = [operators.metrics.BindHalo(old.fields[name])
            for name in old.field_names]  # physical registry order on all ranks
active = [binding.begin() for binding in bindings]
sweep = problem.begin_sgs(rhs, old, forward, update)
sweep.lower(LinearRegion.Core)
for transfer in active:
    transfer.wait()
sweep.lower(LinearRegion.HaloDependent)
sweep.upper(LinearRegion.HaloDependent)
sweep.upper(LinearRegion.Core)
sweep.finish()
```

The default freezes remote old values across both scans. The optional middle
exchange remains an explicit exposed communication point. Never call phases
concurrently or let forward/output alias input. Abandoning unfinished numerical
work can invalidate scratch; accepted state is not silently repaired.

## Native nonlinear correction and spatial FAS

`build_stage_solver(state, workspace, engine, stream, implicit_json="{}")`
binds the production equation nonlinear controller to its original state and
workspace. It retains those Python owners and the engine; the native stream
value is copied. Optional JSON overrides affect only `implicit` controls and
use the production configuration schema/defaults/validation. Spatial and
physical controls come from the immutable FV plan. Use named
`implicit.residualComponents`; `state.resolve_components` separately resolves
explicit update-component masks for low-level `NonlinearSolveControl`.

`correct_once` is exactly one native pseudo-time/Jacobian/linear/PP update. It
does not contain a nonlinear convergence loop and does not decide acceptance.
`solve` invokes the common native nonlinear loop. Both receive frozen
`source=resOther`, `inverse_dt` and `rhs_scale` in the normalized defect
`resOther + rhs_scale*F(U) - inverse_dt*M*U`. The mass map `M` is identity
for compressible equations and omits the algebraic pressure component in ACM. The physical rate F is volume
normalized; the FV residual entrypoint above is integrated. Do not substitute
the integrated residual directly into that time-equation formula.

`build_multigrid(mesh, stage_solver)` uses the native p-hierarchy and its
configured `implicit.multigrid` controls. The supplied mesh and operator owner
must be the original finest ones. The returned hierarchy retains the finest
solver, its state/workspace/engine owners and the mesh. `apply_cycle` performs
one spatial FAS update using native recursion. For Python control, call
`Cauchy.apply_v_cycle(hierarchy, source, inverse_dt, rhs_scale)`. Python visits
each level, orders pre-smoothing, restriction, coarse smoothing, prolongation
and post-smoothing, and submits the complete sequence to `apply_schedule`.
The native calls use the same configured smoothing counts, form the FAS
forcing on restriction, and apply the child correction relative to its saved
entrance state on prolongation. Submit the same sequence on all domain ranks.
The call releases the GIL and returns when the cycle completes. `solve` uses
the shared finest convergence controller.
Keep other-coupledStage source terms frozen throughout the complete cycle.
Invoke separate cycles per coupledStage for DITR; no temporal hierarchy is
introduced. A caller building its own iteration loop owns CFL progression,
convergence and trial rollback. Native Domain/Simulation remain the simple
way to obtain method-owned stage/step history and accepted-step transactions.

## Standalone physical steps

`Cauchy.advance_domain(domain, dt)` runs the standalone time-group and
nonlinear iteration loops in Python. In DITR it corrects and refreshes the
midpoint, then the endpoint, before observing I+H. A serialized DIRK group
observes before a correction. The domain retains its method history, time
equations, reconstruction warm starts, CFL schedule and restart cursor. On
numerical rejection the controller discards the trial and returns the reason.
After acceptance it publishes the step. Call on all ranks of the domain.
Connected domains require simulation-wide coordination of interface refreshes.

## Registered multidomain iteration policies

Importing `DNDSR.Cauchy` registers two additional choices for
`defaults.implicit.updatePolicy`. Use `pythonSingleStageImplicit` for BDF or
an implicit DIRK stage, or `pythonDITRSequential2Stage` for DITR. Then call
`simulation.advance(dt)` on all control ranks. Each Python loop posts the
configured interface refresh, advances the domains on its rank, finishes
the publication, and agrees termination across domains. DITR does this for
the midpoint before it samples the endpoint. Stage-frozen packets follow the
existing stage cadence. The native FV operators still import the publication
event at their first dependent operation.

The group handle expires when the policy returns and can be used only by its
MPI controller thread. The bindings release the GIL for numerical work,
transfers and waits. Restart metadata records the selected policy name. Use
the same registration and configuration when resuming an exact checkpoint.
The standalone C++ executable registers its native policies.

## Queued BDF and DIRK equations

For a custom time controller, keep the accepted states and stage rates as
`EquationDofs` and pass the compute stream to `Cauchy.BDFEquations` or
`Cauchy.DIRKEquations`. Both return named `inverse_dt`, `rhs_scale` and
`time` coefficients for the native nonlinear solver. BDF assembles its source
from the current step's initial state and, for BDF2, the preceding accepted
state and interval. Replace history only after step acceptance. Omitting
history gives BDF1 startup. Steady BDF1 has a zero source and inverse dt.

For DIRK, call `stage_source(out, initial, rates, time, dt)` in stage order.
An empty rate list selects stage zero. Save its explicit spatial rate, then
supply one more completed rate for each following implicit stage. The method
reads the same tableau as C++, including the six-stage ESDIRK4 constants.
Only state terms receive the physical mass matrix. Python controls the
history lifetime and restart persistence when using these low-level operations.

## Queued DITR equations

`Cauchy.DITREquations(stream, fraction, theta, variant)` uses the native
U2R1/U2R2 weights. Pass the resolved time settings from the case. With caller-owned
means and volume-normalized rates, `midpoint_source` and `endpoint_source`
write their respective frozen source to the supplied destination and return the
multiplier of the unknown stage's spatial rate. After both corrections and rate
refreshes, `stopping_residual` writes I+H. These calls enqueue native operations
on the supplied stream. Before reading a field on the host, call
`stream.record().wait()`. This algebra can also feed another computation on the
same stream. The solver's stage-specific convergence and ACM continuity gates
remain separate from equation assembly.

## Scope and verification

The Python interface is a thin operator composition boundary, not a binding of
every internal point kernel or a new Python callback nonlinear controller.
Method-owned ODE history and accepted-step transactions still run through native
Domain/Simulation. The binding tests compare nonconstant P3/WBAP residuals
against the Domain path, compose explicit auxiliary/Jacobian/matrix/linear work,
and compare staged/unsplit SGS with actual active MPI, one native correction and
a spatial FAS cycle. Independent physical
and matrix oracles remain in C++ tests. The standalone DITR restart comparison
runs with two MPI ranks on CPU and one rank on CUDA. Its device case checks
that accepted flow fields reside on the GPU and copies them to host only
for comparison with the native controller.

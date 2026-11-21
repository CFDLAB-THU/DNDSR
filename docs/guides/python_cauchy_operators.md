# Python Cauchy finite-volume operators

Python and the C++ frontend use the same native CPU operators. `Domain` and
`Simulation` remain the production time controllers. `FiniteVolumeOperators`
provides the existing lower-level FV and linear entrypoints for testing,
tuning and custom orchestration; it does not implement another numerical loop.
The native solver also has CUDA operators for supported equation and storage
combinations. The examples below exercise the CPU stream.

## Driving a single implicit stage in Python

`Cauchy.solve_single_stage(stage_solver, res_other, inverse_dt, rhs_scale, time=..., stage=...)` runs the single-domain nonlinear iteration loop in Python. The prepared native session owns the working state, reconstruction, CFL schedule, selected-component convergence test and linear correction. Python begins each iteration, observes the globally reduced domain residual, advances coupled stage zero if another correction is needed, and publishes the terminal decision. The returned report has the same fields as `stage_solver.solve`. The caller supplies the time-method coefficients and the frozen source term for this stage.

Python can also order a bound halo transfer with `binding.submit_after(engine, comm_stream, producer.record())` and `operation.enqueue_wait(consumer)`. The ordinary MPI-to-CUDA edge currently requires controller progress. A prepared NCCL global reduction returns a CUDA event from `binding.submit(engine, comm_stream)`. Use `consumer.wait_for(event)` before reading its result on that stream. All ranks participate in matching collective order; a local Python reference may be released after submission.

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
and matrix oracles remain in C++ tests. None of this establishes Cauchy GPU
execution or a new physical validation campaign.

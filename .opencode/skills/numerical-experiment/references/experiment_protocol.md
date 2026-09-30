# DNDSR Numerical Experiment Protocol

## Workspace and Data Layout

Use these ownership boundaries:

| Location | Purpose | Expected Git ownership |
|---|---|---|
| `<repo>/workspace/<workspace_name>/` | Plans, scripts, config variants, manifests, compact derived data, plots, conclusions | Independent Git repository; not a submodule |
| `<repo>/data/...` | Meshes and ordinary raw solver output following case config conventions | DNDSR data layout; do not copy large runs into the experiment repository |
| `<repo>/cases/...` | Maintained case definitions | DNDSR repository; edit narrowly and preserve comments/notes |
| `<repo>/build.../` | Binaries and build products | Untracked build state |

Do not assume an existing directory under `workspace/` is independent. Verify with `git -C <workspace> rev-parse --show-toplevel`; initialize it only if creating a new workspace or if the user explicitly wants an existing directory converted. Never add the workspace as a DNDSR submodule. Do not commit automatically.

Keep a plan and a run manifest in the workspace. A useful run record contains:

- experiment/run identifier and hypothesis;
- runner hostname or remote label, scheduler allocation if any, `np`, thread counts, and relevant CPU/GPU limits;
- repository commit and branch plus a statement/hash of relevant dirty changes;
- executable path and hash when practical;
- exact invocation working directory, command, and relevant non-secret environment variables;
- config source, copied experiment config, and hash;
- mesh/mechanism identifiers and hashes when practical;
- start/end timestamps, exit status, wall time, and stopping reason;
- raw output location and compact artifact paths;
- pass/fail criteria, observed metrics, and anomalies.

Never record credentials or secret environment values.

## Runner Contract

For `local`, confirm the requested allocation is compatible with visible resources and existing load before launch. Do not seize all CPUs merely because they exist.

For a remote runner, establish the host label, repository/build location, invocation mechanism, scheduler requirements, allocation, and artifact-transfer path. Use non-interactive commands and existing credentials; do not expose secrets. A remote run must use a source/config state tied to the local record. Record whether outputs were generated remotely and which compact artifacts were fetched.

For MPI experiments, start the first timed run at the maximum rank count authorized by the user. Reduce the rank count only when the runner cannot supply that allocation, the executable or mesh imposes a stricter limit, or measured scaling shows that fewer ranks are preferable; record the reason.

No solver launch is authorized by a direction alone: runner and `np` or CPU limit must also be known.

## Runtime Budget and Pilot Design

Every normal result-producing trial targets less than 10 minutes wall time. Longer production campaigns require the user to knowingly expand that budget; they are not normal trials.

Before a trial, estimate:

`predicted_wall = observed_pilot_wall * target_steps / pilot_steps * scaling_factor`

Use at least one measured comparable quantity when available. If none exists, run a deliberately short pilot that cannot exceed a small fraction of the 10-minute budget. Pilot outputs are diagnostic only unless their horizon independently satisfies the scientific criterion.

Bound both levels of iteration:

- physical horizon: `nTimeStep`, final time, or equivalent;
- per-step nonlinear/pseudo-time effort: `nTimeStepInternal`, Newton iterations, linear iterations, or equivalent;
- convergence: residual/internal threshold, including whether reaching it is required;
- I/O: output interval and retained fields so writing does not dominate runtime or storage.

State the stop conditions before launch. Stop or decline to retry when the estimate exceeds the authorized budget, progress stalls beyond the declared criterion, numerical state becomes non-finite/unphysical, or repeated runs fail for the same unresolved reason.

## Monitoring

Use Luna by default, or the user's named available low-cost model, as a monitoring subagent when the expected run is long enough that unattended failure would waste material time. Give it only the exact job/process/log targets, expected progress markers, timeout, and stop/escalation criteria. Monitoring must not start new cases, change configs, kill unrelated jobs, or retry silently.

The primary agent may wait for the polling subagent for at least 20 minutes when waiting on an explicitly authorized long run, and may use a longer bounded wait when the user explicitly authorizes it. Retain the ordinary 10-minute wait bound for normal trials. Prefer one appropriately long wait over frequent short polls; the monitor should still return early on completion, failure, or a stop-condition breach.

The primary agent remains responsible for interpreting completion and numerical validity. A clean exit proves execution, not correctness.

## Slow-Trial Adaptation Ladder

When the predicted wall time is too high, evaluate these options and change the smallest number of experimental factors:

1. **More cores.** Prefer this first when the runner has an authorized allocation and the case is expected to scale. Confirm that MPI/thread oversubscription is absent.
2. **Smaller mesh.** Generate a smaller mesh directly only when geometry and meshing are simple and reproducible. Ask the user before simplifying a complex geometry. Preserve relevant physical layers/features and state the resolution loss.
3. **Coarser physical time step.** Treat this as a change in the numerical method. Check stability, splitting/truncation error, propagation speed, and transient phase accuracy against a finer-step reference.
4. **Relaxed internal threshold or capped internal steps.** Record whether each physical step actually reaches its requested convergence threshold. Non-converged inner solves can bias the result and must not be described as equivalent.
5. **CFL tuning.** Tune within a short bracket using convergence rate and robustness, not the largest stable value alone. Separate pseudo-time CFL effects from physical-step changes.

Mesh scale and physical time step directly affect output quality and numerical behavior. Whenever either changes, repeat the key observable and at least one local profile/invariant comparison before accepting the accelerated setup.

## Evidence and Reporting

Prefer compact, auditable artifacts:

- CSV/JSON tables of time histories, critical locations, extrema, integrated quantities, convergence summaries, and fitted rates;
- profiles containing coordinates, units, field names, time, and source run ID;
- plots generated by tracked scripts with captions that identify run IDs and normalization;
- a report that separates configuration, observation, interpretation, and limitations.

For numerical comparisons, use identical controls unless the changed control is the factor being studied. Sanity-check dimensional and nondimensional invariants, conservation, positivity, expected propagation direction/rate, mesh/time-step sensitivity, and agreement with reference solutions where available.

Label outcomes as:

- **verified:** the declared run and numerical checks completed;
- **partial:** useful evidence exists but a required horizon/check did not complete;
- **failed:** execution or numerical acceptance criterion failed;
- **unrun:** planned but not launched, including cases blocked on runner/allocation.

---
name: cluster-run
description: Prepare, submit, monitor, and record jobs on Slurm or another batch-scheduled cluster. Use when compilation or execution must obtain compute resources through a scheduler; use numerical-experiment instead for local or direct-SSH runs without a scheduler.
---

# Cluster Run

Run reproducible jobs through a scheduler without treating a login node like a
compute node or losing source provenance on a slow shared filesystem.

## Routing Boundary

Use this skill when the machine requires `sbatch`, `srun`, `salloc`, or an
equivalent scheduler. Use `numerical-experiment` for local or direct-SSH
execution. For scheduled numerical experiments, use both skills: this skill
owns cluster preparation and job lifecycle; `numerical-experiment` owns the
scientific plan, runtime bounds, data placement, and result validation.

## Required Cluster Contract

Before compiling or submitting, establish:

- cluster SSH alias and scheduler;
- the user-provided canonical DNDSR root, normally `~/<name>/DNDSR`, and its
  matching build directory; ask for `<name>` or the full path rather than
  inventing a personal or task-specific checkout root;
- desired source commit or branch;
- whether compilation is allowed on login nodes or must be scheduled;
- partition, account/QoS if needed, nodes, tasks, CPUs per task, GPUs, memory,
  wall time, and threading model;
- launcher or submission template, executable command, working directory, log
  directory, and output location.

If fields are missing, inspect and prepare without launching. Ask at the first
operation whose correctness depends on the missing value.

## Subagent Policy

When subagents are available, dispatch regular or mechanical cluster work to
Luna, or to the user's specified available low-cost model. This includes file
transfer, remote checkout or repository updates, compilation, job launch, and
polling. Give each dispatch the cluster profile, exact paths, desired source
state, scheduler request, bounds, stopping conditions, and evidence to return.

For one continuous cluster task, reuse the same subagent thread for successive
mechanical actions such as remote preparation, submission, polling, fetch, and
recording. Start a new thread only when the prior one is unavailable or a
separate independent task needs parallel work; state why continuity was not
possible.

Use a maximum wait of 20 minutes for one ordinary subagent polling round. For
an explicitly authorized long-running task expected to finish within one hour,
one polling round may wait up to one hour. If the job is expected to run for
more than one hour, the main agent defaults to stopping active checks after
verifying that it is safely detached. Report the detached state, scheduler job
ID or process handle, log and record paths, last verified progress, stop
conditions, and ETA; resume polling only in a later turn or when the user
explicitly requests it. A polling timeout is only an observation timeout and
never authorizes restarting the job. Subagent completion is not a persistent
hook that can wake a main agent after its turn has ended, so do not rely on it
for detached-job reporting. These polling limits do not extend the job's
declared wall time.

For source-state, scheduler, numerical, or provenance audits, use the default
subagent model rather than deliberately selecting a cheaper model, unless the
user specifies otherwise. The main agent retains source-state and authorization
decisions, technical reasoning, result interpretation, and final verification;
delegation does not authorize checkout mutation, job submission, cancellation,
or other external side effects by itself.

## Validation and Submission Gates

- Keep dry runs allocation-free. A dry run may inspect files, validate schemas
  and paths, construct commands, and check scheduler scripts on the login node,
  but it must not invoke the solver, MPI launcher, `srun`, `salloc`, `sbatch`,
  or any other compute workload.
- Reuse relevant pilot evidence from a local machine or a direct-SSH runner.
  Do not schedule another cluster pilot when that evidence already validates
  the source, executable behavior, case setup, and resource estimate needed for
  the production launch. Use a cluster pilot only for a material
  platform-specific uncertainty that existing evidence cannot resolve.
- For numerous production submissions, submit one intended production member
  first. Confirm from its scheduler state, manifest, and application log that
  it has entered the solver and is making healthy progress. Only then submit
  the remaining array or batch. This first member belongs to the production
  matrix; do not create an extra pilot run. If it fails, diagnose and retry
  only that member before releasing the rest.

## Workflow

1. Read [the cluster protocol](references/cluster_protocol.md). If the chosen
   cluster has a profile, read it and revalidate its time-sensitive facts.
2. Bundle read-only discovery into one SSH call. Scheduler state and remote
   filesystems can be slow; avoid repeated Git status, recursive searches, and
   many short logins.
3. Build and smoke-test the required target locally or on an authorized
   direct-SSH runner at the desired source and config state before spending
   cluster allocation time. Treat that evidence as the pilot when it covers
   the production question; do not repeat it through the scheduler.
4. Inspect the remote checkout once. By default, update the established remote
   checkout rather than creating another clone. Never silently destroy dirty
   work; follow the source-state decision tree in the protocol.
5. Treat the canonical checkout as the owner of the cluster's one shared
   `venv/` and populated `external/` dependencies. Use the build directory
   paired with each source checkout, but do not create another venv or rebuild
   another private copy of externals for a standalone build/worktree. Link or
   reference the verified canonical resources as described in the protocol.
   Reuse existing build configuration and build only required targets unless
   reconfiguration is demonstrably necessary.
6. Compile only where the cluster contract allows it. If compute allocation is
   required, compile inside `srun` or `salloc`, never on the login node.
7. Materialize a run-specific submission script and config record. Print the
   absolute log directory, exact command, source commit, and scheduler request
   before submission. For a multi-job campaign, submit and confirm one
   production member before submitting the remainder. Capture every scheduler
   job ID and never infer success from submission alone.
8. Monitor by job ID with bounded, low-frequency polls. Distinguish queued,
   running, completed, timed out, cancelled, and failed states; a zero exit
   status does not establish numerical correctness.
9. Fetch compact logs, manifests, profiles, tables, and plots. Leave ordinary
   raw solver output in the configured remote data tree.

## Source-State Rule

Resolve the desired commit before changing the remote checkout. If it is clean,
fetch once and move it to that commit. If it is dirty:

- when every dirty or colliding file is byte-identical to the desired commit,
  record that comparison and align the checkout to the commit;
- otherwise, stop and ask whether to stash the changes or create an isolated
  checkout/worktree.

Do not infer that similarly named edits are already committed. For an isolated
checkout, keep source and build paired and link the venv and populated
dependencies from the canonical checkout only after verifying each target.

## Canonical Python Environment

The user supplies the canonical DNDSR root, normally `~/<name>/DNDSR`. Its
`venv/` is the cluster-wide DNDSR Python environment, and its populated
`external/` tree is the dependency source for canonical and standalone builds.
Do not create another DNDSR venv or independently download/build the same
externals in an isolated checkout.

Do not install the DNDSR package into that venv with `pip install .` or
`pip install -e .`. For a Python-interface job:

1. Configure the checkout-specific build with the canonical venv's Python.
2. Build `dnds_pybind11`, `geom_pybind11`, `cfv_pybind11`, and
   `eulerP_pybind11` as required.
3. Run `cmake --install <build> --component py`; this places the extension
   modules in that checkout's `python/DNDSR/`, outside the venv.
4. Launch with the canonical interpreter and the active checkout on the module
   path, for example:

   ```bash
   PYTHONPATH=<checkout>/python ~/<name>/DNDSR/venv/bin/python <script.py>
   ```

After changing commits or C++ binding sources, rebuild and reinstall the
bindings before the Python run. Keep `PYTHONPATH` tied to the active checkout
so a standalone build cannot silently import the canonical checkout's modules.

## Cluster Profiles

- For `ssh thtj1`, read [references/thtj1.md](references/thtj1.md). Its DNDSR
  compilation must use `srun` on `debug6`; do not compile on the login node.

## Safety

- Do not submit, cancel, requeue, or mutate a remote checkout without the
  authorization required for that action.
- Do not put credentials, tokens, or private environment values in job scripts
  or logs.
- Do not reuse a build directory configured for a different source tree.
- Do not copy a cluster example script blindly: reconcile its partition,
  task/thread layout, time limit, environment, output path, and launcher.
- If repeated scheduler or remote-filesystem failures prevent progress, report
  the exact job/command and pause instead of retrying indefinitely.

# Cauchy output contract

Cauchy constructs one `OutputCoordinator` before stepping. It owns the
immutable `OutputPlan`, progress controllers, logger, append cursors and
observer lifetime; the executable retains only timestep selection, retry and
accepted-step control. The plan owns cadence inheritance, deterministic names,
exact physical-time targets and the schemas advertised by manifests. Numerical
operators publish typed progress; they do not open files or select output
cadence.

## Artifact bundle

For prefix `run`, domain `D`, accepted step `S` and exact physical time `T`, the
frontend writes:

```text
run-resolved.json
run-run-manifest.json
run-D<domain>-history.csv
run-D<domain>-history_1.csv ...
run-events-rank<global-rank>.log         # rank 0 and domain progress writers
run-D<domain>-S<step>.vtkhdf
run-D<domain>-T<round-trip-time>.vtkhdf
run-D<domain>-S<step>_surface_rank<domain-rank>.csv
run-D<domain>-S<step>_<stage>_<iteration>.vtkhdf
run-D0-S<step>_<stage>_<iteration>.h5  # exact internal, single domain
run-S<step>.h5                 # multi-domain simulation archive
run-D0-S<step>.h5              # single-domain archive
run-S<step>-manifest.json
run-summary.json
```

`-final` replaces the step/time token for the unconditional final checkpoint
and visualization. A snapshot manifest records the accepted step and time plus
checkpoint, volume and surface-shard paths. The run manifest records the
detailed-history schema, status enumeration and logical component names once,
so CSV rows remain compact numeric data.

The canonical numerical history remains one combined segmented `DNDS::CsvLog`
stream per domain with a typed schema. It contains nonlinear iterations,
stage ends and physical-step rows. Force integrations remain the additional
`force_x`, `force_y`, `force_z` columns on step rows, summed over
`output.forceZones`. Values use `output.historyPrecision` (default 10 digits
after the decimal, as in Euler), independently of the shorter CLI display.
The event log is not a numerical data substitute.

Segments repeat the same header. `stage=-1,iteration=-1` denotes an accepted
step; `iteration=-1` with a nonnegative stage denotes a stage end. Empty cells
mean that a quantity does not apply, never false zero. Console messages and
CSV rows use the same `ProgressOutputController` reports.

The current force query integrates the last completed RHS wall momentum flux,
including numerical pressure and viscous/RANS traction, in mesh axes with sign
`pressure*n - traction`. It can select individual wall zones, but the frontend
currently writes only their configured aggregate. **Deferred:** add Euler-style
per-BC selection and segmented integration files for force/moment and mass,
energy or species flows. Reuse frozen quadrature outputs and extend reduction
payloads as needed; output writers must not repeat BC/EOS arithmetic. This
extension is annotated at the force API and history emission point.

Each domain's communicator root writes its history and console progress.
In dislocated placement this can be a nonzero world rank, so the output
coordinator explicitly enables its local console sink. World rank zero also
retains its operational logger. No residual gathering or new collective is
needed for printing. Either the step or implicit console interval can enable
the sink; their individual cadences still control which records are printed.
Monitoring must read all numbered CSV segments, not only the initial file.
Iteration `wall` records observation time, not attempt start or flush time.
Captured rows are emitted at each existing quiescent nonlinear boundary even
if internal checkpoints/visualizations are disabled. This local publication
adds no collective; production sink exceptions propagate to the abort handler.

## Controls

The main controls are:

```text
output.checkpointInterval
output.visualizationInterval            # -1 inherits checkpoint cadence
output.checkpointTimeInterval
output.visualizationTimeInterval        # -1 inherits checkpoint time cadence
output.endpointSnapRelativeTolerance
output.consoleInterval
output.consoleColor                      # auto, always, never
output.consoleBold                       # auto, always, never (independent)
output.eventLog                          # true by default
output.historyInterval
output.historySegmentRows
output.historyPrecision
output.implicit.consoleInterval
output.implicit.historyInterval
output.diagnostics.{perRank,level,maxBytes,rotatedFiles}
```

The deprecated `output.interval` spelling is accepted only when the new
checkpoint spelling is absent. `spdlog` provides synchronous rank-zero console
and optional rotating per-rank operational diagnostics. It is not used for
scientific CSV, MPI aggregation, checkpoints or recovery.

By default the same centralized CLI messages also go to a rotating plain-text
`prefix-events-rank<R>.log` for world rank zero and each domain's progress
writer. `output.diagnostics.perRank` additionally enables diagnostic files on
every rank. Both file sinks use timestamped plain text and the configured
diagnostics level/rotation limits; formatting adds ANSI codes only in the
terminal sink. Files flush on ordinary info messages for live monitoring.
No `tee` is required; Slurm stdout capture can remain a separate duplicate.
Messages printed directly by external libraries or early mesh ingress are not
intercepted by this logger.

`--color always --bold always` forces readable styled progress even through a
pipe. `--color never --bold never` disables all terminal styling. Both flags
override their named `output` settings, after ordinary `-k`/`-v` overrides.
Auto styling uses terminal capability detection; numerical files stay plain.

Each physical-step summary includes stage/cap counts, nonlinear/linear update
counts, retries and its own elapsed step time, in addition to total run wall
time. The step timer starts at the first attempt and includes retries and
iteration progress; it excludes timestep selection, subsequent accepted-state
diagnostics and file output. It is the domain-root host elapsed time, not a
new collective maximum or GPU event measurement. Existing CSV `wall` remains
the cumulative elapsed observation time; its schema and append cursor remain
unchanged. The new CLI timer introduces no device fence or timing reduction.

## Exact-time stepping

Positive time cadences and `time.tEnd` are mandatory endpoints. Event times are
computed from an integer ordinal and interval. With proposed step `dt`, current
time `t`, target `target` and configured relative tolerance `r`, Cauchy snaps
when

```text
target - t <= dt + r*dt.
```

The submitted step uses exactly `target-t`; after acceptance the clock is set
to `target`, rather than incremented. If an unsnapped `t+dt` cannot advance the
floating-point clock, the frontend fails explicitly. Step- and time-based
triggers at the same accepted state are coalesced into one snapshot bundle.

## Internal-iteration boundary

Iteration and stage progress are fully represented in the detailed history and
console. A positive `output.implicit.visualizationInterval` writes the current
trial at a quiescent nonlinear observation boundary. The observer copies the
trial mean and the current warm reconstruction into detached storage, then
prepares output fields with a separate workspace. Its halo publication,
reconstruction iterations and diagnostics therefore cannot rebind or mutate
the active time-stepper state. Each per-domain product has an
`internal-snapshot.v1` manifest whose `restartScope` is `visualization_only`
when no checkpoint is written at that boundary.

A positive `output.implicit.checkpointInterval` enables exact internal
checkpoints for a single implicit domain. The archive stores the accepted state
and history, active ODE method payloads and cursor, current nonlinear controller
JSON, trial state, endpoint reconstruction publications and limited-VR history.
DITR additionally stores its midpoint state/rates and independent midpoint
reconstruction publications. BDF2, serialized ESDIRK and DITR restore the
active solve at that boundary rather than restarting its stage; continuation
tests compare the following nonlinear result and accepted step with an
uninterrupted run.

The active time payload uses schema 2 and includes DITR's variant, stage
fraction and transformed-equation weight. Exact internal restore rejects a
different mathematical signature before replacing live stage state; it never
reuses saved stages/rates with new coefficients. Older active payloads without
this signature are rejected. Accepted-step restart remains separately flexible
because it starts a new step rather than resuming an existing stage equation.

Exact internal output is rejected for SSPRK3 and, currently, for multi-domain
simulations. A domain callback cannot publish a truthful simulation-wide exact
snapshot while another colocated or dislocated domain is at a different
nonlinear boundary. Multi-domain exact restart therefore requires a future
simulation-level coordinated quiescent boundary; Cauchy fails configuration
instead of emitting a partial archive. Internal visualization remains
available independently for every implicit domain.

An internal observer is a domain-quiescent callback, not a simulation-wide
barrier. It flushes and broadcasts only that domain's CSV cursor on the domain
communicator. It must not enter simulation-control collectives: dislocated
domains may reach different nonlinear iteration counts. Accepted-step and final
snapshots are coordinated by `OutputCoordinator` after `Simulation::Advance`
and may therefore capture all domain cursors with simulation-wide collectives.

Checkpoint controller metadata includes exact-time event ordinals, resolved
cadence and typed CSV segment cursors. Continuing with the same output prefix
validates and appends at those cursors without duplicating the initial/accepted
row. Choosing a new prefix starts a new output bundle from the restored
numerical state.

## Asynchronous accepted-step output

`output.asynchronous` defaults to `false`. When enabled, `OutputCoordinator`
uses one FIFO writer with capacity for two accepted-step output batches,
including the currently running batch. It waits for capacity **before** making
another snapshot. Capturing fields, numerical output preparation and required
device-to-host transfers remain ordered controller work; subsequent computation
can overlap file writing. Enqueue does not mean the files are complete.

`PreparedCheckpoint` owns the accepted means, optional reconstruction/history,
nonlinear/control metadata, persistent IDs and mesh envelope. CPU snapshots
are independent copies; device payloads use accounted pinned host allocations.
`PreparedVisualization` owns prepared cell fields and serialized surface rows,
and retains a mesh read until writing finishes. No live Domain, coefficient
buffer or physics worker is borrowed by the writer. Mesh mutation or external
CGNS/HDF5 operations must first drain this service.

The current backend is parallel HDF5 and **requires MPI_THREAD_MULTIPLE**.
FUNNELED/SERIALIZED configurations reject this option explicitly. Dedicated
congruent control/domain communicators are duplicated once at service setup
and shared by requests, separating I/O collectives from solver collectives.
SerializerH5 file sessions and Geom VTKHDF transactions share a process-local
recursive lock, including builds with HDF5 thread safety disabled. This lock
does not order distributed collectives and does not cover external raw
HDF5/CGNS calls; callers must respect the drain boundary.

Accepted-step schemas, per-domain shards and naming stay unchanged. The worker
publishes completion manifests only after every required file has closed
successfully across its ranks/domains. Exceptions from collective file paths
trigger an independent abort, with no collective exception recovery. Internal
nonlinear output retains its synchronous semantics after draining older work.
Final output and shutdown also drain; the final summary never reports merely
queued files as complete. The Phase C implementation remains subject to its
grouped build and distributed validation gate.


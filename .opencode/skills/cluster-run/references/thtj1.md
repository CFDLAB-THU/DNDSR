# `thtj1` Slurm Profile

This profile was inspected on 2026-09-11. Re-query scheduler state and remote
Git status before every use; they are not static facts.

## Observed Layout

- SSH alias: `thtj1`; resolved login host: `th-ex-ln1`.
- DNDSR checkout: `~/zhy/DNDSR`.
- Required matching build directory: `~/zhy/DNDSR/build`.
- Build configuration: CMake `Release`, GCC 12.2 OpenMPI wrapper, Ninja-aware;
  invoke it through `cmake --build build` rather than choosing a generator.
- Environment: `~/zhy/DNDSR/running/thtj1_running_var.sh` configures UCX/GLEX
  and OpenMPI transport variables.
- Example solver launcher: `~/zhy/DNDSR/running/srunApp.sh`.

The checkout was on `dev/harry` at `51902b6c93e20907a6a61db098e98d81ab9c12f1`
and dirty when inspected. This is only a warning to re-check; do not act on the
stale branch, commit, or dirty-file list.

## Observed Slurm Resources

Slurm version was 22.05.2.

| Partition | Nodes | CPUs/node | Memory/node | Time limit | Sharing |
|---|---:|---:|---:|---:|---|
| `debug6` | 26 | 56 | 250000 MB | 30 minutes | exclusive |
| `cp6` | approximately 2874 | at least 56 | at least 250000 MB | unlimited | cluster-defined |

No generic GPU resource was reported for these partitions. Confirm account,
node state, exclusions, and availability at submission time.

## Compilation Rule

Do not compile DNDSR on the login node. First compile and smoke-test locally.
When the remote binary must be updated, use the existing checkout and `build/`
inside a `debug6` `srun` allocation:

```bash
ssh thtj1
cd ~/zhy/DNDSR
srun --partition=debug6 --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --time=00:30:00 \
  bash -lc 'source ~/zhy/DNDSR/running/thtj1_running_var.sh && \
            cd ~/zhy/DNDSR && \
            cmake --build build --target eulerEX -j16'
```

Replace only the target and build CPU count. Keep `-j` no larger than
`--cpus-per-task`. If the build cannot complete within 30 minutes, stop and ask
before using another partition or strategy.

## Example Launcher Audit

The observed `running/srunApp.sh`:

- defaults to `cp6`, one node, and 56 tasks per node;
- contains node-exclusion lists that can become stale;
- hard-codes `OMP_NUM_THREADS=32`;
- sets `DNDS_DISABLE_ASYNC_MPI=1` because this MPI stack does not support the
  required `MPI_THREAD_MULTIPLE` behavior;
- sources `running/thtj1_running_var.sh` and launches with `mpirun "$@"`;
- writes a `.job_record/<job_id>` entry.

Treat it as a cluster-specific example, not a ready-made universal script. A
pure-MPI job must not combine 56 MPI ranks with 32 OpenMP threads per rank.
Create a run-local copy or equivalent script with consistent tasks,
`--cpus-per-task`, and `OMP_NUM_THREADS`. Preserve the transport environment and
the `DNDS_DISABLE_ASYNC_MPI=1` workaround unless a deliberate test proves a
different setup.

For a short scheduled run, a safe script derived from the example has this
shape:

```bash
#!/bin/bash
#SBATCH --partition=debug6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=<absolute-log-dir>/slurm-%j.out

source ~/zhy/DNDSR/running/thtj1_running_var.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export DNDS_DISABLE_ASYNC_MPI=1
cd ~/zhy/DNDSR
echo "JOBID=${SLURM_JOB_ID} HOST=$(hostname) PWD=$PWD"
mpirun "$@"
```

Before submission, print the absolute log directory and preserve this rendered
script with the run record. Increase rank count, nodes, or time only from the
user-approved allocation; use the maximum allowed rank count for the first
timed numerical run when the numerical-experiment protocol applies.

## Checkout and Dependency Notes

Default to updating `~/zhy/DNDSR` after the source-state checks in the general
protocol. Do not use some other existing build directory. If dirty work cannot
be stashed and the user chooses an isolated checkout/worktree, create a matching
build directory there and link populated dependencies from the established
checkout. The inspected checkout had a populated `external/cfd_externals`.
Header-only dependencies appear as per-library directories under `external/`,
not as one verified `external/headeronlys` directory; probe and link only the
actual directories required by CMake.

## Accidental `.json.txt` Artifact

The fetched `cases/eulerSA3D/eulerSA3D_config_CRM.json.txt` is not a CRM config.
It is Slurm job `11015570` stdout: environment variables, a listing of the
`running/` directory, and an `mpirun` no-command error. Do not pass it to the
solver or compare it as a configuration variant.

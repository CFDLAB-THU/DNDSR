# Plan 61-2 CUDA optimization

Design and execution evidence are maintained separately in the planning repo:
`workspace/dndsr-next-plan/docs/61-2-cuda-kernel-optimization.md`.
The baseline is periodic 64³ TGV, P3 reconstruction and degree-5 quadrature,
uncached basis, non-overlap, combined FP64 AoS, eight fixed linear block-Jacobi
updates per nonlinear update. VR sweep count is a separate control.

## Timeline instrumentation

Build with `DNDS_USE_CALIPER=ON` and point `caliper_DIR` at the pinned external
installation's `share/cmake/caliper`. Set `DNDS_TIMELINE=1` at process startup
to enable `DNDS::TimelineScope` Caliper regions and typed integer context.
Caliper works with CPU-only builds too; builds without it retain a no-op API.
Caliper is a compiled optional library, placed under `external/`; dependency
bundle inclusion remains a separate decision. No numerical kernel calls NVTX.

For reports without Nsight, use `CALI_CONFIG=runtime-report` together with
`DNDS_TIMELINE=1`. For Nsight correlation use Caliper's NVTX service through
`CALI_SERVICES_ENABLE=nvtx`, then a bounded
`nsys profile --trace=cuda,nvtx,osrt` capture and export both
kernel execution and CUDA API/copy summaries. Record profiler/toolkit versions.

Ranges identify step and stage, nonlinear update/residual observation, FV
evaluation, auxiliary/VR/RHS submission, polynomial PP, Jacobian setup, linear
solve/Jacobi sweep and Runtime CUDA launch callback. Integer payloads identify
step/stage/iteration when applicable. CPU ranges delimit host work or submission;
their durations are not GPU kernel timings. GPU launches may finish after their
host range closes. No scope introduces a device wait or numerical policy.
Caliper may allocate its own diagnostic metadata; measure instrumentation overhead.
Enable CUPTI GPU profiling only in a separate standalone Caliper run with that
service built; do not compete with Nsight's CUPTI collection in the same run.

Compare instrumented and uninstrumented results and use unprofiled repetitions
for throughput. Keep startup/setup and final output separate from warmed solve
work. A profiler/API duration alone does not identify an expensive GPU kernel.

## Queued execution optimization contract

An event provider may declare an autonomous host-wait priority. The conservative
default keeps polling. After every registered controller-progress operation has
retired, the engine can block on autonomous prefixes in priority order instead
of repeatedly querying device events. Native submitted work precedes CPU lanes
that may consume it. While MPI/other controller progress remains, polling stays
active. Completed CUDA prefixes cache retirement only after publishing errors,
releasing captured owners and finishing access reservations; repeated observation
must preserve the original failure.

Fixed-count Jacobi uses two prebound fields and queues sweeps on the same shared
CPU/CUDA schedule. It never rebinds the public solution allocation or its halo.
Even counts finish in that allocation; odd counts copy once. Solve still observes
finiteness/action status and returns a completed result. No per-sweep residual
observation is added. Real MPI payloads retain their explicit controller bridge.

An empty direct-transfer route is defined by empty send **and** receive row maps
with no reducer. It has no payload access: preserve caller/route/device/active
checks and the normal collective-preflight sequence, then retain the binding and
return an ordered stream completion. This applies to any locally empty rank,
including participants with nonempty peers. Repeated empty publications may
queue because they own no mutable request workspace. The ordinary halo
publication/freshness protocol remains responsible for publishing the result.

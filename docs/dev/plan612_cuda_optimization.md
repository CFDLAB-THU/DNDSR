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

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

CUDA prefix events use blocking host synchronization at explicit host observation
points. This prevents driver-side spin inside `cudaEventSynchronize`; native
stream waits and nonblocking event queries retain their ordinary semantics.
It does not permit the engine to block while a controller-progress operation
still needs polling. Host wake latency and GPU throughput are measured separately.

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

## Kernel-local layout and arithmetic experiments

`execution.kernels` selects operator-local policies without changing the
solver schedule, primary state, restart format, geometry or residual sums:

| Setting | Values / scope |
| --- | --- |
| `stateInputLayout` | `aos` (default), `aosoa32`; RHS, Jacobian and linear input shadows |
| `reconstructionInputLayout` | `aos` (default), `aosoa32`; RHS coefficient/viscous input shadows |
| `polynomialArithmetic` | `fp64` (default), `fp32`, `f32f32`; RHS point-local basis/value/gradient |
| `dissipationArithmetic` | Same types; Roe-family dissipation, FP64 central physical flux |
| `fluxArithmetic` | Same types; full flow5 perfect-gas local thermo/convection/molecular diffusion |
| `rhsMapping` | `thread` (default, ordered quadrature) or `warpQuadrature` (FP64 flow5 experiment) |

These are experimental CUDA choices. Reduced arithmetic is flow5 only; full
reduced flux excludes primitive viscous gradients, RANS and QCR. One launch
uses one common reduced arithmetic type, and full flux already includes its
dissipation. Unsupported combinations fail explicitly. PP decisions and stored
means/coefficients remain FP64; failed reduced-range point arithmetic returns
a numerical status rather than silently switching precision.

The warp mapping assigns a face to one full warp, evaluates its quadrature
points in lanes, and uses a fixed warp reduction for the integrated flux.
Inactive lanes still participate in collectives. It changes summation order,
so compare against the CPU/default mapping with the declared FP64 tolerance,
not byte equality. Physical boundary points retain separate unique output rows.

The host wrapper validates residency/layout/shape once, then selects templated
`__global__` entries and shared host/device numerical helpers. Native FP64
flow5 RHS has static AoS/tiled input indexing; precision experiments use uniform
layout-aware views to bound template instantiations. No metadata owner or lease
is created inside the numerical kernel.

`Algebra::ArrayInputCache` owns a setup-allocated tiled shadow. Refresh is queued
on its consumer lane and invalidated by source storage identity, shape and
submission mutation epoch, or writes to the shadow. Kernel operands retain
ordinary ownership/access declarations. A source producer on another lane must
be ordered before refresh by the caller; this cache is bound to its first lane
and is not a cross-stream readiness oracle. Counts/versions are host metadata;
no device-to-host scan occurs to decide a cache hit. Shadow conversion costs
must be included in end-to-end layout comparisons.

Float-float uses optional header-only CCCL (`DNDS_USE_CCCL_COMPUTE=ON`, pinned
`DNDS_CCCL_SOURCE_ROOT`). It has an FP32-like exponent range and is not IEEE
FP64. A coherent source libcudacxx/Thrust/CUB family must precede toolkit headers
in the enabled build. The external checkout remains separate from the packaged
dependency bundle. Qualification results belong in the plan execution record.

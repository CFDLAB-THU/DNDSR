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

### Queued descriptor access and retirement

Queued operators declare read access to pattern, stencil, quadrature and metric
Arrays just as they declare numerical inputs. Retaining an immediate read lease
inside a queued command is insufficient for efficient scheduling: its destructor
must conservatively fence unless an explicit completion is attached. Use
`ReadAs`/`SubmitArrays` or the extra typed operands of `SubmitArrayBatch` instead.
The operation record retains value and CSR shape allocations, excludes immediate
mutation and publishes completion/errors through the existing stream prefix.
Generation-only `FiniteVolumePlan::AcquireMeshRead` protects mesh identity without
acquiring unrelated metric payloads. Synchronous APIs and host result observations
retain ordinary immediate access semantics.

`MetadataBuffer` is a separate immutable publication: construction and placement
complete initialization and no mutable alias is exported. A queued callback may
borrow `MetadataArrayView` while retaining its metadata owner through completion.
This exception must not be applied to mutable Arrays or used to bypass residency
checks. Long-lived immutable resident publications still use exclusion seals;
these are acquired at setup, not on every numerical call.

Native CUDA waits on the same stream are redundant with FIFO execution and the
existing prefix chain. Omitting them preserves the checked dependency frontier,
terminal errors and captured owners. Repeated identical pending foreign-prefix
waits are also coalesced; different producer prefixes retain their real dependency.
Immediate shared completions cache terminal success/error so copy, lease and
destructor observers do not repeatedly synchronize the same event. Host waits
use blocking events. The conservative no-completion fence remains unchanged.

Required host PP/convergence decisions and the portable CUDA-aware MPI progress
bridge are not removed by this optimization. Fixed-count queued kernels remain
distinct from dynamic stopping, which still has an explicit observation boundary.

The matched A5000 two-step follow-up preserves every final flow5 value bitwise
and all 167 non-timing history rows. In the warmed second step, stream syncs
fall 10,351 -> 1,495; event syncs 5,109 -> 3,217; stream waits 1,757 -> 1,666.
Event creations remain 3,627, kernel count remains 5,828 and tiny D2H copies
remain 1,118 / 49,540 bytes. Stream-sync API wall time falls 1,609.7 -> 3.64 ms,
mostly by moving waits to real prefix observation boundaries rather than by
shortening GPU work. GPU idle is 624.7 -> 615.9 ms, while the complete GPU span
is 56.670 -> 56.827 s. One serial matched pair establishes bookkeeping removal
and exact numerical preservation; it does not demonstrate a throughput speedup.
The reference still has one capped nonlinear stage. Focused validation covers
eight CPU/MPI CTests and ten CUDA executables / 54 cases on the A5000. Full
records and the retained traces are indexed in the plan repository's
`evidence/plan61-2-queued-access-retirement/` directory.

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
| `rhsBlockSize` | 64, 128 (default), 256 threads per block |
| `rhsDoubleBuffer` | False (default); two shared changing-input banks with async prefetch |

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

Double buffering cooperatively stages both parent means, reconstruction and
optional distinct viscous coefficients, and prepared auxiliary scalars. The
next grid-stride batch is prefetched into the other bank with `cp.async` on
Ampere while the current batch is consumed. Two banks are bounded to32KiB;
the thread mapping may have inactive compute lanes when P3 rows are large.
All lanes participate in bank readiness/reuse barriers. Geometry/quadrature,
basis and point-shaped interface packets remain resident common inputs.
The grid uses up to four blocks per SM, from a setup-cached device property.
No hot-path hardware query, allocation, lease or host copy is introduced.

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

### Quantized VR accumulation

`execution.reconstructionKernel` keeps quantization separate from RHS arithmetic:

| Choice | Sweep maps | Local product and update arithmetic | Stored coefficients |
| --- | --- | --- | --- |
| `fp32` | FP32 | FP32 dense products; FP64 outer/mean/BC/relaxation sum | FP64 |
| `fp32Accum` | FP32 | Entire local update in FP32 | FP64 |
| `fp32FloatFloatAccum` | FP32 | Entire local update in float-float | FP64 |

The complete-update choices include periodic rotations, neighbor/mean and
boundary contributions and relaxation. FP64 input values convert directly to
the selected scalar; float-float retains both input limbs, but cannot recover
precision already removed from the maps. Only the final result converts to
FP64. WBAP and PP are unaffected. Unsupported build/device/model choices fail
at setup; float-float requires the optional CCCL headers. New policies validate
selected output rows before guarded commit; nonfinite arithmetic sets retained
device status without exception recovery or a new host wait. This validation
pass is absent from the unchanged default/mixed policies.

## Measured dispatch guidance

The uncached 64³ P3/degree-5 TGV assessment completed both thread/warp shared
staging sweeps on A100 and A5000. Every variant passed its frozen short-pilot
numerical budget. Shared banks did not beat unbuffered thread-per-face on either
device. A 32 KiB double bank fits only nine faces for this five-component,
19-mode shared reconstruction/viscous input; thread-per-face therefore leaves
most loader lanes idle while computing. Warp consumers use more lanes, but
retain the heavy point arithmetic and add bank synchronization. More queued
blocks do not guarantee more resident blocks under register/shared limits.

Keep defaults unchanged. For this measured shape, unbuffered 256-thread RHS
plus FP64 tensor reconstruction is a qualified A100 candidate; tiled input
access benefits A5000. Whole-update quantized FP32 and float-float pass their
declared budgets but do not accelerate this case. The full tables, numerical
limits, compiler/runtime hashes and per-operator timeline belong in the separate
Plan 61-2 evidence record, not a universal kernel-selection rule. These two-step
pilots retain one capped nonlinear stage and do not establish long-time accuracy.

## Per-kernel profiling status and memory interpretation

The earlier whole-update assessment did not complete the requested individual
kernel investigation. The reopened evidence is maintained separately under
`workspace/dndsr-next-plan/evidence/plan61-2-profiling-completion/`. It includes
all 35 baseline hot launch configurations on A100 and A5000, all 25 qualified
A5000 normal variant captures, matched initial RHS/VR counter comparisons and
the full CPU submission/event/copy/idle review. The reopened profiling
investigation is complete under the user's 2026-11-15 instruction to skip
occupied A100. Additional A100 candidate captures remain explicitly deferred;
A5000 measurements do not substitute for those architecture-specific results.
The final report and requirement audit are `INVESTIGATION-REPORT.md` and
`CLOSURE-AUDIT.json` in the evidence directory. They retain raw hashes,
fresh numerical revalidation and the five normal traces' unknown controller
exit status with independently verified complete raw/numerical evidence.

In these counter reports, DRAM is device-local GDDR6 on A5000 and HBM on A100.
It is distinct from CPU RAM and host/device copies. Cumulative DRAM bytes for
one selected invocation include repeated field/geometry accesses, outputs and
cache misses from CUDA thread-private local memory. Local memory can contain
addressable automatic arrays and device-call stack state; register spilling
is only one cause. The sampled A100 spill counters report zero despite large local
load/store counts. Do not equate local-sector bytes with off-chip traffic or
sum L1 local/global sectors with DRAM bytes across the same memory hierarchy.

Nsight Compute combines counters from replay passes into one invocation's
report. The 197.83 GB A5000 RHS observation is neither a 197.83 GB allocation nor
the sum of twelve replayed invocations. Nsight may separately save replay
buffers in host RAM; that profiler overhead is outside the selected kernel.
The warmed normal step 2 trace has no H2D payload. Replay timings with cache
flushing diagnose memory/instruction behavior; ordinary qualified NSYS kernel
distributions and unprofiled repetitions provide throughput measurements.

The accepted coefficient-only tiled RHS control reduces device-DRAM traffic
to 102.62 GB while L1 local requests remain about 255.3 GB. This distinguishes
input-layout effects from persistent local/helper traffic. The aggregate
counters do not identify the exact source array behind every cache miss.
Use the separate native execution-pipe counts as instructions, not FLOPs or
exact opcode counts; precision variants also differ in physical-view
projection, so their timings are complete-implementation comparisons.

References: [CUDA local memory](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#device-memory-accesses),
[Nsight Compute kernel replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#kernel-replay).

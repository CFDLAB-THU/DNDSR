#pragma once
#include "CFV/Detail/ReconstructionDeviceView.hpp"
#include "CFV/Plans/ReconstructionSweepLaunch.hpp"
namespace DNDS::CFV::Detail
{
    bool FP64TensorCoreAvailable(Runtime::Device);
    bool TF32TensorCoreAvailable(Runtime::Device);
    struct ReconstructionSweepResources
    {
        index sharedBytes = 0;
        int registersPerThread = 0, activeBlocksPerSM = 0, gridBlocks = 0, threads = 0;
    };
    enum class ReconstructionSweepArithmetic
    {
        FP64,
        FloatFloat,
        FP64TensorCore,
        FP32,
        TF32,
        FP64MatrixShared,
        FP32Reference,
        FP32FullAccumulation,
        FP32FloatFloatAccumulation,
        TF32Synchronous
    };
    /// Full modal batches, with explicit checked mapping/launch/shared-memory
    /// policy. The caller retains every view through stream completion.
    void LaunchStagedJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                            Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>,
                            Runtime::ArrayView<real>, Runtime::ArrayView<const int>, const ReconstructionSweepLaunch &, ReconstructionSweepArithmetic);
    /// Production pipeline with statically selected LVR; alpha=1 excludes all
    /// neighbor, boundary and inverse reads. The caller validates alpha first.
    void LaunchLimitedStagedJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                   Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>,
                                   Runtime::ArrayView<real>, Runtime::ArrayView<const int>, Runtime::ArrayView<const real>,
                                   Runtime::ArrayView<const real>, const ReconstructionSweepLaunch &);
    /// Untimed launch feasibility/resource inquiry; never enqueues a kernel.
    ReconstructionSweepResources InspectStagedJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                                     Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>,
                                                     Runtime::ArrayView<real>, Runtime::ArrayView<const int>, const ReconstructionSweepLaunch &, ReconstructionSweepArithmetic);
    /// Explicit reduced-mantissa tensor arithmetic on FP32 sweep maps, with
    /// FP32 accumulation and FP64 coefficient updates. Not full FP32 MMA.
    void LaunchJacobiTF32(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                          Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    void LaunchJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                      Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    /// FP32 maps with a complete FP32 or float-float local update, FP64 store.
    /// Reports nonfinite selected results before the guarded publication.
    void LaunchJacobiFullAccumulation(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                      Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<int>, bool floatFloat);
    /// Retained block-team, synchronous matrix-shared FP64 sweep.
    void LaunchJacobiMatrixShared(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                  Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    /// Device-only finite-result guard for narrow-exponent compute policies.
    void LaunchValidateJacobiResult(Runtime::ExecutionContext, DeviceWorkSelection, Runtime::ArrayView<const real>, Runtime::ArrayView<int>);
    /// Limited Jacobi preserves CPU alpha=1 neighbor exclusion and reports
    /// invalid penalties through the retained device status, without a drain.
    void LaunchLimitedJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                             Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>,
                             Runtime::ArrayView<const real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<int>,
                             bool pipeline, const ReconstructionSweepLaunch &);
    void LaunchWBAP(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                    Runtime::PairReadView<real>, Runtime::ArrayView<real>, Runtime::ArrayView<real>, DeviceCharacteristicView,
                    Runtime::ArrayView<const real>, Runtime::ArrayView<int>, index, index, real, index);
    void LaunchCommit(Runtime::ExecutionContext, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
}

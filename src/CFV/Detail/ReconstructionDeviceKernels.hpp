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
    /// Full modal batches, with explicit checked mapping/launch/shared-memory
    /// policy. The caller retains every view through stream completion.
    void LaunchStagedJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                            Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>,
                            Runtime::ArrayView<real>, Runtime::ArrayView<const int>, const ReconstructionSweepLaunch &, bool tensor);
    /// Untimed launch feasibility/resource inquiry; never enqueues a kernel.
    ReconstructionSweepResources InspectStagedJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                                     Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>,
                                                     Runtime::ArrayView<real>, Runtime::ArrayView<const int>, const ReconstructionSweepLaunch &, bool tensor);
    /// Explicit reduced-mantissa tensor arithmetic on FP32 sweep maps, with
    /// FP32 accumulation and FP64 coefficient updates. Not full FP32 MMA.
    void LaunchJacobiTF32(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                          Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    void LaunchJacobiTensorCore(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>, bool wide = false);
    void LaunchJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                      Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    /// FP32 maps with a complete FP32 or float-float local update, FP64 store.
    /// Reports nonfinite selected results before the guarded publication.
    void LaunchJacobiFullAccumulation(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                      Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<int>, bool floatFloat);
    void LaunchJacobiShared(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                            Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    /// Explicit FP64 mapping experiments: 0=wide grid with the reference
    /// arithmetic, 1=static modal sizes, 2=shared neighbor matrix and row.
    void LaunchJacobiOptimized(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                               Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>, int mapping);
    /// Limited Jacobi preserves CPU alpha=1 neighbor exclusion and reports
    /// invalid penalties through the retained device status, without a drain.
    void LaunchLimitedJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                             Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>,
                             Runtime::ArrayView<const real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<int>);
    /// Original B gather followed by a local inverse, with independently laid-out
    /// matrix/vector/inverse payloads. Mapping -1/0/1/2 = reference/wide/static/shared.
    void LaunchRawJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                         Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>,
                         Runtime::ArrayView<real>, Runtime::ArrayView<const int>, Runtime::ArrayView<const real>,
                         Runtime::ArrayView<const real>, bool limited, int mapping);
    void LaunchWBAP(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                    Runtime::PairReadView<real>, Runtime::ArrayView<real>, Runtime::ArrayView<real>, DeviceCharacteristicView,
                    Runtime::ArrayView<const real>, Runtime::ArrayView<int>, index, index, real, index);
    void LaunchCommit(Runtime::ExecutionContext, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
}

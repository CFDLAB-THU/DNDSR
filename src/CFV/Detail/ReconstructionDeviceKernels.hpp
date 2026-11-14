#pragma once
#include "CFV/Detail/ReconstructionDeviceView.hpp"
namespace DNDS::CFV::Detail
{
    bool FP64TensorCoreAvailable(Runtime::Device);
    bool TF32TensorCoreAvailable(Runtime::Device);
    /// Explicit reduced-mantissa tensor arithmetic on FP32 sweep maps, with
    /// FP32 accumulation and FP64 coefficient updates. Not full FP32 MMA.
    void LaunchJacobiTF32(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                          Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    void LaunchJacobiTensorCoreShared(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                      Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>, bool async);
    void LaunchJacobiTensorCore(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    void LaunchJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                      Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    void LaunchJacobiShared(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                            Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    /// Limited Jacobi preserves CPU alpha=1 neighbor exclusion and reports
    /// invalid penalties through the retained device status, without a drain.
    void LaunchLimitedJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                             Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>,
                             Runtime::ArrayView<const real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<int>);
    void LaunchWBAP(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                    Runtime::PairReadView<real>, Runtime::ArrayView<real>, Runtime::ArrayView<real>, DeviceCharacteristicView,
                    Runtime::ArrayView<const real>, Runtime::ArrayView<int>, index, index, real, index);
    void LaunchCommit(Runtime::ExecutionContext, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
}

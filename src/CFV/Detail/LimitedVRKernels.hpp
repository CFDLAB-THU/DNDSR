#pragma once
#include "CFV/Detail/ReconstructionDeviceView.hpp"
#include "CFV/Math/LimitedVR.hpp"

namespace DNDS::CFV::Detail
{
    using LVRIndices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using LVRField = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using LVRInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using LVROutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    struct LimitedVRStencilView
    {
        LVRIndices offsets, neighbors, boundary;
        void ValidateDevice(Runtime::Device device) const { Runtime::ValidateViewDevices(device, offsets, neighbors, boundary); }
    };
    void BuildLimitedVRSeeds(LimitedVRStencilView, Runtime::RowSelection, LVRField scalars, LVRField gates,
                             LVRInput previous, LVROutput output, LimitedVRKernelParameters);
    void LimitedVRMaximum(LimitedVRStencilView, Runtime::RowSelection, LVRField input, LVROutput output,
                          std::array<bool, 3> active, bool excludeBoundary, bool saturatedOnly, std::array<real, 3> caps);
    void LaunchLimitedVRSeeds(Runtime::ExecutionContext, LimitedVRStencilView, Runtime::RowSelection, LVRField scalars,
                              LVRField gates, LVRInput previous, LVROutput output, LimitedVRKernelParameters);
    void LaunchLimitedVRMaximum(Runtime::ExecutionContext, LimitedVRStencilView, Runtime::RowSelection, LVRField input, LVROutput output,
                                std::array<bool, 3> active, bool excludeBoundary, bool saturatedOnly, std::array<real, 3> caps);
}

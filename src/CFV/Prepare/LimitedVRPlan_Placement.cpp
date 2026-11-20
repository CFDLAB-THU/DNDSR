#include "CFV/Plans/LimitedVRPlan.hpp"

namespace DNDS::CFV
{
    LimitedVRPlan LimitedVRPlan::Placed(FiniteVolumePlan metrics, Runtime::ExecutionContext context) const
    {
        _metrics.Validate();
        metrics.Validate();
        DNDS_check_throw_info(metrics.Space().Same(_metrics.Space()) && metrics.Device() == context.device, "LVR placement origin/device differs");
        if (metrics.Device() == Device())
            return *this;
        DNDS_check_throw_info(Device() == Runtime::Device::Host() && metrics.Device().space == Runtime::MemorySpace::CUDA,
                              "LVR placement requires host preparation and a CUDA target");
        auto result = *this;
        result._metrics = std::move(metrics);
        const auto device = result.Device();
        result._offsets = _offsets.To(device, context);
        result._neighbors = _neighbors.To(device, context);
        result._boundary = _boundary.To(device, context);
        result._regionRows.Place(context, "CFV/limitedVR");
        result._empty = Runtime::Array<real>(Runtime::RowLayout::Dense(0, 0), device);
        return result;
    }
}

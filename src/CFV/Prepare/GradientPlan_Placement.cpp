#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "DNDS/Core/OMP.hpp"
#include <limits>

namespace DNDS::CFV
{
    namespace R = Runtime;
    GradientPlan GradientPlan::Placed(R::Device device, R::ExecutionContext context) const
    {
        _data->metrics.Validate();
        DNDS_check_throw_info(context.device == device, "gradient placement and context differ");
        if (device == Device())
            return *this;
        DNDS_check_throw_info(Device() == R::Device::Host(), "gradient replacement placement requires its CPU setup plan");
        return Placed(_data->metrics.Placed(device, context), context);
    }
    GradientPlan GradientPlan::Placed(FiniteVolumePlan metrics, R::ExecutionContext context) const
    {
        _data->metrics.Validate();
        metrics.Validate();
        DNDS_check_throw_info(metrics.Space().Same(_data->metrics.Space()) && metrics.Device() == context.device,
                              "gradient placement metrics have a different origin/device");
        if (metrics.Device() == Device())
            return *this;
        DNDS_check_throw_info(Device() == R::Device::Host() && metrics.Device().space == R::MemorySpace::CUDA,
                              "gradient placement supports CPU preparation to CUDA");
        auto placed = std::make_shared<Data>(*_data);
        placed->metrics = std::move(metrics);
        const auto device = placed->metrics.Device();
        placed->offsets = _data->offsets.To(device, context);
        placed->boundaryIds = _data->boundaryIds.To(device, context);
        placed->regionRows.Place(context, "CFV/gradient");
        placed->weights = _data->weights.To(device, context);
        placed->offsets.SetName("CFV/gradient/offsets", DNDS_ALLOCATION_SITE());
        placed->boundaryIds.SetName("CFV/gradient/boundaryIds", DNDS_ALLOCATION_SITE());
        placed->weights.SetName("CFV/gradient/weights", DNDS_ALLOCATION_SITE());
        // Boundary coordinates are setup inputs, not device gradient operands.
        // The boundary-row count survives without retaining their CPU payload.
        std::vector<BoundaryQuadraturePoint>().swap(placed->boundary);
        return GradientPlan(std::move(placed));
    }
    GradientWorkspace GradientPlan::BuildWorkspace(index components, const std::vector<index> &periodicVectorStarts, bool withLimiter) const
    {
        _data->metrics.Validate();
        DNDS_check_throw_info(components > 0 && components <= std::numeric_limits<index>::max() / (6 * _data->metrics.Dimension()),
                              "gradient component count is invalid");
        if (Device().space == R::MemorySpace::CUDA)
            DNDS_check_throw_info(components >= 4 && components <= 7, "CUDA gradient requires a combined table with 4 to 7 components");
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        R::Array<index> starts(R::RowLayout::Dense(index(periodicVectorStarts.size()), 1));
        {
            auto write = starts.Write();
            const auto view = write.View<R::FixedRows<1>>();
            for (index i = 0; i < index(periodicVectorStarts.size()); ++i)
                view(i, 0) = periodicVectorStarts[i];
        }
        if (Device() != R::Device::Host())
            starts = starts.To(Device(), {Device(), nullptr});
        GradientWorkspace work{_data->identity, std::move(starts), R::Array<real>(R::RowLayout::Dense(0, 0), Device()), components};
        work.vectorIdentity = work.periodicVectorStarts.Identity();
        work.vectorVersion = work.periodicVectorStarts.Version();
        work.vectorShapeVersion = work.periodicVectorStarts.ShapeVersion();
        work.withLimiter = withLimiter;
        if (withLimiter)
        {
            work.limiterScratch = R::Array<real>(Device() == R::Device::Host()
                                                     ? R::RowLayout::Dense(OMP::DistributedThreads(), 6 * components * _data->metrics.Dimension())
                                                     : R::RowLayout::Dense(0, 0),
                                                 Device());
            work.deviceStatus = R::Array<int>(R::RowLayout::Dense(Device() == R::Device::Host() ? 0 : 1, 1), Device());
        }
        return work;
    }
}

#include "CFV/Plans/LimitedVRPlan.hpp"
#include "CFV/Detail/LimitedVRKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "Geom/Access/MeshAccess.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    void LimitedVRPlan::BuildExtremumSeeds(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                           const FVField &meanScalars, const FVField &baseGates, const R::Array<real> &previous,
                                           FVField &output, const LimitedVRSettings &settings, ReconstructionRegion region) const
    {
        ValidateSeeds(meanScalars, baseGates, previous, output, region);
        const auto parameters = LimitedVRKernelParameters::FromSettings(settings);
        DNDS_check_throw_info(stream.Context().device == Device(), "LVR stream placement differs");
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(Device() == R::Device::Host(), "CUDA LVR kernels are not built");
#endif
        auto geometry = std::make_shared<Geom::MeshReadAccess>(_metrics.AcquireMeshRead());
        const auto &selected = region == ReconstructionRegion::HaloDependent ? _dependent : _core;
        const index count = region == ReconstructionRegion::All ? _metrics.NumCell() : selected.RowCount();
        R::SubmitArrays(engine, stream, [geometry, region, count, parameters](R::ExecutionContext context, auto offsets, auto neighbors, auto boundary, auto rows, auto scalar, auto scalarGhost, auto gate, auto gateGhost, auto prior, auto result)
                        {
                            const Detail::LimitedVRStencilView stencil{offsets, neighbors, boundary};
                            const Detail::DeviceWorkSelection selection{rows.DynamicView(), count, region != ReconstructionRegion::All};
                            const Detail::LVRField scalars{scalar, scalarGhost}, gates{gate, gateGhost};
                            if (context.device == R::Device::Host())
                                Detail::BuildLimitedVRSeeds(stencil, selection, scalars, gates, prior, result, parameters);
#ifdef DNDS_USE_CUDA
                            else
                                Detail::LaunchLimitedVRSeeds(context, stencil, selection, scalars, gates, prior, result, parameters);
#endif
                        },
                        R::ReadAs<R::FixedRows<1>>(_offsets), R::ReadAs<R::FixedRows<1>>(_neighbors), R::ReadAs<R::FixedRows<1>>(_boundary), R::ReadAs<R::FixedRows<1>>(selected), R::ReadAs<R::UniformTableRows>(meanScalars.Owned()), R::ReadAs<R::UniformTableRows>(region == ReconstructionRegion::Core ? _empty : meanScalars.Ghost()), R::ReadAs<R::UniformTableRows>(baseGates.Owned()), R::ReadAs<R::UniformTableRows>(region == ReconstructionRegion::Core ? _empty : baseGates.Ghost()), R::ReadAs<R::UniformTableRows>(previous), R::WriteAs<R::UniformTableRows>(output.Owned()));
    }
    void LimitedVRPlan::NeighborMaximum(R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &input, FVField &output,
                                        std::array<bool, 3> active, bool excludeBoundary, bool saturatedOnly,
                                        std::array<real, 3> caps, ReconstructionRegion region) const
    {
        ValidateMaximum(input, output, caps, region);
        DNDS_check_throw_info(stream.Context().device == Device(), "LVR stream placement differs");
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(Device() == R::Device::Host(), "CUDA LVR kernels are not built");
#endif
        auto geometry = std::make_shared<Geom::MeshReadAccess>(_metrics.AcquireMeshRead());
        const auto &selected = region == ReconstructionRegion::HaloDependent ? _dependent : _core;
        const index count = region == ReconstructionRegion::All ? _metrics.NumCell() : selected.RowCount();
        R::SubmitArrays(engine, stream, [geometry, region, count, active, excludeBoundary, saturatedOnly, caps](R::ExecutionContext context, auto offsets, auto neighbors, auto boundary, auto rows, auto owned, auto ghost, auto result)
                        {
                            const Detail::LimitedVRStencilView stencil{offsets, neighbors, boundary};
                            const Detail::DeviceWorkSelection selection{rows.DynamicView(), count, region != ReconstructionRegion::All};
                            const Detail::LVRField source{owned, ghost};
                            if (context.device == R::Device::Host())
                                Detail::LimitedVRMaximum(stencil, selection, source, result, active, excludeBoundary, saturatedOnly, caps);
#ifdef DNDS_USE_CUDA
                            else
                                Detail::LaunchLimitedVRMaximum(context, stencil, selection, source, result, active, excludeBoundary, saturatedOnly, caps);
#endif
                        },
                        R::ReadAs<R::FixedRows<1>>(_offsets), R::ReadAs<R::FixedRows<1>>(_neighbors), R::ReadAs<R::FixedRows<1>>(_boundary), R::ReadAs<R::FixedRows<1>>(selected), R::ReadAs<R::UniformTableRows>(input.Owned()), R::ReadAs<R::UniformTableRows>(region == ReconstructionRegion::Core ? _empty : input.Ghost()), R::WriteAs<R::UniformTableRows>(output.Owned()));
    }
}

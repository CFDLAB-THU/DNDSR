#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Detail/GradientKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "Geom/Access/MeshAccess.hpp"
#include <limits>

namespace DNDS::CFV
{
    namespace R = Runtime;
    void GradientPlan::ValidateEvaluation(const FVField &mean, const R::Array<real> &boundaryValues,
                                          const FVField &gradient, ReconstructionRegion region) const
    {
        _data->metrics.Validate();
        _data->partition.Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                              "unknown gradient work region");
        const index components = mean.Width(), dim = _data->metrics.Dimension();
        DNDS_check_throw_info(components > 0 && components <= std::numeric_limits<index>::max() / dim,
                              "gradient component count is invalid");
        _data->metrics.ValidateField(mean, components);
        _data->metrics.ValidateField(gradient, dim * components);
        DNDS_check_throw_info(boundaryValues.GetDevice() == Device() && boundaryValues.Layout().Kind() == R::RowLayoutKind::Table &&
                                  boundaryValues.RowCount() == NumBoundaryQuadrature() && boundaryValues.Layout().Width() == components,
                              "gradient boundary value shape or placement differs");
        for (const auto *input : {&mean.Owned(), &mean.Ghost(), &boundaryValues})
            DNDS_check_throw_info(gradient.Owned().Identity() != input->Identity() &&
                                      !Algebra::Detail::Overlaps(gradient.Owned().Storage(), input->Storage()),
                                  "gradient output overlaps its inputs");
        if (Device().space == R::MemorySpace::CUDA)
            for (const auto *array : {&mean.Owned(), &mean.Ghost(), &gradient.Owned()})
                DNDS_check_throw_info(array->Layout().Kind() == R::RowLayoutKind::Table, "CUDA gradient requires combined AoS fields");
    }
    void GradientPlan::ValidateWorkspace(const GradientWorkspace &workspace, index components) const
    {
        DNDS_check_throw_info(workspace.plan == _data->identity &&
                                  workspace.components == components && workspace.empty.GetDevice() == Device() &&
                                  workspace.empty.RowCount() == 0 && workspace.empty.Layout().Kind() == R::RowLayoutKind::Table &&
                                  workspace.periodicVectorStarts.GetDevice() == Device() &&
                                  workspace.periodicVectorStarts.Identity() == workspace.vectorIdentity && workspace.periodicVectorStarts.Version() == workspace.vectorVersion &&
                                  workspace.periodicVectorStarts.ShapeVersion() == workspace.vectorShapeVersion &&
                                  workspace.periodicVectorStarts.Layout().Kind() == R::RowLayoutKind::Table && workspace.periodicVectorStarts.Layout().Width() == 1 &&
                                  workspace.periodicVectorStarts.Layout().Stride() == 1,
                              "gradient workspace metadata/placement changed");
    }
    void GradientPlan::Evaluate(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                const FVField &mean, const R::Array<real> &boundaryValues, FVField &gradient,
                                const GradientWorkspace &workspace, ReconstructionRegion region) const
    {
        ValidateEvaluation(mean, boundaryValues, gradient, region);
        ValidateWorkspace(workspace, mean.Width());
        DNDS_check_throw_info(stream.Context().device == Device(), "gradient stream placement differs");
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(Device() == R::Device::Host(), "CUDA gradients are not built");
#endif
        // The kernel consumes no metric arrays. Pin only its mesh generation;
        // the captured guard retires with this operation, not an engine drain.
        auto geometry = std::make_shared<Geom::MeshReadAccess>(_data->metrics.AcquireMeshRead());
        const auto &selected = region == ReconstructionRegion::HaloDependent ? _data->dependent : _data->core;
        const index count = region == ReconstructionRegion::All ? _data->metrics.NumCell() : selected.RowCount();
        const int dimension = _data->metrics.Dimension();
        R::SubmitArrays(engine, stream, [geometry, region, count, dimension, rotated = _data->hasRotatedFrames](R::ExecutionContext context, auto offsets, auto neighbors, auto boundaryIds, auto weights, auto rotations, auto rotationActive, auto rows, auto owned, auto ghost, auto boundary, auto starts, auto output)
                        {
                            const Detail::GradientKernelView plan{offsets, neighbors, boundaryIds, weights, rotations, rotationActive, dimension, rotated};
                            const Detail::DeviceWorkSelection selection{rows.DynamicView(), count, region != ReconstructionRegion::All};
                            const Detail::GradientField values{owned, ghost};
                            if (context.device == R::Device::Host())
                                Detail::EvaluateGradient(plan, selection, {{starts.data, starts.RowCount()}}, values, boundary, output);
#ifdef DNDS_USE_CUDA
                            else
                                Detail::LaunchGradient(context, plan, selection, {starts.DynamicView()}, values, boundary, output);
#endif
                        },
                        R::ReadAs<R::FixedRows<1>>(_data->offsets), R::ReadAs<R::FixedRows<1>>(_data->neighbors), R::ReadAs<R::FixedRows<1>>(_data->boundaryIds), R::ReadAs<R::FixedRows<3>>(_data->weights), R::ReadAs<R::FixedRows<9>>(_data->rotations), R::ReadAs<R::FixedRows<1>>(_data->rotationActive), R::ReadAs<R::FixedRows<1>>(selected), R::ReadAs<R::UniformTableRows>(mean.Owned()), R::ReadAs<R::UniformTableRows>(region == ReconstructionRegion::Core ? workspace.empty : mean.Ghost()), R::ReadAs<R::UniformTableRows>(boundaryValues), R::ReadAs<R::FixedRows<1>>(workspace.periodicVectorStarts), R::WriteAs<R::UniformTableRows>(gradient.Owned()));
    }
}

#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeGradientKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeGradientKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::SubmitGradients(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                  const EquationDofs &state, CFV::ReconstructionRegion region)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() && state.Device() == owner->metrics.Device(),
                              "gradient submission backend or stream differs");
        state.RequireSchema(owner->equation.Schema());
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                  region == CFV::ReconstructionRegion::HaloDependent,
                              "invalid gradient submission region");
        DNDS_check_throw_info(state.Space().Same(owner->metrics.Space()) && state.PhysicalFieldNumber() == index(fields.size()),
                              "gradient submission row space/grouping differs");
        for (index iField = 0; iField < index(fields.size()); ++iField)
            if (auto &field = fields[iField]; field.gradient)
            {
                DNDS_check_throw_info(field.gradientBoundary && field.gradientExecution, "gradient execution workspace is missing");
                owner->gradient.Evaluate(engine, stream, state.PhysicalField(iField), *field.gradientBoundary,
                                         *field.gradient, *field.gradientExecution, region);
            }
        if (!primitiveGradient)
            return;
        DNDS_check_throw_info(primitive && primitiveBoundary && primitiveGradientExecution && state.PhysicalFieldNumber() == 1,
                              "primitive-gradient execution workspace is incomplete");
        const auto fill = [&](bool ghost)
        {
            const auto &source = ghost ? state.CoupledField().Ghost() : state.CoupledField().Owned();
            auto &target = ghost ? primitive->Ghost() : primitive->Owned();
            const auto gas = owner->equation.Gas();
            if (stream.Context().device == R::Device::Host())
                R::SubmitArrays(engine, stream, [gas](R::ExecutionContext, auto input, auto output)
                                { CPU::FillPrimitiveMeans(input, output, gas); }, R::ReadAs<R::UniformTableRows>(source), R::WriteAs<R::UniformTableRows>(target));
            else
            {
#ifdef DNDS_USE_CUDA
                DNDS_check_throw_info(preparationStatus && preparationStatus->pending, "primitive conversion requires a preparation status segment");
                auto &status = ghost ? preparationStatus->meanGhost : preparationStatus->meanOwned;
                R::SubmitArrays(engine, stream, [gas](R::ExecutionContext context, auto input, auto output, auto status)
                                { CUDA::FillPrimitiveMeans(context, input, output, gas, status); }, R::ReadAs<R::TableRows>(source), R::WriteAs<R::TableRows>(target), R::WriteAs<R::FixedRows<1>>(status));
                AppendPreparationStatus(engine, stream, status);
#else
                DNDS_check_throw_info(false, "CUDA primitive gradients are not built");
#endif
            }
        };
        if (region != CFV::ReconstructionRegion::HaloDependent)
            fill(false);
        if (region != CFV::ReconstructionRegion::Core)
            fill(true);
        owner->gradient.Evaluate(engine, stream, *primitive, *primitiveBoundary, *primitiveGradient, *primitiveGradientExecution, region);
    }

    void FiniteVolume::Workspace::SubmitDeviceGradientLimiter(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                              const EquationDofs &state, bool barth, CFV::ReconstructionRegion region)
    {
#ifdef DNDS_USE_CUDA
        state.RequireSchema(owner->equation.Schema());
        DNDS_check_throw_info(nativeDirectGradient && fields.size() == 1 && directGradients.size() == 1 && fields.front().gradient &&
                                  state.Space().Same(owner->metrics.Space()) && owner->metrics.Device().space == R::MemorySpace::CUDA &&
                                  engine.Owns(stream) && stream.Context().device == owner->metrics.Device() && state.Device() == owner->metrics.Device(),
                              "device direct-gradient limiter workspace or backend differs");
        const auto &source = fields.front().gradient->Owned();
        auto &target = directGradients.front().Owned();
        const auto &means = state.CoupledField();
        DNDS_check_throw_info(!source.Storage().Overlaps(target.Storage()) &&
                                  !means.Owned().Storage().Overlaps(target.Storage()) && !means.Ghost().Storage().Overlaps(target.Storage()),
                              "device gradient limiter requires disjoint output");
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            FiniteVolumeStencilStorage::ReadAccess stencil;
            explicit Inputs(std::shared_ptr<const Data> data) : owner(std::move(data)),
                                                                stencil(owner->stencil.Read({owner->metrics.Device(), nullptr})) {}
        };
        const auto pin = std::make_shared<Inputs>(owner);
        const bool ghosts = region != CFV::ReconstructionRegion::Core;
        R::SubmitArrays(engine, stream, [pin, barth, ghosts](R::ExecutionContext context, auto mean, auto remote, auto input, auto output, auto cells)
                        {
                CUDA::GradientLimiterGeometry geometry{pin->stencil.KernelDescriptor(), pin->owner->metrics.NumCell(), pin->owner->metrics.NumCellProc()};
                CUDA::LimitPhysicalGradients(context, geometry, {mean, remote}, input, output, cells, pin->owner->metrics.Dimension(), barth, ghosts); }, R::ReadAs<R::TableRows>(means.Owned()), R::ReadAs<R::TableRows>(ghosts ? means.Ghost() : owner->deviceEmpty), R::ReadAs<R::TableRows>(source), R::WriteAs<R::TableRows>(target), R::ReadAs<R::FixedRows<1>>(DeviceCellSelection(region)));
#else
        DNDS_check_throw_info(false, "CUDA gradient limiting is not built");
#endif
    }

    void FiniteVolume::Workspace::SubmitDeviceGradientPositivity(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state)
    {
#ifdef DNDS_USE_CUDA
        state.RequireSchema(owner->equation.Schema());
        DNDS_check_throw_info(nativeDirectGradient && owner->settings.numerics.positivity && fields.size() == 1 && directGradients.size() == 1 &&
                                  state.Space().Same(owner->metrics.Space()) && owner->metrics.Device().space == R::MemorySpace::CUDA &&
                                  preparationStatus && preparationStatus->pending && gradientStatistics && engine.Owns(stream) &&
                                  stream.Context().device == owner->metrics.Device() && state.Device() == owner->metrics.Device(),
                              "device gradient positivity workspace or backend differs");
        const auto &means = state.CoupledField().Owned();
        auto &gradients = directGradients.front().Owned();
        DNDS_check_throw_info(!means.Storage().Overlaps(gradients.Storage()), "gradient positivity cannot mutate means");
        CUDA::GradientPositivityParameters parameters;
        parameters.bounds = thermal.View();
        parameters.dimension = owner->metrics.Dimension();
        parameters.rans = owner->equation.HasRANS() ? ResolveRANSReconstructionPolicy(owner->equation) : RANSReconstructionView{};
        parameters.betaPower = owner->settings.numerics.positivityBetaPower;
        parameters.betaCutoff = owner->settings.numerics.positivityBetaCutoff;
        parameters.clippedPolicy = owner->settings.numerics.positivityClippedReferencePolicy == "backtracking"
                                       ? ReconstructionClippedReferencePolicy::Backtracking
                                       : ReconstructionClippedReferencePolicy::FallbackToBase;
        // Only immutable sample geometry is pinned here. Mutable means,
        // gradients, beta and mandatory status are declared engine operands.
        const auto stencil = owner->stencil.Read({owner->metrics.Device(), nullptr});
        R::SubmitArrays(engine, stream, [stencil, parameters, count = owner->metrics.NumCell(), all = owner->metrics.NumCellProc()](R::ExecutionContext context, auto mean, auto gradients, auto cells, auto status, auto counts, auto minima)
                        {
                CUDA::GradientLimiterGeometry geometry{stencil.KernelDescriptor(), count, all};
                CUDA::LimitPositivePhysicalGradients(context, geometry, mean, gradients, cells, parameters, {status, counts, minima}); }, R::ReadAs<R::TableRows>(means), R::ReadWriteAs<R::TableRows>(gradients), R::ReadAs<R::FixedRows<1>>(DeviceCellSelection(CFV::ReconstructionRegion::All)), R::WriteAs<R::FixedRows<1>>(gradientStatistics->partialStatus), R::WriteAs<R::FixedRows<1>>(gradientStatistics->partialCount), R::WriteAs<R::FixedRows<1>>(gradientStatistics->partialMinimum));
#else
        DNDS_check_throw_info(false, "CUDA gradient positivity is not built");
#endif
    }
}

#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::SubmitDeviceVolumeSource(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const EquationDofs &state, EquationDofs &result, real time, CFV::ReconstructionRegion region,
        OptionalRef<EquationDofs> lowOrder, OptionalRef<R::Array<real>> sourceDerivative)
    {
#ifdef DNDS_USE_CUDA
        state.RequireSchema(owner->equation.Schema());
        DNDS_check_throw_info(owner->metrics.Device().space == R::MemorySpace::CUDA &&
                                  engine.Owns(stream) && stream.Context().device == owner->metrics.Device() &&
                                  state.Device() == owner->metrics.Device() && state.Space().Same(owner->metrics.Space()) &&
                                  state.PhysicalFieldNumber() == 1 && std::isfinite(time),
                              "device source requires combined fields and matching stream/placement");
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            FiniteVolumeStencilStorage::ReadAccess stencil;
            Geom::MeshReadAccess mesh;
            explicit Inputs(std::shared_ptr<const Data> data)
                : owner(std::move(data)), stencil(owner->stencil.Read({owner->metrics.Device(), nullptr})), mesh(owner->metrics.AcquireMeshRead(bool(owner->wallDistance)))
            {
                if (owner->equation.HasRANS())
                {
                    DNDS_check_throw_info(owner->wallDistance, "RANS source requires prepared wall distance");
                }
            }
        };
        const auto pin = std::make_shared<Inputs>(owner);
        const auto rows = pin->stencil.View().RHSRows(region, false, owner->metrics.NumCell());
        const auto frame = owner->motion.At(time).KernelParameters();
        if (frame.moving)
        {
            auto &output = result.CoupledField().Owned();
            const int firstMomentum = owner->equation.IsACM() ? 0 : 1;
            if (lowOrder)
                R::SubmitArrays(engine, stream, [pin, frame, firstMomentum, rows](R::ExecutionContext context, auto means, auto output, auto low, auto metric)
                                { CUDA::AddRotationSource(context, metric, frame, means, output, low, true, rows, firstMomentum); }, R::ReadAs<R::TableRows>(state.CoupledField().Owned()), R::ReadWriteAs<R::TableRows>(output), R::ReadWriteAs<R::TableRows>(lowOrder->CoupledField().Owned()), R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics()));
            else
                R::SubmitArrays(engine, stream, [pin, frame, firstMomentum, rows](R::ExecutionContext context, auto means, auto output, auto metric)
                                { CUDA::AddRotationSource(context, metric, frame, means, output, output, false, rows, firstMomentum); }, R::ReadAs<R::TableRows>(state.CoupledField().Owned()), R::ReadWriteAs<R::TableRows>(output), R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics()));
        }
        if (!sourceOffsets.RowCount())
            return;
        DNDS_check_throw_info(owner->equation.HasRANS() && state.PhysicalFieldNumber() == 1,
                              "device volume source requires one combined RANS field");
        if (sourceSampling)
            sourceSampling->Validate();
        DNDS_check_throw_info(preparationStatus && preparationStatus->pending, "device source requires an active preparation status segment");
        const int dimension = owner->metrics.Dimension(), components = int(state.ComponentNumber());
        const auto &empty = owner->deviceEmpty;
        const auto &grad = sourceSampling ? empty : directGradients.front().Owned();
        VolumeSourceBasis basis;
        if (sourceSampling)
        {
            basis.polynomial = sourceSampling->BasisParameters();
            basis.cachedDerivatives = sourceSampling->CachedDerivativeNumber();
        }
        auto evaluate = [pin, dimension, components, basis, rows](R::ExecutionContext context, auto means, auto coefficients, auto grad, auto points, auto cached, auto moments, auto geometry, auto offsets, auto weights, auto lengths, auto output, auto status, auto wallPoints, auto... derivative)
        {
            CUDA::SourceRequest request;
            ((request.preparedDerivative = derivative), ...);
            if (pin->owner->equation.IsACM())
                request.acm = R::MetadataArrayView(pin->owner->deviceACMPhysics, context.device);
            else
                request.parameters = R::MetadataArrayView(pin->owner->devicePhysics, context.device);
            request.means = means; request.cellInputs=basis.polynomial.size==0;
            request.coefficients=coefficients;
            request.basis=basis;
            request.basis.points=points; request.basis.cached=cached; request.basis.moments=moments; request.basis.geometry=geometry;
            for(int d=0; d<dimension; ++d) request.gradients[d]={grad,dimension,d};
            request.offsets = offsets; request.weights = weights; request.lengths = lengths;
            request.wallPoints = {wallPoints.data, wallPoints.offsets, wallPoints.rows, wallPoints.width, wallPoints.stride, wallPoints.sizes, wallPoints.kind, wallPoints.device};
            request.output = output; request.status = status;
            request.components = components; request.dimension = dimension;
            CUDA::AddRANSSources(context, request, rows); };
        const auto submit = [&](auto... record)
        {
            R::SubmitArrays(engine, stream, evaluate, R::ReadAs<R::TableRows>(state.CoupledField().Owned()), R::ReadAs<R::TableRows>(sourceSampling ? FluxCoefficients().CoupledField().Owned() : empty), R::ReadAs<R::TableRows>(grad),
                            R::ReadAs<R::UniformTableRows>(sourceSampling ? sourceSampling->PointStorage() : empty),
                            R::ReadAs<R::UniformTableRows>(sourceSampling ? sourceSampling->BasisStorage() : empty),
                            R::ReadAs<R::UniformTableRows>(sourceSampling ? sourceSampling->MomentStorage() : empty),
                            R::ReadAs<R::UniformTableRows>(sourceSampling ? sourceSampling->CellGeometryStorage() : empty), R::ReadAs<R::FixedRows<1>>(sourceOffsets), R::ReadAs<R::FixedRows<1>>(sourceWeights), R::ReadAs<R::FixedRows<1>>(owner->ransCellLength), R::ReadWriteAs<R::UniformTableRows>(result.CoupledField().Owned()), R::WriteAs<R::FixedRows<1>>(preparationStatus->meanOwned), R::ReadAs<R::CSRRows>(owner->wallDistance->CellValues()), record...);
        };
        if (sourceDerivative)
            submit(R::WriteAs<R::UniformTableRows>(*sourceDerivative));
        else
            submit();
        // The core and dependent submissions fill disjoint owned slots. The
        // intermediate direct-gradient publication writes only ghost status.
        if (region != CFV::ReconstructionRegion::Core)
            AppendPreparationStatus(engine, stream, preparationStatus->meanOwned);
#else
        DNDS_check_throw_info(false, "CUDA volume sources are not built");
#endif
    }
}

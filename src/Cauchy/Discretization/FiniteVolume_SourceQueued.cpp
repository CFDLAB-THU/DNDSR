#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::SubmitSourceSamples(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state)
    {
        state.RequireSchema(owner->equation.Schema());
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() &&
                                  state.Device() == owner->metrics.Device() && state.Space().Same(owner->metrics.Space()),
                              "source sampling stream, placement or row space differs");
        if (!sourceOffsets.RowCount())
            return;
        const auto timing = BeginQueuedPhase(engine, stream, SamplingPhase);
        DNDS_check_throw_info(index(sourceValues.size()) == state.PhysicalFieldNumber(), "source sample grouping differs");
        const bool polynomial = owner->settings.numerics.reconstruction == "variational";
        const bool gradients = owner->equation.HasRANS();
        const int dim = owner->metrics.Dimension();
        if (polynomial)
            DNDS_check_throw_info(sourceSampling, "polynomial source sampling plan is missing");
        // All point sampling goes through CFV. Only the direct mean/gradient
        // replication leaf is Cauchy-owned; neither branch acquires immediate
        // payload leases or performs communication inside queued execution.
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            const auto replicate = [&](const R::Array<real> &input, R::Array<real> &output, int stride, int direction)
            {
                R::SubmitArrays(engine, stream, [stride, direction](R::ExecutionContext context, auto offsets, auto input, auto output)
                                {
                                    if (context.device == R::Device::Host())
                                        CPU::ReplicateSourceValues(offsets, input, output, stride, direction);
#ifdef DNDS_USE_CUDA
                                    else
                                        CUDA::ReplicateSourceValues(context, offsets,
                                                                    {input.data, input.offsets, input.rows, input.width, input.stride, input.sizes, input.kind, input.device},
                                                                    {output.data, output.offsets, output.rows, output.width, output.stride, output.sizes, output.kind, output.device},
                                                                    stride, direction);
#else
                                    else
                                        DNDS_check_throw_info(false, "CUDA source sampling is not built");
#endif
                                },
                                R::ReadAs<R::FixedRows<1>>(sourceOffsets), R::ReadAs<R::UniformTableRows>(input), R::WriteAs<R::UniformTableRows>(output));
            };
            if (polynomial)
                sourceSampling->Evaluate(engine, stream, state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), sourceValues[iField]);
            else
                replicate(state.PhysicalField(iField).Owned(), sourceValues[iField], 1, 0);
            if (gradients && !sourceValuesAreMeans)
                for (int d = 0; d < dim; ++d)
                    if (polynomial)
                        sourceSampling->Evaluate(engine, stream, state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), sourceGradients[iField][d], 1 + d);
                    else
                        replicate(directGradients[iField].Owned(), sourceGradients[iField][d], dim, d);
        }
        EndQueuedPhase(engine, stream, timing);
    }

    void FiniteVolume::Workspace::SubmitDeviceVolumeSource(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const EquationDofs &state, OptionalRef<EquationDofs> result, real time,
        OptionalRef<EquationDofs> lowOrder)
    {
#ifdef DNDS_USE_CUDA
        state.RequireSchema(owner->equation.Schema());
        DNDS_check_throw_info(owner->metrics.Device().space == R::MemorySpace::CUDA &&
                                  engine.Owns(stream) && stream.Context().device == owner->metrics.Device() &&
                                  state.Device() == owner->metrics.Device() && state.Space().Same(owner->metrics.Space()) &&
                                  state.PhysicalFieldNumber() == 1 && std::isfinite(time),
                              "device source requires combined fields and matching stream/placement");
        if (result)
        {
            state.RequireCompatible(*result);
            EquationDofOperations::RequireDistinct(state, *result);
            if (activeFluxCoefficients)
                EquationDofOperations::RequireDistinct(FluxCoefficients(), *result);
            if (lowOrder)
            {
                state.RequireCompatible(*lowOrder);
                EquationDofOperations::RequireDistinct(state, *lowOrder);
                EquationDofOperations::RequireDistinct(*result, *lowOrder);
                if (activeFluxCoefficients)
                    EquationDofOperations::RequireDistinct(FluxCoefficients(), *lowOrder);
            }
        }
        DNDS_check_throw_info(!lowOrder || result, "low-order source output requires the primary residual");
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            Geom::MeshReadAccess mesh;
            explicit Inputs(std::shared_ptr<const Data> data)
                : owner(std::move(data)), mesh(owner->metrics.AcquireMeshRead(bool(owner->wallDistance)))
            {
                if (owner->equation.HasRANS())
                {
                    DNDS_check_throw_info(owner->wallDistance, "RANS source requires prepared wall distance");
                }
            }
        };
        const auto pin = std::make_shared<Inputs>(owner);
        const auto frame = owner->motion.At(time).KernelParameters();
        if (result && frame.moving)
        {
            auto &output = result->CoupledField().Owned();
            const auto &selection = DeviceCellSelection(CFV::ReconstructionRegion::All);
            if (lowOrder)
                R::SubmitArrays(engine, stream, [pin, frame](R::ExecutionContext context, auto means, auto output, auto low, auto cells, auto metric)
                                { CUDA::AddRotationSource(context, metric, frame, means, output, low, true, cells); }, R::ReadAs<R::TableRows>(state.CoupledField().Owned()), R::ReadWriteAs<R::TableRows>(output), R::ReadWriteAs<R::TableRows>(lowOrder->CoupledField().Owned()), R::ReadAs<R::FixedRows<1>>(selection), R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics()));
            else
                R::SubmitArrays(engine, stream, [pin, frame](R::ExecutionContext context, auto means, auto output, auto cells, auto metric)
                                { CUDA::AddRotationSource(context, metric, frame, means, output, output, false, cells); }, R::ReadAs<R::TableRows>(state.CoupledField().Owned()), R::ReadWriteAs<R::TableRows>(output), R::ReadAs<R::FixedRows<1>>(selection), R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics()));
        }
        if (!sourceOffsets.RowCount())
            return;
        DNDS_check_throw_info(owner->equation.HasRANS() && sourceValues.size() == 1 && sourceGradients.size() == 1,
                              "device volume source currently supports perfect-gas RANS only");
        SubmitSourceSamples(engine, stream, state);
        if (!result)
            return;
        DNDS_check_throw_info(preparationStatus && preparationStatus->pending, "device source requires an active preparation status segment");
        const int dimension = owner->metrics.Dimension(), components = int(state.ComponentNumber());
        const auto &empty = owner->deviceEmpty;
        const auto &g0 = sourceValuesAreMeans ? directGradients.front().Owned() : sourceGradients.front()[0];
        const auto &g1 = sourceValuesAreMeans ? directGradients.front().Owned() : sourceGradients.front()[1];
        const auto &g2 = dimension == 3 ? (sourceValuesAreMeans ? directGradients.front().Owned() : sourceGradients.front()[2]) : empty;
        R::SubmitArrays(engine, stream, [pin, dimension, components, interleaved = sourceValuesAreMeans](R::ExecutionContext context, auto means, auto samples, auto g0, auto g1, auto g2, auto offsets, auto weights, auto lengths, auto selected, auto output, auto status, auto cells, auto wallMean, auto wallPoints)
                        {
            CUDA::SourceRequest request;
            request.parameters = R::MetadataArrayView(pin->owner->devicePhysics, context.device);
            request.means = means; request.samples = samples;
            const std::array<CUDA::SourceInput, 3> gradients{g0, g1, g2};
            for (int d = 0; d < dimension; ++d)
                request.gradients[d] = {gradients[d], interleaved ? dimension : 1, interleaved ? d : 0};
            request.offsets = offsets; request.weights = weights; request.lengths = lengths;
            request.wallMean = {wallMean.data, wallMean.offsets, wallMean.rows, wallMean.width, wallMean.stride, wallMean.sizes, wallMean.kind, wallMean.device};
            request.wallPoints = {wallPoints.data, wallPoints.offsets, wallPoints.rows, wallPoints.width, wallPoints.stride, wallPoints.sizes, wallPoints.kind, wallPoints.device};
            request.cells = cells;
            request.output = output; request.status = status;
            request.components = components; request.dimension = dimension;
            CUDA::AddRANSSources(context, request, selected); }, R::ReadAs<R::TableRows>(state.CoupledField().Owned()), R::ReadAs<R::TableRows>(sourceValues.front()), R::ReadAs<R::TableRows>(g0), R::ReadAs<R::TableRows>(g1), R::ReadAs<R::TableRows>(g2), R::ReadAs<R::FixedRows<1>>(sourceOffsets), R::ReadAs<R::FixedRows<1>>(sourceWeights), R::ReadAs<R::FixedRows<1>>(owner->ransCellLength), R::ReadAs<R::FixedRows<1>>(DeviceCellSelection(CFV::ReconstructionRegion::All)), R::ReadWriteAs<R::UniformTableRows>(result->CoupledField().Owned()), R::WriteAs<R::FixedRows<1>>(preparationStatus->meanOwned), R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics()), R::ReadAs<R::FixedRows<1>>(owner->wallDistance->CellMeans()), R::ReadAs<R::CSRRows>(owner->wallDistance->CellValues()));
        AppendPreparationStatus(engine, stream, preparationStatus->meanOwned);
#else
        DNDS_check_throw_info(false, "CUDA volume sources are not built");
#endif
    }
}

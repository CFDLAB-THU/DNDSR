#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::SubmitVolumeSource(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const EquationDofs &state, OptionalRef<EquationDofs> result, real time,
        std::shared_ptr<PointEvaluator> evaluator, OptionalRef<EquationDofs> lowOrder)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() && std::isfinite(time),
                              "source submission stream, placement or time differs");
        FiniteVolume(owner).ValidateWorkspace(state, *this);
        DNDS_check_throw_info(!lowOrder || result, "low-order source output requires the primary residual");
        const auto checkOutput = [&](const EquationDofs &output)
        {
            state.RequireCompatible(output);
            EquationDofOperations::RequireDistinct(state, output);
            if (activeFluxCoefficients)
                EquationDofOperations::RequireDistinct(FluxCoefficients(), output);
            for (index iField = 0; iField < output.PhysicalFieldNumber(); ++iField)
            {
                const auto &target = output.PhysicalField(iField).Owned();
                for (const auto &values : sourceValues)
                    DNDS_check_throw_info(!target.Storage().Overlaps(values.Storage()), "source residual aliases sampling scratch");
                if (ransFirst >= 0)
                    for (index iSource = 0; iSource < state.PhysicalFieldNumber(); ++iSource)
                        if (!sourceSampling)
                            DNDS_check_throw_info(!target.Storage().Overlaps(directGradients[iSource].Owned().Storage()),
                                                  "source residual aliases direct gradients");
                        else
                            for (int d = 0; d < owner->metrics.Dimension(); ++d)
                                DNDS_check_throw_info(!target.Storage().Overlaps(sourceGradients[iSource][d].Storage()),
                                                      "source residual aliases sampled gradients");
            }
        };
        if (result)
            checkOutput(*result);
        if (lowOrder)
        {
            checkOutput(*lowOrder);
            EquationDofOperations::RequireDistinct(*result, *lowOrder);
        }
        if (sourceOffsets.RowCount())
        {
            sourceState.reset();
            sourceCoefficients.reset();
        }
        if (owner->metrics.Device() != R::Device::Host())
        {
            SubmitDeviceVolumeSource(engine, stream, state, result, time, lowOrder);
            return;
        }
        const auto timing = BeginQueuedPhase(engine, stream, SourcePhase);
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            Geom::MeshReadAccess geometry;
            bool walls = false;
            R::MetadataBuffer<index> logicalFirst;
            R::MetadataBuffer<EquationBlockLocation> locations;
            std::shared_ptr<PointEvaluator> evaluator;
            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
            Physics::Thermodynamics::MixturePropertyRequest propertyRequest;
            std::vector<index> meanSlots, sampleSlots, resultSlots;
            std::vector<std::array<index, 3>> gradientSlots;
            std::optional<EquationBlockLocation> ransMean, ransOutput, speciesOutput;
            index components = 0, flowFirst = -1, speciesFirst = -1, ransFirst = -1;
            bool chemical = false, interleavedGradients = false;
            explicit Inputs(std::shared_ptr<const Data> data)
                : owner(std::move(data)), geometry(owner->metrics.AcquireMeshRead(bool(owner->wallDistance))) {}
        };
        auto pin = std::make_shared<Inputs>(owner);
        pin->evaluator = std::move(evaluator);
        pin->workers = workers;
        const bool viscous = !meanOnly && owner->equation.Viscous();
        pin->propertyRequest = {viscous || owner->equation.HasRANS(), viscous, viscous, chemical, false};
        const auto frame = owner->motion.At(time);
        using Input = R::ArrayOperand<real, false, R::UniformTableRows>;
        using Output = R::ArrayOperand<real, true, R::UniformTableRows>;
        if (result && frame.moving)
        {
            const auto flow = state.LocateBlock("flow"), target = result->LocateBlock("flow");
            const auto low = lowOrder ? std::optional<EquationBlockLocation>(lowOrder->LocateBlock("flow"))
                                      : std::optional<EquationBlockLocation>{};
            std::vector<Output> outputs{R::ReadWriteAs<R::UniformTableRows>(result->PhysicalField(target.physicalField).Owned())};
            if (low)
                outputs.push_back(R::ReadWriteAs<R::UniformTableRows>(lowOrder->PhysicalField(low->physicalField).Owned()));
            R::SubmitArrayBatch(engine, stream, [pin, flow, target, low, frame](R::ExecutionContext, auto inputs, auto outputs, auto cells)
                                {
                const EquationBlockReadView mean{{inputs[0], flow.firstComponent, flow.componentNumber}, {{}, flow.firstComponent, flow.componentNumber}};
                const EquationBlockArrayView<real> output{outputs[0], target.firstComponent, target.componentNumber};
                std::optional<EquationBlockArrayView<real>> lowOutput;
                if (low) lowOutput.emplace(EquationBlockArrayView<real>{outputs[1], low->firstComponent, low->componentNumber});
                CFV::FiniteVolumeHostView metric;
                metric.cells = cells;
                CPU::AddEquationRotationSource(metric, frame, mean, output, lowOutput); }, std::vector<Input>{R::ReadAs<R::UniformTableRows>(state.PhysicalField(flow.physicalField).Owned())}, std::move(outputs), std::make_tuple(R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics())));
        }
        if (!sourceOffsets.RowCount())
        {
            EndQueuedPhase(engine, stream, timing);
            return;
        }
        SubmitSourceSamples(engine, stream, state);
        if (!result)
        {
            EndQueuedPhase(engine, stream, timing);
            return;
        }
        pin->walls = bool(owner->wallDistance);
        pin->logicalFirst = logicalFirst;
        pin->locations = R::MetadataBuffer<EquationBlockLocation>::Generate(state.BlockNumber(),
                                                                            [&](index iBlock)
                                                                            { return state.LocateBlock(iBlock); });
        pin->components = state.ComponentNumber();
        pin->flowFirst = flowFirst;
        pin->speciesFirst = speciesFirst;
        pin->ransFirst = ransFirst;
        pin->chemical = chemical;
        pin->interleavedGradients = !sourceSampling;
        if (ransFirst >= 0)
        {
            pin->ransMean = state.LocateBlock("rans");
            pin->ransOutput = result->LocateBlock("rans");
        }
        if (speciesFirst >= 0)
            pin->speciesOutput = result->LocateBlock("species");
        std::vector<Input> inputs;
        std::vector<Output> outputs;
        const auto read = [&](const R::Array<real> &array)
        {
            const index slot = index(inputs.size());
            inputs.push_back(R::ReadAs<R::UniformTableRows>(array));
            return slot;
        };
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            pin->meanSlots.push_back(read(state.PhysicalField(iField).Owned()));
        if (sourceSampling)
            for (const auto &array : sourceValues)
                pin->sampleSlots.push_back(read(array));
        else
            pin->sampleSlots = pin->meanSlots;
        for (index iField = 0; ransFirst >= 0 && iField < state.PhysicalFieldNumber(); ++iField)
        {
            std::array<index, 3> slots{-1, -1, -1};
            if (!sourceSampling)
            {
                const index slot = read(directGradients[iField].Owned());
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    slots[d] = slot;
            }
            else
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    slots[d] = read(sourceGradients[iField][d]);
            pin->gradientSlots.push_back(slots);
        }
        for (index iField = 0; iField < result->PhysicalFieldNumber(); ++iField)
        {
            pin->resultSlots.push_back(index(outputs.size()));
            outputs.push_back(R::ReadWriteAs<R::UniformTableRows>(result->PhysicalField(iField).Owned()));
        }
        R::SubmitArrayBatch(engine, stream, [pin](R::ExecutionContext, auto inputs, auto outputs, auto offsets, auto weights, auto lengths, auto wallCells, auto wallMean)
                            {
            // Reserve a chemistry worker only after predecessors execute.
            // Holding it while queueing could starve an earlier auxiliary
            // refresh that shares a one-worker pool.
            std::vector<RHSScalarView> samples;
            for (const index slot : pin->sampleSlots) samples.push_back(inputs[slot]);
            std::vector<std::array<CPU::SourceGradientView, 3>> gradients(pin->gradientSlots.size());
            for (index iField = 0; iField < index(pin->gradientSlots.size()); ++iField)
                for (int d = 0; d < pin->owner->metrics.Dimension(); ++d)
                    gradients[iField][d] = {inputs[pin->gradientSlots[iField][d]],
                        pin->interleavedGradients ? pin->owner->metrics.Dimension() : 1,
                        pin->interleavedGradients ? d : 0};
            CPU::SourceKernelRequest request;
            request.offsets = offsets;
            request.weights = weights;
            if (pin->ransFirst >= 0)
                request.lengths = {lengths.data, lengths.offsets, lengths.rows, lengths.width, lengths.stride, lengths.sizes, lengths.kind, lengths.device};
            if (pin->walls)
            {
                request.walls.cells = {wallCells.data, wallCells.offsets, wallCells.rows, wallCells.width, wallCells.stride, wallCells.sizes, wallCells.kind, wallCells.device};
                request.walls.cellMean = {wallMean.data, wallMean.offsets, wallMean.rows, wallMean.width, wallMean.stride, wallMean.sizes, wallMean.kind, wallMean.device};
            }
            request.cells = pin->owner->metrics.NumCell();
            request.components = pin->components;
            request.dimension = pin->owner->metrics.Dimension();
            request.flowFirst = pin->flowFirst;
            request.speciesFirst = pin->speciesFirst;
            request.ransFirst = pin->ransFirst;
            request.independentSpecies = pin->owner->equation.HasMixture() ? pin->owner->equation.Mixture().SpeciesNumber() - 1 : 0;
            request.chemical = pin->chemical;
            request.cellInputs = pin->interleavedGradients;
            request.locations = pin->locations.HostView();
            request.logicalFirst = pin->logicalFirst.HostView();
            request.samples = {samples.data(), index(samples.size())};
            request.gradients = {gradients.data(), index(gradients.size())};
            const auto outputBlock = [&](EquationBlockLocation location)
            {
                return EquationBlockArrayView<real>{outputs[pin->resultSlots[location.physicalField]], location.firstComponent, location.componentNumber};
            };
            if (pin->speciesOutput) request.speciesOutput = outputBlock(*pin->speciesOutput);
            if (pin->ransOutput) request.ransOutput = outputBlock(*pin->ransOutput);
            if (pin->ransMean)
            {
                const auto location = *pin->ransMean;
                request.ransMean = {{inputs[pin->meanSlots[location.physicalField]], location.firstComponent, location.componentNumber},
                                   {{}, location.firstComponent, location.componentNumber}};
            }
            if (pin->evaluator)
                RequireNumericalSuccess(CPU::EvaluateVolumeSource(request, pin->owner->equation, *pin->evaluator), "queued volume source");
            else
            {
                auto workers = MakeCPUPointWorkers<CPU::SourceWorkerScratch>(pin->owner->equation, pin->workers,
                    pin->propertyRequest, pin->components);
                RequireNumericalSuccess(CPU::EvaluateVolumeSource(request, pin->owner->equation,
                    SpanView<CPU::SourceWorkerScratch>{workers.data(), index(workers.size())}), "queued volume source");
            } }, std::move(inputs), std::move(outputs), std::make_tuple(R::ReadAs<R::FixedRows<1>>(sourceOffsets), R::ReadAs<R::FixedRows<1>>(sourceWeights), R::ReadAs<R::RowLayout>(ransFirst >= 0 ? owner->ransCellLength : owner->deviceEmpty), R::ReadAs<R::RowLayout>(pin->walls ? owner->wallDistance->CellValues() : owner->deviceEmpty), R::ReadAs<R::RowLayout>(pin->walls ? owner->wallDistance->CellMeans() : owner->deviceEmpty)));
        EndQueuedPhase(engine, stream, timing);
    }
}

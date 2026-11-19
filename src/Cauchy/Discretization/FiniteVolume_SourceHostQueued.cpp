#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::SubmitVolumeSource(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const EquationDofs &state, EquationDofs &result, real time,
        std::shared_ptr<PointEvaluator> evaluator, OptionalRef<EquationDofs> lowOrder, OptionalRef<R::Array<real>> sourceDerivative)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() && std::isfinite(time),
                              "source submission stream, placement or time differs");
        FiniteVolume(owner).ValidateWorkspace(state, *this);

        const auto checkOutput = [&](const EquationDofs &output)
        {
            state.RequireCompatible(output);
            EquationDofOperations::RequireDistinct(state, output);
            if (activeFluxCoefficients)
                EquationDofOperations::RequireDistinct(FluxCoefficients(), output);
            for (index iField = 0; iField < output.PhysicalFieldNumber(); ++iField)
            {
                const auto &target = output.PhysicalField(iField).Owned();
                if (!sourceSampling && ransFirst >= 0)
                    for (const auto &g : directGradients)
                        DNDS_check_throw_info(!target.Storage().Overlaps(g.Owned().Storage()), "source residual aliases direct gradients");
            }
        };
        checkOutput(result);
        if (lowOrder)
        {
            checkOutput(*lowOrder);
            EquationDofOperations::RequireDistinct(result, *lowOrder);
        }
        if (owner->metrics.Device() != R::Device::Host())
        {
            SubmitDeviceVolumeSource(engine, stream, state, result, time, lowOrder, sourceDerivative);
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
            std::vector<index> meanSlots, coefficientSlots, resultSlots;
            std::vector<std::array<index, 3>> gradientSlots;
            std::optional<EquationBlockLocation> ransMean, ransOutput, speciesOutput;
            index components = 0, flowFirst = -1, speciesFirst = -1, ransFirst = -1;
            bool chemical = false, interleavedGradients = false;
            std::optional<index> derivativeSlot, ransP1Slot;
            std::array<index, 4> basisSlots{};
            VolumeSourceBasis basis;
            explicit Inputs(std::shared_ptr<const Data> data)
                : owner(std::move(data)), geometry(owner->metrics.AcquireMeshRead(bool(owner->wallDistance))) {}
        };
        auto pin = std::make_shared<Inputs>(owner);
        pin->evaluator = std::move(evaluator);
        pin->workers = workers;
        const bool viscous = !meanOnly && owner->equation.Viscous();
        pin->propertyRequest = {viscous || owner->equation.HasRANS(), viscous, viscous, chemical, chemical && bool(sourceDerivative)};
        const auto frame = owner->motion.At(time);
        using Input = R::ArrayOperand<real, false, R::UniformTableRows>;
        using Output = R::ArrayOperand<real, true, R::UniformTableRows>;
        if (frame.moving)
        {
            const auto flow = state.LocateBlock("flow"), target = result.LocateBlock("flow");
            const auto low = lowOrder ? std::optional<EquationBlockLocation>(lowOrder->LocateBlock("flow"))
                                      : std::optional<EquationBlockLocation>{};
            std::vector<Output> outputs{R::ReadWriteAs<R::UniformTableRows>(result.PhysicalField(target.physicalField).Owned())};
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
        if (sourceSampling)
            sourceSampling->Validate();
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
            pin->ransOutput = result.LocateBlock("rans");
        }
        if (speciesFirst >= 0)
            pin->speciesOutput = result.LocateBlock("species");
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
        {
            pin->basis.polynomial = sourceSampling->BasisParameters();
            pin->basis.cachedDerivatives = sourceSampling->CachedDerivativeNumber();
            pin->basisSlots = {read(sourceSampling->PointStorage()), read(sourceSampling->BasisStorage()), read(sourceSampling->MomentStorage()), read(sourceSampling->CellGeometryStorage())};
            for (index f = 0; f < state.PhysicalFieldNumber(); ++f)
                pin->coefficientSlots.push_back(read(FluxCoefficients().PhysicalField(f).Owned()));
            if (owner->settings.numerics.ransP1Overlay)
            {
                DNDS_check_throw_info(meanGradients && pin->ransMean, "RANS P1 source overlay requires mean gradients");
                pin->ransP1Slot = read(meanGradients->fields[pin->ransMean->physicalField].Owned());
            }
        }
        else if (ransFirst >= 0)
            for (const auto &gradient : directGradients)
            {
                const index slot = read(gradient.Owned());
                pin->gradientSlots.push_back({slot, slot, slot});
            }
        for (index iField = 0; iField < result.PhysicalFieldNumber(); ++iField)
        {
            pin->resultSlots.push_back(index(outputs.size()));
            outputs.push_back(R::ReadWriteAs<R::UniformTableRows>(result.PhysicalField(iField).Owned()));
        }
        if (sourceDerivative)
        {
            pin->derivativeSlot = index(outputs.size());
            outputs.push_back(R::WriteAs<R::UniformTableRows>(*sourceDerivative));
        }
        R::SubmitArrayBatch(engine, stream, [pin](R::ExecutionContext, auto inputs, auto outputs, auto offsets, auto weights, auto lengths, auto wallCells, auto wallMean, auto cells)
                            {
            // Reserve a chemistry worker only after predecessors execute.
            // Holding it while queueing could starve an earlier auxiliary
            // refresh that shares a one-worker pool.
            std::vector<RHSScalarView> samples;
            for (const index slot : pin->meanSlots) samples.push_back(inputs[slot]);
            std::vector<std::array<CPU::SourceGradientView, 3>> gradients(pin->gradientSlots.size());
            for (index iField = 0; iField < index(pin->gradientSlots.size()); ++iField)
                for (int d = 0; d < pin->owner->metrics.Dimension(); ++d)
                    gradients[iField][d] = {inputs[pin->gradientSlots[iField][d]],
                        pin->interleavedGradients ? pin->owner->metrics.Dimension() : 1,
                        pin->interleavedGradients ? d : 0};
            CPU::SourceKernelRequest request;
            request.metric.cells = cells;
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
            if (pin->derivativeSlot)
            {
                request.fullBlock = true;
                request.preparedDerivative = outputs[*pin->derivativeSlot];
            }
            request.cellInputs = pin->interleavedGradients;
            request.locations = pin->locations.HostView();
            request.logicalFirst = pin->logicalFirst.HostView();
            std::vector<RHSScalarView> coefficients;
            for(const index slot:pin->coefficientSlots) coefficients.push_back(inputs[slot]);
            if(!coefficients.empty())
            {
                auto basis=pin->basis;
                basis.points=inputs[pin->basisSlots[0]]; basis.cached=inputs[pin->basisSlots[1]];
                basis.moments=inputs[pin->basisSlots[2]]; basis.geometry=inputs[pin->basisSlots[3]];
                request.basis=basis;
                request.coefficients={coefficients.data(),index(coefficients.size())};
            }
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
                if (pin->ransP1Slot)
                {
                    request.ransP1Gradients = inputs[*pin->ransP1Slot];
                    request.ransP1First = location.firstComponent;
                }
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
            } }, std::move(inputs), std::move(outputs), std::make_tuple(R::ReadAs<R::FixedRows<1>>(sourceOffsets), R::ReadAs<R::FixedRows<1>>(sourceWeights), R::ReadAs<R::RowLayout>(ransFirst >= 0 ? owner->ransCellLength : owner->deviceEmpty), R::ReadAs<R::RowLayout>(pin->walls ? owner->wallDistance->CellValues() : owner->deviceEmpty), R::ReadAs<R::RowLayout>(pin->walls ? owner->wallDistance->CellMeans() : owner->deviceEmpty), R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics())));
        EndQueuedPhase(engine, stream, timing);
    }
}

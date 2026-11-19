#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeRHSKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        using ReadRows = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
        using WriteRows = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
        using ReadOperand = Runtime::ArrayOperand<real, false, Runtime::UniformTableRows>;
        using WriteOperand = Runtime::ArrayOperand<real, true, Runtime::UniformTableRows>;
        struct PairSlots
        {
            index owned = -1, ghost = -1;
            RHSFieldView View(SpanView<const ReadRows> views) const
            {
                return {owned >= 0 ? views[owned] : ReadRows{}, ghost >= 0 ? views[ghost] : ReadRows{}};
            }
        };
        struct ReadBlocks
        {
            std::vector<EquationBlockReadView> blocks;
            EquationBlockReadView Block(index iBlock) const { return blocks[iBlock]; }
        };
        struct WriteBlocks
        {
            std::vector<EquationBlockArrayView<real>> blocks;
            EquationBlockArrayView<real> Block(index iBlock) const { return blocks[iBlock]; }
        };
        struct ReadEquation
        {
            EquationDofs shape;
            std::vector<PairSlots> fields;
            ReadBlocks View(SpanView<const ReadRows> rows) const
            {
                ReadBlocks result;
                result.blocks.reserve(size_t(shape.BlockNumber()));
                for (index iBlock = 0; iBlock < shape.BlockNumber(); ++iBlock)
                {
                    const auto location = shape.LocateBlock(iBlock);
                    const auto field = fields[location.physicalField].View(rows);
                    result.blocks.push_back({{field.owned, location.firstComponent, location.componentNumber},
                                             {field.ghost, location.firstComponent, location.componentNumber}});
                }
                return result;
            }
        };
        struct WriteEquation
        {
            EquationDofs shape;
            std::vector<index> fields;
            WriteBlocks View(SpanView<const WriteRows> rows) const
            {
                WriteBlocks result;
                result.blocks.reserve(size_t(shape.BlockNumber()));
                for (index iBlock = 0; iBlock < shape.BlockNumber(); ++iBlock)
                {
                    const auto location = shape.LocateBlock(iBlock);
                    result.blocks.push_back({rows[fields[location.physicalField]], location.firstComponent, location.componentNumber});
                }
                return result;
            }
        };
        template <class T>
        Runtime::ArrayView<T, Runtime::AccessBoundsChecks, RHSFlowRows> FlowRows(
            Runtime::ArrayView<T, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> view)
        {
            DNDS_check_throw_info(view.Width() == 5 || (!view.RowCount() && !view.data), "queued fixed-flow view width differs");
            return {view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device};
        }
        RHSFlowFieldView FlowPair(RHSFieldView pair) { return {FlowRows(pair.owned), FlowRows(pair.ghost)}; }
    }

    std::shared_ptr<FiniteVolume::Workspace::RHSSubmission> FiniteVolume::Workspace::SubmitRHS(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state, EquationDofs &output,
        CFV::ReconstructionRegion region, real time, std::shared_ptr<PointEvaluator> evaluator,
        OptionalRef<const AuxiliaryValues> auxiliary, OptionalRef<R::Array<real>> faceOutput,
        OptionalRef<const R::Array<real>> incoming)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() &&
                                  state.Device() == owner->metrics.Device() && std::isfinite(time),
                              "RHS submission backend, stream or time differs");
        state.RequireSchema(owner->equation.Schema());
        FiniteVolume(owner).ValidateWorkspace(state, *this);
        state.RequireCompatible(output);
        EquationDofOperations::RequireDistinct(state, output);
        DNDS_check_throw_info(state.Space().Same(owner->metrics.Space()) &&
                                  (region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                   region == CFV::ReconstructionRegion::HaloDependent),
                              "RHS submission row space or region differs");
        if (owner->metrics.Device() != R::Device::Host())
        {
            DNDS_check_throw_info(!faceOutput, "queued CUDA RHS currently publishes its workspace face-flux array");
            SubmitDeviceRHS(engine, stream, state, output, region, time, auxiliary, incoming);
            return {};
        }
        DNDS_check_throw_info(!fixedFlowRHS || (state.PhysicalFieldNumber() == 1 && output.PhysicalFieldNumber() == 1 &&
                                                state.ComponentNumber() == 5 && output.ComponentNumber() == 5),
                              "queued fixed-flow leaf requires one flow5 physical field");
        const bool ghosts = region != CFV::ReconstructionRegion::Core;
        const auto &n = owner->settings.numerics;
        const bool modal = n.reconstruction == "variational";
        const bool separateViscous = !n.viscousLimited && (n.limiter == "WBAP" || limitedVR || n.positivity);
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            ThermalPolicy thermal;
            R::MetadataBuffer<index> logicalFirst, vectorStarts;
            Geom::MeshReadAccess geometry;
            FiniteVolumeStencilStorage::ReadAccess stencil;
            bool basis = false, walls = false;
            std::optional<RHSOperator> fixed;
            std::shared_ptr<PointEvaluator> evaluator;
            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
            Physics::Thermodynamics::MixturePropertyRequest properties;
            ReadEquation means;
            WriteEquation result;
            std::optional<ReadEquation> coefficients, viscousCoefficients;
            std::optional<WriteEquation> lowResult;
            std::vector<PairSlots> gradients, viscousGradients;
            PairSlots primitive, waveJump, pressureMinimum, derived, ransP1;
            index ransP1First = 0;
            std::optional<DerivedFields> derivedOwned, derivedGhost;
            std::array<int, 5> derivedColumns{0, 1, 2, -1, -1};
            index derivedWidth = 3;
            index meanThermodynamics = -1, incoming = -1, faces = -1, lowFaces = -1, wallFlux = -1;
            index flowFirst = -1, speciesFirst = -1, ransFirst = -1, ransBlock = -1;
            bool genericFluxOracle = false, meanOnly = false, hasPrimitive = false, captureBoundaryFlux = false, boundaryFacesOnly = false;

            Inputs(std::shared_ptr<const Data> data, const ThermalPolicy &policy,
                   R::MetadataBuffer<index> offsets, R::MetadataBuffer<index> vectors,
                   std::shared_ptr<PointEvaluator> pointEvaluator, const EquationDofs &state, EquationDofs &output)
                : owner(std::move(data)), thermal(policy), logicalFirst(std::move(offsets)), vectorStarts(std::move(vectors)),
                  geometry(owner->metrics.AcquireMeshRead(bool(owner->wallDistance))), stencil(owner->stencil.Read()), evaluator(std::move(pointEvaluator)),
                  means{state.NumericAlias(), {}}, result{output.NumericAlias(), {}} {}
        };
        auto pin = std::make_shared<Inputs>(owner, thermal, logicalFirst, vectorStarts, std::move(evaluator), state, output);
        pin->fixed = fixedFlowRHS;
        pin->genericFluxOracle = genericFluxOracle;
        pin->workers = workers;
        const bool viscous = !meanOnly && owner->equation.Viscous();
        pin->properties = {viscous || owner->equation.HasRANS(), viscous, viscous, chemical, false};
        pin->flowFirst = flowFirst;
        pin->speciesFirst = speciesFirst;
        pin->ransFirst = ransFirst;
        pin->meanOnly = meanOnly;
        pin->captureBoundaryFlux = captureBoundaryFlux;
        pin->boundaryFacesOnly = !faceOutput;
        pin->hasPrimitive = bool(primitiveGradient);
        if (!pin->fixed)
        {
            if (modal)
                pin->basis = true;
            if (owner->wallDistance)
                pin->walls = true;
        }
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            if (state.Schema().blocks[iBlock].name == "rans")
                pin->ransBlock = iBlock;
        std::vector<ReadOperand> reads;
        std::vector<WriteOperand> writes;
        const auto addRead = [&](const R::Array<real> &array)
        {
            const index slot = index(reads.size());
            reads.push_back(R::ReadAs<R::UniformTableRows>(array));
            return slot;
        };
        const auto addWrite = [&](R::Array<real> &array)
        {
            const index slot = index(writes.size());
            writes.push_back(R::WriteAs<R::UniformTableRows>(array));
            return slot;
        };
        const auto addField = [&](const CFV::FVField &field)
        {
            return PairSlots{addRead(field.Owned()), ghosts ? addRead(field.Ghost()) : -1};
        };
        const auto addEquation = [&](const EquationDofs &values)
        {
            ReadEquation group{values.NumericAlias(), {}};
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                group.fields.push_back(addField(values.PhysicalField(iField)));
            return group;
        };
        const auto addOutput = [&](EquationDofs &values)
        {
            WriteEquation group{values.NumericAlias(), {}};
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                group.fields.push_back(addWrite(values.PhysicalField(iField).Owned()));
            return group;
        };
        pin->means = addEquation(state);
        pin->result = addOutput(output);
        if (lowOrderResidual)
            pin->lowResult = addOutput(*lowOrderResidual);
        if (modal)
        {
            pin->coefficients = addEquation(FluxCoefficients());
            if (separateViscous && (pin->fixed || owner->equation.Viscous()))
                pin->viscousCoefficients = addEquation(Coefficients());
        }
        else if (!meanOnly && !n.firstOrderConvection && n.reconstruction == "gradient")
        {
            DNDS_check_throw_info(directGradients.size() == size_t(state.PhysicalFieldNumber()), "direct FV requires physical gradients");
            for (const auto &gradient : directGradients)
                pin->gradients.push_back(addField(gradient));
            if (!pin->fixed || separateViscous)
                for (const auto &gradient : directViscousGradients)
                    pin->viscousGradients.push_back(addField(gradient));
        }
        // The fixed leaf retains a separate viscous gradient even for a
        // first-order convective trial, matching its original input contract.
        if (pin->fixed && !modal && separateViscous && pin->viscousGradients.empty() && !directViscousGradients.empty())
            pin->viscousGradients.push_back(addField(directViscousGradients.front()));
        if (UseRANSP1FaceOverlay())
        {
            DNDS_check_throw_info(meanGradients, "RANS P1 face overlay requires mean gradients");
            const auto location = state.LocateBlock("rans");
            pin->ransP1 = addField(meanGradients->fields[location.physicalField]);
            pin->ransP1First = location.firstComponent;
        }
        if (primitiveGradient)
            pin->primitive = addField(*primitiveGradient);
        if (auxiliary)
        {
            if (auxiliary->meanThermodynamics)
                pin->meanThermodynamics = addRead(*auxiliary->meanThermodynamics);
            else if (pin->fixed && auxiliary->derivedOwned)
            {
                pin->derivedOwned = *auxiliary->derivedOwned;
                pin->derived.owned = addRead(pin->derivedOwned->Values());
                const auto columns = [](const DerivedFields &values)
                {
                    std::array<int, 5> result{values.Column(DerivedQuantity::Pressure), values.Column(DerivedQuantity::SoundSpeed),
                                              values.Column(DerivedQuantity::DiffusionBound), -1, -1};
                    if (values.Request().Has(DerivedQuantity::TotalEnthalpy) && values.Request().Has(DerivedQuantity::PressureInternalEnergyDerivative))
                    {
                        result[3] = values.Column(DerivedQuantity::TotalEnthalpy);
                        result[4] = values.Column(DerivedQuantity::PressureInternalEnergyDerivative);
                    }
                    return result;
                };
                pin->derivedColumns = columns(*pin->derivedOwned);
                pin->derivedWidth = pin->derivedColumns[3] >= 0 ? 5 : 3;
                if (ghosts && auxiliary->derivedGhost)
                {
                    pin->derivedGhost = *auxiliary->derivedGhost;
                    pin->derived.ghost = addRead(pin->derivedGhost->Values());
                    DNDS_check_throw_info(columns(*pin->derivedGhost) == pin->derivedColumns, "derived owned/ghost columns differ");
                }
            }
            if (auxiliary->cellWaveJump)
                pin->waveJump = addField(*auxiliary->cellWaveJump);
            if (auxiliary->cellPressureMinimum)
                pin->pressureMinimum = addField(*auxiliary->cellPressureMinimum);
        }
        if (incoming)
            pin->incoming = addRead(*incoming);
        if (pin->fixed)
        {
            if (faceOutput || captureBoundaryFlux)
                pin->faces = addWrite(faceOutput ? *faceOutput : faceFlux);
            if (captureBoundaryFlux)
                pin->wallFlux = addWrite(boundaryMomentumFlux);
        }
        else
        {
            pin->faces = addWrite(faceOutput ? *faceOutput : faceFlux);
            R::FixedRows<3>::Validate(boundaryMomentumFlux.Layout());
            pin->wallFlux = addWrite(boundaryMomentumFlux);
            if (pin->lowResult)
            {
                DNDS_check_throw_info(lowOrderFaceFlux && state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                      "low-order equation flux requires coupled flow5");
                pin->lowFaces = addWrite(*lowOrderFaceFlux);
            }
        }
        for (const auto &write : writes)
            for (const auto &read : reads)
                DNDS_check_throw_info(!write.storage->Values().Overlaps(read.storage->Values()), "queued RHS output aliases an input");
        for (index i = 0; i < index(writes.size()); ++i)
            for (index j = 0; j < i; ++j)
                DNDS_check_throw_info(!writes[i].storage->Values().Overlaps(writes[j].storage->Values()), "queued RHS outputs overlap");
        auto result = std::make_shared<RHSSubmission>();
        result->region = region;
        const TimelineScope timeline("Cauchy/FV/RHS/HostSubmit");
        const auto timing = BeginQueuedPhase(engine, stream, RHSPhase);
        R::SubmitArrayBatch(engine, stream, [pin, result, region, time](R::ExecutionContext, auto inputs, auto outputs, auto cached, auto moments, auto wallFaces, auto wallMean)
                            {
            if (pin->fixed)
            {
                RHSReadViews input;
                RHSWriteViews output;
                input.mean = FlowPair(pin->means.fields.front().View(inputs));
                input.includeViscosity = !pin->meanOnly;
                input.meanOnly = pin->meanOnly;
                if (pin->coefficients) input.coefficients = pin->coefficients->fields.front().View(inputs);
                if (pin->viscousCoefficients) input.viscousCoefficients = pin->viscousCoefficients->fields.front().View(inputs);
                if (!pin->gradients.empty()) input.conservativeGradient = pin->gradients.front().View(inputs);
                if (!pin->viscousGradients.empty()) input.viscousConservativeGradient = pin->viscousGradients.front().View(inputs);
                if (pin->hasPrimitive) input.primitiveGradient = pin->primitive.View(inputs);
                if (pin->meanThermodynamics >= 0) input.auxiliary.meanThermodynamics.emplace(inputs[pin->meanThermodynamics]);
                else if (pin->derivedOwned)
                {
                    // Publication is completed-version based today. Validate
                    // at execution, after preceding queued source updates;
                    // do not acquire immediate source/payload read leases here.
                    DNDS_check_throw_info(pin->derivedOwned->Current(pin->means.shape, pin->owner->equation) &&
                                              (!pin->derivedGhost || pin->derivedGhost->Current(pin->means.shape, pin->owner->equation)),
                                          "queued RHS derived properties are stale or unpublished");
                    input.auxiliary.meanThermodynamics.emplace(pin->derived.View(inputs), pin->derivedColumns, pin->derivedWidth);
                }
                if (pin->waveJump.owned >= 0) input.auxiliary.cellWaveJump = pin->waveJump.View(inputs);
                if (pin->pressureMinimum.owned >= 0) input.auxiliary.cellPressureMinimum = pin->pressureMinimum.View(inputs);
                if (pin->incoming >= 0) input.boundary = inputs[pin->incoming];
                output.residual = FlowRows(outputs[pin->result.fields.front()]);
                if (pin->lowResult) output.lowOrder = FlowRows(outputs[pin->lowResult->fields.front()]);
                if (pin->faces >= 0) output.faceFlux = FlowRows(outputs[pin->faces]);
                output.boundaryFacesOnly = pin->boundaryFacesOnly;
                if (pin->wallFlux >= 0)
                {
                    const auto wall = outputs[pin->wallFlux];
                    output.boundaryMomentum = {wall.data, wall.offsets, wall.rows, wall.width, wall.stride, wall.sizes, wall.kind, wall.device};
                }
                result->fixedDiagnostics = pin->fixed->ApplyFluxBorrowedFatal(input, output, region, time);
            }
            else
            {
                auto evaluator = pin->evaluator;
                const auto means = pin->means.View(inputs);
                const auto destination = pin->result.View(outputs);
                const EquationViewLayout<const real> owned(pin->means.shape, means), remote(pin->means.shape, means, true);
                const EquationViewLayout<real> output(pin->result.shape, destination);
                std::optional<WriteBlocks> lowBlocks;
                std::optional<EquationViewLayout<real>> lowLayout;
                if (pin->lowResult)
                {
                    lowBlocks.emplace(pin->lowResult->View(outputs));
                    lowLayout.emplace(pin->lowResult->shape, *lowBlocks);
                }
                const auto rec = pin->coefficients ? pin->coefficients->View(inputs) : ReadBlocks{};
                const auto viscous = pin->viscousCoefficients ? pin->viscousCoefficients->View(inputs) : ReadBlocks{};
                std::vector<EquationBlockLocation> locations;
                if (!pin->basis)
                    for (index iBlock = 0; iBlock < pin->means.shape.BlockNumber(); ++iBlock)
                        locations.push_back(pin->means.shape.LocateBlock(iBlock));
                std::vector<RHSFieldView> gradients, viscousGradients;
                for (const auto &field : pin->gradients) gradients.push_back(field.View(inputs));
                for (const auto &field : pin->viscousGradients) viscousGradients.push_back(field.View(inputs));
                CPU::EquationRHSRequest request;
                request.stencil = pin->stencil.View();
                request.metric = request.stencil.metric;
                if (pin->basis)
                {
                    const auto &settings = pin->owner->reconstruction->Settings();
                    request.basis = {cached, moments, request.metric, pin->geometry.HostView(),
                                     pin->owner->metrics.Dimension(), settings.maxOrder, pin->owner->reconstruction->NRecDOF(),
                                     settings.cacheDiffBase ? int(cached.width / pin->owner->reconstruction->NRecDOF()) : 0};
                }
                request.conditions = {pin->owner->boundaryConditions.data(), index(pin->owner->boundaryConditions.size())};
                request.meanBlocks = {means.blocks.data(), index(means.blocks.size())};
                request.recBlocks = {rec.blocks.data(), index(rec.blocks.size())};
                request.viscousRecBlocks = {viscous.blocks.data(), index(viscous.blocks.size())};
                request.directLocations = {locations.data(), index(locations.size())};
                request.directViews = {gradients.data(), index(gradients.size())};
                request.directViscousViews = {viscousGradients.data(), index(viscousGradients.size())};
                request.logicalFirst = pin->logicalFirst.HostView();
                request.vectorStarts = pin->vectorStarts.HostView();
                if (pin->ransBlock >= 0) request.ransMean = means.Block(pin->ransBlock);
                if (pin->meanThermodynamics >= 0) request.preparedMean = inputs[pin->meanThermodynamics];
                request.waveJump = pin->waveJump.View(inputs);
                request.pressureMinimum = pin->pressureMinimum.View(inputs);
                request.primitiveGradientValues = pin->primitive.View(inputs);
                request.ransP1Gradients = pin->ransP1.View(inputs);
                request.ransP1First = pin->ransP1First;
                if (pin->incoming >= 0) request.incoming = inputs[pin->incoming];
                request.faceValues = outputs[pin->faces];
                if (pin->lowFaces >= 0) request.lowFaceValues = outputs[pin->lowFaces];
                const auto wall = outputs[pin->wallFlux];
                DNDS_check_throw_info(wall.Width() == 3, "queued boundary momentum width differs");
                request.wallFlux = {wall.data, wall.offsets, wall.rows, wall.width, wall.stride, wall.sizes, wall.kind, wall.device};
                request.captureBoundaryFlux = pin->captureBoundaryFlux;
                request.genericFluxOracle = pin->genericFluxOracle;
                if (pin->walls)
                {
                    request.wall.faces = {wallFaces.data, wallFaces.offsets, wallFaces.rows, wallFaces.width, wallFaces.stride, wallFaces.sizes, wallFaces.kind, wallFaces.device};
                    request.wall.cellMean = {wallMean.data, wallMean.offsets, wallMean.rows, wallMean.width, wallMean.stride, wallMean.sizes, wallMean.kind, wallMean.device};
                }
                request.thermal = pin->thermal.View();
                request.frame = pin->owner->motion.At(time);
                request.fluxPolicy = pin->owner->fluxPolicy;
                request.ownedCells = pin->owner->metrics.NumCell();
                request.width = pin->means.shape.ComponentNumber();
                request.flowFirst = pin->flowFirst;
                request.speciesFirst = pin->speciesFirst;
                request.ransFirst = pin->ransFirst;
                request.dimension = pin->owner->metrics.Dimension();
                request.modes = pin->owner->reconstruction ? pin->owner->reconstruction->NRecDOF() : 0;
                request.species = pin->owner->equation.HasMixture() ? pin->owner->equation.Mixture().SpeciesNumber() : 0;
                const auto &settings = pin->owner->settings.numerics;
                request.direct = settings.reconstruction != "variational";
                request.linear = settings.reconstruction == "gradient";
                request.firstOrder = pin->meanOnly || settings.firstOrderConvection;
                request.ransMeanFaceValues = settings.ransMeanFaceValues;
                request.viscous = !pin->meanOnly && pin->owner->equation.Viscous();
                request.primitiveGradient = pin->hasPrimitive;
                request.hasWall = bool(pin->walls);
                request.positivity = settings.positivity;
                request.faceFallback = settings.faceFallback;
                request.useStencil = settings.riemann.useStencil;
                request.preparedWaveJump = pin->waveJump.owned >= 0;
                request.preparedPressureMinimum = pin->pressureMinimum.owned >= 0;
                request.produceLow = bool(pin->lowResult);
                request.time = time;
                std::vector<CPU::EquationRHSWorkerScratch> workers;
                if (!evaluator)
                    workers = MakeCPUPointWorkers<CPU::EquationRHSWorkerScratch>(pin->owner->equation, pin->workers,
                        pin->properties, request.width, request.species, request.produceLow);
                const auto evaluated = evaluator
                    ? CPU::EvaluateEquationFlux(request, pin->owner->equation, pin->owner->boundaryPoint,
                        pin->owner->flux, *evaluator, owned.Descriptor(), remote.Descriptor(),
                        output.Descriptor(), lowLayout ? lowLayout->Descriptor() : output.Descriptor(), region)
                    : CPU::EvaluateEquationFlux(request, pin->owner->equation, pin->owner->boundaryPoint,
                        pin->owner->flux, SpanView<CPU::EquationRHSWorkerScratch>{workers.data(), index(workers.size())},
                        owned.Descriptor(), remote.Descriptor(), output.Descriptor(),
                        lowLayout ? lowLayout->Descriptor() : output.Descriptor(), region);
                RequireNumericalSuccess(evaluated.status, "queued equation RHS");
                result->faceFallbacks = evaluated.value;
            }
            result->completed.store(true, std::memory_order_release); }, std::move(reads), std::move(writes), std::make_tuple(R::ReadAs<R::TableRows>(pin->basis ? owner->reconstruction->SamplingFaceBasis() : owner->deviceEmpty), R::ReadAs<R::TableRows>(pin->basis ? owner->reconstruction->SamplingMoments() : owner->deviceEmpty), R::ReadAs<R::RowLayout>(pin->walls ? owner->wallDistance->FaceValues() : owner->deviceEmpty), R::ReadAs<R::RowLayout>(pin->walls ? owner->wallDistance->CellMeans() : owner->deviceEmpty)));
        EndQueuedPhase(engine, stream, timing);
        return result;
    }

    void FiniteVolume::Workspace::PublishRHS(const std::shared_ptr<RHSSubmission> &result)
    {
        if (!result)
            return;
        DNDS_check_throw_info(result->completed.load(std::memory_order_acquire) && !result->published,
                              "RHS diagnostics are incomplete or already published");
        if (result->fixedDiagnostics)
            diagnostics.Merge(*result->fixedDiagnostics);
        else
        {
            const index slot = result->region == CFV::ReconstructionRegion::HaloDependent ? 1 : 0;
            faceFallbacks[slot] = result->faceFallbacks;
            if (result->region == CFV::ReconstructionRegion::All)
                faceFallbacks[1] = 0;
        }
        result->published = true;
    }
}

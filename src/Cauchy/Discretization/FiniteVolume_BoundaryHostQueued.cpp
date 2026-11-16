#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::SubmitBoundary(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const EquationDofs &state, bool polynomial, real time,
        std::shared_ptr<PointEvaluator> evaluator,
        OptionalRef<const R::Array<real>> incoming,
        OptionalRef<const R::Array<real>> incomingCenters)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() &&
                                  state.Device() == owner->metrics.Device() && std::isfinite(time),
                              "boundary submission backend, stream or time differs");
        state.RequireSchema(owner->equation.Schema());
        DNDS_check_throw_info(state.Space().Same(owner->metrics.Space()) && state.PhysicalFieldNumber() == index(fields.size()),
                              "boundary submission row space or grouping differs");
        if (!owner->boundaryFaces.RowCount())
            return; // Empty consumer: no values, mutable worker or launch needed.
        if (owner->metrics.Device() != R::Device::Host())
        {
            SubmitDeviceBoundary(engine, stream, state, polynomial, time, incoming, incomingCenters);
            return;
        }
        const auto validatePacket = [&](OptionalRef<const R::Array<real>> packet, index rows, bool required)
        {
            DNDS_check_throw_info(!required || packet, "boundary producer lacks a required interface section");
            if (packet)
                DNDS_check_throw_info(packet->RowCount() == rows && packet->Layout().Width() == state.ComponentNumber() &&
                                          !packet->Layout().IsVariable() && packet->GetDevice() == owner->metrics.Device(),
                                      "boundary interface section shape or placement differs from selected location");
        };
        validatePacket(incoming, owner->BoundaryValueRows(), owner->hasLocalInterfaceRows);
        const bool independentCenters = primitiveBoundary || std::any_of(fields.begin(), fields.end(), [](const auto &field)
                                                                         { return field.gradientBoundary && field.gradientBoundary->SharedStorage() != field.boundaryValues.SharedStorage(); });
        validatePacket(incomingCenters, owner->boundaryFaces.RowCount(), independentCenters && owner->hasLocalInterfaceRows);
        const auto timing = BeginQueuedPhase(engine, stream, BoundaryPhase);
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            ThermalPolicy thermal;
            R::MetadataBuffer<index> logicalFirst;
            FiniteVolumeStencilStorage::ReadAccess stencil;
            R::Array<BoundaryFaceRange>::ReadLease faces;
            std::optional<CFV::ReconstructionBasisRead> basis;
            bool walls = false, mirrored = false;
            std::vector<EquationBlockLocation> means, coefficients, targets;
            std::optional<EquationBlockLocation> ransMean;
            std::vector<index> gradientOutputs;
            index fieldNumber = 0, componentNumber = 0;
            index primitiveOutput = -1, incomingInput = -1, centerInput = -1, coefficientFirst = -1;
            index ransFirst = -1, flowFirst = -1;
            std::shared_ptr<PointEvaluator> evaluator;
            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
            Physics::Thermodynamics::MixturePropertyRequest properties;
            Inputs(std::shared_ptr<const Data> data, const ThermalPolicy &policy,
                   R::MetadataBuffer<index> offsets, std::shared_ptr<PointEvaluator> pointEvaluator)
                : owner(std::move(data)), thermal(policy), logicalFirst(std::move(offsets)),
                  stencil(owner->stencil.Read()), faces(owner->boundaryFaces.Read()), evaluator(std::move(pointEvaluator)) {}
        };
        const auto pin = std::make_shared<Inputs>(owner, thermal, logicalFirst, std::move(evaluator));
        pin->workers = workers;
        const bool viscous = !meanOnly && owner->equation.Viscous();
        pin->properties = {viscous || owner->equation.HasRANS(), viscous, viscous, chemical, false};
        pin->fieldNumber = state.PhysicalFieldNumber();
        pin->componentNumber = state.ComponentNumber();
        pin->ransFirst = ransFirst;
        pin->flowFirst = flowFirst;
        pin->mirrored = owner->settings.numerics.reconstruction == "variational" &&
                        owner->settings.numerics.vrBoundaryValue == "physicalMirrored";
        if (polynomial && owner->reconstruction)
            pin->basis.emplace(owner->reconstruction->AcquireBasisRead());
        if (owner->equation.HasRANS())
        {
            pin->ransMean = state.LocateBlock("rans");
            pin->walls = owner->equation.RANS().ComponentNumber() == 2;
        }
        std::vector<R::ArrayOperand<real, false, R::UniformTableRows>> inputs;
        std::vector<R::ArrayOperand<real, true, R::UniformTableRows>> outputs;
        for (index iField = 0; iField < pin->fieldNumber; ++iField)
            inputs.push_back(R::ReadAs<R::UniformTableRows>(state.PhysicalField(iField).Owned()));
        if (pin->basis)
        {
            pin->coefficientFirst = index(inputs.size());
            for (index iField = 0; iField < pin->fieldNumber; ++iField)
                inputs.push_back(R::ReadAs<R::UniformTableRows>(Coefficients().PhysicalField(iField).Owned()));
        }
        const auto addPacket = [&](OptionalRef<const R::Array<real>> packet)
        {
            if (!packet)
                return index(-1);
            const index result = index(inputs.size());
            inputs.push_back(R::ReadAs<R::UniformTableRows>(*packet));
            return result;
        };
        pin->incomingInput = addPacket(incoming);
        pin->centerInput = addPacket(incomingCenters);
        for (index iField = 0; iField < pin->fieldNumber; ++iField)
            outputs.push_back(R::WriteAs<R::UniformTableRows>(boundary.PhysicalField(iField).Owned()));
        for (auto &field : fields)
        {
            if (!field.gradientBoundary)
                pin->gradientOutputs.push_back(-1);
            else if (field.gradientBoundary->SharedStorage() == field.boundaryValues.SharedStorage())
                pin->gradientOutputs.push_back(index(pin->gradientOutputs.size())); // same center read role
            else
            {
                pin->gradientOutputs.push_back(index(outputs.size()));
                outputs.push_back(R::WriteAs<R::UniformTableRows>(*field.gradientBoundary));
            }
        }
        if (primitiveBoundary)
        {
            pin->primitiveOutput = index(outputs.size());
            outputs.push_back(R::WriteAs<R::UniformTableRows>(*primitiveBoundary));
        }
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            pin->means.push_back(state.LocateBlock(iBlock));
            pin->targets.push_back(boundary.LocateBlock(iBlock));
            if (pin->basis)
                pin->coefficients.push_back(Coefficients().LocateBlock(iBlock));
        }
        for (const auto &output : outputs)
            for (const auto &input : inputs)
                DNDS_check_throw_info(!output.storage->Values().Overlaps(input.storage->Values()),
                                      "boundary output aliases a mutable input");
        R::SubmitArrayBatch(engine, stream, [pin, time](R::ExecutionContext, auto inputs, auto outputs, auto wallMean)
                            {
            // Mutable chemistry contexts are reserved at execution, one per
            // active row worker. The controller retains immutable geometry.
            auto evaluator = pin->evaluator;
            std::vector<CPU::BoundaryTraceWorkerScratch> workers;
            if (!evaluator)
                workers = MakeCPUPointWorkers<CPU::BoundaryTraceWorkerScratch>(pin->owner->equation, pin->workers,
                    pin->properties, pin->componentNumber);
            const auto block = [&](EquationBlockLocation location, index first)
            {
                return EquationBlockReadView{{inputs[first + location.physicalField], location.firstComponent, location.componentNumber},
                                             {{}, location.firstComponent, location.componentNumber}};
            };
            std::vector<EquationBlockReadView> means, coefficients;
            std::vector<EquationBlockArrayView<real>> targets;
            for (const auto &location : pin->means) means.push_back(block(location, 0));
            for (const auto &location : pin->coefficients) coefficients.push_back(block(location, pin->coefficientFirst));
            for (const auto &location : pin->targets)
                targets.push_back({outputs[location.physicalField], location.firstComponent, location.componentNumber});
            CPU::BoundaryTraceRequest request;
            request.stencil = pin->stencil.View();
            request.faces = pin->faces.View<R::FixedRows<1>>();
            request.means = {means.data(), index(means.size())};
            request.coefficients = {coefficients.data(), index(coefficients.size())};
            request.output = {targets.data(), index(targets.size())};
            request.logicalFirst = pin->logicalFirst.HostView();
            request.componentNumber = pin->componentNumber;
            request.polynomial = bool(pin->basis);
            request.allowPointFallback = pin->owner->settings.numerics.positivity || pin->owner->settings.numerics.faceFallback;
            request.thermal = pin->thermal.View();
            if (pin->basis) request.basis = pin->basis->EvaluationView();
            if (pin->ransMean)
            {
                const auto &rans = pin->owner->equation.RANS();
                request.rans = {rans.Kind(), rans.ComponentNumber()};
                request.ransMeans = block(*pin->ransMean, 0);
            }
            if (pin->walls)
                request.wallDistance.cellMean = {wallMean.data, wallMean.offsets, wallMean.rows, wallMean.width,
                                                wallMean.stride, wallMean.sizes, wallMean.kind, wallMean.device};
            request.frame = pin->owner->motion.At(time);
            request.ransFirst = pin->ransFirst;
            request.modes = pin->owner->reconstruction ? pin->owner->reconstruction->NRecDOF() : 0;
            request.time = time;
            request.centers = !pin->owner->stencil.quadratureBoundaryValues;
            if (pin->incomingInput >= 0) request.incoming = inputs[pin->incomingInput];
            const SpanView<const BoundaryCondition> conditions{pin->owner->boundaryConditions.data(), index(pin->owner->boundaryConditions.size())};
            const auto fill = [&]
            {
                const auto status = evaluator
                    ? CPU::FillBoundaryTraces(request, conditions, pin->owner->boundaryPoint, *evaluator)
                    : CPU::FillBoundaryTraces(request, conditions, pin->owner->boundaryPoint,
                        SpanView<CPU::BoundaryTraceWorkerScratch>{workers.data(), index(workers.size())});
                RequireNumericalSuccess(status, "queued boundary traces");
            };
            request.stateKind = pin->mirrored ? BoundaryPointStateKind::ExteriorState : BoundaryPointStateKind::FaceTrace;
            fill();
            const bool gradient = std::any_of(pin->gradientOutputs.begin(), pin->gradientOutputs.end(), [&](index i) { return i >= pin->fieldNumber; });
            if (gradient)
            {
                // A center trace is an independent evaluation of point physics,
                // not a quadrature-row copy or an effective mirrored cell.
                targets.clear();
                for (const auto &location : pin->targets)
                {
                    const index output = pin->gradientOutputs[location.physicalField];
                    DNDS_assert(output >= 0);
                    targets.push_back({outputs[output], location.firstComponent, location.componentNumber});
                }
                request.output = {targets.data(), index(targets.size())};
                request.centers = true;
                request.stateKind = BoundaryPointStateKind::FaceTrace;
                request.incoming = pin->centerInput >= 0 ? InterfaceBoundaryRows(inputs[pin->centerInput]) : InterfaceBoundaryRows{};
                fill();
            }
            if (pin->primitiveOutput >= 0)
            {
                CPU::PrimitiveBoundaryRequest primitive;
                primitive.stencil = request.stencil;
                primitive.faces = request.faces;
                primitive.means = request.means;
                primitive.componentNumber = request.componentNumber;
                primitive.output = outputs[pin->primitiveOutput];
                primitive.gas = pin->owner->equation.Gas();
                primitive.frame = request.frame;
                primitive.incoming = pin->centerInput >= 0 ? InterfaceBoundaryRows(inputs[pin->centerInput]) : InterfaceBoundaryRows{};
                primitive.flowFirst = pin->flowFirst;
                primitive.time = time;
                RequireNumericalSuccess(CPU::FillPrimitiveBoundary(primitive, conditions), "queued primitive center boundary");
            } }, std::move(inputs), std::move(outputs), std::make_tuple(R::ReadAs<R::RowLayout>(pin->walls ? owner->wallDistance->CellMeans() : owner->deviceEmpty)));
        EndQueuedPhase(engine, stream, timing);
    }
}

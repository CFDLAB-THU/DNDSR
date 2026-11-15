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
        OptionalRef<const R::Array<real>> incoming)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() &&
                                  state.Device() == owner->metrics.Device() && std::isfinite(time),
                              "boundary submission backend, stream or time differs");
        state.RequireSchema(owner->equation.Schema());
        DNDS_check_throw_info(state.Space().Same(owner->metrics.Space()) && state.PhysicalFieldNumber() == index(fields.size()),
                              "boundary submission row space or grouping differs");
        if (owner->metrics.Device() != R::Device::Host())
        {
            SubmitDeviceBoundary(engine, stream, state, polynomial, time, incoming);
            return;
        }
        const auto timing = BeginQueuedPhase(engine, stream, BoundaryPhase);
        // Array-backed geometry and fields use scheduled operands alike. A
        // queued VR reader cannot block this producer by reserving BC reads
        // before the CPU worker has reached the producer's execution callback.
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            ThermalPolicy thermal;
            R::MetadataBuffer<index> logicalFirst;
            Geom::MeshReadAccess geometry;
            bool basis = false, walls = false;
            std::vector<EquationBlockLocation> means, coefficients, targets;
            std::optional<EquationBlockLocation> ransMean;
            std::vector<index> gradientOutputs;
            index fieldNumber = 0, componentNumber = 0;
            index primitiveOutput = -1, incomingInput = -1, coefficientFirst = -1;
            index ransFirst = -1, flowFirst = -1;
            bool mirrored = false;
            std::shared_ptr<PointEvaluator> evaluator;
            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
            Physics::Thermodynamics::MixturePropertyRequest properties;

            Inputs(std::shared_ptr<const Data> data, const ThermalPolicy &policy,
                   R::MetadataBuffer<index> offsets, std::shared_ptr<PointEvaluator> pointEvaluator)
                : owner(std::move(data)), thermal(policy), logicalFirst(std::move(offsets)),
                  geometry(owner->metrics.AcquireMeshRead(owner->equation.HasRANS() && owner->equation.RANS().ComponentNumber() == 2)),
                  evaluator(std::move(pointEvaluator)) {}
        };
        auto pin = std::make_shared<Inputs>(owner, thermal, logicalFirst, std::move(evaluator));
        pin->workers = workers;
        const bool viscous = !meanOnly && owner->equation.Viscous();
        pin->properties = {viscous || owner->equation.HasRANS(), viscous, viscous, chemical, false};
        pin->fieldNumber = state.PhysicalFieldNumber();
        pin->componentNumber = state.ComponentNumber();
        pin->ransFirst = ransFirst;
        pin->flowFirst = flowFirst;
        pin->mirrored = owner->settings.numerics.reconstruction == "variational" &&
                        owner->settings.numerics.vrBoundaryValue == "physicalMirrored";
        if (polynomial && owner->settings.numerics.reconstruction == "variational")
            pin->basis = true;
        if (owner->equation.HasRANS())
        {
            pin->ransMean = state.LocateBlock("rans");
            if (owner->equation.RANS().ComponentNumber() == 2)
                pin->walls = true;
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
        if (incoming)
        {
            pin->incomingInput = index(inputs.size());
            inputs.push_back(R::ReadAs<R::UniformTableRows>(*incoming));
        }
        for (index iField = 0; iField < pin->fieldNumber; ++iField)
            outputs.push_back(R::WriteAs<R::UniformTableRows>(boundary.PhysicalField(iField).Owned()));
        for (auto &field : fields)
        {
            pin->gradientOutputs.push_back(field.gradientBoundary ? index(outputs.size()) : -1);
            if (field.gradientBoundary)
                outputs.push_back(R::WriteAs<R::UniformTableRows>(*field.gradientBoundary));
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
        R::SubmitArrayBatch(engine, stream, [pin, time](R::ExecutionContext, auto inputs, auto outputs, auto points, auto rows, auto cells, auto faceOffsets, auto quadrature, auto radial, auto faces, auto cached, auto moments, auto wallMean)
                            {
            // Acquire scarce chemistry workers only when this operation runs.
            // Earlier queued auxiliary refreshes may still own the pool slot
            // while the controller is submitting this boundary producer.
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
            request.points = points;
            request.means = {means.data(), index(means.size())};
            request.coefficients = {coefficients.data(), index(coefficients.size())};
            request.output = {targets.data(), index(targets.size())};
            request.logicalFirst = pin->logicalFirst.HostView();
            request.componentNumber = pin->componentNumber;
            request.polynomial = bool(pin->basis);
            request.allowPointFallback = pin->owner->settings.numerics.positivity || pin->owner->settings.numerics.faceFallback;
            request.thermal = pin->thermal.View();
            if (pin->basis)
            {
                const auto &settings = pin->owner->reconstruction->Settings();
                request.basis = {cached, moments, {cells, faceOffsets, quadrature, radial, faces},
                                 pin->geometry.HostView(), pin->owner->metrics.Dimension(), settings.maxOrder,
                                 pin->owner->reconstruction->NRecDOF(),
                                 settings.cacheDiffBase ? int(cached.width / pin->owner->reconstruction->NRecDOF()) : 0};
            }
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
            if (pin->incomingInput >= 0) request.incoming = inputs[pin->incomingInput];
            request.ransFirst = pin->ransFirst;
            request.modes = pin->owner->reconstruction ? pin->owner->reconstruction->NRecDOF() : 0;
            request.time = time;
            const SpanView<const BoundaryCondition> conditions{pin->owner->boundaryConditions.data(), index(pin->owner->boundaryConditions.size())};
            const bool gradient = std::any_of(pin->gradientOutputs.begin(), pin->gradientOutputs.end(), [](index i) { return i >= 0; });
            const auto fill = [&](BoundaryPointStateKind kind)
            {
                request.stateKind = kind;
                const auto status = evaluator
                    ? CPU::FillBoundaryTraces(request, conditions, pin->owner->boundaryPoint, *evaluator)
                    : CPU::FillBoundaryTraces(request, conditions, pin->owner->boundaryPoint,
                                             SpanView<CPU::BoundaryTraceWorkerScratch>{workers.data(), index(workers.size())});
                RequireNumericalSuccess(status, "queued FillBoundaryTraces");
            };
            fill(pin->mirrored && !gradient ? BoundaryPointStateKind::ExteriorState : BoundaryPointStateKind::FaceTrace);
            for (index iField = 0; iField < pin->fieldNumber; ++iField)
                if (const auto target = pin->gradientOutputs[iField]; target >= 0)
                    CPU::CopyBoundaryRows(outputs[iField].ConstView(), outputs[target], rows);
            if (pin->mirrored && gradient) fill(BoundaryPointStateKind::ExteriorState);
            if (pin->primitiveOutput >= 0)
            {
                CPU::PrimitiveBoundaryRequest primitive;
                primitive.points = request.points;
                primitive.sourceRows = rows;
                primitive.means = request.means;
                primitive.componentNumber = request.componentNumber;
                primitive.output = outputs[pin->primitiveOutput];
                primitive.gas = pin->owner->equation.Gas();
                primitive.frame = request.frame;
                primitive.incoming = request.incoming;
                primitive.flowFirst = pin->flowFirst;
                primitive.time = time;
                RequireNumericalSuccess(CPU::FillPrimitiveBoundary(primitive, conditions), "queued primitive boundary evaluation");
            } }, std::move(inputs), std::move(outputs), std::make_tuple(R::ReadAs<R::FixedRows<1>>(owner->boundaryTraceGeometry), R::ReadAs<R::FixedRows<1>>(owner->gradientBoundaryRows), R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics()), R::ReadAs<R::FixedRows<1>>(owner->metrics.FaceOffsets()), R::ReadAs<R::FixedRows<7>>(owner->metrics.FaceQuadrature()), R::ReadAs<R::FixedRows<3>>(owner->metrics.RadialMoments()), R::ReadAs<R::FixedRows<7>>(owner->metrics.FaceMetrics()), R::ReadAs<R::TableRows>(pin->basis ? owner->reconstruction->SamplingFaceBasis() : owner->deviceEmpty), R::ReadAs<R::TableRows>(pin->basis ? owner->reconstruction->SamplingMoments() : owner->deviceEmpty), R::ReadAs<R::RowLayout>(pin->walls ? owner->wallDistance->CellMeans() : owner->deviceEmpty)));
        EndQueuedPhase(engine, stream, timing);
    }
}

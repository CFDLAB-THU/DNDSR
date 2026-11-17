#include "Cauchy/Implicit/FiniteVolumeDeviceJacobian.hpp"
#include "Cauchy/Implicit/CUDA/ACMJacobianKernels.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void FiniteVolume::PrepareDeviceJacobian(const EquationDofs &state, const AuxiliaryValues &auxiliary,
                                             const R::Array<real> &inversePseudoTime, LinearProblemContext &output,
                                             JacobianPreparation controls, real time, const JacobianContext &context) const
    {
        const TimelineScope timeline("Cauchy/Linear/PrepareJacobian");
#ifdef DNDS_USE_CUDA
        auto &work = context.workspace.get();
        ValidateWorkspace(state, work);
        output.Validate();
        const index n = state.ComponentNumber(), owned = Metrics().NumCell(), all = Metrics().NumCellProc();
        const auto device = Device();
        const bool acm = _data->equation.IsACM();
        DNDS_check_throw_info(context.stream && context.stream->Context().device == device &&
                                  device.space == R::MemorySpace::CUDA && state.Device() == device &&
                                  state.PhysicalFieldNumber() == 1 && output.Device() == device &&
                                  output.Space().Same(state.Space()) && output.Components() == n,
                              "device Jacobian requires one matching combined-AoS state, matrix and execution lane");
        const bool full = output.DiagonalMode() == DiagonalStorage::Block;
        DNDS_check_throw_info(acm || !controls.roeDissipation || (n == 5 && full),
                              "Roe Jacobian requires the flow5 full-block profile");
        DNDS_check_throw_info(std::isfinite(time) && std::isfinite(controls.physicalMass) && controls.physicalMass >= 0 &&
                                  std::isfinite(controls.residualScale) && controls.residualScale >= 0,
                              "invalid device Jacobian time or coefficients");
        DNDS_check_throw_info(_data->settings.equation.source == "none" && auxiliary.faceSpectralRadius,
                              "device Jacobian requires prepared signals and no chemical source");
        const auto checkTable = [&](const R::Array<real> &array, index rows, index width)
        {
            DNDS_check_throw_info(array.GetDevice() == device && array.Layout().Kind() == R::RowLayoutKind::Table &&
                                      array.RowCount() == rows && array.Layout().Width() == width,
                                  "device Jacobian input placement or AoS shape differs");
        };
        const auto &field = state.CoupledField();
        output.ConfigureInputLayout(_data->settings.execution.kernels.stateInputLayout == "aosoa32"
                                        ? LinearInputLayout::Tiled32
                                        : LinearInputLayout::Native);
        checkTable(field.Owned(), owned, n);
        checkTable(field.Ghost(), all - owned, n);
        checkTable(inversePseudoTime, owned, 1);
        checkTable(*auxiliary.faceSpectralRadius, _data->stencil.stencilFaces.RowCount(), 2);
        output.ValidateColumns(_data->jacobianColumns);
        const auto checkAlias = [&](const R::Array<real> &input)
        {
            DNDS_check_throw_info(!input.Storage().Overlaps(output.Diagonal().Storage()) &&
                                      (!output.HasAssembledOffDiagonal() || !input.Storage().Overlaps(output.OffDiagonal().Values().Storage())),
                                  "device Jacobian output aliases an input");
        };
        checkAlias(field.Owned());
        checkAlias(field.Ghost());
        checkAlias(inversePseudoTime);
        checkAlias(*auxiliary.faceSpectralRadius);
        if (acm)
        {
            DNDS_check_throw_info(full && output.HasAssembledOffDiagonal() && n >= 4 && n <= 6 &&
                                      !HasInterfaces() && !_data->motion.At(time).moving,
                                  "CUDA ACM Jacobian requires stationary single-domain full blocks");
            auto session = work.session.Acquire();
            const auto &lane = *context.stream;
            auto engine = lane.LockEngine();
            if (!work.deviceJacobian)
                work.deviceJacobian = std::make_shared<Workspace::DeviceJacobianData>(_data, n, lane.Context());
            const auto scratch = work.deviceJacobian;
            auto physics = _data->equation.ACMParameters();
            physics.roe = _data->settings.numerics.flux == "Roe";
            const auto rans = _data->equation.HasRANS() ? _data->equation.RANS().KernelParameters() : Physics::RANS::PointParameters{};
            std::array<bool, 6> active{};
            const Algebra::ComponentSelection selection(n, controls.updateComponents);
            for (index j = 0; j < n; ++j)
                active[j] = selection.Includes(j);
            R::SubmitArrays(engine, lane, [](auto where, auto result)
                            { CUDA::ClearNumericalStatus(where, result); }, R::WriteAs<R::FixedRows<1>>(scratch->result));
            auto blocks = output.OffDiagonal().Values();
            R::SubmitArrays(engine, lane, [scratch, physics, rans, active, scale = controls.residualScale, mass = controls.physicalMass, rowMajor = output.RowMajorBlocks()](auto where, auto owned, auto ghost, auto tau, auto radius, auto offsets, auto diagonal, auto blocks, auto status)
                            {
                CUDA::ACMJacobianInput q{};
                q.input = {owned,ghost};
                q.stencil = scratch->stencil.KernelDescriptor();
                const auto bc = scratch->owner->deviceACMBoundaryConditions.View(where.device);
                q.conditions.rows = {bc.data(),nullptr,index(bc.size()),1,1,nullptr,R::RowLayoutKind::Table,where.device};
                q.physics = physics;
                q.tau = tau;
                q.radius = radius;
                if (scratch->walls) q.wallMean = scratch->walls->View().cellMean;
                if (scratch->acmLengths) q.lengths = scratch->acmLengths->View<R::FixedRows<1>>();
                q.rans = rans;
                q.diagonal = diagonal;
                q.offDiagonal = {blocks,offsets,owned.Width(),rowMajor};
                q.mass = mass;
                q.residualScale = scale;
                q.active = active;
                CUDA::PrepareACMJacobian(where,q,status); }, R::ReadAs<R::TableRows>(field.Owned()), R::ReadAs<R::TableRows>(field.Ghost()), R::ReadAs<R::TableRows>(inversePseudoTime), R::ReadAs<R::TableRows>(*auxiliary.faceSpectralRadius), R::ReadAs<R::FixedRows<1>>(output.OffDiagonal().OffsetArray()), R::WriteAs<R::UniformTableRows>(output.Diagonal()), R::WriteAs<R::UniformTableRows>(blocks), R::WriteAs<R::FixedRows<1>>(scratch->rowsStatus));
            scratch->AppendStatus(engine, lane, scratch->rowsStatus);
            engine.Synchronize();
            const auto result = scratch->result.To(R::Device::Host(), lane.Context());
            RequireNumericalSuccess(result.Read().View()(0, 0), "CUDA ACM Jacobian preparation");
            return;
        }
        const auto physics = PerfectGasSpatialParameters::FromEquation(_data->equation, Metrics().Dimension());
        physics.Validate(int(n));
        const bool wave = controls.roeDissipation && _data->settings.numerics.riemann.useStencil && _data->flux.UsesTransverseSignal();
        const bool pressure = controls.roeDissipation && _data->settings.numerics.riemann.useStencil && _data->flux.UsesPressureStencil();
        DNDS_check_throw_info((!wave || auxiliary.cellWaveJump) && (!pressure || auxiliary.cellPressureMinimum),
                              "Roe Jacobian requires the configured flux stencil signals");
        for (const auto *signal : {wave ? &*auxiliary.cellWaveJump : nullptr, pressure ? &*auxiliary.cellPressureMinimum : nullptr})
            if (signal)
            {
                DNDS_check_throw_info(signal->Space().Same(state.Space()), "Jacobian signal row space differs");
                checkTable(signal->Owned(), owned, 1);
                checkTable(signal->Ghost(), all - owned, 1);
                checkAlias(signal->Owned());
                checkAlias(signal->Ghost());
            }
        if (_data->equation.HasRANS())
            ValidateEquationSourceJacobian(state, context, time);
        auto workspaceSession = work.session.Acquire();
        std::optional<R::SessionAccess::Lease> reconstructionSession;
        std::optional<FinalizedReconstruction> finalized;
        std::optional<Workspace::ActiveReconstruction> active;
        if (context.reconstruction)
        {
            DNDS_check_throw_info(context.reconstruction->owner == _data, "foreign Jacobian reconstruction");
            reconstructionSession.emplace(context.reconstruction->session.Acquire());
            if (_data->reconstruction)
                finalized.emplace(AcquireFinalizedReconstruction(*context.reconstruction));
            active.emplace(work.BindReconstruction(*context.reconstruction));
        }
        const auto boundary = PreparedBoundaryInputs(work);
        DNDS_check_throw_info(!HasInterfaces() || boundary, "device interface Jacobian requires its centered boundary publication");
        if (boundary)
            ValidateBoundaryInputs(*boundary, time);
        const auto &incoming = boundary ? boundary->FaceCenterValues().CoupledField().Owned() : _data->deviceEmpty;
        checkAlias(incoming);
        auto engine = context.stream->LockEngine();
        const auto &stream = *context.stream;
        if (!work.deviceJacobian)
            work.deviceJacobian = std::make_shared<Workspace::DeviceJacobianData>(_data, n, stream.Context());
        const auto scratch = work.deviceJacobian;
        const R::Array<real> *inputOwned = &field.Owned(), *inputGhost = &field.Ghost();
        if (scratch->inputOwned)
        {
            scratch->inputOwned->Refresh(engine, stream, *inputOwned);
            scratch->inputGhost->Refresh(engine, stream, *inputGhost);
            inputOwned = &scratch->inputOwned->Values();
            inputGhost = &scratch->inputGhost->Values();
        }
        const auto slot = scratch->Acquire(controls.roeDissipation);
        auto &values = slot->values;
        const auto &cells = work.DeviceCellSelection(CFV::ReconstructionRegion::All);
        CUDA::FrozenJacobianParameters parameters;
        parameters.frame = _data->motion.At(time).KernelParameters();
        parameters.gas = physics.gas;
        parameters.components = int(n);
        parameters.kineticComponent = n == 7 ? 5 : -1;
        parameters.scale = controls.residualScale;
        parameters.roe = controls.roeDissipation;
        const Algebra::ComponentSelection selection(n, controls.updateComponents);
        for (index j = 0; j < n; ++j)
            parameters.active[j] = selection.Includes(j);

        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto result)
                        { CUDA::ClearNumericalStatus(context, result); }, R::WriteAs<R::FixedRows<1>>(scratch->result));
        R::SubmitArrays(engine, stream, [gas = physics.gas](R::ExecutionContext context, auto owned, auto ghost, auto means, auto p, auto dp, auto status)
                        { CUDA::FreezeJacobianMeans(context, gas, owned, ghost, means, p, dp, status); }, R::ReadAs<R::UniformTableRows>(*inputOwned), R::ReadAs<R::UniformTableRows>(*inputGhost), R::WriteAs<R::UniformTableRows>(values.means), R::WriteAs<R::TableRows>(values.pressure), R::WriteAs<R::TableRows>(values.pressureDerivative), R::WriteAs<R::FixedRows<1>>(scratch->meansStatus));
        scratch->AppendStatus(engine, stream, scratch->meansStatus);
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto input, auto output)
                        { Algebra::ApplyRowViews(context, Algebra::PointwiseOperation::Copy, output.DynamicView(), input.DynamicView()); }, R::ReadAs<R::TableRows>(*auxiliary.faceSpectralRadius), R::WriteAs<R::TableRows>(values.radius));
        R::SubmitArrays(engine, stream, [scratch, parameters, controls, full, rowMajorBlocks = output.RowMajorBlocks()](R::ExecutionContext context, auto tau, auto diagonal, auto status, auto cells)
                        { CUDA::InitializeJacobianDiagonal(context, scratch->metric.View().cells, tau, diagonal, status, cells,
                                                           parameters.frame, controls.physicalMass, parameters.scale, parameters.components, full, rowMajorBlocks); }, R::ReadAs<R::TableRows>(inversePseudoTime), R::WriteAs<R::UniformTableRows>(output.Diagonal()), R::WriteAs<R::FixedRows<1>>(scratch->rowsStatus), R::ReadAs<R::FixedRows<1>>(cells));
        scratch->AppendStatus(engine, stream, scratch->rowsStatus);
        scratch->SubmitSource(work, engine, stream, state, output.Diagonal(), full, parameters.scale);

        const auto freezeSignal = [&](const auto &input, R::Array<real> &target, const char *name)
        {
            if (target.GetDevice() != device || target.RowCount() != all || target.Layout().Width() != 1)
            {
                target = R::Array<real>(R::RowLayout::Dense(all, 1), device);
                target.SetName(name, DNDS_ALLOCATION_SITE());
            }
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto owned, auto ghost, auto output)
                            { CUDA::FreezeJacobianSignal(context, owned, ghost, output); }, R::ReadAs<R::TableRows>(input.Owned()), R::ReadAs<R::TableRows>(input.Ghost()), R::WriteAs<R::TableRows>(target));
        };
        if (wave)
            freezeSignal(*auxiliary.cellWaveJump, scratch->waveJump, "Cauchy/Jacobian/waveJump");
        if (pressure)
            freezeSignal(*auxiliary.cellPressureMinimum, scratch->pressureMinimum, "Cauchy/Jacobian/pressureMinimum");
        R::SubmitArrays(engine, stream, [scratch, parameters, physics, controls, full, time, rowMajorBlocks = output.RowMajorBlocks()](R::ExecutionContext context, auto means, auto p, auto dp, auto radius, auto incoming, auto wave, auto minimum, auto cells, auto diagonal, auto magnitudes, auto status)
                        {
            const auto conditions = scratch->conditions.View<PerfectGasBoundaryParameters>();
            CUDA::JacobianFaceInput request;
            request.stencil = scratch->stencil.KernelDescriptor();
            request.ownedCells = scratch->owner->metrics.NumCell();
            request.incidences = scratch->owner->stencil.IncidenceNumber();
            request.conditions = {conditions.data, nullptr, index(conditions.size), 1, 1, nullptr, R::RowLayoutKind::Table, context.device};
            request.means = means; request.pressure = p; request.pressureDerivative = dp; request.radius = radius;
            request.incoming = incoming; request.waveJump = wave; request.pressureMinimum = minimum;
            if (scratch->walls)
            {
                const auto mean = scratch->walls->View().cellMean;
                request.wallMean = {mean.data, mean.offsets, mean.rows, mean.width, mean.stride, mean.sizes, mean.kind, mean.device};
            }
            request.parameters = parameters; request.rans = physics.rans.kind;
            request.solver = scratch->owner->flux.Kind(); request.fluxSettings = scratch->owner->settings.numerics.riemann.KernelSettings();
            request.rowMajorBlocks = rowMajorBlocks;
            request.time = time; request.physicalBoundary = controls.scalarBoundaryDerivative;
            CUDA::AddJacobianFaces(context, request, cells, diagonal, full, magnitudes, status); }, R::ReadAs<R::UniformTableRows>(values.means), R::ReadAs<R::TableRows>(values.pressure), R::ReadAs<R::TableRows>(values.pressureDerivative), R::ReadAs<R::TableRows>(values.radius), R::ReadAs<R::TableRows>(incoming), R::ReadAs<R::TableRows>(wave ? scratch->waveJump : _data->deviceEmpty), R::ReadAs<R::TableRows>(pressure ? scratch->pressureMinimum : _data->deviceEmpty), R::ReadAs<R::FixedRows<1>>(cells), R::ReadWriteAs<R::UniformTableRows>(output.Diagonal()), R::WriteAs<R::TableRows>(values.magnitudes), R::ReadWriteAs<R::FixedRows<1>>(scratch->rowsStatus));
        scratch->AppendStatus(engine, stream, scratch->rowsStatus);
        if (output.HasAssembledOffDiagonal())
        {
            // Array handles share storage; submission owns the write reservation.
            auto blocks = output.OffDiagonal().Values();
            R::SubmitArrays(engine, stream, [scratch, parameters, rowMajorBlocks = output.RowMajorBlocks()](R::ExecutionContext context, auto means, auto p, auto dp, auto radius, auto magnitudes, auto cells, auto blocks, auto offsets, auto status)
                            {
                const CUDA::FrozenJacobianInput request{scratch->geometryRead->View(), means, p, dp, radius, magnitudes, parameters};
                CUDA::AssembleFrozenJacobian(context, request, cells, {blocks, offsets, parameters.components, rowMajorBlocks}, status); }, R::ReadAs<R::UniformTableRows>(values.means), R::ReadAs<R::TableRows>(values.pressure), R::ReadAs<R::TableRows>(values.pressureDerivative), R::ReadAs<R::TableRows>(values.radius), R::ReadAs<R::TableRows>(values.magnitudes), R::ReadAs<R::FixedRows<1>>(cells), R::WriteAs<R::UniformTableRows>(blocks), R::ReadAs<R::FixedRows<1>>(output.OffDiagonal().OffsetArray()), R::WriteAs<R::FixedRows<1>>(scratch->assemblyStatus));
            scratch->AppendStatus(engine, stream, scratch->assemblyStatus);
        }
        // Explicit publication boundary, not a per-kernel wait. The CPU path
        // is synchronous too. A future queued publication can retain the same
        // operation records and defer this status decision to the linear solve.
        engine.Synchronize();
        const auto result = scratch->result.To(R::Device::Host(), stream.Context());
        RequireNumericalSuccess(result.Read().View()(0, 0), "device Jacobian preparation");
        if (!output.HasAssembledOffDiagonal())
        {
            auto publication = std::make_shared<FrozenJacobianOperator>(output.Space().Pin(), scratch->publicationGeometry,
                                                                        values, parameters, stream.Context());
            output.PublishOffDiagonalEvaluator(publication);
            slot->publication = publication;
        }
#else
        (void)state;
        (void)auxiliary;
        (void)inversePseudoTime;
        (void)output;
        (void)controls;
        (void)time;
        (void)context;
        DNDS_check_throw_info(false, "CUDA Jacobian preparation is not compiled");
#endif
    }
}

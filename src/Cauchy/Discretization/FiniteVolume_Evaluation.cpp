#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/FiniteVolumeAuxiliaryInternal.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;
    // Entry validation must allow ordered pending producers. Access reservations
    // belong to queued operators/transfers; immediate read/write probes would
    // incorrectly reject valid work on the same compute lane. A non-mutating
    // reservation preflight may reject external immediate leases before MPI.
    void FiniteVolume::PrepareCorrectionAux(EquationDofs &state, ReconstructionState &reconstruction,
                                            Workspace &work, AuxiliaryValues &auxiliary,
                                            Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const
    {
        PrepareCorrectionAux(state, reconstruction, work, auxiliary, engine,
                             stream, stream, time);
    }
    void FiniteVolume::PrepareCorrectionAux(EquationDofs &state, ReconstructionState &reconstruction,
                                            Workspace &work, AuxiliaryValues &auxiliary,
                                            Runtime::EagerEngine &engine,
                                            const Runtime::ExecutionStream &stream,
                                            const Runtime::ExecutionStream &communication,
                                            real time,
                                            OptionalRef<const QueuedEvaluationDependency> dependency) const
    {
        {
            auto reconstructionSession = reconstruction.session.Acquire();
            auto workspaceSession = work.session.Acquire();
            reconstruction.binding.Check(state);
            const auto local = [&](auto &&action)
            {
                Metrics().Communication()->LocalStage(std::forward<decltype(action)>(action));
            };
            std::optional<AuxiliaryPreparation> preparation;
            std::optional<BoundaryInputs> boundary;
            const bool gradientAuxiliary = _data->equation.HasRANS() &&
                                           _data->equation.RANS().ComponentNumber() == 2;
            local([&]
                  {
            ValidateWorkspace(state, work);
            DNDS_check_throw_info(engine.Owns(stream) && engine.Owns(communication) &&
                                      stream.Context().device == Device() && communication.Context().device == Device(),
                                  "equation correction stream or placement differs");
            AuxiliaryInputs inputs;
            inputs.deferBoundary = bool(dependency);
            if (dependency)
            {
                DNDS_check_throw_info(dependency->incoming, "correction interface refresh has no stable receiver buffers");
                boundary = *dependency->incoming;
                ValidateBoundaryLayout(*boundary);
            }
            if (work.incoming && !dependency)
            {
                boundary = *work.incoming;
                inputs.boundary.Emplace(*boundary);
            }
            if (!gradientAuxiliary)
                preparation.emplace(BeginAuxValues(state, auxiliary, time, work.workers, inputs)); });
            if (gradientAuxiliary)
                EnsureMeanGradientWorkspace(state, work);
            OperatorSchedule schedule(engine, stream, communication,
                                      _data->settings.execution.overlap);
            std::vector<Algebra::DofHalo<>::Operation> transfers;
            transfers.reserve(reconstruction.meanHalo.size());
            ++reconstruction.meanEpoch;
            for (auto &halo : reconstruction.meanHalo)
                transfers.push_back(halo.BeginAfter(reconstruction.meanEpoch, engine, communication, stream.Record(),
                                                    _data->settings.CheckHaloFreshness()));
            if (preparation && _data->settings.execution.overlap)
            {
                if (dependency)
                    preparation->ApplyInterfaceIndependent(engine, stream);
                else
                    preparation->Apply(engine, stream, CFV::ReconstructionRegion::Core);
            }
            if (dependency)
            {
                schedule.PublishAfter(dependency->completion, [&]
                                      {
                dependency->publish();
                ValidateBoundaryPublication(*boundary, time); });
                if (preparation)
                {
                    preparation->BindBoundary(*boundary);
                    if (_data->settings.execution.overlap)
                        preparation->Apply(engine, stream, CFV::ReconstructionRegion::Core);
                }
            }
            for (auto &transfer : transfers)
                schedule.AwaitPublished(std::move(transfer));
            transfers.clear();
            const auto rest = schedule.DependentRegion();
            local([&]
                  {
            if (gradientAuxiliary)
            {
                AuxiliaryInputs inputs;
                if (boundary) inputs.boundary.Emplace(*boundary);
                inputs.deferBoundary = bool(dependency);
                preparation.emplace(QueueMeanGradientAuxValues(state, auxiliary, *work.meanGradients,
                    engine, stream, communication, time, work.workers, inputs, false));
            }
            else preparation->Apply(engine, stream, rest);
            schedule.Join();
            if (gradientAuxiliary)
            {
                FinishMeanGradientAuxValues(state, auxiliary, *work.meanGradients, *preparation);
                work.centerGradientInput.emplace(state, boundary, time);
            }
            else preparation->Finish(auxiliary);
                if (_data->settings.CheckHaloFreshness())
                    for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                        reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch);

            work.PublishQueuedPhases(); });
        }
        EnsureCenterSourceGradient(state, reconstruction, work, engine, stream, communication, time);
    }
    void FiniteVolume::EnsureCenterSourceGradient(EquationDofs &state, ReconstructionState &reconstruction, Workspace &work,
                                                  R::EagerEngine &engine, const R::ExecutionStream &compute,
                                                  const R::ExecutionStream &communication, real time) const
    {
        if (!_data->equation.HasRANS() || _data->settings.implicit.sourceJacobianMode != "center")
            return;
        if (work.centerGradientInput && work.centerGradientInput->Current(state, work.incoming, time))
            return;
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        const auto context = Metrics().Communication();
        OptionalRef<const R::Array<real>> incoming;
        context->LocalStage([&]
                            {
            ValidateWorkspace(state, work);
            reconstruction.binding.Check(state);
            DNDS_check_throw_info(!HasInterfaces() || work.incoming, "centered gradient needs a current interface packet");
            if (work.incoming)
            {
                ValidateBoundaryInputs(*work.incoming, time);
                incoming.Emplace(work.incoming->FaceCenterValues().CoupledField().Owned());
            }
            if (_data->settings.CheckHaloFreshness())
                for (index field = 0; field < state.PhysicalFieldNumber(); ++field)
                    reconstruction.meanHalo[field].RequireCurrent(state.PhysicalField(field), reconstruction.meanEpoch); });
        EnsureMeanGradientWorkspace(state, work);
        auto &gradients = *work.meanGradients;
        OperatorSchedule schedule(engine, compute, communication, false);
        SubmitMeanGradientBoundary(engine, compute, state, gradients, time, work.workers, incoming);
        SubmitMeanGradientRows(engine, compute, state, gradients, CFV::ReconstructionRegion::All);
        context->LocalStage([&]
                            {
            schedule.Join();
            CheckMeanGradientBoundary(gradients);
            work.centerGradientInput.emplace(state, work.incoming, time); });
    }
    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, ReconstructionState &reconstruction, Workspace &work,
                                Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time, OptionalRef<R::Array<real>> sourceDerivative) const
    {
        EvaluateImpl(state, OptionalRef<EquationDofs>(residual), reconstruction,
                     work, engine, stream, stream, time, {}, {}, sourceDerivative);
    }
    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, ReconstructionState &reconstruction, Workspace &work,
                                AuxiliaryValues &auxiliary, Runtime::EagerEngine &engine,
                                const Runtime::ExecutionStream &stream, real time, OptionalRef<R::Array<real>> sourceDerivative) const
    {
        EvaluateImpl(state, OptionalRef<EquationDofs>(residual), reconstruction,
                     work, engine, stream, stream, time,
                     OptionalRef<AuxiliaryValues>(auxiliary), {}, sourceDerivative);
    }
    void FiniteVolume::PrepareReconstruction(EquationDofs &state, ReconstructionState &reconstruction, Workspace &work, Runtime::EagerEngine &engine,
                                             const Runtime::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, {}, reconstruction, work, engine, stream, stream,
                     time);
    }
    void FiniteVolume::RefreshSamplingInputs(EquationDofs &state, ReconstructionState &reconstruction, Workspace &work, Runtime::EagerEngine &engine,
                                             const Runtime::ExecutionStream &stream,
                                             const Runtime::ExecutionStream &communication) const
    {
        ValidateWorkspace(state, work);
        auto reconstructionSession = reconstruction.session.Acquire();
        auto session = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction);
        Metrics().Communication()->LocalStage([&]
                                              { DNDS_check_throw_info(engine.Owns(stream) && engine.Owns(communication) &&
                                                                          stream.Context().device == Device() && communication.Context().device == Device(),
                                                                      "sampling refresh stream or placement differs"); });
        ++reconstruction.meanEpoch;
        std::vector<Algebra::DofHalo<>::Operation> means;
        means.reserve(reconstruction.meanHalo.size());
        OperatorSchedule schedule(engine, stream, communication, true);
        for (auto &halo : reconstruction.meanHalo)
            means.push_back(halo.BeginAfter(reconstruction.meanEpoch, engine,
                                            communication, stream.Record(), _data->settings.CheckHaloFreshness()));
        // Modal coefficients are distinct from means whenever this list is
        // nonempty. Post all independent payloads before the final join;
        // gradient/P0 representations have no modal-coefficient exchange.
        auto coefficients = schedule.BeginAfterCompute(reconstruction.FinalizedHalo());
        for (auto &operation : means)
            schedule.AwaitPublished(std::move(operation));
        means.clear();
        schedule.Await(coefficients);
        // This public refresh promises completed sampling inputs. Its transfer
        // segment uses the same ordering protocol on every backend.
        schedule.Join();
        if (_data->settings.CheckHaloFreshness())
            Metrics().Communication()->LocalStage([&]
                                                  {
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch); });
    }
    void FiniteVolume::Execute(EquationDofs &state,
                               ReconstructionState &reconstruction,
                               Workspace &work,
                               DomainExecutionContext &context,
                               FiniteVolumeEvaluation evaluation) const
    {
        auto &engine = context.Engine();
        const auto &stream = context.ComputeStream();
        const auto &communication = context.CommunicationStream();
        switch (evaluation.mode)
        {
        case FiniteVolumeEvaluationMode::SpatialResidual:
            DNDS_check_throw_info(evaluation.residual,
                                  "spatial FV evaluation requires a residual output");
            {
                auto dependency = context.AcquireEvaluationDependency();
                if (evaluation.auxiliary)
                    EvaluateImpl(state, OptionalRef<EquationDofs>(*evaluation.residual),
                                 reconstruction, work, engine, stream, communication,
                                 evaluation.time,
                                 OptionalRef<AuxiliaryValues>(*evaluation.auxiliary),
                                 dependency ? OptionalRef<const QueuedEvaluationDependency>(*dependency)
                                            : OptionalRef<const QueuedEvaluationDependency>{},
                                 evaluation.sourceDerivative);
                else
                    EvaluateImpl(state, OptionalRef<EquationDofs>(*evaluation.residual),
                                 reconstruction, work, engine, stream, communication,
                                 evaluation.time, {},
                                 dependency ? OptionalRef<const QueuedEvaluationDependency>(*dependency)
                                            : OptionalRef<const QueuedEvaluationDependency>{},
                                 evaluation.sourceDerivative);
                return;
            }
        case FiniteVolumeEvaluationMode::ReconstructionOnly:
            DNDS_check_throw_info(!evaluation.residual && !evaluation.auxiliary && !evaluation.sourceDerivative,
                                  "reconstruction-only FV evaluation accepts no spatial outputs");
            EvaluateImpl(state, {}, reconstruction, work, engine, stream,
                         communication, evaluation.time);
            return;
        case FiniteVolumeEvaluationMode::SamplingInputs:
            DNDS_check_throw_info(!evaluation.residual && !evaluation.auxiliary && !evaluation.sourceDerivative,
                                  "sampling-input FV evaluation accepts no spatial outputs");
            RefreshSamplingInputs(state, reconstruction, work, engine, stream,
                                  communication);
            return;
        }
        DNDS_check_throw_info(false, "unknown finite-volume evaluation mode");
    }
    void FiniteVolume::EvaluateMean(EquationDofs &state, EquationDofs &residual, MeanWorkspace &work,
                                    Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, OptionalRef<EquationDofs>(residual),
                     *work.reconstruction, *work.work, engine, stream, stream,
                     time);
    }
    void FiniteVolume::EvaluateImpl(EquationDofs &state,
                                    OptionalRef<EquationDofs> residual,
                                    ReconstructionState &reconstruction, Workspace &work, Runtime::EagerEngine &engine,
                                    const Runtime::ExecutionStream &stream,
                                    const Runtime::ExecutionStream &communication, real time,
                                    OptionalRef<AuxiliaryValues> auxiliary,
                                    OptionalRef<const QueuedEvaluationDependency> dependency, OptionalRef<R::Array<real>> sourceDerivative) const
    {
        const TimelineScope timeline("Cauchy/FV/Evaluate");
        if (!auxiliary && work.rhsAuxiliary)
            auxiliary.Emplace(*work.rhsAuxiliary);
        const bool gradientAuxiliary = auxiliary && _data->equation.HasRANS() &&
                                       _data->equation.RANS().ComponentNumber() == 2;
        const bool centerSourceGradient = auxiliary && _data->equation.HasRANS() &&
                                          _data->settings.implicit.sourceJacobianMode == "center";
        const bool ransP1Overlay = !work.meanOnly && _data->settings.numerics.ransP1Overlay;
        auto reconstructionSession = reconstruction.session.Acquire();
        auto session = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction);
        const Workspace::PhaseTime timing(work, Workspace::TotalPhase);
        work.centerGradientInput.reset();
        work.rhsState.reset();
        work.rhsCoefficients.reset();
        work.relativeUpdate = !work.meanOnly && _data->settings.numerics.reconstruction == "variational" ? std::numeric_limits<real>::infinity() : 0;
        work.faceFallbacks = {};
        work.diagnostics.Reset();
        work.characteristicFallbacks = 0;
        const MPIInfo mpi(Metrics().Communicator());
        // Repeated entry/CPU-join agreement follows the same checked/eager
        // policy as later local kernels. Unchecked execution still validates
        // locally and propagates failures to the fail-fast frontend. Payload
        // transfer preflight remains mandatory and unchanged.
        const auto local = [&](auto &&action)
        {
            Metrics().Communication()->LocalStage(std::forward<decltype(action)>(action));
        };
        DNDS_check_throw_info(engine.Owns(stream) &&
                                  engine.Owns(communication),
                              "FV execution lanes belong to another runtime");
        std::optional<AuxiliaryPreparation> auxiliaryPreparation;
        std::optional<BoundaryInputs> boundary;
        OptionalRef<const R::Array<real>> incoming, incomingCenters;
        std::vector<std::shared_ptr<Workspace::RHSSubmission>> rhsSubmissions;
        std::shared_ptr<Workspace::CharacteristicSubmission> characteristicSubmission;
        std::shared_ptr<Workspace::GradientPositivitySubmission> gradientPositivitySubmission;
        const auto finishAuxiliary = [&]
        {
            if (auxiliaryPreparation)
            {
                if (gradientAuxiliary)
                    FinishMeanGradientAuxValues(state, *auxiliary, *work.meanGradients, *auxiliaryPreparation);
                else
                    auxiliaryPreparation->Finish(*auxiliary);
                auxiliaryPreparation.reset();
            }
        };
        local([&]
              {
            DNDS_check_throw_info(work.owner == _data && reconstruction.owner == _data &&
                                      std::isfinite(time) &&
                                      engine.Owns(stream) && engine.Owns(communication) &&
                                      stream.Context().device == Device() &&
                                      communication.Context().device == Device(),
                                  "foreign equation workspace/state, invalid time or stream placement");
            ValidateWorkspace(state, work);
            reconstruction.binding.Check(state);
            if (residual)
            {
                DNDS_check_throw_info(work.meanOnly || !_data->settings.numerics.ransCellCenterViscosity ||
                                          (auxiliary && auxiliary->cellTurbulentViscosity),
                                      "RANS face viscosity requires the prepared cell turbulent viscosity Array");
                auto &output = *residual;
                output.RequireSchema(_data->equation.Schema());
                state.RequireCompatible(output);
                EquationDofOperations::RequireDistinct(state, output);
                if (reconstruction.coefficients)
                {
                    EquationDofOperations::RequireDistinct(reconstruction.WarmValues(), output);
                    EquationDofOperations::RequireDistinct(reconstruction.FinalizedValues(), output);
                }
                // Reject an external output reader on every checked rank before
                // posting any halo or mutating an earlier physical field. Late
                // RHS reservation alone is insufficient: a UT/library caller
                // could catch on one rank while peers enter the final reduction.
                // This is a dry run, not an immediate write lease or GPU fence.
                for (index iField = 0; iField < output.PhysicalFieldNumber(); ++iField)
                    output.PhysicalField(iField).Owned().Storage().ValidateReservation(
                        R::AccessMode::Write, Device(), engine.ChecksAccess());
            }
            if (sourceDerivative)
            {
                const auto &record = *sourceDerivative;
                const index n = state.ComponentNumber();
                DNDS_check_throw_info(residual && _data->quadrature.source && _data->settings.implicit.sourceJacobianMode == "rhsQuadrature" &&
                                          record.GetDevice() == Device() && record.RowCount() == Metrics().NumCell() &&
                                          record.Layout().Kind() == R::RowLayoutKind::Table && record.Layout().Width() == n * n,
                                      "invalid RHS source derivative destination");
                const auto distinct = [&](const EquationDofs &fields)
                {
                    for (index field = 0; field < fields.PhysicalFieldNumber(); ++field)
                    {
                        const auto &values = fields.PhysicalField(field);
                        DNDS_check_throw_info(!record.Storage().Overlaps(values.Owned().Storage()) &&
                                              !record.Storage().Overlaps(values.Ghost().Storage()),
                                              "source derivative aliases an evaluation field");
                    }
                };
                distinct(state);
                distinct(*residual);
                if (reconstruction.coefficients)
                {
                    distinct(reconstruction.WarmValues());
                    distinct(reconstruction.FinalizedValues());
                }
                record.Storage().ValidateReservation(R::AccessMode::Write, Device(), engine.ChecksAccess());
            }
            DNDS_check_throw_info(!HasInterfaces() || work.incoming || dependency,
                                  "FV evaluation needs a ready interface packet");
            if (dependency)
            {
                DNDS_check_throw_info(dependency->incoming,
                                      "queued interface evaluation requires stable receiver buffers");
                boundary = *dependency->incoming;
                ValidateBoundaryLayout(*boundary);
            }
            else if (work.incoming)
            {
                boundary = *work.incoming;
                ValidateBoundaryInputs(*boundary, time);
            }
            if (boundary)
            {
                incomingCenters.Emplace(boundary->FaceCenterValues().CoupledField().Owned());
                incoming.Emplace(_data->stencil.quadratureBoundaryValues
                    ? boundary->QuadratureValues().CoupledField().Owned() : *incomingCenters);
            }
            _data->partition.Validate();
            work.boundaryMomentumCurrent = work.boundaryFacialFluxCurrent = false; });
        if (gradientAuxiliary || centerSourceGradient || ransP1Overlay)
            EnsureMeanGradientWorkspace(state, work);
        const bool needsAuxiliaryPreparation = auxiliary && !gradientAuxiliary;
        if (needsAuxiliaryPreparation)
            local([&]
                  {
                      AuxiliaryInputs inputs;
                      inputs.deferBoundary = bool(dependency);
                      if (work.incoming && !dependency)
                          inputs.boundary.Emplace(*work.incoming);
                      auxiliaryPreparation.emplace(BeginAuxValues(state, *auxiliary, time, work.workers, inputs)); });
        const bool overlap = _data->settings.execution.overlap;
        const auto sweep = [&](CFV::ReconstructionRegion region)
        {
            const TimelineScope vrTimeline("Cauchy/CFV/Reconstruction/Submit");
            const auto timing = work.BeginQueuedPhase(engine, stream, Workspace::VRPhase);
            if (_data->reconstructionAMG)
            {
                DNDS_check_throw_info(region == CFV::ReconstructionRegion::All,
                                      "AMG reconstruction is an unsplit single-device operator");
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                {
                    auto &field = work.fields[iField];
                    DNDS_check_throw_info(field.reconstruction && field.reconstructionBoundary && field.reconstructionAMG,
                                          "AMG reconstruction workspace is incomplete");
                    _data->reconstructionAMG->SubmitDefectCorrection(
                        engine, stream, state.PhysicalField(iField), reconstruction.WarmValues().PhysicalField(iField),
                        *field.reconstructionBoundary, field.reconstruction->temporary, *field.reconstructionAMG);
                }
                work.EndQueuedPhase(engine, stream, timing);
                return;
            }
            if (work.limitedVR)
            {
                auto &field = work.fields.front();
                DNDS_check_throw_info(_data->reconstruction && field.reconstruction &&
                                          field.reconstructionBoundary,
                                      "limited VR schedule lacks polynomial workspace");
                _data->reconstruction->SubmitReconstructionIterLimitedTo(
                    engine, stream, state.CoupledField(), reconstruction.WarmValues().CoupledField(),
                    *field.reconstruction, *field.reconstructionBoundary, work.limitedVR->targetCoefficients,
                    work.limitedVR->alpha, region, _data->reconstructionKernel);
                work.EndQueuedPhase(engine, stream, timing);
                return;
            }
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                auto &field = work.fields[iField];
                DNDS_check_throw_info(_data->reconstruction && field.reconstruction &&
                                          field.reconstructionBoundary,
                                      "VR schedule lacks polynomial workspace");
                _data->reconstruction->SubmitReconstructionIterTo(
                    engine, stream, state.PhysicalField(iField), reconstruction.WarmValues().PhysicalField(iField),
                    *field.reconstruction, *field.reconstructionBoundary, region, _data->reconstructionKernel);
            }
            work.EndQueuedPhase(engine, stream, timing);
        };
        const auto gradient = [&](CFV::ReconstructionRegion region)
        {
            const auto timing = work.BeginQueuedPhase(engine, stream, Workspace::GradientPhase);
            if (work.meanGradients && !gradientAuxiliary)
                SubmitMeanGradientRows(engine, stream, state, *work.meanGradients, region);
            work.SubmitPrimitiveGradients(engine, stream, state, region);
            work.EndQueuedPhase(engine, stream, timing);
        };
        OperatorSchedule schedule(engine, stream, communication, overlap);
        const auto rest = schedule.DependentRegion();
        work.BeginNumericalStatus(engine, stream, bool(residual));
        // Payload operations retain their own inputs; queued callbacks only
        // borrow objects declared before schedule (or capture values explicitly).
        std::vector<Algebra::DofHalo<>::Operation> meanOperations;
        meanOperations.reserve(reconstruction.meanHalo.size());
        ++reconstruction.meanEpoch;
        for (auto &halo : reconstruction.meanHalo)
            meanOperations.push_back(halo.BeginAfter(
                reconstruction.meanEpoch, engine, communication, stream.Record(),
                _data->settings.CheckHaloFreshness()));
        std::vector<R::TransferOperation> firstCoefficients;
        const auto &numerics = _data->settings.numerics;
        const bool variational = !work.meanOnly && numerics.reconstruction == "variational";
        const bool firstSweep = overlap && variational && !_data->reconstructionAMG && !work.limitedVR && numerics.reconstructionIterations > 0;
        if (firstSweep)
            firstCoefficients = schedule.BeginAfterCompute(reconstruction.coefficientHalo);
        // Post both transfers before queueing core work. A CPU-visible MPI
        // bridge may otherwise join that work while checking producer readiness.
        if (auxiliaryPreparation && overlap)
        {
            if (dependency)
                auxiliaryPreparation->ApplyInterfaceIndependent(engine, stream);
            else
                auxiliaryPreparation->Apply(engine, stream, CFV::ReconstructionRegion::Core);
        }
        if (!dependency)
            work.SubmitBoundary(engine, stream, state, variational && numerics.reconstructionBoundaryFromPolynomial,
                                time, {}, incoming, incomingCenters);
        // P0 RANS sources still need physical gradients during a mean-only
        // timestep forecast; disabling reconstruction must not skip this pass.
        const bool directGradientWork = (!work.meanOnly && numerics.reconstruction == "gradient") ||
                                        (_data->equation.HasRANS() &&
                                         (numerics.reconstruction == "constant" ||
                                          work.meanOnly));
        // The unlimited mean gradient has one producer. Direct P1, LVR, SST
        // auxiliary properties and the centered source Jacobian share it.
        const bool separateMeanGradient = !gradientAuxiliary &&
                                          (directGradientWork || centerSourceGradient || ransP1Overlay || bool(work.limitedVR));
        const bool gradientWork = separateMeanGradient || bool(work.primitiveGradient);
        const auto gradientBoundary = [&]
        {
            if (separateMeanGradient)
                SubmitMeanGradientBoundary(engine, stream, state, *work.meanGradients, time, work.workers, incomingCenters);
        };
        if (!dependency)
            gradientBoundary();
        if (gradientWork && overlap && !dependency)
            gradient(CFV::ReconstructionRegion::Core);
        if (firstSweep && !dependency)
            sweep(CFV::ReconstructionRegion::Core);
        if (dependency)
        {
            schedule.PublishAfter(dependency->completion, [&]
                                  {
                dependency->publish();
                ValidateBoundaryPublication(*boundary, time); });
            if (needsAuxiliaryPreparation)
            {
                // Binding declares known allocations now; the one-round slot
                // becomes published on the ordered lane before numerical use.
                auxiliaryPreparation->BindBoundary(*boundary);
                if (overlap)
                    auxiliaryPreparation->Apply(engine, stream, CFV::ReconstructionRegion::Core);
            }
            work.SubmitBoundary(engine, stream, state, variational && numerics.reconstructionBoundaryFromPolynomial,
                                time, {}, incoming, incomingCenters);
            gradientBoundary();
            // MPI-local core does not imply interface independence.
            if (gradientWork && overlap)
                gradient(CFV::ReconstructionRegion::Core);
            if (firstSweep)
                sweep(CFV::ReconstructionRegion::Core);
        }
        for (auto &operation : meanOperations)
            schedule.AwaitPublished(std::move(operation));
        meanOperations.clear();
        if (needsAuxiliaryPreparation)
            auxiliaryPreparation->Apply(engine, stream, rest);
        else if (gradientAuxiliary)
        {
            // Two-equation transport explicitly needs mean gradients; those
            // are independent of the reconstruction being computed below.
            // Reuse the already ordered mean halo. The private session queues
            // boundary/gradient/gradient-halo/auxiliary work; publication waits
            // for this enclosing schedule's final completion, on every backend.
            AuxiliaryInputs inputs;
            if (boundary)
                inputs.boundary.Emplace(*boundary);
            inputs.deferBoundary = bool(dependency);
            auxiliaryPreparation.emplace(QueueMeanGradientAuxValues(state, *auxiliary, *work.meanGradients,
                                                                    engine, stream, communication, time, work.workers, inputs, false));
        }
        if (gradientWork)
            gradient(rest);
        if (directGradientWork)
            for (index iField = 0; iField < index(work.fields.size()); ++iField)
            {
                auto &field = work.fields[iField];
                if (!work.meanOnly && numerics.reconstruction == "gradient" && !work.nativeDirectGradient)
                {
                    DNDS_check_throw_info(_data->reconstruction,
                                          "gradient-to-modal conversion requires a polynomial plan");
                    _data->gradient.ConvertToCoefficients(engine, stream, *field.gradient, *_data->reconstruction,
                                                          reconstruction.WarmValues().PhysicalField(iField));
                }
                else if (_data->reconstruction)
                    Algebra::Fill(engine, stream, reconstruction.WarmValues().PhysicalField(iField), 0.);
            }
        std::optional<R::TransferOperation> primitiveGradientOperation;
        if (work.primitiveGradientHalo)
            primitiveGradientOperation.emplace(schedule.BeginAfterCompute(*work.primitiveGradientHalo));
        if (work.limitedVR)
        {
            if (work.flowCharacteristic)
                characteristicSubmission = work.SubmitCharacteristics(engine, stream, state, OptionalRef<const AuxiliaryValues>(auxiliary));
            DNDS_check_throw_info(reconstruction.limitedVRHistory, "LVR reconstruction state has no history");
            work.limitedVR->Build(*_data, state.CoupledField(), *work.fields.front().gradient, *reconstruction.limitedVRHistory,
                                  work.flowCharacteristic ? CFV::FaceCharacteristicTransformRef(std::cref(*work.flowCharacteristic))
                                                          : std::nullopt,
                                  engine, stream, communication);
        }
        if (variational)
        {
            for (int iteration = 0; iteration < numerics.reconstructionIterations; ++iteration)
            {
                std::vector<R::TransferOperation> operations;
                if (iteration == 0 && firstSweep)
                {
                    operations = std::move(firstCoefficients);
                }
                else
                {
                    if (numerics.reconstructionBoundaryFromPolynomial)
                        work.SubmitBoundary(engine, stream, state, true, time, {}, incoming, incomingCenters);
                    operations = schedule.BeginAfterCompute(reconstruction.coefficientHalo);
                    if (overlap && !_data->reconstructionAMG)
                        sweep(CFV::ReconstructionRegion::Core);
                }
                schedule.Await(operations);
                // Native AMG currently joins at the provider boundary. It is
                // callable in either overlap schedule, but is not split into
                // core/halo and does not promise communication overlap here.
                sweep(_data->reconstructionAMG ? CFV::ReconstructionRegion::All : rest);
                bool converged = false;
                // Explicit host stopping boundary. Fixed-count sweeps do not
                // observe a residual. Device-side conditional iteration / queue
                // mutation is future work, not hidden inside this segment.
                if (numerics.reconstructionTolerance > 0 &&
                    ((iteration + 1) % numerics.reconstructionResidualCheckInterval == 0 ||
                     iteration + 1 == numerics.reconstructionIterations))
                {
                    std::array<real, 2> localNorms{};
                    local([&]
                          {
                        work.SubmitReconstructionNorm(engine, stream, reconstruction.WarmValues());
                        schedule.Join();
                        localNorms = work.ReadReconstructionNorm(); });
                    real norms[2]{localNorms[0], localNorms[1]};
                    MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                        MPI::ContiguousView<real>{norms, 2}, mpi.comm);
                    work.relativeUpdate = std::sqrt(norms[0] / std::max(norms[1], real(1e-300)));
                    converged = work.relativeUpdate <= numerics.reconstructionTolerance;
                }
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                    _data->reconstruction->SubmitCommitReconstruction(engine, stream,
                                                                      reconstruction.WarmValues().PhysicalField(iField),
                                                                      *work.fields[iField].reconstruction);
                // Fixed-count sweeps enqueue commit and the next boundary
                // producer in order. Only requested norm observations above
                // and the portable-MPI readiness bridge need host visibility.
                work.iterations = iteration + 1;
                if (converged)
                    break;
            }
        }
        else if (!directGradientWork && _data->reconstruction)
            local([&]
                  {
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                    Algebra::Fill(engine, stream, reconstruction.WarmValues().PhysicalField(iField), 0.); });
        const bool useFluxScratch = reconstruction.distinctFinalized;
        if (useFluxScratch)
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                Algebra::Copy(engine, stream, reconstruction.MutableFinalized().PhysicalField(iField),
                              reconstruction.WarmValues().PhysicalField(iField));
        if (work.nativeDirectGradient && numerics.limiter != "WBAP" &&
            (numerics.limiter == "Barth" || _data->settings.GradientPositivity()))
            work.SubmitGradientLimiter(engine, stream, state, numerics.limiter == "Barth");
        // Mean-only forecasts do not publish a reconstructed face field, even
        // when the regular operator is configured with gradient/modal WBAP.
        if (!work.meanOnly && numerics.limiter == "WBAP")
        {
            if (work.nativeDirectGradient)
            {
                if (work.gradientFlowCharacteristic)
                    characteristicSubmission = work.SubmitCharacteristics(engine, stream, state, OptionalRef<const AuxiliaryValues>(auxiliary));
                auto gradientOperations = gradientAuxiliary ? std::vector<R::TransferOperation>{} : schedule.BeginAfterCompute(work.directGradientInputHalo);
                const auto limit = [&](CFV::ReconstructionRegion region)
                {
                    for (index iField = 0; iField < index(work.fields.size()); ++iField)
                        _data->gradient.LimitWBAP3(
                            engine, stream, *work.fields[iField].gradient, work.directGradients[iField],
                            *work.fields[iField].gradientExecution,
                            iField == work.flowCharacteristicField && work.gradientFlowCharacteristic
                                ? CFV::GradientFaceCharacteristicTransformRef(
                                      std::cref(*work.gradientFlowCharacteristic))
                                : std::nullopt,
                            region, std::nullopt, 0, numerics.cfv.WBAP_nStd);
                };
                if (overlap)
                    limit(CFV::ReconstructionRegion::Core);
                schedule.Await(gradientOperations);
                limit(rest);
            }
            else
            {
                if (work.flowCharacteristic && !work.limitedVR)
                    characteristicSubmission = work.SubmitCharacteristics(engine, stream, state, OptionalRef<const AuxiliaryValues>(auxiliary));
                for (int order = numerics.cfv.maxOrder; order >= 1; --order)
                {
                    auto limiterOperations = schedule.BeginAfterCompute(reconstruction.FinalizedHalo());
                    const auto limit = [&, order](CFV::ReconstructionRegion region)
                    {
                        for (index iField = 0; iField < index(work.fields.size()); ++iField)
                            _data->reconstruction->SubmitLimiterWBAP_3OrderTo(
                                engine, stream, reconstruction.MutableFinalized().PhysicalField(iField),
                                *work.fields[iField].reconstruction, order,
                                iField == work.flowCharacteristicField && work.flowCharacteristic
                                    ? CFV::FaceCharacteristicTransformRef(std::cref(*work.flowCharacteristic))
                                    : std::nullopt,
                                region);
                    };
                    if (overlap)
                        limit(CFV::ReconstructionRegion::Core);
                    schedule.Await(limiterOperations);
                    limit(rest);
                    for (index iField = 0; iField < index(work.fields.size()); ++iField)
                        _data->reconstruction->SubmitCommitReconstruction(
                            engine, stream, reconstruction.MutableFinalized().PhysicalField(iField),
                            *work.fields[iField].reconstruction);
                    // Commit precedes the next order's producer-prefix
                    // exchange; no host observation is needed between orders.
                }
            }
        }
        const auto positivityTiming = (work.nativeDirectGradient && _data->settings.GradientPositivity()) || work.positivity
                                          ? work.BeginQueuedPhase(engine, stream, Workspace::PPPhase)
                                          : nullptr;
        if (work.nativeDirectGradient && _data->settings.GradientPositivity())
            gradientPositivitySubmission = work.SubmitGradientPositivity(engine, stream, state);
        if (work.positivity)
            local([&]
                  {
                // One controller sequence for inline CPU, worker CPU and CUDA.
                // PP performs point-local recovery inside its queued producer.
                // Only the enclosing reporting drain observes its compact
                // summary; no host decision interrupts local backtracking.
                work.positivity->Apply(engine, stream, state, reconstruction.WarmValues(), reconstruction.MutableFinalized(), false); });
        work.EndQueuedPhase(engine, stream, positivityTiming);
        // Freeze the RANS polynomial before publishing either flux coefficient halo.
        // The unlimited viscous route can read warm coefficients independently.
        if (ransP1Overlay)
        {
            const auto block = state.LocateBlock("rans");
            const bool barth = numerics.UsesRANSGradientBarth();
            if (barth)
                work.SubmitGradientLimiter(engine, stream, state, true);
            const auto &gradient = barth ? work.directGradients[block.physicalField] : work.meanGradients->fields[block.physicalField];
            const auto force = [&](EquationDofs &coefficients)
            {
                _data->gradient.SetLinearCoefficients(engine, stream, gradient, *_data->reconstruction,
                                                      coefficients.PhysicalField(block.physicalField), block.firstComponent, block.componentNumber);
            };
            force(reconstruction.WarmValues());
            if (useFluxScratch)
                force(reconstruction.MutableFinalized());
        }
        std::vector<R::TransferOperation> operations;
        if (!work.nativeDirectGradient)
            operations = schedule.BeginAfterCompute(reconstruction.FinalizedHalo());
        else if (!gradientAuxiliary || numerics.limiter != "none" || _data->settings.GradientPositivity())
            operations = schedule.BeginAfterCompute(work.directGradientHalo);
        std::vector<R::TransferOperation> unlimitedViscousOperations;
        if (useFluxScratch && _data->equation.Viscous() && numerics.viscousGradient == "polynomial" && !numerics.viscousLimited)
            unlimitedViscousOperations = ransP1Overlay ? schedule.BeginAfterCompute(reconstruction.coefficientHalo)
                                                       : schedule.Begin(reconstruction.coefficientHalo);
        auto directViscousOperations = gradientAuxiliary && work.nativeDirectGradient ? std::vector<R::TransferOperation>{} : schedule.Begin(work.directViscousGradientHalo);
        // Publish the direct physical gradients after limiter/PP, once per row.
        // The physical-gradient RHS itself never interprets coefficient data.
        work.SubmitDirectGradients(engine, stream, state, false);
        if (overlap && residual)
            rhsSubmissions.push_back(work.SubmitRHS(engine, stream, state, *residual,
                                                    CFV::ReconstructionRegion::Core, time, {}, OptionalRef<const AuxiliaryValues>(auxiliary), {}, incoming));
        schedule.Await(operations);
        schedule.Await(unlimitedViscousOperations);
        schedule.Await(directViscousOperations);
        if (primitiveGradientOperation)
            schedule.Await(*primitiveGradientOperation);
        work.SubmitDirectGradients(engine, stream, state, true);
        if (residual)
        {
            rhsSubmissions.push_back(work.SubmitRHS(engine, stream, state, *residual,
                                                    rest, time, {}, OptionalRef<const AuxiliaryValues>(auxiliary), {}, incoming));
            work.SubmitVolumeSource(engine, stream, state, *residual, time, {},
                                    work.lowOrderResidual ? OptionalRef<EquationDofs>(*work.lowOrderResidual) : OptionalRef<EquationDofs>{}, sourceDerivative);
        }
        local([&]
              {
            schedule.Join();
            work.CheckNumericalStatus(bool(residual));
            if (separateMeanGradient)
                CheckMeanGradientBoundary(*work.meanGradients);
            if (work.positivity)
                (void)work.positivity->LocalStatistics();
            reconstruction.PublishFinalized();
            if (_data->settings.CheckHaloFreshness())
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch);
            finishAuxiliary();
            if (characteristicSubmission) work.PublishCharacteristics(characteristicSubmission);
            if (gradientPositivitySubmission) work.PublishGradientPositivity(gradientPositivitySubmission);
            work.PublishDirectGradients(state, false, time);
            work.PublishDirectGradients(state, true, time);
            for (const auto &record : rhsSubmissions) work.PublishRHS(record);
            work.PublishQueuedPhases(); });
        work.boundaryMomentumCurrent = bool(residual);
        work.boundaryFacialFluxCurrent = bool(residual) && work.captureBoundaryFlux;
        work.rhsState.emplace(state.NumericAlias());
        if (reconstruction.coefficients)
            work.rhsCoefficients.emplace(reconstruction.FinalizedValues());
        if (separateMeanGradient || gradientAuxiliary)
            work.centerGradientInput.emplace(state, boundary, time);
        work.rhsTime = time;
    }

}

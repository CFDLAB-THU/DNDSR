#include "Cauchy/Discretization/BoundaryFluxInput.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::ValidateSourceDerivative(const R::Array<real> &record, const EquationDofs &state, const EquationDofs &output,
                                                           const ReconstructionState &reconstruction, const R::EagerEngine &engine) const
    {
        const auto device = owner->metrics.Device();
        const index n = state.ComponentNumber();
        DNDS_check_throw_info(owner->quadrature.source && owner->settings.implicit.sourceJacobianMode == "rhsQuadrature" &&
                                  record.GetDevice() == device && record.RowCount() == owner->metrics.NumCell() &&
                                  record.Layout().Kind() == R::RowLayoutKind::Table && record.Layout().Width() == n * n,
                              "invalid RHS source derivative destination");
        const auto distinct = [&](const EquationDofs &fields)
        {
            for (index f = 0; f < fields.PhysicalFieldNumber(); ++f)
            {
                const auto &field = fields.PhysicalField(f);
                DNDS_check_throw_info(!record.Storage().Overlaps(field.Owned().Storage()) && !record.Storage().Overlaps(field.Ghost().Storage()),
                                      "source derivative aliases an evaluation field");
            }
        };
        distinct(state);
        distinct(output);
        if (reconstruction.coefficients)
        {
            distinct(reconstruction.WarmValues());
            distinct(reconstruction.FinalizedValues());
        }
        record.Storage().ValidateReservation(R::AccessMode::Write, device, engine.ChecksAccess());
    }

    R::StreamEvent FiniteVolume::SubmitPreparedResidual(
        const EquationDofs &state, EquationDofs &residual, const std::shared_ptr<ReconstructionState> &reconstruction,
        const std::shared_ptr<Workspace> &work, R::EagerEngine &engine, const R::ExecutionStream &stream, real time,
        OptionalRef<const AuxiliaryValues> auxiliary, OptionalRef<R::Array<real>> sourceDerivative) const
    {
        DNDS_check_throw_info(work && reconstruction && reconstruction->owner == _data && engine.Owns(stream) && stream.Context().device == Device(),
                              "prepared residual workspace, reconstruction or stream differs");
        auto workspaceLease = work->session.Acquire();
        auto reconstructionLease = reconstruction->session.Acquire();
        ValidateWorkspace(state, *work);
        state.RequireCompatible(residual);
        EquationDofOperations::RequireDistinct(state, residual);
        std::optional<FinalizedReconstruction> finalized;
        if (_data->reconstruction)
        {
            finalized.emplace(AcquireFinalizedReconstruction(*reconstruction));
            EquationDofOperations::RequireDistinct(reconstruction->WarmValues(), residual);
            EquationDofOperations::RequireDistinct(finalized->Values(), residual);
        }
        work->CheckPreparedResidual(state, *reconstruction, finalized, time);
        if (!auxiliary && work->rhsAuxiliary)
            auxiliary.Emplace(*work->rhsAuxiliary);
        DNDS_check_throw_info(work->meanOnly || !_data->settings.numerics.ransCellCenterViscosity ||
                                  (auxiliary && auxiliary->cellTurbulentViscosity),
                              "prepared residual is missing the cell turbulent viscosity");
        if (auxiliary)
        {
            ValidateAuxValues(state, *auxiliary, time);
            DNDS_check_throw_info(AuxiliaryBoundaryCurrent(*auxiliary, *work), "prepared flux auxiliary values belong to another receiver packet");
        }
        OptionalRef<const R::Array<real>> incoming;
        if (work->incoming)
        {
            ValidateBoundaryInputs(*work->incoming, time);
            incoming.Emplace(FiniteVolumeDetail::BoundaryFluxInput(_data->stencil, *work->incoming));
        }
        if (sourceDerivative)
            work->ValidateSourceDerivative(*sourceDerivative, state, residual, *reconstruction, engine);
        for (index f = 0; f < residual.PhysicalFieldNumber(); ++f)
            residual.PhysicalField(f).Owned().Storage().ValidateReservation(R::AccessMode::Write, Device(), engine.ChecksAccess());

        struct Submission
        {
            std::shared_ptr<Workspace> work;
            std::shared_ptr<ReconstructionState> reconstruction;
            R::SessionAccess::Lease workspaceLease, reconstructionLease;
            std::shared_ptr<Workspace::RHSSubmission> flux;
            bool success = false;
        };
        auto held = std::make_shared<Submission>(Submission{work, reconstruction, workspaceLease, reconstructionLease, {}});
        auto active = work->BindReconstruction(*reconstruction);
        work->boundaryMomentumCurrent = work->boundaryFacialFluxCurrent = false;
        try
        {
            work->BeginNumericalStatus(engine, stream, true);
            constexpr auto all = CFV::ReconstructionRegion::All;
            held->flux = work->SubmitRHS(engine, stream, state, residual, all, time, {}, auxiliary, {}, incoming);
            work->SubmitVolumeSource(engine, stream, state, residual, time, all, {},
                                     work->lowOrderResidual ? OptionalRef<EquationDofs>(*work->lowOrderResidual) : OptionalRef<EquationDofs>{}, sourceDerivative);
            work->StagePreparedStatus(engine, stream);
            // Retirement distinguishes successful device completion from a rejected
            // or failed prefix. The following cleanup runs on either outcome.
            engine.SubmitOperation(stream, [](R::ExecutionContext) {}, [held](std::exception_ptr failure) noexcept
                                   { held->success = !failure; });
        }
        catch (...)
        {
            // Only the local prefix owns these operands. Retain both sessions
            // through accepted work, then propagate submission failure to the caller.
            engine.SubmitCleanup(stream, [held](R::ExecutionContext)
                                 { held->work->RetirePreparedStatus(false); });
            throw;
        }
        engine.SubmitCleanup(stream, [held](R::ExecutionContext)
                             {
            auto &w = *held->work;
            w.RetirePreparedStatus(held->success);
            if (held->success)
            {
                w.PublishRHS(held->flux);
                w.PublishQueuedPhases();
                w.boundaryMomentumCurrent = true;
                w.boundaryFacialFluxCurrent = w.captureBoundaryFlux;
            } });
        return stream.Record();
    }
}

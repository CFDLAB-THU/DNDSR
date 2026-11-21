#include "Cauchy/Discretization/BoundaryFluxInput.hpp"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    R::Array<real> FiniteVolume::BuildFaceFluxBuffer(index components) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(components > 0, "face flux requires positive component count");
        return R::Array<real>(R::RowLayout::Dense(_data->stencil.stencilFaces.RowCount(), components));
    }
    void FiniteVolume::EvaluateFlux(const EquationDofs &state, EquationDofs &residual,
                                    const ReconstructionState &reconstruction, Workspace &work,
                                    CFV::ReconstructionRegion region, real time,
                                    OptionalRef<R::Array<real>> faceOutput,
                                    OptionalRef<const AuxiliaryValues> auxiliary) const
    {
        ValidateWorkspace(state, work);
        DNDS_check_throw_info(reconstruction.owner == _data, "foreign equation reconstruction state");
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        std::optional<FinalizedReconstruction> finalized;
        if (_data->reconstruction)
            finalized.emplace(AcquireFinalizedReconstruction(reconstruction));
        auto activeReconstruction = work.BindReconstruction(reconstruction);
        state.RequireCompatible(residual);
        EquationDofOperations::RequireDistinct(state, residual);
        if (finalized)
        {
            EquationDofOperations::RequireDistinct(finalized->Values(), residual);
            if (finalized->UsesDistinctStorage())
                EquationDofOperations::RequireDistinct(reconstruction.WarmValues(), residual);
        }
        work.CheckPreparedResidual(state, reconstruction, finalized, time);
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                  region == CFV::ReconstructionRegion::HaloDependent,
                              "unknown flux region");
        if (faceOutput)
        {
            const auto &output = *faceOutput;
            DNDS_check_throw_info(output.GetDevice() == R::Device::Host() && !output.Layout().IsVariable() &&
                                      output.RowCount() == _data->stencil.stencilFaces.RowCount() && output.Layout().Width() == state.ComponentNumber(),
                                  "equation face output shape/device differs");
            for (const auto fields : std::array<OptionalRef<const EquationDofs>, 3>{
                     OptionalRef<const EquationDofs>(state), OptionalRef<const EquationDofs>(residual),
                     finalized ? OptionalRef<const EquationDofs>(finalized->Values()) : OptionalRef<const EquationDofs>{}})
                if (fields)
                    for (index iField = 0; iField < fields->PhysicalFieldNumber(); ++iField)
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(output.Storage(), fields->PhysicalField(iField).Owned().Storage()) &&
                                                  !Algebra::Detail::Overlaps(output.Storage(), fields->PhysicalField(iField).Ghost().Storage()),
                                              "face output aliases equation fields");
        }
        if (!work.meanOnly && _data->settings.numerics.ransCellCenterViscosity)
        {
            if (!auxiliary && work.rhsAuxiliary)
                auxiliary.Emplace(*work.rhsAuxiliary);
            DNDS_check_throw_info(auxiliary && auxiliary->cellTurbulentViscosity,
                                  "RANS face viscosity requires the prepared cell turbulent viscosity Array");
        }
        if (auxiliary)
        {
            ValidateAuxValues(state, *auxiliary, time);
            DNDS_check_throw_info(AuxiliaryBoundaryCurrent(*auxiliary, work), "prepared flux auxiliary values belong to another receiver packet");
        }
        std::optional<R::Array<real>::ReadLease> incomingRead;
        RHSScalarView incoming;
        DNDS_check_throw_info(!HasInterfaces() || work.incoming,
                              "equation flux evaluation needs an incoming interface packet");
        if (work.incoming)
        {
            ValidateBoundaryInputs(*work.incoming, time);
            incomingRead.emplace(
                FiniteVolumeDetail::BoundaryFluxInput(_data->stencil, *work.incoming).Read());
            incoming = incomingRead->View<R::UniformTableRows>();
        }
        auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                            {true, true, true, false, false});
        work.ApplyRHS(state, residual, region, evaluator, time, auxiliary, faceOutput, incoming);
        work.boundaryMomentumCurrent = region == CFV::ReconstructionRegion::All;
        work.boundaryFacialFluxCurrent = work.boundaryMomentumCurrent && work.captureBoundaryFlux && !faceOutput;
    }
}

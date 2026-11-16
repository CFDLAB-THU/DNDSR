#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void FiniteVolume::AddSource(const EquationDofs &state, EquationDofs &residual,
                                 const ReconstructionState &reconstruction, Workspace &work, real time) const
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
            EquationDofOperations::RequireDistinct(finalized->Values(), residual);
        DNDS_check_throw_info(work.rhsState && work.rhsTime == time, "source needs a completed spatial preparation at this time");
        work.rhsState->Check(state, true);
        DNDS_check_throw_info((finalized ? work.rhsCoefficients && work.rhsCoefficients->Current(finalized->Values())
                                         : !work.rhsCoefficients),
                              "source reconstruction changed after spatial preparation");
        auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                            {true, false, false, work.chemical, false});
        work.PrepareVolumeSource(state, std::ref(residual), evaluator, time);
        work.PublishSourceState(state, time);
    }

    void FiniteVolume::ValidateEquationSourceJacobian(
        const EquationDofs &state, const JacobianContext &context, real time) const
    {
        DNDS_check_throw_info(context.reconstruction,
                              "source Jacobian context has no reconstruction publication");
        const auto &reconstruction = *context.reconstruction;
        auto &work = context.workspace.get();
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        std::optional<FinalizedReconstruction> finalized;
        if (_data->reconstruction)
            finalized.emplace(AcquireFinalizedReconstruction(reconstruction));
        auto activeReconstruction = work.BindReconstruction(reconstruction);
        DNDS_check_throw_info(work.owner == _data && (work.chemical || work.ransFirst >= 0), "foreign or source-free reconstruction");
        if (!work.sourceSampling)
            return;
        DNDS_check_throw_info(finalized && work.sourceState && work.sourceCoefficients && work.sourceTime == time, "source quadrature requires a current completed sample publication");
        work.sourceState->Check(state, true);
        DNDS_check_throw_info(work.sourceCoefficients->Current(finalized->Values()),
                              "source reconstruction changed after sample publication");
    }
    bool FiniteVolume::AddEquationSourceJacobian(const EquationDofs &state,
                                                 const JacobianContext &context,
                                                 const R::Array<real>::WriteLease &diagonal, DiagonalStorage storage, real scale, PointEvaluator &evaluator,
                                                 OptionalRef<const EquationFVDetail::MeanLinearizationOutput> linearization, bool rowMajorBlocks) const
    {
        DNDS_check_throw_info(context.reconstruction,
                              "source Jacobian context has no reconstruction publication");
        const auto &reconstruction = *context.reconstruction;
        auto &work = context.workspace.get();
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        std::optional<FinalizedReconstruction> finalized;
        if (_data->reconstruction)
            finalized.emplace(AcquireFinalizedReconstruction(reconstruction));
        auto activeReconstruction = work.BindReconstruction(reconstruction);
        return work.AddMeanSourceJacobian(state, diagonal, storage, scale, evaluator, linearization, rowMajorBlocks);
    }
}

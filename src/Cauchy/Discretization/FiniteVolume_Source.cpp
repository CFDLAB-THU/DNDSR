#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    std::optional<Runtime::Array<real>> FiniteVolume::BuildSourceDerivative(const EquationDofs &state) const
    {
        if (_data->settings.implicit.sourceJacobianMode != "rhsQuadrature" || !_data->quadrature.source)
            return std::nullopt;
        state.RequireSchema(_data->equation.Schema());
        const index n = state.ComponentNumber();
        Runtime::Array<real> record(Runtime::RowLayout::Dense(Metrics().NumCell(), n * n), Device());
        record.SetName("Cauchy/source/integratedDerivative", DNDS_ALLOCATION_SITE());
        return record;
    }

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
    }

    void FiniteVolume::ValidateEquationSourceJacobian(
        const EquationDofs &state, const JacobianContext &context, real time) const
    {
        const auto &work = context.workspace.get();
        DNDS_check_throw_info(work.owner == _data && std::isfinite(time), "invalid source Jacobian context");
        if (_data->settings.implicit.sourceJacobianMode == "rhsQuadrature")
        {
            DNDS_check_throw_info(context.sourceDerivative, "rhsQuadrature requires the derivative produced with this rate");
            const auto &record = *context.sourceDerivative;
            const index n = state.ComponentNumber();
            DNDS_check_throw_info(record.GetDevice() == Device() && record.RowCount() == Metrics().NumCell() &&
                                      record.Layout().Kind() == R::RowLayoutKind::Table && record.Layout().Width() == n * n,
                                  "source derivative record differs from correction shape or device");
        }
        else if (_data->equation.HasRANS())
            DNDS_check_throw_info(work.meanGradients && work.centerGradientInput &&
                                      work.centerGradientInput->Current(state, work.incoming, time),
                                  "centered RANS Jacobian requires this stage's prepared mean gradient");
    }
    bool FiniteVolume::AddEquationSourceJacobian(const EquationDofs &state,
                                                 const JacobianContext &context,
                                                 const R::Array<real>::WriteLease &diagonal, DiagonalStorage storage, real scale, PointEvaluator &evaluator,
                                                 OptionalRef<const EquationFVDetail::MeanLinearizationOutput> linearization, bool rowMajorBlocks) const
    {
        if (_data->settings.implicit.sourceJacobianMode == "rhsQuadrature")
        {
            const auto read = context.sourceDerivative->Read();
            CPU::ApplyVolumeSourceDerivative(read.View<R::TableRows>(), diagonal.View<R::UniformTableRows>(),
                                             state.ComponentNumber(), storage == DiagonalStorage::Block, rowMajorBlocks, scale);
            return false; // The flux assembler prepares its own mean pressure and derivatives.
        }
        auto &work = context.workspace.get();
        auto workspaceSession = work.session.Acquire();
        return work.AddMeanSourceJacobian(state, diagonal, storage, scale, evaluator, linearization, rowMajorBlocks);
    }
}

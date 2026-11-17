#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
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
    void FiniteVolume::InitializeConservative(EquationDofs &state, const std::vector<real> &values) const
    {
        Metrics().Communication()->LocalStage([&]
                                              {
            Metrics().Validate();
            state.RequireSchema(_data->equation.Schema());
            _data->equation.ResolveKernelProfile(state.Device());
            index width = 0;
            for (const auto &block : _data->equation.Schema().blocks) width += index(block.components.size());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && index(values.size()) == width &&
                                      std::all_of(values.begin(), values.end(), [](real v) { return std::isfinite(v); }),
                                  "conservative initializer row space or values differ from equation");
            FlowState flow;
            std::copy_n(values.begin(), 5, flow.begin());
            flow = _data->motion.At(0).LocalState(flow);
            auto output = state.WriteOwned();
            index offset = 0;
            for (const auto &block : _data->equation.Schema().blocks)
            {
                const auto rows = output.Block(block.name);
                CPU::FillConstantRows(rows, block.name == "flow"
                    ? SpanView<const real>{flow.data(), index(flow.size())}
                    : SpanView<const real>{values.data() + offset, rows.ComponentNumber()});
                offset += index(block.components.size());
            } });
    }
    void FiniteVolume::InitializeConstant(EquationDofs &state, const FlowState &primitive,
                                          const std::map<std::string, real> &massFractions, real temperature,
                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                          const std::map<std::string, real> &ransConservative) const
    {
        Metrics().Communication()->LocalStage([&]
                                              {
            Metrics().Validate();
            state.RequireSchema(_data->equation.Schema());
            _data->equation.ResolveKernelProfile(state.Device());
            const auto rans = ResolveRANSValues(_data->equation, ransConservative);
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && _data->equation.HasMixture() == bool(workers),
                                  "equation initialization row space/worker profile differs");
            auto fractions = _data->equation.IndependentMassFractions(massFractions);
            for (auto &value : fractions) value *= primitive[0];
            auto evaluator = _data->equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
            const auto value = _data->motion.At(0).LocalState(evaluator.FromPrimitive(primitive, {fractions.data(), int(fractions.size())}, temperature));
            // Resolve/validate EOS input before obtaining any state writer.
            auto output = state.WriteOwned();
            const auto flow = output.Block("flow");
            CPU::FillConstantRows(flow, {value.data(), index(value.size())});
            if (!fractions.empty())
            {
                const auto species = output.Block("species");
                CPU::FillConstantRows(species, {fractions.data(), index(fractions.size())});
            }
            if (_data->equation.HasRANS())
            {
                const auto scalars = output.Block("rans");
                CPU::FillConstantRows(scalars, {rans.data(), scalars.ComponentNumber()});
            } });
    }

    std::shared_ptr<FiniteVolume::Workspace> FiniteVolume::BuildWorkspace(EquationDofs &state,
                                                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                                                          WorkspaceOptions options) const
    {
        return BuildWorkspaceImpl(state, std::move(workers), false, options);
    }
    std::shared_ptr<FiniteVolume::ReconstructionState> FiniteVolume::BuildReconstructionState(EquationDofs &state) const
    {
        std::shared_ptr<ReconstructionState> result;
        Metrics().Communication()->LocalStage([&]
                                              {
            state.RequireSchema(_data->equation.Schema());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && state.Device() == Device(),
                                  "reconstruction state belongs to another row space or device");
            result = std::make_shared<ReconstructionState>(_data, state); });
        return result;
    }
    std::shared_ptr<FiniteVolume::MeanWorkspace> FiniteVolume::BuildMeanWorkspace(EquationDofs &state,
                                                                                  std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        auto inner = BuildWorkspaceImpl(state, std::move(workers), true);
        auto reconstruction = BuildReconstructionState(state);
        std::shared_ptr<MeanWorkspace> result;
        Metrics().Communication()->LocalStage([&]
                                              { result = std::make_shared<MeanWorkspace>(MeanWorkspace{std::move(reconstruction), std::move(inner)}); });
        return result;
    }
    std::shared_ptr<FiniteVolume::Workspace> FiniteVolume::BuildWorkspaceImpl(EquationDofs &state,
                                                                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                                                              bool meanOnly, WorkspaceOptions options) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        const auto &n = _data->settings.numerics;
        // Resolve an operator-local fixed-width leaf without changing the one
        // EquationDofs workspace or outer schedule. Unsupported layouts and
        // physics retain the generic equation kernel.
        const bool flow5Profile = !_data->equation.HasMixture() && !_data->equation.HasRANS() &&
                                  state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1;
        std::string operationControls;
        Metrics().Communication()->LocalStage([&]
                                              {
            state.RequireSchema(_data->equation.Schema());
            if (state.Device() == R::Device::Host())
                _data->equation.ResolveKernelProfile(state.Device());
            else
            {
                // Placement/workspace construction does not enable the full
                // solver backend. Each queued operator resolves its own leaf;
                // Execute keeps its capability gate until all paths are wired.
                const auto physics = PerfectGasSpatialParameters::FromEquation(_data->equation, Metrics().Dimension());
                physics.Validate(int(state.ComponentNumber()));
                const auto &field = state.CoupledField();
                DNDS_check_throw_info(field.Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                          field.Ghost().Layout().Kind() == R::RowLayoutKind::Table,
                                      "CUDA FV workspace requires combined AoS state");
            }
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && state.Device() == Device(),
                                  "equation FV state belongs to another row space or device");
            DNDS_check_throw_info(options.rhsKernel != RHSKernelPolicy::FixedFlow5 || flow5Profile,
                                  "fixed flow5 RHS kernel requires one perfect-gas flow5 physical field");
            DNDS_check_throw_info((n.reconstruction == "constant" || n.reconstruction == "gradient" || n.reconstruction == "variational") &&
                                  n.reconstructionIterations >= 0 && n.reconstructionResidualCheckInterval > 0,
                                  "invalid equation reconstruction controls");
            DNDS_check_throw_info(!n.reconstructionSolver.UsesAMG() || _data->reconstructionAMG,
                                  "AMG reconstruction requires an explicit CUDA placement and available single-device provider");
            DNDS_check_throw_info(!_data->equation.HasMixture() || _data->fluxPolicy.rusanov,
                "extended EOS or stencil flux needs an explicit preparation provider");
            DNDS_check_throw_info(n.limiter == "none" || n.limiter == "Barth" || n.limiter == "LVR" || n.limiter == "WBAP",
                "requested extended FV flux/PP/limiter profile is not connected yet");
            DNDS_check_throw_info(!n.positivity || n.positivityReconstruction != "Segment" || flow5Profile,
                                  "Segment reconstruction PP is available only for one perfect-gas flow5 field");
            DNDS_check_throw_info(!_data->equation.Viscous() || n.viscousGradient == "polynomial" || flow5Profile,
                                  "extended viscosity requires the conservative polynomial gradient");
            DNDS_check_throw_info(_data->settings.equation.source == "none" ||
                (_data->settings.equation.source == "chemical" && _data->equation.HasMixture()), "unavailable equation volume source");
            _data->boundary.ValidateEquationProducers(_data->boundaryZones);
            if (_data->equation.HasMixture())
            {
                DNDS_check_throw_info(workers, "mixture FV requires explicit chemistry workers");
                auto checked = _data->equation.MakePointEvaluator(workers->Acquire(), {});
            }
            else DNDS_check_throw_info(!workers, "perfect-gas FV does not require chemistry workers");
            operationControls = Json{{"meanOnly", meanOnly}, {"rhsKernel", int(options.rhsKernel)},
                {"numerics", _data->settings.numerics}, {"equation", _data->settings.equation},
                {"motion", _data->settings.motion}, {"boundaries", _data->settings.boundaries}, {"overlap", _data->settings.execution.overlap},
                {"collectiveChecks", _data->settings.implicit.collectiveChecks}, {"properties", _data->equation.PropertyKey().parameters},
                {"rans", _data->equation.HasRANS() ? Json(_data->equation.RANS().Settings()) : Json()}}.dump(); });
        AgreeEquationBinding(state, mpi.comm, operationControls);
        std::shared_ptr<Workspace> result;
        Metrics().Communication()->LocalStage([&]
                                              { result = std::make_shared<Workspace>(_data, state, std::move(workers), meanOnly); });
        result->BindHalos(state);
        if (flow5Profile && state.Device() == R::Device::Host() && options.rhsKernel != RHSKernelPolicy::Generic)
            result->fixedFlowRHS.emplace(AcquireRHSOperator());
        auto request = CompleteRHSAuxiliaryRequest({});
        if (request.meanThermodynamics || request.cellWaveJump || request.cellPressureMinimum)
        {
            request.meanThermodynamics = true;
            result->rhsAuxiliary.emplace(BuildAuxiliaryValues(request));
        }
        if (!meanOnly && _data->equation.HasRANS() && _data->equation.RANS().ComponentNumber() == 2)
            result->meanGradients.emplace(BuildMeanGradients(state));
        return result;
    }
}

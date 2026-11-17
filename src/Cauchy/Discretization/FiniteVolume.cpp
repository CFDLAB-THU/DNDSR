#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FlowErrorOperator.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Input/FlowProfile.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "Cauchy/State/MeanStateCheck.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Core/OMP.hpp"
#include <limits>
#include <climits>
#include <cstdlib>
#include <chrono>
#include <cstdio>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    FiniteVolume::FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation,
                               Diagnostics::SetupProgress progress)
    {
        // Agree the actual selection before optional stages can diverge. The
        // metric plan retains this immutable policy for subsequent operators.
        R::CommunicationDetail::AgreePolicy(mesh->getMPI().comm, settings.implicit.collectiveChecks);
        R::CommunicationDetail::LocalStage(mesh->getMPI().comm, settings.implicit.collectiveChecks, [&]
                                           {
            settings.numerics.ValidatePositivity();
            settings.numerics.reconstructionSolver.Validate();
            DNDS_check_throw_info(!settings.numerics.reconstructionSolver.UsesAMG() ||
                                      (settings.numerics.reconstruction == "variational" && settings.numerics.limiter != "LVR"),
                                  "AMG reconstruction requires variational reconstruction without LVR; LVR-AMG is deferred");
            DNDS_check_throw_info(settings.numerics.vrBoundaryValue == "physicalTrace" || settings.numerics.vrBoundaryValue == "physicalMirrored",
                                  "vrBoundaryValue must be physicalTrace or physicalMirrored");
            DNDS_check_throw_info(settings.initialState.is_null() && settings.referenceState.is_null(),
                                  "resolve tagged physical inputs before constructing FiniteVolume directly");
            if (equation.HasRANS())
                DNDS_check_throw_info(std::isfinite(settings.numerics.wallDistanceFloor) && settings.numerics.wallDistanceFloor > 0 && mesh->HasNodeWallDist(),
                                      "RANS FV requires a published nodal wall field with a positive sampling floor");
            settings.numerics.limitedVR.Validate();
            DNDS_check_throw_info(settings.numerics.limiter != "Barth" ||
                                      settings.numerics.reconstruction == "gradient",
                                  "Barth needs direct gradient reconstruction");
            DNDS_check_throw_info(settings.numerics.limiter != "WBAP" ||
                                      settings.numerics.reconstruction != "constant",
                                  "WBAP needs gradient or variational reconstruction");
            DNDS_check_throw_info(!equation.HasMixture() || settings.numerics.limiter != "LVR",
                                  "mixture LVR policy is not connected yet");
            DNDS_check_throw_info(settings.numerics.limiter != "LVR" || (settings.numerics.reconstruction == "variational" && settings.numerics.cfv.maxOrder >= 2),
                                  "LVR requires degree2/3 variational reconstruction");
            DNDS_check_throw_info(settings.numerics.viscousGradient == "primitive" || settings.numerics.viscousGradient == "polynomial",
                                  "unknown Cauchy viscous gradient scheme"); }, "invalid Cauchy finite-volume setup");
        auto data = std::make_shared<Data>(std::move(mesh), ResolveSweepOrdering(settings), equation, progress);
        if (equation.HasRANS())
        {
            auto setupStage = progress.Stage("FV/RANS-geometry");
            data->metrics.Communication()->LocalStage([&]
                                                      {
                data->wallDistance.emplace(WallDistanceSamples::Construct(data->mesh, settings.numerics.RHSIntegrationOrder(), settings.numerics.wallDistanceFloor));
                data->ransCellLength = data->metrics.BuildCellMaxLenScale(); });
        }
        const bool limitedVR = settings.numerics.limiter == "LVR" && (settings.numerics.limitedVR.alphaMax > 0 || settings.numerics.limitedVR.contactAlphaMax > 0);
        {
            auto setupStage = progress.Stage("FV/stencil");
            data->BuildStencil();
        }
        {
            auto setupStage = progress.Stage("FV/source-quadrature");
            data->BuildSourceQuadrature();
        }
        if (!equation.IsACM())
        {
            auto setupStage = progress.Stage("FV/positivity-quadrature");
            data->BuildPositivityQuadrature(limitedVR);
        }
        data->stencil.Freeze(data->metrics);
        if (limitedVR)
        {
            auto setupStage = progress.Stage("FV/limited-VR");
            data->limitedVRGeometry = std::make_shared<LimitedVRGeometry>(*data);
        }
        _data = std::move(data);
    }
    const CFV::FiniteVolumePlan &FiniteVolume::Metrics() const { return _data->metrics; }
    R::TransferBinding FiniteVolume::BindHalo(CFV::FVField &field) const { return _data->BindHalo(field); }
    const EquationSystem &FiniteVolume::Equation() const { return _data->equation; }
    const DomainSettings &FiniteVolume::Settings() const { return _data->settings; }
    OptionalRef<const CFV::VariationalReconstructionPlan> FiniteVolume::PolynomialReconstruction() const
    {
        return _data->reconstruction ? OptionalRef<const CFV::VariationalReconstructionPlan>(*_data->reconstruction)
                                     : OptionalRef<const CFV::VariationalReconstructionPlan>();
    }
    const Geom::UnstructuredMesh &FiniteVolume::Mesh() const { return *_data->mesh; }
    Geom::MeshReadAccess FiniteVolume::AcquireMeshRead(Geom::MeshReadRequirements requirements) const
    {
        Metrics().Validate();
        return _data->mesh->AcquireRead(requirements);
    }
    const RigidMotion &FiniteVolume::Motion() const { return _data->motion; }

    Json FiniteVolume::ReconstructionMetadata() const
    {
        index components = 0;
        Json blocks = Json::array();
        for (const auto &block : _data->equation.Schema().blocks)
        {
            components += index(block.components.size());
            blocks.push_back({{"name", block.name}, {"components", block.components.size()}});
        }
        return {{"dimension", Metrics().Dimension()},
                {"family", _data->settings.numerics.reconstruction},
                {"coefficient_dofs", _data->reconstruction ? _data->reconstruction->NRecDOF() : 0},
                {"components", components},
                {"blocks", std::move(blocks)},
                {"coefficient_role", "unlimitedGuess"},
                {"cfv", _data->settings.numerics.cfv}};
    }
    void FiniteVolume::Initialize(EquationDofs &state, const FlowProfileSettings &profile, const FlowState &primitive) const
    {
        state.RequireSchema(_data->equation.Schema());
        DNDS_check_throw_info(!_data->equation.HasMixture() && !_data->equation.HasRANS() && state.ComponentNumber() == 5,
                              "analytic flow-profile initialization currently requires a pure flow5 equation");
        Metrics().Communication()->LocalStage([&]
                                              {
            auto meshRead = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = meshRead.HostView();
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow");
            const auto frame = _data->motion.At(0);
            RequireNumericalSuccess(CPU::InitializeFlowProfile(mesh, flow, profile, primitive, _data->equation.Gas(), frame), "flow profile initialization"); });
    }
    std::array<FlowState, 3> FiniteVolume::Error(const EquationDofs &state, const FlowProfileSettings &profile, const FlowState &primitive, real time,
                                                 std::optional<R::ExecutionStream> stream) const
    {
        auto engine = stream ? stream->LockEngine() : R::EagerEngine{};
        const auto lane = stream ? *stream : engine.CreateCPUStream();
        std::optional<FlowErrorOperator> operation;
        Metrics().Communication()->LocalStage([&]
                                              { operation.emplace(*this, lane.Context()); });
        return operation->Evaluate(engine, lane, state, profile, primitive, time);
    }
    AdmissibilityPolicy FiniteVolume::PrepareAdmissibility(
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        DNDS_check_throw_info(
            _data->equation.HasMixture() == bool(workers),
            "equation admissibility needs exactly its mixture worker pool");
        std::optional<AdmissibilityPolicy> result;
        Metrics().Communication()->LocalStage([&]
                                              { result.emplace(
                                                    _data->equation, _data->settings.numerics.densityFloor,
                                                    _data->settings.numerics.temperatureFloor,
                                                    _data->settings.numerics.sensibleEnergyFloor,
                                                    _data->settings.numerics.pressureFloor,
                                                    _data->equation.HasRANS()
                                                        ? ResolveRANSStatePolicy(
                                                              _data->equation,
                                                              _data->settings.numerics.ransState)
                                                        : RANSStateView{},
                                                    workers ? workers->Acquire()
                                                            : Physics::Chemistry::ChemicalWorkerPool::Lease{}); });
        return std::move(*result);
    }
    const ThermalBounds &FiniteVolume::ThermalConstraints() const
    {
        return _data->FixedFlowThermalBounds();
    }
    bool FiniteVolume::Admissible(const EquationDofs &state) const
    {
        DNDS_check_throw_info(
            _data->defaultAdmissibility,
            "mixture finite-volume admissibility needs an explicitly prepared policy");
        return Admissible(state, *_data->defaultAdmissibility);
    }
    bool FiniteVolume::Admissible(
        const EquationDofs &state,
        const AdmissibilityPolicy &policy) const
    {
        state.RequireSchema(_data->equation.Schema());
        DNDS_check_throw_info(
            policy.PropertyKey() == _data->equation.PropertyKey(),
            "admissibility policy belongs to another equation material");
        const auto bounds = policy.View();
        DNDS_check_throw_info(
            bounds.thermal.components == state.ComponentNumber(),
            "admissibility policy and equation state widths differ");
        int valid = 1;
        Metrics().Communication()->LocalStage([&]
                                              {
            DNDS_check_throw_info(state.Device() == Device() && state.Space().Same(Metrics().Space()),
                                  "finite-volume admissibility state placement or row space differs");
            // Synchronous public convenience, sharing the same resident
            // acceptance operator as Domain. Only its scalar status crosses
            // to the controller; no device field is downloaded for CPU physics.
            Runtime::EagerEngine engine;
            Runtime::ExecutionStream stream;
            if (Device() == Runtime::Device::Host()) stream = engine.CreateCPUStream();
#ifdef DNDS_USE_CUDA
            else stream = engine.CreateStream(std::make_shared<Runtime::CUDALauncher>(Device().ordinal));
#else
            else DNDS_check_throw_info(false, "CUDA admissibility is not compiled");
#endif
            MeanStateCheck check(state.Space(), policy, stream.Context());
            valid = bool(check.Evaluate(engine, stream, state)); });
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, Metrics().Communicator());
        return valid;
    }
}

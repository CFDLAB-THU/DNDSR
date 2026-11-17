#include "Cauchy/Core/Simulation.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/MPI/Failure.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

namespace DNDS::Cauchy
{
    namespace
    {
        /// Shared completion/publication gate for one globally ordered
        /// interface value round. Every local domain imports the same event;
        /// the first dependent consumer publishes once, later consumers only
        /// bind their receiver packet.
        class CoordinatedInterfaceRefresh
        {
            std::mutex _mutex;
            InterfaceCoupling::RefreshOperation _operation;
            Runtime::StreamEvent _completion;

        public:
            explicit CoordinatedInterfaceRefresh(
                InterfaceCoupling::RefreshOperation operation)
                : _operation(std::move(operation)),
                  _completion(_operation.CompletionEvent()) {}
            Runtime::StreamEvent CompletionEvent() const
            {
                return _completion;
            }
            void Complete()
            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (!_operation.Finished())
                    _operation.PublishReady();
            }
            /// Controller-side policy boundary, including ranks with no local
            /// receiver domain. Those ranks have no FV consumer to drive this
            /// event, but must retire their accepted interface transfers too.
            void Finish()
            {
                CompletionEvent().WaitHost();
                Complete();
            }
            std::optional<BoundaryInputs> Inputs(index domain)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return _operation.Inputs(domain);
            }
            std::optional<BoundaryInputs> PendingInputs(index domain)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return _operation.PendingInputs(domain);
            }
        };
    }
    StepResult Simulation::Advance(real dt,
                                   std::optional<real> acceptedTime)
    {
        const TimelineScope timeline("Cauchy/Step", std::int64_t(_step));
        real limits[2]{dt, -dt};
        int valid = std::isfinite(dt) && dt > 0;
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _placement.Control());
        DNDS_check_throw_info(valid, "invalid coordinated Cauchy timestep");
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<real>{limits, 2}, _placement.Control());
        DNDS_check_throw_info(limits[0] == -limits[1], "simulation ranks requested different timesteps");
        int targetFlags[2]{int(acceptedTime.has_value()), -int(acceptedTime.has_value())};
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<int>{targetFlags, 2}, _placement.Control());
        DNDS_check_throw_info(targetFlags[0] == -targetFlags[1], "simulation ranks disagree on exact accepted-time snapping");
        real acceptedTarget = acceptedTime.value_or(0);
        if (acceptedTime)
        {
            real targets[2]{acceptedTarget, -acceptedTarget};
            MPI::AllreduceInPlace<MPI::Reduction::Min>(
                MPI::ContiguousView<real>{targets, 2}, _placement.Control());
            DNDS_check_throw_info(targets[0] == -targets[1] && std::isfinite(acceptedTarget) && acceptedTarget > _time,
                                  "simulation ranks requested different or invalid exact accepted times");
            const real expected = _time + dt;
            const real tolerance = 8 * std::numeric_limits<real>::epsilon() *
                                   std::max({real(1), std::abs(expected), std::abs(acceptedTarget)});
            DNDS_check_throw_info(std::abs(expected - acceptedTarget) <= tolerance,
                                  "exact accepted time is inconsistent with the requested timestep");
        }
        // A one-domain simulation is the degenerate coordinated case. Keep one
        // accepted-step transaction and one stage cursor regardless of rank
        // placement; standalone Domain::Advance remains a debugging oracle.
        {
            uint64_t trialIdentifier = 0;
            const auto localPhase = [&](const auto &action, const std::string &name)
            {
                // Numerical rejection is a normal result, already agreed on
                // each domain. Unexpected exceptions abort before this control
                // agreement; they must never masquerade as a rejected step.
                int ready = 1;
                std::string error;
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                    {
                        if constexpr (std::is_same_v<decltype(action(iDomain, *_domains[iDomain])), StepResult>)
                        {
                            auto result = action(iDomain, *_domains[iDomain]);
                            if (!result)
                            {
                                ready = 0;
                                error = std::move(result.reason);
                            }
                        }
                        else
                            action(iDomain, *_domains[iDomain]);
                    }
                ready = MPI::AllreduceValue<MPI::Reduction::Min>(ready, _placement.Control());
                return ready ? StepResult{} : StepResult::Rejected("coordinated " + name + " rejected; accepted states unchanged: " + error);
            };
            const auto ports = [&](bool prepared, int coupledStage)
            {
                std::vector<InterfaceDomainInput> inputs(_domains.size());
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (auto &domain = _domains[iDomain])
                        inputs[iDomain] = {
                            domain->PublishInterfaceDonor(coupledStage, prepared)};
                return inputs;
            };
            const auto refresh = [&](real time, int stage, bool prepared,
                                     bool acceptedSnapshot = false,
                                     int coupledStage = 0,
                                     std::optional<index> nonlinearIteration = {})
            {
                StateEpoch epoch;
                epoch.acceptedStep = _step + uint64_t(acceptedSnapshot);
                epoch.trial = trialIdentifier;
                epoch.stage = stage;
                epoch.coupledStage = coupledStage;
                epoch.couplingIteration = nonlinearIteration
                                              ? uint64_t(*nonlinearIteration)
                                              : uint64_t(prepared);
                _interfaces->Refresh(ports(prepared, coupledStage), time, epoch);
                localPhase([&](index iDomain, Domain &domain)
                           {
                    if (const auto input = _interfaces->Inputs(iDomain))
                        domain.BindInterfaceInputs(*input, coupledStage); }, "interface binding");
            };
            const auto beginNonlinearRefresh =
                [&](real stageTime, int stage, int coupledStage,
                    index iteration)
            {
                DNDS_check_throw_info(
                    _interfaces &&
                        _interfaces->RefreshEveryNonlinearIteration(),
                    "nonlinear interface refresh requested for a frozen cadence");
                localPhase(
                    [&](index, Domain &domain)
                    {
                        domain.PrepareInterface(coupledStage, true, stageTime);
                    },
                    "nonlinear interface donor reconstruction");
                StateEpoch epoch;
                epoch.acceptedStep = _step;
                epoch.trial = trialIdentifier;
                epoch.stage = stage;
                epoch.coupledStage = coupledStage;
                epoch.couplingIteration = uint64_t(iteration);
                auto publication =
                    std::make_shared<CoordinatedInterfaceRefresh>(
                        _interfaces->BeginValueRefresh(
                            ports(true, coupledStage), stageTime, epoch));
                const auto completion = publication->CompletionEvent();
                for (index iDomain = 0; iDomain < index(_domains.size());
                     ++iDomain)
                    if (auto &domain = _domains[size_t(iDomain)])
                    {
                        if (!domain->Discretization().HasInterfaces())
                            continue;
                        auto receiver = domain->PrepareInterfaceReceiver(coupledStage);
                        QueuedEvaluationDependency dependency;
                        dependency.completion = completion;
                        dependency.owner = publication;
                        if (auto packet = publication->PendingInputs(iDomain))
                            dependency.incoming = std::make_shared<const BoundaryInputs>(std::move(*packet));
                        dependency.publish =
                            [publication, receiver = std::move(receiver), iDomain]
                        {
                            publication->Complete();
                            if (const auto input = publication->Inputs(iDomain))
                                receiver(*input);
                        };
                        domain->SetEvaluationDependency(
                            std::move(dependency));
                    }
                return publication;
            };
            const auto solveGroup = [&](auto &sessions,
                                        const DomainStageGroup &description)
            {
                const TimelineScope stageTimeline("Cauchy/Stage", description.stage);
                NonlinearSolveGroup group;
                group.communication = _placement.ControlContext();
                group.policy = _updatePolicy;
                group.collectiveChecks = _nonlinearCollectiveChecks;
                for (auto &session : sessions)
                    if (session)
                        group.domains.emplace_back(*session);
                std::shared_ptr<CoordinatedInterfaceRefresh> publication;
                group.refreshInterfaces = [&](int coupledStage, index iteration)
                {
                    if (!_interfaces ||
                        !_interfaces->RefreshEveryNonlinearIteration())
                        return;
                    publication = beginNonlinearRefresh(
                        description.times[size_t(coupledStage)],
                        description.stage, coupledStage,
                        iteration);
                };
                group.finishInterfaces = [&]
                {
                    for (auto &domain : _domains)
                        if (domain)
                            domain->DrainEvaluationDependency();
                    if (publication)
                        publication->Finish();
                    publication.reset();
                };
                // Exactly one dispatch. The selected policy owns its entire
                // nonlinear loop and orders interface/domain sweeps itself.
                SolveNonlinearGroup(group);
            };
            std::vector<std::optional<Domain::CandidateReservation>> candidates(_domains.size());
            const auto prepareAndCommit = [&]() -> StepResult
            {
                if (auto start = localPhase([&](index, Domain &domain)
                                            {
                    auto &control = domain;
                    MPI::LocalStage(MPIInfo(control.Communicator()),[&]
                    {
                        control.RequireAcceptedClock(_time, _step);
                    });
                    return control.BeginStep(dt); }, "step preparation");
                    !start)
                    return start;
                if (Local(0))
                    trialIdentifier = _domains[0]->TrialIdentifier();
                MPI_Bcast(&trialIdentifier, 1, MPI_UINT64_T, _placement.SimulationRank(0, 0), _placement.Control());
                int more = Local(0) ? int(_domains[0]->HasNextGroup()) : 0;
                MPI_Bcast(&more, 1, MPI_INT, _placement.SimulationRank(0, 0),
                          _placement.Control());
                while (more)
                {
                    std::vector<std::optional<DomainStageGroup>> prepared(_domains.size());
                    localPhase([&](index iDomain, Domain &domain)
                               { prepared[size_t(iDomain)] = domain.PrepareGroup(); },
                               "time-group preparation");
                    int shape[3]{};
                    std::array<real, 2> times{};
                    if (Local(0))
                    {
                        const auto &group = *prepared[0];
                        shape[0] = group.stage;
                        shape[1] = group.coupledStages;
                        shape[2] = int(group.implicit);
                        times = group.times;
                    }
                    MPI_Bcast(shape, 3, MPI_INT, _placement.SimulationRank(0, 0),
                              _placement.Control());
                    MPI_Bcast(times.data(), 2, MPI_DOUBLE,
                              _placement.SimulationRank(0, 0), _placement.Control());
                    const DomainStageGroup group{shape[0], shape[1], shape[2] != 0, times};
                    localPhase([&](index iDomain, Domain &)
                               {
                        const auto &local = *prepared[size_t(iDomain)];
                        DNDS_check_throw_info(
                            local.stage == group.stage &&
                                local.coupledStages == group.coupledStages &&
                                local.implicit == group.implicit &&
                                local.times == group.times,
                            "domain time-group descriptions differ"); },
                               "time-group agreement");
                    // This loop prepares independent frozen stage packets.
                    // Nonlinear update ordering belongs to the chosen policy.
                    if (_interfaces)
                        for (int slot = 0; slot < group.coupledStages; ++slot)
                        {
                            const real time = group.times[size_t(slot)];
                            localPhase([&](index, Domain &domain)
                                       { domain.PrepareInterface(slot, false, time); },
                                       "interface predictor halos");
                            refresh(time, group.stage, false, false, slot);
                            localPhase([&](index, Domain &domain)
                                       { domain.PrepareInterface(slot, true, time); },
                                       "interface donor reconstruction");
                            refresh(time, group.stage, true, false, slot);
                        }

                    if (group.implicit)
                    {
                        std::vector<std::optional<NonlinearSession>> sessions(_domains.size());
                        localPhase([&](index iDomain, Domain &domain)
                                   {
                            sessions[size_t(iDomain)] = domain.PrepareGroupSolve();
                            DNDS_check_throw_info(sessions[size_t(iDomain)],
                                                  "implicit group has no solve record"); },
                                   "nonlinear group preparation");
                        solveGroup(sessions, group);
                        if (auto result = localPhase(
                                [&](index iDomain, Domain &domain)
                                { return domain.FinishGroupSolve(*sessions[size_t(iDomain)]); },
                                "nonlinear group completion");
                            !result)
                            return result;
                    }
                    else if (auto result = localPhase(
                                 [](index, Domain &domain)
                                 { return domain.ApplyExplicitGroup(); },
                                 "explicit group update");
                             !result)
                        return result;
                    if (auto result = localPhase(
                            [](index, Domain &domain)
                            { return domain.CompleteGroup(); },
                            "time-group completion");
                        !result)
                        return result;
                    more = Local(0) ? int(_domains[0]->HasNextGroup()) : 0;
                    MPI_Bcast(&more, 1, MPI_INT, _placement.SimulationRank(0, 0),
                              _placement.Control());
                }
                if (_interfaces)
                {
                    localPhase([](index, Domain &domain)
                               { domain.PrepareInterface(0, false, 0); }, "accepted-time predictor halos");
                    refresh(_config.Time().steady ? _time : _time + dt, 0, false, true);
                }
                // Only local lease acquisition belongs in this control-stage
                // catch. Domain solves above contain MPI and must not be caught
                // and followed by an unrelated control reduction on local failure.
                try
                {
                    MPI::LocalStage(MPIInfo(_placement.Control()), [&]
                                    {
                    for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                        if (auto &domain = _domains[iDomain])
                            candidates[iDomain].emplace(domain->ReserveCandidate()); });
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("Cauchy coordinated state reservation failed");
                    candidates.clear();
                    for (auto &domain : _domains)
                        if (domain)
                            domain->DiscardStep();
                    if (_interfaces)
                    {
                        _interfaces->Discard();
                        for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                            if (_domains[iDomain])
                                if (const auto input = _interfaces->Inputs(iDomain))
                                    _domains[iDomain]->BindInterfaceInputs(*input);
                    }
                    throw;
                }
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                        _domains[iDomain]->CommitCandidate(*candidates[iDomain]);
                if (_interfaces)
                    _interfaces->Commit();
                return {};
            };
            const auto result = prepareAndCommit();
            if (!result)
            {
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                {
                    candidates[iDomain].reset();
                    if (_domains[iDomain])
                        _domains[iDomain]->DiscardStep();
                }
                if (_interfaces)
                {
                    _interfaces->Discard();
                    // Restore only ready accepted packets. No donor search,
                    // payload transfer or physical recomputation is needed.
                    for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                        if (_domains[iDomain])
                            if (const auto input = _interfaces->Inputs(iDomain))
                                _domains[iDomain]->BindInterfaceInputs(*input);
                }
                return result;
            }
            if (!_config.Time().steady)
            {
                _time = acceptedTime.value_or(_time + dt);
                for (auto &domain : _domains)
                    if (domain)
                        domain->SnapAcceptedTime(_time);
            }
            ++_step;
            return {};
        }
    }
}

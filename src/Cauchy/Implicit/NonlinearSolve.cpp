#include "Cauchy/Implicit/NonlinearSolve.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "Cauchy/Implicit/ResidualConvergence.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

namespace DNDS::Cauchy
{
    struct NonlinearSession::Data
    {
        enum class Phase
        {
            Idle,
            Updating,
            Observing,
            ReadyForUpdate,
            AwaitFinish,
            Complete
        };
        ImplicitSettings settings;
        EquationSchema schema;
        int rank = 0, stage = 0;
        std::shared_ptr<const UpdatePolicy> policy;
        NonlinearProblem problem;
        NonlinearIterationReport report;
        BasicResidualConvergence<std::vector<real>> convergence;
        index limit = 0, nextIteration = 0, currentIteration = -1;
        int nextCoupledStage = 0;
        bool localConverged = false, localTerminal = false, held = false;
        Phase phase = Phase::Idle;
        std::optional<NonlinearProgress> progress;
        std::chrono::steady_clock::time_point iterationStarted;

        Data(ImplicitSettings settingsIn, EquationSchema schemaIn, int rankIn,
             int stageIn, std::shared_ptr<const UpdatePolicy> policyIn,
             NonlinearProblem problemIn)
            : settings(std::move(settingsIn)), schema(std::move(schemaIn)),
              rank(rankIn), stage(stageIn), policy(std::move(policyIn)),
              problem(std::move(problemIn)), report(std::move(problem.initial)),
              convergence(settings, schema, [&]
                          {
                              index components = 0;
                              for (const auto &block : schema.blocks)
                                  components += index(block.components.size());
                              if (report.residual.empty())
                                  report.residual.assign(size_t(components), 0);
                              return report.residual; }())
        {
            (void)rank;
            limit = problem.control.correctionLimit < 0
                        ? settings.iterations
                        : problem.control.correctionLimit;
            const bool after = policy->Descriptor().observation ==
                               NonlinearDetail::ResidualObservation::AfterUpdate;
            DNDS_check_throw_info(!after || limit > 0,
                                  "post-update residual observation requires at least one update");
            nextIteration = after ? 1 : 0;
            if (problem.control.resume)
            {
                DNDS_check_throw_info(problem.control.state &&
                                          problem.control.state->schemaVersion == 1 &&
                                          problem.control.state->iteration >= 0 &&
                                          !problem.control.state->rejected,
                                      "nonlinear resume needs a valid live controller state");
                const auto &state = *problem.control.state;
                DNDS_check_throw_info(
                    state.stage == stage &&
                        state.residual.size() == report.residual.size() &&
                        state.initialResidual.size() == report.residual.size() &&
                        state.referenceResidual.size() == report.residual.size() &&
                        std::isfinite(state.pseudoCFL) && state.pseudoCFL > 0,
                    "nonlinear resume state differs from the active equation");
                report.iterations = state.iteration;
                report.residual = state.residual;
                report.initialResidual = state.initialResidual;
                report.referenceResidual = state.referenceResidual;
                report.residualTargetRatio = state.residualTargetRatio;
                report.pseudoCFL = state.pseudoCFL;
                report.linearIterations = state.linearIterations;
                report.multigridCycles = state.multigridCycles;
                report.levelCorrections = state.levelCorrections;
                report.rhsLimited = state.rhsLimited;
                report.incrementLimited = state.incrementLimited;
                report.minimumRHSAlpha = state.minimumRHSAlpha;
                report.minimumIncrementAlpha = state.minimumIncrementAlpha;
                report.reconstructionIterations = state.reconstructionIterations;
                report.reconstructionRelativeUpdate = state.reconstructionRelativeUpdate;
                report.reconstructionUpdateMeasured = state.reconstructionUpdateMeasured;
                convergence.Restore(report.initialResidual,
                                    report.referenceResidual);
                nextIteration = state.iteration + 1;
                DNDS_check_throw_info(state.terminal || nextIteration <= limit,
                                      "restored nonlinear cursor exceeds its correction cap");
                if (state.terminal)
                {
                    report.converged = state.converged;
                    localConverged = state.converged;
                    localTerminal = true;
                    phase = Phase::Complete;
                }
            }
            else
            {
                report.pseudoCFL = problem.control.initialPseudoCFL == 0
                                       ? settings.pseudoCFL
                                       : problem.control.initialPseudoCFL;
                if (problem.control.state)
                {
                    auto &state = *problem.control.state;
                    state.stage = stage;
                    state.coupledStage = -1;
                    state.iteration = 0;
                    state.converged = state.terminal = state.capped =
                        state.rejected = false;
                    state.pseudoCFL = report.pseudoCFL;
                }
            }
        }

        void PublishState(bool terminal)
        {
            if (!problem.control.state)
                return;
            auto &state = *problem.control.state;
            state.iteration = currentIteration;
            state.converged = localConverged;
            state.terminal = terminal;
            state.capped = terminal && !localConverged;
            state.residual = report.residual;
            state.initialResidual = report.initialResidual;
            state.referenceResidual = report.referenceResidual;
            state.residualTargetRatio = report.residualTargetRatio;
            state.pseudoCFL = report.pseudoCFL;
            state.linearIterations = report.linearIterations;
            state.multigridCycles = report.multigridCycles;
            state.levelCorrections = report.levelCorrections;
            state.rhsLimited = report.rhsLimited;
            state.incrementLimited = report.incrementLimited;
            state.minimumRHSAlpha = report.minimumRHSAlpha;
            state.minimumIncrementAlpha = report.minimumIncrementAlpha;
            state.reconstructionIterations = report.reconstructionIterations;
            state.reconstructionRelativeUpdate =
                report.reconstructionRelativeUpdate;
            state.reconstructionUpdateMeasured =
                report.reconstructionUpdateMeasured;
        }
    };

    NonlinearSolveSetup::NonlinearSolveSetup(
        const ImplicitSettings &settings, EquationSchema schema, int rank,
        const std::string &policy,
        const UpdatePolicyRequirements &requirements)
        : settings(settings), schema(std::move(schema)), rank(rank),
          policy(UpdatePolicyRegistry::Resolve(policy, requirements))
    {
        DNDS_check_throw_info(rank >= 0,
                              "nonlinear solve needs a domain rank");
    }

    NonlinearSession NonlinearSession::Prepare(
        const NonlinearSolveSetup &setup, int stage, NonlinearProblem problem)
    {
        DNDS_check_throw_info(setup.policy, "nonlinear solve has no update policy");
        const auto &descriptor = setup.policy->Descriptor();
        problem.Validate(descriptor);
        return NonlinearSession(std::make_shared<NonlinearSession::Data>(
            setup.settings, setup.schema, setup.rank, stage, setup.policy,
            std::move(problem)));
    }

    void NonlinearSession::BeginIteration()
    {
        DNDS_check_throw_info(_data && _data->phase == Data::Phase::Idle,
                              "nonlinear session is not ready for another iteration");
        auto &data = *_data;
        data.currentIteration = data.nextIteration;
        data.nextCoupledStage = 0;
        data.localConverged = data.localTerminal = false;
        data.progress.reset();
        data.iterationStarted = std::chrono::steady_clock::now();
        data.phase = ObservationBeforeUpdate() ? Data::Phase::Observing
                                               : Data::Phase::Updating;
    }

    void NonlinearSession::AdvanceStage(int coupledStage)
    {
        DNDS_check_throw_info(_data &&
                                  (_data->phase == Data::Phase::Updating ||
                                   _data->phase == Data::Phase::ReadyForUpdate) &&
                                  coupledStage == _data->nextCoupledStage &&
                                  coupledStage < CoupledStageNumber(),
                              "nonlinear coupled-stage update is out of order");
        auto &data = *_data;
        const TimelineScope timeline("Cauchy/NonlinearUpdate", data.currentIteration);
        data.problem.advanceStage(coupledStage, data.report, !data.held);
        ++data.nextCoupledStage;
        if (data.nextCoupledStage == CoupledStageNumber())
            data.phase = ObservationBeforeUpdate() ? Data::Phase::AwaitFinish
                                                   : Data::Phase::Observing;
        else
            data.phase = Data::Phase::Updating;
    }

    void NonlinearSession::Observe()
    {
        const TimelineScope timeline("Cauchy/Nonlinear/ObserveResidual");
        DNDS_check_throw_info(_data && _data->phase == Data::Phase::Observing,
                              "nonlinear residual observation is out of order");
        auto &data = *_data;
        {
            const NonlinearDetail::PhaseTimer timer(
                data.problem.control.profile, NonlinearProfile::Residual);
            data.problem.evaluate();
            if (data.problem.control.profile)
                ++data.problem.control.profile->residualEvaluations;
        }
        std::tie(data.report.reconstructionIterations,
                 data.report.reconstructionRelativeUpdate) =
            data.problem.reconstruction();
        data.report.reconstructionUpdateMeasured =
            std::isfinite(data.report.reconstructionRelativeUpdate);
        if (!data.report.reconstructionUpdateMeasured)
            data.report.reconstructionRelativeUpdate = 0;
        {
            const NonlinearDetail::PhaseTimer timer(
                data.problem.control.profile, NonlinearProfile::Convergence);
            data.problem.measure(data.report.residual);
            data.convergence.Observe(data.report.residual);
            data.report.initialResidual = data.convergence.Initial();
            data.report.referenceResidual = data.convergence.Reference();
            data.report.residualTargetRatio = data.convergence.TargetRatio();
            const real constraintRatio = data.problem.constraintTargetRatio ? data.problem.constraintTargetRatio() : 0;
            data.report.residualTargetRatio = std::max(data.report.residualTargetRatio, constraintRatio);
            data.report.iterations = data.currentIteration;
            data.localConverged =
                data.currentIteration >=
                    std::min(data.settings.minIterations, data.limit) &&
                (data.problem.reconstructionTolerance <= 0 ||
                 (data.report.reconstructionUpdateMeasured &&
                  data.report.reconstructionRelativeUpdate <=
                      data.problem.reconstructionTolerance)) &&
                data.convergence.Accepted() && constraintRatio <= 1;
            data.localTerminal =
                data.localConverged || data.currentIteration >= data.limit;
            if (!data.localTerminal && ObservationBeforeUpdate())
                data.held = false;
        }
        data.phase = ObservationBeforeUpdate() && !data.localTerminal
                         ? Data::Phase::ReadyForUpdate
                         : Data::Phase::AwaitFinish;
    }

    void NonlinearSession::FinishIteration(bool coordinatedTerminal)
    {
        DNDS_check_throw_info(_data &&
                                  _data->phase == Data::Phase::AwaitFinish &&
                                  (!coordinatedTerminal || _data->localTerminal),
                              "nonlinear iteration cannot finish before local readiness");
        auto &data = *_data;
        data.report.converged = data.localConverged;
        data.PublishState(coordinatedTerminal);
        const index reportedIteration = std::max(data.currentIteration, index(1));
        const bool initialBeforeUpdate = ObservationBeforeUpdate() &&
                                         data.currentIteration == 0;
        const auto copy = [](const auto &values)
        { return std::vector<real>(values.begin(), values.end()); };
        NonlinearProgress progress{
            data.stage,
            -1,
            reportedIteration,
            data.localConverged,
            coordinatedTerminal,
            copy(data.report.residual),
            copy(data.report.initialResidual),
            copy(data.report.referenceResidual),
            data.report.residualTargetRatio,
            data.report.pseudoCFL,
            data.report.linearIterations,
            data.report.rhsLimited,
            data.report.incrementLimited,
            data.report.minimumRHSAlpha,
            data.report.minimumIncrementAlpha,
            data.report.reconstructionIterations,
            data.report.reconstructionRelativeUpdate,
            data.report.reconstructionUpdateMeasured,
            std::chrono::duration<real>(std::chrono::steady_clock::now() - data.iterationStarted).count(),
            data.convergence.RelativeRatio()};
        if (data.problem.control.progress && !initialBeforeUpdate)
            data.problem.control.progress(progress);
        if (coordinatedTerminal)
        {
            if (data.problem.control.quiescent)
            {
                DNDS_check_throw_info(data.problem.control.state,
                                      "quiescent nonlinear observer needs live controller state");
                data.problem.control.quiescent(*data.problem.control.state);
            }
            data.phase = Data::Phase::Complete;
            return;
        }
        data.held = data.localTerminal;
        if (!data.held)
            data.report.pseudoCFL = std::min(
                data.settings.maxPseudoCFL,
                data.report.pseudoCFL * data.settings.pseudoGrowth);
        if (data.problem.control.state)
            data.problem.control.state->pseudoCFL = data.report.pseudoCFL;
        if (data.problem.control.quiescent)
        {
            DNDS_check_throw_info(data.problem.control.state,
                                  "quiescent nonlinear observer needs live controller state");
            data.problem.control.quiescent(*data.problem.control.state);
        }
        ++data.nextIteration;
        data.phase = Data::Phase::Idle;
    }

    bool NonlinearSession::ObservationBeforeUpdate() const
    {
        DNDS_check_throw_info(_data, "empty nonlinear session");
        return _data->policy->Descriptor().observation ==
               NonlinearDetail::ResidualObservation::BeforeUpdate;
    }
    int NonlinearSession::CoupledStageNumber() const
    {
        DNDS_check_throw_info(_data, "empty nonlinear session");
        return _data->policy->Descriptor().coupledStages;
    }
    index NonlinearSession::Iteration() const
    {
        DNDS_check_throw_info(_data, "empty nonlinear session");
        return _data->currentIteration;
    }
    index NonlinearSession::NextIteration() const
    {
        DNDS_check_throw_info(_data, "empty nonlinear session");
        return _data->nextIteration;
    }
    bool NonlinearSession::LocalTerminal() const
    {
        DNDS_check_throw_info(_data, "empty nonlinear session");
        return _data->localTerminal;
    }
    bool NonlinearSession::Complete() const
    {
        DNDS_check_throw_info(_data, "empty nonlinear session");
        return _data->phase == Data::Phase::Complete;
    }
    const NonlinearIterationReport &NonlinearSession::Report() const
    {
        DNDS_check_throw_info(Complete(),
                              "nonlinear report requested before coordinated completion");
        return _data->report;
    }

    const UpdatePolicy &NonlinearSession::Policy() const
    {
        DNDS_check_throw_info(_data, "empty nonlinear session");
        return *_data->policy;
    }

    index NonlinearSolveGroup::BeginIteration()
    {
        for (auto reference : domains)
        {
            auto &domain = reference.get();
            domain.BeginIteration();
            DNDS_check_throw_info(_iteration == domain.Iteration(),
                                  "colocated domains entered different nonlinear iterations");
        }
        if (communication && collectiveChecks)
        {
            const auto minimum = MPI::AllreduceValue<MPI::Reduction::Min>(
                _iteration, communication->Communicator());
            const auto maximum = MPI::AllreduceValue<MPI::Reduction::Max>(
                _iteration, communication->Communicator());
            DNDS_check_throw_info(minimum == maximum,
                                  "dislocated domains entered different nonlinear iterations");
        }
        return _iteration;
    }

    bool NonlinearSolveGroup::FinishIteration()
    {
        int terminal = 1;
        for (auto reference : domains)
            terminal = terminal && reference.get().LocalTerminal();
        // Termination agreement is part of the algorithm, not a diagnostic
        // preflight. It remains required when collectiveChecks is false.
        if (communication)
            terminal = MPI::AllreduceValue<MPI::Reduction::Min>(
                terminal, communication->Communicator());
        for (auto reference : domains)
            reference.get().FinishIteration(terminal != 0);
        if (!terminal)
            ++_iteration;
        return terminal != 0;
    }

    void NonlinearSolveGroup::RefreshInterfaces(int coupledStage,
                                                index iteration) const
    {
        if (refreshInterfaces)
            refreshInterfaces(coupledStage, iteration);
    }

    void NonlinearSolveGroup::FinishInterfaces() const
    {
        if (finishInterfaces)
            finishInterfaces();
    }

    void SolveNonlinearGroup(NonlinearSolveGroup &group)
    {
        DNDS_check_throw_info(group.policy || !group.domains.empty(),
                              "nonlinear group has no prepared policy");
        const auto &policy = group.policy ? *group.policy
                                          : group.domains.front().get().Policy();
        const auto descriptor = policy.Descriptor().Metadata();
        // One setup agreement makes resumed cursors and terminal status
        // unambiguous, including control ranks with no local domain. This is
        // required algorithmic ordering, independent of optional diagnostics.
        index bounds[]{1, 1, std::numeric_limits<index>::max(),
                       std::numeric_limits<index>::max()};
        for (auto reference : group.domains)
        {
            const auto &session = reference.get();
            DNDS_check_throw_info(
                session.Policy().Descriptor().Metadata() == descriptor,
                "domain nonlinear policy differs from its coordinated group");
            bounds[0] = std::min(bounds[0], index(session.Complete()));
            bounds[1] = std::min(bounds[1], index(!session.Complete()));
            bounds[2] = std::min(bounds[2], session.NextIteration());
            bounds[3] = std::min(bounds[3], -session.NextIteration());
        }
        if (group.communication)
            MPI::AllreduceInPlace<MPI::Reduction::Min>(
                MPI::ContiguousView<index>{bounds, 4},
                group.communication->Communicator());
        DNDS_check_throw_info((bounds[0] || bounds[1]) &&
                                  bounds[2] != std::numeric_limits<index>::max() &&
                                  bounds[2] == -bounds[3],
                              "domain restored nonlinear cursors differ");
        group._iteration = bounds[2];
        if (!bounds[0])
            policy.Solve(group);
    }

    NonlinearIterationReport SolveNonlinear(NonlinearSession &session)
    {
        NonlinearSolveGroup group;
        group.domains.emplace_back(session);
        SolveNonlinearGroup(group);
        return session.Report();
    }

    NonlinearIterationReport SolveNonlinear(const NonlinearSolveSetup &setup,
                                            int stage, NonlinearProblem problem)
    {
        auto session = NonlinearSession::Prepare(setup, stage, std::move(problem));
        return SolveNonlinear(session);
    }
}

#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include "Solver/TimeIntegration/DITRMethod.hpp"
#include "Solver/TimeIntegration/SerializedMethods.hpp"
#include <functional>
#include <algorithm>
#include <optional>
#include <limits>
#include <type_traits>

namespace DNDS::ODE
{
    /// Conservative semidiscrete time equations on native fields. Spatial
    /// operators supply the spatial rate F; the stage solver receives
    ///     mass*M U - rhsScale*F(U,t) = source.
    /// It owns nonlinear iteration, PP, physical accumulation and MPI policy.
    /// This facade owns transaction/order checks and exactly one method owner.
    /// The method owns its tableau, history, trial scratch and source arithmetic.
    /// State terms use physical accumulation M; rate and state publication
    /// use ordinary arithmetic even when M has zero algebraic rows.
    /// Operations supplies NewVector, CheckCompatible, RequireDistinct, a
    /// binding/version snapshot, and the five pointwise algebra actions. It
    /// owns storage/backend access; this class owns one set of time mathematics.
    template <class State, class Operations>
    class StateTimeStepper
    {
        static_assert(std::is_nothrow_swappable_v<State>, "time state handles must support allocation-free, nonthrowing history swaps");

    public:
        using Field = State;
        using DITR = DITRMethod<State, Operations>;
        using Evaluate = std::function<void(Field &, Field &, real, int)>;
        using LocalStage = std::function<void(const std::function<void()> &)>;
        /// Immutable group ticket. It identifies method-owned state without
        /// exporting a concrete method or granting unvalidated scratch access.
        class StageGroup
        {
            friend class StateTimeStepper;
            std::shared_ptr<const uint8_t> _session;
            uint64_t _generation;
            StageGroup(std::shared_ptr<const uint8_t> session, uint64_t generation,
                       int s, int count, bool i, std::array<real, 2> times)
                : _session(std::move(session)), _generation(generation),
                  stage(s), coupledStages(count), implicit(i), times(times) {}

        public:
            const int stage;
            const int coupledStages;
            const bool implicit;
            const std::array<real, 2> times;
        };
        /// Deep, backend-agnostic payload captured only at a nonlinear
        /// iteration boundary. Runtime handles/leases are deliberately absent;
        /// restore recreates bindings against the caller-provided trial.
        struct ActiveCheckpoint
        {
            int schemaVersion = 2;
            std::string method;
            /// Mathematical method parameters, not stopping/output controls.
            /// Active stages cannot be interpreted using another DITR tableau.
            std::optional<DITRSettings> ditrSettings;
            real pendingDt = 0, startTime = 0;
            bool pendingSteady = false, stagesActive = false, stagePrepared = false;
            int nextStage = 0;
            TimeEquationCoefficients equation;
            bool bdfHistory = false;
            real bdfPreviousDt = 0;
            real ditrTime = 0, ditrDt = 0;
            std::vector<std::string> names;
            std::vector<Field> fields;
        };

    private:
        Operations _operations;
        std::shared_ptr<const uint8_t> _session = std::make_shared<const uint8_t>(0);
        uint64_t _stageGeneration = 0;
        std::string _method;
        std::unique_ptr<BDFMethod<State, Operations>> _bdf;
        std::unique_ptr<DIRKMethod<State, Operations>> _dirk;
        real _pendingDt = 0;
        bool _pending = false, _pendingSteady = false;
        LocalStage _local;
        DITRSettings _ditrSettings;
        std::unique_ptr<DITR> _ditr;
        bool _stagesActive = false, _stagePrepared = false;
        std::optional<TimeEquationCoefficients> _preparedEquation;
        using Binding = typename Operations::Binding;
        std::optional<Binding> _preparedSource, _preparedState;
        int _nextStage = 0;
        real _startTime = 0;
        template <class F>
        void RunLocal(F &&run) const
        {
            if (_local)
                _local(std::forward<F>(run));
            else
                run();
        }
        const Field &Prototype() const
        {
            return _bdf ? _bdf->Source() : _dirk ? _dirk->Source()
                                                 : _ditr->IntermediateState();
        }
        const Field &Source() const
        {
            DNDS_check_throw_info(_bdf || _dirk, "coupled time methods own their own source");
            return _bdf ? _bdf->Source() : _dirk->Source();
        }
        void Check(const Field &field) const { _operations.CheckCompatible(Prototype(), field); }
        void BeginInitial(const Field &initial, Field &trial)
        {
            Check(initial);
            Check(trial);
            Operations::RequireDistinct(initial, trial);
            if (_bdf)
                _bdf->Begin(initial);
            else if (_dirk)
                _dirk->Begin(initial);
        }
        void CheckPrepared(const Field &trial) const
        {
            RunLocal([&]
                     {
                DNDS_check_throw_info(_preparedState, "time stage has no reserved state binding");
                _preparedState->Check(trial);
                if (_preparedSource) _preparedSource->Check(Source(), true); });
        }
        Field Clone(const Field &source) const
        {
            auto result = Operations::NewVector(source);
            _operations.Copy(result, source);
            return result;
        }

    public:
        StateTimeStepper(const Field &prototype, std::string method, LocalStage local = {}, DITRSettings ditr = {}, Operations operations = {})
            : _operations(std::move(operations)), _method(std::move(method)), _local(std::move(local)), _ditrSettings(ditr)
        {
            if (_method == "bdf1" || _method == "bdf2")
                _bdf = std::make_unique<BDFMethod<State, Operations>>(prototype, _method == "bdf2" ? 2 : 1, _operations);
            else if (_method == "ditr")
                _ditr = std::make_unique<DITR>(prototype, ditr.fraction, ditr.theta, ditr.variant, _local, _operations);
            else
                _dirk = std::make_unique<DIRKMethod<State, Operations>>(prototype, _method, _operations);
        }
        StateTimeStepper(const StateTimeStepper &) = delete;
        StateTimeStepper &operator=(const StateTimeStepper &) = delete;
        const std::string &Method() const { return _method; }
        /// Avoid a spurious roundoff-sized final step. Times are finite and
        /// nonnegative; relative scaling also preserves tiny physical horizons.
        static bool ReachedTimeTarget(real time, real target)
        {
            // Allow modest accumulated addition error, e.g. 800 IV steps or
            // 2500 cylinder steps, while scaling down for tiny time horizons.
            return time >= target || target - time <= 1024 * std::numeric_limits<real>::epsilon() * std::max(std::abs(time), std::abs(target));
        }
        /// Allocate an independent empty-history workspace with identical method
        /// and local-error policy, e.g. to prepare a restart transaction before
        /// replacing the live accepted history.
        std::unique_ptr<StateTimeStepper> Fresh() const
        {
            return std::make_unique<StateTimeStepper>(Prototype(), _method, _local, _ditrSettings, _operations);
        }
        bool HasHistory() const { return _bdf && _bdf->HasHistory(); }
        bool HasPendingTrial() const { return _pending; }
        /// Serialized group count: DIRK tableau stages, otherwise one group.
        int GroupNumber() const { return _dirk ? _dirk->StageNumber() : 1; }
        int CoupledStageNumber() const { return _ditr ? 2 : 1; }
        /// Physical spatial rate at a completed coupled update. Both rates
        /// are retained by the method. A singular mass operator can require
        /// separate constraint norms even when I+H already meets its target.
        const Field &CompletedCoupledRate(const StageGroup &group, int coupledStage) const
        {
            DNDS_check_throw_info(_ditr && _stagePrepared && group._session == _session &&
                                      group._generation == _stageGeneration && (coupledStage == 0 || coupledStage == 1),
                                  "completed coupled rate needs a DITR stage");
            (void)_ditr->CheckpointControl();
            return coupledStage == 0 ? _ditr->IntermediateRate() : _ditr->EndpointRate();
        }
        real PreviousDt() const { return _bdf ? _bdf->PreviousDt() : 0; }
        const Field &PreviousState() const
        {
            DNDS_check_throw_info(_bdf, "this method has no accepted-step history");
            return _bdf->PreviousState();
        }

        /**
         * @brief Begin one physical trial for every supported time method.
         * @details No nonlinear solve is called here. DITR initializes its
         * midpoint/end rate guesses with the supplied initial-state evaluator;
         * BDF/DIRK prepare their initial/history operands. Accepted state and
         * history remain unchanged until Commit. Steady means BDF1 with zero
         * physical mass and no infinite floating-point operand.
         */
        void BeginStep(const Field &initial, Field &trial, real time, real dt,
                       const Evaluate &evaluate = {}, bool steady = false)
        {
            Discard();
            RunLocal([&]
                     {
                DNDS_check_throw_info(
                    std::isfinite(time) &&
                        (steady ? _method == "bdf1" :
                         std::isfinite(dt) && dt > 0 &&
                         std::isfinite(time + dt) && time + dt > time),
                    "invalid time-step controls");
                BeginInitial(initial, trial); });
            _pendingSteady = steady;
            _pendingDt = steady ? 0 : dt;
            _startTime = time;
            _nextStage = 0;
            if (_ditr)
                _ditr->Begin(initial, trial, time, dt, evaluate);
            _stagesActive = true;
        }

        /// Prepare the next mathematical group. Ordinary DIRK stages contain
        /// one unknown; the one DITR group contains midpoint and endpoint.
        /// A restored active group is obtained through ActiveGroup instead.
        StageGroup PrepareGroup(Field &trial)
        {
            DNDS_check_throw_info(
                _stagesActive && !_stagePrepared && _nextStage < GroupNumber(),
                "no next time group, or previous group is unfinished");
            DNDS_check_throw_info(_stageGeneration != std::numeric_limits<uint64_t>::max(),
                                  "time group generation exhausted");
            RunLocal([&]
                     {
                Check(trial);
                _preparedState.emplace(trial);
                if (!_ditr)
                {
                    _preparedEquation = _bdf
                        ? _bdf->Prepare(trial, _startTime, _pendingDt, _pendingSteady)
                        : _dirk->Prepare(trial, _nextStage, _startTime, _pendingDt);
                    if (!_dirk || _dirk->Implicit(_nextStage))
                        _preparedSource.emplace(Source());
                } });
            _stagePrepared = true;
            ++_stageGeneration;
            return ActiveGroup();
        }

        /// Return the current group ticket, including after exact restoration.
        StageGroup ActiveGroup() const
        {
            DNDS_check_throw_info(_stagesActive && _stagePrepared,
                                  "time method has no prepared group");
            const bool implicit = !_dirk || _dirk->Implicit(_nextStage);
            std::array<real, 2> times{};
            if (_ditr)
            {
                times[0] = _ditr->CoupledStageTime(0);
                times[1] = _ditr->CoupledStageTime(1);
            }
            else
                times[0] = _preparedEquation->time;
            return StageGroup(_session, _stageGeneration, _nextStage,
                              CoupledStageNumber(), implicit, times);
        }

        /// Validate ticket identity and retained field bindings at a group
        /// boundary. Values may change; replacing an allocation is forbidden.
        void ValidateGroup(const StageGroup &group, const Field &trial) const
        {
            RunLocal([&]
                     { DNDS_check_throw_info(
                           _stagesActive && _stagePrepared &&
                               group._session == _session &&
                               group._generation == _stageGeneration &&
                               group.stage == _nextStage &&
                               group.coupledStages == CoupledStageNumber(),
                           "expired or foreign time-group ticket"); });
            CheckPrepared(trial);
        }

        /// Borrow a group's actual unknown. The caller retains this stepper and
        /// trial through the scoped solve. No field storage is copied or owned
        /// by the ticket.
        Field &GroupState(const StageGroup &group, Field &trial, int coupledStage)
        {
            ValidateGroup(group, trial);
            DNDS_check_throw_info(coupledStage >= 0 &&
                                      coupledStage < group.coupledStages,
                                  "invalid time-group unknown");
            return _ditr && coupledStage == 0 ? _ditr->IntermediateState() : trial;
        }
        const Field &GroupState(const StageGroup &group, const Field &trial,
                                int coupledStage) const
        {
            ValidateGroup(group, trial);
            DNDS_check_throw_info(coupledStage >= 0 &&
                                      coupledStage < group.coupledStages,
                                  "invalid time-group unknown");
            return _ditr && coupledStage == 0 ? _ditr->IntermediateState() : trial;
        }

        /// Freeze the equation operands for a named update. The same returned
        /// shape serves one-stage methods and each DITR equation. DITR enforces
        /// its midpoint/end mathematical ordering internally.
        CoupledStageEquation<Field> PrepareEquation(const StageGroup &group,
                                                    Field &trial, int coupledStage)
        {
            ValidateGroup(group, trial);
            DNDS_check_throw_info(group.implicit && coupledStage >= 0 &&
                                      coupledStage < group.coupledStages,
                                  "invalid implicit time-group equation");
            if (_ditr)
                return _ditr->PrepareCoupledStage(trial, coupledStage);
            return {trial, Source(), *_preparedEquation, 0};
        }

        /// Method-provided rate guess, if applicable, for the currently
        /// prepared equation. Ordinary stages evaluate their residual normally.
        OptionalRef<const Field> CachedRate(const StageGroup &group) const
        {
            DNDS_check_throw_info(group._session == _session &&
                                      group._generation == _stageGeneration &&
                                      _stagePrepared,
                                  "expired time-group rate query");
            return _ditr ? OptionalRef<const Field>(_ditr->CurrentRate())
                         : OptionalRef<const Field>{};
        }

        /// Finish one update's mathematical dependency publication. DITR
        /// refreshes the corrected stage rate; DIRK needs its final rate only
        /// when CompleteGroup is called after convergence.
        void CompleteEquation(const StageGroup &group, Field &trial,
                              const Evaluate &evaluate)
        {
            ValidateGroup(group, trial);
            if (_ditr)
                _ditr->CompleteCoupledStage(trial, evaluate);
        }

        /// Method-specific coupled stopping residual. Ordinary one-stage
        /// defects are assembled by the common spatial/mass operator instead.
        void AssembleStoppingResidual(const StageGroup &group,
                                      const Field &trial, Field &residual)
        {
            ValidateGroup(group, trial);
            DNDS_check_throw_info(_ditr, "this group uses the ordinary stage defect");
            _ditr->AssembleStoppingResidual(trial, residual);
        }

        /// Complete a successful group, publishing only method-owned stage
        /// history. Numerical rejection leaves the group active until Discard.
        /// No nonlinear solve or hidden exchange is launched.
        void CompleteGroup(const StageGroup &group, Field &trial,
                           const Evaluate &evaluate = {})
        {
            ValidateGroup(group, trial);
            if (_ditr)
                (void)_ditr->CheckpointControl();
            else if (_dirk)
            {
                DNDS_check_throw_info(evaluate, "DIRK completion needs a rate evaluator");
                const Binding input(trial), output(_dirk->Rate(_nextStage));
                evaluate(trial, _dirk->Rate(_nextStage),
                         _dirk->StageTime(_nextStage, _startTime, _pendingDt),
                         _nextStage);
                RunLocal([&]
                         { input.Check(trial, true); output.Check(_dirk->Rate(_nextStage)); });
            }
            _preparedSource.reset();
            _preparedState.reset();
            _preparedEquation.reset();
            _stagePrepared = false;
            if (++_nextStage == GroupNumber())
            {
                _pending = true;
                _stagesActive = false;
            }
        }
        /// Only call once the matching physical trial was published. Pure
        /// handle swaps allocate nothing and cannot fail midway through history.
        void Commit()
        {
            DNDS_check_throw_info(_pending, "no complete time trial to commit");
            if (_ditr)
            {
                _pending = false;
                _ditr->Discard();
                return;
            }
            if (_bdf)
                _bdf->Commit(_pendingDt, _pendingSteady);
            _pending = _pendingSteady = false;
        }
        void Discard() noexcept
        {
            _pending = _pendingSteady = _stagesActive = _stagePrepared = false;
            if (_ditr)
                _ditr->Discard();
            _preparedSource.reset();
            _preparedState.reset();
            _preparedEquation.reset();
        }
        /// After an externally replaced state or incompatible restart, take a
        /// BDF1 startup step rather than silently consuming stale history.
        void ResetHistory() noexcept
        {
            Discard();
            if (_bdf)
                _bdf->ResetHistory();
        }
        void RestoreHistory(const Field &previous, real dt)
        {
            Check(previous);
            DNDS_check_throw_info(_bdf, "this time method cannot restore BDF history");
            Discard();
            _bdf->RestoreHistory(previous, dt);
        }

        ActiveCheckpoint CaptureActive(const Field &trial) const
        {
            ActiveCheckpoint checkpoint;
            checkpoint.method = _method;
            if (_ditr)
                checkpoint.ditrSettings = _ditrSettings;
            checkpoint.pendingDt = _pendingDt;
            checkpoint.startTime = _startTime;
            checkpoint.pendingSteady = _pendingSteady;
            const auto append = [&](const std::string &name, const Field &field)
            {
                checkpoint.names.push_back(name);
                checkpoint.fields.push_back(Clone(field));
            };
            RunLocal([&]
                     {
                Check(trial);
                append("trial", trial);
                if (_ditr)
                {
                    const auto control = _ditr->CheckpointControl();
                    checkpoint.ditrTime = control.time;
                    checkpoint.ditrDt = control.dt;
                    append("ditr_initial", _ditr->InitialState());
                    append("ditr_middle", _ditr->IntermediateState());
                    append("ditr_initial_rate", _ditr->InitialRate());
                    append("ditr_middle_rate", _ditr->IntermediateRate());
                    append("ditr_end_rate", _ditr->EndpointRate());
                    append("ditr_res_other", _ditr->ResidualOther());
                    return;
                }
                DNDS_check_throw_info(_stagesActive && _stagePrepared && _preparedEquation,
                                      "serialized time checkpoint needs an active prepared stage");
                checkpoint.stagesActive = _stagesActive;
                checkpoint.stagePrepared = _stagePrepared;
                checkpoint.nextStage = _nextStage;
                checkpoint.equation = *_preparedEquation;
                if (_bdf)
                {
                    checkpoint.bdfHistory = _bdf->HasHistory();
                    checkpoint.bdfPreviousDt = _bdf->PreviousDt();
                    append("time_initial", _bdf->InitialState());
                    append("time_source", _bdf->PreparedSource());
                    append("bdf_previous", _bdf->PreviousStorage());
                }
                else
                {
                    append("time_initial", _dirk->InitialState());
                    append("time_source", _dirk->Source());
                    for (index stage = 0; stage < index(_dirk->Rates().size()); ++stage)
                        append("dirk_rate_" + std::to_string(stage), _dirk->Rates()[size_t(stage)]);
                } });
            return checkpoint;
        }

        void RestoreActive(const ActiveCheckpoint &checkpoint, Field &trial)
        {
            RunLocal([&]
                     {
                DNDS_check_throw_info(checkpoint.schemaVersion == 2 && checkpoint.method == _method &&
                                          checkpoint.names.size() == checkpoint.fields.size(),
                                      "time checkpoint method or payload schema differs");
                // Check the mathematical signature before mutating trial or
                // discarding an existing group. Accepted-step restarts may
                // change methods; an exact internal restart must not combine
                // stored stage/rate values with different method coefficients.
                DNDS_check_throw_info(bool(checkpoint.ditrSettings) == bool(_ditr),
                                      "time checkpoint method parameters are missing or unexpected");
                if (_ditr)
                {
                    checkpoint.ditrSettings->Validate();
                    DNDS_check_throw_info(checkpoint.ditrSettings->fraction == _ditrSettings.fraction &&
                                              checkpoint.ditrSettings->theta == _ditrSettings.theta &&
                                              checkpoint.ditrSettings->variant == _ditrSettings.variant,
                                          "time checkpoint DITR coefficients differ");
                }
                const auto payload = [&](const std::string &name) -> const Field &
                {
                    const auto found = std::find(checkpoint.names.begin(), checkpoint.names.end(), name);
                    DNDS_check_throw_info(found != checkpoint.names.end(), "time checkpoint payload is missing: " + name);
                    return checkpoint.fields[size_t(found - checkpoint.names.begin())];
                };
                _operations.Copy(trial, payload("trial"));
                Discard();
                _pendingDt = checkpoint.pendingDt;
                _startTime = checkpoint.startTime;
                _pendingSteady = checkpoint.pendingSteady;
                if (_ditr)
                {
                    _ditr->RestoreActive(trial, {checkpoint.ditrTime, checkpoint.ditrDt},
                                         payload("ditr_initial"), payload("ditr_middle"),
                                         payload("ditr_initial_rate"), payload("ditr_middle_rate"),
                                         payload("ditr_end_rate"), payload("ditr_res_other"));
                    _stagesActive = _stagePrepared = true;
                    _nextStage = 0;
                    _preparedState.emplace(trial);
                    _session = std::make_shared<const uint8_t>(0);
                    ++_stageGeneration;
                    return;
                }
                DNDS_check_throw_info(checkpoint.stagesActive && checkpoint.stagePrepared &&
                                          checkpoint.nextStage >= 0 && checkpoint.nextStage < GroupNumber(),
                                      "serialized time checkpoint has no prepared stage");
                checkpoint.equation.Validate();
                if (_bdf)
                    _bdf->RestoreActive(payload("time_initial"), payload("bdf_previous"), payload("time_source"),
                                        checkpoint.bdfHistory, checkpoint.bdfPreviousDt);
                else
                {
                    std::vector<std::reference_wrapper<const Field>> rates;
                    for (index stage = 0; stage < index(_dirk->Rates().size()); ++stage)
                        rates.push_back(std::cref(payload("dirk_rate_" + std::to_string(stage))));
                    _dirk->RestoreActive(payload("time_initial"), payload("time_source"), rates);
                }
                _stagesActive = true;
                _stagePrepared = true;
                _nextStage = checkpoint.nextStage;
                _preparedEquation = checkpoint.equation;
                _preparedState.emplace(trial);
                _preparedSource.emplace(Source());
                _session = std::make_shared<const uint8_t>(0);
                ++_stageGeneration; });
        }
    };
}

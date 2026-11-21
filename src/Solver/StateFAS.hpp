#pragma once

#include "DNDS/Algebra/ComponentSelection.hpp"
#include <algorithm>
#include <functional>
#include <cstring>
#include <memory>
#include <optional>

namespace DNDS::Multigrid
{
    template <class TState>
    struct StateFASLevel
    {
        /// Shallow retained state owner. Storage is distinct between levels;
        /// callbacks may bind its allocation before constructing the hierarchy.
        TState state;
        /// F(U), including physical mass when present. Own communication here;
        /// input ghosts may be refreshed but owned inputs remain unchanged.
        std::function<void(TState &, TState &)> apply;
        /// Relax F(U)=source, at most the requested number of corrections.
        /// Return actual corrections; throw on an unusable numerical result.
        std::function<index(TState &, const TState &, index)> smooth;
        /// Optional physical update policy (e.g. flow PP). Default is U+=delta.
        std::function<void(TState &, const TState &)> update;
        /// Required for a proper subset cycle. Solve only the selected
        /// equations with all remaining owned components frozen. Selection is
        /// normalized, nonempty and identical across levels. This is a
        /// principal-system smoother, not a full solve followed by masking.
        std::function<index(TState &, const TState &, index, const std::vector<index> &)> smoothSelected;
        /// Optional operator with a common linear term removed analytically.
        /// Used only by an identity-state edge declaring matching cancellation.
        /// The full apply/smoother still describes the actual level equation.
        std::function<void(TState &, TState &)> applyForIdentityTau;
    };
    template <class TState>
    struct StateFASTransfer
    {
        /// State and residual restriction are different operators in general.
        std::function<void(const TState &, TState &)> restrictState, restrictResidual;
        std::function<void(const TState &, TState &)> prolongCorrection;
        /// Caller proves restriction is identity in physical state and the
        /// removed linear term is identical on both levels. Both level tau
        /// operators must exist. This avoids subtracting large common U/dt.
        bool cancelIdentityLinearTerm = false;
    };
    struct FASCycleSettings
    {
        index preSmooth = 1, postSmooth = 1, coarseSmooth = 8;
        /// Empty selects the full state. Otherwise select distinct component
        /// indices, common to all levels; restriction still transfers the full
        /// background state and residual for nonlinear tau construction.
        std::vector<index> updateComponents;
        /// Optional counts indexed from the finest level, excluding the
        /// coarsest (which uses coarseSmooth). Empty retains the scalar policy.
        /// This expresses e.g. one fine and four intermediate corrections
        /// without multiplying the work on every upper level.
        std::vector<index> preSmoothByLevel, postSmoothByLevel;
    };
    /// One operation on the current level. Restrict moves the cursor to its child after
    /// forming the FAS source. Prolong applies the correction and returns to the parent.
    enum class FASOperation
    {
        PreSmooth,
        Restrict,
        CoarseSmooth,
        Prolong,
        PostSmooth
    };

    struct FASCycleReport
    {
        index cycles = 0, operatorEvaluations = 0;
        std::vector<index> corrections;
    };
    /// Full Approximation Scheme V-cycle over explicit levels and transfers.
    /// The hierarchy does not own mesh construction, equation physics, stopping
    /// criteria, time history or communicator policy. Operations supplies owned
    /// algebra, allocation/layout checks, binding/version checks and selected
    /// component access; it must reject unsupported backends. New mesh/coarsening families
    /// provide level operators and three transfer maps, not a new recursion.
    ///
    /// Callbacks complete synchronously and coordinate distributed errors before
    /// communication. All ranks use the same hierarchy/cycle order. LocalStage
    /// optionally coordinates local algebra failures. Caller serializes access,
    /// retains callback resources and releases conflicting views before Apply.
    /// The Operations contract is NewVector, Binding::Check, CheckCompatible,
    /// RequireDistinct, ComponentNumber, Copy/LinearCombination/Axpy, CheckFrozen,
    /// ProjectCorrection and AddSelected. It performs no hidden communication.
    /// Write provided allocations without rebinding outputs. A failed cycle may
    /// leave partial level states; restore the trial and rebuild before retry.
    template <class TState, class Operations>
    class StateFASCycle
    {
    public:
        using Field = TState;
        using LevelDefinition = StateFASLevel<Field>;
        using TransferDefinition = StateFASTransfer<Field>;
        using LocalStage = std::function<void(const std::function<void()> &)>;
        enum class ScratchKind
        {
            Value,
            Residual,
            Correction
        };
        struct Scratch
        {
            Field field;
            /// Keeps a reserved borrow alive through all consumers. Synchronous
            /// callbacks must release numerical views before this scope ends.
            std::shared_ptr<void> retention;
        };
        /// Optional prepared workspace adapter. Lane 0 is the current level,
        /// lane 1 its child. Acquisitions never span recursive descent.
        using AcquireScratch = std::function<Scratch(index, ScratchKind, index)>;

    private:
        Operations _operations;
        struct Level
        {
            LevelDefinition operation;
            Field source, reference;
            std::optional<Field> value, residual, correction;
            std::optional<Field> frozen;
            typename Operations::Binding binding;
            explicit Level(LevelDefinition input, bool pooled)
                : operation(std::move(input)), source(Operations::NewVector(operation.state)), reference(Operations::NewVector(operation.state)),
                  binding(operation.state)
            {
                if (!pooled)
                {
                    value.emplace(Operations::NewVector(operation.state));
                    residual.emplace(Operations::NewVector(operation.state));
                    correction.emplace(Operations::NewVector(operation.state));
                }
            }
        };
        std::vector<std::unique_ptr<Level>> _levels;
        std::vector<TransferDefinition> _transfers;
        FASCycleSettings _settings;
        LocalStage _local;
        AcquireScratch _acquireScratch;
        Scratch Acquire(index level, ScratchKind kind, index lane)
        {
            if (_acquireScratch)
            {
                auto result = _acquireScratch(level, kind, lane);
                Operations::CheckCompatible(result.field, _levels[level]->operation.state);
                for (const auto &other : _levels)
                {
                    Operations::RequireDistinct(result.field, other->operation.state);
                    Operations::RequireDistinct(result.field, other->source);
                    Operations::RequireDistinct(result.field, other->reference);
                    if (other->frozen)
                        Operations::RequireDistinct(result.field, *other->frozen);
                }
                return result;
            }
            auto &data = *_levels[level];
            return {*(kind == ScratchKind::Value ? data.value : kind == ScratchKind::Residual ? data.residual
                                                                                              : data.correction),
                    {}};
        }
        bool _failed = false;
        bool Selected() const { return !_settings.updateComponents.empty(); }
        void SaveFrozen(Level &level)
        {
            if (Selected())
                _operations.Copy(*level.frozen, level.operation.state);
        }
        void CheckFrozen(const Level &level) const
        {
            if (!Selected())
                return;
            _operations.CheckFrozen(level.operation.state, *level.frozen, _settings.updateComponents);
        }
        void ProjectCorrection(Field &correction) const
        {
            if (Selected())
                _operations.ProjectCorrection(correction, _settings.updateComponents);
        }
        template <class F>
        void RunLocal(F &&run) const
        {
            if (_local)
                _local(std::forward<F>(run));
            else
                run();
        }
        void Relax(index level, index corrections, FASCycleReport &report)
        {
            if (!corrections)
                return;
            auto &data = *_levels[level];
            if (Selected())
                RunLocal([&]
                         { SaveFrozen(data); });
            const index actual = Selected() ? data.operation.smoothSelected(data.operation.state, data.source, corrections, _settings.updateComponents)
                                            : data.operation.smooth(data.operation.state, data.source, corrections);
            RunLocal([&]
                     {
                DNDS_check_throw_info(actual >= 0 && actual <= corrections, "FAS smoother correction count is invalid");
                CheckBinding(data); CheckFrozen(data); });
            report.corrections[level] += actual;
        }
        static void CheckBinding(const Level &level)
        {
            level.binding.Check(level.operation.state);
        }
        void ApplyOperator(Level &level, Field &value, FASCycleReport &report, bool identityTau = false)
        {
            std::optional<typename Operations::Binding> input, output;
            RunLocal([&]
                     {
                CheckBinding(level);
                input.emplace(level.operation.state); output.emplace(value); });
            const auto &apply = identityTau ? level.operation.applyForIdentityTau : level.operation.apply;
            apply(level.operation.state, value);
            RunLocal([&]
                     {
                input->Check(level.operation.state, true); output->Check(value); });
            ++report.operatorEvaluations;
        }
        void Restrict(index level, FASCycleReport &report)
        {
            auto &fine = *_levels[level];
            auto &transfer = _transfers[level];
            auto &coarse = *_levels[level + 1];
            {
                auto fineValue = Acquire(level, ScratchKind::Value, 0);
                auto fineResidual = Acquire(level, ScratchKind::Residual, 0);
                auto coarseValue = Acquire(level + 1, ScratchKind::Value, 1);
                auto coarseResidual = Acquire(level + 1, ScratchKind::Residual, 1);
                ApplyOperator(fine, fineValue.field, report, transfer.cancelIdentityLinearTerm);
                if (!transfer.cancelIdentityLinearTerm)
                    RunLocal([&]
                             { _operations.LinearCombination(fineResidual.field, real(1), fine.source, real(-1), fineValue.field); });
                transfer.restrictState(fine.operation.state, coarse.operation.state);
                std::optional<typename Operations::Binding> sourceBinding, differenceBinding;
                RunLocal([&]
                         { CheckBinding(coarse); });
                RunLocal([&]
                         {
                _operations.Copy(coarse.reference, coarse.operation.state);
                sourceBinding.emplace(coarse.source);
                if (transfer.cancelIdentityLinearTerm) differenceBinding.emplace(coarseResidual.field); });
                ApplyOperator(coarse, coarseValue.field, report, transfer.cancelIdentityLinearTerm);
                if (transfer.cancelIdentityLinearTerm)
                {
                    transfer.restrictResidual(fineValue.field, coarseResidual.field);
                    RunLocal([&]
                             {
                    differenceBinding->Check(coarseResidual.field);
                    _operations.Axpy(coarseResidual.field, real(-1), coarseValue.field); });
                    transfer.restrictResidual(fine.source, coarse.source);
                    RunLocal([&]
                             {
                    sourceBinding->Check(coarse.source);
                    // Form the small spatial difference first, then add the
                    // history source once. Two separate additions to a large
                    // resOther can lose a representable net tau correction.
                    _operations.Axpy(coarse.source, real(-1), coarseResidual.field); });
                }
                else
                {
                    transfer.restrictResidual(fineResidual.field, coarse.source);
                    RunLocal([&]
                             {
                    sourceBinding->Check(coarse.source);
                    _operations.Axpy(coarse.source, real(1), coarseValue.field); });
                }
            } // Release temporary values before recursive child work.
            // F_c(U_c)=F_c(R U_f)+R(b_f-F_f(U_f)). The entrance reference
            // belongs to the incoming edge and is never overwritten by a
            // recursive child cycle, including in a hierarchy of 3+ levels.
        }

        void Prolong(index level)
        {
            auto &fine = *_levels[level];
            auto &coarse = *_levels[level + 1];
            auto &transfer = _transfers[level];
            {
                auto coarseCorrection = Acquire(level + 1, ScratchKind::Correction, 1);
                auto fineCorrection = Acquire(level, ScratchKind::Correction, 0);
                std::optional<typename Operations::Binding> correctionBinding;
                RunLocal([&]
                         {
                _operations.LinearCombination(coarseCorrection.field, real(1), coarse.operation.state, real(-1), coarse.reference);
                correctionBinding.emplace(fineCorrection.field); });
                transfer.prolongCorrection(coarseCorrection.field, fineCorrection.field);
                RunLocal([&]
                         {
                correctionBinding->Check(fineCorrection.field);
                ProjectCorrection(fineCorrection.field); SaveFrozen(fine); });
                if (fine.operation.update)
                    fine.operation.update(fine.operation.state, fineCorrection.field);
                else
                    RunLocal([&]
                             { _operations.AddSelected(fine.operation.state, fineCorrection.field, _settings.updateComponents); });

                RunLocal([&]
                         { CheckBinding(fine); CheckFrozen(fine); });
            } // Correction remains borrowed through physical PP/update.
        }
        index PreCount(index level) const
        {
            return _settings.preSmoothByLevel.empty() ? _settings.preSmooth : _settings.preSmoothByLevel[level];
        }
        index PostCount(index level) const
        {
            return _settings.postSmoothByLevel.empty() ? _settings.postSmooth : _settings.postSmoothByLevel[level];
        }
        void Cycle(index level, FASCycleReport &report)
        {
            if (level + 1 == index(_levels.size()))
            {
                Relax(level, _settings.coarseSmooth, report);
                return;
            }
            Relax(level, PreCount(level), report);
            Restrict(level, report);
            Cycle(level + 1, report);
            Prolong(level);
            Relax(level, PostCount(level), report);
        }

    public:
        StateFASCycle(std::vector<LevelDefinition> levels, std::vector<TransferDefinition> transfers, FASCycleSettings settings = {}, LocalStage local = {}, AcquireScratch acquireScratch = {}, Operations operations = {})
            : _operations(std::move(operations)), _transfers(std::move(transfers)), _settings(settings), _local(std::move(local)), _acquireScratch(std::move(acquireScratch))
        {
            RunLocal([&]
                     {
                DNDS_check_throw_info(!levels.empty() && _transfers.size() + 1 == levels.size() && settings.preSmooth >= 0 && settings.postSmooth >= 0 && settings.coarseSmooth > 0,
                                      "invalid FAS hierarchy or sweep controls");
                for (const auto *counts : {&settings.preSmoothByLevel, &settings.postSmoothByLevel})
                    DNDS_check_throw_info(counts->empty() ||
                                              (counts->size() == _transfers.size() &&
                                               std::all_of(counts->begin(), counts->end(), [](index count) { return count >= 0; })),
                                          "FAS per-level smoothing needs one nonnegative count per noncoarsest level");
                const index width = Operations::ComponentNumber(levels.front().state);
                _settings.updateComponents = Algebra::ComponentSelection(width, settings.updateComponents).Indices();
                _levels.reserve(levels.size());
                for (auto &level : levels)
                {
                    DNDS_check_throw_info(level.apply && level.smooth, "FAS level is missing an operator or smoother");
                    DNDS_check_throw_info(!Selected() || (Operations::ComponentNumber(level.state) == width && level.smoothSelected),
                                          "selected FAS needs equal level widths and principal-system smoothers");
                    Operations::CheckCompatible(level.state, level.state);
                    for (const auto &existing : _levels)
                        Operations::RequireDistinct(level.state, existing->operation.state);
                    _levels.push_back(std::make_unique<Level>(std::move(level), bool(_acquireScratch)));
                    if (Selected())
                        _levels.back()->frozen.emplace(Operations::NewVector(_levels.back()->operation.state));
                }
                for (const auto &transfer : _transfers)
                    DNDS_check_throw_info(transfer.restrictState && transfer.restrictResidual && transfer.prolongCorrection, "FAS edge is missing a transfer"); });
            RunLocal([&]
                     {
                for (index edge = 0; edge < index(_transfers.size()); ++edge)
                    DNDS_check_throw_info(!_transfers[edge].cancelIdentityLinearTerm ||
                                              (_levels[edge]->operation.applyForIdentityTau && _levels[edge + 1]->operation.applyForIdentityTau),
                                          "identity FAS cancellation requires both reduced operators"); });
        }
        StateFASCycle(const StateFASCycle &) = delete;
        StateFASCycle &operator=(const StateFASCycle &) = delete;
        const Field &State(index level = 0) const { return _levels.at(level)->operation.state; }
        /// Apply fixed V-cycles in place to the bound fine state. This reports
        /// work, not convergence; the physical caller checks its original
        /// unmodified finest equation. Coarse forcing stays internal.
        FASCycleReport Apply(const Field &source, index cycles = 1)
        {
            return ApplyWith(source, cycles, [&](auto &report)
                             { Cycle(0, report); });
        }

        /// Execute an externally ordered traversal. As with Apply, every domain rank must
        /// supply the same order. Validate the entire cursor walk before mutating fields.
        FASCycleReport ApplySchedule(const Field &source, const std::vector<FASOperation> &steps)
        {
            const index last = index(_levels.size()) - 1;
            RunLocal([&]
                     {
                index level = 0;
                bool smoothedCoarse = false;
                for (auto op : steps)
                {
                    switch (op)
                    {
                    case FASOperation::PreSmooth:
                    case FASOperation::PostSmooth:
                        DNDS_check_throw_info(level < last, "FAS pre/post smoother requires a finer level");
                        break;
                    case FASOperation::CoarseSmooth:
                        DNDS_check_throw_info(level == last, "FAS coarse smoother requires the coarsest level");
                        smoothedCoarse = true;
                        break;
                    case FASOperation::Restrict:
                        DNDS_check_throw_info(level < last, "FAS restriction has no child level");
                        ++level;
                        break;
                    case FASOperation::Prolong:
                        DNDS_check_throw_info(level > 0, "FAS prolongation has no parent level");
                        --level;
                        break;
                    default:
                        DNDS_check_throw_info(false, "unknown FAS operation");
                    }
                }
                DNDS_check_throw_info(level == 0 && smoothedCoarse, "FAS traversal must smooth the coarsest level and return to the finest"); });
            return ApplyWith(source, 1, [&](auto &report)
                             {
                index level = 0;
                for (auto op : steps)
                    switch (op)
                    {
                    case FASOperation::PreSmooth: Relax(level, PreCount(level), report); break;
                    case FASOperation::Restrict: Restrict(level++, report); break;
                    case FASOperation::CoarseSmooth: Relax(level, _settings.coarseSmooth, report); break;
                    case FASOperation::Prolong: Prolong(--level); break;
                    case FASOperation::PostSmooth: Relax(level, PostCount(level), report); break;
                    } });
        }

    private:
        template <class Traverse>
        FASCycleReport ApplyWith(const Field &source, index cycles, Traverse traverse)
        {
            RunLocal([&]
                     {
                Operations::CheckCompatible(State(), source);
                DNDS_check_throw_info(!_failed, "FAS callback failed; rebuild the hierarchy before reuse");
                DNDS_check_throw_info(cycles > 0, "invalid FAS cycle count");
                for (const auto &level : _levels)
                {
                    CheckBinding(*level);
                    Operations::RequireDistinct(source, level->operation.state);
                }
                _operations.Copy(_levels.front()->source, source); });
            FASCycleReport report;
            report.corrections.assign(_levels.size(), 0);
            try
            {
                for (index cycle = 0; cycle < cycles; ++cycle)
                {
                    traverse(report);
                    ++report.cycles;
                }
            }
            catch (...)
            {
                _failed = true;
                throw;
            }
            return report;
        }
    };
}

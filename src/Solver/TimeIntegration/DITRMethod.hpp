#pragma once

#include "Solver/TimeIntegration/DITRCoefficients.hpp"
#include "Solver/TimeIntegration/TimeEquation.hpp"
#include <functional>
#include <optional>

namespace DNDS::ODE
{
    /// DITR U2R2/U2R1 arithmetic and stage workspace, without a nonlinear loop.
    /// The controller visits intermediate (coupledStage=0), then endpoint (1),
    /// performing ONE correction or spatial V-cycle per visit. Completion
    /// refreshes that stage's RHS; stopping uses I+H after both updates, exactly
    /// as Euler's default thetaM2=0, thetaMMG=1 protocol. No temporal pMG here.
    ///
    /// Begin initializes all rate guesses with F(initial,t), as Euler does.
    /// The method owns independent initial, intermediate and rate buffers; the
    /// caller owns the endpoint trial throughout Begin..Discard. No accepted
    /// state/history is mutated. This method has no accepted-step history.
    /// Evaluate may refresh input ghosts but must preserve owned input and all
    /// bindings. It completes its output before returning. All calls are
    /// serialized; LocalAction optionally coordinates local failures before
    /// the caller's distributed evaluations. Arithmetic posts no hidden MPI.
    template <class State, class Operations>
    class DITRMethod
    {
    public:
        struct ActiveControl
        {
            real time = 0, dt = 0;
        };
        using Evaluate = std::function<void(State &, State &, real, int)>;
        using LocalAction = std::function<void(const std::function<void()> &)>;

    private:
        using Binding = typename Operations::Binding;
        Operations _operations;
        DITRCoefficients _coefficients;
        real _fraction, _theta, _time = 0, _inverseDt = 0, _dt = 0;
        LocalAction _local;
        State _initial, _middle, _initialRate, _middleRate, _endRate, _resOther;
        Binding _middleBinding;
        std::optional<Binding> _endpointBinding, _preparedState, _sourceBinding, _otherBinding, _otherRateBinding;
        bool _active = false, _prepared = false;
        int _nextCoupledStage = 0;

        template <class F>
        void Local(F &&action) const
        {
            if (_local)
                _local(std::forward<F>(action));
            else
                action();
        }
        void EvaluateChecked(const Evaluate &evaluate, State &state, State &rate, real time, int coupledStage)
        {
            const Binding input(state), output(rate);
            evaluate(state, rate, time, coupledStage);
            Local([&]
                  { input.Check(state, true); output.Check(rate); });
        }
        void CheckEndpoint(const State &endpoint) const
        {
            DNDS_check_throw_info(_active && _endpointBinding, "DITR method has no active trial");
            _endpointBinding->Check(endpoint);
        }

    public:
        DITRMethod(const State &prototype, real fraction = .55, real theta = .9146,
                   DITRVariant variant = DITRVariant::U2R2, LocalAction local = {}, Operations operations = {})
            : _operations(std::move(operations)), _coefficients(fraction, variant), _fraction(fraction), _theta(theta), _local(std::move(local)),
              _initial(Operations::NewVector(prototype)), _middle(Operations::NewVector(prototype)),
              _initialRate(Operations::NewVector(prototype)), _middleRate(Operations::NewVector(prototype)),
              _endRate(Operations::NewVector(prototype)), _resOther(Operations::NewVector(prototype)), _middleBinding(_middle)
        {
            DNDS_check_throw_info(std::isfinite(theta) && theta > 0, "DITR intermediate equation weight must be positive");
        }
        DITRMethod(const DITRMethod &) = delete;
        DITRMethod &operator=(const DITRMethod &) = delete;
        static constexpr int CoupledStageNumber() { return 2; }
        /// Stable storage for binding an independent physical-stage workspace.
        /// Caller may write its owned values only through a prepared equation.
        State &IntermediateState() { return _middle; }
        const State &IntermediateState() const { return _middle; }
        const State &InitialState() const { return _initial; }
        const State &InitialRate() const { return _initialRate; }
        const State &IntermediateRate() const { return _middleRate; }
        const State &EndpointRate() const { return _endRate; }
        const State &ResidualOther() const { return _resOther; }
        real CoupledStageTime(int coupledStage) const
        {
            DNDS_check_throw_info(
                _active && (coupledStage == 0 || coupledStage == 1),
                "DITR stage time needs an active midpoint or endpoint");
            return _time + (coupledStage == 0 ? _fraction : 1) * _dt;
        }
        ActiveControl CheckpointControl() const
        {
            DNDS_check_throw_info(_active && !_prepared && _nextCoupledStage == 0,
                                  "DITR checkpoint needs a completed coupled-iteration boundary");
            return {_time, _dt};
        }
        /// The physical RHS selected by this method for its prepared equation.
        /// In the first iteration this is the initial-time guess; subsequently
        /// it is the refreshed rate of this stage from the preceding update.
        const State &CurrentRate() const
        {
            DNDS_check_throw_info(_active && _prepared, "DITR rate needs a prepared coupled stage");
            return _nextCoupledStage == 0 ? _middleRate : _endRate;
        }
        void Begin(const State &initial, State &endpoint, real time, real dt, const Evaluate &evaluate)
        {
            Discard();
            Local([&]
                  {
                DNDS_check_throw_info(evaluate && std::isfinite(time) && std::isfinite(dt) && dt > 0 &&
                                          std::isfinite(1 / dt) && std::isfinite(time + dt) && time + dt > time,
                                      "invalid DITR method trial");
                _operations.CheckCompatible(_initial, initial);
                _operations.CheckCompatible(_initial, endpoint);
                _middleBinding.Check(_middle);
                Operations::RequireDistinct(initial, endpoint);
                Operations::RequireDistinct(_middle, endpoint);
                _operations.Copy(_initial, initial);
                _operations.Copy(endpoint, initial);
                _operations.Copy(_middle, initial);
                _endpointBinding.emplace(endpoint); });
            _time = time;
            _dt = dt;
            _inverseDt = 1 / dt;
            EvaluateChecked(evaluate, endpoint, _initialRate, time, 1);
            Local([&]
                  {
                _operations.Copy(_middleRate, _initialRate);
                _operations.Copy(_endRate, _initialRate); });
            _active = true;
        }

        /// Freeze resOther and the other stage until CompleteCoupledStage.
        /// Mutating the returned state is permitted; changing its binding or
        /// the other stage/rate is not. Outputs guarantee owned values only.
        CoupledStageEquation<State> PrepareCoupledStage(State &endpoint, int coupledStage)
        {
            Local([&]
                  {
                CheckEndpoint(endpoint);
                _middleBinding.Check(_middle);
                DNDS_check_throw_info(!_prepared && coupledStage == _nextCoupledStage && coupledStage < 2,
                                      "DITR coupled stages must be prepared/completed in intermediate-endpoint order");
                const auto &c = _coefficients.interpolation;
                const auto &w = _coefficients.integration;
                if (coupledStage == 0)
                {
                    _operations.PhysicalLinearCombination(_resOther, (c[0] + _theta) * _inverseDt, _initial,
                                                          (c[1] - _theta) * _inverseDt, endpoint);
                    _operations.Axpy(_resOther, c[2] + _theta * w[0], _initialRate);
                    _operations.Axpy(_resOther, c[3] + _theta * w[2], _endRate);
                    _otherBinding.emplace(endpoint); _otherRateBinding.emplace(_endRate);
                }
                else
                {
                    _operations.PhysicalStateAndRate(_resOther, _inverseDt, _initial, w[0], _initialRate);
                    _operations.Axpy(_resOther, w[1], _middleRate);
                    _otherBinding.emplace(_middle); _otherRateBinding.emplace(_middleRate);
                }
                _sourceBinding.emplace(_resOther);
                _preparedState.emplace(coupledStage == 0 ? _middle : endpoint); });
            _prepared = true;
            return {coupledStage == 0 ? _middle : endpoint, _resOther, {_inverseDt, coupledStage == 0 ? _theta * _coefficients.integration[1] : _coefficients.integration[2], _time + (coupledStage == 0 ? _fraction : 1) * _dt}, coupledStage};
        }
        void CompleteCoupledStage(State &endpoint, const Evaluate &evaluate)
        {
            Local([&]
                  {
                CheckEndpoint(endpoint);
                DNDS_check_throw_info(_prepared && evaluate, "DITR completion requires a prepared equation and rate evaluator");
                _sourceBinding->Check(_resOther, true);
                _preparedState->Check(_nextCoupledStage == 0 ? _middle : endpoint);
                _otherBinding->Check(_nextCoupledStage == 0 ? endpoint : _middle, true);
                _otherRateBinding->Check(_nextCoupledStage == 0 ? _endRate : _middleRate, true); });
            const bool middle = _nextCoupledStage == 0;
            EvaluateChecked(evaluate, middle ? _middle : endpoint, middle ? _middleRate : _endRate,
                            _time + (middle ? _fraction : 1) * _dt, _nextCoupledStage);
            _prepared = false;
            ++_nextCoupledStage;
        }
        /// I+H, after both corrections and their RHS refreshes. This is not
        /// max(norm(H),norm(I)); the controller alone selects norms/tolerances.
        /// Calling this closes one iteration and permits its next stage visit.
        void AssembleStoppingResidual(const State &endpoint, State &residual)
        {
            Local([&]
                  {
                CheckEndpoint(endpoint);
                DNDS_check_throw_info(!_prepared && _nextCoupledStage == 2, "DITR stopping residual needs both updated stages");
                for (const State *input : std::array<const State *, 6>{&_initial, &_middle, &endpoint, &_initialRate, &_middleRate, &_endRate})
                    Operations::RequireDistinct(residual, *input);
                const auto &c = _coefficients.interpolation;
                const auto &w = _coefficients.integration;
                _operations.PhysicalLinearCombination(residual, (1 + c[0]) * _inverseDt, _initial, (c[1] - 1) * _inverseDt, endpoint);
                _operations.PhysicalAxpy(residual, -_inverseDt, _middle);
                _operations.Axpy(residual, w[0] + c[2], _initialRate);
                _operations.Axpy(residual, w[1], _middleRate);
                _operations.Axpy(residual, w[2] + c[3], _endRate); });
            _nextCoupledStage = 0;
        }
        void RestoreActive(State &endpoint, ActiveControl control,
                           const State &initial, const State &middle,
                           const State &initialRate, const State &middleRate,
                           const State &endRate, const State &resOther)
        {
            Discard();
            Local([&]
                  {
                DNDS_check_throw_info(std::isfinite(control.time) && std::isfinite(control.dt) &&
                                          control.dt > 0 && std::isfinite(1 / control.dt) &&
                                          std::isfinite(control.time + control.dt) && control.time + control.dt > control.time,
                                      "invalid restored DITR control");
                _operations.CheckCompatible(_initial, endpoint);
                Operations::RequireDistinct(_middle, endpoint);
                _operations.Copy(_initial, initial);
                _operations.Copy(_middle, middle);
                _operations.Copy(_initialRate, initialRate);
                _operations.Copy(_middleRate, middleRate);
                _operations.Copy(_endRate, endRate);
                _operations.Copy(_resOther, resOther);
                _endpointBinding.emplace(endpoint); });
            _time = control.time;
            _dt = control.dt;
            _inverseDt = 1 / control.dt;
            _active = true;
            _prepared = false;
            _nextCoupledStage = 0;
        }
        /// Caller commits its endpoint only after acceptance/publication. No
        /// hidden time/history commit is required for this one-step method.
        void Discard() noexcept
        {
            _active = _prepared = false;
            _nextCoupledStage = 0;
            _endpointBinding.reset();
            _preparedState.reset();
            _sourceBinding.reset();
            _otherBinding.reset();
            _otherRateBinding.reset();
        }
    };
}

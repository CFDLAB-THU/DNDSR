#pragma once

#include "Solver/TimeIntegration/BDFCoefficients.hpp"
#include "Solver/TimeIntegration/DIRKTableau.hpp"
#include <type_traits>

namespace DNDS::ODE
{
    /// BDF-owned history and source arithmetic. The transaction facade validates
    /// stage ordering/bindings and commits only after accepted-state publication.
    /// No RHS cache, DIRK tableau, MPI or nonlinear iteration is owned here.
    template <class State, class Operations>
    class BDFMethod
    {
        static_assert(std::is_nothrow_swappable_v<State>);
        Operations _operations;
        int _order;
        State _initial, _previous, _source;
        real _previousDt = 0;
        bool _history = false;

        // Backends may attach allocation names without changing the field
        // interface. The history storage slots can exchange roles on commit.
        template <class Ops>
        static auto NewStorage(const State &prototype, const char *name, int)
            -> decltype(Ops::NewVector(prototype, name))
        {
            return Ops::NewVector(prototype, name);
        }
        template <class Ops>
        static State NewStorage(const State &prototype, const char *, long)
        {
            return Ops::NewVector(prototype);
        }

    public:
        BDFMethod(const State &prototype, int order, Operations operations = {})
            : _operations(std::move(operations)), _order(order),
              _initial(NewStorage<Operations>(prototype, "BDF/stateSlot0", 0)),
              _previous(NewStorage<Operations>(prototype, "BDF/stateSlot1", 0)),
              _source(NewStorage<Operations>(prototype, "BDF/resOther", 0))
        {
            DNDS_check_throw_info(order == 1 || order == 2, "unsupported BDF order");
        }
        const State &Source() const { return _source; }
        void Begin(const State &initial) { _operations.Copy(_initial, initial); }
        TimeEquationCoefficients Prepare(State &trial, real time, real dt, bool steady)
        {
            _operations.Copy(trial, _initial);
            const BDFCoefficients coefficients(_order, time, dt, _history ? _previousDt : 0, steady);
            coefficients.template AssembleResOther<Operations>(_source, _initial, &_previous, _operations);
            return coefficients.equation;
        }
        bool HasHistory() const { return _history; }
        int Order() const { return _order; }
        real PreviousDt() const { return _previousDt; }
        const State &InitialState() const { return _initial; }
        const State &PreparedSource() const { return _source; }
        const State &PreviousStorage() const { return _previous; }
        const State &PreviousState() const
        {
            DNDS_check_throw_info(_history, "no accepted BDF history");
            return _previous;
        }
        void Commit(real dt, bool steady) noexcept
        {
            if (steady)
                ResetHistory();
            else
            {
                std::swap(_previous, _initial);
                _previousDt = dt;
                _history = true;
            }
        }
        void ResetHistory() noexcept
        {
            _history = false;
            _previousDt = 0;
        }
        void RestoreHistory(const State &previous, real dt)
        {
            DNDS_check_throw_info(std::isfinite(dt) && dt > 0, "invalid restored BDF step");
            _operations.Copy(_previous, previous);
            _previousDt = dt;
            _history = true;
        }
        void RestoreActive(const State &initial, const State &previous, const State &source,
                           bool history, real previousDt)
        {
            DNDS_check_throw_info(!history || (std::isfinite(previousDt) && previousDt > 0),
                                  "invalid active BDF history");
            _operations.Copy(_initial, initial);
            _operations.Copy(_previous, previous);
            _operations.Copy(_source, source);
            _history = history;
            _previousDt = history ? previousDt : 0;
        }
    };

    /// DIRK-owned initial state, resOther and completed-stage RHS cache. No
    /// accepted-step history is allocated. Stage ordering and rate completion
    /// are enforced by the transaction facade before exposing the next source.
    template <class State, class Operations>
    class DIRKMethod
    {
        Operations _operations;
        DIRKTableau _tableau;
        State _initial, _source;
        std::vector<State> _rates;

    public:
        DIRKMethod(const State &prototype, const std::string &method, Operations operations = {})
            : _operations(std::move(operations)), _tableau(method), _initial(Operations::NewVector(prototype)), _source(Operations::NewVector(prototype))
        {
            _rates.reserve(_tableau.a.size());
            for (index stage = 0; stage < index(_tableau.a.size()); ++stage)
                _rates.push_back(Operations::NewVector(prototype));
        }
        int StageNumber() const { return int(_tableau.a.size()); }
        bool Implicit(int stage) const { return _tableau.a.at(stage).at(stage) != 0; }
        real StageTime(int stage, real time, real dt) const { return time + _tableau.c.at(stage) * dt; }
        const State &Source() const { return _source; }
        const State &InitialState() const { return _initial; }
        const std::vector<State> &Rates() const { return _rates; }
        State &Rate(int stage) { return _rates.at(stage); }
        void Begin(const State &initial) { _operations.Copy(_initial, initial); }
        TimeEquationCoefficients Prepare(State &trial, int stage, real time, real dt)
        {
            DNDS_check_throw_info(std::isfinite(time) && std::isfinite(dt) && dt > 0 &&
                                      std::isfinite(1 / dt) && std::isfinite(time + dt) && time + dt > time,
                                  "invalid DIRK timestep");
            const bool implicit = Implicit(stage); // Validate stage before writing.
            _operations.Copy(trial, _initial);
            const real inverseDt = 1 / dt;
            if (implicit)
            {
                _operations.PhysicalCopyScale(_source, inverseDt, _initial);
                for (int previous = 0; previous < stage; ++previous)
                    if (_tableau.a[stage][previous] != 0)
                        _operations.Axpy(_source, _tableau.a[stage][previous], _rates[previous]);
            }
            return {implicit ? inverseDt : 0, _tableau.a[stage][stage], StageTime(stage, time, dt)};
        }
        void RestoreActive(const State &initial, const State &source,
                           const std::vector<std::reference_wrapper<const State>> &rates)
        {
            DNDS_check_throw_info(rates.size() == _rates.size(), "DIRK checkpoint rate count differs");
            _operations.Copy(_initial, initial);
            _operations.Copy(_source, source);
            for (index stage = 0; stage < index(_rates.size()); ++stage)
                _operations.Copy(_rates[size_t(stage)], rates[size_t(stage)].get());
        }
    };
}

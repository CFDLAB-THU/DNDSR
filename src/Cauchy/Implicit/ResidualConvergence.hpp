#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Physics/EquationSchema.hpp"

namespace DNDS::Cauchy
{
    /// Small rank-local policy over already globally reduced component norms.
    /// All components remain observable; selection only controls acceptance.
    /// One instance belongs to one solve, including a finest FAS solve. It
    /// performs no MPI and never changes residuals, corrections or tolerances.
    template <class Norms>
    class BasicResidualConvergence
    {
        const ImplicitSettings &_settings;
        Norms _initial{}, _reference{};
        std::vector<index> _components;
        bool _started = false, _accepted = false;
        real _ratio = 0;

    public:
        /// Settings must be validated and outlive this policy.
        explicit BasicResidualConvergence(const ImplicitSettings &settings, const EquationSchema &schema, Norms prototype = {})
            : _settings(settings), _initial(prototype), _reference(prototype), _components(schema.ResolveComponents(
                                                                                   settings.ResolveResidualComponents(schema.AccumulationKind() == PhysicalAccumulationKind::ConstantDensityVelocityPressure)))
        {
            index width = 0;
            for (const auto &block : schema.blocks)
                width += index(block.components.size());
            DNDS_check_throw_info(width == index(prototype.size()), "residual norms differ from equation schema");
        }
        void Observe(const Norms &residual)
        {
            DNDS_check_throw_info(residual.size() == _initial.size(), "residual component count changed");
            for (auto value : residual)
                DNDS_check_throw_info(std::isfinite(value) && value >= 0, "invalid global residual norm");
            if (!_started)
                _initial = _reference = residual;
            else if (_settings.residualReference == "runningMaximum")
                for (index j = 0; j < index(residual.size()); ++j)
                    _reference[j] = std::max(_reference[j], residual[j]);
            _started = true;
            _accepted = true;
            _ratio = 0;
            const index count = _components.empty() ? index(residual.size()) : index(_components.size());
            for (index iComponent = 0; iComponent < count; ++iComponent)
            {
                const index j = _components.empty() ? iComponent : _components[iComponent];
                const real target = std::max(_settings.absoluteTolerance, _settings.relativeTolerance * _reference[j]);
                _accepted = _accepted && residual[j] <= target;
                // A zero requested tolerance means exact zero. Keep diagnostics
                // finite rather than serializing an infinity as JSON null.
                const real ratio = target > 0 ? residual[j] / target : residual[j] == 0 ? 0
                                                                                        : std::numeric_limits<real>::max();
                _ratio = std::max(_ratio, std::min(ratio, std::numeric_limits<real>::max()));
            }
        }
        void Restore(const Norms &initial, const Norms &reference)
        {
            DNDS_check_throw_info(initial.size() == _initial.size() && reference.size() == _reference.size(),
                                  "restored residual reference shape differs");
            for (index j = 0; j < index(initial.size()); ++j)
                DNDS_check_throw_info(std::isfinite(initial[j]) && initial[j] >= 0 &&
                                          std::isfinite(reference[j]) && reference[j] >= initial[j],
                                      "invalid restored residual reference");
            _initial = initial;
            _reference = reference;
            _started = true;
            _accepted = false;
            _ratio = 0;
        }
        const Norms &Initial() const { return _initial; }
        const Norms &Reference() const { return _reference; }
        bool Accepted() const { return _started && _accepted; }
        real TargetRatio() const { return _ratio; }
    };
    using ResidualConvergence = BasicResidualConvergence<FlowState>;
}

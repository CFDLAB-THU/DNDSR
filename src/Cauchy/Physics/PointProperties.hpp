#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"
#include "Physics/Thermodynamics/ConservativeConversion.hpp"
#include "Physics/Thermodynamics/IdealMixtureGradient.hpp"
#include <limits>

namespace DNDS::Cauchy
{
    /// Scalars own their values. Species views borrow the evaluator's worker
    /// scratch until its next point or destruction; no layout/Array assumption.
    struct PointProperties
    {
        Physics::Thermodynamics::Properties thermo;
        Physics::Transport::Properties transport;
        Physics::Chemistry::ConstSpeciesBufferView fractions, diffusivity, speciesEnthalpy, speciesGasConstant;
        Physics::Chemistry::ConstSpeciesBufferView chemicalSource;
        Physics::Chemistry::ConstJacobianBufferView sourceJacobian;
        /// NaN when viscous mixture diffusivity was not requested.
        real diffusionBound = std::numeric_limits<real>::quiet_NaN();
    };

    struct PrimitiveDerivative
    {
        /// Directional derivative of [rho,u,v,w,T], in code units.
        FlowState rhoVelocityTemperature;
        real pressure;
    };
    /// Pure directional chain rule at a ready EOS point. Input/output can be
    /// gathered/scattered through any layout. Species derivatives are Ns-1
    /// raw d(rhoY), with operator[] (a strided row span is sufficient). Mixtures
    /// require full species enthalpies requested when making the evaluator.
    /// Retain its worker/material and consume the borrowed point before the next
    /// evaluation. Caller supplies a point from this state and prevalidates the
    /// species extent; no implicit property refresh, repair or gamma closure.
    /// Extent assertions are programming preconditions, not numerical fallback
    /// signals; failed EOS evaluations are handled before this arithmetic call.
    template <class TSpeciesDerivative>
    PrimitiveDerivative PreparePrimitiveDerivative(const FlowState &state, const FlowState &derivative,
                                                   const PointProperties &point, const TSpeciesDerivative &speciesDerivative)
    {
        namespace T = Physics::Thermodynamics;
        const auto velocity = T::ConservativeVelocity<3>(state);
        const real internalEnergy = T::ConservativeSpecificInternalEnergy<3>(state);
        const real internalDerivative = T::ConservativeInternalEnergyDensityDerivative<3>(state, derivative);
        const int species = point.fractions.nSpecies;
        DNDS_assert_info(!species || (point.speciesEnthalpy.nSpecies == species && point.speciesGasConstant.nSpecies == species),
                         "primitive mixture derivative needs full species enthalpies and gas constants");
        struct SpeciesEnergy
        {
            const PointProperties &point;
            real operator[](int k) const { return point.speciesEnthalpy[k] - point.speciesGasConstant[k] * point.thermo.temperature; }
        };
        const real temperatureDerivative = T::IdealMixtureTemperatureDerivative(state[0], internalEnergy, point.thermo.cv,
                                                                                derivative[0], internalDerivative, point.fractions, SpeciesEnergy{point}, speciesDerivative, species);
        PrimitiveDerivative result;
        result.rhoVelocityTemperature[0] = derivative[0];
        for (int d = 0; d < 3; ++d)
            result.rhoVelocityTemperature[1 + d] = T::ConservativeSpecificDerivative(state[0], velocity[d], derivative[0], derivative[1 + d]);
        result.rhoVelocityTemperature[4] = temperatureDerivative;
        result.pressure = T::IdealMixturePressureDerivative(state[0], point.thermo.temperature, point.thermo.gasConstant,
                                                            derivative[0], temperatureDerivative, point.fractions, point.speciesGasConstant, speciesDerivative, species);
        return result;
    }

    /// CPU point producer shared by explicit derived refresh and fused/tiled
    /// operator work. Owns one exclusive chemistry worker for mixtures. Names,
    /// backend, units and material compatibility are resolved before the loop.
    /// Caller serializes use; move this evaluator into asynchronous CPU work.
    class PointEvaluator
    {
        friend class EquationSystem;
        std::optional<PerfectGas> _gas;
        bool _viscous;
        std::optional<Physics::Thermodynamics::IdealGasMixture::PointEvaluator> _mixture;
        Physics::Thermodynamics::MixturePropertyRequest _request;
        PointEvaluator(const EquationSystem &equation, Physics::Chemistry::ChemicalWorkerPool::Lease worker,
                       Physics::Thermodynamics::MixturePropertyRequest request);

    public:
        PointEvaluator(const PointEvaluator &) = delete;
        PointEvaluator &operator=(const PointEvaluator &) = delete;
        PointEvaluator(PointEvaluator &&) noexcept = default;
        PointEvaluator &operator=(PointEvaluator &&) noexcept = default;
        Physics::Chemistry::SpeciesBufferView SpeciesInput() const;
        /// Perfect-gas profile parameter for setup-validated analytic boundary
        /// producers. Mixtures must use an equation-specific profile adapter.
        const PerfectGas &PerfectGasModel() const
        {
            DNDS_assert_info(_gas, "point evaluator has no perfect-gas material");
            return *_gas;
        }
        /// Construct flow5 from [rho,u,v,w,p] and Ns-1 conservative species.
        /// A positive temperature overrides p; zero selects the pressure input.
        /// Full thermochemical energy is retained for mixtures. The species
        /// buffer is borrowed, never modified, and may be this worker's input
        /// scratch. No new worker or point-time payload allocation is needed.
        /// Numerical adapter; failed values must not be consumed. External EOS
        /// exceptions are intercepted here and are never a fallback signal.
        NumericalResult<FlowState> TryFromPrimitive(const FlowState &primitive,
                                                    Physics::Chemistry::ConstSpeciesBufferView rhoY = {}, real temperature = 0) noexcept;
        FlowState FromPrimitive(const FlowState &primitive, Physics::Chemistry::ConstSpeciesBufferView rhoY = {},
                                real temperature = 0);
        /// Shared total-state inlet closure, with fixed composition. Returns
        /// accepted speed after the Euler enthalpy cap; inputs use code units.
        NumericalResult<Physics::Thermodynamics::TotalConditionStaticState> TryTotalToStatic(
            real pressure, real temperature, real speedSquared,
            Physics::Chemistry::ConstSpeciesBufferView independentFractions = {}) noexcept;
        /// Nonthrowing numerical property entry. All expected validation and
        /// backend failures return an explicit code. output is valid only on
        /// success; its species spans still borrow this worker until next use.
        NumericalStatus TryEvaluate(const FlowState &flow, Physics::Chemistry::ConstSpeciesBufferView rhoY,
                                    real temperatureGuess, Physics::Thermodynamics::MixturePropertyRequest request,
                                    PointProperties &output) noexcept;
        NumericalStatus TryEvaluate(const FlowState &flow, Physics::Chemistry::ConstSpeciesBufferView rhoY,
                                    real temperatureGuess, PointProperties &output) noexcept
        {
            return TryEvaluate(flow, rhoY, temperatureGuess, _request, output);
        }
        /// Input flow is [rho,rhou,rhov,rhow,rhoE], with full thermal energy for
        /// mixtures. rhoY has Ns-1 independent conservative species; it may be
        /// worker scratch gathered from any physical layout or allocation.
        /// A stale positive T guess seeds UV inversion, never its final answer.
        /// Checked setup/compatibility adapter; numerical kernels use TryEvaluate.
        PointProperties Evaluate(const FlowState &flow, Physics::Chemistry::ConstSpeciesBufferView rhoY = {},
                                 real temperatureGuess = 0);
        /// Per-point subset of the reserved outputs; intended for fused FV
        /// face/source phases sharing one explicit worker context.
        PointProperties Evaluate(const FlowState &flow, Physics::Chemistry::ConstSpeciesBufferView rhoY,
                                 real temperatureGuess, Physics::Thermodynamics::MixturePropertyRequest request);
        /// Numerical fluxes consume these actual EOS values; no gamma closure.
        static Riemann::AcousticState FluxState(const FlowState &flow, const PointProperties &point)
        {
            return {{flow, point.thermo.pressure}, point.thermo.soundSpeed};
        }
    };
}

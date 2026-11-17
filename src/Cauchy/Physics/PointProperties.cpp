#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include <new>
#include <limits>

namespace DNDS::Cauchy
{
    NumericalResult<Physics::Thermodynamics::TotalConditionStaticState> PointEvaluator::TryTotalToStatic(
        real pressure, real temperature, real speedSquared,
        Physics::Chemistry::ConstSpeciesBufferView independentFractions) noexcept
    {
        if (_mixture)
            return _mixture->TryTotalToStatic(pressure, temperature, speedSquared, independentFractions);
        if (!_gas || independentFractions.nSpecies)
            return NumericalStatus{NumericalError::InvalidState, 1, -1};
        return Physics::Thermodynamics::TotalToStaticPerfectGas(
            pressure, temperature, speedSquared, _gas->gasConstant, _gas->gamma);
    }
    PointEvaluator::PointEvaluator(const EquationSystem &equation,
                                   Physics::Chemistry::ChemicalWorkerPool::Lease worker, Physics::Thermodynamics::MixturePropertyRequest request)
        : _viscous(equation.Viscous()), _request(request)
    {
        if (equation.HasMixture())
        {
            request.diffusivity = _viscous && request.diffusivity;
            request.transport = _viscous && (request.transport || request.diffusivity);
            _request = request;
            _mixture.emplace(equation.Mixture().Borrow(std::move(worker), request));
        }
        else
        {
            DNDS_check_throw_info(!worker, "perfect gas requires no chemistry worker");
            DNDS_check_throw_info(!request.chemicalSource && !request.sourceJacobian, "perfect gas has no chemical source profile");
            if (!equation.IsACM())
                _gas = equation.Gas();
        }
    }
    Physics::Chemistry::SpeciesBufferView PointEvaluator::SpeciesInput() const
    {
        return _mixture ? _mixture->SpeciesInput() : Physics::Chemistry::SpeciesBufferView{};
    }
    FlowState PointEvaluator::FromPrimitive(const FlowState &primitive,
                                            Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperature)
    {
        const auto result = TryFromPrimitive(primitive, rhoY, temperature);
        DNDS_check_throw_info(result.status, "primitive property evaluation failed");
        return result.value;
    }
    NumericalResult<FlowState> PointEvaluator::TryFromPrimitive(const FlowState &primitive,
                                                                Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperature) noexcept
    {
        // Adapter boundary: third-party thermodynamics may throw. No numerical
        // kernel catches exceptions, and these failures never request fallback.
        try
        {
            if (!(std::isfinite(temperature) && temperature >= 0 && primitive[0] > 0 &&
                  std::all_of(primitive.begin(), primitive.end(), [](real value)
                              { return std::isfinite(value); }) &&
                  (temperature > 0 || primitive[4] > 0)))
                return NumericalStatus{NumericalError::InvalidState, 1, -1};
            if (!_mixture)
            {
                if (!_gas || rhoY.nSpecies)
                    return NumericalStatus{NumericalError::InvalidState, 2, -1};
                auto value = primitive;
                if (temperature > 0)
                    value[4] = value[0] * _gas->gasConstant * temperature;
                return _gas->Conservative(value);
            }
            const auto point = temperature > 0 ? _mixture->FromRhoT(primitive[0], temperature, rhoY)
                                               : _mixture->FromRhoP(primitive[0], primitive[4], rhoY);
            const Vector3 velocity{primitive[1], primitive[2], primitive[3]};
            return FlowState{primitive[0], primitive[0] * velocity[0], primitive[0] * velocity[1], primitive[0] * velocity[2],
                             Physics::Thermodynamics::ConservativeTotalEnergyDensity<3>(primitive[0], velocity, point.thermo.internalEnergy)};
        }
        catch (const std::bad_alloc &)
        {
            return NumericalStatus{NumericalError::AllocationFailure};
        }
        catch (...)
        {
            return NumericalStatus{NumericalError::BackendFailure};
        }
    }

    PointProperties PointEvaluator::Evaluate(const FlowState &flow,
                                             Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess)
    {
        return Evaluate(flow, rhoY, temperatureGuess, _request);
    }
    PointProperties PointEvaluator::Evaluate(const FlowState &flow,
                                             Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess,
                                             Physics::Thermodynamics::MixturePropertyRequest request)
    {
        PointProperties output;
        const auto status = TryEvaluate(flow, rhoY, temperatureGuess, request, output);
        DNDS_check_throw_info(status, fmt::format("point evaluation failed (code={}, detail={})",
                                                  int(status.code), status.detail));
        return output;
    }
    NumericalStatus PointEvaluator::TryEvaluate(const FlowState &flow,
                                                Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess,
                                                Physics::Thermodynamics::MixturePropertyRequest request,
                                                PointProperties &output) noexcept
    {
        using Physics::EvaluationError;
        using Physics::EvaluationFailure;
        try
        {
            // PP POLICY -- rhoY is a borrowed raw conservative point value. The
            // mixture evaluator clips/normalizes it only into worker-local mass
            // fractions for EOS, transport and temperature inversion. It never
            // writes that temporary composition back to reconstruction coefficients,
            // cell means, halos, or the species values used by conservative fluxes.
            request.diffusivity = _viscous && request.diffusivity;
            request.transport = _viscous && (request.transport || request.diffusivity);
            if ((request.transport && !_request.transport) || (request.diffusivity && !_request.diffusivity) ||
                (request.speciesEnthalpy && !_request.speciesEnthalpy) || (request.sourceJacobian && !_request.sourceJacobian) ||
                ((request.chemicalSource || request.sourceJacobian) && !_request.chemicalSource && !_request.sourceJacobian))
                return EvaluationError(NumericalError::UnsupportedRequest, EvaluationFailure::Request);
            PointProperties result;
            if (!_mixture)
            {
                if (!_gas || rhoY.nSpecies)
                    return EvaluationError(NumericalError::InvalidInput, EvaluationFailure::Composition);
                const auto point = TryPerfectGasPoint(*_gas, flow, request.transport);
                if (!point)
                    return point.status;
                result.thermo = point.value.thermo;
                result.transport = point.value.transport;
                result.diffusionBound = point.value.diffusionBound;
                output = result;
                return {};
            }
            Physics::Thermodynamics::MixturePoint point;
            const auto status = _mixture->TryFromConservative(flow[0], flow[4], {flow[1], flow[2], flow[3]},
                                                              rhoY, temperatureGuess, request, point);
            if (!status)
                return status;
            result.thermo = point.thermo;
            result.transport = _viscous ? point.transport : Physics::Transport::Properties{0, 0};
            result.fractions = point.fractions;
            result.diffusivity = point.diffusivity;
            result.speciesEnthalpy = point.speciesEnthalpy;
            result.speciesGasConstant = point.speciesGasConstant;
            result.chemicalSource = point.chemicalSource;
            result.sourceJacobian = point.sourceJacobian;
            if (!_viscous)
                result.diffusionBound = 0;
            if (_viscous && request.diffusivity)
            {
                result.diffusionBound = std::max(real(4. / 3) * result.transport.viscosity / flow[0],
                                                 result.transport.conductivity / (flow[0] * result.thermo.cv));
                for (int k = 0; k < result.diffusivity.nSpecies; ++k)
                    result.diffusionBound = std::max(result.diffusionBound, result.diffusivity[k]);
            }
            output = result;
            return {};
        }
        catch (const std::bad_alloc &)
        {
            return EvaluationError(NumericalError::AllocationFailure, EvaluationFailure::Properties);
        }
        catch (...)
        {
            return EvaluationError(NumericalError::UnexpectedFailure, EvaluationFailure::Properties);
        }
    }
}

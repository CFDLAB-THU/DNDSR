#pragma once

#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "Physics/Thermodynamics/ConservativeConversion.hpp"

namespace DNDS::Cauchy
{
    /** Perfect-gas specialization of PrepareViscousFlux, using the same
     * conservative chain rule as the generic CPU operator. Transport/EOS have
     * already been evaluated at state. Turbulent coefficients are explicit;
     * this function neither selects a RANS model nor performs a point repair.
     * Positive viscous flux is subtracted from the convective normal flux.
     */
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FlowStateT<Scalar> PreparePerfectGasViscousFlux(
        const FlowStateT<Scalar> &state, const FlowGradientT<Scalar> &gradient, const PerfectGasPointT<Scalar> &point,
        const Vector3T<Scalar> &normal, Scalar turbulentViscosity, Scalar turbulentConductivity,
        bool adiabatic, bool useQCR) noexcept
    {
        namespace T = Physics::Thermodynamics;
        const auto velocity = T::ConservativeVelocity<3>(state);
        const Scalar internalEnergy = T::ConservativeSpecificInternalEnergy<3>(state);
        const Scalar viscosity = point.transport.viscosity + turbulentViscosity;
        const Scalar conductivity = adiabatic ? 0 : point.transport.conductivity + turbulentConductivity;
        struct VelocityGradient
        {
            std::array<Vector3T<Scalar>, 3> values{};
            DNDS_DEVICE_CALLABLE Scalar operator()(int i, int d) const { return values[i][d]; }
        };
        VelocityGradient derivative;
        Vector3T<Scalar> temperatureDerivative{};
        for (int d = 0; d < 3; ++d)
        {
            for (int i = 0; i < 3; ++i)
                derivative.values[i][d] = T::ConservativeSpecificDerivative(state[0], velocity[i], gradient[0][d], gradient[1 + i][d]);
            if (conductivity != 0)
            {
                FlowStateT<Scalar> direction;
                for (int j = 0; j < 5; ++j)
                    direction[j] = gradient[j][d];
                const Scalar internalDerivative = T::ConservativeInternalEnergyDensityDerivative<3>(state, direction);
                temperatureDerivative[d] = (internalDerivative - internalEnergy * gradient[0][d]) / (state[0] * point.thermo.cv);
            }
        }
        const auto flux = Physics::Transport::NewtonianFourierNormalFlux(velocity, derivative, temperatureDerivative, normal,
                                                                         viscosity, conductivity, useQCR, turbulentViscosity / (viscosity + DNDS::Math::ScalarMax(Scalar(verySmallReal), DNDS::Math::ScalarMinPositive<Scalar>())));
        return {0, flux.traction[0], flux.traction[1], flux.traction[2], flux.energy};
    }

    /** @brief Conservative polynomial face averaging and jump correction.
     * @details Both sides and gradients are in one frame. An explicit wall
     * state, when supplied, replaces the thermodynamic face average after the
     * derivative construction; it does not replace those derivatives. This
     * flow5 helper uses molecular Newtonian/Fourier transport only. It performs
     * no primitive-gradient reconstruction, RANS augmentation or QCR correction.
     * The input state/geometry contracts are validated outside the point loop;
     * a failed EOS after arithmetic narrowing is returned without repair.
     */
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline NumericalResult<FlowStateT<Scalar>> PreparePerfectGasPolynomialViscousFlux(
        const PerfectGasT<Scalar> &gas, const FlowStateT<Scalar> &left, const FlowStateT<Scalar> &right,
        const FlowGradientT<Scalar> &leftGradient, const FlowGradientT<Scalar> &rightGradient,
        const Vector3T<Scalar> &normal, Scalar inverseDistance, bool adiabatic,
        bool hasFaceState = false, const FlowStateT<Scalar> &faceState = {}) noexcept
    {
        FlowStateT<Scalar> average;
        FlowGradientT<Scalar> gradient;
        for (int j = 0; j < 5; ++j)
        {
            average[j] = Scalar(.5) * (left[j] + right[j]);
            for (int d = 0; d < 3; ++d)
                gradient[j][d] = Scalar(.5) * (leftGradient[j][d] + rightGradient[j][d]) +
                                 normal[d] * (right[j] - left[j]) * inverseDistance;
        }
        if (hasFaceState)
            average = faceState;
        const auto point = TryPerfectGasPoint(gas, average, true);
        if (!point)
            return point.status;
        return PreparePerfectGasViscousFlux(average, gradient, point.value, normal, Scalar(0), Scalar(0), adiabatic, false);
    }

}

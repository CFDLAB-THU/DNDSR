#pragma once

#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"

namespace DNDS::Cauchy
{
    /// Value owner for optional model-prepared data. Input() creates borrowed
    /// pointers after this owner reaches its final location; copies/moves never
    /// leave embedded pointers referring to an old stack object.
    template <class Scalar>
    struct PreparedRiemannDataT
    {
        Riemann::AcousticStateT<Scalar> leftMean, rightMean;
        Riemann::RoeDataT<Scalar> roe;
        Riemann::WaveBoundsT<Scalar> outerWaves;
        Riemann::AUSMDataT<Scalar> ausm;
        Riemann::Requirements required;
        DNDS_DEVICE_CALLABLE Riemann::FluxInputT<Scalar> Input(Riemann::AcousticStateT<Scalar> left, Riemann::AcousticStateT<Scalar> right,
                                                               Vector3T<Scalar> normal, Vector3T<Scalar> gridVelocity = {}) const
        {
            return {left, right, normal, gridVelocity, required.meanStates ? &leftMean : nullptr, required.meanStates ? &rightMean : nullptr,
                    required.roe ? &roe : nullptr, required.outerWaveBounds ? &outerWaves : nullptr, required.ausmFaceSpeed ? &ausm : nullptr};
        }
    };
    using PreparedRiemannData = PreparedRiemannDataT<real>;
    /** @brief Perfect-gas Roe, shock and AUSM physical input producer.
     * @details Host/device shared arithmetic. Inputs are already admissible,
     * in code units and one common Cartesian frame. This is the physics side
     * of the gas-agnostic Riemann boundary: it supplies enthalpy, wave estimates
     * and the acoustic scale, without allocating or acquiring any service.
     * Keep the returned owner alive while its Input() view is consumed.
     */
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline PreparedRiemannDataT<Scalar> PreparePerfectGasRiemann(
        const PerfectGasT<Scalar> &gas, Riemann::Solver solver,
        const Riemann::AcousticStateT<Scalar> &left, const Riemann::AcousticStateT<Scalar> &right,
        const Riemann::AcousticStateT<Scalar> &leftMeanData, const Riemann::AcousticStateT<Scalar> &rightMeanData,
        const Vector3T<Scalar> &normal, const Vector3T<Scalar> &gridVelocity = {}) noexcept
    {
        PreparedRiemannDataT<Scalar> data{};
        data.required = Riemann::RequiredInputs(solver);
        if (!data.required.meanStates)
            return data;
        data.leftMean = leftMeanData;
        data.rightMean = rightMeanData;
        const auto &leftMean = leftMeanData.flow.conservative, &rightMean = rightMeanData.flow.conservative;
        const auto velocity = [](const FlowStateT<Scalar> &state)
        { Vector3T<Scalar> value; for (int d=0;d<3;++d) value[d]=state[1+d]/state[0]; return value; };
        const auto dot = [](const Vector3T<Scalar> &a, const Vector3T<Scalar> &b)
        { Scalar sum=0; for(int d=0;d<3;++d) sum+=a[d]*b[d];return sum; };
        if (data.required.roe)
        {
            const Scalar rootL = DNDS::Math::ScalarSqrt(leftMean[0]), rootR = DNDS::Math::ScalarSqrt(rightMean[0]), sum = rootL + rootR;
            auto &roe = data.roe;
            const auto vL = velocity(leftMean), vR = velocity(rightMean);
            for (int d = 0; d < 3; ++d)
                roe.velocity[d] = (rootL * vL[d] + rootR * vR[d]) / sum;
            const Scalar hL = (leftMean[4] + data.leftMean.flow.pressure) / leftMean[0], hR = (rightMean[4] + data.rightMean.flow.pressure) / rightMean[0];
            roe.enthalpy = (rootL * hL + rootR * hR) / sum;
            const Scalar speedSquared = dot(roe.velocity, roe.velocity), denominator = roe.enthalpy - Scalar(.5) * speedSquared;
            roe.soundSpeed = DNDS::Math::ScalarSqrt((gas.gamma - 1) * denominator);
            roe.contactEnergy = Scalar(.5) * speedSquared;
            roe.contactProjection = {(roe.enthalpy - speedSquared) / denominator, roe.velocity[0] / denominator, roe.velocity[1] / denominator, roe.velocity[2] / denominator, -1 / denominator};
        }
        if (data.required.outerWaveBounds)
        {
            const auto vL = velocity(leftMean), vR = velocity(rightMean);
            const Scalar unL = dot(vL, normal) - dot(gridVelocity, normal), unR = dot(vR, normal) - dot(gridVelocity, normal);
            const Scalar rho = DNDS::Math::ScalarSqrt(leftMean[0] * rightMean[0]);
            const Scalar star = DNDS::Math::ScalarMax(Scalar(0), Scalar(.5) * (data.leftMean.flow.pressure + data.rightMean.flow.pressure) - Scalar(.5) * (unL - unR) * rho * data.roe.soundSpeed);
            const auto shock = [&](Scalar pressure)
            { return star <= pressure ? Scalar(1) : DNDS::Math::ScalarSqrt(1 + (gas.gamma + 1) / (2 * gas.gamma) * (star / pressure - 1)); };
            data.outerWaves = {unL - data.leftMean.soundSpeed * shock(data.leftMean.flow.pressure), unR + data.rightMean.soundSpeed * shock(data.rightMean.flow.pressure)};
        }
        if (data.required.ausmFaceSpeed)
        {
            auto vL = velocity(left.flow.conservative), vR = velocity(right.flow.conservative);
            for (int d = 0; d < 3; ++d)
            {
                vL[d] -= gridVelocity[d];
                vR[d] -= gridVelocity[d];
            }
            const Scalar unL = dot(vL, normal), unR = dot(vR, normal), factor = 2 * (gas.gamma - 1) / (gas.gamma + 1);
            const Scalar hL = gas.gamma / (gas.gamma - 1) * left.flow.pressure / left.flow.conservative[0], hR = gas.gamma / (gas.gamma - 1) * right.flow.pressure / right.flow.conservative[0];
            if (solver == Riemann::Solver::AUSMPWPlus)
            {
                const Scalar aSquared = Scalar(.5) * factor * (hL + Scalar(.5) * unL * unL + hR + Scalar(.5) * unR * unR);
                data.ausm.faceSoundSpeed = aSquared / DNDS::Math::ScalarMax(DNDS::Math::ScalarSqrt(aSquared), unL + unR >= 0 ? DNDS::Math::ScalarAbs(unL) : DNDS::Math::ScalarAbs(unR));
            }
            else
            {
                const Scalar aLSquared = factor * (hL + Scalar(.5) * dot(vL, vL)), aRSquared = factor * (hR + Scalar(.5) * dot(vR, vR));
                data.ausm.faceSoundSpeed = DNDS::Math::ScalarMin(aLSquared / DNDS::Math::ScalarMax(DNDS::Math::ScalarSqrt(aLSquared), unL), aRSquared / DNDS::Math::ScalarMax(DNDS::Math::ScalarSqrt(aRSquared), -unR));
            }
        }
        return data;
    }
}

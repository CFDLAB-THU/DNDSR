#pragma once

#include "Cauchy/Implicit/CUDA/FrozenJacobianKernels.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Physics/PerfectGasCharacteristics.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    /// Stored-neighbor perturbations are rotated together with momentum.
    /// Pressure derivatives remain in the stored frame. Exact identity
    /// faces skip this operation entirely, as in the CPU operator.
    DNDS_DEVICE_CALLABLE inline FlowState RotateJacobianFlow(FlowState value, const JacobianPointRecord &point)
    {
        if (!point.changesFrame)
            return value;
        const Vector3 momentum{value[1], value[2], value[3]};
        for (int i = 0; i < 3; ++i)
        {
            value[1 + i] = 0;
            for (int j = 0; j < 3; ++j)
                value[1 + i] += point.rotation[i + 3 * j] * momentum[j];
        }
        return value;
    }

    /** @brief Accumulate one weighted frozen physical Jacobian action.
     * @details state and pressureDerivative are in the stored neighbor frame;
     * the incidence rotates flow and its perturbation into the current face
     * frame. Selection applies to the stored input and final output components.
     * Boundary chain rules pass an already transformed exterior perturbation
     * with selectComponents=false, and project the completed diagonal later.
     * This common arithmetic is used by CUDA assembly, evaluated application
     * and diagonal preparation; it acquires no storage or execution resource.
     */
    template <int N>
    DNDS_DEVICE_CALLABLE NumericalStatus AddFrozenJacobianPoint(const FrozenJacobianParameters &p, index iCell,
                                                                const JacobianPointRecord &point, const FlowState &left,
                                                                const std::array<real, N> &state, const std::array<real, N> &pressureDerivative,
                                                                real pressure, real convection, real diffusion, const Vector3 &magnitudes,
                                                                const std::array<real, N> &input, std::array<real, N> &result,
                                                                bool selectComponents = true)
    {
        FlowState flow{}, delta{};
        real dp = 0;
        for (int j = 0; j < N; ++j)
        {
            if (!selectComponents || p.active[j])
                dp += pressureDerivative[j] * input[j];
            if (j < 5)
            {
                flow[j] = state[j];
                delta[j] = (!selectComponents || p.active[j]) ? input[j] : 0;
            }
        }
        flow = RotateJacobianFlow(flow, point);
        delta = RotateJacobianFlow(delta, point);
        const auto grid = p.frame.GridVelocity(point.location);
        auto derivative = PrepareCompressibleFluxDerivative(flow, delta, pressure, dp, point.normal, grid);
        // No derivative of the former isotropic 2*rho*k/3 stress or energy work.
        const real lambda = convection + diffusion;
        FlowState dissipation{};
        for (int j = 0; j < 5; ++j)
            dissipation[j] = lambda * delta[j];
        if constexpr (N == 5)
            if (p.roe)
            {
                const auto basis = TryPerfectGasCharacteristics(p.gas, left, flow, point.normal);
                if (!basis)
                    return basis.status.At(iCell, point.sample);
                if (basis.value.componentwiseFallback)
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell, point.sample};
                FlowState characteristic{};
                for (int i = 0; i < 5; ++i)
                {
                    for (int j = 0; j < 5; ++j)
                        characteristic[i] += basis.value.forward[i + 5 * j] * delta[j];
                    characteristic[i] *= magnitudes[i < 3 ? 0 : i - 2];
                }
                for (int i = 0; i < 5; ++i)
                {
                    dissipation[i] = 0;
                    for (int j = 0; j < 5; ++j)
                        dissipation[i] += basis.value.inverse[i + 5 * j] * characteristic[j];
                    dissipation[i] += diffusion * delta[i];
                }
            }
        real speed = 0, deltaSpeed = 0;
        if constexpr (N > 5)
            for (int d = 0; d < 3; ++d)
            {
                const real velocity = flow[1 + d] / flow[0];
                speed += (velocity - grid[d]) * point.normal[d];
                deltaSpeed += (delta[1 + d] - velocity * delta[0]) / flow[0] * point.normal[d];
            }
        const real factor = .5 * p.scale * point.weight;
        for (int j = 0; j < N; ++j)
            if (!selectComponents || p.active[j])
            {
                const real value = j < 5 ? derivative[j] - dissipation[j]
                                         : PrepareAdvectedScalarFluxDerivative(state[j], input[j], speed, deltaSpeed) - lambda * input[j];
                result[j] += factor * value;
                if (!std::isfinite(result[j]))
                    return NumericalStatus{NumericalError::InvalidState, 2, iCell, point.sample};
            }
        return {};
    }

}

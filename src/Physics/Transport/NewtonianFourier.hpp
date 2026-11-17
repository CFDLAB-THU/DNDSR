#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Math/ComputeScalar.hpp"
#include "Physics/Transport/QuadraticStress.hpp"
#include <array>
#include <limits>
#include <type_traits>

namespace DNDS::Physics::Transport
{
    template <class Scalar>
    struct NewtonianFourierFluxT
    {
        std::array<Scalar, 3> traction{};
        Scalar energy = 0;
    };
    using NewtonianFourierFlux = NewtonianFourierFluxT<real>;
    /// Deviatoric Newtonian normal stress, velocity gradient indexed (component, direction).
    template <class Scalar, class Gradient, class Normal>
    DNDS_DEVICE_CALLABLE inline std::array<Scalar, 3> NewtonianTraction(const Gradient &g, const Normal &n, Scalar viscosity)
    {
        const Scalar trace = (g(0, 0) + g(1, 1) + g(2, 2)) * Scalar(2. / 3.);
        std::array<Scalar, 3> traction{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                traction[i] += viscosity * (g(i, j) + g(j, i) - (i == j ? trace : Scalar(0))) * n[j];
        return traction;
    }
    /// Tangential viscous force exerted by the fluid on the solid per unit area.
    /// n is the outward unit normal of the fluid, traction is tau*n. Pressure
    /// and the isotropic turbulent stress have no tangential component.
    template <class Traction, class Normal>
    DNDS_DEVICE_CALLABLE inline std::array<real, 3> WallShear(const Traction &traction, const Normal &normal)
    {
        real normalStress = 0;
        for (int d = 0; d < 3; ++d)
            normalStress += traction[d] * normal[d];
        std::array<real, 3> shear{};
        for (int d = 0; d < 3; ++d)
            shear[d] = normalStress * normal[d] - traction[d];
        return shear;
    }
    /// Stokes/Newtonian stress and Fourier heat flux in a common Cartesian
    /// frame. Inputs are ready point values in one unit system; velocityGradient
    /// has (component,direction) indexing. No EOS, allocation, validation, mesh,
    /// normal normalization or communication. Caller supplies effective mu/k,
    /// including any explicitly prepared turbulence augmentation. This kernel
    /// uses the linear Newtonian law by default. QCR is an explicit optional
    /// stress correction with a ready turbulent/total viscosity ratio.
    template <class TVelocity, class TVelocityGradient, class TTemperatureGradient, class TNormal, class Scalar>
    DNDS_DEVICE_CALLABLE NewtonianFourierFluxT<Scalar> NewtonianFourierNormalFlux(const TVelocity &velocity, const TVelocityGradient &velocityGradient,
                                                                                  const TTemperatureGradient &temperatureGradient, const TNormal &normal,
                                                                                  Scalar viscosity, Scalar conductivity,
                                                                                  bool useQCR = false, Scalar turbulentViscosityRatio = Scalar(0))
    {
        Scalar divergence = 0;
        for (int d = 0; d < 3; ++d)
            divergence += velocityGradient(d, d);
        NewtonianFourierFluxT<Scalar> result;
        // Reduced arithmetic currently qualifies the linear Newtonian law
        // only. Host dispatch rejects QCR; a direct unsupported call returns
        // a nonfinite result for the caller's numerical status check, never
        // silently substitutes Newtonian stress or promotes to FP64.
        if constexpr (!std::is_same_v<Scalar, real>)
            if (useQCR)
            {
                const Scalar invalid(std::numeric_limits<real>::quiet_NaN());
                return {{invalid, invalid, invalid}, invalid};
            }
        if constexpr (std::is_same_v<Scalar, real>)
        {
            if (useQCR)
            {
                Eigen::Matrix<real, 3, 3> derivative, stress;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                    {
                        derivative(i, j) = velocityGradient(j, i);
                        stress(i, j) = viscosity * (velocityGradient(i, j) + velocityGradient(j, i)) +
                                       (i == j ? (-2. / 3.) * viscosity * divergence : 0);
                    }
                ApplyQCRStress<3>(stress, derivative, turbulentViscosityRatio);
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                        result.traction[i] += stress(i, j) * normal[j];
                    result.energy += velocity[i] * result.traction[i];
                }
                for (int d = 0; d < 3; ++d)
                    result.energy += conductivity * temperatureGradient[d] * normal[d];
                return result;
            }
        }
        result.traction = NewtonianTraction(velocityGradient, normal, viscosity);
        for (int i = 0; i < 3; ++i)
        {
            result.energy += velocity[i] * result.traction[i];
        }
        for (int d = 0; d < 3; ++d)
            result.energy += conductivity * temperatureGradient[d] * normal[d];
        return result;
    }
}

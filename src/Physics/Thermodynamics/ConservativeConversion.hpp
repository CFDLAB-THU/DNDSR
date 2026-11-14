#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Math/ComputeScalar.hpp"
#include <array>
#include <type_traits>

namespace DNDS::Physics::Thermodynamics
{
    /// Pure compressible kinematics, with no EOS/energy-reference assumption.
    /// Flow rows are [rho, momentum(Dim), rhoE]. Callers validate positive rho,
    /// sizes and access once; row types need only operator[]. Dim is 2 or 3.
    template <int Dim, class TFlow>
    DNDS_DEVICE_CALLABLE auto ConservativeVelocity(const TFlow &flow)
    {
        static_assert(Dim == 2 || Dim == 3);
        using Scalar = std::remove_cv_t<std::remove_reference_t<decltype(flow[0])>>;
        std::array<Scalar, Dim> velocity;
        for (int d = 0; d < Dim; ++d)
            velocity[d] = flow[1 + d] / flow[0];
        return velocity;
    }
    template <int Dim, class TFlow>
    DNDS_DEVICE_CALLABLE auto ConservativeSpecificInternalEnergy(const TFlow &flow)
    {
        using Scalar = std::remove_cv_t<std::remove_reference_t<decltype(flow[0])>>;
        const auto velocity = ConservativeVelocity<Dim>(flow);
        Scalar speedSquared = 0;
        for (int d = 0; d < Dim; ++d)
            speedSquared += DNDS::Math::ScalarSquare(velocity[d]);
        return flow[Dim + 1] / flow[0] - Scalar(.5) * speedSquared;
    }
    /// specificEnergy includes any thermochemical formation/reference energy.
    template <int Dim, class TVelocity, class Scalar>
    DNDS_DEVICE_CALLABLE Scalar ConservativeTotalEnergyDensity(Scalar rho, const TVelocity &velocity, Scalar specificEnergy)
    {
        static_assert(Dim == 2 || Dim == 3);
        Scalar speedSquared = 0;
        for (int d = 0; d < Dim; ++d)
            speedSquared += DNDS::Math::ScalarSquare(velocity[d]);
        return rho * (specificEnergy + Scalar(.5) * speedSquared);
    }
    /// One directional derivative d(rho*e), including all kinetic terms.
    template <int Dim, class TFlow, class TDerivative>
    DNDS_DEVICE_CALLABLE auto ConservativeInternalEnergyDensityDerivative(const TFlow &flow, const TDerivative &derivative)
    {
        using Scalar = std::remove_cv_t<std::remove_reference_t<decltype(flow[0])>>;
        const auto velocity = ConservativeVelocity<Dim>(flow);
        Scalar speedSquared = 0, momentumDerivative = 0;
        for (int d = 0; d < Dim; ++d)
        {
            speedSquared += DNDS::Math::ScalarSquare(velocity[d]);
            momentumDerivative += velocity[d] * derivative[1 + d];
        }
        return derivative[Dim + 1] - momentumDerivative + Scalar(.5) * speedSquared * derivative[0];
    }
    /// d(phi) from conservative d(rho*phi), e.g. velocity, species or RANS.
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar ConservativeSpecificDerivative(Scalar rho, Scalar phi, Scalar densityDerivative, Scalar conservedDerivative)
    {
        return (conservedDerivative - phi * densityDerivative) / rho;
    }
}

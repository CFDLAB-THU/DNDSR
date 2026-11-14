#pragma once

#include "DNDS/Core/Defines.hpp"
#include <array>

namespace DNDS::Cauchy
{
    /// Conservative compressible flow profile, including transverse momentum
    /// on planar geometry. Its energy origin is defined by the equation model.
    template <class Scalar>
    using FlowStateT = std::array<Scalar, 5>;
    template <class Scalar>
    using Vector3T = std::array<Scalar, 3>;
    using FlowState = FlowStateT<real>;
    using Vector3 = Vector3T<real>;
    /// Five field gradients, each with three physical-space directions. The
    /// consumer declares whether components are conservative or [rho,u,v,w,T].
    template <class Scalar>
    using FlowGradientT = std::array<Vector3T<Scalar>, 5>;
    using FlowGradient = FlowGradientT<real>;
    template <class Scalar>
    struct PrimitiveStateT
    {
        Scalar density, pressure, temperature, soundSpeed;
        Vector3T<Scalar> velocity;
    };
    using PrimitiveState = PrimitiveStateT<real>;
}

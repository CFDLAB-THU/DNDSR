#pragma once

#include <limits>

namespace DNDS::Physics::Transport
{
    /// Newtonian/Fourier molecular coefficients at one point. Species
    /// coefficients use separate caller-owned buffers; these two scalars do
    /// not prescribe a diagonal or matrix species-transport representation.
    template <class Scalar>
    struct PropertiesT
    {
        Scalar viscosity = Scalar(std::numeric_limits<double>::quiet_NaN());
        Scalar conductivity = Scalar(std::numeric_limits<double>::quiet_NaN());
    };
    using Properties = PropertiesT<double>;
}

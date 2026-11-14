#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Math/ComputeScalar.hpp"
#include <cmath>

namespace DNDS::Physics::Transport
{
    enum class ViscosityModel
    {
        Constant,
        Sutherland,
        ConstantKinematic
    };
    /// Euler's reference-temperature law, with all temperatures in the same
    /// units. The caller validates positive T/Tref and nonnegative C at setup
    /// or its physical-state boundary. No state repair or reference fallback.
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar SutherlandViscosity(Scalar reference, Scalar temperature, Scalar referenceTemperature, Scalar constant)
    {
        const Scalar ratio = temperature / referenceTemperature;
        return reference * ratio * DNDS::Math::ScalarSqrt(ratio) * (referenceTemperature + constant) / (temperature + constant);
    }
    /// Pure code-unit molecular law. ConstantKinematic preserves Euler's
    /// muModel=2 convention: mu_code=reference*rho_code. Its reference argument
    /// is the dynamic viscosity at code density 1, not an inferred state rho.
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar MolecularViscosity(ViscosityModel model, Scalar reference, Scalar density, Scalar temperature,
                                                          Scalar referenceTemperature, Scalar constant)
    {
        if (model == ViscosityModel::Sutherland)
            return SutherlandViscosity(reference, temperature, referenceTemperature, constant);
        if (model == ViscosityModel::ConstantKinematic)
            return reference * density;
        return reference;
    }
}

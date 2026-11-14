#pragma once

#include <limits>

namespace DNDS::Physics::Thermodynamics
{
    /// One thermodynamic point, in the units declared by its producer. Backend
    /// ChemicalSource queries produce SI; solver-facing material queries use
    /// ReferenceScales. Internal energy and enthalpy include thermochemical
    /// reference energy, not the separate bookkeeping base-energy offset.
    template <class Scalar>
    struct PropertiesT
    {
        Scalar density = Scalar(std::numeric_limits<double>::quiet_NaN());
        Scalar temperature = Scalar(std::numeric_limits<double>::quiet_NaN());
        Scalar pressure = Scalar(std::numeric_limits<double>::quiet_NaN());
        Scalar soundSpeed = Scalar(std::numeric_limits<double>::quiet_NaN());
        Scalar internalEnergy = Scalar(std::numeric_limits<double>::quiet_NaN());
        Scalar enthalpy = Scalar(std::numeric_limits<double>::quiet_NaN());
        Scalar cp = Scalar(std::numeric_limits<double>::quiet_NaN());
        Scalar cv = Scalar(std::numeric_limits<double>::quiet_NaN());
        Scalar gasConstant = Scalar(std::numeric_limits<double>::quiet_NaN());
    };
    using Properties = PropertiesT<double>;
}

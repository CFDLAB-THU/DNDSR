#pragma once

namespace DNDS::Cauchy
{
    /// Physical boundary family, resolved from configuration before execution.
    enum class BoundaryKind
    {
        Interior,
        Extrapolate,
        Farfield,
        TotalConditionInlet,
        SlipWall,
        NoSlipWall,
        Analytic,
        DoubleMachBottom,
        Interface,
        VelocityInlet,
        PressureOutlet
    };
}

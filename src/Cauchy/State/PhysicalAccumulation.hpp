#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::Cauchy
{
    enum class PhysicalOperation
    {
        CopyScale,
        Combination,
        StateAndRate,
        Axpy,
        TimeResidual
    };

    /// One constant-density row. The momentum and turbulence variables carry
    /// physical mass. Continuity retains the spatial rate and source only.
    template <PhysicalOperation Op, class Out, class X, class Y, class Z>
    DNDS_DEVICE_CALLABLE inline void ApplyAccumulatedRow(index row, index width, Out out, X x, Y y, Z z, real a, real b)
    {
        auto differential = [&](index j)
        {
            if constexpr (Op == PhysicalOperation::CopyScale)
                out(row, j) = a * x(row, j);
            else if constexpr (Op == PhysicalOperation::Combination || Op == PhysicalOperation::StateAndRate)
                out(row, j) = a * x(row, j) + b * y(row, j);
            else if constexpr (Op == PhysicalOperation::Axpy)
                out(row, j) += a * x(row, j);
            else
                out(row, j) = x(row, j) - a * y(row, j) + b * z(row, j);
        };
        for (index j = 0; j < 3; ++j)
            differential(j);
        if constexpr (Op == PhysicalOperation::CopyScale || Op == PhysicalOperation::Combination)
            out(row, 3) = 0;
        else if constexpr (Op == PhysicalOperation::TimeResidual)
            out(row, 3) = x(row, 3) + b * z(row, 3);
        else if constexpr (Op == PhysicalOperation::StateAndRate)
            out(row, 3) = b * y(row, 3);
        for (index j = 4; j < width; ++j)
            differential(j);
    }

    template <class Launch>
    void DispatchPhysicalOperation(PhysicalOperation op, Launch launch)
    {
        switch (op)
        {
        case PhysicalOperation::CopyScale:
            launch(std::integral_constant<PhysicalOperation, PhysicalOperation::CopyScale>{});
            break;
        case PhysicalOperation::Combination:
            launch(std::integral_constant<PhysicalOperation, PhysicalOperation::Combination>{});
            break;
        case PhysicalOperation::StateAndRate:
            launch(std::integral_constant<PhysicalOperation, PhysicalOperation::StateAndRate>{});
            break;
        case PhysicalOperation::Axpy:
            launch(std::integral_constant<PhysicalOperation, PhysicalOperation::Axpy>{});
            break;
        case PhysicalOperation::TimeResidual:
            launch(std::integral_constant<PhysicalOperation, PhysicalOperation::TimeResidual>{});
            break;
        }
    }
}

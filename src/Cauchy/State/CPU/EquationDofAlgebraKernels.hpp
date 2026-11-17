#pragma once

#include "Cauchy/State/PhysicalAccumulation.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    template <class Output, class X, class Y, class Z>
    void ApplyConstantDensityAccumulation(PhysicalOperation operation, Output output, X x, Y y, Z z, real a, real b)
    {
        DispatchPhysicalOperation(operation, [&](auto op)
                                  { OMP::ForRows(0, output.RowCount(), [&](index row)
                                                 { ApplyAccumulatedRow<decltype(op)::value>(row, output.ComponentNumber(), output, x, y, z, a, b); }); });
    }
}

#pragma once

#include "Physics/RANS/Point.hpp"

namespace DNDS::Physics::Incompressible
{
    /// Gather [u,v,w,p,scalars...] into the shared mechanical closure. Energy
    /// and its derivatives are unused. The selected SA/Wilcox/SST source and
    /// normal diffusion formulas consume velocity/scalar gradients directly.
    template <class State, class Gradient>
    DNDS_DEVICE_CALLABLE inline RANS::Point RANSPoint(const State &state, const Gradient &gradient, int scalars,
                                                      real viscosity, real wallDistance, real cellLength)
    {
        RANS::Point result;
        result.incompressible = true;
        result.cp = result.gamma = 0;
        result.state[0] = 1;
        for (int j = 0; j < 4 + scalars; ++j)
            if (j != 3)
            {
                result.state[j + 1] = state[j];
                for (int d = 0; d < 3; ++d)
                    result.gradient(d, j + 1) = gradient(j, d);
            }
        result.molecularViscosity = viscosity;
        result.wallDistance = wallDistance;
        result.cellLength = cellLength;
        return result;
    }
}

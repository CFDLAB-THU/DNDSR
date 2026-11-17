#pragma once
#include "Cauchy/Physics/ACMFlux.hpp"
#include "Physics/Incompressible/RANSPoint.hpp"
#include "Physics/RANS/PointKernels.hpp"

namespace DNDS::Cauchy::ACM
{
    struct SpatialParameters
    {
        Parameters flow;
        Physics::RANS::PointParameters rans;
    };
    template <int C>
    struct KernelPoint
    {
        std::array<real, C> values{};
        std::array<std::array<real, 3>, C> gradients{};
        template <class Means, class Gradients>
        DNDS_DEVICE_CALLABLE void Load(index cell, const Means &mean, const Gradients &gradient, int dim, bool useGradient)
        {
            for (int j = 0; j < C; ++j)
            {
                values[j] = mean(cell, j);
                for (int d = 0; d < 3; ++d)
                    gradients[j][d] = useGradient && d < dim ? gradient(cell, j * dim + d) : 0;
            }
        }
        DNDS_DEVICE_CALLABLE void Rotate(const std::array<real, 9> &rotation)
        {
            const auto old = *this;
            for (int j = 0; j < C; ++j)
                for (int d = 0; d < 3; ++d)
                {
                    real value = 0;
                    for (int k = 0; k < 3; ++k)
                        value += rotation[d + 3 * k] * old.gradients[j][k];
                    gradients[j][d] = value;
                }
            const auto spatial = gradients;
            for (int j = 0; j < 3; ++j)
            {
                values[j] = 0;
                for (int k = 0; k < 3; ++k)
                    values[j] += rotation[j + 3 * k] * old.values[k];
                for (int d = 0; d < 3; ++d)
                {
                    gradients[j][d] = 0;
                    for (int k = 0; k < 3; ++k)
                        gradients[j][d] += rotation[j + 3 * k] * spatial[k][d];
                }
            }
        }
        DNDS_DEVICE_CALLABLE auto Turbulence(const SpatialParameters &p, real distance, real length = 0) const
        {
            return Physics::Incompressible::RANSPoint(values, [&](int j, int d)
                                                      { return gradients[j][d]; }, C - 4, p.flow.kinematicViscosity, distance, length);
        }
        DNDS_DEVICE_CALLABLE NumericalResult<std::array<real, 3>> Auxiliary(const SpatialParameters &p, real distance) const
        {
            real mu = p.flow.kinematicViscosity;
            if constexpr (C > 4)
            {
                const auto eddy = Physics::RANS::TryTransport(p.rans, Turbulence(p, distance));
                if (!eddy)
                    return eddy.status;
                mu += eddy.value.viscosity;
            }
            return std::array<real, 3>{values[3], 0, (4. / 3.) * mu};
        }
    };
}

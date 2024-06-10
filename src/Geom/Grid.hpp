#pragma once
#include "Geometric.hpp"
#include "Solver/Scalar.hpp"

namespace DNDS::Geom
{
    inline void GetTanhDistributionBilateral(real x0, real x1, index NInterval, real d0, Eigen::Vector<real, Eigen::Dynamic> &d)
    {
        DNDS_assert(x1 > x0 && NInterval > 0);
        real d00 = (x1 - x0) / NInterval;
        if (d00 <= d0)
        {
            d.setLinSpaced(NInterval + 1, x0, x1);
            return;
        }
        real h = Scalar::BisectSolveLower(
            [&](real h)
            { return -1 + std::tanh(h - 2 * h / NInterval) / (std::tanh(h) + verySmallReal) + d0 * 2 / (x1 - x0); },
            1e-10, 20, 0.0, 20);
        real Mh = std::tanh(h);

        d = (Eigen::VectorXd::LinSpaced(NInterval + 1, -h, h).array().tanh() + Mh) / (2 * Mh) * (x1 - x0) + x0;
        d(0) = x0;
        d(NInterval) = x1;
    }
}
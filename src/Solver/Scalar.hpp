#pragma once

#include "DNDS/HardEigen.hpp"
#include "DNDS/Defines.hpp"
#include "fmt/core.h"

namespace DNDS::Scalar
{
    template <class TF>
    real BisectSolveLower(TF &&F, real v0, real v1, real fTarget, int maxIter)
    {
        real f0 = F(v0);
        real f1 = F(v1);
        DNDS_assert_info(f0 <= fTarget && f1 >= fTarget && v0 < v1,
                         fmt::format(" v0 {} v1 {} f0 {} f1 {} fT {}", v0, v1, f0, f1, fTarget));
        for (int iter = 1; iter <= maxIter; iter++)
        {
            if (f0 >= fTarget - verySmallReal)
                break;
            real vm = 0.5 * (v0 + v1);
            v0 = std::min(v0, vm);
            v1 = std::max(v1, vm);
            if (F(vm) <= fTarget)
                v0 = vm;
            else
                v1 = vm;
        }

        return v0;
    }

}
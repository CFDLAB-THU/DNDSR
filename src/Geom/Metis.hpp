#pragma once

#include "Geometric.hpp"
#include <metis.h>
#include <parmetis.h>

namespace _METIS
{
}

namespace _METIS
{
    static idx_t indexToIdx(DNDS::index v)
    {
        if constexpr (sizeof(DNDS::index) <= sizeof(idx_t))
            return v;
        else
            return DNDS::checkedIndexTo32(v);
    }
}

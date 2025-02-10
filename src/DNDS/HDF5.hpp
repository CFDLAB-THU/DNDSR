#pragma once

#include "Defines.hpp"
#include <hdf5.h>

namespace DNDS
{
    inline hid_t DNDS_H5T_INDEX()
    {
        if constexpr (std::is_same_v<index, int64_t>)
            return H5T_NATIVE_INT64;
        else
        {
            static_assert(std::is_same_v<index, int64_t>, "index type not right");
            return H5T_NATIVE_INT64;
        }
    }

    inline hid_t DNDS_H5T_ROWSIZE()
    {
        if constexpr (std::is_same_v<rowsize, int32_t>)
            return H5T_NATIVE_INT32;
        else
        {
            static_assert(std::is_same_v<rowsize, int32_t>, "rowsize type not right");
            return H5T_NATIVE_INT32;
        }
    }

    inline hid_t DNDS_H5T_REAL()
    {
        if constexpr (std::is_same_v<real, double>)
            return H5T_NATIVE_DOUBLE;
        else
        {
            static_assert(std::is_same_v<real, double>, "real type not right");
            return H5T_NATIVE_DOUBLE;
        }
    }
}
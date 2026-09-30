#pragma once
#include "Errors.hpp"
#include <limits>
#include <type_traits>

namespace DNDS::CheckedSize
{
    // Host-side size arithmetic. Validate operands before evaluating expressions.
    template <class T>
    T Add(T a, T b)
    {
        static_assert(std::is_integral_v<T>);
        DNDS_check_throw_info(a >= 0 && b >= 0 && a <= std::numeric_limits<T>::max() - b,
                              "negative size or size addition overflow");
        return a + b;
    }

    template <class T>
    T Multiply(T a, T b)
    {
        static_assert(std::is_integral_v<T>);
        DNDS_check_throw_info(a >= 0 && b >= 0 && (b == 0 || a <= std::numeric_limits<T>::max() / b),
                              "negative size or size multiplication overflow");
        return a * b;
    }
}

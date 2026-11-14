#pragma once

#include "DNDS/Core/Defines.hpp"
#include <cmath>
#include <limits>
#ifdef DNDS_USE_CCCL_COMPUTE
#    include <cuda/fpmp>
#endif

namespace DNDS::Math
{
    enum class ComputePrecision
    {
        FP64,
        FP32,
        FloatFloat
    };
    /// Kernel-local arithmetic types. Persistent DNDS::real storage stays FP64.
    using real_f32 = float;
#ifdef DNDS_USE_CCCL_COMPUTE
    /// Unevaluated FP32 high/low pair: greater precision than float, float-like
    /// exponent range. Explicit conversion preserves both parts of FP64 inputs.
    /// Not IEEE double, not a persistent state/serialization scalar in this phase.
    using real_f32f32 = cuda::fp32mp2;
#endif
    namespace Detail
    {
        template <class T>
        struct ComputeScalarLimits : std::numeric_limits<T>
        {
        };
#ifdef DNDS_USE_CCCL_COMPUTE
        template <>
        struct ComputeScalarLimits<real_f32f32> : cuda::std::numeric_limits<real_f32f32>
        {
        };
#endif
    }

    /// Explicit arithmetic-boundary conversion; no allocation or residency change.
    template <class To, class From>
    DNDS_DEVICE_CALLABLE inline To ScalarCast(const From &value) noexcept
    {
        return static_cast<To>(value);
    }
    template <class T>
    DNDS_DEVICE_CALLABLE inline T ScalarAbs(T value) noexcept
    {
        // Floating arithmetic providers expose fabs through ADL. std::abs
        // can instead select their implicit double conversion, discarding
        // the low limb (or failing an explicit conversion back to the pair).
        using std::fabs;
        return fabs(value);
    }
    template <class T>
    DNDS_DEVICE_CALLABLE inline T ScalarSqrt(T value) noexcept
    {
        using std::sqrt;
        return sqrt(value);
    }
    template <class T>
    DNDS_DEVICE_CALLABLE inline bool ScalarFinite(T value) noexcept
    {
        using std::isfinite;
        return isfinite(value);
    }
    /// Preserve std::min/max operand order and NaN behavior; no implicit repair.
    template <class T>
    DNDS_DEVICE_CALLABLE inline T ScalarMin(T a, T b) noexcept { return b < a ? b : a; }
    template <class T>
    DNDS_DEVICE_CALLABLE inline T ScalarMax(T a, T b) noexcept { return a < b ? b : a; }
    /// Pure multiply, also valid for software arithmetic types rejected by sqr's
    /// std::is_arithmetic guard. Preserve grouping of the canonical expression.
    template <class T>
    DNDS_DEVICE_CALLABLE inline T ScalarSquare(T value) noexcept { return value * value; }

    template <class T>
    DNDS_DEVICE_CALLABLE inline T ScalarEpsilon() noexcept
    {
        return Detail::ComputeScalarLimits<T>::epsilon();
    }
    /// Smallest positive normal value of the compute type, not the state type.
    template <class T>
    DNDS_DEVICE_CALLABLE inline T ScalarMinPositive() noexcept
    {
        return Detail::ComputeScalarLimits<T>::min();
    }
    /// Largest finite value of the compute type; avoid narrowing double::max.
    template <class T>
    DNDS_DEVICE_CALLABLE inline T ScalarMaxFinite() noexcept
    {
        return Detail::ComputeScalarLimits<T>::max();
    }
}

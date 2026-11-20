#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::Runtime
{
    enum class MemorySpace
    {
        Host,
        CUDA,
        /// Only for an unbound borrowed view; never an allocation backend.
        Unspecified,
        SYCL
    };

    /// Allocation placement, independent of layout, access and stream readiness.
    /// Host remains the default; Unspecified is reserved for borrowed views.
    struct Device
    {
        MemorySpace space = MemorySpace::Host;
        int ordinal = -1;
        DNDS_DEVICE_CALLABLE static constexpr Device Host() { return {}; }
        DNDS_DEVICE_CALLABLE static constexpr Device CUDA(int ordinal) { return {MemorySpace::CUDA, ordinal}; }
        DNDS_DEVICE_CALLABLE static constexpr Device SYCL(int ordinal) { return {MemorySpace::SYCL, ordinal}; }
        DNDS_DEVICE_CALLABLE static constexpr Device Unspecified() { return {MemorySpace::Unspecified, -1}; }
        DNDS_DEVICE_CALLABLE constexpr bool IsValid() const
        {
            return (space == MemorySpace::Host && ordinal == -1) || ((space == MemorySpace::CUDA || space == MemorySpace::SYCL) && ordinal >= 0);
        }
        DNDS_DEVICE_CALLABLE constexpr bool operator==(Device r) const { return space == r.space && ordinal == r.ordinal; }
        DNDS_DEVICE_CALLABLE constexpr bool operator!=(Device r) const { return !(*this == r); }
    };
}

#pragma once

#include <string_view>

namespace DNDS::Cauchy
{
    /// Stationary methods apply their own correction. GMRES applies the configured
    /// preconditioner. The returned name borrows storage from settings.
    template <class Settings>
    std::string_view EffectiveLinearProvider(const Settings &settings)
    {
        return settings.method == "gmres" ? settings.preconditioner : settings.method;
    }
    inline bool KnownLinearProvider(std::string_view name)
    {
        return name == "sgs" || name == "jacobi" || name == "ilu" || name == "amg" || name == "amgx";
    }
    inline bool DeviceLinearProvider(std::string_view name)
    {
        return name == "jacobi" || name == "amgx";
    }
}

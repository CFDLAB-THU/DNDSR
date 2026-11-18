#pragma once

#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <string_view>

namespace DNDS::Cauchy
{
    struct KernelExecutionSettings;

    /// Qualified local operator features. Callers pass no profile when other
    /// physics, input representations, geometry or schedules are in use.
    struct RHSKernelProfile
    {
        std::string_view equation;
        std::string_view boundary;
        int components = 0, modes = 0, dimension = 0, order = 0, points = 0;
    };

    struct RHSKernelLaunchSelection
    {
        std::string mapping, reduction;
        int blockSize = 128, gridBlocks = 0;
        bool hoistFaceRows = false, doubleBuffer = false;
    };

    /// Lookup an exact RHS profile within a matching device preset. Returns
    /// the standard kernels JSON fields or null for an unqualified profile.
    nlohmann::json FindRHSKernelPreset(std::string_view deviceName, const RHSKernelProfile &profile);
    std::string RHSKernelDeviceName(int ordinal);

    /// Resolve requested settings once for a prepared RHS operator. An
    /// explicit mapping uses that mapping's defaults for any unspecified
    /// fields. A direct-gradient operator always selects the plain thread path.
    RHSKernelLaunchSelection ResolveRHSKernelLaunch(const KernelExecutionSettings &settings,
                                                    bool directGradient, std::string_view deviceName = {},
                                                    std::optional<RHSKernelProfile> profile = {});
}

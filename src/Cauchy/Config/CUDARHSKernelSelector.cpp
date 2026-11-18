#include "Cauchy/Config/CUDARHSKernelSelector.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif

namespace DNDS::Cauchy
{
    namespace
    {
        // Keep the experimental registry in this TU so editing a preset does
        // not invalidate any CUDA translation units.
        const nlohmann::json &Registry()
        {
            static const auto registry = nlohmann::json::parse(R"preset(
{
  "schemaVersion": 1,
  "devices": [{
    "namePattern": "NVIDIA A100 80GB PCIe",
    "presets": [
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 2, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 2, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 2, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 5, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 5, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 5, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 9, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 64, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 9, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 64, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 9, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 64, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 3, "D": 3, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 64, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 3, "D": 3, "Q": 3, "points": 4}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 3, "D": 3, "Q": 5, "points": 9}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 9, "D": 3, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 9, "D": 3, "Q": 3, "points": 4}, "kernels": {"rhsMapping": "ctaSegments", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 864}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 9, "D": 3, "Q": 5, "points": 9}, "kernels": {"rhsMapping": "ctaSegments", "rhsHoistFaceRows": false, "rhsBlockSize": 256, "rhsGridBlocks": 216}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 19, "D": 3, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 19, "D": 3, "Q": 3, "points": 4}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "periodic", "C": 5, "M": 19, "D": 3, "Q": 5, "points": 9}, "kernels": {"rhsMapping": "ctaSegments", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 864}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 2, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 2, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 2, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 5, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 5, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 5, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 9, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 64, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 9, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 64, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 9, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 64, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 3, "D": 3, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 216}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 3, "D": 3, "Q": 3, "points": 4}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 3, "D": 3, "Q": 5, "points": 9}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 9, "D": 3, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 9, "D": 3, "Q": 3, "points": 4}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 9, "D": 3, "Q": 5, "points": 9}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 19, "D": 3, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 19, "D": 3, "Q": 3, "points": 4}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "SA", "boundary": "periodic", "C": 6, "M": 19, "D": 3, "Q": 5, "points": 9}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 2, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 2, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 2, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 5, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 5, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 5, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 9, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 9, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 64, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 9, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 64, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 3, "D": 3, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 216}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 3, "D": 3, "Q": 3, "points": 4}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 3, "D": 3, "Q": 5, "points": 9}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 9, "D": 3, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 9, "D": 3, "Q": 3, "points": 4}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 9, "D": 3, "Q": 5, "points": 9}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 19, "D": 3, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 256, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 19, "D": 3, "Q": 3, "points": 4}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "KOmegaSST", "boundary": "periodic", "C": 7, "M": 19, "D": 3, "Q": 5, "points": 9}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": true, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "extrapolation", "C": 5, "M": 9, "D": 2, "Q": 1, "points": 1}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "extrapolation", "C": 5, "M": 9, "D": 2, "Q": 3, "points": 2}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 0}},
      {"profile": {"equation": "none", "boundary": "extrapolation", "C": 5, "M": 9, "D": 2, "Q": 5, "points": 3}, "kernels": {"rhsMapping": "thread", "rhsHoistFaceRows": false, "rhsBlockSize": 128, "rhsGridBlocks": 0}}
    ]
  }]
}
)preset");
            return registry;
        }

        bool FP64(const KernelExecutionSettings &s)
        {
            return s.polynomialArithmetic == "fp64" && s.dissipationArithmetic == "fp64" &&
                   s.fluxArithmetic == "fp64";
        }
    }

    nlohmann::json FindRHSKernelPreset(std::string_view deviceName, const RHSKernelProfile &profile)
    {
        const auto &registry = Registry();
        DNDS_check_throw_info(registry.at("schemaVersion") == 1, "unknown RHS kernel preset schema");
        const nlohmann::json pattern{{"equation", profile.equation}, {"boundary", profile.boundary}, {"C", profile.components}, {"M", profile.modes}, {"D", profile.dimension}, {"Q", profile.order}, {"points", profile.points}};
        for (const auto &device : registry.at("devices"))
        {
            const auto &name = device.at("namePattern").get_ref<const std::string &>();
            if (deviceName.find(name) == std::string_view::npos)
                continue;
            for (const auto &preset : device.at("presets"))
                if (preset.at("profile") == pattern)
                    return preset.at("kernels");
        }
        return nullptr;
    }

    std::string RHSKernelDeviceName(int ordinal)
    {
#ifdef DNDS_USE_CUDA
        cudaDeviceProp properties{};
        DNDS_check_throw_info(cudaGetDeviceProperties(&properties, ordinal) == cudaSuccess,
                              "cannot read the selected CUDA RHS device name");
        return properties.name;
#else
        (void)ordinal;
        DNDS_check_throw_info(false, "CUDA RHS device names require a CUDA build");
        return {};
#endif
    }

    RHSKernelLaunchSelection ResolveRHSKernelLaunch(const KernelExecutionSettings &settings,
                                                    bool directGradient, std::string_view deviceName,
                                                    std::optional<RHSKernelProfile> profile)
    {
        RHSKernelLaunchSelection launch;
        if (directGradient)
        {
            launch.mapping = "thread";
            launch.reduction = "serial";
            launch.blockSize = settings.rhsBlockSize.value_or(128);
            launch.gridBlocks = settings.rhsGridBlocks.value_or(0);
            return launch;
        }

        const bool fp64 = FP64(settings);
        const bool automatic = settings.rhsMapping == "auto";
        const bool requestedHoist = settings.rhsHoistFaceRows.value_or(false);
        const auto preset = automatic && fp64 && profile && !settings.rhsDoubleBuffer &&
                                    settings.rhsReduction == "auto"
                                ? FindRHSKernelPreset(deviceName, *profile)
                                : nlohmann::json(nullptr);
        if (!automatic)
            launch.mapping = settings.rhsMapping;
        else if (requestedHoist)
            launch.mapping = "thread";
        else if (!preset.is_null())
            launch.mapping = preset.at("rhsMapping").get<std::string>();
        else
            launch.mapping = fp64 ? "ctaSegments" : "thread";

        launch.hoistFaceRows = launch.mapping == "thread" &&
                               settings.rhsHoistFaceRows.value_or(!preset.is_null() &&
                                                                  preset.at("rhsHoistFaceRows").get<bool>());
        const bool presetCompatible = !preset.is_null() &&
                                      launch.mapping == preset.at("rhsMapping").get<std::string>() &&
                                      launch.hoistFaceRows == preset.at("rhsHoistFaceRows").get<bool>();
        launch.blockSize = settings.rhsBlockSize.value_or(presetCompatible ? preset.at("rhsBlockSize").get<int>() : 128);
        launch.gridBlocks = settings.rhsGridBlocks.value_or(presetCompatible ? preset.at("rhsGridBlocks").get<int>() : 0);
        launch.doubleBuffer = settings.rhsDoubleBuffer;
        launch.reduction = settings.rhsReduction == "auto"
                               ? (launch.mapping == "ctaSegments" ? "segmented" : "serial")
                               : settings.rhsReduction;
        DNDS_check_throw_info(launch.mapping == "thread" || launch.mapping == "ctaSegments", "invalid CUDA RHS mapping");
        DNDS_check_throw_info(launch.reduction == "serial" || launch.reduction == "segmented" ||
                                  launch.reduction == "boundedTree" || launch.reduction == "warpTree",
                              "invalid CUDA RHS reduction");
        return launch;
    }
}

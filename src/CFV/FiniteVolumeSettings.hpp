#pragma once

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif
#include "DNDS/Defines.hpp"
#include "DNDS/Serializer/JsonUtil.hpp"
#include "DNDS/Config/ConfigParam.hpp"
// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif

namespace DNDS::CFV
{
    /**
     * @brief
     * A means to translate nlohmann json into c++ primitive data types and back;
     * and stores then during computation.
     *
     */
    struct FiniteVolumeSettings
    {
        using json = nlohmann::ordered_json;

        int maxOrder{1}; /// @brief polynomial degree of reconstruction
        int intOrder{1}; /// @brief integration degree globally set @note this is actually reduced somewhat
        bool ignoreMeshGeometryDeficiency = false;

        int nIterCellSmoothScale = 15;

        // VRSettings()
        // {
        // }

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(FiniteVolumeSettings, FiniteVolumeSettings)

        DNDS_DEVICE_CALLABLE FiniteVolumeSettings() = default;
        DNDS_DEVICE_CALLABLE FiniteVolumeSettings(int dim)
        {
        }

        DNDS_DECLARE_CONFIG(FiniteVolumeSettings)
        {
            DNDS_FIELD(maxOrder,                      "Polynomial degree of reconstruction",
                       DNDS::Config::range(0));
            DNDS_FIELD(intOrder,                      "Global integration degree",
                       DNDS::Config::range(0));
            DNDS_FIELD(ignoreMeshGeometryDeficiency,  "Ignore mesh geometry deficiency warnings");
            DNDS_FIELD(nIterCellSmoothScale,          "Cell smooth scale iterations",
                       DNDS::Config::range(0));
        }

        /// @brief Backward-compatible write (used by Python bindings and VRSettings).
        DNDS_HOST void WriteIntoJson(json &jsonSetting) const
        {
            to_json(jsonSetting, *this);
        }

        /// @brief Backward-compatible read (used by Python bindings and VRSettings).
        DNDS_HOST void ParseFromJson(const json &jsonSetting)
        {
            from_json(jsonSetting, *this);
        }
    };
}

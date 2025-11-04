#pragma once

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif
#include "DNDS/Defines.hpp"
#include "DNDS/JsonUtil.hpp"
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

        FiniteVolumeSettings(int dim)
        {
        }

        /**
         * @brief write any data into jsonSetting member
         *
         */
        void WriteIntoJson(json &jsonSetting) const
        {
            jsonSetting["maxOrder"] = maxOrder;
            jsonSetting["intOrder"] = intOrder;
            jsonSetting["ignoreMeshGeometryDeficiency"] = ignoreMeshGeometryDeficiency;
            jsonSetting["nIterCellSmoothScale"] = nIterCellSmoothScale;
        }

        /**
         * @brief read any data from jsonSetting member
         *
         */
        void ParseFromJson(const json &jsonSetting)
        {
            maxOrder = jsonSetting["maxOrder"]; ///@todo //TODO: update to better
            intOrder = jsonSetting["intOrder"];
            ignoreMeshGeometryDeficiency = jsonSetting["ignoreMeshGeometryDeficiency"];
            nIterCellSmoothScale = jsonSetting["nIterCellSmoothScale"];
        }
        friend void from_json(const json &j, FiniteVolumeSettings &s)
        {
            s.ParseFromJson(j);
        }

        friend void to_json(json &j, const FiniteVolumeSettings &s)
        {
            s.WriteIntoJson(j);
        }
    };
}

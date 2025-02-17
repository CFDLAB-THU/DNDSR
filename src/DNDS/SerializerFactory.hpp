#pragma once

#include "SerializerBase.hpp"
#include "SerializerJSON.hpp"
#include "SerializerH5.hpp"
#include "JsonUtil.hpp"
#include <json.hpp>

namespace DNDS::Serializer
{
    struct SerializerFactory
    {
        std::string type = "JSON";
        int hdfDeflateLevel = 0;
        int hdfChunkSize = 0;
        bool hdfCollOnMeta = true;
        bool hdfCollOnData = false;
        int jsonBinaryDeflateLevel = 5;
        bool jsonUseCodecOnUInt8 = true;

        SerializerFactory() = default;
        SerializerFactory(const std::string &_type): type(_type){}

        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
            SerializerFactory,
            type,
            hdfDeflateLevel, hdfChunkSize,
            hdfCollOnMeta, hdfCollOnData,
            jsonBinaryDeflateLevel, jsonUseCodecOnUInt8)

        SerializerBaseSSP BuildSerializer(const MPIInfo& mpi)
        {
            SerializerBaseSSP serializerP;
            if (type == "JSON")
            {
                serializerP = std::make_shared<SerializerJSON>();
                std::dynamic_pointer_cast<SerializerJSON>(serializerP)->SetUseCodecOnUint8(jsonUseCodecOnUInt8);
                std::dynamic_pointer_cast<SerializerJSON>(serializerP)->SetDeflateLevel(jsonBinaryDeflateLevel);
            }
            else if (type == "H5")
            {
                serializerP = std::make_shared<SerializerH5>(mpi);
                std::dynamic_pointer_cast<SerializerH5>(serializerP)->SetChunkAndDeflate(hdfChunkSize, hdfDeflateLevel);
                std::dynamic_pointer_cast<SerializerH5>(serializerP)->SetCollectiveRW(hdfCollOnMeta, hdfCollOnData);
            }
            else
                DNDS_assert_info(false, "type of serializer not existent: " + type);
            return serializerP;
        }

        // std::tuple<std::string, std::string> ModifyFilePath(std::string fname, const MPIInfo &mpi)
        // {
        //     if (type == "JSON")
        //     {
        //         std::filesystem::path outPath;
        //         outPath = {fname + ".dir"};
        //         std::filesystem::create_directories(outPath);
        //         char BUF[512];
        //         std::sprintf(BUF, "%06d", mpi.rank);
        //         fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));
        //         return std::make_tuple(fname, getStringForcePath(outPath));
        //     }
        //     else if (type == "H5")
        //     {

        //         fname += ".dnds.h5";
        //         std::filesystem::path outPath = fname;
        //         std::filesystem::create_directories(outPath.parent_path() / ".");
        //         return std::make_tuple(fname, fname);
        //     }
        //     else
        //         DNDS_assert_info(false, "type of serializer not existent: " + type);
        // }
    };

}
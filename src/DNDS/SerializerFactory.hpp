#pragma once

#include "SerializerBase.hpp"
#include "SerializerJSON.hpp"
#include "SerializerH5.hpp"
#include "JsonUtil.hpp"
#include "ConfigParam.hpp"

namespace DNDS::Serializer
{
    struct SerializerFactory
    {
        std::string type = "JSON";
        int hdfDeflateLevel = 0;
        int hdfChunkSize = 0;
        bool hdfCollOnData = false;
        bool hdfCollOnMeta = true;
        int jsonBinaryDeflateLevel = 5;
        bool jsonUseCodecOnUInt8 = true;

        SerializerFactory() = default;
        SerializerFactory(const std::string &_type) : type(_type) {}

        DNDS_DECLARE_CONFIG(SerializerFactory)
        {
            DNDS_FIELD(type,                    "Serializer backend: \"JSON\" or \"H5\"",
                       DNDS::Config::enum_values({"JSON", "H5"}));
            DNDS_FIELD(hdfDeflateLevel,         "HDF5 deflate compression level",
                       DNDS::Config::range(0, 9));
            DNDS_FIELD(hdfChunkSize,            "HDF5 chunk size (0=auto)",
                       DNDS::Config::range(0));
            DNDS_FIELD(hdfCollOnData,           "HDF5 collective I/O on data arrays");
            DNDS_FIELD(hdfCollOnMeta,           "HDF5 collective I/O on metadata");
            DNDS_FIELD(jsonBinaryDeflateLevel,  "JSON binary deflate level",
                       DNDS::Config::range(0, 9));
            DNDS_FIELD(jsonUseCodecOnUInt8,     "Apply codec on uint8 arrays in JSON");
        }

        SerializerBaseSSP BuildSerializer(const MPIInfo &mpi)
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

        std::tuple<std::string, std::string> ModifyFilePath(std::string fname, const MPIInfo &mpi, std::string rank_part_fmt = "%06d", bool read = false)
        {
            if (type == "JSON")
            {
                std::filesystem::path outPath;
                outPath = {fname + ".dir"};
                if (!read)
                    std::filesystem::create_directories(outPath);
                char BUF[512];
                std::sprintf(BUF, rank_part_fmt.c_str(), mpi.rank);
                fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));
                return std::make_tuple(fname, getStringForcePath(outPath));
            }
            else if (type == "H5")
            {

                fname += ".dnds.h5";
                std::filesystem::path outPath = fname;
                if (!read)
                    std::filesystem::create_directories(outPath.parent_path() / ".");
                return std::make_tuple(fname, fname);
            }
            else
            {
                DNDS_assert_info(false, "type of serializer not existent: " + type);
                return std::make_tuple(std::string(""), std::string(""));
            }
        }
    };

}
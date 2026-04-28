#pragma once
/// @file SerializerFactory.hpp
/// @brief Configurable factory that builds either a @ref DNDS::SerializerJSON "SerializerJSON" or a
/// @ref DNDS::SerializerH5 "SerializerH5" with all tunables exposed through the DNDS config system.

#include "SerializerBase.hpp"
#include "SerializerJSON.hpp"
#include "SerializerH5.hpp"
#include "JsonUtil.hpp"
#include "DNDS/Config/ConfigParam.hpp"

namespace DNDS::Serializer
{
    /**
     * @brief Config-backed factory selecting between JSON and HDF5 serializers.
     *
     * @details Exposes the tunables of both backends under a single config
     * schema so users can switch formats by changing one JSON field. See
     * @ref DNDS_DECLARE_CONFIG body for the full list of fields.
     */
    struct SerializerFactory
    {
        /// @brief Backend selector: `"JSON"` or `"H5"`.
        std::string type = "JSON";
        /// @brief HDF5 gzip deflate level (0 = off, 9 = max).
        int hdfDeflateLevel = 0;
        /// @brief HDF5 chunk size (0 = library-chosen auto).
        int hdfChunkSize = 0;
        /// @brief Use collective HDF5 I/O for data arrays.
        bool hdfCollOnData = false;
        /// @brief Use collective HDF5 I/O for metadata (usually safe).
        bool hdfCollOnMeta = true;
        /// @brief Compression level used by the JSON backend's binary codec.
        int jsonBinaryDeflateLevel = 5;
        /// @brief Whether to apply the byte-codec to uint8 arrays in JSON (faster writes).
        bool jsonUseCodecOnUInt8 = true;

        SerializerFactory() = default;
        /// @brief Construct with a specific backend name; other fields stay at defaults.
        SerializerFactory(const std::string &_type) : type(_type) {}

        DNDS_DECLARE_CONFIG(SerializerFactory)
        {
            // clang-format off
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
            // clang-format on
        }

        /// @brief Instantiate the selected serializer and apply its tunables.
        /// @param mpi MPI context (used only by the H5 backend).
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

        /**
         * @brief Expand a user-supplied base file name into the backend-specific path layout.
         *
         * @details JSON uses one file per rank under a `<name>.dir/` directory
         * (`<rank>.json`); H5 produces a single `<name>.dnds.h5` file.
         *
         * @param fname         Base name supplied by the user (without suffix).
         * @param mpi           MPI context (rank used for JSON per-rank filename).
         * @param rank_part_fmt `printf`-style format used for the JSON rank-id component.
         * @param read          `true` for read (skips directory creation).
         * @return Tuple `(finalFilePath, displayPath)` -- the display path is
         *         the JSON dir or the H5 file, depending on backend.
         */
        std::tuple<std::string, std::string> ModifyFilePath(std::string fname, const MPIInfo &mpi, const std::string &rank_part_fmt = "%06d", bool read = false)
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
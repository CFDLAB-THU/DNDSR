#include "SerializerJSON.hpp"
#include <algorithm>
#include <string>
#include "base64_rfc4648.hpp"
#include <zlib.h>
#include <fmt/core.h>

namespace DNDS::Serializer
{

    /**
     * @brief returns good path and if the path is absolute
     */
    bool processPath(std::vector<std::string> &pth)
    {
        bool ifAbs = pth.empty() || pth[0].empty();
        pth.erase(
            std::remove_if(pth.begin(), pth.end(), [](const std::string &v)
                           { return v.empty(); }), // only keep non-zero sized path names
            pth.end());

        return ifAbs;
    }

    std::string constructPath(std::vector<std::string> &pth)
    {
        std::string ret;
        for (auto &name : pth)
            ret.append(std::string("/") + name);
        return ret;
    }

    void SerializerJSON::OpenFile(const std::string &fName, bool read)
    {
        DNDS_assert(!fileStream.is_open());
        reading = read;
        if (read)
            fileStream.open(fName, std::ios::in);
        else
            fileStream.open(fName, std::ios::out);
        DNDS_assert_info(fileStream.is_open(), fmt::format(" attempted to {} file [{}]", read ? "read" : "write", fName));

        if (reading)
            fileStream >> jObj;
        cP = "";
    }
    void SerializerJSON::CloseFile()
    {
        CloseFileNonVirtual();
    }
    void SerializerJSON::CloseFileNonVirtual()
    {
        DNDS_assert(fileStream.is_open());
        if (!reading)
            fileStream << jObj;
        fileStream.close();
        cPathSplit.clear();
        ptr_2_pth.clear();
        pth_2_ssp.clear();
        jObj.clear();
    }
    void SerializerJSON::CreatePath(const std::string &p)
    {
        auto pth = splitSString(p, '/');
        // std::cout << pth.size() << std::endl;
        // std::cout << cP + constructPath(pth) << std::endl;
        bool isAbs = processPath(pth);
        if (isAbs)
            jObj[nlohmann::json::json_pointer(constructPath(pth))] = nlohmann::json::object();
        else
            jObj[nlohmann::json::json_pointer(cP + constructPath(pth))] = nlohmann::json::object();
    }
    void SerializerJSON::GoToPath(const std::string &p)
    {
        auto pth = splitSString(p, '/');
        bool isAbs = processPath(pth);
        if (isAbs)
            cPathSplit = std::move(pth);
        else
            for (auto &name : pth)
                cPathSplit.push_back(name);
        cP = constructPath(cPathSplit);
        DNDS_assert(jObj[nlohmann::json::json_pointer(cP)].is_object());
    }
    std::string SerializerJSON::GetCurrentPath()
    {
        return cP;
    }

    void SerializerJSON::WriteInt(const std::string &name, int v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteIndex(const std::string &name, index v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteReal(const std::string &name, real v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteIndexVector(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteRealVector(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteString(const std::string &name, const std::string &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteSharedIndexVector(const std::string &name, const ssp<std::vector<index>> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        if (ptr_2_pth.count(v.get()))
            jObj[cPointer][name]["ref"] = ptr_2_pth[v.get()];
        else
        {
            jObj[cPointer][name] = *v;
            ptr_2_pth[v.get()] = cP + "/" + name;
        }
    }
    void SerializerJSON::WriteSharedRowsizeVector(const std::string &name, const ssp<std::vector<rowsize>> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        if (ptr_2_pth.count(v.get()))
            jObj[cPointer][name]["ref"] = ptr_2_pth[v.get()];
        else
        {
            jObj[cPointer][name] = *v;
            ptr_2_pth[v.get()] = cP + "/" + name;
        }
    }

    void SerializerJSON::ReadInt(const std::string &name, int &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_number_integer());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
    }
    void SerializerJSON::ReadIndex(const std::string &name, index &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_number_integer());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
    }
    void SerializerJSON::ReadReal(const std::string &name, real &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_number_float());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
    }
    void SerializerJSON::ReadIndexVector(const std::string &name, std::vector<index> &v, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_array());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::ReadRowsizeVector(const std::string &name, std::vector<rowsize> &v, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_array());
        v = jObj[nlohmann::json::json_pointer(cP)][name].get<std::remove_reference_t<decltype(v)>>();
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::ReadRealVector(const std::string &name, std::vector<real> &v, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_array());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::ReadString(const std::string &name, std::string &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_string());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
    }
    void SerializerJSON::ReadSharedIndexVector(const std::string &name, ssp<std::vector<index>> &v, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        using tValue = std::vector<index>;
        std::string refPath;
        if (jObj[cPointer][name].is_object() && !jObj[cPointer][name].is_array())
        {
            DNDS_assert(jObj[cPointer][name]["ref"].is_string());
            refPath = jObj[cPointer][name]["ref"].get<std::string>();
        }
        else
        {
            refPath = cP + "/" + name;
        }

        if (pth_2_ssp.count(refPath))
        {
            v = *((ssp<tValue> *)(pth_2_ssp[refPath]));
        }
        else
        {
            DNDS_assert(jObj[nlohmann::json::json_pointer(refPath)].is_array());
            v = std::make_shared<tValue>(jObj[nlohmann::json::json_pointer(refPath)].get<tValue>()); // vector's copy constructor
            pth_2_ssp[refPath] = &v;
        }
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::ReadSharedRowsizeVector(const std::string &name, ssp<std::vector<rowsize>> &v, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        using tValue = std::vector<rowsize>;
        std::string refPath;
        if (jObj[cPointer][name].is_object() && !jObj[cPointer][name].is_array())
        {
            DNDS_assert(jObj[cPointer][name]["ref"].is_string());
            refPath = jObj[cPointer][name]["ref"].get<std::string>();
        }
        else
        {
            refPath = cP + "/" + name;
        }

        if (pth_2_ssp.count(refPath))
        {
            v = *((ssp<tValue> *)(pth_2_ssp[refPath]));
        }
        else
        {
            DNDS_assert(jObj[nlohmann::json::json_pointer(refPath)].is_array());
            v = std::make_shared<tValue>(jObj[nlohmann::json::json_pointer(refPath)].get<tValue>()); // vector's copy constructor
            pth_2_ssp[refPath] = &v;
        }
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::WriteUint8Array(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset)
    {
        auto zlibCompressedSize = [&](index sizeC)
        {
            return sizeC + (sizeC + 999) / 1000 + 12; // form vtk
        };
        int compressLevel = 5;
        auto zlibCompressData = [&](const uint8_t *buf, index sizeC)
        {
            std::vector<uint8_t> ret(zlibCompressedSize(sizeC));
            uLongf retSize = ret.size();
            auto ret_v = compress2(ret.data(), &retSize, buf, sizeC, compressLevel);
            if (ret_v != Z_OK)
                DNDS_assert_info(false, "compression failed");
            ret.resize(retSize);
            return ret;
        };

        auto cPointer = nlohmann::json::json_pointer(cP);
        if (!useCodecOnUint8)
            jObj[cPointer][name] = std::vector<uint8_t>(data, data + size);
        else
        {
            jObj[cPointer][name]["size"] = size;
            jObj[cPointer][name]["encoded"] = cppcodec::base64_rfc4648::encode(
                zlibCompressData(data, size));
        }
    }
    void SerializerJSON::ReadUint8Array(const std::string &name, uint8_t *data, index &size, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_object() || jObj[cPointer][name].is_array());
        if (jObj[cPointer][name].is_array())
        {
            size = jObj[cPointer][name].size();
            if (data != nullptr)
                std::copy(jObj[cPointer][name].begin(), jObj[cPointer][name].end(), data);
        }
        else if (jObj[cPointer][name].is_object() &&
                 jObj[cPointer][name]["size"].is_number_integer() &&
                 jObj[cPointer][name]["encoded"].is_string())
        {
            size = jObj[cPointer][name]["size"];
            auto &dataIn = jObj[cPointer][name]["encoded"].get_ref<nlohmann::json::string_t &>();
            if (data != nullptr)
            {
                std::vector<uint8_t> decodedData;
                cppcodec::base64_rfc4648::decode(decodedData, dataIn);
                std::vector<uint8_t> decodedDataUncompress(size);
                uLongf sizeU{(uLongf)size};
                auto ret = uncompress(decodedDataUncompress.data(), &sizeU, decodedData.data(), decodedData.size());
                DNDS_assert_info(ret == Z_OK, "zlib uncompress failed");
                DNDS_assert_info(sizeU == size, "zlib uncompress failed");
                // DNDS_assert(decodedData.size() == size);
                std::copy(decodedDataUncompress.begin(), decodedDataUncompress.end(), data);
            }
        }
        else
        {
            DNDS_assert(false);
        }
        offset = ArrayGlobalOffset_Unknown;
    }
} // namespace DNDS::Serializer
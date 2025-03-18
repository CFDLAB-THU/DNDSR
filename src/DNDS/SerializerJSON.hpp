#pragma once
#include "SerializerBase.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <map>

namespace DNDS::Serializer
{
    class SerializerJSON : public SerializerBase
    {
        std::fstream fileStream;
        nlohmann::json jObj;
        bool reading = true;
        std::vector<std::string> cPathSplit;
        std::string cP; // current path
        std::map<void *, std::string> ptr_2_pth;
        std::map<std::string, void *> pth_2_ssp;

        bool useCodecOnUint8{false};
        int deflateLevel{5};

    public:
        void SetUseCodecOnUint8(bool v) { useCodecOnUint8 = v; }
        void SetDeflateLevel(int v) { deflateLevel = v; }

        void OpenFile(const std::string &fName, bool read) override;
        void CloseFile() override;
        void CloseFileNonVirtual();
        void CreatePath(const std::string &p) override;
        void GoToPath(const std::string &p) override;
        bool IsPerRank() override { return true; }
        std::string GetCurrentPath() override;

        void WriteInt(const std::string &name, int v) override;
        void WriteIndex(const std::string &name, index v) override;
        void WriteReal(const std::string &name, real v) override;
        void WriteString(const std::string &name, const std::string &v) override;

        void WriteIndexVector(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset) override;
        void WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset) override;
        void WriteRealVector(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset) override;
        void WriteSharedIndexVector(const std::string &name, const ssp<std::vector<index>> &v, ArrayGlobalOffset offset) override;
        void WriteSharedRowsizeVector(const std::string &name, const ssp<std::vector<rowsize>> &v, ArrayGlobalOffset offset) override;

        void WriteUint8Array(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset) override;

        void WriteIndexVectorPerRank(const std::string &name, const std::vector<index> &v) override
        {
            this->WriteIndexVector(name, v, ArrayGlobalOffset_Unknown);
        }

        void ReadInt(const std::string &name, int &v) override;
        void ReadIndex(const std::string &name, index &v) override;
        void ReadReal(const std::string &name, real &v) override;
        void ReadString(const std::string &name, std::string &v) override;

        void ReadIndexVector(const std::string &name, std::vector<index> &v, ArrayGlobalOffset &offset) override;
        void ReadRowsizeVector(const std::string &name, std::vector<rowsize> &v, ArrayGlobalOffset &offset) override;
        void ReadRealVector(const std::string &name, std::vector<real> &v, ArrayGlobalOffset &offset) override;
        void ReadSharedIndexVector(const std::string &name, ssp<std::vector<index>> &v, ArrayGlobalOffset &offset) override;
        void ReadSharedRowsizeVector(const std::string &name, ssp<std::vector<rowsize>> &v, ArrayGlobalOffset &offset) override;

        void ReadUint8Array(const std::string &name, uint8_t *data, index &size, ArrayGlobalOffset &offset) override;

        ~SerializerJSON() override
        {
            CloseFileNonVirtual();
        }
    };
}
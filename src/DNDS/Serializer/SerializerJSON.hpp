#pragma once
/// @file SerializerJSON.hpp
/// @brief Per-rank JSON serializer implementing the SerializerBase interface.
/// @par Unit Test Coverage (test_Serializer.cpp, MPI np=1,2,4)
/// - Scalar round-trip: WriteInt/ReadInt, WriteIndex/ReadIndex,
///   WriteReal/ReadReal, WriteString/ReadString
/// - Vector round-trip: WriteRealVector, WriteIndexVector, WriteRowsizeVector
/// - uint8 array: with and without codec (base64 + zlib)
/// - Path operations: CreatePath, GoToPath, GetCurrentPath, ListCurrentPath
/// - Shared pointer deduplication: WriteSharedIndexVector / ReadSharedIndexVector
/// @par Not Yet Tested
/// - WriteSharedRowsizeVector / ReadSharedRowsizeVector
/// - WriteIndexVectorPerRank
/// - SetDeflateLevel impact verification
/// - Error handling (nonexistent file, duplicate paths)
#include "SerializerBase.hpp"

#include "DNDS/Serializer/JsonUtil.hpp"
#include <fstream>
#include <map>

namespace DNDS::Serializer
{
    /// @brief Per-rank JSON file serializer; each MPI rank writes its own .json file.
    class SerializerJSON : public SerializerBase
    {
        std::fstream fileStream;
        nlohmann::json jObj;
        bool reading = true;
        std::vector<std::string> cPathSplit;
        std::string cP; // current path
        // ptr_2_pth and pth_2_ssp are inherited from SerializerBase

        bool useCodecOnUint8{false};
        int deflateLevel{5};

        MPIInfo mpi; // NULL

    public:
        // Rule-of-five closure. Owns `fstream` + JSON DOM; copy / move
        // are inherited-deleted from SerializerBase but must be re-declared
        // explicitly because the non-trivial dtor suppresses implicit defaults.
        SerializerJSON() = default;
        SerializerJSON(const SerializerJSON &) = delete;
        SerializerJSON &operator=(const SerializerJSON &) = delete;
        SerializerJSON(SerializerJSON &&) = delete;
        SerializerJSON &operator=(SerializerJSON &&) = delete;

        void SetUseCodecOnUint8(bool v) { useCodecOnUint8 = v; }
        void SetDeflateLevel(int v) { deflateLevel = v; }

        void OpenFile(const std::string &fName, bool read) override;
        void CloseFile() override;
        void CloseFileNonVirtual();
        void CreatePath(const std::string &p) override;
        void GoToPath(const std::string &p) override;
        bool IsPerRank() override { return true; }
        std::string GetCurrentPath() override;
        std::set<std::string> ListCurrentPath() override;
        int GetMPIRank() override { return -1; }
        int GetMPISize() override { return -1; }
        const MPIInfo &getMPI() override { return mpi; }

        void WriteInt(const std::string &name, int v) override;
        void WriteIndex(const std::string &name, index v) override;
        void WriteReal(const std::string &name, real v) override;
        void WriteString(const std::string &name, const std::string &v) override;

        void WriteIndexVector(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset) override;
        void WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset) override;
        void WriteRealVector(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset) override;
        void WriteSharedIndexVector(const std::string &name, const ssp<host_device_vector<index>> &v, ArrayGlobalOffset offset) override;
        void WriteSharedRowsizeVector(const std::string &name, const ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset offset) override;

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
        void ReadSharedIndexVector(const std::string &name, ssp<host_device_vector<index>> &v, ArrayGlobalOffset &offset) override;
        void ReadSharedRowsizeVector(const std::string &name, ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset &offset) override;

        void ReadUint8Array(const std::string &name, uint8_t *data, index &size, ArrayGlobalOffset &offset) override;

        ~SerializerJSON() override
        {
            // Destructors must not throw; swallow any exception from the
            // close path (fstream flush, bad-path cleanup). A failure here
            // is reported when the user invoked CloseFile() explicitly.
            // NOLINTBEGIN(bugprone-empty-catch)
            try
            {
                CloseFileNonVirtual();
            }
            catch (...)
            {
            }
            // NOLINTEND(bugprone-empty-catch)
        }
    };
}
#pragma once
/// @file SerializerH5.hpp
/// @brief MPI-parallel HDF5 serializer implementing the SerializerBase interface.
/// @par Unit Test Coverage (test_Serializer.cpp, MPI np=1,2,4)
/// - Scalar round-trip: WriteInt/ReadInt, WriteIndex/ReadIndex,
///   WriteReal/ReadReal, WriteString/ReadString
/// - Vector round-trip: WriteRealVector, WriteIndexVector with explicit
///   ArrayGlobalOffset; ReadRealVector/ReadIndexVector with ArrayGlobalOffset_Unknown
/// - Distributed vector: non-uniform per-rank sizes, write with explicit offset,
///   read with both ArrayGlobalOffset_Unknown (auto-detect from ::rank_offsets)
///   and explicit offset
/// - uint8 distributed round-trip: two-pass read (nullptr size query, then read)
/// - Path operations: CreatePath, GoToPath, GetCurrentPath, ListCurrentPath
///   (groups materialized by writing content), WriteInt/ReadInt on nested paths
/// - String round-trip: WriteString/ReadString with fixed-length HDF5 attributes
/// @par Collective I/O and zero-size partitions
/// All Read/Write vector and byte-array methods use MPI-collective HDF5 calls.
/// Every rank must call them in the same order, even when its local element
/// count is 0 (which happens when nGlobal < nRanks under EvenSplit).
///
/// Internally, ReadDataVector uses a two-pass pattern:
///   - Pass 1 (buf == nullptr): queries dataset size and resolves the offset.
///   - Pass 2 (buf != nullptr): performs the collective H5Dread.
///
/// When local size is 0, std::vector<>::data() / host_device_vector<>::data()
/// may return nullptr, which would skip the H5Dread block (guarded by
/// `if (buf != nullptr)`) and hang the other ranks. To prevent this, each
/// Read*Vector / ReadShared*Vector caller passes a dummy stack pointer when
/// size == 0. Callers of ReadUint8Array must do the same (see SerializerBase).
///
/// H5_ReadDataset and H5_WriteDataset accept nLocal == 0: the hyperslab
/// selection with count == 0 selects nothing, so no data is transferred,
/// but the rank still participates in the collective call.
///
/// @par Not Yet Tested
/// - SetChunkAndDeflate impact, SetCollectiveRW impact
/// - WriteSharedIndexVector / ReadSharedIndexVector (H5 deduplication)
/// - WriteRowsizeVector / ReadRowsizeVector
/// - WriteIndexVectorPerRank
/// - ArrayGlobalOffset_One semantics (rank-0-only write)
/// - ArrayGlobalOffset_Parts auto-offset
#include "SerializerBase.hpp"

#include <hdf5.h>
#include <fstream>
#include <map>
#include <set>

namespace DNDS::Serializer
{
    /// @brief MPI-parallel HDF5 serializer; all ranks collectively read/write a single .h5 file.
    class SerializerH5 : public SerializerBase
    {
        bool reading = true;
        std::vector<std::string> cPathSplit;
        std::string cP; // current path
        std::map<void *, std::string> ptr_2_pth;
        std::map<std::string, void *> pth_2_ssp;

        MPIInfo mpi;
        MPI_Comm commDup{MPI_COMM_NULL};

        hid_t h5file{NULL};

        int64_t chunksize{0};
        int deflateLevel{0};
        bool collectiveMetadataRW = true;
        bool collectiveDataRW = false;

    public:
        SerializerH5(const MPIInfo &_mpi) : SerializerBase(), mpi(_mpi)
        {
            MPI_Comm_dup(mpi.comm, &commDup);
        }

        void SetChunkAndDeflate(int64_t n_chunksize, int n_deflateLevel)
        {
            if (n_deflateLevel > 0)
                DNDS_assert_info(n_chunksize > 0, "chunksize must be positive when using deflate!");
            chunksize = n_chunksize;
            deflateLevel = n_deflateLevel;
        }

        void SetCollectiveRW(bool metadata, bool data)
        {
            collectiveMetadataRW = metadata;
            collectiveDataRW = data;
        }

        void OpenFile(const std::string &fName, bool read) override;
        void CloseFile() override;
        void CloseFileNonVirtual();
        void CreatePath(const std::string &p) override;
        void GoToPath(const std::string &p) override;
        bool IsPerRank() override { return false; }
        std::string GetCurrentPath() override;
        std::set<std::string> ListCurrentPath() override;
        int GetMPIRank() override { return mpi.rank; }
        int GetMPISize() override { return mpi.size; }
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
            this->WriteIndexVector(name, v, ArrayGlobalOffset{index(mpi.size) * index(v.size()), index(mpi.rank) * index(v.size())});
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

        ~SerializerH5() override
        {
            CloseFileNonVirtual();
            MPI_Comm_free(&commDup);
        }
    };
}
#pragma once
/// @file ArrayRedistributor.hpp
/// @brief Redistributes ArrayPair data across different MPI partitions using ArrayTransformer.
///
/// Enables reading HDF5 checkpoint data written with partition A (np=N) into
/// partition B (np=M) by using an "original index" (e.g., CGNS serial cell index)
/// as a partition-independent key.
///
/// The redistribution uses a two-step rendezvous pattern implemented via
/// temporary ArrayTransformer pull operations:
///   Step 1: Each rank reads an even slice of the H5 global data.
///   Step 2: A temporary ArrayTransformer pulls each rank's needed cells
///           (identified by origIndex) from whichever rank read them.

#include "ArrayTransformer.hpp"
#include "Serializer/SerializerBase.hpp"
#include <unordered_map>
#include <algorithm>
#include <numeric>

namespace DNDS
{

    // EvenSplitRange is defined in Defines.hpp

    /// @brief Redistributes ArrayPair data between different MPI partitions using ArrayTransformer.
    ///
    /// The caller provides:
    /// - readOrigIndex: the original indices for data this rank read from HDF5 (even-split)
    /// - readArray:     the data this rank read (father of a temporary ArrayPair)
    /// - newOrigIndex:  the original indices this rank needs (from current mesh's cell2cellOrig)
    ///
    /// The redistributor builds a temporary ArrayTransformer to pull needed data:
    /// 1. Build a global mapping on the "read" arrays (readArray serves as father).
    /// 2. For each entry in newOrigIndex, find which rank's read slice contains it, and
    ///    compute the corresponding global index in the read array's global space.
    /// 3. Use ArrayTransformer::createGhostMapping + pullOnce to fetch the data.
    /// 4. Copy the pulled data into the output array in newOrigIndex order.
    ///
    /// This works for fixed-row-size, dynamic-row-size, and CSR arrays alike, since
    /// ArrayTransformer handles all these layouts internally.

    /// @brief Builds the pulling index mapping for redistribution.
    ///
    /// Given the origIndex arrays from the even-split read (distributed across all ranks),
    /// and the newOrigIndex for this rank, determines which global indices in the
    /// read array space need to be pulled.
    ///
    /// Uses a 3-round MPI_Alltoallv rendezvous:
    ///   Round 1: Send (origIdx, globalReadIdx) pairs to directory ranks.
    ///   Round 2: Send query origIndex entries to directory ranks.
    ///   Round 3: Directory ranks reply with resolved globalReadIdx.
    ///
    /// @par Memory scaling
    /// Temporary memory per rank is O(nLocal) where nLocal is the larger of
    /// readOrigIndex.size() and newOrigIndex.size(). Specifically:
    ///   - 7 index vectors of size O(nLocal): sendBuf, recvBuf, querySendBuf,
    ///     queryRecvBuf, queryReplyBuf, replyRecvBuf, pullingIndexGlobal
    ///   - 1 unordered_map<index,index> of size O(nGlobal/nRanks) for the directory
    ///   - Total: ~7 × nLocal × sizeof(index) + O(nGlobal/nRanks) hash table overhead
    /// For a 100M-cell mesh with 8 ranks: ~12.5M entries × 8 B × 7 ≈ 700 MB per rank,
    /// plus ~200 MB for the directory hash table.
    ///
    /// @param mpi           MPI communicator info
    /// @param readOrigIndex Original indices from the even-split read (this rank's slice)
    /// @param newOrigIndex  Original indices this rank needs (from current partition)
    /// @param readGlobalMapping Global mapping of the read array (prefix-sum offsets)
    /// @return pullingIndexGlobal: vector of global read-array indices to pull
    ///         (ordered to match newOrigIndex)
    inline std::vector<index> BuildRedistributionPullingIndex(
        const MPIInfo &mpi,
        const std::vector<index> &readOrigIndex,
        const std::vector<index> &newOrigIndex,
        const ssp<GlobalOffsetsMapping> &readGlobalMapping)
    {
        // Build a distributed directory mapping origIdx -> global read index via
        // a rendezvous pattern using MPI_Alltoallv.
        //
        // Each rank holds readOrigIndex for its read slice. The global read index
        // for readOrigIndex[i] on rank r is: readGlobalMapping(r, i).
        //
        // Directory rank assignment: directoryRank(origIdx) = origIdx * nRanks / nGlobal
        // (block partitioning for good locality).
        //
        // Steps:
        //   1. Send (origIdx, globalReadIdx) pairs to directory ranks via Alltoallv.
        //   2. Send newOrigIndex queries to directory ranks via Alltoallv.
        //   3. Directory ranks look up and reply with globalReadIdx.

        const int nRanks = mpi.size;
        index nGlobal = readGlobalMapping->globalSize();
        DNDS_assert_info(nGlobal > 0, "Redistribution requires nGlobal > 0");
        auto directoryRank = [&](index origIdx) -> int
        {
            if (nGlobal == 0)
                return 0;
            return static_cast<int>(std::min(index(nRanks - 1), origIdx * index(nRanks) / nGlobal));
        };

        // Step 3: Send read entries to directory via Alltoallv.
        // Each entry is (origIdx, globalReadIdx).

        // Count entries per directory rank
        std::vector<int> sendCounts(nRanks, 0);
        for (index i = 0; i < index(readOrigIndex.size()); i++)
        {
            int dr = directoryRank(readOrigIndex[i]);
            sendCounts[dr]++;
        }

        std::vector<int> sendDisps(nRanks + 1, 0);
        std::partial_sum(sendCounts.begin(), sendCounts.end(), sendDisps.begin() + 1);

        // Pack send buffers (origIdx, globalReadIdx) interleaved
        std::vector<index> sendBuf(index(sendDisps[nRanks]) * 2);
        std::vector<int> sendPos(nRanks, 0);
        for (index i = 0; i < index(readOrigIndex.size()); i++)
        {
            int dr = directoryRank(readOrigIndex[i]);
            index pos = sendDisps[dr] + sendPos[dr];
            sendBuf[pos * 2] = readOrigIndex[i];
            sendBuf[pos * 2 + 1] = readGlobalMapping->operator()(mpi.rank, i);
            sendPos[dr]++;
        }

        // Alltoall sizes
        std::vector<int> recvCounts(nRanks, 0);
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, mpi.comm);

        std::vector<int> recvDisps(nRanks + 1, 0);
        std::partial_sum(recvCounts.begin(), recvCounts.end(), recvDisps.begin() + 1);

        // Alltoallv to send pairs
        // Multiply counts/disps by 2 for the interleaved pairs
        std::vector<int> sendCounts2(nRanks), sendDisps2(nRanks);
        std::vector<int> recvCounts2(nRanks), recvDisps2(nRanks);
        for (int r = 0; r < nRanks; r++)
        {
            sendCounts2[r] = sendCounts[r] * 2;
            sendDisps2[r] = sendDisps[r] * 2;
            recvCounts2[r] = recvCounts[r] * 2;
            recvDisps2[r] = recvDisps[r] * 2;
        }

        std::vector<index> recvBuf(index(recvDisps[nRanks]) * 2);
        MPI_Alltoallv(sendBuf.data(), sendCounts2.data(), sendDisps2.data(), DNDS_MPI_INDEX,
                      recvBuf.data(), recvCounts2.data(), recvDisps2.data(), DNDS_MPI_INDEX,
                      mpi.comm);

        // Step 4: Build directory lookup: origIdx -> globalReadIdx
        std::unordered_map<index, index> directoryMap;
        directoryMap.reserve(recvDisps[nRanks]);
        for (index i = 0; i < recvDisps[nRanks]; i++)
        {
            directoryMap[recvBuf[i * 2]] = recvBuf[i * 2 + 1];
        }

        // Step 5: Send queries from newOrigIndex to directory, get back globalReadIdx.
        // Count queries per directory rank
        std::vector<int> querySendCounts(nRanks, 0);
        for (index i = 0; i < index(newOrigIndex.size()); i++)
        {
            int dr = directoryRank(newOrigIndex[i]);
            querySendCounts[dr]++;
        }

        std::vector<int> querySendDisps(nRanks + 1, 0);
        std::partial_sum(querySendCounts.begin(), querySendCounts.end(), querySendDisps.begin() + 1);

        // Pack query send buffer and record the order mapping
        std::vector<index> querySendBuf(querySendDisps[nRanks]);
        // orderMap[pos_in_send] = index in newOrigIndex
        std::vector<index> queryOrderMap(querySendDisps[nRanks]);
        std::vector<int> queryPos(nRanks, 0);
        for (index i = 0; i < index(newOrigIndex.size()); i++)
        {
            int dr = directoryRank(newOrigIndex[i]);
            index pos = querySendDisps[dr] + queryPos[dr];
            querySendBuf[pos] = newOrigIndex[i];
            queryOrderMap[pos] = i;
            queryPos[dr]++;
        }

        // Alltoall query sizes
        std::vector<int> queryRecvCounts(nRanks, 0);
        MPI_Alltoall(querySendCounts.data(), 1, MPI_INT, queryRecvCounts.data(), 1, MPI_INT, mpi.comm);

        std::vector<int> queryRecvDisps(nRanks + 1, 0);
        std::partial_sum(queryRecvCounts.begin(), queryRecvCounts.end(), queryRecvDisps.begin() + 1);

        // Alltoallv queries
        std::vector<index> queryRecvBuf(queryRecvDisps[nRanks]);
        MPI_Alltoallv(querySendBuf.data(), querySendCounts.data(), querySendDisps.data(), DNDS_MPI_INDEX,
                      queryRecvBuf.data(), queryRecvCounts.data(), queryRecvDisps.data(), DNDS_MPI_INDEX,
                      mpi.comm);

        // Step 6: Directory ranks look up and reply with globalReadIdx.
        std::vector<index> queryReplyBuf(queryRecvDisps[nRanks]);
        for (index i = 0; i < queryRecvDisps[nRanks]; i++)
        {
            auto it = directoryMap.find(queryRecvBuf[i]);
            DNDS_assert_info(it != directoryMap.end(),
                             fmt::format("origIdx {} not found in directory on rank {}", queryRecvBuf[i], mpi.rank));
            queryReplyBuf[i] = it->second;
        }

        // Alltoallv replies back (reverse direction)
        std::vector<index> replyRecvBuf(querySendDisps[nRanks]);
        MPI_Alltoallv(queryReplyBuf.data(), queryRecvCounts.data(), queryRecvDisps.data(), DNDS_MPI_INDEX,
                      replyRecvBuf.data(), querySendCounts.data(), querySendDisps.data(), DNDS_MPI_INDEX,
                      mpi.comm);

        // Step 7: Build pullingIndexGlobal in newOrigIndex order.
        std::vector<index> pullingIndexGlobal(newOrigIndex.size());
        for (index i = 0; i < querySendDisps[nRanks]; i++)
        {
            pullingIndexGlobal[queryOrderMap[i]] = replyRecvBuf[i];
        }

        return pullingIndexGlobal;
    }

    /// @brief Redistributes an ArrayPair from a source partition to a target partition.
    ///
    /// Template-free core logic: given a "read" father array (from even-split H5 read)
    /// with known origIndex, and a target partition's newOrigIndex, uses
    /// ArrayTransformer to pull data from wherever it was read to where it's needed.
    ///
    /// @tparam TArray  The ParArray type (e.g., ParArray<real, DynamicSize>)
    /// @param mpi            MPI communicator info
    /// @param readFather     Father array populated from even-split H5 read
    /// @param readOrigIndex  Original indices for each row of readFather
    /// @param newOrigIndex   Original indices this rank needs (from current partition)
    /// @param outFather      Output father array, pre-allocated with correct size and row structure
    template <class TArray>
    void RedistributeArrayWithTransformer(
        const MPIInfo &mpi,
        ssp<TArray> readFather,
        const std::vector<index> &readOrigIndex,
        const std::vector<index> &newOrigIndex,
        ssp<TArray> outFather)
    {
        DNDS_assert(readFather);
        DNDS_assert(outFather);
        DNDS_assert(index(readOrigIndex.size()) == readFather->Size());

        // 1. Create global mapping for the read father array
        readFather->createGlobalMapping();
        auto readGlobalMapping = readFather->pLGlobalMapping;

        // 2. Build the pulling index: for each newOrigIndex[i], find the global read index
        std::vector<index> pullingIndexGlobal = BuildRedistributionPullingIndex(
            mpi, readOrigIndex, newOrigIndex, readGlobalMapping);

        // Save the original order before createGhostMapping sorts it
        std::vector<index> pullingIndexOrig(pullingIndexGlobal);

        // 3. Create a temporary son array and transformer to do the pull
        auto readSon = std::make_shared<TArray>(mpi);

        using TArrayTransformer = typename ArrayTransformerType<TArray>::Type;
        TArrayTransformer trans;
        trans.setFatherSon(readFather, readSon);
        trans.createFatherGlobalMapping(); // reuses the mapping we created above
        trans.createGhostMapping(pullingIndexGlobal);
        // NOTE: createGhostMapping sorts pullingIndexGlobal in-place!
        // After pull, readSon data is in sorted-global-index order, NOT newOrigIndex order.
        trans.createMPITypes();
        trans.pullOnce();

        // 4. Copy pulled data (in son) to outFather in the correct order.
        //    readSon is in sorted ghostIndex order. For each newOrigIndex[i],
        //    find pullingIndexOrig[i] in the sorted ghostIndex to get the son position.
        DNDS_assert(outFather->Size() == index(newOrigIndex.size()));

        // Build a lookup from sorted ghostIndex to son position
        const auto &ghostIndex = trans.pLGhostMapping->ghostIndex;
        std::unordered_map<index, index> globalIdx2SonPos;
        globalIdx2SonPos.reserve(ghostIndex.size());
        for (index j = 0; j < index(ghostIndex.size()); j++)
            globalIdx2SonPos[ghostIndex[j]] = j;

        // For CSR arrays, set the output row sizes from the pulled son data,
        // then compress, before copying element values.
        if constexpr (TArray::_dataLayout == CSR)
        {
            outFather->ResizeRowsAndCompress(
                [&](index i) -> rowsize
                {
                    auto it = globalIdx2SonPos.find(pullingIndexOrig[i]);
                    DNDS_assert(it != globalIdx2SonPos.end());
                    return readSon->RowSize(it->second);
                });
        }

        for (index i = 0; i < index(newOrigIndex.size()); i++)
        {
            index globalReadIdx = pullingIndexOrig[i];
            auto it = globalIdx2SonPos.find(globalReadIdx);
            DNDS_assert_info(it != globalIdx2SonPos.end(),
                             fmt::format("globalReadIdx {} not found in ghostIndex on rank {}", globalReadIdx, mpi.rank));
            index sonPos = it->second;
            outFather->CopyRowFrom(i, *readSon, sonPos);
        }
    }

}

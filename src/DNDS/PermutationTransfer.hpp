#pragma once
/// @file PermutationTransfer.hpp
/// @brief Utility for distributed or local row permutation/transfer of arrays.
///
/// Encapsulates the common pattern: given a partition assignment (or forward map)
/// for a set of entities, compute new global indices and transfer array rows to
/// their target ranks. Supports both distributed (MPI push) and local-only
/// (in-place permutation) paths.

#include "DNDS/ArrayTransformer.hpp"
#include "DNDS/ArrayPair.hpp"
#include "DNDS/MPI.hpp"

namespace DNDS
{
    /// Encapsulates a distributed or local permutation of array rows.
    ///
    /// Given a per-slot target rank assignment, computes:
    /// - New global indices (prefix-sum across ranks grouped by target)
    /// - Push CSR indices for MPI communication
    /// - Local permutation vector (when all rows stay on the same rank)
    ///
    /// Then provides `transferRows` to actually move/permute the data,
    /// and `buildLookup` to create a ghost-pullable old->new global map.
    struct PermutationTransfer
    {
        /// Per father slot: target rank after reorder.
        std::vector<MPI_int> targetRanks;

        /// New global index for each father slot (same size as targetRanks).
        std::vector<index> newGlobalIndices;

        /// Push CSR: pushIndex[pushStart[r]..pushStart[r+1]) are the local
        /// father indices that go to rank r.
        std::vector<index> pushIndex;
        std::vector<index> pushStart; // size = nRanks + 1

        /// Local permutation: localOld2New[i] = new local index for old local i.
        /// Only valid when isLocalOnly == true. Empty otherwise.
        std::vector<index> localOld2New;

        /// Whether this is a pure rank-local permutation (no cross-rank traffic).
        bool isLocalOnly{false};

        /// New global offsets: newGlobalOffsets[r] = first global index owned by
        /// rank r after reorder. Size = nRanks + 1.
        std::vector<index> newGlobalOffsets;

        // =============================================================
        // Factory methods
        // =============================================================

        /// Build from partition assignment (target ranks only).
        /// New global indices are computed automatically via prefix-sum.
        ///
        /// @param partition      Per-slot target rank. Size == father size.
        /// @param oldGlobalMapping  Current global offsets mapping for this entity.
        /// @param mpi            MPI communicator.
        /// @warning Collective.
        static PermutationTransfer fromPartition(
            const std::vector<MPI_int> &partition,
            const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
            const MPIInfo &mpi);

        /// Build from a local-only permutation vector.
        /// All entities stay on the same rank. targetRanks all == mpi.rank.
        ///
        /// @param old2new        Local permutation: old local index -> new local index.
        ///                       Must be a valid permutation of [0, N).
        /// @param oldGlobalMapping  Current global offsets mapping.
        /// @param mpi            MPI communicator (used only for rank/size, no communication).
        /// @note Non-collective: performs zero MPI communication.
        static PermutationTransfer fromLocalPermutation(
            const std::vector<index> &old2new,
            const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
            const MPIInfo &mpi);

        // =============================================================
        // Core operations
        // =============================================================

        /// Transfer (or permute) rows of an ArrayPair.
        ///
        /// - Local-only: in-place row permutation via PermuteRows.
        /// - Distributed: father=old, son=new ArrayTransformer push trick.
        ///
        /// After return, pair.father contains the new data.
        /// pair.son is reset (stale after distributed transfer).
        ///
        /// @warning Collective (when !isLocalOnly).
        template <class TPair>
        void transferRows(TPair &pair, const MPIInfo &mpi) const;

        /// Result of buildLookup: ghost-pullable old-global -> new-global map.
        struct LookupResult
        {
            ArrayAdjacencyPair<1> pair; // pair(localSlot, 0) = newGlobalIndices[localSlot]
            // Ghost-pulled for off-rank entries.

            /// Resolve an old global index to its new global index.
            /// The old global must be in the local father or ghost (son) range.
            index resolve(index oldGlobal) const
            {
                DNDS_assert(pair.trans.pLGhostMapping);
                if (oldGlobal == UnInitIndex)
                    return UnInitIndex;
                MPI_int rank;
                index val;
                bool found = pair.trans.pLGhostMapping->search_indexAppend(
                    oldGlobal, rank, val);
                DNDS_assert_info(found,
                                 fmt::format("LookupResult::resolve: old global {} not found", oldGlobal));
                return pair(val, 0);
            }
        };

        /// Build a ghost-pullable lookup array for old->new global conversion.
        ///
        /// @param pullSet  Sorted, deduplicated set of off-rank old globals that
        ///                 need to be resolvable. Typically collected from adj
        ///                 entries pointing to this entity kind.
        /// @param mpi      MPI communicator.
        /// @return LookupResult with resolve() method.
        /// @warning Collective.
        LookupResult buildLookup(
            const std::vector<index> &pullSet,
            const MPIInfo &mpi) const;

        // =============================================================
        // Queries
        // =============================================================

        /// Number of entities (father slots) in this transfer.
        [[nodiscard]] index size() const { return static_cast<index>(targetRanks.size()); }
    };

    // =====================================================================
    // Implementation: fromPartition
    // =====================================================================

    inline PermutationTransfer PermutationTransfer::fromPartition(
        const std::vector<MPI_int> &partition,
        const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
        const MPIInfo &mpi)
    {
        DNDS_assert(oldGlobalMapping);
        PermutationTransfer pt;
        pt.targetRanks = partition;
        const index nLocal = static_cast<index>(partition.size());

        // --- Push CSR ---
        std::vector<index> pushSizes(mpi.size, 0);
        for (auto r : partition)
        {
            DNDS_assert(r >= 0 && r < mpi.size);
            pushSizes[r]++;
        }
        AccumulateRowSize(pushSizes, pt.pushStart);
        pt.pushIndex.resize(pt.pushStart[mpi.size]);
        pushSizes.assign(mpi.size, 0);
        for (index i = 0; i < nLocal; i++)
            pt.pushIndex[pt.pushStart[partition[i]] + (pushSizes[partition[i]]++)] = i;

        // --- New global indices (prefix-sum grouped by target rank) ---
        // Count how many entities each rank sends to each target:
        std::vector<index> localCounts(mpi.size, 0);
        for (auto r : partition)
            localCounts[r]++;

        // Total entities per target rank (across all senders):
        std::vector<index> totalCounts(mpi.size);
        MPI_Allreduce(localCounts.data(), totalCounts.data(), mpi.size,
                      DNDS_MPI_INDEX, MPI_SUM, mpi.comm);

        // Global offsets per target rank:
        pt.newGlobalOffsets.resize(mpi.size + 1);
        pt.newGlobalOffsets[0] = 0;
        for (int r = 0; r < mpi.size; r++)
            pt.newGlobalOffsets[r + 1] = pt.newGlobalOffsets[r] + totalCounts[r];

        // Exclusive prefix per target rank (how many entities before mine):
        std::vector<index> prevCounts(mpi.size);
        MPI_Scan(localCounts.data(), prevCounts.data(), mpi.size,
                 DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        // MPI_Scan is inclusive, convert to exclusive:
        for (int r = 0; r < mpi.size; r++)
            prevCounts[r] -= localCounts[r];

        // Assign new globals: within each target rank bucket, entities are
        // ordered by (sender rank, local slot within sender).
        pt.newGlobalIndices.resize(nLocal);
        std::vector<index> fillCounters(mpi.size, 0);
        for (index i = 0; i < nLocal; i++)
        {
            MPI_int target = partition[i];
            pt.newGlobalIndices[i] =
                pt.newGlobalOffsets[target] + prevCounts[target] + fillCounters[target];
            fillCounters[target]++;
        }

        // --- Detect local-only ---
        int localFlag = 1;
        for (auto r : partition)
            if (r != mpi.rank)
            {
                localFlag = 0;
                break;
            }
        int globalFlag;
        MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_LAND, mpi.comm);
        pt.isLocalOnly = (globalFlag != 0);

        // --- Build local permutation if local-only ---
        if (pt.isLocalOnly)
        {
            index myOffset = pt.newGlobalOffsets[mpi.rank];
            DNDS_assert(myOffset == (*oldGlobalMapping)(mpi.rank, 0));
            pt.localOld2New.resize(nLocal);
            for (index i = 0; i < nLocal; i++)
                pt.localOld2New[i] = pt.newGlobalIndices[i] - myOffset;
        }

        return pt;
    }

    // =====================================================================
    // Implementation: fromLocalPermutation
    // =====================================================================

    inline PermutationTransfer PermutationTransfer::fromLocalPermutation(
        const std::vector<index> &old2new,
        const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
        const MPIInfo &mpi)
    {
        DNDS_assert(oldGlobalMapping);
        PermutationTransfer pt;
        const index nLocal = static_cast<index>(old2new.size());

#ifndef NDEBUG
        // Validate that old2new is a valid permutation of [0, nLocal)
        std::vector<bool> seen(nLocal, false);
        for (index i = 0; i < nLocal; i++)
        {
            DNDS_assert_info(old2new[i] >= 0 && old2new[i] < nLocal,
                             fmt::format("fromLocalPermutation: old2new[{}] = {} out of [0, {})",
                                         i, old2new[i], nLocal));
            DNDS_assert_info(!seen[old2new[i]],
                             fmt::format("fromLocalPermutation: duplicate mapping to {}",
                                         old2new[i]));
            seen[old2new[i]] = true;
        }
#endif

        pt.isLocalOnly = true;
        pt.localOld2New = old2new;
        pt.targetRanks.assign(nLocal, mpi.rank);

        index myOffset = (*oldGlobalMapping)(mpi.rank, 0);
        pt.newGlobalIndices.resize(nLocal);
        for (index i = 0; i < nLocal; i++)
            pt.newGlobalIndices[i] = myOffset + old2new[i];

        // Push CSR (trivial: all go to self)
        pt.pushStart.assign(mpi.size + 1, 0);
        pt.pushStart[mpi.rank + 1] = nLocal;
        for (int r = mpi.rank + 2; r <= mpi.size; r++)
            pt.pushStart[r] = nLocal;
        pt.pushIndex.resize(nLocal);
        std::iota(pt.pushIndex.begin(), pt.pushIndex.end(), index{0});

        // Global offsets unchanged
        pt.newGlobalOffsets.resize(mpi.size + 1);
        for (int r = 0; r < mpi.size; r++)
            pt.newGlobalOffsets[r] = (*oldGlobalMapping)(r, 0);
        pt.newGlobalOffsets[mpi.size] = oldGlobalMapping->globalSize();

        return pt;
    }

    // =====================================================================
    // Implementation: transferRows
    // =====================================================================

    template <class TPair>
    void PermutationTransfer::transferRows(TPair &pair, const MPIInfo &mpi) const
    {
        DNDS_assert(pair.father);
        DNDS_assert(pair.father->Size() == size());

        if (isLocalOnly)
        {
            // In-place permutation
            DNDS_assert_info(static_cast<index>(localOld2New.size()) == size(),
                             "transferRows: localOld2New size mismatch");
            using TArr = typename decltype(pair.father)::element_type;
            auto tmp = std::make_shared<TArr>(*pair.father); // deep copy
#ifndef NDEBUG
            for (index i = 0; i < size(); i++)
                DNDS_assert_info(localOld2New[i] >= 0 && localOld2New[i] < size(),
                                 fmt::format("transferRows: localOld2New[{}] = {} out of [0, {})",
                                             i, localOld2New[i], size()));
#endif
            if constexpr (TPair::IsCSR())
                pair.father->Decompress();
            for (index i = 0; i < size(); i++)
            {
                index iNew = localOld2New[i];
                if constexpr (TPair::IsCSR())
                    pair.father->ResizeRow(iNew, tmp->RowSize(i));
                for (rowsize j = 0; j < tmp->RowSize(i); j++)
                    pair.father->operator()(iNew, j) = (*tmp)(i, j);
            }
            if constexpr (TPair::IsCSR())
                pair.father->Compress();
        }
        else
        {
            // Distributed: father=old, son=new ArrayTransformer push
            using TArr = typename decltype(pair.father)::element_type;
            auto oldFather = pair.father;
            pair.father = make_ssp<TArr>(ObjName{"PermTransfer.new"}, mpi);

            typename ArrayTransformerType<TArr>::Type trans;
            trans.setFatherSon(oldFather, pair.father);
            trans.createFatherGlobalMapping();
            trans.createGhostMapping(pushIndex, pushStart);
            trans.createMPITypes();
            trans.pullOnce();
            // pair.father is now the new array with redistributed data.
            // Rebuild its global mapping so the result is self-contained.
            pair.father->createGlobalMapping();
        }

        // Son is stale after either path — reset it
        if (pair.son)
            pair.son.reset();
    }

    // =====================================================================
    // Implementation: buildLookup
    // =====================================================================

    inline PermutationTransfer::LookupResult PermutationTransfer::buildLookup(
        const std::vector<index> &pullSet,
        const MPIInfo &mpi) const
    {
        LookupResult result;
        result.pair.InitPair("PermTransfer_lookup", mpi);
        result.pair.father->Resize(size());
        for (index i = 0; i < size(); i++)
            result.pair(i, 0) = newGlobalIndices[i];

        result.pair.TransAttach();
        result.pair.trans.createFatherGlobalMapping();
        std::vector<index> pullSetMut(pullSet); // mutable copy for createGhostMapping
        result.pair.trans.createGhostMapping(pullSetMut);
        result.pair.trans.createMPITypes();
        result.pair.trans.pullOnce();

        return result;
    }

} // namespace DNDS

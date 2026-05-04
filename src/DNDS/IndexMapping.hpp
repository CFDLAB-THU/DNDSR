#pragma once
/// @file IndexMapping.hpp
/// @brief Global-to-local index mapping for distributed arrays.
/// @par Unit Test Coverage (test_IndexMapping.cpp, MPI np=1,2,4)
/// - GlobalOffsetsMapping: setMPIAlignBcast, globalSize, RLengths, ROffsets,
///   operator() (local-to-global), search (global-to-local, both overloads)
///   -- uniform and non-uniform distributions
///   -- edge cases: out-of-range, negative index, rank boundary crossing
/// - OffsetAscendIndexMapping (pull-based): searchInMain, searchInGhost,
///   searchInAllGhost, search, search_indexAppend, operator() (reverse mapping)
///   -- empty ghost set edge case
/// @par Not Yet Tested
/// - Push-based OffsetAscendIndexMapping constructor
/// - sort(), ghostAt(), search_indexRank
/// - Duplicate pull indices

#include <unordered_map>
#include <algorithm>
#include <tuple>

#include "Defines.hpp"
#include "MPI.hpp"

namespace DNDS
{ // mapping from rank-main place to global indices
    // should be global-identical, can broadcast

    /**
     * @brief Table mapping rank-local row indices to the global index space.
     *
     * @details Every rank owns a contiguous chunk in the global index space:
     *
     * ```
     * rank:           0    1    2
     * RankLengths = [ 3,   4,   5 ]
     * RankOffsets = [ 0,   3,   7,  12 ]  # prefix sum, size == nRanks + 1
     * ```
     *
     * After #setMPIAlignBcast, every rank holds an identical copy of the table
     * (it is globally replicated), so both local-to-global
     * (`operator()`) and global-to-local (#search) queries are purely local.
     * Used by @ref DNDS::ParArray "ParArray"::createGlobalMapping and friends.
     */
    class GlobalOffsetsMapping
    {

        ///@brief "dist" sizes of each rank
        t_IndexVec RankLengths;
        t_IndexVec RankOffsets;

    public:
        /// @brief Per-rank lengths vector (`size == nRanks`).
        t_IndexVec &RLengths() { return RankLengths; }
        /// @brief Prefix-sum offsets vector (`size == nRanks + 1`, last element == globalSize).
        t_IndexVec &ROffsets() { return RankOffsets; }

        /// @brief Total number of rows across all ranks.
        [[nodiscard]] index globalSize() const
        {
            if (!RankOffsets.empty())
                return RankOffsets[RankOffsets.size() - 1];
            else
                return 0;
        }

        /// @brief Broadcast each rank's length, then compute the global prefix sums.
        /// @details Collective call. After it returns, every rank holds the full
        /// #RLengths / #ROffsets tables. Called by
        /// `ParArray::createGlobalMapping` during mesh/array setup.
        /// @param mpi       MPI context.
        /// @param myLength  Number of rows owned by the calling rank.
        void setMPIAlignBcast(const MPIInfo &mpi, index myLength)
        {
            RankLengths.resize(mpi.size);
            RankOffsets.resize(mpi.size + 1);
            RankLengths[mpi.rank] = myLength;

            // tMPI_reqVec bcastReqs(mpi.size); // for Ibcast

            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // std::cout << mpi.rank << '\t' << myLength << std::endl;
                MPI::Bcast(RankLengths.data() + r, sizeof(index), MPI_BYTE, r, mpi.comm);
            }
            RankOffsets[0] = 0;
            for (size_t i = 0; i < RankLengths.size(); i++)
            {
                RankOffsets[i + 1] = RankOffsets[i] + RankLengths[i];
                DNDS_assert(RankOffsets[i + 1] >= 0);
            }
        }

        /// @brief Convert a (rank, local) pair to a global index.
        /// @details Equivalent to `RankOffsets[rank] + val`, with asserts.
        [[nodiscard]] index operator()(MPI_int rank, index val) const
        {
            // if (!((rank >= 0 && rank <= RankLengths.size()) &&
            //       (val >= 0 && val <= RankOffsets[rank + 1] - RankOffsets[rank])))
            // {
            //     PrintVec(RankOffsets, std::cout);
            //     std::cout << rank << " KK " << val << std::endl;
            // }
            DNDS_assert((rank >= 0 && rank <= RankLengths.size()) &&
                        (val >= 0 && val <= RankOffsets[rank + 1] - RankOffsets[rank]));
            return RankOffsets[rank] + val;
        }

        /// @brief Convert a global index to `(rank, local)`. Returns `false` if out of range.
        /// @details Uses `std::lower_bound` on the offsets table (O(log nRanks)).
        /// @param globalQuery Global index.
        /// @param[out] rank   Owning rank on success.
        /// @param[out] val    Local offset within the owner.
        /// @return true if `globalQuery` lies within `[0, globalSize())`.
        [[nodiscard]] bool search(index globalQuery, MPI_int &rank, index &val) const
        {
            // find the first larger than query, for this is a [,) interval search, should find the ) position
            // example: RankOffsets = {0,1,3,3,4,4,5,5}
            //                 Rank = {0,1,2,3,4,5,6}
            // query 5 should be rank 7, which is out-of bound, returns false
            // query 4 should be rank 5, query 3 should be rank 3, query 2 should be rank 1
            if (RankOffsets.empty()) // in case the communicator is of size 0 ??
                return false;
            auto place = std::lower_bound(RankOffsets.begin(), RankOffsets.end(), globalQuery, std::less_equal<index>());
            rank = static_cast<MPI_int>(place - 1 - RankOffsets.begin()); // ! could overflow
            if (rank < RankLengths.size() && rank >= 0)
            {
                val = globalQuery - RankOffsets[rank];
                return true;
            }
            return false;
        }

        [[nodiscard]] std::tuple<bool, MPI_int, index> search(index globalQuery) const
        {
            MPI_int rank{-1};
            index val{-1};
            bool ret = this->search(globalQuery, rank, val);
            return std::make_tuple(ret, rank, val);
        }
    };

    /**
     * @brief Mapping between a rank's *main* data and its *ghost* copies.
     *
     * @details `main` data (owned rows) is a contiguous range
     * `[mainOffset, mainOffset + mainSize)` in the global index space. `ghost`
     * data (rows needed from other ranks) is stored in *globally ascending*
     * order so that binary search may be used for the reverse mapping.
     *
     * Two construction modes:
     *  - **Pull**: the caller provides the set of global indices this rank needs.
     *  - **Push**: the caller provides which local rows to send to which ranks.
     * Both modes exchange the complementary information via `MPI_Alltoall(v)`
     * so the resulting mapping is valid on both sides.
     *
     * Internal structure (all vectors are per-peer-rank layouts, sorted by rank):
     *  - `ghostSizes` / `ghostStart` / `ghostIndex`: what this rank *pulls* from
     *    each other rank.
     *  - `pushIndexSizes` / `pushIndexStarts` / `pushingIndexGlobal`: what this
     *    rank *sends* to each other rank.
     *
     * @warning Ghost indices carry only 32-bit payload in MPI calls; the total
     * per-rank ghost count must fit in `int32_t`.
     */
    class OffsetAscendIndexMapping
    {
        using mapIndex = MPI_int;
        using t_MapIndexVec = std::vector<mapIndex>;
        static_assert(std::numeric_limits<mapIndex>::max() >= std::numeric_limits<MPI_int>::max());
        index mainOffset;
        index mainSize;

    public:
        /// @brief Number of ghost rows pulled from each peer rank. (a.k.a. `pullIndexSizes`)
        tMPI_intVec ghostSizes;

        /// @brief Global indices of all ghost rows on this rank, ascending.
        /// (a.k.a. `pullingIndexGlobal`)
        t_IndexVec ghostIndex;

        /// @brief Per-peer prefix-sum offsets into `ghostIndex`
        /// (size == `nRanks + 1`). (a.k.a. `pullIndexStarts`)
        t_MapIndexVec ghostStart;

        /// @brief Per-peer number of rows *this rank* will send out on a push.
        tMPI_intVec pushIndexSizes;
        /// @brief Per-peer prefix-sum offsets into `pushingIndexGlobal`.
        t_MapIndexVec pushIndexStarts;
        /// @brief Global indices this rank sends to others, grouped by receiver.
        t_IndexVec pushingIndexGlobal;

        /// @brief Optional cached local-index form of ghostIndex (unused in current code).
        /// @details Encoding: `>=0` means `array[i]`, `<0` means `ghost_array[-i-1]`.
        /// Retained for potential future reconstruction paths.
        t_IndexVec pullingRequestLocal;

        /**
         * @brief Construct the mapping from a pull specification.
         *
         * @tparam TpullSet         Range of `index` (vector-like, supports sort/unique/erase).
         * @param nmainOffset       Global offset of this rank's main data block.
         * @param nmainSize         Number of main rows owned by this rank.
         * @param pullingIndexGlobal Global indices this rank wants to see as ghosts.
         *                          **Sorted and deduplicated in-place.**
         * @param LGlobalMapping    Globally-known offsets, used to attribute each
         *                          global index to its owner rank.
         * @param mpi               MPI context; the constructor is collective.
         *
         * @warning `pullingIndexGlobal` is mutated (sorted / uniquified / shrunk).
         * The resulting `ghostIndex` ends up in ascending global-index order
         * regardless of the caller's original ordering.
         */
        template <class TpullSet>
        OffsetAscendIndexMapping(index nmainOffset, index nmainSize,
                                 TpullSet &&pullingIndexGlobal,
                                 const GlobalOffsetsMapping &LGlobalMapping,
                                 const MPIInfo &mpi)
            : mainOffset(nmainOffset),
              mainSize(nmainSize)
        {
            ///*make sorted and unique!
            std::sort(pullingIndexGlobal.begin(), pullingIndexGlobal.end());
            auto last = std::unique(pullingIndexGlobal.begin(), pullingIndexGlobal.end());
            pullingIndexGlobal.erase(last, pullingIndexGlobal.end());
            pullingIndexGlobal.shrink_to_fit();
            ///

            ghostSizes.assign(mpi.size, 0);

            for (auto i : pullingIndexGlobal)
            {
                MPI_int rank = -1;
                index loc = -1; // dummy here
                bool search_result = LGlobalMapping.search(i, rank, loc);
                DNDS_assert_info(search_result, "Search Failed");
                // if (rank != mpi.rank) // must not exclude local ones for the sake of scatter/gather
                ghostSizes[rank]++;
            }

            ghostStart.resize(ghostSizes.size() + 1);
            ghostStart[0] = 0;
            for (typename decltype(ghostSizes)::size_type i = 0; i < ghostSizes.size(); i++)
                ghostStart[i + 1] = signedIntSafeAdd<mapIndex>(ghostStart[i], ghostSizes[i]);
            ghostIndex.reserve(ghostStart[ghostSizes.size()]);
            for (auto i : pullingIndexGlobal)
            {
                MPI_int rank = 0;
                index loc = 0; // dummy here
                bool search_result = LGlobalMapping.search(i, rank, loc);
                DNDS_assert_info(search_result, "Search Failed");
                // if (rank != mpi.rank)
                ghostIndex.push_back(i);
            }
            // ghostIndex.shrink_to_fit(); // only for safety
            this->sort();

            // obtain pushIndexSizes and actual push indices
            pushIndexSizes.resize(mpi.size);
            MPI::Alltoall(ghostSizes.data(), 1, MPI_INT, pushIndexSizes.data(), 1, MPI_INT, mpi.comm);
            AccumulateRowSize(pushIndexSizes, pushIndexStarts);
            pushingIndexGlobal.resize(pushIndexStarts[pushIndexStarts.size() - 1]);
            MPI::Alltoallv(ghostIndex.data(), ghostSizes.data(), ghostStart.data(), DNDS_MPI_INDEX,
                           pushingIndexGlobal.data(), pushIndexSizes.data(), pushIndexStarts.data(), DNDS_MPI_INDEX,
                           mpi.comm);

            // stores pullingRequest // ! now cancelled
            // pullingRequestLocal = std::forward<TpullSet>(pullingIndexGlobal);
            // for (auto &i : pullingRequestLocal)
            // {
            //     MPI_int rank;
            //     index loc; // dummy here
            //     search(i, rank, loc);
            //     if (rank != -1)
            //         i = -(1 + ghostStart[rank] + loc);
            // }
        }

        /**
         * @brief Construct the mapping from a push specification.
         *
         * @details The caller supplies, for every peer rank, a list of *local*
         * row indices that this rank should send. The constructor flips this
         * into the pull view via `MPI_Alltoall(v)`.
         *
         * @tparam TpushSet              Range-like of local indices.
         * @tparam TpushStart            Size-`nRanks+1` prefix-sum offsets.
         * @param nmainOffset            Global offset of this rank's main block.
         * @param nmainSize              Number of main rows owned by this rank.
         * @param pushingIndexesLocal    Local indices to push, grouped by receiver
         *                               (flat array; `[pushingStarts[r], pushingStarts[r+1])`
         *                               are the indices going to rank `r`).
         * @param pushingStarts          Prefix sums of per-receiver counts, size `nRanks+1`.
         * @param LGlobalMapping         Rank-offset table (to convert local -> global).
         * @param mpi                    MPI context; collective.
         */
        template <class TpushSet, class TpushStart>
        OffsetAscendIndexMapping(index nmainOffset, index nmainSize,
                                 TpushSet &&pushingIndexesLocal, // which stores local index
                                 TpushStart &&pushingStarts,
                                 const GlobalOffsetsMapping &LGlobalMapping,
                                 const MPIInfo &mpi)
            : mainOffset(nmainOffset),
              mainSize(nmainSize)
        {
            DNDS_assert(pushingStarts.size() == mpi.size + 1 && pushingIndexesLocal.size() == pushingStarts[mpi.size]);
            pushIndexSizes.resize(mpi.size);
            pushIndexStarts.resize(mpi.size + 1, 0);
            for (int i = 0; i < mpi.size; i++)
                pushIndexSizes[i] = pushingStarts[i + 1] - pushingStarts[i],
                pushIndexStarts[i + 1] = pushingStarts[i + 1];
            pushingIndexGlobal.resize(pushingIndexesLocal.size());
            // std::forward<TpushStart>(pushingStarts); //! might delete
            for (size_t i = 0; i < pushingIndexGlobal.size(); i++)
                pushingIndexGlobal[i] = LGlobalMapping(mpi.rank, pushingIndexesLocal[i]); // convert from local to global
            // std::forward<TpushSet>(pushingIndexesLocal);                                  //! might delete

            ghostSizes.assign(mpi.size, 0);
            MPI::Alltoall(pushIndexSizes.data(), 1, MPI_INT, ghostSizes.data(), 1, MPI_INT, mpi.comm); // inverse to the normal pulling
            ghostStart.resize(ghostSizes.size() + 1);
            ghostStart[0] = 0;
            for (size_t i = 0; i < ghostSizes.size(); i++)
                ghostStart[i + 1] = ghostStart[i] + ghostSizes[i];
            ghostIndex.resize(ghostStart[ghostSizes.size()]);
            MPI::Alltoallv(pushingIndexGlobal.data(), pushIndexSizes.data(), pushIndexStarts.data(), DNDS_MPI_INDEX,
                           ghostIndex.data(), ghostSizes.data(), ghostStart.data(), DNDS_MPI_INDEX,
                           mpi.comm); // inverse to the normal pulling

            // !doesn't store pullingRequest
        }

        // auto &ghost() { return ghostIndex; }

        // auto &gStarts() { return ghostStart; }

        /// @brief Sort `ghostIndex` into ascending order. Required invariant
        /// maintained after pull construction.
        // warning: using globally sorted condition
        void sort()
        {
            std::sort(ghostIndex.begin(), ghostIndex.end());
        }

        /// @brief Direct mutable access to the `ighost`-th ghost global index for peer `rank`.
        index &ghostAt(MPI_int rank, index ighost)
        {
            DNDS_assert(ighost >= 0 && ighost < (ghostStart[rank + 1] - ghostStart[rank]));
            return ghostIndex[ghostStart[rank] + ighost];
        }

        // TtMapIndexVec is a std::vector of someint
        // could overflow, accumulated to 32-bit
        // template <class TtMapIndexVec>
        // void allocateGhostIndex(const TtMapIndexVec &ghostSizes)
        // {
        // }

        /// @brief Check whether a global index lies within the main block.
        /// @param globalQuery Global index to look up.
        /// @param[out] val    On success, local offset within the main block.
        /// @return true if `globalQuery` is owned by this rank.
        [[nodiscard]] bool searchInMain(index globalQuery, index &val) const
        {
            // std::cout << mainOffset << mainSize << std::endl;
            if (globalQuery >= mainOffset && globalQuery < mainSize + mainOffset)
            {
                val = globalQuery - mainOffset;
                return true;
            }
            return false;
        }

        /// @brief Search a single peer rank's ghost slab for a global index.
        /// @param[out] val Offset relative to `ghostStart[rank]` on success.
        // returns place relative to ghostStart[rank]
        [[nodiscard]] bool searchInGhost(index globalQuery, MPI_int rank, index &val) const
        {
            DNDS_assert((rank >= 0 && rank < ghostStart.size() - 1));
            if ((ghostStart[rank + 1] - ghostStart[rank]) == 0)
                return false; // size = 0 could result in seg error doing search
            auto start = ghostIndex.begin() + ghostStart[rank];
            auto end = ghostIndex.begin() + ghostStart[rank + 1];
            auto place = std::lower_bound(start, end, globalQuery);
            if (place != end && *place == globalQuery) // dereferencing end could result in seg error
            {
                val = place - (ghostIndex.begin() + ghostStart[rank]);
                return true;
            }
            return false;
        }

        /// @brief Search the concatenated ghost array for a global index (O(log nGhost)).
        /// @param[out] rank Peer rank that owns the found index.
        /// @param[out] val  Offset relative to `ghostStart[0]` (== 0).
        // returns rank & place, place relative to ghostStart[0] (==0)
        [[nodiscard]] bool searchInAllGhost(index globalQuery, MPI_int &rank, index &val) const
        {
            auto start = ghostIndex.begin();
            auto end = ghostIndex.end();
            auto place = std::lower_bound(start, end, globalQuery);
            if (place != end && *place == globalQuery) // dereferencing end could result in seg error
            {
                val = place - start;
                auto s_start = ghostStart.begin();
                auto s_end = ghostStart.end();
                auto s_place = std::lower_bound(s_start, s_end, val, std::less_equal<rowsize>());
                DNDS_assert(s_place != s_end && s_place > s_start);
                rank = static_cast<MPI_int>(s_place - s_start - 1);
                return true;
            }
            return false;
        }

        /// @brief Search main + ghost in one call. `rank == -1` signals "hit in main".
        /// @details Returned `val` is in the *ghost* address space (relative to
        /// `ghostStart[0]`) when `rank != -1`, else it is the main-local index.
        /// See also #search_indexAppend for the combined (father+son) index space.
        /// \brief returns rank and place in ghost array, rank==-1 means main data
        [[nodiscard]] bool search(index globalQuery, MPI_int &rank, index &val) const
        {
            if (searchInMain(globalQuery, val))
            {
                rank = -1;
                return true;
            }
            if (searchInAllGhost(globalQuery, rank, val))
            {
                return true;
            }
            return false;
        }
        /// @brief Tuple-return overload of the main #search. `(success, rank, local)`.
        [[nodiscard]] std::tuple<bool, MPI_int, index> search(index globalQuery) const
        {
            MPI_int rank{-1};
            index val{-1};
            bool ret = this->search(globalQuery, rank, val);
            return std::make_tuple(ret, rank, val);
        }

        /// @brief Like #search, but the returned `val` is in the concatenated
        /// *father+son* address space.
        /// @details `val < mainSize` means main row; `val >= mainSize` means
        /// ghost row at `val - mainSize`. Matches `ArrayPair::operator[]` layout.
        /// \brief returns rank and place in ghost array, rank==-1 means main data;
        /// returned val is used for pair indexing
        [[nodiscard]] bool search_indexAppend(index globalQuery, MPI_int &rank, index &val) const
        {
            if (searchInMain(globalQuery, val))
            {
                rank = -1;
                return true;
            }
            if (searchInAllGhost(globalQuery, rank, val))
            {
                // std::cout << mainSize << std::endl;
                val += mainSize;
                return true;
            }
            return false;
        }
        /// @brief Tuple-return overload of #search_indexAppend.
        [[nodiscard]] std::tuple<bool, MPI_int, index> search_indexAppend(index globalQuery) const
        {
            MPI_int rank{-1};
            index val{-1};
            bool ret = this->search_indexAppend(globalQuery, rank, val);
            return std::make_tuple(ret, rank, val);
        }

        /// @brief Like #search, but `val` is expressed relative to the owner
        /// rank's ghost slab (`ghostStart[rank]`-relative).
        /// @warning Do not confuse with #search, whose `val` is relative to the
        /// concatenated ghost array.
        /// \brief returns rank and place in ghost of rank, rank==-1 means main data
        /// \warning search returns index that applies to local ghost array, this only goes for the ith of rank
        [[nodiscard]] bool search_indexRank(index globalQuery, MPI_int &rank, index &val) const
        {
            if (searchInMain(globalQuery, val))
            {
                rank = -1;
                return true;
            }
            if (searchInAllGhost(globalQuery, rank, val))
            {
                val -= ghostStart[rank];
                return true;
            }
            return false;
        }

        /// @brief Tuple-return overload of #search_indexRank.
        [[nodiscard]] std::tuple<bool, MPI_int, index> search_indexRank(index globalQuery) const
        {
            MPI_int rank{-1};
            index val{-1};
            bool ret = this->search_indexRank(globalQuery, rank, val);
            return std::make_tuple(ret, rank, val);
        }

        /// @brief Reverse (local -> global) mapping.
        /// @details
        ///  - `rank == -1`: `val` indexes the concatenated father+son address
        ///    space. `val < mainSize` returns the main global index;
        ///    `val >= mainSize` returns the corresponding ghost global index.
        ///  - `rank >= 0`: `val` indexes into that peer's ghost slab
        ///    (like #search_indexRank's output).
        /// \brief if rank == -1, return the global index of local main data,
        /// or else return the ghosted global index of local ghost data.
        [[nodiscard]] index operator()(MPI_int rank, index val) const
        {
            if (rank == -1)
            {
                DNDS_assert(val >= 0);
                if (val < mainSize)
                    return val + mainOffset;
                else
                {
                    DNDS_assert_info(val - mainSize < ghostStart.back(), fmt::format("{}, {}, ghostSize {}", val, mainSize, ghostStart.back()));
                    return ghostIndex.at(val - mainSize);
                }
            }
            else
            {
                DNDS_assert(
                    (rank >= 0 && rank < ghostStart.size() - 1) &&
                    (val >= 0 && val < ghostStart[rank + 1] - ghostStart[rank]));
                return ghostIndex[ghostStart[rank] + val];
            }
        }
    };

}

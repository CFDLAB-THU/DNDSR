#pragma once
#include "Mesh_DeviceView.hpp"
#include "DNDS/ArrayTransformer.hpp"

namespace DNDS::Geom
{
    /**
     * \brief Per-adjacency index state tracking.
     *
     * Records whether an adjacency array's entries are global or local
     * indices, and holds a shared reference to the ghost mapping of the
     * **target** entity kind (the entity that the indices refer to).
     *
     * For example, `cell2node` points to nodes, so its `AdjIndexInfo`
     * holds a reference to the node ghost mapping
     * (`coords.trans.pLGhostMapping`).
     */
    struct AdjIndexInfo
    {
        /// Current global/local state of the entries in this adjacency.
        MeshAdjState state{Adj_Unknown};

        /// Ghost mapping of the **target** entity kind.
        /// Used for global<->local conversion of entries.
        /// nullptr until the target entity's ghost layer has been built.
        t_pLGhostMapping targetMapping;

        /// Global-offsets mapping of the target entity kind.
        /// Used for _NoSon conversion paths (father-only, no ghost).
        t_pLGlobalMapping targetGlobalMapping;

        /// \brief Attach the target entity's ghost mapping to this adjacency.
        ///
        /// Asserts that either:
        ///   (a) state == Adj_PointToGlobal (indices are global, mapping is
        ///       meaningful for future toLocal()), or
        ///   (b) state == Adj_Unknown (adjacency not yet populated — the
        ///       mapping is being pre-wired for later use).
        ///
        /// Calling this when state == Adj_PointToLocal is a logic error:
        /// the indices are already local with respect to some (possibly
        /// different) mapping, and silently replacing the mapping would
        /// make the next toGlobal() produce garbage.
        void wireTargetMapping(const t_pLGhostMapping &mapping)
        {
            DNDS_assert_info(
                state != Adj_PointToLocal,
                "wireTargetMapping called while indices are local — "
                "convert to global first, or the stored mapping will "
                "be inconsistent with the index values");
            targetMapping = mapping;
        }

        /// \brief Attach the target entity's global-offsets mapping.
        /// Same safety invariant as wireTargetMapping.
        void wireTargetGlobalMapping(const t_pLGlobalMapping &mapping)
        {
            targetGlobalMapping = mapping;
        }

        /// \brief Bulk-convert all entries in [0, nRows) from global to local.
        ///
        /// Entries that cannot be found in the ghost mapping are encoded as
        /// `(-1 - globalIndex)`, matching the convention of
        /// UnstructuredMesh::IndexGlobal2Local.
        template <class TAdj>
        void toLocal(TAdj &adj, index nRows)
        {
            DNDS_assert(state == Adj_PointToGlobal);
            DNDS_assert(targetMapping);
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index &v = adj(i, j);
                    if (v == UnInitIndex)
                        continue;
                    MPI_int rank;
                    index val;
                    if (targetMapping->search_indexAppend(v, rank, val))
                        v = val;
                    else
                        v = -1 - v; // not-found encoding
                }
            state = Adj_PointToLocal;
        }

        /// \brief Bulk-convert all entries in [0, nRows) from local to global.
        template <class TAdj>
        void toGlobal(TAdj &adj, index nRows)
        {
            DNDS_assert(state == Adj_PointToLocal);
            DNDS_assert(targetMapping);
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index &v = adj(i, j);
                    if (v == UnInitIndex)
                        continue;
                    if (v < 0) // "not-found" encoding from prior G2L
                        v = -1 - v;
                    else
                        v = targetMapping->operator()(-1, v);
                }
            state = Adj_PointToGlobal;
        }

        /// \brief OMP-parallelized variant of toLocal.
        template <class TAdj>
        void toLocalOMP(TAdj &adj, index nRows)
        {
            DNDS_assert(state == Adj_PointToGlobal);
            DNDS_assert(targetMapping);
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index &v = adj(i, j);
                    if (v == UnInitIndex)
                        continue;
                    MPI_int rank;
                    index val;
                    if (targetMapping->search_indexAppend(v, rank, val))
                        v = val;
                    else
                        v = -1 - v;
                }
            state = Adj_PointToLocal;
        }

        /// \brief OMP-parallelized variant of toGlobal.
        template <class TAdj>
        void toGlobalOMP(TAdj &adj, index nRows)
        {
            DNDS_assert(state == Adj_PointToLocal);
            DNDS_assert(targetMapping);
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index &v = adj(i, j);
                    if (v == UnInitIndex)
                        continue;
                    if (v < 0)
                        v = -1 - v;
                    else
                        v = targetMapping->operator()(-1, v);
                }
            state = Adj_PointToGlobal;
        }
    };

    /**
     * \brief Flattened wrapper: inherits from TPair and adds AdjIndexInfo.
     *
     * Because AdjWithState inherits publicly from TPair, all existing
     * code that accesses `.father`, `.son`, `.trans`, `operator[]`,
     * `BorrowAndPull()`, etc. works unchanged.  The only addition is
     * the `idx` member for per-adjacency state tracking.
     *
     * \tparam TPair  The ArrayPair type (e.g. tAdjPair, tAdj2Pair).
     */
    template <class TPair>
    struct AdjWithState : public TPair
    {
        AdjIndexInfo idx;

        // Convenience: convert this adjacency in-place
        void toLocal() { idx.toLocal(*this, this->Size()); }
        void toGlobal() { idx.toGlobal(*this, this->Size()); }
        void toLocalOMP() { idx.toLocalOMP(*this, this->Size()); }
        void toGlobalOMP() { idx.toGlobalOMP(*this, this->Size()); }

        // State queries
        MeshAdjState state() const { return idx.state; }
        bool isLocal() const { return idx.state == Adj_PointToLocal; }
        bool isGlobal() const { return idx.state == Adj_PointToGlobal; }
        bool isBuilt() const { return idx.state != Adj_Unknown; }
    };

} // namespace DNDS::Geom

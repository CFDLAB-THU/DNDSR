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
     *
     * All fields are private.  State transitions are enforced by methods:
     *
     * - markGlobal():           Adj_Unknown  -> Adj_PointToGlobal
     * - wireTargetMapping(m):   requires !isLocal(); stores ghost mapping
     * - toLocal(adj, n):        Adj_PointToGlobal -> Adj_PointToLocal  (requires wired)
     * - toGlobal(adj, n):       Adj_PointToLocal  -> Adj_PointToGlobal (requires wired)
     * - bootstrapToLocal(m,adj,n):  markGlobal + wire + toLocal in one call
     *                               (accepts Adj_Unknown or Adj_PointToGlobal)
     */
    struct AdjIndexInfo
    {
    private:
        /// Current global/local state of the entries in this adjacency.
        MeshAdjState _state{Adj_Unknown};

        /// Ghost mapping of the **target** entity kind.
        /// Used for global<->local conversion of entries.
        /// nullptr until the target entity's ghost layer has been built.
        t_pLGhostMapping _targetMapping;

    public:
        // ============================================================
        // Queries
        // ============================================================

        MeshAdjState state() const { return _state; }
        bool isLocal() const { return _state == Adj_PointToLocal; }
        bool isGlobal() const { return _state == Adj_PointToGlobal; }
        bool isBuilt() const { return _state != Adj_Unknown; }
        bool isWired() const { return _targetMapping != nullptr; }

        /// Read-only access to the stored mapping (for assertions/comparisons).
        const t_pLGhostMapping &mapping() const { return _targetMapping; }

        // ============================================================
        // State transitions
        // ============================================================

        /// \brief Mark this adjacency as containing global indices.
        ///
        /// Valid from Adj_Unknown or Adj_PointToGlobal (idempotent).
        /// Use after building or receiving an adjacency whose entries
        /// are global entity indices.
        void markGlobal()
        {
            DNDS_assert_info(
                _state == Adj_Unknown || _state == Adj_PointToGlobal,
                "markGlobal: expected Adj_Unknown or Adj_PointToGlobal state");
            _state = Adj_PointToGlobal;
        }

        /// \brief Mark this adjacency as containing local indices.
        ///
        /// Valid from Adj_Unknown only.  Requires target mapping to be wired
        /// (needed for future toGlobal calls).
        /// Use when an adjacency is populated directly with local indices,
        /// bypassing the normal global->local pipeline.
        void markLocal()
        {
            DNDS_assert_info(
                _state == Adj_Unknown,
                "markLocal: expected Adj_Unknown state");
            DNDS_assert_info(
                _targetMapping,
                "markLocal: target mapping must be wired before marking local");
            _state = Adj_PointToLocal;
        }

        /// \brief Attach the target entity's ghost mapping.
        ///
        /// Callable when state is Adj_Unknown or Adj_PointToGlobal.
        /// Calling while Adj_PointToLocal is a logic error: indices are
        /// already local w.r.t. some (possibly different) mapping, and
        /// replacing the mapping silently would make toGlobal() produce
        /// garbage.
        void wireTargetMapping(const t_pLGhostMapping &mapping)
        {
            DNDS_assert_info(
                _state != Adj_PointToLocal,
                "wireTargetMapping called while indices are local — "
                "convert to global first, or the stored mapping will "
                "be inconsistent with the index values");
            _targetMapping = mapping;
        }

        // ============================================================
        // Conversion: toLocal / toGlobal (mapping must be wired)
        // ============================================================

        /// \brief Bulk-convert all entries in [0, nRows) from global to local.
        ///
        /// Entries that cannot be found in the ghost mapping are encoded as
        /// `(-1 - globalIndex)`, matching the convention of
        /// UnstructuredMesh::IndexGlobal2Local.
        template <class TAdj>
        void toLocal(TAdj &adj, index nRows)
        {
            DNDS_assert_info(_state == Adj_PointToGlobal,
                             "toLocal: expected Adj_PointToGlobal state");
            DNDS_assert_info(_targetMapping,
                             "toLocal: target mapping not wired");
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index &v = adj(i, j);
                    if (v == UnInitIndex)
                        continue;
                    MPI_int rank;
                    index val;
                    if (_targetMapping->search_indexAppend(v, rank, val))
                        v = val;
                    else
                        v = -1 - v; // not-found encoding
                }
            _state = Adj_PointToLocal;
        }

        /// \brief Bulk-convert all entries in [0, nRows) from local to global.
        template <class TAdj>
        void toGlobal(TAdj &adj, index nRows)
        {
            DNDS_assert_info(_state == Adj_PointToLocal,
                             "toGlobal: expected Adj_PointToLocal state");
            DNDS_assert_info(_targetMapping,
                             "toGlobal: target mapping not wired");
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index &v = adj(i, j);
                    if (v == UnInitIndex)
                        continue;
                    if (v < 0) // "not-found" encoding from prior G2L
                        v = -1 - v;
                    else
                        v = _targetMapping->operator()(-1, v);
                }
            _state = Adj_PointToGlobal;
        }

        /// \brief OMP-parallelized variant of toLocal.
        template <class TAdj>
        void toLocalOMP(TAdj &adj, index nRows)
        {
            DNDS_assert_info(_state == Adj_PointToGlobal,
                             "toLocalOMP: expected Adj_PointToGlobal state");
            DNDS_assert_info(_targetMapping,
                             "toLocalOMP: target mapping not wired");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index &v = adj(i, j);
                    if (v == UnInitIndex)
                        continue;
                    MPI_int rank;
                    index val;
                    if (_targetMapping->search_indexAppend(v, rank, val))
                        v = val;
                    else
                        v = -1 - v;
                }
            _state = Adj_PointToLocal;
        }

        /// \brief OMP-parallelized variant of toGlobal.
        template <class TAdj>
        void toGlobalOMP(TAdj &adj, index nRows)
        {
            DNDS_assert_info(_state == Adj_PointToLocal,
                             "toGlobalOMP: expected Adj_PointToLocal state");
            DNDS_assert_info(_targetMapping,
                             "toGlobalOMP: target mapping not wired");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
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
                        v = _targetMapping->operator()(-1, v);
                }
            _state = Adj_PointToGlobal;
        }

        // ============================================================
        // Bootstrap: wire mapping + convert in one atomic operation
        // ============================================================

        /// \brief Wire target mapping and convert to local in one step.
        ///
        /// Solves the chicken-and-egg problem where the mapping becomes
        /// available at the same time the adjacency needs converting.
        /// Accepts Adj_Unknown or Adj_PointToGlobal.  Sets state to
        /// Adj_PointToLocal on completion.
        template <class TAdj>
        void bootstrapToLocal(const t_pLGhostMapping &mapping, TAdj &adj, index nRows)
        {
            DNDS_assert_info(
                _state == Adj_Unknown || _state == Adj_PointToGlobal,
                "bootstrapToLocal: expected Adj_Unknown or Adj_PointToGlobal state");
            _targetMapping = mapping;
            _state = Adj_PointToGlobal; // ensure toLocal precondition
            toLocal(adj, nRows);
        }

        /// \brief OMP variant of bootstrapToLocal.
        template <class TAdj>
        void bootstrapToLocalOMP(const t_pLGhostMapping &mapping, TAdj &adj, index nRows)
        {
            DNDS_assert_info(
                _state == Adj_Unknown || _state == Adj_PointToGlobal,
                "bootstrapToLocalOMP: expected Adj_Unknown or Adj_PointToGlobal state");
            _targetMapping = mapping;
            _state = Adj_PointToGlobal;
            toLocalOMP(adj, nRows);
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

        /// Bootstrap: wire mapping + convert to local (solves chicken-and-egg).
        void bootstrapToLocal(const t_pLGhostMapping &mapping)
        {
            idx.bootstrapToLocal(mapping, *this, this->Size());
        }
        void bootstrapToLocalOMP(const t_pLGhostMapping &mapping)
        {
            idx.bootstrapToLocalOMP(mapping, *this, this->Size());
        }

        // State queries (delegate to idx)
        MeshAdjState state() const { return idx.state(); }
        bool isLocal() const { return idx.isLocal(); }
        bool isGlobal() const { return idx.isGlobal(); }
        bool isBuilt() const { return idx.isBuilt(); }
        bool isWired() const { return idx.isWired(); }
        const t_pLGhostMapping &mapping() const { return idx.mapping(); }
    };

} // namespace DNDS::Geom

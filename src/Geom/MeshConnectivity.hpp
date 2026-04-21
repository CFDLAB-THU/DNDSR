#pragma once
/// @file MeshConnectivity.hpp
/// @brief Layered DAG of mesh adjacency relations with composable DSL operations.
///
/// MeshConnectivity manages adjacency (connectivity) tables between entity strata
/// of different topological depths (cells, faces, edges, nodes). It provides three
/// core operations:
///   - Inverse:         cone (A→B) → support (B→A)
///   - Compose:         A→B + B→C → A→C
///   - ComposeFiltered: A→B + B→C → A→C with on-the-fly predicate filtering
///
/// Cone adjacencies (downward: higher→lower depth) are ordered by element topology.
/// Support adjacencies (upward: lower→higher depth) are ordered by creation method
/// (typically the order entities were discovered during inversion).
///
/// Periodic bits (pbi) are only stored on cones whose target depth is 0 (nodes),
/// since pbi tracks how each node's coordinates transform under periodicity.

#include "DNDS/ArrayPair.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "Mesh_DeviceView.hpp"

#include <variant>

namespace DNDS::Geom
{
    // =================================================================
    // Type-erased adjacency storage
    // =================================================================

    /// Variant of all supported fixed-width adjacency pair types.
    /// tAdjPair (NonUniformSize) is the general variable-width CSR;
    /// tAdj1Pair..tAdj8Pair are compile-time fixed-width for performance
    /// on hot-path adjacencies (face2cell = Adj2, bnd2face = Adj1, etc.).
    using AdjVariant = std::variant<
        tAdjPair,   // variable-width (cell2node, cell2face, node2cell, cell2cell, ...)
        tAdj1Pair,  // 1 per row (bnd2face, face2bnd, cell2cellOrig, ...)
        tAdj2Pair,  // 2 per row (face2cell, bnd2cell)
        tAdj3Pair,  // 3 per row
        tAdj4Pair,  // 4 per row
        tAdj8Pair>; // 8 per row

    // =================================================================
    // ConeAdj: downward adjacency (higher depth → lower depth)
    // =================================================================

    /// A cone (downward) adjacency from entities at `fromDepth` to entities at
    /// `toDepth`. Row order is defined by element topology and must not be
    /// permuted (e.g., cell2node ordering determines element shape functions).
    ///
    /// Periodic bits (`pbi`) are only meaningful when `toDepth == 0` (nodes).
    /// For non-periodic meshes or non-node targets, `pbi` remains uninitialized.
    struct ConeAdj
    {
        int fromDepth{-1}; ///< Source stratum (e.g., dim for cells, dim-1 for faces)
        int toDepth{-1};   ///< Target stratum (e.g., 0 for nodes, dim-1 for faces)
        AdjVariant adj;    ///< The adjacency pair (typed by row width)
        tPbiPair pbi;      ///< Periodic bits per entry (only for toDepth==0, optional)

        /// Access as variable-width tAdjPair. Throws if adj holds a fixed-width type.
        tAdjPair &asAdj() { return std::get<tAdjPair>(adj); }
        const tAdjPair &asAdj() const { return std::get<tAdjPair>(adj); }

        /// Access as fixed-width tAdj2Pair (e.g., for cell2face with 2 entries).
        tAdj2Pair &asAdj2() { return std::get<tAdj2Pair>(adj); }
        const tAdj2Pair &asAdj2() const { return std::get<tAdj2Pair>(adj); }

        tAdj1Pair &asAdj1() { return std::get<tAdj1Pair>(adj); }
        const tAdj1Pair &asAdj1() const { return std::get<tAdj1Pair>(adj); }

        /// Number of local (father) rows.
        index fatherSize() const
        {
            return std::visit([](const auto &p) -> index
                              { return p.father ? p.father->Size() : 0; }, adj);
        }

        /// Check if the adjacency pair is initialized (father is non-null).
        bool initialized() const
        {
            return std::visit([](const auto &p) -> bool
                              { return bool(p.father); }, adj);
        }

        /// Check if pbi is attached (only valid for toDepth==0).
        bool hasPbi() const { return bool(pbi.father); }
    };

    // =================================================================
    // SupportAdj: upward adjacency (lower depth → higher depth)
    // =================================================================

    /// A support (upward) adjacency from entities at `fromDepth` to entities at
    /// `toDepth`. Row order is determined by the creation method (typically
    /// Inverse), and is stable but not semantically ordered.
    ///
    /// Supports never carry periodic bits — pbi is a property of cones to nodes.
    struct SupportAdj
    {
        int fromDepth{-1}; ///< Source stratum (e.g., 0 for nodes)
        int toDepth{-1};   ///< Target stratum (e.g., dim for cells)
        AdjVariant adj;    ///< The adjacency pair (typed by row width)

        /// Access as variable-width tAdjPair.
        tAdjPair &asAdj() { return std::get<tAdjPair>(adj); }
        const tAdjPair &asAdj() const { return std::get<tAdjPair>(adj); }

        tAdj2Pair &asAdj2() { return std::get<tAdj2Pair>(adj); }
        const tAdj2Pair &asAdj2() const { return std::get<tAdj2Pair>(adj); }

        tAdj1Pair &asAdj1() { return std::get<tAdj1Pair>(adj); }
        const tAdj1Pair &asAdj1() const { return std::get<tAdj1Pair>(adj); }

        index fatherSize() const
        {
            return std::visit([](const auto &p) -> index
                              { return p.father ? p.father->Size() : 0; }, adj);
        }

        bool initialized() const
        {
            return std::visit([](const auto &p) -> bool
                              { return bool(p.father); }, adj);
        }
    };

    // =================================================================
    // SharedCountPredicate
    // =================================================================

    /// Predicate for ComposeFiltered: keep (A, C) pairs sharing >= minShared
    /// intermediate B-entities.
    struct SharedCountPredicate
    {
        int minShared{1};
        bool removeSelf{false};

        bool operator()(index a, index c, int nShared) const
        {
            if (removeSelf && a == c)
                return false;
            return nShared >= minShared;
        }
    };

    // =================================================================
    // MeshConnectivity: the layered DAG
    // =================================================================

    /// Manages the layered DAG of mesh adjacency relations.
    ///
    /// Cones (downward adjacencies) and supports (upward adjacencies) are stored
    /// in separate vectors. Each is identified by a `(fromDepth, toDepth)` pair
    /// using dynamic depth tags (e.g., `(dim, 0)` for cell→node).
    struct MeshConnectivity
    {
        int meshDim{0};
        std::vector<ConeAdj> cones;
        std::vector<SupportAdj> supports;

        // -----------------------------------------------------------------
        // Cone management
        // -----------------------------------------------------------------

        /// Add a new cone for (fromDepth, toDepth). Returns reference.
        /// Asserts no duplicate exists.
        ConeAdj &addCone(int fromDepth, int toDepth);

        /// Find a cone by (fromDepth, toDepth). Returns nullptr if not found.
        ConeAdj *findCone(int fromDepth, int toDepth);
        const ConeAdj *findCone(int fromDepth, int toDepth) const;
        bool hasCone(int fromDepth, int toDepth) const;

        // -----------------------------------------------------------------
        // Support management
        // -----------------------------------------------------------------

        /// Add a new support for (fromDepth, toDepth). Returns reference.
        /// Asserts no duplicate exists.
        SupportAdj &addSupport(int fromDepth, int toDepth);

        /// Find a support by (fromDepth, toDepth). Returns nullptr if not found.
        SupportAdj *findSupport(int fromDepth, int toDepth);
        const SupportAdj *findSupport(int fromDepth, int toDepth) const;
        bool hasSupport(int fromDepth, int toDepth) const;

        // -----------------------------------------------------------------
        // Core DSL operations (static, pure-functional on tAdjPair)
        // -----------------------------------------------------------------
        // These operate on variable-width tAdjPair. Fixed-width adjacencies
        // must be widened before use or the caller extracts tAdjPair from
        // the variant.

        /// Invert a cone to get its support (distributed, MPI push-back).
        ///
        /// Given a cone adjacency A→B (CSR: for each A-entity, list of B-entities),
        /// compute the support adjacency B→A (for each B-entity, list of A-entities
        /// that reference it). Result is globally complete via MPI push-back.
        ///
        /// @param cone         CSR adjacency: from → to (global indices).
        /// @param nToLocal     Local number of "to" entities on this rank.
        /// @param mpi          MPI communicator.
        /// @param fromLocal2Global  Maps local from-entity index to global index.
        /// @param toLocal2Global    Maps local to-entity index to global index.
        /// @param toGlobalMapping   Global mapping for to-entities (ownership lookup).
        /// @return             CSR adjacency: to → from (global, complete). Father-only.
        static tAdjPair Inverse(
            const tAdjPair &cone,
            index nToLocal,
            const MPIInfo &mpi,
            const std::function<index(index)> &fromLocal2Global,
            const std::function<index(index)> &toLocal2Global,
            const ssp<GlobalOffsetsMapping> &toGlobalMapping);

        /// Compose two adjacencies: A→B + B→C → A→C (delegates to ComposeFiltered
        /// with SharedCountPredicate{1}).
        static tAdjPair Compose(
            const tAdjPair &AB,
            const tAdjPair &BC,
            index nALocal,
            const std::unordered_map<index, index> &bGlobal2Local,
            const std::function<index(index)> &aLocal2Global,
            bool removeSelf = false);

        /// Compose two adjacencies with on-the-fly predicate filtering.
        ///
        /// For each row a in AB, iterates b in AB[a], collects c in BC[b].
        /// Counts shared B-entities per candidate, applies predicate.
        ///
        /// @param AB           CSR: A → B (father only, global B indices).
        /// @param BC           CSR: B → C (father+son, global C indices).
        /// @param nALocal      Number of local A-entities.
        /// @param bGlobal2Local Maps global B-index to local-appended index in BC.
        /// @param aLocal2Global Maps local A-index to global A-index.
        /// @param pred         Predicate(a_global, c_global, nShared) → keep?
        /// @return             CSR: A → C (global C indices), father-only.
        template <class Predicate>
        static tAdjPair ComposeFiltered(
            const tAdjPair &AB,
            const tAdjPair &BC,
            index nALocal,
            const std::unordered_map<index, index> &bGlobal2Local,
            const std::function<index(index)> &aLocal2Global,
            Predicate &&pred);
    };

    // =====================================================================
    // Template implementation: ComposeFiltered
    // =====================================================================

    template <class Predicate>
    tAdjPair MeshConnectivity::ComposeFiltered(
        const tAdjPair &AB,
        const tAdjPair &BC,
        index nALocal,
        const std::unordered_map<index, index> &bGlobal2Local,
        const std::function<index(index)> &aLocal2Global,
        Predicate &&pred)
    {
        const auto &mpi = AB.father->getMPI();
        tAdjPair result;
        result.InitPair("ComposeFiltered_result", mpi);
        result.father->Resize(nALocal);

        for (index iA = 0; iA < nALocal; iA++)
        {
            index aGlobal = aLocal2Global(iA);

            // Collect all candidate C-entities with their shared-B count
            std::unordered_map<index, int> candidateSharedCount;
            for (auto iB : AB.father->operator[](iA))
            {
                auto it = bGlobal2Local.find(iB);
                DNDS_assert_info(it != bGlobal2Local.end(),
                                 fmt::format("ComposeFiltered: B-entity {} not found in bGlobal2Local", iB));
                index bLocal = it->second;
                for (auto iC : BC[bLocal])
                    candidateSharedCount[iC]++;
            }

            // Apply predicate and collect passing entries
            std::vector<index> accepted;
            for (auto &[cGlobal, nShared] : candidateSharedCount)
            {
                if (pred(aGlobal, cGlobal, nShared))
                    accepted.push_back(cGlobal);
            }

            // Sort for deterministic output
            std::sort(accepted.begin(), accepted.end());

            result.father->ResizeRow(iA, accepted.size());
            for (rowsize j = 0; j < static_cast<rowsize>(accepted.size()); j++)
                result.father->operator()(iA, j) = accepted[j];
        }

        return result;
    }

} // namespace DNDS::Geom

#pragma once
/// @file MeshConnectivity_StateChecked.hpp
/// @brief Thin free-function wrappers over MeshConnectivity DSL methods
///        that assert per-adjacency state (AdjPairTracked::idx) before
///        delegating to the bare-pair DSL.
///
/// MeshConnectivity operates on bare ArrayAdjacencyPair<rs> and has no
/// knowledge of AdjIndexInfo. These wrappers bridge the gap: they accept
/// AdjPairTracked<T> inputs, validate state, and forward to the DSL.

#include "AdjIndexInfo.hpp"
#include "MeshConnectivity.hpp"

namespace DNDS::Geom
{
    // =================================================================
    // Checked Inverse
    // =================================================================

    /// State-checked wrapper for MeshConnectivity::Inverse.
    ///
    /// Extracts L2G callbacks from cone and toPair ghost mappings.
    /// Returns an AdjPairTracked with:
    ///   - father adopted from the DSL result (no copy)
    ///   - idx.state() == Adj_PointToGlobal
    ///
    /// @param cone    A → B (global state, from-entity = A).
    /// @param toPair  Any ArrayPair for entity B (must have trans.pLGhostMapping).
    /// @param nToLocal  Number of local B-entities.
    /// @param mpi       MPI communicator.
    template <rowsize cone_rs = NonUniformSize, class ToPair>
    AdjPairTracked<tAdjPair> CheckedInverse(
        const AdjPairTracked<ArrayAdjacencyPair<cone_rs>> &cone,
        const ToPair &toPair,
        index nToLocal,
        const MPIInfo &mpi)
    {
        DNDS_assert_info(cone.idx.state() == Adj_PointToGlobal,
                         "CheckedInverse: cone must be in Adj_PointToGlobal state");
        DNDS_assert_info(cone.trans.pLGhostMapping,
                         "CheckedInverse: cone.trans.pLGhostMapping must be set");
        DNDS_assert_info(toPair.trans.pLGhostMapping,
                         "CheckedInverse: toPair.trans.pLGhostMapping must be set");
        DNDS_assert_info(toPair.father->pLGlobalMapping,
                         "CheckedInverse: toPair.father->pLGlobalMapping must be set");

        auto fromL2G = [&](index i) -> index
        { return cone.trans.pLGhostMapping->operator()(-1, i); };
        auto toL2G = [&](index i) -> index
        { return toPair.trans.pLGhostMapping->operator()(-1, i); };

        auto dslResult = MeshConnectivity::Inverse<cone_rs>(
            static_cast<const ArrayAdjacencyPair<cone_rs> &>(cone),
            nToLocal, mpi,
            std::function<index(index)>(fromL2G),
            std::function<index(index)>(toL2G),
            toPair.father->pLGlobalMapping);

        AdjPairTracked<tAdjPair> result;
        result.father = std::move(dslResult.father);
        result.son = make_ssp<tAdjPair::t_arr>(
            ObjName{"CheckedInverse.son"}, result.father->getMPI());
        result.idx.markGlobal();
        return result;
    }

    // =================================================================
    // Checked ComposeFiltered
    // =================================================================

    /// State-checked wrapper for MeshConnectivity::ComposeFiltered.
    ///
    /// Extracts aLocal2Global from AB's ghost mapping.
    /// Returns an AdjPairTracked with:
    ///   - father adopted from the DSL result (no copy)
    ///   - idx.state() == Adj_PointToGlobal
    ///
    /// AB must have a valid trans.pLGhostMapping (e.g., via EnsureGhostMapping).
    template <rowsize rs_AB = NonUniformSize, rowsize rs_BC = NonUniformSize,
              rowsize out_rs = NonUniformSize, class Predicate, class... TArgs>
    AdjPairTracked<ArrayAdjacencyPair<out_rs>> CheckedComposeFiltered(
        const AdjPairTracked<ArrayAdjacencyPair<rs_AB>> &AB,
        const AdjPairTracked<ArrayAdjacencyPair<rs_BC>> &BC,
        const std::unordered_map<index, index> &bGlobal2Local,
        Predicate &&pred,
        TArgs &&...args)
    {
        DNDS_assert_info(AB.idx.state() == Adj_PointToGlobal,
                         "CheckedComposeFiltered: AB must be in Adj_PointToGlobal state");
        DNDS_assert_info(BC.idx.state() == Adj_PointToGlobal,
                         "CheckedComposeFiltered: BC must be in Adj_PointToGlobal state");
        DNDS_assert_info(AB.trans.pLGhostMapping,
                         "CheckedComposeFiltered: AB.trans.pLGhostMapping must be set");

        auto aL2G = [&](index i) -> index
        { return AB.trans.pLGhostMapping->operator()(-1, i); };

        auto dslResult = MeshConnectivity::ComposeFiltered<rs_AB, rs_BC, out_rs>(
            static_cast<const ArrayAdjacencyPair<rs_AB> &>(AB),
            static_cast<const ArrayAdjacencyPair<rs_BC> &>(BC),
            AB.father->Size(), bGlobal2Local,
            std::function<index(index)>(aL2G),
            std::forward<Predicate>(pred),
            std::forward<TArgs>(args)...);

        AdjPairTracked<ArrayAdjacencyPair<out_rs>> result;
        result.father = std::move(dslResult.father);
        result.son = make_ssp<typename ArrayAdjacencyPair<out_rs>::t_arr>(
            ObjName{"CheckedComposeFiltered.son"}, result.father->getMPI());
        result.idx.markGlobal();
        return result;
    }

    // =================================================================
    // Checked InterpolateGlobal
    // =================================================================

    /// State-checked wrapper for MeshConnectivity::InterpolateGlobal.
    template <rowsize p2n_rs = NonUniformSize, rowsize e2p_rs = NonUniformSize>
    auto CheckedInterpolateGlobal(
        const AdjPairTracked<ArrayAdjacencyPair<p2n_rs>> &parent2node,
        const tPbiPair &parent2nodePbi,
        const OffsetAscendIndexMapping &parentGhostMapping,
        const GlobalOffsetsMapping &parentGlobalMapping,
        const OffsetAscendIndexMapping &nodeGhostMapping,
        const SubEntityQueryPbi &query,
        index nLocalParents,
        index nTotalParents,
        index nNode,
        const OwnershipResolverMulti &resolver,
        const MPIInfo &mpi)
    {
        DNDS_assert_info(parent2node.idx.state() == Adj_PointToLocal,
                         "CheckedInterpolateGlobal: parent2node must be in Adj_PointToLocal state");
        // Verify mapping consistency when wired
        DNDS_assert_info(!parent2node.idx.isWired() ||
                             parent2node.idx.mapping().get() == &nodeGhostMapping,
                         "CheckedInterpolateGlobal: nodeGhostMapping parameter does not match "
                         "parent2node.idx.targetMapping");
        return MeshConnectivity::InterpolateGlobal<p2n_rs, e2p_rs>(
            static_cast<const ArrayAdjacencyPair<p2n_rs> &>(parent2node),
            parent2nodePbi,
            parentGhostMapping, parentGlobalMapping, nodeGhostMapping,
            query, nLocalParents, nTotalParents, nNode, resolver, mpi);
    }

} // namespace DNDS::Geom

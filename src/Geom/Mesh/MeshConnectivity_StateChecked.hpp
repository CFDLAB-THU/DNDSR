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
    /// Asserts that the cone adjacency is in Adj_PointToGlobal state
    /// (Inverse requires global indices).
    ///
    /// @tparam cone_rs  Row-size of the cone adjacency.
    /// @param cone      AdjPairTracked cone (A → B, global indices).
    /// @param args      Remaining arguments forwarded to MeshConnectivity::Inverse.
    /// @return          Same as MeshConnectivity::Inverse.
    template <rowsize cone_rs = NonUniformSize, class... TArgs>
    tAdjPair CheckedInverse(
        const AdjPairTracked<ArrayAdjacencyPair<cone_rs>> &cone,
        TArgs &&...args)
    {
        DNDS_assert_info(cone.idx.state() == Adj_PointToGlobal,
                         "CheckedInverse: cone must be in Adj_PointToGlobal state");
        return MeshConnectivity::Inverse<cone_rs>(
            static_cast<const ArrayAdjacencyPair<cone_rs> &>(cone),
            std::forward<TArgs>(args)...);
    }

    // =================================================================
    // Checked ComposeFiltered
    // =================================================================

    /// State-checked wrapper for MeshConnectivity::ComposeFiltered.
    ///
    /// Asserts that both input adjacencies are in Adj_PointToGlobal state.
    template <rowsize rs_AB = NonUniformSize, rowsize rs_BC = NonUniformSize,
              rowsize out_rs = NonUniformSize, class Predicate, class... TArgs>
    ArrayAdjacencyPair<out_rs> CheckedComposeFiltered(
        const AdjPairTracked<ArrayAdjacencyPair<rs_AB>> &AB,
        const AdjPairTracked<ArrayAdjacencyPair<rs_BC>> &BC,
        index nALocal,
        const std::unordered_map<index, index> &bGlobal2Local,
        const std::function<index(index)> &aLocal2Global,
        Predicate &&pred,
        TArgs &&...args)
    {
        DNDS_assert_info(AB.idx.state() == Adj_PointToGlobal,
                         "CheckedComposeFiltered: AB must be in Adj_PointToGlobal state");
        DNDS_assert_info(BC.idx.state() == Adj_PointToGlobal,
                         "CheckedComposeFiltered: BC must be in Adj_PointToGlobal state");
        return MeshConnectivity::ComposeFiltered<rs_AB, rs_BC, out_rs>(
            static_cast<const ArrayAdjacencyPair<rs_AB> &>(AB),
            static_cast<const ArrayAdjacencyPair<rs_BC> &>(BC),
            nALocal, bGlobal2Local, aLocal2Global,
            std::forward<Predicate>(pred),
            std::forward<TArgs>(args)...);
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

// Template implementations for Interpolate family.
// Included at the bottom of MeshConnectivity.hpp.
// Do NOT include this file directly — include MeshConnectivity.hpp instead.

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <fmt/core.h>

namespace DNDS::Geom
{
    template <rowsize p2n_rs>
    InterpolateResult MeshConnectivity::Interpolate(
        const ArrayAdjacencyPair<p2n_rs> &parent2node,
        const SubEntityQuery &query,
        index nParent,
        index nNode,
        const MPIInfo &mpi)
    {
        InterpolateResult result;

        // Allocate parent2entity
        result.parent2entity.InitPair("Interpolate_p2e", mpi);
        result.parent2entity.father->Resize(nParent);

        auto &entityElemInfoV = result.entityElemInfo;
        index &nEntities = result.nEntities;
        nEntities = 0;

        // Temporary storage (compacted into ArrayPair at end)
        std::vector<std::vector<index>> ent2nodeVec;
        std::vector<std::vector<index>> ent2parentVec;  // variable-width: accumulates all parents

        // Per-entity creating (parent, sub) pair — needed for matchExtra callback
        struct EntityOrigin
        {
            index parentIdx;
            int subIdx;
        };
        std::vector<EntityOrigin> ent2origin;

        // Reverse index: for each node, which entities contain it.
        // Used for O(1)-amortized deduplication lookup.
        std::vector<std::vector<index>> node2entity(nNode);

        // Scratch buffer for extracted sub-entity nodes
        constexpr int maxSubEntityNodes = 10; // Quad9 = 9, padded
        std::array<index, maxSubEntityNodes> subNodesBuf{};

        const bool hasMatchExtra = bool(query.matchExtra);

        for (index iParent = 0; iParent < nParent; iParent++)
        {
            int nSubs = query.numSubEntities(iParent);
            result.parent2entity.father->ResizeRow(iParent, nSubs);

            // Adapter: wrap parent2node[iParent] row as a function(int) -> index
            auto parentNodeAccessor = [&](int j) -> index
            {
                return parent2node[iParent][j];
            };

            for (int iSub = 0; iSub < nSubs; iSub++)
            {
                SubEntityDesc desc = query.describe(iParent, iSub);
                DNDS_assert_info(desc.nNodes <= maxSubEntityNodes,
                                 fmt::format("Interpolate: sub-entity has {} nodes, max is {}",
                                             desc.nNodes, maxSubEntityNodes));

                // Extract sub-entity nodes from parent's node list
                query.extractNodes(iParent, iSub, parentNodeAccessor, subNodesBuf.data());

                // Sort vertices for comparison (first nVertices entries)
                std::vector<index> subVerts(subNodesBuf.begin(),
                                            subNodesBuf.begin() + desc.nVertices);
                std::sort(subVerts.begin(), subVerts.end());

                // Deduplicate: search existing entities via reverse index
                index iFound = -1;
                for (auto iV : subVerts)
                {
                    if (iFound >= 0)
                        break;
                    DNDS_assert_info(iV >= 0 && iV < nNode,
                                     fmt::format("Interpolate: node index {} out of range [0, {})",
                                                 iV, nNode));
                    for (auto iEnt : node2entity[iV])
                    {
                        // Type-tag must match
                        if (entityElemInfoV[iEnt].type != desc.typeTag)
                            continue;

                        // Compare sorted vertices
                        auto &entNodes = ent2nodeVec[iEnt];
                        int nEntVerts = desc.nVertices;

                        std::vector<index> entVerts(entNodes.begin(),
                                                    entNodes.begin() + nEntVerts);
                        std::sort(entVerts.begin(), entVerts.end());

                        if (!std::equal(subVerts.begin(), subVerts.end(),
                                        entVerts.begin(), entVerts.end()))
                            continue;

                        // Vertex match found. Apply optional extra predicate.
                        if (hasMatchExtra)
                        {
                            auto &orig = ent2origin[iEnt];
                            if (!query.matchExtra(iParent, iSub,
                                                  iEnt,
                                                  orig.parentIdx, orig.subIdx))
                                continue; // Rejected by extra predicate
                        }

                        iFound = iEnt;
                        break;
                    }
                }

                if (iFound < 0)
                {
                    // New sub-entity
                    std::vector<index> subNodesVec(subNodesBuf.begin(),
                                                   subNodesBuf.begin() + desc.nNodes);
                    ent2nodeVec.emplace_back(std::move(subNodesVec));
                    ent2parentVec.push_back({iParent});
                    entityElemInfoV.emplace_back(ElemInfo{desc.typeTag, 0});
                    ent2origin.push_back({iParent, iSub});

                    // Register in reverse index (vertices only)
                    for (auto iV : subVerts)
                        node2entity[iV].push_back(nEntities);

                    result.parent2entity.father->operator()(iParent, iSub) = nEntities;
                    nEntities++;
                }
                else
                {
                    // Existing sub-entity: append parent to the parent list.
                    ent2parentVec[iFound].push_back(iParent);

                    result.parent2entity.father->operator()(iParent, iSub) = iFound;
                }
            }
        }

        // ----- Compact temporary vectors into ArrayPair structures -----

        // entity2node
        result.entity2node.InitPair("Interpolate_e2n", mpi);
        result.entity2node.father->Resize(nEntities);
        for (index iEnt = 0; iEnt < nEntities; iEnt++)
        {
            auto &nodes = ent2nodeVec[iEnt];
            result.entity2node.father->ResizeRow(iEnt, nodes.size());
            for (rowsize j = 0; j < static_cast<rowsize>(nodes.size()); j++)
                result.entity2node.father->operator()(iEnt, j) = nodes[j];
        }

        // entity2parent (variable-width)
        result.entity2parent.InitPair("Interpolate_e2p", mpi);
        result.entity2parent.father->Resize(nEntities);
        for (index iEnt = 0; iEnt < nEntities; iEnt++)
        {
            auto &parents = ent2parentVec[iEnt];
            result.entity2parent.father->ResizeRow(iEnt, parents.size());
            for (rowsize j = 0; j < static_cast<rowsize>(parents.size()); j++)
                result.entity2parent.father->operator()(iEnt, j) = parents[j];
        }

        return result;
    }

    template <rowsize p2n_rs>
    InterpolateDistributedResult MeshConnectivity::InterpolateDistributed(
        const ArrayAdjacencyPair<p2n_rs> &parent2node,
        const OffsetAscendIndexMapping &parentGhostMapping,
        const SubEntityQuery &query,
        index nLocalParents,
        index nTotalParents,
        index nNode,
        const OwnershipResolver2 &resolver,
        const OffsetAscendIndexMapping &nodeGhostMapping,
        const MPIInfo &mpi)
    {
        // ============================================================
        // Step 1: Local interpolation on all parents (local + ghost)
        // ============================================================
        auto localResult = Interpolate(parent2node, query, nTotalParents, nNode, mpi);

        // ============================================================
        // Step 2: Ownership decision and compaction mapping
        // ============================================================
        index nAllEntities = localResult.nEntities;
        std::vector<index> oldToOwned(nAllEntities, UnInitIndex);
        std::vector<std::vector<index>> pushPerRank(mpi.size);

        index nOwned = 0;
        for (index iEnt = 0; iEnt < nAllEntities; iEnt++)
        {
            index parentL = localResult.entity2parent.father->operator()(iEnt, 0);
            index parentR = (localResult.entity2parent.father->RowSize(iEnt) >= 2)
                                ? localResult.entity2parent.father->operator()(iEnt, 1)
                                : UnInitIndex;

            auto decision = resolver(parentL, parentR, nLocalParents);
            if (decision.owned)
            {
                oldToOwned[iEnt] = nOwned;
                for (auto pr : decision.peerRanks)
                    if (pr >= 0 && pr != mpi.rank)
                        pushPerRank[pr].push_back(nOwned);
                nOwned++;
            }
        }

        // ============================================================
        // Step 3: Build owned entity arrays (father) with global
        //         parent indices in entity2parent
        // ============================================================
        InterpolateDistributedResult result;
        result.nOwnedEntities = nOwned;

        // entity2node (father = owned, local node indices)
        result.entity2node.InitPair("InterpDist_e2n", mpi);
        result.entity2node.father->Resize(nOwned);

        // entity2parent (father = owned, GLOBAL parent indices)
        result.entity2parent.InitPair("InterpDist_e2p", mpi);
        result.entity2parent.father->Resize(nOwned);

        // entityElemInfo (father = owned)
        result.entityElemInfo.InitPair("InterpDist_eInfo", mpi);
        result.entityElemInfo.father->Resize(nOwned);

        for (index iEnt = 0; iEnt < nAllEntities; iEnt++)
        {
            index iNew = oldToOwned[iEnt];
            if (iNew == UnInitIndex)
                continue;

            // entity2node: convert local node indices to GLOBAL
            auto row = localResult.entity2node.father->operator[](iEnt);
            result.entity2node.father->ResizeRow(iNew, row.size());
            for (rowsize j = 0; j < static_cast<rowsize>(row.size()); j++)
                result.entity2node.father->operator()(iNew, j) =
                    nodeGhostMapping(-1, row[j]);

            // entity2parent: convert local-appended → global parent indices.
            // InterpolateLocal produces variable-width; extract first 2 for the
            // fixed-2 InterpolateDistributedResult.
            {
                auto parentRow = localResult.entity2parent.father->operator[](iEnt);
                index pL = parentRow[0];
                index pR = (parentRow.size() >= 2) ? parentRow[1] : UnInitIndex;
                result.entity2parent.father->operator()(iNew, 0) =
                    parentGhostMapping(-1, pL);
                result.entity2parent.father->operator()(iNew, 1) =
                    (pR == UnInitIndex) ? UnInitIndex : parentGhostMapping(-1, pR);
            }

            // entityElemInfo
            result.entityElemInfo.father->operator()(iNew, 0) =
                localResult.entityElemInfo[iEnt];
        }
        result.entity2node.father->Compress();

        // ============================================================
        // Step 4: Push-based ghost exchange
        // ============================================================
        std::vector<index> pushIdx;
        std::vector<index> pushStarts(mpi.size + 1, 0);
        for (MPI_int r = 0; r < mpi.size; r++)
            pushStarts[r + 1] = pushStarts[r] + static_cast<index>(pushPerRank[r].size());
        pushIdx.resize(pushStarts.back());
        for (MPI_int r = 0; r < mpi.size; r++)
            std::copy(pushPerRank[r].begin(), pushPerRank[r].end(),
                      pushIdx.begin() + pushStarts[r]);

        result.entity2parent.TransAttach();
        result.entity2parent.trans.createFatherGlobalMapping();
        result.entity2parent.trans.createGhostMapping(pushIdx, pushStarts);
        result.entity2parent.trans.createMPITypes();
        result.entity2parent.trans.pullOnce();
        // Now entity2parent.son has ghost entities with global parent indices.

        result.entity2node.TransAttach();
        result.entity2node.trans.BorrowGGIndexing(result.entity2parent.trans);
        result.entity2node.trans.createMPITypes();
        result.entity2node.trans.pullOnce();

        result.entityElemInfo.TransAttach();
        result.entityElemInfo.trans.BorrowGGIndexing(result.entity2parent.trans);
        result.entityElemInfo.trans.createMPITypes();
        result.entityElemInfo.trans.pullOnce();

        // ============================================================
        // Step 5: Convert entity2parent from global to local-appended
        //         parent indices (receiver side)
        // ============================================================
        index nGhostEntities = result.entity2parent.son
                                   ? result.entity2parent.son->Size()
                                   : 0;
        for (index iGhost = 0; iGhost < nGhostEntities; iGhost++)
        {
            for (rowsize side = 0; side < 2; side++)
            {
                index &pGlobal = (*result.entity2parent.son)(iGhost, side);
                if (pGlobal == UnInitIndex)
                    continue;
                auto [found, rank, localAppend] =
                    parentGhostMapping.search_indexAppend(pGlobal);
                DNDS_assert_info(found,
                                 fmt::format("InterpolateDistributed: ghost entity parent "
                                             "global {} not found locally",
                                             pGlobal));
                pGlobal = localAppend;
            }
        }
        // Convert father entity2parent back to local-appended too.
        for (index iOwned = 0; iOwned < nOwned; iOwned++)
        {
            for (rowsize side = 0; side < 2; side++)
            {
                index &pGlobal = result.entity2parent.father->operator()(iOwned, side);
                if (pGlobal == UnInitIndex)
                    continue;
                auto [found, rank, localAppend] =
                    parentGhostMapping.search_indexAppend(pGlobal);
                DNDS_assert_info(found,
                                 fmt::format("InterpolateDistributed: owned entity parent "
                                             "global {} not found locally",
                                             pGlobal));
                pGlobal = localAppend;
            }
        }

        // Convert entity2node from global to local-appended node indices
        // (for both father/owned and son/ghost).
        for (index iFace = 0; iFace < nOwned + nGhostEntities; iFace++)
        {
            for (rowsize j = 0; j < result.entity2node.RowSize(iFace); j++)
            {
                index &nGlobal = result.entity2node(iFace, j);
                auto [found, rank, localAppend] =
                    nodeGhostMapping.search_indexAppend(nGlobal);
                DNDS_assert_info(found,
                                 fmt::format("InterpolateDistributed: entity node "
                                             "global {} not found locally",
                                             nGlobal));
                nGlobal = localAppend;
            }
        }

        // ============================================================
        // Step 6: Build parent2entity with local-appended entity indices
        // ============================================================
        // Build a reverse map from ghost entities: for each ghost entity,
        // which parent slots does it fill?
        //
        // First, initialize parent2entity from the local Interpolate result,
        // remapping owned entities via oldToOwned.
        result.parent2entity.InitPair("InterpDist_p2e", mpi);
        result.parent2entity.father->Resize(nLocalParents);
        result.parent2entity.son = std::make_shared<tAdj::element_type>(
            ObjName{"InterpDist_p2e.son"}, mpi);
        result.parent2entity.son->Resize(nTotalParents - nLocalParents);

        for (index iParent = 0; iParent < nTotalParents; iParent++)
        {
            rowsize nSubs = localResult.parent2entity.father->RowSize(iParent);
            result.parent2entity.ResizeRow(iParent, nSubs);

            for (rowsize j = 0; j < nSubs; j++)
            {
                index oldEntIdx = localResult.parent2entity.father->operator()(iParent, j);
                index newEntIdx = oldToOwned[oldEntIdx];
                // Owned → direct mapping. Non-owned → -1 (resolved next).
                result.parent2entity(iParent, j) = (newEntIdx != UnInitIndex)
                                                       ? newEntIdx
                                                       : -1;
            }
        }

        // Resolve -1 entries using ghost entities.
        // For each ghost entity, find which parents reference it and match the
        // correct parent2entity slot by comparing vertex sets.
        for (index iGhost = 0; iGhost < nGhostEntities; iGhost++)
        {
            index entityLocalAppended = nOwned + iGhost;

            // Build sorted vertex set for this ghost entity.
            auto ghostRow = result.entity2node[entityLocalAppended];
            // Note: we need vertices only (first nVertices entries).
            // Since we don't have the SubEntityDesc here, use all nodes for matching.
            // Actually, entity2node stores ALL nodes (vertices + higher-order).
            // For vertex-set matching, we sort the full node set.
            std::vector<index> ghostNodes(ghostRow.begin(), ghostRow.end());
            std::sort(ghostNodes.begin(), ghostNodes.end());

            for (rowsize side = 0; side < 2; side++)
            {
                index parentLocalAppended = result.entity2parent(entityLocalAppended, side);
                if (parentLocalAppended == UnInitIndex)
                    continue;
                rowsize nSubs = result.parent2entity.RowSize(parentLocalAppended);
                bool found = false;
                for (rowsize j = 0; j < nSubs; j++)
                {
                    if (result.parent2entity(parentLocalAppended, j) != -1)
                        continue;

                    // Get the old entity that was in this slot and compare node sets.
                    index oldEntIdx = localResult.parent2entity.father->operator()(parentLocalAppended, j);
                    auto oldRow = localResult.entity2node.father->operator[](oldEntIdx);
                    std::vector<index> oldNodes(oldRow.begin(), oldRow.end());
                    std::sort(oldNodes.begin(), oldNodes.end());

                    if (ghostNodes == oldNodes)
                    {
                        result.parent2entity(parentLocalAppended, j) = entityLocalAppended;
                        found = true;
                        break;
                    }
                }
                DNDS_assert_info(found,
                                 fmt::format("InterpolateDistributed: ghost entity {} "
                                             "could not be matched to parent {}",
                                             iGhost, parentLocalAppended));
            }
        }

        return result;
    }

    // =================================================================
    // InterpolateGlobal: distributed sub-entity creation
    // =================================================================

    template <rowsize p2n_rs, rowsize e2p_rs>
    InterpolateGlobalResultT<e2p_rs> MeshConnectivity::InterpolateGlobal(
        const ArrayAdjacencyPair<p2n_rs> &parent2node,
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
        const bool hasPbi = bool(parent2nodePbi.father);

        // ============================================================
        // Step 1: Local interpolation on all parents (father + ghost)
        // ============================================================
        auto localResult = Interpolate(parent2node, query, nTotalParents, nNode, mpi);
        index nAllEntities = localResult.nEntities;

        // ============================================================
        // Step 2: Extract B→C pbi via SubEntityQueryPbi::extractPbi
        // ============================================================
        // Pbi is extracted from the FIRST parent's perspective.
        // Stored per local entity, used for dedup fingerprint and output.
        constexpr int maxSubEntityNodes = 10;
        std::vector<std::vector<NodePeriodicBits>> localEntityPbi(nAllEntities);
        if (hasPbi && query.extractPbi)
        {
            // For each entity, extract pbi from its first parent
            for (index iEnt = 0; iEnt < nAllEntities; iEnt++)
            {
                index firstParent = localResult.entity2parent.father->operator()(iEnt, 0);
                // Find which sub-entity slot of firstParent corresponds to iEnt
                int iSub = -1;
                for (rowsize j = 0; j < localResult.parent2entity.father->RowSize(firstParent); j++)
                {
                    if (localResult.parent2entity.father->operator()(firstParent, j) == iEnt)
                    {
                        iSub = j;
                        break;
                    }
                }
                DNDS_assert(iSub >= 0);

                auto desc = query.describe(firstParent, iSub);
                localEntityPbi[iEnt].resize(desc.nNodes);
                auto pbiAccessor = [&](int k) -> NodePeriodicBits
                {
                    return parent2nodePbi[firstParent][k];
                };
                query.extractPbi(firstParent, iSub, pbiAccessor,
                                 localEntityPbi[iEnt].data());
            }
        }

        // ============================================================
        // Step 2b: Compute parent2entityPbi for all parents
        // ============================================================
        // For each (parent, sub) pair, compute the uniform XOR between this
        // parent's sub-entity pbi and the entity's stored pbi (first-parent view).
        // This is a single NodePeriodicBits value per (parent, sub) slot.
        //
        // The XOR must be computed per matching node (by node index), not per
        // position, because different parents may enumerate the same face/edge
        // nodes in a different local order.
        std::vector<std::vector<NodePeriodicBits>> localParent2EntityPbi;
        if (hasPbi && query.extractPbi)
        {
            localParent2EntityPbi.resize(nTotalParents);
            std::array<NodePeriodicBits, maxSubEntityNodes> thisPbiBuf{};
            std::array<index, maxSubEntityNodes> thisNodeBuf{};
            for (index iParent = 0; iParent < nTotalParents; iParent++)
            {
                rowsize nSubs = localResult.parent2entity.father->RowSize(iParent);
                localParent2EntityPbi[iParent].resize(nSubs);
                for (rowsize j = 0; j < nSubs; j++)
                {
                    index iEnt = localResult.parent2entity.father->operator()(iParent, j);
                    auto desc = query.describe(iParent, j);

                    // Extract pbi and nodes from THIS parent's view
                    auto pbiAccessor = [&](int k) -> NodePeriodicBits
                    {
                        return parent2nodePbi[iParent][k];
                    };
                    query.extractPbi(iParent, j, pbiAccessor, thisPbiBuf.data());

                    auto nodeAccessor = [&](int k) -> index
                    {
                        return parent2node[iParent][k];
                    };
                    query.extractNodes(iParent, j, nodeAccessor, thisNodeBuf.data());

                    // Entity's stored pbi and nodes (from first parent's view)
                    auto &entPbi = localEntityPbi[iEnt];
                    auto entNodeRow = localResult.entity2node.father->operator[](iEnt);
                    DNDS_assert(desc.nNodes == static_cast<int>(entPbi.size()));

                    if (desc.nNodes > 0)
                    {
                        // Build sorted (node, pbi) pairs for this parent
                        using NP = std::pair<index, uint8_t>;
                        std::vector<NP> thisNP(desc.nNodes), entNP(desc.nNodes);
                        for (int k = 0; k < desc.nNodes; k++)
                        {
                            thisNP[k] = {thisNodeBuf[k], uint8_t(thisPbiBuf[k])};
                            entNP[k] = {entNodeRow[k], uint8_t(entPbi[k])};
                        }
                        auto cmp = [](const NP &a, const NP &b)
                        { return a.first == b.first ? a.second < b.second : a.first < b.first; };
                        std::sort(thisNP.begin(), thisNP.end(), cmp);
                        std::sort(entNP.begin(), entNP.end(), cmp);

                        // Verify same node set and compute uniform XOR
                        uint8_t xorVal = thisNP[0].second ^ entNP[0].second;
                        for (int k = 1; k < desc.nNodes; k++)
                        {
                            DNDS_assert_info(thisNP[k].first == entNP[k].first,
                                             fmt::format("parent2entityPbi: node mismatch at parent {} sub {} node {}: "
                                                         "this={}, ent={}",
                                                         iParent, j, k, thisNP[k].first, entNP[k].first));
                            uint8_t xk = thisNP[k].second ^ entNP[k].second;
                            DNDS_assert_info(xk == xorVal,
                                             fmt::format("parent2entityPbi: non-uniform XOR at parent {} sub {} node {}: "
                                                         "expected 0x{:02x}, got 0x{:02x}",
                                                         iParent, j, k, xorVal, xk));
                        }
                        localParent2EntityPbi[iParent][j] = NodePeriodicBits{xorVal};
                    }
                    else
                    {
                        localParent2EntityPbi[iParent][j] = NodePeriodicBits{0};
                    }
                }
            }
        }

        // ============================================================
        // Step 3: Classify entities and resolve ownership
        // ============================================================
        // For each entity, determine: fully local, fully ghost, or straddling.
        // For straddling, call the ownership resolver.

        // Map parent local-appended → rank
        auto getParentRank = [&](index parentLocal) -> MPI_int
        {
            index parentGlobal = parentGhostMapping(-1, parentLocal);
            MPI_int rank;
            index val;
            auto ret = parentGlobalMapping.search(parentGlobal, rank, val);
            DNDS_assert(ret);
            return rank;
        };

        struct EntityClassification
        {
            bool owned{false};
            bool discard{false}; // fully ghost
            std::vector<MPI_int> peerRanks; // ranks that need this B's global ID
        };
        std::vector<EntityClassification> classifications(nAllEntities);

        // Precompute per-entity: parent list + parent ranks
        for (index iEnt = 0; iEnt < nAllEntities; iEnt++)
        {
            auto parentRow = localResult.entity2parent.father->operator[](iEnt);
            bool anyLocal = false, anyGhost = false;
            std::vector<index> parents;
            std::vector<MPI_int> parentRanks;
            parents.reserve(parentRow.size());
            parentRanks.reserve(parentRow.size());

            for (auto p : parentRow)
            {
                parents.push_back(p);
                MPI_int r = getParentRank(p);
                parentRanks.push_back(r);
                if (p < nLocalParents)
                    anyLocal = true;
                else
                    anyGhost = true;
            }

            if (!anyLocal)
            {
                // Fully ghost — discard
                classifications[iEnt].discard = true;
                continue;
            }

            if (!anyGhost)
            {
                // Fully local — owned, no peers
                classifications[iEnt].owned = true;
                continue;
            }

            // Straddling — call ownership resolver
            auto decision = resolver(parents, parentRanks, nLocalParents);
            classifications[iEnt].owned = decision.owned;
            classifications[iEnt].discard = !decision.owned;
            if (decision.owned)
                classifications[iEnt].peerRanks = std::move(decision.peerRanks);
        }

        // ============================================================
        // Step 4: Compact owned entities, build owned arrays
        // ============================================================
        std::vector<index> oldToOwned(nAllEntities, UnInitIndex);
        index nOwned = 0;
        for (index iEnt = 0; iEnt < nAllEntities; iEnt++)
        {
            if (classifications[iEnt].owned)
            {
                oldToOwned[iEnt] = nOwned;
                nOwned++;
            }
        }

        InterpolateGlobalResultT<e2p_rs> result;
        result.nOwnedEntities = nOwned;

        // entity2node (global C indices)
        result.entity2node.InitPair("InterpGlobal_e2n", mpi);
        result.entity2node.father->Resize(nOwned);

        // entity2parent (global A indices, variable-width)
        result.entity2parent.InitPair("InterpGlobal_e2p", mpi);
        result.entity2parent.father->Resize(nOwned);

        // entity2nodePbi
        if (hasPbi)
            result.entity2nodePbi.InitPair("InterpGlobal_e2nPbi", mpi);
        if (hasPbi)
            result.entity2nodePbi.father->Resize(nOwned);

        // entityElemInfo
        result.entityElemInfo.InitPair("InterpGlobal_eInfo", mpi);
        result.entityElemInfo.father->Resize(nOwned);

        for (index iEnt = 0; iEnt < nAllEntities; iEnt++)
        {
            index iNew = oldToOwned[iEnt];
            if (iNew == UnInitIndex)
                continue;

            // entity2node → global C indices
            auto nodeRow = localResult.entity2node.father->operator[](iEnt);
            result.entity2node.father->ResizeRow(iNew, nodeRow.size());
            for (rowsize j = 0; j < static_cast<rowsize>(nodeRow.size()); j++)
                result.entity2node.father->operator()(iNew, j) =
                    nodeGhostMapping(-1, nodeRow[j]);

            // entity2parent → global A indices
            auto parentRow = localResult.entity2parent.father->operator[](iEnt);
            if constexpr (e2p_rs == NonUniformSize)
            {
                result.entity2parent.father->ResizeRow(iNew, parentRow.size());
                for (rowsize j = 0; j < static_cast<rowsize>(parentRow.size()); j++)
                    result.entity2parent.father->operator()(iNew, j) =
                        parentGhostMapping(-1, parentRow[j]);
            }
            else
            {
                // Fixed-width: fill up to e2p_rs, pad with UnInitIndex.
                for (rowsize j = 0; j < e2p_rs; j++)
                    result.entity2parent.father->operator()(iNew, j) =
                        (j < static_cast<rowsize>(parentRow.size()))
                            ? parentGhostMapping(-1, parentRow[j])
                            : UnInitIndex;
            }

            // entity2nodePbi
            if (hasPbi)
            {
                auto &pbi = localEntityPbi[iEnt];
                result.entity2nodePbi.father->ResizeRow(iNew, pbi.size());
                for (rowsize j = 0; j < static_cast<rowsize>(pbi.size()); j++)
                    result.entity2nodePbi.father->operator()(iNew, j) = pbi[j];
            }

            // entityElemInfo
            result.entityElemInfo.father->operator()(iNew, 0) =
                localResult.entityElemInfo[iEnt];
        }
        result.entity2node.father->Compress();
        if (hasPbi)
            result.entity2nodePbi.father->Compress();

        // ============================================================
        // Step 5: Assign global B indices
        // ============================================================
        result.entity2node.TransAttach();
        result.entity2node.trans.createFatherGlobalMapping();
        auto faceGlobalMapping = result.entity2node.trans.pLGlobalMapping;
        index myFaceOffset = (*faceGlobalMapping)(mpi.rank, 0);

        // ============================================================
        // Step 6: Build parent2entity (A→B) with global B indices
        // ============================================================
        // For local A parents: remap owned entities via oldToOwned → global B ID.
        // For non-owned entities: need to receive global B ID from owner.
        //
        // parent2entityPbi does NOT need to be pushed: Step 2b already computed
        // localParent2EntityPbi[iParent][j] for ALL parents (father+son),
        // because the XOR is a local property of (cell, sub-entity slot) that
        // only depends on cell2nodePbi and entity2nodePbi — both available locally.

        // 6a: Push global B IDs to peer ranks.
        // For each owned B with ghost A-parents, push (globalA, globalB, subIdx)
        // so the peer can fill its parent2entity at the correct slot.
        std::vector<std::vector<std::array<index, 3>>> pushMessages(mpi.size);
        for (index iEnt = 0; iEnt < nAllEntities; iEnt++)
        {
            if (!classifications[iEnt].owned)
                continue;
            index iNew = oldToOwned[iEnt];
            index globalB = myFaceOffset + iNew;

            auto parentRow = localResult.entity2parent.father->operator[](iEnt);
            for (auto pLocal : parentRow)
            {
                if (pLocal >= nLocalParents) // ghost parent
                {
                    MPI_int pRank = getParentRank(pLocal);
                    index pGlobal = parentGhostMapping(-1, pLocal);
                    int subIdx = -1;
                    for (rowsize s = 0; s < localResult.parent2entity.father->RowSize(pLocal); s++)
                    {
                        if (localResult.parent2entity.father->operator()(pLocal, s) == iEnt)
                        {
                            subIdx = s;
                            break;
                        }
                    }
                    DNDS_assert(subIdx >= 0);
                    pushMessages[pRank].push_back({pGlobal, globalB, static_cast<index>(subIdx)});
                }
            }
        }

        // 6b: AllToAllv the (globalA, globalB, subIdx) triplets.
        std::vector<int> sendCounts(mpi.size), recvCounts(mpi.size);
        for (MPI_int r = 0; r < mpi.size; r++)
            sendCounts[r] = static_cast<int>(pushMessages[r].size() * 3);
        MPI::Alltoall(sendCounts.data(), 1, MPI_INT,
                      recvCounts.data(), 1, MPI_INT, mpi.comm);

        std::vector<int> sendDisp(mpi.size + 1, 0), recvDisp(mpi.size + 1, 0);
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            sendDisp[r + 1] = sendDisp[r] + sendCounts[r];
            recvDisp[r + 1] = recvDisp[r] + recvCounts[r];
        }

        std::vector<index> sendBuf(sendDisp.back());
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            index offset = sendDisp[r];
            for (auto &msg : pushMessages[r])
            {
                sendBuf[offset++] = msg[0];
                sendBuf[offset++] = msg[1];
                sendBuf[offset++] = msg[2];
            }
        }
        std::vector<index> recvBuf(recvDisp.back());
        MPI::Alltoallv(sendBuf.data(), sendCounts.data(), sendDisp.data(), DNDS_MPI_INDEX,
                       recvBuf.data(), recvCounts.data(), recvDisp.data(), DNDS_MPI_INDEX,
                       mpi.comm);

        // 6c: Build a map: (globalA, subIdx) → globalB for received triplets.
        std::unordered_map<index, std::unordered_map<int, index>> receivedA2SubB;
        for (index i = 0; i < static_cast<index>(recvBuf.size()); i += 3)
        {
            index gA = recvBuf[i];
            index gB = recvBuf[i + 1];
            int subIdx = static_cast<int>(recvBuf[i + 2]);
            receivedA2SubB[gA][subIdx] = gB;
        }

        // 6d: Build parent2entity and parent2entityPbi for all parents (father + son).
        result.parent2entity.InitPair("InterpGlobal_p2e", mpi);
        result.parent2entity.father->Resize(nLocalParents);
        result.parent2entity.son = std::make_shared<tAdj::element_type>(
            ObjName{"InterpGlobal_p2e.son"}, mpi);
        result.parent2entity.son->Resize(nTotalParents - nLocalParents);

        if (hasPbi)
        {
            result.parent2entityPbi.InitPair("InterpGlobal_p2ePbi", mpi);
            result.parent2entityPbi.father->Resize(nLocalParents);
            result.parent2entityPbi.son = std::make_shared<typename decltype(result.parent2entityPbi.son)::element_type>(
                ObjName{"InterpGlobal_p2ePbi.son"}, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            result.parent2entityPbi.son->Resize(nTotalParents - nLocalParents);
        }

        for (index iParent = 0; iParent < nTotalParents; iParent++)
        {
            rowsize nSubs = localResult.parent2entity.father->RowSize(iParent);
            result.parent2entity.ResizeRow(iParent, nSubs);
            if (hasPbi)
                result.parent2entityPbi.ResizeRow(iParent, nSubs);

            for (rowsize j = 0; j < nSubs; j++)
            {
                // parent2entityPbi is always from localParent2EntityPbi (computed
                // in Step 2b for ALL parents, father+son).
                if (hasPbi)
                    result.parent2entityPbi(iParent, j) =
                        localParent2EntityPbi[iParent][j];

                index oldEntIdx = localResult.parent2entity.father->operator()(iParent, j);
                index newOwnedIdx = oldToOwned[oldEntIdx];
                if (newOwnedIdx != UnInitIndex)
                {
                    // Owned B → global B ID
                    result.parent2entity(iParent, j) = myFaceOffset + newOwnedIdx;
                }
                else
                {
                    // Non-owned B → look up from received messages by (globalA, subIdx).
                    index parentGlobal = parentGhostMapping(-1, iParent);
                    auto it = receivedA2SubB.find(parentGlobal);
                    if (it != receivedA2SubB.end())
                    {
                        auto jt = it->second.find(static_cast<int>(j));
                        if (jt != it->second.end())
                        {
                            result.parent2entity(iParent, j) = jt->second;
                            continue;
                        }
                    }
                    // No received B for this (parent, slot) — fully ghost B entity.
                    // Will be resolved after ghost B pull via cell2face ghost comm.
                    result.parent2entity(iParent, j) = UnInitIndex;
                }
            }
        }

        return result;
    }

} // namespace DNDS::Geom

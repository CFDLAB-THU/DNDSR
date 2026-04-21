#include "MeshConnectivity.hpp"

#include <vector>
#include <algorithm>
#include <fmt/core.h>

namespace DNDS::Geom
{
    InterpolateResult MeshConnectivity::Interpolate(
        const tAdjPair &parent2node,
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
        std::vector<std::pair<index, index>> ent2parentVec;

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
                    ent2parentVec.emplace_back(iParent, UnInitIndex);
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
                    // Existing sub-entity: record second parent
                    if (ent2parentVec[iFound].second != UnInitIndex)
                    {
                        // Third parent detected — deduplication is wrong
                        // (typically missing matchExtra on periodic meshes).
                        // Set overflow flag and skip this match.
                        result.duplicateOverflow = true;
                        // Treat as new entity to avoid corrupting existing data
                        std::vector<index> subNodesVec(subNodesBuf.begin(),
                                                       subNodesBuf.begin() + desc.nNodes);
                        ent2nodeVec.emplace_back(std::move(subNodesVec));
                        ent2parentVec.emplace_back(iParent, UnInitIndex);
                        entityElemInfoV.emplace_back(ElemInfo{desc.typeTag, 0});
                        ent2origin.push_back({iParent, iSub});
                        for (auto iV : subVerts)
                            node2entity[iV].push_back(nEntities);
                        result.parent2entity.father->operator()(iParent, iSub) = nEntities;
                        nEntities++;
                    }
                    else
                    {
                        ent2parentVec[iFound].second = iParent;
                        result.parent2entity.father->operator()(iParent, iSub) = iFound;
                    }
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

        // entity2parent (fixed-2)
        result.entity2parent.InitPair("Interpolate_e2p", mpi);
        result.entity2parent.father->Resize(nEntities);
        for (index iEnt = 0; iEnt < nEntities; iEnt++)
        {
            result.entity2parent.father->operator()(iEnt, 0) = ent2parentVec[iEnt].first;
            result.entity2parent.father->operator()(iEnt, 1) = ent2parentVec[iEnt].second;
        }

        return result;
    }

} // namespace DNDS::Geom

#pragma once

#include "DNDS/Core/OMP.hpp"

#include <map>
#include <queue>

// Included after MeshConnectivity query types are declared. Enumeration uses
// callbacks only; both compatibility arrays and native readers share this math.
namespace DNDS::Geom::ConnectivityRows
{
    enum class SubEntityEnumerationMode
    {
        Auto,
        Append,
        RecordPrefix
    };

    struct SubEntityRows
    {
        std::vector<std::vector<index>> parent2entity, entity2node, entity2parent;
        std::vector<ElemInfo> entityElemInfo;
        index nEntities = 0;
    };

    namespace detail
    {
        constexpr int MaxSubEntityNodes = 10;

        struct SubEntityCandidate
        {
            SubEntityDesc desc{};
            index parent = -1;
            int sub = -1;
            std::array<index, MaxSubEntityNodes> nodes{};
            std::array<index, MaxSubEntityNodes> vertices{};
        };

        inline void ValidateCandidate(const SubEntityCandidate &candidate,
                                      index nNode)
        {
            DNDS_assert_info(candidate.desc.nVertices > 0 &&
                                 candidate.desc.nVertices <= candidate.desc.nNodes &&
                                 candidate.desc.nNodes <= MaxSubEntityNodes,
                             fmt::format("Interpolate: sub-entity has {} nodes, max is {}",
                                         candidate.desc.nNodes, MaxSubEntityNodes));
            for (int vertex = 0; vertex < candidate.desc.nVertices; ++vertex)
                DNDS_assert_info(candidate.vertices[size_t(vertex)] >= 0 &&
                                     candidate.vertices[size_t(vertex)] < nNode,
                                 fmt::format("Interpolate: node index {} out of range [0, {})",
                                             candidate.vertices[size_t(vertex)], nNode));
        }

        template <class ReadNode>
        SubEntityCandidate BuildCandidate(ReadNode &readNode,
                                          const SubEntityQuery &query,
                                          index parent, int sub, index nNode)
        {
            SubEntityCandidate candidate;
            candidate.parent = parent;
            candidate.sub = sub;
            candidate.desc = query.describe(parent, sub);
            DNDS_assert_info(candidate.desc.nVertices > 0 &&
                                 candidate.desc.nVertices <= candidate.desc.nNodes &&
                                 candidate.desc.nNodes <= MaxSubEntityNodes,
                             fmt::format("Interpolate: sub-entity has {} nodes, max is {}",
                                         candidate.desc.nNodes, MaxSubEntityNodes));
            const auto parentNode = [&](int slot) -> index
            { return readNode(parent, slot); };
            query.extractNodes(parent, sub, parentNode, candidate.nodes.data());
            std::copy_n(candidate.nodes.begin(), candidate.desc.nVertices,
                        candidate.vertices.begin());
            std::sort(candidate.vertices.begin(),
                      candidate.vertices.begin() + candidate.desc.nVertices);
            ValidateCandidate(candidate, nNode);
            return candidate;
        }

        inline void ConsumeCandidate(
            const SubEntityCandidate &candidate, const SubEntityQuery &query,
            SubEntityRows &result,
            std::vector<std::vector<index>> &node2entity,
            std::vector<std::pair<index, int>> &origins,
            std::vector<std::array<index, MaxSubEntityNodes>> &entityVertices,
            std::vector<int> &entityVertexCounts)
        {
            index found = -1;
            const auto vertexEnd = candidate.vertices.begin() + candidate.desc.nVertices;
            for (auto entity : node2entity[size_t(candidate.vertices[0])])
            {
                if (result.entityElemInfo[size_t(entity)].type != candidate.desc.typeTag)
                    continue;
                DNDS_assert(result.entity2node[size_t(entity)].size() >=
                            size_t(candidate.desc.nVertices));
                const index *existingVertices = entityVertices[size_t(entity)].data();
                std::array<index, MaxSubEntityNodes> scratch{};
                if (entityVertexCounts[size_t(entity)] != candidate.desc.nVertices)
                {
                    std::copy_n(result.entity2node[size_t(entity)].begin(),
                                candidate.desc.nVertices, scratch.begin());
                    std::sort(scratch.begin(), scratch.begin() + candidate.desc.nVertices);
                    existingVertices = scratch.data();
                }
                if (!std::equal(candidate.vertices.begin(), vertexEnd, existingVertices))
                    continue;
                if (query.matchExtra)
                {
                    const auto [originParent, originSub] = origins[size_t(entity)];
                    if (!query.matchExtra(candidate.parent, candidate.sub, entity,
                                          originParent, originSub))
                        continue;
                }
                found = entity;
                break;
            }

            if (found < 0)
            {
                found = result.nEntities++;
                result.entity2node.emplace_back(candidate.nodes.begin(),
                                                candidate.nodes.begin() + candidate.desc.nNodes);
                result.entity2parent.push_back({candidate.parent});
                result.entityElemInfo.emplace_back(ElemInfo{candidate.desc.typeTag, 0});
                origins.emplace_back(candidate.parent, candidate.sub);
                entityVertices.push_back(candidate.vertices);
                entityVertexCounts.push_back(candidate.desc.nVertices);
                for (auto it = candidate.vertices.begin(); it != vertexEnd; ++it)
                    node2entity[size_t(*it)].push_back(found);
            }
            else
                result.entity2parent[size_t(found)].push_back(candidate.parent);
            result.parent2entity[size_t(candidate.parent)][size_t(candidate.sub)] = found;
        }

        template <class ReadNode>
        SubEntityRows EnumerateSubEntitiesAppend(ReadNode &readNode,
                                                 const SubEntityQuery &query,
                                                 index nParent, index nNode)
        {
            SubEntityRows result;
            result.parent2entity.resize(size_t(nParent));
            std::vector<std::vector<index>> node2entity(static_cast<size_t>(nNode));
            std::vector<std::pair<index, int>> origins;
            std::vector<std::array<index, MaxSubEntityNodes>> entityVertices;
            std::vector<int> entityVertexCounts;
            for (index parent = 0; parent < nParent; ++parent)
            {
                const int nSubs = query.numSubEntities(parent);
                DNDS_check_throw_info(nSubs >= 0, "negative sub-entity count");
                result.parent2entity[size_t(parent)].resize(size_t(nSubs));
                for (int sub = 0; sub < nSubs; ++sub)
                    ConsumeCandidate(BuildCandidate(readNode, query, parent, sub, nNode),
                                     query, result, node2entity, origins,
                                     entityVertices, entityVertexCounts);
            }
            return result;
        }

        template <class ReadNode>
        SubEntityRows EnumerateSubEntitiesRecords(ReadNode &readNode,
                                                  const SubEntityQuery &query,
                                                  index nParent, index nNode)
        {
            std::vector<index> counts(static_cast<size_t>(nParent));
            OMP::ForSetupRows(0, nParent, [&](index parent)
                              {
                const int count = query.numSubEntities(parent);
                DNDS_check_throw_info(count >= 0, "negative sub-entity count");
                counts[size_t(parent)] = count; });
            std::vector<index> starts(size_t(nParent) + 1);
            const index candidateCount = OMP::ExclusivePrefixRows(
                nParent, [&](index parent)
                { return counts[size_t(parent)]; },
                [&](index parent, index offset)
                { starts[size_t(parent)] = offset; });
            SubEntityRows result;
            result.parent2entity.resize(size_t(nParent));
            OMP::ForSetupRows(0, nParent, [&](index parent)
                              { result.parent2entity[size_t(parent)].resize(size_t(counts[size_t(parent)])); });
            if (!candidateCount)
                return result;

            const int workers = int(std::min<index>(
                std::max(1, OMP::DistributedThreads()),
                std::max<index>(nParent, 1)));
            struct CandidateRef
            {
                index parent;
                int sub;
            };
            std::vector<CandidateRef> references(static_cast<size_t>(candidateCount));
            std::vector<int> owners(static_cast<size_t>(candidateCount));
            std::vector<std::map<t_index, int>> workerArities(static_cast<size_t>(workers));
            const index nodeChunk = std::max<index>((nNode + workers - 1) / workers, 1);
            OMP::ForSetupRowsWithThreads(0, workers, workers, [&](index worker)
                                         {
                const index parentBegin = nParent * worker / workers;
                const index parentEnd = nParent * (worker + 1) / workers;
                auto &arities = workerArities[size_t(worker)];
                for (index parent = parentBegin; parent < parentEnd; ++parent)
                    for (int sub = 0; sub < counts[size_t(parent)]; ++sub)
                    {
                        const index ordinal = starts[size_t(parent)] + sub;
                        const auto candidate = BuildCandidate(readNode, query, parent, sub, nNode);
                        references[size_t(ordinal)] = {parent, sub};
                        owners[size_t(ordinal)] = std::min(
                            workers - 1, int(candidate.vertices[0] / nodeChunk));
                        const auto [found, inserted] = arities.emplace(
                            candidate.desc.typeTag, candidate.desc.nVertices);
                        if (!inserted && found->second != candidate.desc.nVertices)
                            found->second = -1;
                    } });

            // A type tag with inconsistent vertex arity historically uses an
            // asymmetric prefix comparison. Re-enter the exact append route for
            // that compatibility-only case rather than changing its semantics.
            std::map<t_index, int> arities;
            bool irregularType = false;
            for (const auto &worker : workerArities)
                for (const auto &[type, arity] : worker)
                {
                    const auto [found, inserted] = arities.emplace(type, arity);
                    irregularType = irregularType || arity < 0 ||
                                    (!inserted && found->second != arity);
                }
            if (irregularType)
                return EnumerateSubEntitiesAppend(readNode, query, nParent, nNode);

            // Stable worker-to-owner aggregation is the setup equivalent of a
            // coalesced per-destination message layout. All equal vertex keys
            // share their minimum vertex and therefore reach one owner.
            std::vector<index> producerOwnerCounts(static_cast<size_t>(workers * workers), 0);
            OMP::ForSetupRowsWithThreads(0, workers, workers, [&](index producer)
                                         {
                const index begin = candidateCount * producer / workers;
                const index end = candidateCount * (producer + 1) / workers;
                auto *row = producerOwnerCounts.data() + size_t(producer * workers);
                for (index ordinal = begin; ordinal < end; ++ordinal)
                    ++row[owners[size_t(ordinal)]]; });
            std::vector<index> ownerStarts(size_t(workers) + 1, 0);
            std::vector<index> producerOffsets(static_cast<size_t>(workers * workers));
            for (int owner = 0; owner < workers; ++owner)
            {
                index cursor = ownerStarts[size_t(owner)];
                for (int producer = 0; producer < workers; ++producer)
                {
                    producerOffsets[size_t(producer * workers + owner)] = cursor;
                    cursor += producerOwnerCounts[size_t(producer * workers + owner)];
                }
                ownerStarts[size_t(owner + 1)] = cursor;
            }
            DNDS_assert(ownerStarts.back() == candidateCount);
            std::vector<index> routedOrdinals(static_cast<size_t>(candidateCount));
            OMP::ForSetupRowsWithThreads(0, workers, workers, [&](index producer)
                                         {
                const index begin = candidateCount * producer / workers;
                const index end = candidateCount * (producer + 1) / workers;
                std::vector<index> cursors(static_cast<size_t>(workers));
                for (int owner = 0; owner < workers; ++owner)
                    cursors[size_t(owner)] = producerOffsets[size_t(producer * workers + owner)];
                for (index ordinal = begin; ordinal < end; ++ordinal)
                {
                    const int owner = owners[size_t(ordinal)];
                    routedOrdinals[size_t(cursors[size_t(owner)]++)] = ordinal;
                } });

            struct CandidateKey
            {
                SubEntityDesc desc{};
                std::array<index, MaxSubEntityNodes> vertices{};
            };
            struct PendingEntity
            {
                index originOrdinal;
                CandidateKey key;
                std::vector<index> ordinals;
            };
            struct OwnerRecords
            {
                std::vector<PendingEntity> entities;
            };
            const auto sameKey = [](const SubEntityCandidate &candidate,
                                    const CandidateKey &key)
            {
                return candidate.desc.typeTag == key.desc.typeTag &&
                       candidate.desc.nVertices == key.desc.nVertices &&
                       std::equal(candidate.vertices.begin(),
                                  candidate.vertices.begin() + candidate.desc.nVertices,
                                  key.vertices.begin());
            };
            std::vector<OwnerRecords> records(static_cast<size_t>(workers));
            OMP::ForSetupRowsWithThreads(0, workers, workers, [&](index owner)
                                         {
                const index begin = ownerStarts[size_t(owner)];
                const index end = ownerStarts[size_t(owner + 1)];
                auto &entities = records[size_t(owner)].entities;
                entities.reserve(size_t(end - begin));
                const index firstNode = owner * nodeChunk;
                const index ownerNodeCount = std::max<index>(
                    std::min(nodeChunk, nNode - firstNode), 0);
                std::vector<std::vector<index>> byMinimumVertex(
                    static_cast<size_t>(ownerNodeCount));
                for (index routed = begin; routed < end; ++routed)
                {
                    const index ordinal = routedOrdinals[size_t(routed)];
                    const auto ref = references[size_t(ordinal)];
                    const auto candidate = BuildCandidate(
                        readNode, query, ref.parent, ref.sub, nNode);
                    auto &possible = byMinimumVertex[size_t(
                        candidate.vertices[0] - firstNode)];
                    index accepted = -1;
                    for (auto entity : possible)
                    {
                        const auto &existing = entities[size_t(entity)];
                        const auto originRef = references[size_t(existing.originOrdinal)];
                        if (sameKey(candidate, existing.key) &&
                            (!query.matchExtra ||
                             query.matchExtra(ref.parent, ref.sub, -1,
                                              originRef.parent, originRef.sub)))
                        {
                            accepted = entity;
                            break;
                        }
                    }
                    if (accepted < 0)
                    {
                        accepted = index(entities.size());
                        entities.push_back({ordinal,
                                            {candidate.desc, candidate.vertices}, {}});
                        possible.push_back(accepted);
                    }
                    entities[size_t(accepted)].ordinals.push_back(ordinal);
                } });

            struct EntityCursor
            {
                index originOrdinal;
                int owner;
                index local;
            };
            const auto laterOrigin = [](const EntityCursor &left,
                                        const EntityCursor &right)
            { return left.originOrdinal > right.originOrdinal; };
            std::priority_queue<EntityCursor, std::vector<EntityCursor>,
                                decltype(laterOrigin)>
                queue(laterOrigin);
            for (int owner = 0; owner < workers; ++owner)
                if (!records[size_t(owner)].entities.empty())
                    queue.push({records[size_t(owner)].entities.front().originOrdinal,
                                owner, 0});
            std::vector<EntityCursor> entities;
            entities.reserve(size_t(candidateCount));
            while (!queue.empty())
            {
                const auto cursor = queue.top();
                queue.pop();
                entities.push_back(cursor);
                const index next = cursor.local + 1;
                if (next < index(records[size_t(cursor.owner)].entities.size()))
                {
                    queue.push({records[size_t(cursor.owner)].entities[size_t(next)].originOrdinal,
                                cursor.owner, next});
                }
            }

            result.nEntities = index(entities.size());
            result.entity2node.resize(entities.size());
            result.entity2parent.resize(entities.size());
            result.entityElemInfo.resize(entities.size());
            OMP::ForSetupRows(0, result.nEntities, [&](index entity)
                              {
                const auto &record = entities[size_t(entity)];
                const auto &pending = records[size_t(record.owner)]
                                          .entities[size_t(record.local)];
                const auto originRef = references[size_t(pending.originOrdinal)];
                const auto origin = BuildCandidate(
                    readNode, query, originRef.parent, originRef.sub, nNode);
                result.entity2node[size_t(entity)].assign(
                    origin.nodes.begin(), origin.nodes.begin() + origin.desc.nNodes);
                result.entityElemInfo[size_t(entity)] = ElemInfo{origin.desc.typeTag, 0};
                auto &parents = result.entity2parent[size_t(entity)];
                parents.reserve(pending.ordinals.size());
                for (auto ordinal : pending.ordinals)
                {
                    const auto ref = references[size_t(ordinal)];
                    parents.push_back(ref.parent);
                    result.parent2entity[size_t(ref.parent)][size_t(ref.sub)] = entity;
                } });
            return result;
        }
    }

    /** Enumerate deduplicated sub-entities in stable parent/sub-entity order.
     *
     * The record path routes candidates by minimum vertex, resolves disjoint key
     * owners in parallel, and merges their creation streams by first candidate
     * ordinal. Entity numbers, first-parent origins, accepted `matchExtra` groups,
     * and parent adjacency ordering therefore match the append implementation.
     * `Auto` keeps append for one distributed setup worker and uses the scalable
     * count/prefix/fill route otherwise.
     */
    template <class ReadNode>
    SubEntityRows EnumerateSubEntities(ReadNode &&readNode,
                                       const SubEntityQuery &query,
                                       index nParent, index nNode,
                                       SubEntityEnumerationMode mode =
                                           SubEntityEnumerationMode::Auto)
    {
        DNDS_check_throw_info(nParent >= 0 && nNode >= 0,
                              "negative sub-entity enumeration extent");
        auto &&reader = readNode;
        if (mode == SubEntityEnumerationMode::Auto)
            mode = OMP::DistributedThreads() > 1 &&
                           (!query.matchExtra || query.matchExtraSupportsParallelGrouping)
                       ? SubEntityEnumerationMode::RecordPrefix
                       : SubEntityEnumerationMode::Append;
        if (mode == SubEntityEnumerationMode::Append)
            return detail::EnumerateSubEntitiesAppend(reader, query, nParent, nNode);
        return detail::EnumerateSubEntitiesRecords(reader, query, nParent, nNode);
    }
}

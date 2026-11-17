#pragma once

#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Connectivity/NativePeriodicMatch.hpp"

// Shared exact subentity matching for topology publication and reader-only
// partition graph scratch. This header is implementation-only.
namespace DNDS::Geom::detail
{
    inline std::vector<index> CanonicalRoleKey(Elem::Element element, const NativePeriodicNodes &pattern, bool periodic)
    {
        DNDS_check_throw_info(!pattern.empty(), "native topology identity requires node occurrences");
        std::array<bool, 8> candidates{};
        const auto firstNode = std::min_element(pattern.begin(), pattern.end())->first;
        for (const auto &[node, frame] : pattern)
        {
            DNDS_check_throw_info(frame < 8, "invalid native topology identity image");
            if (periodic && node == firstNode)
                candidates[frame] = true;
        }
        if (!periodic)
            candidates[0] = true;
        // Linear subentities have at most four vertices and two-node edges.
        // Build the identical flattened key on the stack; only the returned
        // winning vector allocates. In particular Quad4 must retain cyclic
        // edge roles even though every node is a vertex.
        if ((element.type == Elem::Line2 && pattern.size() == 2) ||
            (element.type == Elem::Tri3 && pattern.size() == 3) ||
            (element.type == Elem::Quad4 && pattern.size() == 4))
        {
            auto linearKey = [&](auto extent)
            {
                constexpr size_t count = decltype(extent)::value;
                constexpr size_t edgeCount = count == 2 ? 0 : count;
                constexpr size_t keySize = 4 + 4 * count + 12 * edgeCount;
                std::array<index, keySize> best{}, key{};
                bool haveBest = false;
                for (uint8_t mask = 0; mask < 8; ++mask)
                {
                    if (!candidates[mask])
                        continue;
                    std::array<std::pair<index, uint8_t>, count> vertices{};
                    for (size_t node = 0; node < count; ++node)
                        vertices[node] = {pattern[node].first, uint8_t(pattern[node].second ^ mask)};
                    std::array<std::array<std::pair<index, uint8_t>, 2>, edgeCount> edges{};
                    for (size_t edge = 0; edge < edgeCount; ++edge)
                    {
                        edges[edge] = {vertices[edge], vertices[(edge + 1) % count]};
                        if (edges[edge][1] < edges[edge][0])
                            std::swap(edges[edge][0], edges[edge][1]);
                    }
                    std::sort(vertices.begin(), vertices.end());
                    if constexpr (edgeCount > 1)
                        std::sort(edges.begin(), edges.end());
                    size_t next = 0;
                    key[next++] = element.type;
                    key[next++] = count;
                    for (const auto &[node, frame] : vertices)
                    {
                        key[next++] = node;
                        key[next++] = frame;
                    }
                    key[next++] = count;
                    for (const auto &[node, frame] : vertices)
                    {
                        key[next++] = node;
                        key[next++] = frame;
                    }
                    key[next++] = edgeCount;
                    for (const auto &edge : edges)
                    {
                        key[next++] = 11;
                        key[next++] = Elem::Line2;
                        key[next++] = 2;
                        for (const auto &[node, frame] : edge)
                        {
                            key[next++] = node;
                            key[next++] = frame;
                        }
                        key[next++] = 2;
                        for (const auto &[node, frame] : edge)
                        {
                            key[next++] = node;
                            key[next++] = frame;
                        }
                    }
                    if (!haveBest || key < best)
                    {
                        best = key;
                        haveBest = true;
                    }
                }
                return std::vector<index>(best.begin(), best.end());
            };
            if (element.type == Elem::Line2)
                return linearKey(std::integral_constant<size_t, 2>{});
            if (element.type == Elem::Tri3)
                return linearKey(std::integral_constant<size_t, 3>{});
            return linearKey(std::integral_constant<size_t, 4>{});
        }
        // The key begins with sorted (node, frame) pairs. Its smallest
        // node must have minimum frame zero in the winning XOR image;
        // every other mask loses before vertex/edge roles are compared.
        // Repeated canonical IDs can admit several masks, all retained.
        std::vector<index> best;
        for (uint8_t mask = 0; mask < 8; ++mask)
        {
            if (!candidates[mask])
                continue;
            auto image = pattern;
            for (auto &[node, frame] : image)
            {
                frame ^= mask;
            }
            std::sort(image.begin(), image.end());
            std::vector<index> key{element.type, index(image.size())};
            for (const auto &[node, frame] : image)
            {
                key.push_back(node);
                key.push_back(frame);
            }
            // Interpolation first compares element vertices, then
            // the complete higher-order nodes. Preserve that role
            // distinction: an endpoint cannot become a midside node.
            const auto count = element.GetNumVertices();
            auto vertices = NativePeriodicNodes(pattern.begin(), pattern.begin() + count);
            for (auto &[node, frame] : vertices)
                frame ^= mask;
            std::sort(vertices.begin(), vertices.end());
            key.push_back(count);
            for (const auto &[node, frame] : vertices)
            {
                key.push_back(node);
                key.push_back(frame);
            }
            // Generated 2-D faces have 1-D subfaces (the Element
            // edge API is for 3-D cells). Preserve which midpoint
            // belongs to each edge, and thus Quad9 center identity.
            std::vector<std::vector<index>> edges;
            if (element.GetDim() == 2)
                for (t_index slot = 0; slot < element.GetNumFaces(); ++slot)
                {
                    const auto sub = element.ObtainFace(slot);
                    NativePeriodicNodes occurrences(size_t(sub.GetNumNodes()));
                    element.ExtractFaceNodes(slot, pattern, occurrences);
                    for (auto &[node, frame] : occurrences)
                        frame ^= mask;
                    auto corners = NativePeriodicNodes(occurrences.begin(), occurrences.begin() + sub.GetNumVertices());
                    std::sort(corners.begin(), corners.end());
                    std::sort(occurrences.begin(), occurrences.end());
                    std::vector<index> edge{sub.type, sub.GetNumVertices()};
                    for (const auto &[node, frame] : corners)
                    {
                        edge.push_back(node);
                        edge.push_back(frame);
                    }
                    edge.push_back(index(occurrences.size()));
                    for (const auto &[node, frame] : occurrences)
                    {
                        edge.push_back(node);
                        edge.push_back(frame);
                    }
                    edges.push_back(std::move(edge));
                }
            std::sort(edges.begin(), edges.end());
            key.push_back(index(edges.size()));
            for (const auto &edge : edges)
            {
                key.push_back(index(edge.size()));
                key.insert(key.end(), edge.begin(), edge.end());
            }
            if (best.empty() || key < best)
                best = std::move(key);
        }
        return best;
    }

    template <class Types, class Cone, class Periodic = RelationPeriodicBits>
    SubEntityQueryPbi Query(Types types, const Cone &cone, bool edges, const Periodic *periodic = nullptr)
    {
        SubEntityQueryPbi query;
        query.numSubEntities = [types, edges](index cell)
        { auto element = Elem::Element{types(size_t(cell), 0).getElemType()}; return edges ? element.GetNumEdges() : element.GetNumFaces(); };
        query.describe = [types, edges](index cell, int slot)
        {
            auto element = Elem::Element{types(size_t(cell), 0).getElemType()};
            auto sub = edges ? element.ObtainEdge(t_index(slot)) : element.ObtainFace(t_index(slot));
            return SubEntityDesc{sub.GetNumVertices(), sub.GetNumNodes(), t_index(sub.type)};
        };
        query.extractNodes = [types, edges](index cell, int slot, const std::function<index(int)> &read, index *output)
        {
            auto element = Elem::Element{types(size_t(cell), 0).getElemType()};
            // Extract only the requested subentity. Materializing the entire
            // parent cone repeats mapping lookups for unrelated nodes at every
            // face, including each identity comparison during enumeration.
            struct Nodes
            {
                const std::function<index(int)> &read;
                index operator[](int node) const { return read(node); }
            } nodes{read};
            if (edges)
                element.ExtractEdgeNodes(t_index(slot), nodes, output);
            else
                element.ExtractFaceNodes(t_index(slot), nodes, output);
        };
        const auto extract = query.extractNodes;
        const auto describe = query.describe;
        query.extractPbi = [types, edges](index cell, int slot, const std::function<NodePeriodicBits(int)> &read, NodePeriodicBits *output)
        {
            auto element = Elem::Element{types(size_t(cell), 0).getElemType()};
            struct Nodes
            {
                const std::function<NodePeriodicBits(int)> &read;
                NodePeriodicBits operator[](int node) const { return read(node); }
            } nodes{read};
            if (edges)
                element.ExtractEdgeNodes(t_index(slot), nodes, output);
            else
                element.ExtractFaceNodes(t_index(slot), nodes, output);
        };
        const auto extractPbi = query.extractPbi;
        query.matchExtra = [&cone, extract, describe, extractPbi, periodic](index parent, int sub, index, index originalParent, int originalSub)
        {
            const auto currentDesc = describe(parent, sub), originalDesc = describe(originalParent, originalSub);
            const bool vertexOnlySimplex = currentDesc.typeTag == originalDesc.typeTag &&
                                           currentDesc.nNodes == currentDesc.nVertices &&
                                           originalDesc.nNodes == originalDesc.nVertices &&
                                           (currentDesc.typeTag == Elem::Line2 || currentDesc.typeTag == Elem::Tri3);
            // Enumeration already matched the complete vertex set. A linear
            // edge/triangle has no extra node roles (all vertex pairs are its
            // edges). Quadrilaterals and every high-order type still require
            // the full role key, including cracked-interface detection.
            if (!periodic && vertexOnlySimplex)
                return true;
            auto fullNodes = [&](index cell, int slot)
            {
                const auto desc = describe(cell, slot);
                std::vector<index> nodes(size_t(desc.nNodes));
                extract(cell, slot, [&](int node)
                        { return cone.ConvertedIndex(cell, size_t(node), Adj_PointToGlobal); }, nodes.data());
                std::vector<NodePeriodicBits> bits(nodes.size());
                if (periodic)
                    extractPbi(cell, slot, [&](int node)
                               { return (*periodic)(cell, size_t(node)); }, bits.data());
                NativePeriodicNodes result;
                for (size_t node = 0; node < size_t(desc.nNodes); ++node)
                    result.emplace_back(nodes[node], uint8_t(bits[node]));
                return result;
            };
            const auto current = fullNodes(parent, sub), original = fullNodes(originalParent, originalSub);
            // Different image patterns may share canonical vertices. They
            // are distinct entities, not cracked higher-order geometry.
            if (!NativePeriodicFrame(NativePeriodicNodes(current.begin(), current.begin() + currentDesc.nVertices),
                                     NativePeriodicNodes(original.begin(), original.begin() + originalDesc.nVertices)))
                return false;
            if (vertexOnlySimplex)
                return true;
            DNDS_check_throw_info(CanonicalRoleKey(Elem::Element{static_cast<Elem::ElemType>(currentDesc.typeTag)}, current, periodic != nullptr) ==
                                      CanonicalRoleKey(Elem::Element{static_cast<Elem::ElemType>(originalDesc.typeTag)}, original, periodic != nullptr),
                                  "shared native subentity vertices have inconsistent higher-order nodes or edge roles");
            return true;
        };
        query.matchExtraSupportsParallelGrouping = true;
        return query;
    }

}

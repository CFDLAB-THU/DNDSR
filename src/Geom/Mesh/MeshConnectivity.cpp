#include "MeshConnectivity.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <fmt/core.h>

namespace DNDS::Geom
{
    // -----------------------------------------------------------------
    // adjKindName
    // -----------------------------------------------------------------

    std::string adjKindName(const AdjKind &kind)
    {
        if (kind.isDirect())
            return fmt::format("{}2{}", entityKindName(kind.from), entityKindName(kind.to));
        else
            return fmt::format("{}2{}({})", entityKindName(kind.from),
                               entityKindName(kind.to), entityKindName(kind.via));
    }

    // -----------------------------------------------------------------
    // Adjacency registry
    // -----------------------------------------------------------------

    void MeshConnectivity::registerAdj(AdjKind kind, ssp<AdjVariant> adjPtr)
    {
        adjRegistry[kind] = std::move(adjPtr);
    }

    void MeshConnectivity::registerGlobalMapping(EntityKind kind, const ssp<GlobalOffsetsMapping> &gm)
    {
        globalMappings[kind] = gm;
    }

    ssp<AdjVariant> MeshConnectivity::resolveAdj(AdjKind kind) const
    {
        auto it = adjRegistry.find(kind);
        return (it != adjRegistry.end()) ? it->second : nullptr;
    }

    const ssp<GlobalOffsetsMapping> &MeshConnectivity::getGlobalMapping(EntityKind kind) const
    {
        auto it = globalMappings.find(kind);
        if (it != globalMappings.end())
            return it->second;
        static const ssp<GlobalOffsetsMapping> null_mapping{};
        return null_mapping;
    }

    bool MeshConnectivity::hasAdj(AdjKind kind) const
    {
        return adjRegistry.find(kind) != adjRegistry.end();
    }

    // -----------------------------------------------------------------
    // Cone management
    // -----------------------------------------------------------------

    ConeAdj &MeshConnectivity::addCone(int fromDepth, int toDepth)
    {
        DNDS_assert_info(!hasCone(fromDepth, toDepth),
                         fmt::format("Cone ({}, {}) already exists", fromDepth, toDepth));
        cones.push_back(ConeAdj{fromDepth, toDepth, makeAdjVariant<tAdjPair>(), {}});
        return cones.back();
    }

    ConeAdj *MeshConnectivity::findCone(int fromDepth, int toDepth)
    {
        for (auto &c : cones)
            if (c.fromDepth == fromDepth && c.toDepth == toDepth)
                return &c;
        return nullptr;
    }

    const ConeAdj *MeshConnectivity::findCone(int fromDepth, int toDepth) const
    {
        for (auto &c : cones)
            if (c.fromDepth == fromDepth && c.toDepth == toDepth)
                return &c;
        return nullptr;
    }

    bool MeshConnectivity::hasCone(int fromDepth, int toDepth) const
    {
        return findCone(fromDepth, toDepth) != nullptr;
    }

    // -----------------------------------------------------------------
    // Support management
    // -----------------------------------------------------------------

    SupportAdj &MeshConnectivity::addSupport(int fromDepth, int toDepth)
    {
        DNDS_assert_info(!hasSupport(fromDepth, toDepth),
                         fmt::format("Support ({}, {}) already exists", fromDepth, toDepth));
        supports.push_back(SupportAdj{fromDepth, toDepth, makeAdjVariant<tAdjPair>()});
        return supports.back();
    }

    SupportAdj *MeshConnectivity::findSupport(int fromDepth, int toDepth)
    {
        for (auto &s : supports)
            if (s.fromDepth == fromDepth && s.toDepth == toDepth)
                return &s;
        return nullptr;
    }

    const SupportAdj *MeshConnectivity::findSupport(int fromDepth, int toDepth) const
    {
        for (auto &s : supports)
            if (s.fromDepth == fromDepth && s.toDepth == toDepth)
                return &s;
        return nullptr;
    }

    bool MeshConnectivity::hasSupport(int fromDepth, int toDepth) const
    {
        return findSupport(fromDepth, toDepth) != nullptr;
    }

} // namespace DNDS::Geom

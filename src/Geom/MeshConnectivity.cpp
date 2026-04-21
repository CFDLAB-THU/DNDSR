#include "MeshConnectivity.hpp"

#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <fmt/core.h>

namespace DNDS::Geom
{
    // -----------------------------------------------------------------
    // Cone management
    // -----------------------------------------------------------------

    ConeAdj &MeshConnectivity::addCone(int fromDepth, int toDepth)
    {
        DNDS_assert_info(!hasCone(fromDepth, toDepth),
                         fmt::format("Cone ({}, {}) already exists", fromDepth, toDepth));
        cones.push_back(ConeAdj{fromDepth, toDepth, tAdjPair{}, {}}); // default to variable-width
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
        supports.push_back(SupportAdj{fromDepth, toDepth, tAdjPair{}});
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

    // -----------------------------------------------------------------
    // Inverse
    // -----------------------------------------------------------------

    tAdjPair MeshConnectivity::Inverse(
        const tAdjPair &cone,
        index nToLocal,
        const MPIInfo &mpi,
        const std::function<index(index)> &fromLocal2Global,
        const std::function<index(index)> &toLocal2Global,
        const ssp<GlobalOffsetsMapping> &toGlobalMapping)
    {
        // ----- Step 1: Local inversion -----
        // For each "from" entity in cone, record which "to" entities it touches.
        // Build: toGlobal -> set of fromGlobal
        std::unordered_map<index, std::unordered_set<index>> to2fromRecord;
        std::vector<index> ghostToIndices;    // to-indices that are off-rank
        std::unordered_set<index> ghostToSet;

        index nFromLocal = cone.father->Size();
        for (index iFrom = 0; iFrom < nFromLocal; iFrom++)
        {
            index fromGlobal = fromLocal2Global(iFrom);
            for (auto iTo : cone.father->operator[](iFrom))
            {
                to2fromRecord[iTo].insert(fromGlobal);

                // Check if this to-entity is off-rank
                auto [ret, rank, val] = toGlobalMapping->search(iTo);
                DNDS_assert_info(ret, fmt::format("Inverse: to-entity {} not found in global mapping", iTo));
                if (rank != mpi.rank && ghostToSet.find(iTo) == ghostToSet.end())
                {
                    ghostToIndices.push_back(iTo);
                    ghostToSet.insert(iTo);
                }
            }
        }

        // ----- Step 2: Build initial support pair (father = local to-entities, son = ghost to-entities) -----
        tAdjPair support;
        support.InitPair("Inverse_support", mpi);
        support.father->Resize(nToLocal);

        // Fill father with local records
        for (index iTo = 0; iTo < nToLocal; iTo++)
        {
            index toGlobal = toLocal2Global(iTo);
            auto it = to2fromRecord.find(toGlobal);
            if (it != to2fromRecord.end())
            {
                support.father->ResizeRow(iTo, it->second.size());
                rowsize j = 0;
                for (auto fromG : it->second)
                    support.father->operator()(iTo, j++) = fromG;
            }
        }

        // Set up ghost communication: we have local partial data about off-rank to-entities.
        // We need to push our partial knowledge to the ranks that own those to-entities,
        // then merge all contributions.
        support.TransAttach();
        support.trans.createFatherGlobalMapping();

        // Ghost mapping for the off-rank to-entities
        support.trans.createGhostMapping(ghostToIndices);

        // Fill son with our local partial data for off-rank to-entities
        support.son->Resize(support.trans.pLGhostMapping->ghostIndex.size());
        for (auto &[toGlobal, fromSet] : to2fromRecord)
        {
            MPI_int rank{-1};
            index val{-1};
            if (!support.trans.pLGhostMapping->search(toGlobal, rank, val))
                DNDS_assert_info(false, "Inverse: ghost search failed");
            if (rank >= 0)
            {
                // This is a ghost entry — fill son
                support.son->ResizeRow(val, fromSet.size());
                rowsize j = 0;
                for (auto fromG : fromSet)
                    support.son->operator()(val, j++) = fromG;
            }
        }

        // ----- Step 3: Reverse-push to collect remote contributions -----
        // The "son" currently holds OUR local partial data for entities owned by OTHER ranks.
        // We need to push this to those owning ranks, who will merge it with their own data.

        using tSupportArr = decltype(support.son)::element_type;
        ssp<tSupportArr> supportPast = make_ssp<tSupportArr>(ObjName{"Inverse_supportPast"}, mpi);

        DNDS::ArrayTransformerType<tSupportArr>::Type supportPastTrans;
        supportPastTrans.setFatherSon(support.son, supportPast);
        supportPastTrans.createFatherGlobalMapping();

        // Create ghost mapping for the reverse push
        std::vector<index> pushSonSeries(support.son->Size());
        for (index i = 0; i < support.son->Size(); i++)
            pushSonSeries[i] = i;
        supportPastTrans.createGhostMapping(pushSonSeries, support.trans.pLGhostMapping->ghostStart);
        supportPastTrans.createMPITypes();
        supportPastTrans.pullOnce();

        // ----- Step 4: Merge remote contributions into local record -----
        DNDS_assert(DNDS::size_to_index(support.trans.pLGhostMapping->ghostIndex.size()) == support.son->Size());
        DNDS_assert(DNDS::size_to_index(support.trans.pLGhostMapping->pushingIndexGlobal.size()) == supportPast->Size());

        for (index i = 0; i < supportPast->Size(); i++)
        {
            index toGlobal = support.trans.pLGhostMapping->pushingIndexGlobal[i];
            for (auto fromG : (*supportPast)[i])
                to2fromRecord[toGlobal].insert(fromG);
        }

        // ----- Step 5: Rebuild father with complete merged data -----
        tAdjPair result;
        result.InitPair("Inverse_result", mpi);
        result.father->Resize(nToLocal);

        for (index iTo = 0; iTo < nToLocal; iTo++)
        {
            index toGlobal = toLocal2Global(iTo);
            auto it = to2fromRecord.find(toGlobal);
            if (it != to2fromRecord.end())
            {
                // Sort for deterministic output
                std::vector<index> fromVec(it->second.begin(), it->second.end());
                std::sort(fromVec.begin(), fromVec.end());
                result.father->ResizeRow(iTo, fromVec.size());
                for (rowsize j = 0; j < static_cast<rowsize>(fromVec.size()); j++)
                    result.father->operator()(iTo, j) = fromVec[j];
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // Compose
    // -----------------------------------------------------------------

    tAdjPair MeshConnectivity::Compose(
        const tAdjPair &AB,
        const tAdjPair &BC,
        index nALocal,
        const std::unordered_map<index, index> &bGlobal2Local,
        const std::function<index(index)> &aLocal2Global,
        bool removeSelf)
    {
        return ComposeFiltered(
            AB, BC, nALocal, bGlobal2Local, aLocal2Global,
            SharedCountPredicate{.minShared = 1, .removeSelf = removeSelf});
    }

} // namespace DNDS::Geom

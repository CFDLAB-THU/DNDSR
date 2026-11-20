#pragma once

#include "Geom/Access/PublishedAdjacency.hpp"
#include "Geom/Transition/MeshReorderTypes.hpp"
#include <map>

namespace DNDS::Geom
{
    /// First-touch ordering of owned targets while traversing owned anchors.
    /// Anchor kind is unrestricted; adjacency descriptors specify the exact
    /// input relations. No entity ownership, local element slots or mesh values
    /// change during planning. Untouched owned targets append in old order.
    /// Apply via the ordinary mesh transition to remap every reference/companion.
    class MeshLocalityOrdering
    {
        struct Data
        {
            ReorderSourceOrder sourceOrder;
            std::map<EntityKind, std::vector<index>> permutations;
        };
        std::shared_ptr<const Data> _data;
        explicit MeshLocalityOrdering(std::shared_ptr<const Data> data) : _data(std::move(data)) {}

    public:
        static MeshLocalityOrdering Build(UnstructuredMesh &mesh, EntityKind anchor,
                                          const std::vector<AdjacencyDescriptor> &adjacencies)
        {
            auto lease = mesh.versions.Read();
            auto data = std::make_shared<Data>();
            data->sourceOrder.generation = mesh.versions.Stamp(Runtime::Change::Topology | Runtime::Change::Distribution |
                                                               Runtime::Change::Layout | Runtime::Change::Interface);
            const auto anchors = mesh.Entities(anchor);
            data->sourceOrder.families.push_back(anchors);
            for (const auto &descriptor : adjacencies)
            {
                auto adjacency = ReadPublishedAdjacency(mesh, descriptor);
                DNDS_check_throw_info(adjacency.Source().SameOrder(anchors) && descriptor.to != anchor,
                                      "locality adjacency must map the chosen anchors to a distinct target kind");
                const auto target = adjacency.Target();
                auto [entry, inserted] = data->permutations.emplace(descriptor.to, std::vector<index>(size_t(target.NumOwned()), -1));
                DNDS_check_throw_info(inserted, "locality ordering specifies a target kind more than once");
                data->sourceOrder.families.push_back(target);
                auto &oldToNew = entry->second;
                index next = 0;
                for (index iAnchor = 0; iAnchor < anchors.NumOwned(); ++iAnchor)
                    for (index slot = 0; slot < adjacency.RowSize(iAnchor); ++slot)
                    {
                        const auto incidence = adjacency.ReadEntry(iAnchor, size_t(slot));
                        if (incidence.absent)
                            continue;
                        DNDS_check_throw_info(!incidence.unavailable, "locality ordering requires complete target ghost mapping");
                        const auto local = target.GlobalToLocal(incidence.global);
                        DNDS_check_throw_info(local, "locality target is absent from its mapping");
                        if (*local < target.NumOwned() && oldToNew[*local] < 0)
                            oldToNew[*local] = next++;
                    }
                for (index iTarget = 0; iTarget < target.NumOwned(); ++iTarget)
                    if (oldToNew[iTarget] < 0)
                        oldToNew[iTarget] = next++;
            }
            return MeshLocalityOrdering(std::move(data));
        }
        void Validate() const
        {
            DNDS_check_throw_info(_data, "locality ordering was moved");
            _data->sourceOrder.generation.Validate();
        }
        /// Borrow first-touch permutations from this plan while its row ordering is current.
        const std::map<EntityKind, std::vector<index>> &Permutations() const
        {
            Validate();
            return _data->permutations;
        }
        ReorderInput Input() const
        {
            Validate();
            ReorderInput input;
            input.sourceOrder = _data->sourceOrder;
            input.localPermutations = _data->permutations;
            return input;
        }
    };
}

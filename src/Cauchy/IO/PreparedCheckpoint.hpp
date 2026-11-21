#pragma once

#include "Cauchy/Core/DomainStateSnapshot.hpp"
#include "Cauchy/IO/CheckpointMetadata.hpp"
#include "DNDS/Runtime/Communication/CommunicationContext.hpp"

namespace DNDS::Cauchy
{
    class OutputStaging;
    struct CheckpointPreparation
    {
        bool includeReconstruction = false;
        /// Background writing requires a detached copy. Synchronous CPU output
        /// may retain a read-only source snapshot until Write returns.
        bool detached = true;
    };

    /** @brief Frozen accepted-step or internal-stage archive input, with no live Domain borrow.
     * @details Owns matching field/history/control values, persistent cell IDs,
     * mesh envelope, physics/frame metadata and a shared I/O communicator.
     * Write preserves the existing v5 domain HDF schema and performs no point
     * physics, reconstruction or source lookup. The controller stages GPU values into pinned host slots on the domain compute stream. A detached
     * image can outlive subsequent solver updates. Background Write requires
     * MPI_THREAD_MULTIPLE, an independent ordering communicator, serialized
     * HDF5 access and a terminal failure handler supplied by the output service.
     */
    class PreparedCheckpoint
    {
        friend class Domain;
        struct InternalFields
        {
            size_t ownedEquations = 0;
            std::vector<std::pair<std::string, EquationDofs>> equations;
            std::vector<std::pair<std::string, Runtime::Array<real>>> arrays;
        };
        struct Data;
        std::shared_ptr<Data> _data;
        PreparedCheckpoint(DomainStateSnapshot, Json metadata, Runtime::Array<index> ids,
                           CheckpointMetadata::MeshEnvelope,
                           std::shared_ptr<Runtime::CommunicationContext>, bool detached, std::optional<InternalFields> internal, Runtime::ExecutionContext compute, OutputStaging &staging);

    public:
        bool Detached() const;
        void Write(const std::string &path, const Json &resolved, const Json &outputController = Json()) const;
    };
}

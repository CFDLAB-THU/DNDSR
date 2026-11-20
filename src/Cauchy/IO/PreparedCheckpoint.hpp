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
     * physics, reconstruction or source lookup. A detached GPU snapshot stages its values on its retained copy stream before opening HDF5. A detached
     * image can outlive subsequent solver updates. Background Write requires
     * MPI_THREAD_MULTIPLE, an independent ordering communicator, serialized
     * HDF5 access and a terminal failure handler supplied by the output service.
     */
    class PreparedCheckpoint
    {
        friend class Domain;
        friend class AsyncOutput;
        void BindWriter(std::shared_ptr<OutputStaging>);
        struct InternalFields
        {
            std::vector<std::pair<std::string, EquationDofs>> equations;
            std::vector<std::pair<std::string, Runtime::Array<real>>> arrays;
        };
        struct Data;
        std::shared_ptr<Data> _data;
        PreparedCheckpoint(DomainStateSnapshot, Json metadata, Runtime::Array<index> ids,
                           CheckpointMetadata::MeshEnvelope,
                           std::shared_ptr<Runtime::CommunicationContext>, bool detached, std::optional<InternalFields> internal, Runtime::ExecutionContext writerCopy);

    public:
        bool Detached() const;
        void Write(const std::string &path, const Json &resolved, const Json &outputController = Json()) const;
    };
}

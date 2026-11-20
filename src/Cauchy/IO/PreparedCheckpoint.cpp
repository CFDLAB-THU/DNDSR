#include "Cauchy/IO/PreparedCheckpoint.hpp"
#include "Cauchy/IO/StateArchive.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"

namespace DNDS::Cauchy
{
    struct PreparedCheckpoint::Data
    {
        DomainStateSnapshot snapshot;
        Json metadata;
        Runtime::Array<index> ids;
        CheckpointMetadata::MeshEnvelope mesh;
        std::shared_ptr<Runtime::CommunicationContext> communication;
        bool detached;
        Runtime::ExecutionContext writerCopy;
        std::optional<InternalFields> internal;
        Data(DomainStateSnapshot state, Json description, Runtime::Array<index> cellIds,
             CheckpointMetadata::MeshEnvelope envelope,
             std::shared_ptr<Runtime::CommunicationContext> context, bool copied, std::optional<InternalFields> active, Runtime::ExecutionContext copy)
            : snapshot(std::move(state)), metadata(std::move(description)), ids(std::move(cellIds)),
              mesh(std::move(envelope)), communication(std::move(context)), detached(copied), writerCopy(std::move(copy)), internal(std::move(active)) {}
    };
    PreparedCheckpoint::PreparedCheckpoint(DomainStateSnapshot snapshot, Json metadata, Runtime::Array<index> ids,
                                           CheckpointMetadata::MeshEnvelope mesh, std::shared_ptr<Runtime::CommunicationContext> communication, bool detached, std::optional<InternalFields> internal, Runtime::ExecutionContext writerCopy)
        : _data(std::make_shared<Data>(std::move(snapshot), std::move(metadata), std::move(ids),
                                       std::move(mesh), std::move(communication), detached, std::move(internal), std::move(writerCopy)))
    {
        DNDS_check_throw_info(_data->communication && _data->snapshot.Means().Device() == _data->writerCopy.device &&
                                  (_data->writerCopy.device == Runtime::Device::Host() || (_data->detached && _data->writerCopy.stream)),
                              "prepared checkpoint requires an I/O communicator and host fields or a detached device snapshot with a copy stream");
    }
    bool PreparedCheckpoint::Detached() const { return _data->detached; }
    void PreparedCheckpoint::Write(const std::string &path, const Json &resolved, const Json &outputController) const
    {
        const auto &data = *_data;
        std::optional<DomainStateSnapshot> staged;
        std::optional<InternalFields> stagedInternal;
        if (data.writerCopy.device != Runtime::Device::Host())
        {
            staged.emplace(data.snapshot.DetachedHost(data.writerCopy));
            if (data.internal)
            {
                stagedInternal.emplace();
                for (const auto &[key, field] : data.internal->equations)
                    stagedInternal->equations.emplace_back(key, EquationDofOperations::CopyOwnedTo(field, Runtime::Device::Host(), data.writerCopy, {true}));
                for (const auto &[key, field] : data.internal->arrays)
                {
                    const auto layout = field.Layout();
                    auto host = Runtime::Array<real>::FromStorage(layout, Runtime::Buffer::AllocatePinnedHost(CheckedSize::Multiply(layout.Elements(), sizeof(real))));
                    host.SetName(field.Name(), DNDS_ALLOCATION_SITE());
                    host.CopyFrom(field, data.writerCopy).Wait();
                    stagedInternal->arrays.emplace_back(key, std::move(host));
                }
            }
        }
        const auto &snapshot = staged ? *staged : data.snapshot;
        const auto &internal = staged ? stagedInternal : data.internal;
        const MPIInfo mpi(data.communication->Communicator());
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, false);
        serializer->WriteInt("cauchy_format", 5);
        if (internal)
            serializer->WriteInt("cauchy_internal_format", 1);
        serializer->WriteString("resolved_config", resolved.dump());
        serializer->WriteString("metadata_json", data.metadata.dump());
        if (!outputController.is_null())
            serializer->WriteString("output_controller_json", outputController.dump());
        serializer->WriteString("domain_name", snapshot.Name());
        serializer->WriteReal("accepted_time", snapshot.Time());
        serializer->WriteReal("last_accepted_dt", snapshot.LastAcceptedDt());
        serializer->WriteIndex("accepted_step", index(snapshot.Epoch().acceptedStep));
        Runtime::WriteArray(data.ids, serializer, "cell_ids", mpi);
        Runtime::WriteArray(data.mesh.topology, serializer, "mesh_topology", mpi);
        Runtime::WriteArray(data.mesh.coordinates, serializer, "mesh_coordinates", mpi);
        Runtime::WriteArray(data.mesh.periodicTransforms, serializer, "periodic_transforms", mpi);
        WriteEquationArrays(snapshot.Means(), serializer, "state_field_", mpi);
        if (snapshot.Reconstruction())
            WriteEquationArrays(*snapshot.Reconstruction(), serializer, "reconstruction_field_", mpi);
        if (snapshot.LimitedVRHistory())
            WriteStateArray(*snapshot.LimitedVRHistory(), serializer, "limited_vr_history", mpi, Runtime::ExecutionContext::Host());
        if (snapshot.PreviousState())
        {
            serializer->WriteReal("history_dt", snapshot.PreviousDt());
            WriteEquationArrays(*snapshot.PreviousState(), serializer, "history_field_", mpi);
        }
        if (internal)
        {
            for (const auto &[key, values] : internal->equations)
                WriteEquationArrays(values, serializer, key, mpi);
            for (const auto &[key, values] : internal->arrays)
                WriteStateArray(values, serializer, key, mpi, Runtime::ExecutionContext::Host());
        }
        serializer->CloseFile();
    }
}

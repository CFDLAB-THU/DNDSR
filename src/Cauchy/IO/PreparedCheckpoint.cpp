#include "Cauchy/IO/OutputMeasurement.hpp"
#include "Cauchy/IO/PreparedCheckpoint.hpp"
#include "Cauchy/IO/StateArchive.hpp"
#include "Cauchy/IO/OutputStaging.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    struct PreparedCheckpoint::Data
    {
        DomainStateSnapshot snapshot;
        Json metadata;
        Runtime::Array<index> ids;
        CheckpointMetadata::MeshEnvelope mesh;
        std::shared_ptr<Runtime::CommunicationContext> communication;
        std::shared_ptr<OutputStaging> staging;
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
    void PreparedCheckpoint::BindWriter(std::shared_ptr<OutputStaging> staging)
    {
        DNDS_check_throw_info(_data.unique(), "bind the file writer before sharing a checkpoint");
        _data->staging = std::move(staging);
    }
    void PreparedCheckpoint::Write(const std::string &path, const Json &resolved, const Json &outputController) const
    {
        const auto &data = *_data;
        const auto &snapshot = data.snapshot;
        const MPIInfo mpi(data.communication->Communicator());
        OutputStaging local;
        OutputStaging::Batch batch(data.staging ? *data.staging : local);
        std::vector<std::pair<std::string, Runtime::Array<real>>> fields;
        MPI::LocalStage(mpi, [&]
                        {
            const auto add = [&](const auto &array, const std::string &name)
            { fields.emplace_back(name, batch.HostValues(array, data.writerCopy, name)); };
            const auto equation = [&](const EquationDofs &state, const std::string &prefix)
            {
                for (index i = 0; i < state.PhysicalFieldNumber(); ++i)
                    add(state.PhysicalField(i).Owned(), prefix + std::to_string(i));
            };
            equation(snapshot.Means(), "state_field_");
            if (snapshot.Reconstruction())
                equation(*snapshot.Reconstruction(), "reconstruction_field_");
            if (snapshot.LimitedVRHistory())
                add(*snapshot.LimitedVRHistory(), "limited_vr_history");
            if (snapshot.PreviousState())
                equation(*snapshot.PreviousState(), "history_field_");
            if (data.internal)
            {
                for (const auto &[key, values] : data.internal->equations)
                    equation(values, key);
                for (const auto &[key, values] : data.internal->arrays)
                    add(values, key);
            } });
        const OutputMeasurement file("checkpoint_file");
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, false);
        serializer->WriteInt("cauchy_format", 5);
        if (data.internal)
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
        if (snapshot.PreviousState())
            serializer->WriteReal("history_dt", snapshot.PreviousDt());
        for (const auto &[key, values] : fields)
            Runtime::WriteArray(values, serializer, key, mpi);
        serializer->CloseFile();
    }
}

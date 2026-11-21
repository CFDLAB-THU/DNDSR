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
        // Only synchronous, aliased host output retains solver read leases.
        std::optional<DomainStateSnapshot> pin;
        std::string name;
        real time, dt;
        index step;
        std::optional<real> historyDt;
        Json metadata;
        Runtime::Array<index> ids;
        CheckpointMetadata::MeshEnvelope mesh;
        std::shared_ptr<Runtime::CommunicationContext> communication;
        std::vector<std::pair<std::string, Runtime::Array<real>>> fields;
        bool internal;
        Data(const DomainStateSnapshot &state, Json description, Runtime::Array<index> cellIds,
             CheckpointMetadata::MeshEnvelope envelope, std::shared_ptr<Runtime::CommunicationContext> comm, bool active)
            : name(state.Name()), time(state.Time()), dt(state.LastAcceptedDt()), step(index(state.Epoch().acceptedStep)),
              historyDt(state.PreviousState() ? std::make_optional(state.PreviousDt()) : std::nullopt),
              metadata(std::move(description)), ids(std::move(cellIds)), mesh(std::move(envelope)),
              communication(std::move(comm)), internal(active) {}
    };
    PreparedCheckpoint::PreparedCheckpoint(DomainStateSnapshot snapshot, Json metadata, Runtime::Array<index> ids,
                                           CheckpointMetadata::MeshEnvelope mesh, std::shared_ptr<Runtime::CommunicationContext> communication,
                                           bool detached, std::optional<InternalFields> internal, Runtime::ExecutionContext compute, OutputStaging &staging)
        : _data(std::make_shared<Data>(snapshot, std::move(metadata), std::move(ids), std::move(mesh), std::move(communication), bool(internal)))
    {
        DNDS_check_throw_info(_data->communication, "prepared checkpoint requires an I/O communicator");
        const auto add = [&](const auto &array, const std::string &name, bool copyHost)
        { _data->fields.emplace_back(name, staging.HostValues(array, compute, name, copyHost)); };
        const auto equation = [&](const EquationDofs &state, const std::string &prefix, bool copyHost)
        {
            for (index i = 0; i < state.PhysicalFieldNumber(); ++i)
                add(state.PhysicalField(i).Owned(), prefix + std::to_string(i), copyHost);
        };
        equation(snapshot.Means(), "state_field_", detached);
        if (snapshot.Reconstruction())
            equation(*snapshot.Reconstruction(), "reconstruction_field_", detached);
        if (snapshot.LimitedVRHistory())
            add(*snapshot.LimitedVRHistory(), "limited_vr_history", detached);
        if (snapshot.PreviousState())
            equation(*snapshot.PreviousState(), "history_field_", detached);
        if (internal)
        {
            for (size_t i = 0; i < internal->equations.size(); ++i)
            {
                const auto &[name, state] = internal->equations[i];
                equation(state, name, detached && i >= internal->ownedEquations);
            }
            for (const auto &[name, array] : internal->arrays)
                add(array, name, detached);
        }
        if (!detached)
            _data->pin.emplace(std::move(snapshot));
    }
    bool PreparedCheckpoint::Detached() const { return !_data->pin; }
    void PreparedCheckpoint::Write(const std::string &path, const Json &resolved, const Json &outputController) const
    {
        const auto &data = *_data;
        const MPIInfo mpi(data.communication->Communicator());
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
        serializer->WriteString("domain_name", data.name);
        serializer->WriteReal("accepted_time", data.time);
        serializer->WriteReal("last_accepted_dt", data.dt);
        serializer->WriteIndex("accepted_step", data.step);
        Runtime::WriteArray(data.ids, serializer, "cell_ids", mpi);
        Runtime::WriteArray(data.mesh.topology, serializer, "mesh_topology", mpi);
        Runtime::WriteArray(data.mesh.coordinates, serializer, "mesh_coordinates", mpi);
        Runtime::WriteArray(data.mesh.periodicTransforms, serializer, "periodic_transforms", mpi);
        if (data.historyDt)
            serializer->WriteReal("history_dt", *data.historyDt);
        for (const auto &[key, values] : data.fields)
            Runtime::WriteArray(values, serializer, key, mpi);
        serializer->CloseFile();
    }
}

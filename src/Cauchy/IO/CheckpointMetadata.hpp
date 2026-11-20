#pragma once

#include "Cauchy/IO/CheckpointLoadMode.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Serializer/SerializerBase.hpp"
#include <nlohmann/json.hpp>

namespace DNDS::Cauchy
{
    class Domain;
    class EquationSystem;
    class FiniteVolume;
}

/// Archive metadata helpers. Owned-cell envelopes validate restart identity;
/// scientific field payloads are transferred by StateArchive without flattening.
namespace DNDS::Cauchy::CheckpointMetadata
{
    using Json = nlohmann::ordered_json;
    struct MeshEnvelope
    {
        Runtime::Array<index> topology;
        Runtime::Array<real> coordinates;
        Runtime::Array<real> periodicTransforms;
    };

    /// Normalize inactive historical fields only; retain physical distinctions.
    nlohmann::json CompatibleEquationMetadata(Json value);
    void AgreeLoadMode(const MPIInfo &mpi, CheckpointLoadMode mode);
    nlohmann::json CompatibleConfig(Json config, CheckpointLoadMode mode = CheckpointLoadMode::Resume);
    Json EquationMetadata(const EquationSystem &equation);
    Json FrameMetadata(const FiniteVolume &fv, real time);
    bool CompatibleFrame(const Json &metadata, const FiniteVolume &fv, real time);
    index StoredRows(const Serializer::SerializerBaseSSP &serializer, const std::string &name);
    Runtime::Array<index> CellIds(const Domain &domain);
    MeshEnvelope DescribeMesh(const Domain &domain);
    bool SameRows(const Runtime::Array<real> &left, const Runtime::Array<real> &right);
    std::string RowDifference(const Runtime::Array<real> &left, const Runtime::Array<real> &right,
                              const Runtime::Array<index> &ids);
    std::string TopologyDifference(const Runtime::Array<index> &left, const Runtime::Array<index> &right,
                                   const Runtime::Array<index> &ids, bool legacyFaceImage);
}

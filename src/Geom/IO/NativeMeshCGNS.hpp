#pragma once

#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "DNDS/Diagnostics/SetupProgress.hpp"

namespace DNDS::Geom
{
    /// Explicit BC-zone main/donor transforms and absolute geometric tolerance.
    struct NativePeriodicImportOptions
    {
        Periodicity frames;
        real tolerance = 1e-9;
        Diagnostics::SetupProgress progress;
    };
    enum class NativeSerialDistribution
    {
        Even,
        ReaderOwned
    };

    /// Process-local work counters for reader-owned copies that retain the
    /// CGNS storage family and stride. They measure code-path selection only;
    /// no per-read atomic is touched by ordinary distributed ranks.
    struct NativeSerialCopyCounters
    {
        uint64_t bulkCalls = 0, bulkRows = 0;
        uint64_t rowCalls = 0, rowRows = 0;
    };
    NativeSerialCopyCounters GetNativeSerialCopyCounters();
    void ResetNativeSerialCopyCounters();

    /// Explicit serial import result, held entirely in Runtime arrays. Global
    /// serial scratch is confined to the reader rank; it is not ghost storage.
    /// Numbering follows zone-major, node-major deterministic deduplication.
    struct NativeSerialMeshData
    {
        Runtime::Array<real> coordinates{Runtime::RowLayout::Dense(0, 3)};
        Runtime::Array<index> cellNodes, boundaryNodes;
        Runtime::Array<index> boundaryCells{Runtime::RowLayout::Dense(0, 2)};
        Runtime::Array<ElemInfo> cellInfo{Runtime::RowLayout::Dense(0, 1)}, boundaryInfo{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> cellIds{Runtime::RowLayout::Dense(0, 1)}, nodeIds{Runtime::RowLayout::Dense(0, 1)}, boundaryIds{Runtime::RowLayout::Dense(0, 1)};
        bool periodic = false;
        std::optional<Runtime::Array<NodePeriodicBits>> cellNodesPbi, boundaryNodesPbi;
        Periodicity periodicInfo;
        MeshIdentity identity;
    };

    /// Collective owned-only ingress for construction pipelines. Distributes
    /// primary arrays and computes global incidence without assembling ghosts
    /// or publishing a mesh. Partition, permutation and face construction may
    /// follow before the caller supplies its final GhostSpec to the assembler.
    NativeOwnedMeshData DistributeNativeSerialMeshOwned(
        const MPIInfo &mpi, NativeSerialMeshData serial, int dimension, int readerRank = 0,
        NativeMeshBuildOptions options = {}, std::optional<NativePeriodicImportOptions> periodic = {},
        NativeSerialDistribution distribution = NativeSerialDistribution::Even);

    /// File-reading counterpart of DistributeNativeSerialMeshOwned. Only the
    /// reader holds serial global scratch; distributed closure stays sparse.
    NativeOwnedMeshData ReadNativeCGNSOwned(
        const MPIInfo &mpi, const std::string &path, int dimension,
        const t_FBCName_2_ID &boundaryId, int readerRank = 0,
        NativeMeshBuildOptions options = {},
        std::optional<NativePeriodicImportOptions> periodic = {},
        NativeSerialDistribution distribution = NativeSerialDistribution::Even);

    /// Collective native serial ingress. Only readerRank supplies payload rows;
    /// every rank supplies identical options. Optional periodic matching runs
    /// once before sparse direct distribution, and owned IDs remain payload.
    std::shared_ptr<const UnstructuredMeshState> DistributeNativeSerialMesh(
        const MPIInfo &mpi, NativeSerialMeshData serial, int dimension, int readerRank = 0,
        NativeMeshBuildOptions options = {}, std::optional<NativePeriodicImportOptions> periodic = {},
        NativeSerialDistribution distribution = NativeSerialDistribution::Even);

    /// Local file operation, no MPI collectives. Supports the existing CGNS
    /// multi-zone, mixed-element and boundary PointList/PointRange conventions.
    NativeSerialMeshData ReadNativeCGNSSerial(const std::string &path, int dimension,
                                              const t_FBCName_2_ID &boundaryId);

    /// Collective explicit serial import, native direct row distribution to
    /// even contiguous ownership per family, then native sparse construction.
    /// All ranks supply identical options/path; boundaryId runs on readerRank.
    /// Optional BC-zone main/donor identification runs on readerRank before
    /// distribution. CGNS periodic connection properties are explicitly rejected;
    /// transforms must be supplied via the BC-zone convention. No legacy
    /// mesh/Array workspace is used.
    std::shared_ptr<const UnstructuredMeshState> ReadNativeCGNS(
        const MPIInfo &mpi, const std::string &path, int dimension,
        const t_FBCName_2_ID &boundaryId, int readerRank = 0,
        NativeMeshBuildOptions options = {},
        std::optional<NativePeriodicImportOptions> periodic = {},
        NativeSerialDistribution distribution = NativeSerialDistribution::Even);
}

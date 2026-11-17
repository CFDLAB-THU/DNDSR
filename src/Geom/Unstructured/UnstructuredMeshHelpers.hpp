/** @file Mesh_Helpers.hpp
 *  @brief Unified mesh helpers: read, prepare, build boundary, serialize.
 *
 *  Thin wrappers that compose the canonical mesh assembly pipeline from
 *  lower-level UnstructuredMesh / UnstructuredMeshSerialRW methods.
 *  Mirror the Python helpers in DNDSR.Geom.utils.
 *
 *  See docs/dev/mesh_helpers_design.md for the design document.
 */
#pragma once

#include "DNDS/MPI/Failure.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/IO/NativeMeshOpenFOAM.hpp"
#include "Geom/Refinement/NativeMeshRefinement.hpp"
#include "Geom/IO/NativeMeshArchive.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Compat/NativeMeshLegacyArchive.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "DNDS/Serializer/SerializerFactory.hpp"

#include <functional>
#include <fstream>
#include <string>

namespace DNDS::Geom
{

    /// Options for the face-complete solver preparation stage.
    struct PrepareMeshOptions
    {
        int reorderCells = 0;       ///< 0 = natural, 1 = reorder
        int reorderParts = 1;       ///< nParts for ReorderLocalCells
        bool buildSerialOut = true; ///< build serial output arrays
        int reorderPartsInner = 1;  ///< nested local partitioning
        int ghostLayers = 1;        ///< explicit final cell-ring depth
        bool buildEdges = false;    ///< additionally construct 3D edges
    };

    inline NativeOwnedMeshData DeriveAndPartitionNativeMeshOwned(
        NativeOwnedMeshData owned, const MPIInfo &mpi,
        const PartitionOptions &partOpts, int elevation, int bisect);

    inline std::shared_ptr<const UnstructuredMeshState> PrepareNativeMeshForSolverOwned(
        NativeOwnedMeshData owned, const MPIInfo &mpi, const PrepareMeshOptions &opts);

    /// Read and partition without publishing an intermediate primary state.
    inline NativeOwnedMeshData ReadAndDeriveNativeMeshFromCGNSOwned(
        const MPIInfo &mpi, int dimension, const std::string &meshFile,
        const PartitionOptions &partOpts, const t_FBCName_2_ID &nameMapper,
        std::optional<NativePeriodicImportOptions> periodic = {},
        int elevation = 0, int bisect = 0, int readerRank = 0,
        MeshAuditOptions audit = {}, std::shared_ptr<Runtime::CommunicationContext> communication = {});
    inline NativeOwnedMeshData ReadAndDeriveNativeMeshFromOpenFOAMOwned(
        const MPIInfo &mpi, const std::string &meshFile, const PartitionOptions &partOpts,
        const std::map<std::string, std::string> &nameMapping,
        const t_FBCName_2_ID &nameMapper, std::optional<NativePeriodicImportOptions> periodic = {},
        int elevation = 0, int bisect = 0, int readerRank = 0,
        MeshAuditOptions audit = {}, std::shared_ptr<Runtime::CommunicationContext> communication = {});

    /// Partition the source before optional order derivation, matching source
    /// mesh setup. Refinement retains parent ownership without repartitioning;
    /// all metadata comes from native publications and existing O2 is retained.
    /// Elevation retains Cell IDs for Euler legacy restart compatibility;
    /// bisection establishes fresh rank-prefix child IDs as in source setup.
    inline NativeOwnedMeshData DeriveAndPartitionNativeMeshOwned(
        NativeOwnedMeshData owned, const MPIInfo &mpi,
        const PartitionOptions &partOpts, int elevation, int bisect)
    {
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !owned.communication || owned.communication->ChecksCollectivePreflight());
        owned.LocalStage(mpi, [&]
                         { DNDS_check_throw_info((elevation == 0 || elevation == 1) && bisect >= 0 && bisect <= 4,
                                                 "native mesh setup requires elevation 0/1 and bisection count 0..4"); });
        std::array<int, 2> derived{elevation, bisect}, expected = derived;
        MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, 0, mpi.comm);
        owned.LocalStage(mpi, [&]
                         { DNDS_check_throw_info(derived == expected, "native mesh derivation options differ across ranks"); });
        NativePartitionOptions partition{partOpts.metisType, partOpts.metisUfactor, partOpts.metisSeed,
                                         partOpts.edgeWeightMethod, std::max(partOpts.metisNcuts, 1), 1};
        owned = NativeMeshPreparation::PartitionOwned(owned, mpi, partition).owned;
        auto isO2 = [&]
        {
            int local = 1;
            owned.LocalStage(mpi, [&]
                             {
                auto info = owned.cellInfo->Read();
                for (size_t row = 0; row < owned.cellInfo->RowCount(); ++row)
                    local = local && Elem::Element{info.View()(row, 0).getElemType()}.GetOrder() == 2; });
            MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT, MPI_MIN, mpi.comm);
            return bool(local);
        };
        const NativeMeshBuildOptions build{1, true};
        if (elevation && !isO2())
            owned = NativeMeshRefinement::ElevateOwned(owned, mpi, build, true).owned;
        for (int pass = 0; pass < bisect; ++pass)
        {
            if (!isO2())
                owned = NativeMeshRefinement::ElevateOwned(owned, mpi, build, true).owned;
            owned = NativeMeshRefinement::BisectOwned(owned, mpi, build).owned;
        }
        return owned;
    }

    inline std::shared_ptr<const UnstructuredMeshState> DeriveAndPartitionNativeMesh(
        NativeOwnedMeshData owned, const MPIInfo &mpi,
        const PartitionOptions &partOpts, int elevation, int bisect, bool deferNewFacesUntilPreparation = false)
    {
        owned = DeriveAndPartitionNativeMeshOwned(std::move(owned), mpi, partOpts, elevation, bisect);
        if (!elevation && !bisect && !deferNewFacesUntilPreparation)
            owned = NativeMeshTopology::Build(std::move(owned), mpi);
        GhostSpec recipe;
        owned.LocalStage(mpi, [&]
                         { recipe = DefaultNativeMeshGhostSpec(owned, 1, true); });
        return NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    }

    /// Completed-state compatibility entry; construction after capture is owned
    /// through partition/refinement and invokes the assembler only at return.
    inline std::shared_ptr<const UnstructuredMeshState> DeriveAndPartitionNativeMesh(
        std::shared_ptr<const UnstructuredMeshState> state, const MPIInfo &mpi,
        const PartitionOptions &partOpts, int elevation, int bisect, bool deferNewFacesUntilPreparation = false)
    {
        return DeriveAndPartitionNativeMesh(NativeMeshAssembler::CaptureOwned(*state, mpi), mpi,
                                            partOpts, elevation, bisect, deferNewFacesUntilPreparation);
    }

    inline ssp<UnstructuredMesh> ReadNativeMeshFromCGNS(
        const MPIInfo &mpi, int dimension, const std::string &meshFile,
        const PartitionOptions &partOpts, const t_FBCName_2_ID &nameMapper,
        std::optional<NativePeriodicImportOptions> periodic = {},
        int elevation = 0, int bisect = 0, int readerRank = 0, bool deferNewFacesUntilPreparation = false,
        MeshAuditOptions audit = {})
    {
        // Source setup historically partitions the reader's serial graph with
        // METIS for both KWAY and RB. An even split selects ParMETIS instead
        // and changes the partition despite identical user options.
        const auto distribution = NativeSerialDistribution::ReaderOwned;
        auto owned = ReadNativeCGNSOwned(mpi, meshFile, dimension, nameMapper, readerRank, {1, true, audit}, periodic, distribution);
        return UnstructuredMesh::FromNative(mpi, DeriveAndPartitionNativeMesh(std::move(owned), mpi, partOpts, elevation, bisect,
                                                                              deferNewFacesUntilPreparation));
    }

    inline NativeOwnedMeshData ReadAndDeriveNativeMeshFromCGNSOwned(
        const MPIInfo &mpi, int dimension, const std::string &meshFile,
        const PartitionOptions &partOpts, const t_FBCName_2_ID &nameMapper,
        std::optional<NativePeriodicImportOptions> periodic, int elevation, int bisect,
        int readerRank, MeshAuditOptions audit, std::shared_ptr<Runtime::CommunicationContext> communication)
    {
        auto owned = ReadNativeCGNSOwned(mpi, meshFile, dimension, nameMapper, readerRank,
                                         {1, true, audit, std::move(communication)}, periodic, NativeSerialDistribution::ReaderOwned);
        return DeriveAndPartitionNativeMeshOwned(std::move(owned), mpi, partOpts, elevation, bisect);
    }

    inline NativeOwnedMeshData ReadAndDeriveNativeMeshFromOpenFOAMOwned(
        const MPIInfo &mpi, const std::string &meshFile, const PartitionOptions &partOpts,
        const std::map<std::string, std::string> &nameMapping,
        const t_FBCName_2_ID &nameMapper, std::optional<NativePeriodicImportOptions> periodic,
        int elevation, int bisect, int readerRank, MeshAuditOptions audit, std::shared_ptr<Runtime::CommunicationContext> communication)
    {
        auto owned = ReadNativeOpenFOAMOwned(mpi, meshFile, nameMapping, nameMapper, readerRank,
                                             {1, true, audit, std::move(communication)}, periodic, NativeSerialDistribution::ReaderOwned);
        return DeriveAndPartitionNativeMeshOwned(std::move(owned), mpi, partOpts, elevation, bisect);
    }

    inline std::shared_ptr<const UnstructuredMeshState> PrepareNativeMeshForSolverOwned(
        NativeOwnedMeshData owned, const MPIInfo &mpi, const PrepareMeshOptions &opts)
    {
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !owned.communication || owned.communication->ChecksCollectivePreflight());
        std::array<int, 6> settings{opts.reorderCells, opts.reorderParts, int(opts.buildSerialOut),
                                    opts.reorderPartsInner, opts.ghostLayers, int(opts.buildEdges)},
            expected = settings;
        MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, 0, mpi.comm);
        owned.LocalStage(mpi, [&]
                         { DNDS_check_throw_info(settings == expected && opts.ghostLayers > 0 &&
                                                     (opts.reorderCells == 0 || opts.reorderCells == 1),
                                                 "native preparation options differ across ranks or invalid depth"); });
        if (opts.reorderCells == 1)
            owned = NativeMeshTransition::ReorderOwnedCells(owned, mpi, opts.reorderParts, opts.reorderPartsInner).owned;
        owned = NativeMeshTopology::Build(std::move(owned), mpi, true, opts.buildEdges);
        GhostSpec recipe;
        owned.LocalStage(mpi, [&]
                         {
            recipe = DefaultNativeMeshGhostSpec(owned, opts.ghostLayers, true);
            if (owned.ghostSpec)
                recipe.chains.insert(recipe.chains.end(), owned.ghostSpec->chains.begin(), owned.ghostSpec->chains.end());
            owned.ghostSpec = recipe; });
        return NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    }

    inline ssp<UnstructuredMesh> ReadNativeMeshFromOpenFOAM(
        const MPIInfo &mpi, const std::string &meshFile, const PartitionOptions &partOpts,
        const std::map<std::string, std::string> &nameMapping,
        const t_FBCName_2_ID &nameMapper, std::optional<NativePeriodicImportOptions> periodic = {},
        int elevation = 0, int bisect = 0, int readerRank = 0, bool deferNewFacesUntilPreparation = false,
        MeshAuditOptions audit = {})
    {
        const auto distribution = NativeSerialDistribution::ReaderOwned;
        auto owned = ReadNativeOpenFOAMOwned(mpi, meshFile, nameMapping, nameMapper, readerRank, {1, true, audit}, periodic, distribution);
        return UnstructuredMesh::FromNative(mpi, DeriveAndPartitionNativeMesh(std::move(owned), mpi, partOpts, elevation, bisect,
                                                                              deferNewFacesUntilPreparation));
    }

    /// Open an existing full native or legacy primary archive. Schema detection
    /// is explicit and collective; legacy decode preserves original IDs but
    /// creates a new namespace because the old format stores none.
    inline ssp<UnstructuredMesh> ReadNativeMeshFromArchive(
        const MPIInfo &mpi, Serializer::SerializerFactory factory, const std::string &inputPath,
        bool repartition = false, const PartitionOptions &partOpts = {}, int ghostLayers = 1)
    {
        std::string signature;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(ghostLayers >= 1 && (factory.type == "JSON" || factory.type == "H5") && factory.hdfDeflateLevel >= 0 &&
                                  factory.hdfDeflateLevel <= 9 && factory.hdfChunkSize >= 0 &&
                                  factory.jsonBinaryDeflateLevel >= 0 && factory.jsonBinaryDeflateLevel <= 9,
                                  "native archive serializer configuration invalid");
            signature = nlohmann::json{{"path", inputPath}, {"type", factory.type}, {"deflate", factory.hdfDeflateLevel},
                {"chunk", factory.hdfChunkSize}, {"collectiveData", factory.hdfCollOnData}, {"collectiveMeta", factory.hdfCollOnMeta},
                {"jsonDeflate", factory.jsonBinaryDeflateLevel}, {"codec", factory.jsonUseCodecOnUInt8}, {"repartition", repartition}, {"ghostLayers", ghostLayers},
                {"partitionType", partOpts.metisType}, {"ufactor", partOpts.metisUfactor}, {"seed", partOpts.metisSeed},
                {"weight", partOpts.edgeWeightMethod}, {"cuts", partOpts.metisNcuts}}.dump();
            DNDS_check_throw_info(signature.size() <= size_t(INT_MAX), "native archive configuration too large"); });
        int length = mpi.rank == 0 ? int(signature.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string canonical;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { canonical = mpi.rank == 0 ? signature : std::string(size_t(length), '\0'); });
        MPI_Bcast(canonical.data(), length, MPI_CHAR, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(signature == canonical, "native archive reader options differ across ranks"); });
        Serializer::SerializerBaseSSP serializer;
        std::string path;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            path = std::get<0>(factory.ModifyFilePath(inputPath, mpi, "part_%d", true));
            std::ifstream probe(path, std::ios::binary);
            DNDS_check_throw_info(probe.good(), "native archive input file is not readable"); });
        // HDF5 serializer construction duplicates a communicator.
        serializer = factory.BuildSerializer(mpi);
        struct CloseArchive
        {
            Serializer::SerializerBaseSSP serializer;
            bool opened = false, json = false;
            ~CloseArchive()
            {
                if (opened || json)
                    try
                    {
                        serializer->CloseFile();
                    }
                    catch (...)
                    {
                        MPI::AbortIfSolverFailure("native mesh derivation failed");
                    }
            }
        } close{serializer, false, factory.type == "JSON"};
        // JSON opens/parses one independent file per rank. Agree on parse
        // success before any rank enters the schema collectives: otherwise a
        // rank-local malformed file can strand peers in schema Bcast while a
        // coordinated caller proceeds to its failure/cleanup agreement.
        // HDF5 OpenFile may itself be collective and must not be wrapped in a
        // recoverable local-stage guard. Solver catches still terminate.
        if (factory.type == "JSON")
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { serializer->OpenFile(path, true); });
        else
            serializer->OpenFile(path, true);
        close.opened = true;
        const auto partition = repartition && !serializer->IsPerRank() ? Serializer::ArrayGlobalOffset_EvenSplit : Serializer::ArrayGlobalOffset_Unknown;
        auto owned = ReadCompatibleNativeMeshArchiveOwned(mpi, serializer, "meshPart", partition);
        serializer->CloseFile();
        close.opened = close.json = false;
        if (repartition)
        {
            NativePartitionOptions options{partOpts.metisType, partOpts.metisUfactor, partOpts.metisSeed,
                                           partOpts.edgeWeightMethod, partOpts.metisNcuts, ghostLayers};
            owned = NativeMeshPreparation::PartitionOwned(owned, mpi, options).owned;
            if (!owned.requirements.faces)
            {
                owned = NativeMeshTopology::Build(std::move(owned), mpi);
                FieldTransferDetail::LocalStage(mpi, [&]
                                                { owned.ghostSpec = ExtendNativeMeshGhostSpecForTopology(owned, true, false, ghostLayers); });
            }
        }
        return UnstructuredMesh::FromNative(mpi, AssembleNativeMesh(mpi, std::move(owned), ghostLayers));
    }

    // -----------------------------------------------------------------------
    // BuildGhostPrimary: explicit completed primary operation
    // -----------------------------------------------------------------------

    /** @brief Finalize primary closure for an already readable native mesh.
     *
     *  Compatibility entry for callers requesting an immediate read state.
     *  Owned construction pipelines call the assembler only at their end.
     *
     *  @param nGhostLayers  Number of cell2cell hops for ghost cells (default 1).
     */
    inline void BuildGhostPrimary(UnstructuredMesh &mesh, int nGhostLayers = 1)
    {
        mesh.PrepareNativeTopology(false, false, nGhostLayers);
    }

    // -----------------------------------------------------------------------
    // ReadMeshFromCGNS
    // -----------------------------------------------------------------------

    /** @brief Read a CGNS mesh, partition, and optionally elevate/bisect.
     *
     *  After this call the mesh is distributed with ghost cells and local
     *  indices, but NOT solver-ready (no faces, no ghost N2CB).
     *
     *  @param mesh         Fresh UnstructuredMesh (constructed with mpi, dim).
     *  @param reader       UnstructuredMeshSerialRW bound to @p mesh.
     *  @param meshFile     Path to the CGNS file.
     *  @param partOpts     Metis partition options.
     *  @param periodicTol  Tolerance for periodic boundary deduplication.
     *  @param elevation    0 = none, 1 = O1->O2.
     *  @param bisect       Number of bisection passes (0..4).
     *  @param nameMapper   Optional callback mapping BC names to IDs
     *                      (for the CGNS reader).  If nullptr, the reader's
     *                      default AutoAppendName2ID is used.
     */
    inline void ReadMeshFromCGNS(
        ssp<UnstructuredMesh> &mesh,
        ssp<UnstructuredMeshSerialRW> &reader,
        const std::string &meshFile,
        const PartitionOptions &partOpts = {},
        real periodicTol = 1e-9,
        int elevation = 0,
        int bisect = 0,
        std::function<t_index(const std::string &)> nameMapper = nullptr)
    {
        const auto mpi = mesh->getMPI();
        const int dimension = mesh->getDim();
        int readerRank = 0;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(reader && reader->mesh.get() == mesh.get(), "native CGNS helper requires the bound reader");
            readerRank = reader->mRank; });
        NativePeriodicImportOptions periodic{mesh->PeriodicMetadata(), periodicTol};
        AutoAppendName2ID generatedNames;
        const t_FBCName_2_ID names = nameMapper ? nameMapper : t_FBCName_2_ID([&](const std::string &name)
                                                                              { return generatedNames(name); });
        auto prepared = ReadNativeMeshFromCGNS(mpi, dimension, meshFile, partOpts, names,
                                               periodic, elevation, bisect, readerRank, true);
        mesh = std::move(prepared);
        reader = make_ssp<UnstructuredMeshSerialRW>(mesh, readerRank);
    }

    // -----------------------------------------------------------------------
    // ReadMeshFromH5
    // -----------------------------------------------------------------------

    /** @brief Read a mesh from DNDSR H5 with even-split + ParMetis repartition.
     *
     *  Works with any number of MPI ranks regardless of how the file was written.
     */
    inline void ReadMeshFromH5(
        UnstructuredMesh &mesh,
        Serializer::SerializerFactory factory,
        const std::string &h5Path,
        const PartitionOptions &partOpts = {})
    {
        auto result = ReadNativeMeshFromArchive(mesh.getMPI(), std::move(factory), h5Path, true, partOpts);
        mesh.ReplaceNativeForSetup(result->CurrentNativeState());
    }

    /** @brief Read a pre-partitioned mesh from H5 (exact np match required).
     *
     *  Uses ReadSerialize (no repartitioning).  The file must have been written
     *  with the same number of MPI ranks.
     */
    inline void ReadMeshFromH5Parallel(
        UnstructuredMesh &mesh,
        Serializer::SerializerFactory factory,
        const std::string &h5Path)
    {
        auto result = ReadNativeMeshFromArchive(mesh.getMPI(), std::move(factory), h5Path, false);
        mesh.ReplaceNativeForSetup(result->CurrentNativeState());
    }

    // -----------------------------------------------------------------------
    // PrepareMesh
    // -----------------------------------------------------------------------

    /** @brief Prepare a distributed mesh for solver use.
     *
     *  Steps: cell reorder, face interpolation, ghost N2CB, optional serial output.
     *  Does NOT include: elevation smoothing, wall distance, coord transforms,
     *  periodic nodes, VTK connectivity, boundary mesh.
     *  Those remain solver / caller responsibility.
     *
     *  @param mesh     Distributed mesh (output of ReadMeshFromCGNS / ReadMeshFromH5).
     *  @param reader   The reader bound to @p mesh.
     *  @param opts     Options controlling reordering and serial output.
     */
    inline void PrepareMesh(
        UnstructuredMesh &mesh,
        UnstructuredMeshSerialRW &reader,
        const PrepareMeshOptions &opts = {})
    {
        const auto mpi = mesh.getMPI();
        const auto communication = mesh.PreparationCommunication();
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, checks);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(mesh.HasNativeState() && reader.mesh.get() == &mesh &&
                                                                    (opts.reorderCells == 0 || opts.reorderCells == 1),
                                                                "native preparation requires the bound output adapter and supported reorder option"); });
        mesh.CheckCollectiveMutationAvailable(checks);
        auto prepared = PrepareNativeMeshForSolverOwned(
            NativeMeshAssembler::CaptureOwned(*mesh.CurrentNativeState(), mpi), mpi, opts);
        mesh.ReplaceNativeForSetup(std::move(prepared));
        if (opts.buildSerialOut)
            reader.BuildSerialOut();
    }

    // -----------------------------------------------------------------------
    // BuildBndMesh
    // -----------------------------------------------------------------------

    /** @brief Extract the boundary surface mesh from a solver-ready volume mesh.
     *
     *  @param mesh       The volume mesh (must have faces built).
     *  @param meshBnd    Output boundary mesh (constructed with dim-1).
     *  @param readerBnd  Reader bound to @p meshBnd.
     *  @param buildSerialOut  Whether to build serial output for the bnd mesh.
     */
    inline void BuildBndMesh(
        UnstructuredMesh &mesh,
        UnstructuredMesh &meshBnd,
        UnstructuredMeshSerialRW &readerBnd,
        bool buildSerialOut = true)
    {
        FieldTransferDetail::LocalStage(mesh.getMPI(), [&]
                                        { DNDS_check_throw_info(mesh.HasNativeState() && readerBnd.mesh.get() == &meshBnd,
                                                                "native boundary helper requires native input and bound output adapter"); });
        int expected = buildSerialOut;
        MPI_Bcast(&expected, 1, MPI_INT, 0, mesh.getMPI().comm);
        FieldTransferDetail::LocalStage(mesh.getMPI(), [&]
                                        { DNDS_check_throw_info(expected == int(buildSerialOut), "native boundary output options differ across ranks"); });
        mesh.ConstructBndMesh(meshBnd);
        if (buildSerialOut)
            readerBnd.BuildSerialOut();
    }

    // -----------------------------------------------------------------------
    // SerializeMesh
    // -----------------------------------------------------------------------

    /** @brief Write a full native archive for matching or redistributed restart.
     *
     *  Rejects nonnative input collectively before creating files or directories.
     *  The Python equivalent of the C++ ``partitionMeshOnly`` path.  The
     *  caller controls where this occurs in the pipeline; in Euler solvers it
     *  occurs before coordinate transforms, boundary mesh extraction, and solver
     *  setup.  Redistribution is supported by the collective H5 serializer, not
     *  by per-rank serializers.
     */
    inline void SerializeMesh(
        UnstructuredMesh &mesh,
        const std::string &outputPath,
        Serializer::SerializerFactory &factory)
    {
        auto &mpi = mesh.getMPI();
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(mesh.HasNativeState(), "native serialization helper requires native mesh input"); });
        auto [pathMod, pathPart] = factory.ModifyFilePath(outputPath, mpi, "part_%d", false);
        Serializer::SerializerBaseSSP serializerP = factory.BuildSerializer(mpi);

        serializerP->OpenFile(pathMod, false);
        mesh.WriteNativeArchive(serializerP, "meshPart");
        serializerP->CloseFile();
    }

    // -----------------------------------------------------------------------
    // MeshH5Path (naming convention)
    // -----------------------------------------------------------------------

    /** @brief Build the conventional H5 filename for a partitioned mesh.
     *
     *  Convention: ``{base}_part_{mpiSize}[_elevated][_bisectN]``
     *  (The caller / SerializerFactory appends the actual extension.)
     */
    inline std::string MeshH5Path(
        const std::string &base, int mpiSize,
        int elevation = 0, int bisect = 0)
    {
        using namespace std::literals;
        std::string name = base + "_part_" + std::to_string(mpiSize);
        if (elevation == 1)
            name += "_elevated";
        if (bisect > 0)
            name += "_bisect" + std::to_string(bisect);
        return name;
    }

} // namespace DNDS::Geom

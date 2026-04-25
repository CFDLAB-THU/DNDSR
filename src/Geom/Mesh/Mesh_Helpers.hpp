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

#include "Mesh.hpp"
#include "DNDS/Serializer/SerializerFactory.hpp"

#include <functional>
#include <string>

namespace DNDS::Geom
{

    // -----------------------------------------------------------------------
    // BuildGhostPrimary: the 5-step connectivity+ghost sequence
    // -----------------------------------------------------------------------

    /** @brief Build connectivity and ghost layer for a freshly-distributed mesh.
     *
     *  This 5-step sequence is identical across all read paths (CGNS serial,
     *  H5 parallel, H5 distributed).
     *
     *  @param nGhostLayers  Number of cell2cell hops for ghost cells (default 1).
     */
    inline void BuildGhostPrimary(UnstructuredMesh &mesh, int nGhostLayers = 1)
    {
        mesh.RecoverNode2CellAndNode2Bnd();
        mesh.RecoverCell2CellAndBnd2Cell();
        mesh.BuildGhostPrimary(nGhostLayers);
        mesh.AdjGlobal2LocalPrimary();
        mesh.AdjGlobal2LocalN2CB();
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
        auto &mpi = mesh->getMPI();
        int dim = mesh->getDim();

        if (nameMapper)
            reader->ReadFromCGNSSerial(meshFile, nameMapper);
        else
            reader->ReadFromCGNSSerial(meshFile);

        reader->Deduplicate1to1Periodic(periodicTol);
        reader->BuildCell2Cell();
        reader->MeshPartitionCell2Cell(partOpts);
        reader->PartitionReorderToMeshCell2Cell();

        Geom::BuildGhostPrimary(*mesh);

        // --- Optional O2 elevation ---
        if (elevation == 1)
        {
            ssp<UnstructuredMesh> meshO2;
            DNDS_MAKE_SSP(meshO2, mpi, dim);
            meshO2->BuildO2FromO1Elevation(*mesh);
            std::swap(meshO2, mesh);

            reader->mesh = mesh;
            Geom::BuildGhostPrimary(*mesh);
        }

        // --- Optional bisection ---
        DNDS_assert(bisect >= 0 && bisect <= 4);
        for (int iter = 1; iter <= bisect; iter++)
        {
            ssp<UnstructuredMesh> meshO2;
            DNDS_MAKE_SSP(meshO2, mpi, dim);
            meshO2->BuildO2FromO1Elevation(*mesh);

            meshO2->RecoverNode2CellAndNode2Bnd();
            meshO2->RecoverCell2CellAndBnd2Cell();
            meshO2->BuildGhostPrimary();

            ssp<UnstructuredMesh> meshO1B;
            DNDS_MAKE_SSP(meshO1B, mpi, dim);
            meshO1B->BuildBisectO1FormO2(*meshO2);

            std::swap(meshO1B, mesh);
            reader->mesh = mesh;
            Geom::BuildGhostPrimary(*mesh);

            index nCell = mesh->NumCellGlobal();
            index nNode = mesh->NumNodeGlobal();
            if (mpi.rank == 0)
                log() << fmt::format("Mesh Direct Bisect {} done, nCell [{}], nNode [{}]",
                                     iter, nCell, nNode)
                      << std::endl;
        }
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
        auto &mpi = mesh.getMPI();
        auto [pathMod, pathPart] = factory.ModifyFilePath(h5Path, mpi, "part_%d", true);
        Serializer::SerializerBaseSSP serializerP = factory.BuildSerializer(mpi);

        if (mpi.rank == 0)
            log() << "ReadMeshFromH5: distributed read via [" << factory.type
                  << "]" << std::endl;

        serializerP->OpenFile(pathMod, true);
        mesh.ReadSerializeAndDistribute(serializerP, "meshPart", partOpts);
        serializerP->CloseFile();

        Geom::BuildGhostPrimary(mesh);
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
        auto &mpi = mesh.getMPI();
        auto [pathMod, pathPart] = factory.ModifyFilePath(h5Path, mpi, "part_%d", true);
        Serializer::SerializerBaseSSP serializerP = factory.BuildSerializer(mpi);

        if (mpi.rank == 0)
            log() << "ReadMeshFromH5Parallel: reading via [" << factory.type
                  << "]" << std::endl;

        serializerP->OpenFile(pathMod, true);
        mesh.ReadSerialize(serializerP, "meshPart");
        serializerP->CloseFile();

        Geom::BuildGhostPrimary(mesh);
    }

    // -----------------------------------------------------------------------
    // PrepareMesh
    // -----------------------------------------------------------------------

    /** @brief Options for PrepareMesh. */
    struct PrepareMeshOptions
    {
        int reorderCells = 0;       ///< 0 = natural, 1 = reorder
        int reorderParts = 1;       ///< nParts for ReorderLocalCells
        bool buildSerialOut = true; ///< build serial output arrays
    };

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
        auto &mpi = mesh.getMPI();

        if (opts.reorderCells == 1)
            mesh.ReorderLocalCells(opts.reorderParts);

        mesh.InterpolateFace();
        mesh.AssertOnFaces();

        mesh.AdjLocal2GlobalN2CB();
        mesh.BuildGhostN2CB();
        mesh.AdjGlobal2LocalN2CB();
        log() << fmt::format("{}, NumBndGhost {}", mpi.rank, mesh.NumBndGhost())
              << std::endl;

        // Owned nodes must have all cell neighbors resolved to local indices.
        // Ghost (son) nodes may have unresolved neighbors encoded as negative.
        for (index iNode = 0; iNode < mesh.NumNode(); iNode++)
            for (index iCell : mesh.node2cell.father->operator[](iNode))
                DNDS_assert(iCell >= 0);

        if (opts.buildSerialOut)
        {
            mesh.AdjLocal2GlobalPrimary();
            reader.BuildSerialOut();
            mesh.AdjGlobal2LocalPrimary();
        }
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
        mesh.ConstructBndMesh(meshBnd);
        if (buildSerialOut)
        {
            meshBnd.AdjLocal2GlobalPrimaryForBnd();
            readerBnd.BuildSerialOut();
            meshBnd.AdjGlobal2LocalPrimaryForBnd();
        }
    }

    // -----------------------------------------------------------------------
    // SerializeMesh
    // -----------------------------------------------------------------------

    /** @brief Write a partitioned mesh to H5 for later distributed read.
     *
     *  The Python equivalent of the C++ ``partitionMeshOnly`` path.
     */
    inline void SerializeMesh(
        UnstructuredMesh &mesh,
        const std::string &outputPath,
        Serializer::SerializerFactory &factory)
    {
        auto &mpi = mesh.getMPI();
        auto [pathMod, pathPart] = factory.ModifyFilePath(outputPath, mpi, "part_%d", false);
        Serializer::SerializerBaseSSP serializerP = factory.BuildSerializer(mpi);

        serializerP->OpenFile(pathMod, false);
        mesh.AdjLocal2GlobalPrimary();
        mesh.WriteSerialize(serializerP, "meshPart");
        mesh.AdjGlobal2LocalPrimary();
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

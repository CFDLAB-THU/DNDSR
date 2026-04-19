from DNDSR import Geom
from DNDSR import DNDS

import os


def default_serializer_factory():
    fac = DNDS.Serializer.SerializerFactory()
    fac.from_dict(
        {
            "type": "H5",
            "hdfDeflateLevel": 0,
            "hdfChunkSize": 0,
            "hdfCollOnMeta": False,
            "hdfCollOnData": False,
            "jsonBinaryDeflateLevel": 5,
            "jsonUseCodecOnUInt8": True,
        }
    )
    return fac


def create_mesh_from_CGNS(
    meshFile: str,
    mpi: DNDS.MPIInfo,
    dim: int = 2,
    periodic_tolerance: float = 1e-9,
    inner_process_parts: int = 1,
    second_level_parts: int = 1,
    periodic_geometry={
        "translation1": [1, 0, 0],
        "rotationCenter1": [0, 0, 0],
        "eulerAngles3": [0, 0, 0],
    },
    readMeshMode: str = "Serial",  # or "Parallel"
    meshElevation: str = "",  # or "O2"
    meshDirectBisect: int = 0,
    serializerFactory: DNDS.Serializer.SerializerFactory = default_serializer_factory(),
    outPltMode: str = "Serial",
):

    mesh = Geom.UnstructuredMesh(mpi, dim)
    meshReader = Geom.UnstructuredMeshSerialRW(mesh, 0)
    assert os.path.isfile(meshFile)
    mesh.SetPeriodicGeometry(
        **periodic_geometry,
    )
    if readMeshMode == "Serial":
        name2ID = meshReader.ReadFromCGNSSerial(meshFile)
        # print(name2ID.n2id_map)
        meshReader.Deduplicate1to1Periodic(periodic_tolerance)
        meshReader.BuildCell2Cell()
        meshReader.MeshPartitionCell2Cell(
            {
                "metisType": "KWAY",
                "metisUfactor": 5,
                "metisSeed": 0,
                "metisNcuts": 3,
            }
        )
        meshReader.PartitionReorderToMeshCell2Cell()

        mesh.RecoverNode2CellAndNode2Bnd()
        mesh.RecoverCell2CellAndBnd2Cell()
        mesh.BuildGhostPrimary()
        mesh.AdjGlobal2LocalPrimary()
        mesh.AdjGlobal2LocalN2CB()

        if meshElevation == "O2":
            meshO2 = Geom.UnstructuredMesh(mpi, dim)
            meshO2.BuildO2FromO1Elevation(mesh)
            meshO2, mesh = mesh, meshO2

            meshReader.mesh = mesh
            mesh.RecoverNode2CellAndNode2Bnd()
            mesh.RecoverCell2CellAndBnd2Cell()
            mesh.BuildGhostPrimary()
            mesh.AdjGlobal2LocalPrimary()
            mesh.AdjGlobal2LocalN2CB()

            del meshO2

        if not (0 <= meshDirectBisect <= 4):
            raise ValueError(f"bad input: meshDirectBisect is {meshDirectBisect}")
        for iter in range(1, 1 + meshDirectBisect):
            meshO2 = Geom.UnstructuredMesh(mpi, dim)
            meshO2.BuildO2FromO1Elevation(mesh)

            meshO2.RecoverNode2CellAndNode2Bnd()
            meshO2.RecoverCell2CellAndBnd2Cell()
            meshO2.BuildGhostPrimary()
            meshO1B = Geom.UnstructuredMesh(mpi, dim)
            meshO1B.BuildBisectO1FormO2(meshO2)

            mesh, meshO1B = meshO1B, mesh
            meshReader.mesh = mesh
            mesh.RecoverNode2CellAndNode2Bnd()
            mesh.RecoverCell2CellAndBnd2Cell()
            mesh.BuildGhostPrimary()
            mesh.AdjGlobal2LocalPrimary()
            mesh.AdjGlobal2LocalN2CB()
            nCell = mesh.NumCellGlobal()
            nNode = mesh.NumNodeGlobal()
            if mpi.rank == 0:
                print(
                    str.format(
                        "Mesh Direct Bisect {} done, nCell [{}], nNode [{}]",
                        iter,
                        nCell,
                        nNode,
                    )
                )
    elif readMeshMode == "Parallel":
        meshOutName = (
            meshFile
            + "_part_"
            + f"{mpi.size}"
            + ("_elevated" if meshElevation == "O2" else "")
            + (f"_bisect{meshDirectBisect}" if meshDirectBisect > 0 else "")
        )
        meshOutNameMod, meshPartPath = serializerFactory.ModifyFilePath(
            meshOutName, mpi, "part_%d", True
        )
        serializer = serializerFactory.BuildSerializer(mpi)
        if mpi.rank == 0:
            print(f"Mesh reader: to read via [{serializerFactory.to_dict()['type']}]")
        serializer.OpenFile(meshPartPath, True)
        mesh.ReadSerialize(serializer, "meshPart")
        serializer.CloseFile()

        mesh.RecoverNode2CellAndNode2Bnd()
        mesh.RecoverCell2CellAndBnd2Cell()
        mesh.BuildGhostPrimary()
        mesh.AdjGlobal2LocalPrimary()
        mesh.AdjGlobal2LocalN2CB()
    elif readMeshMode == "Distributed":
        # Read from H5 with even-split distribution + ParMetis repartition.
        # Works with any number of MPI ranks, regardless of how the file was written.
        meshOutName = (
            meshFile
            + "_part_"
            + f"{mpi.size}"
            + ("_elevated" if meshElevation == "O2" else "")
            + (f"_bisect{meshDirectBisect}" if meshDirectBisect > 0 else "")
        )
        meshOutNameMod, meshPartPath = serializerFactory.ModifyFilePath(
            meshOutName, mpi, "part_%d", True
        )
        serializer = serializerFactory.BuildSerializer(mpi)
        if mpi.rank == 0:
            print(f"Mesh reader: distributed read via [{serializerFactory.to_dict()['type']}]")
        serializer.OpenFile(meshPartPath, True)
        mesh.ReadSerializeAndDistribute(
            serializer,
            "meshPart",
            {
                "metisType": "KWAY",
                "metisUfactor": 5,
                "metisSeed": 0,
                "metisNcuts": 3,
            },
        )
        serializer.CloseFile()

        mesh.RecoverNode2CellAndNode2Bnd()
        mesh.RecoverCell2CellAndBnd2Cell()
        mesh.BuildGhostPrimary()
        mesh.AdjGlobal2LocalPrimary()
        mesh.AdjGlobal2LocalN2CB()
    else:
        raise ValueError(f"Unknown readMeshMode: {readMeshMode}")

    mesh.ReorderLocalCells(nParts=inner_process_parts, nPartsInner=second_level_parts)

    mesh.InterpolateFace()
    mesh.AssertOnFaces()

    mesh.AdjLocal2GlobalN2CB()
    mesh.BuildGhostN2CB()
    mesh.AdjGlobal2LocalN2CB()
    print(str.format("{}, NumBndGhost {}", mpi.rank, mesh.NumBndGhost()))

    # TODO: elevation smoothing
    """
    if (config.dataIOControl.meshElevation == 1 && config.dataIOControl.readMeshMode == 0)
    {
        mesh->elevationInfo.nIter = config.dataIOControl.meshElevationIter;
        mesh->elevationInfo.nSearch = config.dataIOControl.meshElevationNSearch;
        mesh->elevationInfo.RBFRadius = config.dataIOControl.meshElevationRBFRadius;
        mesh->elevationInfo.RBFPower = config.dataIOControl.meshElevationRBFPower;
        mesh->elevationInfo.kernel = config.dataIOControl.meshElevationRBFKernel;
        mesh->elevationInfo.MaxIncludedAngle = config.dataIOControl.meshElevationMaxIncludedAngle;
        mesh->elevationInfo.refDWall = config.dataIOControl.meshElevationRefDWall;
        mesh->ElevatedNodesGetBoundarySmooth(
            [&](Geom::t_index bndId)
            {
                auto bType = pBCHandler->GetTypeFromID(bndId);
                if (bType == BCWall || bType == BCWallIsothermal)
                    return true;
                if (config.dataIOControl.meshElevationBoundaryMode == 1 &&
                    (bType == BCWallInvis || bType == BCSym))
                    return true;
                return false;
            });
        if (config.dataIOControl.meshElevationInternalSmoother == 0)
            mesh->ElevatedNodesSolveInternalSmooth();
        else if (config.dataIOControl.meshElevationInternalSmoother == 1)
            mesh->ElevatedNodesSolveInternalSmoothV1();
        else if (config.dataIOControl.meshElevationInternalSmoother == 2)
            mesh->ElevatedNodesSolveInternalSmoothV2();
        else if (config.dataIOControl.meshElevationInternalSmoother == -1)
        {
            if (mpi.rank == 0)
                log() << " WARNING !!! Not Smoothing internal, abandoning boundary smooth displacements" << std::endl;
        }
        else
            DNDS_assert(false);
    }
    """

    if outPltMode == "Serial":
        mesh.AdjLocal2GlobalPrimary()
        meshReader.BuildSerialOut()
        mesh.AdjGlobal2LocalPrimary()

    mesh.RecreatePeriodicNodes()
    mesh.BuildVTKConnectivity()

    return mesh, meshReader, name2ID


def create_bnd_mesh(mesh: Geom.UnstructuredMesh, outPltMode: str = "Serial"):
    meshBnd = Geom.UnstructuredMesh(mesh.getMPI(), mesh.getDim() - 1)
    readerBnd = Geom.UnstructuredMeshSerialRW(meshBnd, 0)
    mesh.ConstructBndMesh(meshBnd)
    if outPltMode == "Serial":
        meshBnd.AdjLocal2GlobalPrimaryForBnd()
        readerBnd.BuildSerialOut()
        meshBnd.AdjGlobal2LocalPrimaryForBnd()
    meshBnd.RecreatePeriodicNodes()
    meshBnd.BuildVTKConnectivity()

    return meshBnd, readerBnd

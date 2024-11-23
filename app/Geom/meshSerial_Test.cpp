#include "Geom/Mesh.hpp"
#include "stdio.h"
#include "unistd.h"

std::vector<double> argD;

/*
running:
valgrind --log-file=log_valgrind.log
*/

void testCGNS()
{
    auto mpi = DNDS::MPIInfo();
    mpi.setWorld();
    // DNDS::Debug::MPIDebugHold(mpi);
    char buf[512];
    // std::cout << getcwd(buf, 512) << std::endl;
    int dim = 2;
    auto mesh = std::make_shared<DNDS::Geom::UnstructuredMesh>(mpi, dim);
    auto reader = DNDS::Geom::UnstructuredMeshSerialRW(mesh, 0);
    // "../data/mesh/FourTris_V1.pw.cgns"
    // "../data/mesh/SC20714_MixedA.cgns"
    // "../data/mesh/UniformDM240_E120.cgns"
    // "../data/mesh/Ball.cgns"
    // "../data/mesh/Uniform32_Periodic.cgns"
    // "../data/mesh/Uniform128.cgns"
    reader.ReadFromCGNSSerial("../data/mesh/Uniform32_Periodic.cgns");
    reader.Deduplicate1to1Periodic();
    reader.BuildCell2Cell();
    reader.MeshPartitionCell2Cell(DNDS::Geom::UnstructuredMeshSerialRW::PartitionOptions{});
    reader.PartitionReorderToMeshCell2Cell();
    reader.BuildSerialOut();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->InterpolateFace();
    // mesh->AssertOnFaces();

    auto meshBnd = std::make_shared<DNDS::Geom::UnstructuredMesh>(mpi, dim - 1);
    auto readerBnd = DNDS::Geom::UnstructuredMeshSerialRW(meshBnd, 0);
    mesh->ConstructBndMesh(*meshBnd);
    meshBnd->AdjLocal2GlobalPrimaryForBnd();
    readerBnd.BuildSerialOut();
    meshBnd->AdjGlobal2LocalPrimaryForBnd();

    for (int i = 0; i < 4; i++)
    {
        mesh->AdjLocal2GlobalFacial();
        mesh->AdjGlobal2LocalFacial();
        mesh->AdjLocal2GlobalPrimary();
        mesh->AdjGlobal2LocalPrimary(); // test on the ptr mapping completeness
    }

    mesh->TransformCoords(
        [&](const DNDS::Geom::tPoint p)
        { return p * 2; });

    reader.PrintSerialPartPltBinaryDataArray(
        "../data/out/debug",
        1, 1,
        [](int)
        { return "vCell"; },
        [](int, DNDS::index)
        { return 1; },
        [](int)
        { return "vPoint"; },
        [](int, DNDS::index)
        { return 1; },
        0.0,
        0);
    reader.PrintSerialPartPltBinaryDataArray(
        "../data/out/debug",
        1, 1,
        [](int)
        { return "vCell"; },
        [](int, DNDS::index)
        { return 1; },
        [](int)
        { return "vPoint"; },
        [](int, DNDS::index)
        { return 1; },
        0.0,
        1);
    reader.PrintSerialPartVTKDataArray(
        "../data/out/debug",
        "../data/out/debug",
        1, 1, 1, 1,
        [](int)
        { return "aScalar"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVector"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        [](int)
        { return "aScalarP"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVectorP"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        0.0,
        0);
    reader.PrintSerialPartVTKDataArray(
        "../data/out/debug",
        "../data/out/debug",
        1, 1, 1, 1,
        [](int)
        { return "aScalar"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVector"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        [](int)
        { return "aScalarP"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVectorP"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        0.0,
        1);

    readerBnd.PrintSerialPartPltBinaryDataArray(
        "../data/out/debugBnd",
        1, 1,
        [](int)
        { return "vCell"; },
        [](int, DNDS::index)
        { return 1; },
        [](int)
        { return "vPoint"; },
        [](int, DNDS::index)
        { return 1; },
        0.0,
        0);
    readerBnd.PrintSerialPartPltBinaryDataArray(
        "../data/out/debugBnd",
        1, 1,
        [](int)
        { return "vCell"; },
        [](int, DNDS::index)
        { return 1; },
        [](int)
        { return "vPoint"; },
        [](int, DNDS::index)
        { return 1; },
        0.0,
        0);
    readerBnd.PrintSerialPartVTKDataArray(
        "../data/out/debugBnd",
        "../data/out/debugBnd",
        1, 1, 1, 1,
        [](int)
        { return "aScalar"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVector"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        [](int)
        { return "aScalarP"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVectorP"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        0.0,
        0);
    readerBnd.PrintSerialPartVTKDataArray(
        "../data/out/debugBnd",
        "../data/out/debugBnd",
        1, 1, 1, 1,
        [](int)
        { return "aScalar"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVector"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        [](int)
        { return "aScalarP"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVectorP"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        0.0,
        1);
    // char c;
    // std::cin >> c;
}

int main(int argc, char *argv[])
{
    // ! Disable MPI call to help serial mem check
    MPI_Init(&argc, &argv);

    for (int i = 1; i < argc; i++)
    {
        double v = std::atof(argv[i]);
        argD.push_back(v);
    }
    testCGNS();

    MPI_Finalize();

    return 0;
}
#include "Geom/Mesh.hpp"
#include "DNDS/SerializerJSON.hpp"
#include "stdio.h"

#include <filesystem>

std::vector<double> argD;

/*
running: with openmpi:
mpirun --mca threads_pthreads_yield_strategy nanosleep --mca threads_pthreads_nanosleep_time 1000000000  --ove
rsubscribe -np 512 test/partitionMeshSerial.exe
*/

void doPartitioning(const std::string &meshName, int dim)
{
    auto mpi = DNDS::MPIInfo();
    mpi.setWorld();
    // DNDS::Debug::MPIDebugHold(mpi);
    char buf[512];
    // std::cout << getcwd(buf, 512) << std::endl;
    auto mesh = std::make_shared<DNDS::Geom::UnstructuredMesh>(mpi, dim);
    auto reader = DNDS::Geom::UnstructuredMeshSerialRW(mesh, 0);
    // "../data/mesh/FourTris_V1.pw.cgns"
    // "../data/mesh/SC20714_MixedA.cgns"
    // "../data/mesh/UniformDM240_E120.cgns"
    // "../data/mesh/Ball.cgns"
    mesh->periodicInfo.translation[3] *= 10;
    reader.ReadFromCGNSSerial(meshName);
    reader.Deduplicate1to1Periodic();

    reader.BuildCell2Cell();
    // MPI_Request bReq{MPI_REQUEST_NULL};
    // MPI_Ibarrier(mpi.comm, &bReq);
    // bool barrierOk = false;
    // MPI_Start(&bReq);
    // while (!barrierOk)
    // {
    //     int flag{0};
    //     MPI_Status stat;
    //     MPI_Test(&bReq, &flag, &stat);
    //     barrierOk = flag != 0;
    //     // sleep(5);
    // }

    reader.MeshPartitionCell2Cell(DNDS::Geom::UnstructuredMeshSerialRW::PartitionOptions{});
    reader.PartitionReorderToMeshCell2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();

    /**************************/
    // about serializerPP:
    std::filesystem::path meshPath{meshName};
    auto meshOutName = std::string(meshName) + "_part_" + std::to_string(mpi.size) + ".dir";
    std::filesystem::path meshOutDir{meshOutName};
    std::filesystem::create_directories(meshOutDir);
    std::string meshPartPath = DNDS::getStringForcePath(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));

    DNDS::SerializerJSON serializerJSON;
    serializerJSON.SetUseCodecOnUint8(true);
    DNDS::Serializer::SerializerBaseSSP serializerP = &serializerJSON;

    serializerP->OpenFile(meshPartPath, false);
    mesh->WriteSerialize(serializerP, "meshPart");
    serializerPP->CloseFile();

    MPISerialDo(mpi, [&]()
                { DNDS::log() << "    Rank: " << mpi.rank << " nCell " << mesh->NumCell() << " nCellGhost " << mesh->NumCellGhost() << std::endl; });
    MPISerialDo(mpi, [&]()
                { DNDS::log() << "    Rank: " << mpi.rank << " nNode " << mesh->NumNode() << " nNodeGhost " << mesh->NumNodeGhost() << std::endl; });

    /**************************/
}

int main(int argc, char *argv[])
{
    // ! Disable MPI call to help serial mem check
    MPI_Init(&argc, &argv);
    DNDS_assert_info(argc == 3, "need 2 arguments of [mesh file] and [dim]");

    doPartitioning(argv[1], std::stoi(argv[2]));

    MPI_Finalize();

    return 0;
}
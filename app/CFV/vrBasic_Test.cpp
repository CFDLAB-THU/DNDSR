#include "CFV/VariationalReconstruction.hpp"
#include "CFV/VariationalReconstruction_Reconstruction.hxx"
#include "CFV/VariationalReconstruction_LimiterProcedure.hxx"

#include <cstdlib>
#include <omp.h>

#include "DNDS/SerializerJSON.hpp"

#include <filesystem>

std::vector<double> argD;

/*
running:
valgrind --log-file=log_valgrind.log
*/

void testConstruct()
{
    // "../data/mesh/FourTris_V1.pw.cgns"
    // "../data/mesh/SC20714_MixedA.cgns"
    // "../data/mesh/UniformDM240_E120.cgns"
    // "../data/mesh/Ball.cgns"
    // "../data/mesh/Uniform32_Periodic.cgns"
    // "../data/mesh/RotPeriodicA.cgns";
    auto meshName = "../data/mesh/Uniform32_Periodic.cgns";
    static const int dim = 2;

    auto mpi = DNDS::MPIInfo();
    mpi.setWorld();
    // DNDS::Debug::MPIDebugHold(mpi);
    char buf[512];
    // std::cout << getcwd(buf, 512) << std::endl;
    auto mesh = std::make_shared<DNDS::Geom::UnstructuredMesh>(mpi, dim);
    auto reader = DNDS::Geom::UnstructuredMeshSerialRW(mesh, 0);
    
    reader.ReadFromCGNSSerial(meshName);
    reader.Deduplicate1to1Periodic();
    reader.BuildCell2Cell();
    reader.MeshPartitionCell2Cell();
    reader.PartitionReorderToMeshCell2Cell();
    reader.BuildSerialOut();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->InterpolateFace();
    mesh->AssertOnFaces();

    /**************************/
    // about serializer: test the coherence of mesh serializer
    std::filesystem::path meshPath{meshName};
    auto meshOutName = std::string(meshName) + "_part_" + std::to_string(mpi.size) + ".dir";
    std::filesystem::path meshOutDir{meshOutName};
    std::filesystem::create_directories(meshOutDir);
    std::string meshPartPath = DNDS::getStringForcePath(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));

    DNDS::SerializerJSON serializerJSON;
    serializerJSON.SetUseCodecOnUint8(true);
    DNDS::SerializerBase *serializer = &serializerJSON;

    serializer->OpenFile(meshPartPath, false);
    mesh->WriteSerialize(serializer, "meshPart");
    serializer->CloseFile();

    serializer->OpenFile(meshPartPath, true);
    mesh->ReadSerialize(serializer, "meshPart");
    serializer->CloseFile();

    /**************************/
    mesh->InterpolateFace();
    mesh->AssertOnFaces();

    auto vr = DNDS::CFV::VariationalReconstruction<dim>(mpi, mesh);
#ifdef DNDS_USE_OMP
    omp_set_num_threads(DNDS::MPIWorldSize() == 1 ? std::min(omp_get_num_procs(), omp_get_max_threads()) : 1);
#endif
    // omp_set_num_threads(1);
    vr.ConstructMetrics();
    vr.ConstructBaseAndWeight();
    vr.ConstructRecCoeff();
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

    testConstruct();

    MPI_Finalize();

    return 0;
}
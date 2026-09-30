#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh_Elevation_SmoothHelpers.hpp"
#include <filesystem>

using namespace DNDS;
using namespace DNDS::Geom;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context(argc, argv);
    int result = context.run();
    MPI_Finalize();
    return result;
}

static ssp<UnstructuredMesh> ReadSquare(MPIInfo mpi)
{
    auto mesh = std::make_shared<UnstructuredMesh>(mpi, 2);
    UnstructuredMeshSerialRW reader(mesh, 0);
    auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    reader.ReadFromCGNSSerial((root / "data/mesh/UniformSquare_10.cgns").string());
    reader.BuildCell2Cell();
    UnstructuredMeshSerialRW::PartitionOptions partition;
    partition.metisType = "KWAY";
    partition.metisSeed = 42;
    partition.metisNcuts = 1;
    reader.MeshPartitionCell2Cell(partition);
    reader.PartitionReorderToMeshCell2Cell();
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->AdjGlobal2LocalN2CB();
    mesh->InterpolateFace();
    return mesh;
}

TEST_CASE("Audit batch 2: wall quadrature preserves square distances")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = ReadSquare(mpi);
    double lower[2] = {1e30, 1e30}, upper[2] = {-1e30, -1e30};
    for (DNDS::index i = 0; i < mesh->NumNode(); ++i)
        for (int d = 0; d < 2; ++d)
        {
            lower[d] = std::min(lower[d], mesh->coords[i](d));
            upper[d] = std::max(upper[d], mesh->coords[i](d));
        }
    MPI_Allreduce(MPI_IN_PLACE, lower, 2, MPI_DOUBLE, MPI_MIN, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, upper, 2, MPI_DOUBLE, MPI_MAX, mpi.comm);
    UnstructuredMesh::WallDistOptions options;
    options.method = 1;
    options.minWallDist = 0;
    options.verbose = 0;
    for (int method : {0, 1, 20})
    {
        options.method = method;
        mesh->BuildNodeWallDist([](auto)
                                { return true; }, options);
        for (DNDS::index i = 0; i < mesh->nodeWallDist.Size(); ++i)
        {
            const auto p = mesh->coords[i];
            const double expected = std::min({p(0) - lower[0], upper[0] - p(0), p(1) - lower[1], upper[1] - p(1)});
            CHECK(mesh->nodeWallDist[i].allFinite());
            CHECK(mesh->nodeWallDist[i].norm() == doctest::Approx(expected).epsilon(1e-11));
        }
        mesh->BuildNodeWallDist([](auto)
                                { return false; }, options);
        for (DNDS::index i = 0; i < mesh->nodeWallDist.Size(); ++i)
            CHECK(mesh->nodeWallDist[i].norm() == std::pow(DNDS::veryLargeReal, 0.25));
    }
}

TEST_CASE("Audit batch 2: both smoothing units share helper contracts")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    auto linear = ReadSquare(mpi);
    auto mesh = std::make_shared<UnstructuredMesh>(mpi, 2);
    mesh->BuildO2FromO1Elevation(*linear);
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->AdjGlobal2LocalN2CB();
    mesh->InterpolateFace();
    mesh->ElevatedNodesGetBoundarySmooth([](auto)
                                         { return false; });
    // Exercise both linked implementations' common no-displacement contract.
    // A finite-iteration smoothing algorithm comparison is outside this test.
    CHECK(mesh->nTotalMoved == 0);
    CHECK_NOTHROW(mesh->ElevatedNodesSolveInternalSmooth());
    CHECK_NOTHROW(mesh->ElevatedNodesSolveInternalSmoothV2());
    CoordPairDOF values;
    values.InitPair("audit smooth helper", mpi);
    values.father->Resize(1);
    values.son->Resize(0);
    values.setConstant(2);
    CHECK(values.dot(values) == doctest::Approx(12 * mpi.size));
    PointCloudKDTreeCoordPair cloud(values.father);
    CHECK(cloud.kdtree_get_point_count() == 1);
    CHECK(cloud.kdtree_get_pt(0, 2) == 2);
}

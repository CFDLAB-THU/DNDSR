/**
 * @file test_MeshCGNSMultiZone.cpp
 * @brief Tests for CGNS multizone node deduplication.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/IO/CGNS.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace DNDS;
using namespace DNDS::Geom;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();
    MPI_Finalize();
    return res;
}

static std::filesystem::path repoRoot()
{
    std::filesystem::path p(__FILE__);
    for (int i = 0; i < 4; i++)
        p = p.parent_path();
    return p;
}

static void generateMultiblockMeshes(const std::string &directory)
{
    const auto script = repoRoot() / "scripts" / "generate_multiblock_cgns.py";
    REQUIRE(std::filesystem::exists(script));
    // CTest supplies the configured CGNS library. Generate in the test's
    // working directory so shared input meshes are never overwritten.
    const std::string command = "python3 \"" + script.string() + "\" --output " + directory + " --blocks 2 3";
    int ret = std::system(command.c_str());
    REQUIRE(ret == 0);
}

static void checkGeneratedOneSidedBlocks(const std::filesystem::path &directory, int blocks)
{
    auto mpi = MPIInfo();
    mpi.setWorld();
    REQUIRE(mpi.size == 1);

    std::string name = fmt::format("GeneratedMultiBlock{}x{}_OneSided.cgns", blocks, blocks);
    auto serial = ReadNativeCGNSSerial((directory / name).string(), 2, FBC_Name_2_ID_Default);

    CHECK(serial.cellNodes.RowCount() == blocks * blocks);
    CHECK(serial.coordinates.RowCount() == (blocks + 1) * (blocks + 1));

    // Every assembled node coordinate should be initialized exactly once after
    // union deduplication of the one-sided block interfaces.
    auto coordinates = serial.coordinates.Read();
    for (size_t iNode = 0; iNode < serial.coordinates.RowCount(); iNode++)
    {
        CHECK(!DNDS::IsUnInitReal(coordinates.View()(iNode, 0)));
        CHECK(!DNDS::IsUnInitReal(coordinates.View()(iNode, 1)));
    }
}

TEST_CASE("CGNS multizone one-sided connectivity deduplicates 2x2 and 3x3 blocks")
{
    const auto directory = fmt::format("cgns-multizone-{}", std::chrono::steady_clock::now().time_since_epoch().count());
    REQUIRE(std::filesystem::create_directory(directory));
    generateMultiblockMeshes(directory);
    checkGeneratedOneSidedBlocks(directory, 2);
    checkGeneratedOneSidedBlocks(directory, 3);
    std::filesystem::remove_all(directory);
}

TEST_CASE("Audit regression: CGNS single-entry boundary PointList")
{
    const auto path = std::filesystem::current_path() / "audit_single_point_list.cgns";
    int file, base, zone, coord, section, bc;
    REQUIRE(cg_open(path.c_str(), CG_MODE_WRITE, &file) == CG_OK);
    REQUIRE(cg_base_write(file, "Base", 2, 2, &base) == CG_OK);
    cgsize_t sizes[3] = {4, 1, 0};
    REQUIRE(cg_zone_write(file, base, "Zone", sizes, Unstructured, &zone) == CG_OK);
    double x[4] = {0, 1, 1, 0}, y[4] = {0, 0, 1, 1};
    REQUIRE(cg_coord_write(file, base, zone, RealDouble, "CoordinateX", x, &coord) == CG_OK);
    REQUIRE(cg_coord_write(file, base, zone, RealDouble, "CoordinateY", y, &coord) == CG_OK);
    cgsize_t quad[4] = {1, 2, 3, 4};
    cgsize_t edges[8] = {1, 2, 2, 3, 3, 4, 4, 1};
    REQUIRE(cg_section_write(file, base, zone, "Cells", QUAD_4, 1, 1, 0, quad, &section) == CG_OK);
    REQUIRE(cg_section_write(file, base, zone, "Edges", BAR_2, 2, 5, 0, edges, &section) == CG_OK);
    cgsize_t single[1] = {2}, rest[2] = {3, 5};
    REQUIRE(cg_boco_write(file, base, zone, "WALL", BCWall, PointList, 1, single, &bc) == CG_OK);
    REQUIRE(cg_boco_gridlocation_write(file, base, zone, bc, EdgeCenter) == CG_OK);
    REQUIRE(cg_boco_write(file, base, zone, "FAR", BCFarfield, PointRange, 2, rest, &bc) == CG_OK);
    REQUIRE(cg_boco_gridlocation_write(file, base, zone, bc, EdgeCenter) == CG_OK);
    REQUIRE(cg_close(file) == CG_OK);
    MPIInfo mpi;
    mpi.setWorld();
    auto serial = ReadNativeCGNSSerial(path.string(), 2, FBC_Name_2_ID_Default);
    REQUIRE(serial.cellNodes.RowCount() == 1);
    REQUIRE(serial.boundaryNodes.RowCount() == 4);
    int walls = 0, farfields = 0;
    auto info = serial.boundaryInfo.Read();
    for (size_t i = 0; i < serial.boundaryInfo.RowCount(); ++i)
    {
        auto code = info.View()(i, 0).zone;
        walls += code == BC_ID_DEFAULT_WALL;
        farfields += code == BC_ID_DEFAULT_FAR;
    }
    CHECK(walls == 1);
    CHECK(farfields == 3);
    std::filesystem::remove(path);
}

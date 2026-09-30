/**
 * @file test_MeshCGNSMultiZone.cpp
 * @brief Tests for CGNS multizone node deduplication.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/CGNS.hpp"

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

static std::filesystem::path meshPath(const std::string &name)
{
    return repoRoot() / "data" / "mesh" / name;
}

static void ensureGeneratedMultiblockMeshes()
{
    auto root = repoRoot();
    auto script = root / "scripts" / "generate_multiblock_cgns.py";
    auto out = root / "data" / "mesh";
    auto lib = root / "external" / "cfd_externals" / "install" / "lib" / "libcgns.so";
    REQUIRE(std::filesystem::exists(script));
    REQUIRE(std::filesystem::exists(lib));

    std::string libPath = (root / "external" / "cfd_externals" / "install" / "lib").string();
    std::string command =
        "LD_LIBRARY_PATH=\"" + libPath + ":$LD_LIBRARY_PATH\" "
                                         "python3 \"" +
        script.string() + "\" --output \"" + out.string() + "\" --blocks 2 3";
    int ret = std::system(command.c_str());
    REQUIRE(ret == 0);
}

static void checkGeneratedOneSidedBlocks(int blocks)
{
    auto mpi = MPIInfo();
    mpi.setWorld();
    REQUIRE(mpi.size == 1);

    std::string name = fmt::format("GeneratedMultiBlock{}x{}_OneSided.cgns", blocks, blocks);
    auto mesh = make_ssp<UnstructuredMesh>(mpi, 2);
    UnstructuredMeshSerialRW reader(mesh, 0);
    reader.ReadFromCGNSSerial(meshPath(name).string());

    CHECK(reader.cell2nodeSerial->Size() == blocks * blocks);
    CHECK(reader.coordSerial->Size() == (blocks + 1) * (blocks + 1));

    // Every assembled node coordinate should be initialized exactly once after
    // union deduplication of the one-sided block interfaces.
    for (DNDS::index iNode = 0; iNode < reader.coordSerial->Size(); iNode++)
    {
        CHECK(!DNDS::IsUnInitReal((*reader.coordSerial)[iNode](0)));
        CHECK(!DNDS::IsUnInitReal((*reader.coordSerial)[iNode](1)));
    }
}

TEST_CASE("CGNS multizone one-sided connectivity deduplicates 2x2 and 3x3 blocks")
{
    ensureGeneratedMultiblockMeshes();
    checkGeneratedOneSidedBlocks(2);
    checkGeneratedOneSidedBlocks(3);
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
    auto mesh = make_ssp<UnstructuredMesh>(mpi, 2);
    UnstructuredMeshSerialRW reader(mesh, 0);
    reader.ReadFromCGNSSerial(path.string());
    REQUIRE(reader.cell2nodeSerial->Size() == 1);
    REQUIRE(reader.bnd2nodeSerial->Size() == 4);
    int walls = 0, farfields = 0;
    for (DNDS::index i = 0; i < reader.bndElemInfoSerial->Size(); ++i)
    {
        auto code = (*reader.bndElemInfoSerial)(i, 0).zone;
        walls += code == BC_ID_DEFAULT_WALL;
        farfields += code == BC_ID_DEFAULT_FAR;
    }
    CHECK(walls == 1);
    CHECK(farfields == 3);
    std::filesystem::remove(path);
}

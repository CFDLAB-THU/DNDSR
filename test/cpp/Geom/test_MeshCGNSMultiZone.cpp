/**
 * @file test_MeshCGNSMultiZone.cpp
 * @brief Tests for CGNS multizone node deduplication.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"

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

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/IO/CGNS.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Access/EntityField.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    struct Oracle
    {
        std::vector<std::array<DNDS::real, 3>> points;
        std::vector<std::vector<I>> cells, boundaries;
    };

    Oracle Expected(bool multizone)
    {
        if (multizone)
            return {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {2, 0, 0}, {2, 1, 0}},
                    {{0, 1, 2, 3}, {1, 4, 5, 2}},
                    {{0, 1}, {3, 0}, {2, 3}, {1, 4}, {4, 5}, {5, 2}}};
        return {{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}},
                {{0, 1, 4, 3}, {1, 2, 5, 4}},
                {{0, 1}, {3, 0}, {4, 3}, {1, 2}, {2, 5}, {5, 4}}};
    }

    void CG(int result)
    {
        if (result != CG_OK)
            throw std::runtime_error(cg_get_error());
    }

    void WriteFixture(const std::string &path, bool multizone)
    {
        int file = -1;
        CG(cg_open(path.c_str(), CG_MODE_WRITE, &file));
        struct Close
        {
            int file;
            ~Close() { cg_close(file); }
        } close{file};
        int base;
        CG(cg_base_write(file, "Base", 2, 2, &base));
        auto zone = [&](const char *name, const std::vector<double> &x, const std::vector<double> &y,
                        const std::vector<cgsize_t> &cells, const std::vector<cgsize_t> &edges,
                        bool wall)
        {
            int result, coordinate, section, boundary;
            const cgsize_t count = cgsize_t(cells.size() / 4);
            cgsize_t sizes[]{cgsize_t(x.size()), count, 0};
            CG(cg_zone_write(file, base, name, sizes, Unstructured, &result));
            CG(cg_coord_write(file, base, result, RealDouble, "CoordinateX", x.data(), &coordinate));
            CG(cg_coord_write(file, base, result, RealDouble, "CoordinateY", y.data(), &coordinate));
            CG(cg_section_write(file, base, result, "Cells", QUAD_4, 1, count, 0, cells.data(), &section));
            CG(cg_section_write(file, base, result, "Exterior", BAR_2, count + 1,
                                count + cgsize_t(edges.size() / 2), 0, edges.data(), &section));
            if (wall)
            {
                const cgsize_t single[]{count + 1};
                CG(cg_boco_write(file, base, result, "WALL_SINGLE", BCWall, PointList, 1, single, &boundary));
                CG(cg_boco_gridlocation_write(file, base, result, boundary, EdgeCenter));
            }
            const cgsize_t remaining[]{count + 1 + int(wall), count + cgsize_t(edges.size() / 2)};
            CG(cg_boco_write(file, base, result, "FAR_REST", BCFarfield, PointRange, 2, remaining, &boundary));
            CG(cg_boco_gridlocation_write(file, base, result, boundary, EdgeCenter));
            return result;
        };
        if (multizone)
        {
            const int left = zone("Left", {0, 1, 1, 0}, {0, 0, 1, 1}, {1, 2, 3, 4}, {1, 2, 4, 1, 3, 4}, true);
            zone("Right", {1, 2, 2, 1}, {0, 0, 1, 1}, {1, 2, 3, 4}, {1, 2, 2, 3, 3, 4}, false);
            // Deliberately no reciprocal Right-to-Left connectivity.
            const cgsize_t local[]{2, 3}, donor[]{1, 4};
            int connection;
            CG(cg_conn_write(file, base, left, "LeftToRight", Vertex, Abutting1to1, PointList, 2, local,
                             "Right", Unstructured, PointListDonor,
                             sizeof(cgsize_t) == 8 ? LongInteger : Integer, 2, donor, &connection));
        }
        else
            zone("Whole", {0, 1, 2, 0, 1, 2}, {0, 0, 0, 1, 1, 1},
                 {1, 2, 5, 4, 2, 3, 6, 5}, {1, 2, 4, 1, 5, 4, 2, 3, 3, 6, 6, 5}, true);
    }

    struct Fixture
    {
        DNDS::MPIInfo mpi;
        std::string path;

        Fixture(const DNDS::MPIInfo &mpiIn, bool multizone) : mpi(mpiIn)
        {
            long long pid = getpid();
            MPI_Bcast(&pid, 1, MPI_LONG_LONG, 0, mpi.comm);
            path = (std::filesystem::temp_directory_path() /
                    ("dndsr-native-cgns-" + std::to_string(pid) + (multizone ? "-multi.cgns" : "-single.cgns")))
                       .string();
            int ready = 1;
            if (mpi.rank == 0)
                try
                {
                    WriteFixture(path, multizone);
                }
                catch (const std::exception &error)
                {
                    INFO(error.what());
                    CHECK(false);
                    ready = 0;
                }
            MPI_Bcast(&ready, 1, MPI_INT, 0, mpi.comm);
            REQUIRE(ready == 1);
        }

        ~Fixture()
        {
            MPI_Barrier(mpi.comm);
            if (mpi.rank == 0)
                std::filesystem::remove(path);
        }
    };

    G::t_FBCName_2_ID BoundaryIds(int *calls = nullptr)
    {
        return [calls](const std::string &name) -> G::t_index
        {
            if (calls)
                ++*calls;
            if (name == "WALL_SINGLE")
                return 21;
            if (name == "FAR_REST")
                return 22;
            throw std::runtime_error("unexpected fixture boundary name: " + name);
        };
    }

    void CheckSerial(const G::NativeSerialMeshData &mesh, const Oracle &expected)
    {
        CHECK(mesh.coordinates.RowCount() == 6);
        CHECK(mesh.cellNodes.RowCount() == 2);
        CHECK(mesh.boundaryNodes.RowCount() == 6);
        auto coordinates = mesh.coordinates.Read();
        auto cells = mesh.cellNodes.Read();
        auto boundaries = mesh.boundaryNodes.Read();
        auto owners = mesh.boundaryCells.Read();
        auto cellInfo = mesh.cellInfo.Read();
        auto boundaryInfo = mesh.boundaryInfo.Read();
        auto cellIds = mesh.cellIds.Read();
        auto nodeIds = mesh.nodeIds.Read();
        auto boundaryIds = mesh.boundaryIds.Read();
        for (size_t row = 0; row < 6; ++row)
        {
            for (size_t component = 0; component < 3; ++component)
                CHECK(coordinates.View()(row, component) == expected.points[row][component]);
            CHECK(nodeIds.View()(row, 0) == I(row));
            const auto cone = boundaries.View()[row];
            CHECK(std::vector<I>(cone.begin(), cone.end()) == expected.boundaries[row]);
            CHECK(owners.View()(row, 0) == I(row / 3));
            CHECK(owners.View()(row, 1) == DNDS::UnInitIndex);
            CHECK(boundaryInfo.View()(row, 0).getElemType() == G::Elem::Line2);
            CHECK(boundaryInfo.View()(row, 0).zone == (row == 0 ? 21 : 22));
            CHECK(boundaryIds.View()(row, 0) == I(row));
        }
        for (size_t row = 0; row < 2; ++row)
        {
            const auto cone = cells.View()[row];
            CHECK(std::vector<I>(cone.begin(), cone.end()) == expected.cells[row]);
            CHECK(cellInfo.View()(row, 0).getElemType() == G::Elem::Quad4);
            CHECK(cellInfo.View()(row, 0).zone == G::BC_ID_INTERNAL);
            CHECK(cellIds.View()(row, 0) == I(row));
        }
    }

    std::vector<I> Decode(const G::EntitySpace &space, G::MeshRowSpan<I> row)
    {
        std::vector<I> result;
        for (I value : row)
            result.push_back(value == DNDS::UnInitIndex ? value : value >= 0 ? space.LocalToGlobal(value)
                                                                             : -1 - value);
        return result;
    }

    void CheckState(const std::shared_ptr<const G::NativeMeshState> &state, const DNDS::MPIInfo &mpi, const Oracle &expected)
    {
        const auto nodes = state->Entities(G::EntityKind::Node);
        const auto cells = state->Entities(G::EntityKind::Cell);
        const auto boundaries = state->Entities(G::EntityKind::Bnd);
        auto facade = G::UnstructuredMesh::FromNative(mpi, state);
        CHECK(facade->HasNativeState());
        auto read = facade->AcquireRead(state->Requirements());
        const auto view = read.HostView();
        for (const auto &space : {nodes, cells, boundaries})
        {
            const I total = space.Kind() == G::EntityKind::Cell ? 2 : 6;
            CHECK(space.NumOwned() == total / mpi.size + (mpi.rank < total % mpi.size));
            for (I row = 0; row < space.NumLocal(); ++row)
                CHECK(state->OriginalEntityId(space.Kind(), row) == space.LocalToGlobal(row));
        }
        for (I row = 0; row < cells.NumLocal(); ++row)
        {
            const I global = cells.LocalToGlobal(row);
            CHECK(Decode(nodes, view.CellNodes(row)) == expected.cells.at(size_t(global)));
            CHECK(Decode(cells, view.CellCells(row)) == std::vector<I>{1 - global});
            CHECK(view.GetCellInfo(row).getElemType() == G::Elem::Quad4);
            CHECK(view.GetCellInfo(row).zone == G::BC_ID_INTERNAL);
        }
        for (I row = 0; row < boundaries.NumLocal(); ++row)
        {
            const I global = boundaries.LocalToGlobal(row);
            CHECK(Decode(nodes, view.BoundaryNodes(row)) == expected.boundaries.at(size_t(global)));
            CHECK(Decode(cells, view.BoundaryCells(row)) == std::vector<I>{global / 3, DNDS::UnInitIndex});
            CHECK(view.GetBndInfo(row).zone == (global == 0 ? 21 : 22));
        }
        for (I row = 0; row < nodes.NumLocal(); ++row)
        {
            const I global = nodes.LocalToGlobal(row);
            for (size_t component = 0; component < 3; ++component)
                CHECK(view.Coord(row)[component] == expected.points.at(size_t(global))[component]);
            std::vector<I> incidentCells, incidentBoundaries;
            for (size_t cell = 0; cell < expected.cells.size(); ++cell)
                if (std::find(expected.cells[cell].begin(), expected.cells[cell].end(), global) != expected.cells[cell].end())
                    incidentCells.push_back(I(cell));
            for (size_t boundary = 0; boundary < expected.boundaries.size(); ++boundary)
                if (std::find(expected.boundaries[boundary].begin(), expected.boundaries[boundary].end(), global) != expected.boundaries[boundary].end())
                    incidentBoundaries.push_back(I(boundary));
            CHECK(Decode(cells, view.NodeCells(row)) == incidentCells);
            CHECK(Decode(boundaries, view.NodeBoundaries(row)) == incidentBoundaries);
        }

        R::Array<DNDS::real> values(R::RowLayout::Dense(size_t(nodes.NumOwned()), 1));
        {
            auto write = values.Write();
            for (I row = 0; row < nodes.NumOwned(); ++row)
            {
                const auto point = expected.points.at(size_t(nodes.LocalToGlobal(row)));
                write.View()(size_t(row), 0) = 5 + point[0] + 2 * point[1];
            }
        }
        G::EntityField<DNDS::real> field(nodes, values);
        auto route = field.BuildHaloRoute(nodes);
        auto binding = route.Bind(field, nodes);
        binding.Begin().Wait();
        auto owned = field.Read(nodes);
        auto ghost = binding.GhostValues().Read();
        const DNDS::real stencils[]{26, 30};
        for (I row = 0; row < cells.NumOwned(); ++row)
        {
            DNDS::real sum = 0;
            for (I node : view.CellNodes(row))
                sum += node < nodes.NumOwned() ? owned.View()(size_t(node), 0) : ghost.View()(size_t(node - nodes.NumOwned()), 0);
            CHECK(sum == stencils[cells.LocalToGlobal(row)]);
        }
    }
}

TEST_CASE("Native CGNS serial rows and distributed construction preserve exact topology")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (bool multizone : {false, true})
    {
        CAPTURE(multizone);
        Fixture fixture(mpi, multizone);
        const auto expected = Expected(multizone);
        const int reader = mpi.size - 1;
        int serialReady = 1;
        if (mpi.rank == reader)
            try
            {
                CheckSerial(G::ReadNativeCGNSSerial(fixture.path, 2, BoundaryIds()), expected);
            }
            catch (const std::exception &error)
            {
                INFO(error.what());
                CHECK(false);
                serialReady = 0;
            }
        MPI_Bcast(&serialReady, 1, MPI_INT, reader, mpi.comm);
        REQUIRE(serialReady == 1);
        int calls = 0;
        const auto state = G::ReadNativeCGNS(mpi, fixture.path, 2, BoundaryIds(&calls), reader, {1, true});
        CHECK(calls == (mpi.rank == reader ? (multizone ? 3 : 2) : 0));
        CheckState(state, mpi, expected);
    }
}

TEST_CASE("Native CGNS import errors agree collectively and permit valid retry")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    Fixture fixture(mpi, false);
    const int reader = mpi.size - 1;
    CHECK_THROWS(G::ReadNativeCGNS(mpi, fixture.path + ".missing", 2, BoundaryIds(), reader));
    const G::t_FBCName_2_ID badBoundary = [](const std::string &) -> G::t_index
    { throw std::runtime_error("fixture boundary callback failure on reader only"); };
    CHECK_THROWS(G::ReadNativeCGNS(mpi, fixture.path, 2, badBoundary, reader));
    if (mpi.size > 1)
    {
        CHECK_THROWS(G::ReadNativeCGNS(mpi, fixture.path + (mpi.rank == 0 ? ".mismatch" : ""), 2, BoundaryIds(), reader));
        CHECK_THROWS(G::ReadNativeCGNS(mpi, fixture.path, mpi.rank == 0 ? 3 : 2, BoundaryIds(), reader));
    }
    CheckState(G::ReadNativeCGNS(mpi, fixture.path, 2, BoundaryIds(), reader, {1, true}), mpi, Expected(false));
}

int main(int argc, char **argv)
{
    // Native row kernels may create OMP/CPU workers even with controller MPI.
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    const int result = doctest::Context(argc, argv).run();
    DNDS::MPI::Finalize();
    return result;
}

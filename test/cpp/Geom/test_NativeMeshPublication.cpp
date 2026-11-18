#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Legacy/ArrayPair.hpp"
#include "DNDS/Legacy/Array.hpp"
#include "NativeMeshFacadeFixture.hpp"
#include "NativeMeshOverlapFixture.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Solver/Direct.hpp"
#include <optional>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

TEST_CASE("Managed mesh readers retain state metadata across placement replacement and moves")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = NativeFacadeFixture::Make(mpi).Publish(mpi);
    std::weak_ptr<const G::NativeMeshState> original = mesh->CurrentNativeState();
    std::optional<G::MeshReadAccess> retained;
    {
        auto read = mesh->AcquireRead();
        const auto &frames = mesh->PeriodicMetadata();
        const auto &neighbors = mesh->LocalCellNeighbors();
        const auto expectedFrames = frames;
        const auto expectedNeighbors = neighbors;
        mesh->PlaceNative(R::Device::Host(), R::ExecutionContext::Host());
        REQUIRE_FALSE(original.expired());
        CHECK(&frames == &original.lock()->BorrowHostView().Periodic());
        CHECK(&neighbors == &original.lock()->LocalCellNeighbors());
        CHECK(frames.translation[1].map() == expectedFrames.translation[1].map());
        CHECK(neighbors == expectedNeighbors);
        retained.emplace(std::move(read));
    }
    CHECK_FALSE(original.expired());
    CHECK(retained->HostView().NumCell() == mesh->NumCell());
    retained.reset();
    CHECK(original.expired());
    CHECK(mesh->HasNativeState());
}

TEST_CASE("Native publication owns semantic queries and current numbering without a legacy workspace")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto input = NativeFacadeFixture::Make(mpi);
    auto aliasedCoordinates = input.data.coords->owned;
    auto aliasedNodes = input.data.relations[size_t(G::MeshRelation::CellNodes)]->owned;
    auto mesh = input.Publish(mpi);
    const auto original = mesh->CurrentNativeState();
    const auto count = mesh->NumCell();
    const G::tPoint point = mesh->Coord(0);
    CHECK(mesh->HasNativeState());
    CHECK_NOTHROW(mesh->PlaceNative(R::Device::Host(), R::ExecutionContext::Host()));
    CHECK_THROWS(mesh->PlaceNative(R::Device::Host(), R::ExecutionContext::CUDA(0)));
    CHECK_THROWS(mesh->PlaceNative(R::Device::CUDA(-1), R::ExecutionContext::CUDA(-1)));
    struct WrongDeviceStream final : R::StreamResource
    {
        R::Device GetDevice() const noexcept override { return R::Device::CUDA(0); }
        void SynchronizeBackend() const override {}
    };
    CHECK_THROWS(mesh->PlaceNative(
        R::Device::Host(), {R::Device::Host(), std::make_shared<WrongDeviceStream>()}));
    const auto cells = mesh->Entities(G::EntityKind::Cell);
    const auto nodes = mesh->Entities(G::EntityKind::Node);
    const auto boundary = mesh->Entities(G::EntityKind::Bnd);
    const auto faces = mesh->Entities(G::EntityKind::Face);
    const auto global = cells.LocalToGlobal(0);
    CHECK(mesh->CellIndexLocal2Global(0) == global);
    CHECK(mesh->CellIndexGlobal2Local(global) == 0);
    CHECK(mesh->CellIndexLocal2Global_NoSon(0) == global);
    CHECK(mesh->CellIndexGlobal2Local_NoSon(global) == 0);
    CHECK(mesh->NodeIndexGlobal2Local(mesh->NodeIndexLocal2Global(0)) == 0);
    CHECK(mesh->BndIndexGlobal2Local(mesh->BndIndexLocal2Global(0)) == 0);
    CHECK(mesh->FaceIndexGlobal2Local(mesh->FaceIndexLocal2Global(0)) == 0);
    CHECK(mesh->CellIndexLocal2Global(DNDS::UnInitIndex) == DNDS::UnInitIndex);
    CHECK(mesh->CellIndexGlobal2Local(DNDS::UnInitIndex) == DNDS::UnInitIndex);
    CHECK(mesh->CellIndexLocal2Global(-1 - global) == global);
    CHECK_THROWS(mesh->CellIndexLocal2Global_NoSon(cells.NumOwned()));
    std::optional<G::MeshReadAccess> first, second;
    first.emplace(mesh->AcquireRead(NativeFacadeFixture::Requirements()));
    second.emplace(mesh->AcquireRead(NativeFacadeFixture::Requirements()));
    CHECK(first->HostView().CellNodes(0).data() == second->HostView().CellNodes(0).data());
    CHECK_THROWS(aliasedCoordinates.Write());
    CHECK_THROWS(aliasedNodes.Write());
    auto next = NativeFacadeFixture::Make(mpi).Publish(mpi)->CurrentNativeState();
    CHECK_THROWS(mesh->ReplaceNativeForSetup(next));
    CHECK(mesh->NumCell() == count);
    CHECK(mesh->Coord(0) == point);
    first.reset();
    second.reset();
    auto detached = G::NativeMeshSnapshot::Capture(*mesh);
    mesh->ReplaceNativeForSetup(next);
    CHECK_FALSE(cells.Valid());
    CHECK_FALSE(nodes.Valid());
    CHECK_FALSE(boundary.Valid());
    CHECK_FALSE(faces.Valid());
    CHECK(detached.AcquireRead().HostView().Coord(0) == point);
    CHECK(mesh->HasNativeState());
}

TEST_CASE("Published entity maps cannot be changed through a solver field export")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi).mesh;
    auto space = mesh->Entities(G::EntityKind::Cell);
    const auto firstGlobal = space.LocalToGlobal(0), total = space.NumGlobal();
    DNDS::ArrayPair<DNDS::ParArray<DNDS::real, 1>> field;
    field.InitPair("publication.export", mpi);
    field.father->Resize(space.NumOwned());
    field.son->Resize(0);
    field.TransAttach();
    G::LegacyFieldBridge::BorrowIndexing(field.trans, space);
    field.trans.pLGlobalMapping->RLengths()[size_t(mpi.rank)] = 0;
    field.trans.pLGlobalMapping->ROffsets().back() = 0;
    if (space.NumGhost())
    {
        const auto ghost = space.LocalToGlobal(space.NumOwned());
        field.trans.pLGhostMapping->ghostIndex[0] = -123;
        CHECK(space.LocalToGlobal(space.NumOwned()) == ghost);
    }
    CHECK(space.LocalToGlobal(0) == firstGlobal);
    CHECK(space.NumGlobal() == total);
    CHECK(space.Owner(firstGlobal) == mpi.rank);
    CHECK(space.SameOrder(mesh->Entities(G::EntityKind::Cell)));
}

TEST_CASE("Published periodic rows retain readonly incidence companions")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto input = NativeFacadeFixture::Make(mpi, true);
    auto bits = input.data.cellNodesPbi->owned;
    auto cone = input.data.relations[size_t(G::MeshRelation::CellNodes)]->owned;
    auto mesh = input.Publish(mpi);
    const auto cell = mesh->getCell2NodeIndexPbiRow(0);
    const auto boundary = mesh->getBnd2NodeIndexPbiRow(0);
    auto read = mesh->AcquireRead();
    CHECK_THROWS(bits.Write());
    CHECK_THROWS(cone.Write());
    CHECK(mesh->getCell2NodeIndexPbiRow(0) == cell);
    CHECK(mesh->getBnd2NodeIndexPbiRow(0) == boundary);
    CHECK(read.HostView().CellNodePeriodicBits(0)[2] == G::nodePB1);
    CHECK(read.HostView().IsPeriodic());
}

TEST_CASE("Native managed replacement rejects reads and preserves detached snapshots")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = NativeFacadeFixture::Make(mpi).Publish(mpi);
    auto old = mesh->Entities(G::EntityKind::Cell);
    auto detached = G::NativeMeshSnapshot::Capture(*mesh);
    const G::tPoint point = mesh->Coord(0);
    {
        auto read = mesh->AcquireRead();
        const auto stamp = mesh->versions.Stamp();
        CHECK_THROWS(mesh->TransformCoords([](const G::tPoint &p)
                                           { return p; }));
        CHECK(stamp.Valid());
        CHECK(read.HostView().Coord(0) == point);
    }
    mesh->TransformCoords([](const G::tPoint &p)
                          { return (p + G::tPoint{1, 0, 0}).eval(); });
    CHECK(mesh->HasNativeState());
    CHECK(old.Valid()); // Geometry publication preserves numbering.
    CHECK(detached.AcquireRead().HostView().Coord(0) == point);
    CHECK(mesh->Coord(0) == point + G::tPoint{1, 0, 0});
}

TEST_CASE("Native original IDs reject absent authority invalid rows and malformed candidate extents")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    G::UnstructuredMesh empty(mpi, 2);
    CHECK_THROWS(empty.OriginalEntityId(G::EntityKind::Cell, 0));
    CHECK_THROWS(empty.OriginalEntityId(G::EntityKind::Face, 0));
    auto input = NativeFacadeFixture::Make(mpi);
    const auto expected = input.ids.at(G::EntityKind::Cell).Read().View()(0, 0);
    auto mesh = input.Publish(mpi);
    CHECK_THROWS(mesh->OriginalEntityId(G::EntityKind::Cell, -1));
    CHECK_THROWS(mesh->OriginalEntityId(G::EntityKind::Cell, mesh->NumCellProc()));
    CHECK(mesh->OriginalEntityId(G::EntityKind::Cell, 0) == expected);
    for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd})
    {
        auto bad = NativeFacadeFixture::Make(mpi);
        bad.ids.at(kind) = R::Array<I>(R::RowLayout::Dense(0, 1));
        CHECK_THROWS(bad.Publish(mpi));
        CHECK(mesh->OriginalEntityId(G::EntityKind::Cell, 0) == expected);
    }
    CHECK_NOTHROW(NativeFacadeFixture::Make(mpi).Publish(mpi));
}

TEST_CASE("Native symbolic factorization retains detached setup metadata exactly")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = G::OverlapTest::BuildNativeFixture(mpi).mesh->CurrentNativeState();
    const auto count = source->NumOwned(G::EntityKind::Cell), split = count / 2;
    std::vector<I> parts{0, split, count};
    G::tLocalMatStruct graph(static_cast<size_t>(count));
    for (int part = 0; part < 2; ++part)
        for (I cell = parts[size_t(part)]; cell < parts[size_t(part + 1)]; ++cell)
        {
            if (cell > parts[size_t(part)])
                graph[size_t(cell)].push_back(cell - 1);
            if (cell + 1 < parts[size_t(part + 1)])
                graph[size_t(cell)].push_back(cell + 1);
        }
    source = source->WithLocalPartitions(parts);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, source->WithLocalCellSetup(parts, graph));
    DNDS::Direct::DirectPrecControl control;
    control.useDirectPrec = true;
    DNDS::Direct::SerialSymLUStructure expected(mpi, count), actual(mpi, count);
    expected.ObtainSymmetricSymbolicFactorization(graph, parts, control.iluCode);
    parts.assign(1, 1234);
    graph.assign(1, {1234});
    mesh->ObtainSymmetricSymbolicFactorization(actual, control);
    CHECK(actual.localPartStarts == expected.localPartStarts);
    CHECK(actual.lowerTriStructure == expected.lowerTriStructure);
    CHECK(actual.upperTriStructure == expected.upperTriStructure);
    CHECK(actual.cell2cellFaceVLocal2FullRowPos == expected.cell2cellFaceVLocal2FullRowPos);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}

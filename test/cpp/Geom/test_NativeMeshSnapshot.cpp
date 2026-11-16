#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NativeMeshOverlapFixture.hpp"
#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include "Geom/Access/CellFaceIncidence.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "NativePeriodicFixtures.hpp"
#include <optional>
#include <set>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;

TEST_CASE("Canonical incidence projection shares host rows and guards source numbering")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    const auto projection = G::CellFaceIncidenceStorage::Capture(fixture.mesh);
    CHECK(projection.GetDevice() == R::Device::Host());
    {
        const auto read = projection.AcquireRead();
        const auto ordinary = fixture.mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        const auto projected = read.View();
        const auto canonical = ordinary.HostView().CellFaceIncidences();
        CHECK(projected.cellFaces.owned.data == canonical.cellFaces.owned.data);
        CHECK(projected.cellFaces.ghost.data == canonical.cellFaces.ghost.data);
        CHECK(projected.faceCells.owned.data == canonical.faceCells.owned.data);
        CHECK(projected.faceInfo.owned.data == canonical.faceInfo.owned.data);
        CHECK_FALSE(fixture.mesh->versions.CanMutate());
    }
    CHECK(fixture.mesh->versions.CanMutate());
    std::set<uint64_t> compact, complete;
    projection.VisitAllocations([&](uint64_t id, size_t, R::Device)
                                { compact.insert(id); });
    fixture.mesh->CurrentNativeState()->Snapshot(R::Device::Host()).VisitAllocations([&](uint64_t id, size_t, R::Device)
                                                                                     { complete.insert(id); });
    CHECK(compact.size() < complete.size());
    for (const auto id : compact)
        CHECK(complete.count(id) == 1);
    // A cache owner does not permanently pin facade mutation. The operator
    // read scope does; after reordering, old numbering cannot be reacquired.
    fixture.mesh->ReorderLocalCells(2, 1);
    CHECK_THROWS(projection.AcquireRead());
}

TEST_CASE("Detached native mesh snapshots retain values metadata and independent validity")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    std::optional<G::NativeMeshSnapshot> snapshot;
    snapshot.emplace(G::NativeMeshSnapshot::Capture(*fixture.mesh, G::MeshReadRequirements::CellFlux()));
    auto first = snapshot->AcquireRead();
    auto second = snapshot->AcquireRead();
    const auto view = first.HostView(), again = second.HostView();
    REQUIRE(view.NumCell() > 0);
    const auto originalNodes = std::vector<DNDS::index>(view.CellNodes(0).begin(), view.CellNodes(0).end());
    const G::tPoint originalPoint = view.Coord(0);
    CHECK(view.CellNodes(0).data() == again.CellNodes(0).data());
    CHECK(view.Coord(0).data() == again.Coord(0).data());
    {
        auto source = fixture.mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        CHECK(view.CellNodes(0).data() != source.HostView().CellNodes(0).data());
    }
    CHECK(fixture.mesh->versions.CanMutate());
    fixture.mesh->ReorderLocalCells(2, 1);
    CHECK_FALSE(snapshot->SourceStamp().Valid());
    CHECK(first.Stamp().Valid());
    CHECK(snapshot->AcquireRead().Stamp().Valid());
    // Source structural replacement and deletion cannot touch snapshot shape.
    fixture.mesh = fixture.mesh->ExtractBoundary();
    CHECK(fixture.mesh->CurrentNativeState()->BorrowHostView().Dimension() == 1);
    fixture.mesh.reset();
    CHECK(view.NumCell() > 0);
    CHECK(std::vector<DNDS::index>(view.CellNodes(0).begin(), view.CellNodes(0).end()) == originalNodes);
    CHECK(view.Coord(0).isApprox(originalPoint));
    snapshot.reset();
    CHECK(first.HostView().Coord(0).isApprox(originalPoint));
    CHECK(second.HostView().CellNodes(0).size() == originalNodes.size());
}

TEST_CASE("Detached snapshot publication retains validation and explicit placement")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto state = G::ConstructNativeMesh(NativePeriodicFixtures::TwoCells(mpi), {1, true});
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    auto snapshot = G::NativeMeshSnapshot::Capture(*mesh);
    CHECK_FALSE(snapshot.AcquireRead().HostView().HasRelation(G::MeshRelation::CellFaces));
    CHECK_THROWS(snapshot.AcquireRead(R::ExecutionContext::CUDA(0)));
    auto copy = snapshot.To(R::Device::Host());
    auto first = snapshot.AcquireRead(), second = copy.AcquireRead();
    if (first.HostView().NumCellProc() > 0)
        CHECK(first.HostView().CellNodes(0).data() != second.HostView().CellNodes(0).data());
    // Reject malformed detached native input before publication; the source
    // and its already-retained detached snapshot remain valid and unchanged.
    auto malformed = G::NativeMeshAssembler::CaptureOwned(*state, mpi);
    if (mpi.rank == 0)
    {
        auto &rows = *malformed.relations[size_t(G::MeshRelation::CellNodes)];
        rows = rows.To(R::Device::Host());
        auto write = rows.Write();
        write.View()(0, 0) = state->NumGlobal(G::EntityKind::Node);
    }
    CHECK_THROWS(G::AssembleNativeMesh(mpi, std::move(malformed)));
    CHECK_NOTHROW(G::AssembleNativeMesh(mpi, G::NativeMeshAssembler::CaptureOwned(*state, mpi)));
    CHECK(first.Stamp().Valid());
    auto moved = std::move(copy);
    CHECK_THROWS(copy.GetDevice());
    CHECK_THROWS(copy.SourceStamp());
    CHECK_THROWS(copy.AcquireRead());
    CHECK_THROWS(copy.To(R::Device::Host()));
    CHECK(moved.AcquireRead().HostView().NumCell() == first.HostView().NumCell());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}

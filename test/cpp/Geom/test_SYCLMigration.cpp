#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NativeMeshOverlapFixture.hpp"
#include "DNDS/Runtime/Execution/SYCLQueue.hpp"

namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;
using I = DNDS::index;

static DNDS::MPIInfo mpi;
static R::SYCLTarget target;

TEST_CASE("SYCL queue and allocation owners cover table, strided and ragged rows")
{
    const auto device = R::SelectSYCLDevice(target);
    const auto context = R::CreateSYCLQueue(device);
    // Dropping the last caller-owned source handle after submission leaves the
    // copy operation responsible for its storage until the queue prefix retires.
    R::Array<I> destination(R::RowLayout::Dense(1, 1), device);
    R::CopyOperation transfer;
    {
        R::Array<I> source(R::RowLayout::Dense(1, 1));
        source.Write().View()(0, 0) = 37;
        auto target = destination.Storage();
        transfer = R::Copy(source.Storage(), target, context);
    }
    transfer.Wait();
    CHECK(destination.To(R::Device::Host(), context).Read().View()(0, 0) == 37);
    const R::Array<I> empty(R::RowLayout::Dense(0, 1));
    CHECK(empty.To(device, context).To(R::Device::Host(), context).RowCount() == 0);
    for (const auto &layout : {R::RowLayout::Dense(3, 2), R::RowLayout::ColMajor(3, 2), R::RowLayout::Ragged({2, 0, 3})})
    {
        R::Array<I> host(layout);
        {
            auto write = host.Write();
            auto v = write.View();
            for (I i = 0; i < 3; ++i)
                for (I j = 0; j < v.RowSize(i); ++j)
                    v(i, j) = 10 * i + j;
        }
        auto placed = host.To(device, context);
        auto access = placed.Write(context);
        const auto v = access.View();
        R::SYCLQueue(context).parallel_for(sycl::range<1>{3}, [=](sycl::id<1> id)
                                           {
            const I i = I(id[0]);
            for (I j = 0; j < v.RowSize(i); ++j) v(i,j) += 7; });
        access.Complete(R::Completion::Record(context));
        access.Wait();
        const auto out = placed.To(R::Device::Host(), context).Read();
        for (I i = 0; i < 3; ++i)
            for (I j = 0; j < layout.RowSize(i); ++j)
                CHECK(out.View()(i, j) == 10 * i + j + 7);
    }
}

TEST_CASE("SYCL mesh placement preserves owned and ghost incidence and coordinates")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi).mesh;
    const auto device = R::SelectSYCLDevice(target);
    const auto context = R::CreateSYCLQueue(device);
    auto requirements = G::MeshReadRequirements::CellFlux();
    const auto stamp = mesh->versions.Stamp();
    const auto cellOrder = mesh->Entities(G::EntityKind::Cell);
    const auto firstId = mesh->OriginalEntityId(G::EntityKind::Cell, 0);
    mesh->PlaceNative(device, context);
    CHECK(stamp.Valid());
    CHECK(cellOrder.SameOrder(mesh->Entities(G::EntityKind::Cell)));
    CHECK(firstId == mesh->OriginalEntityId(G::EntityKind::Cell, 0));
    auto host = mesh->AcquireRead(requirements);
    auto placed = mesh->AcquireRead(requirements, context);
    const auto reference = host.HostView();
    const auto count = reference.NumCellProc();
    REQUIRE((mpi.size == 1 || reference.NumCellGhost() > 0));
    const auto descriptor = R::MetadataBuffer<G::UnstructuredMeshView>({placed.DeviceView()}).CopyTo(device, context);
    auto descRead = descriptor.Read(context);
    const auto *view = descRead.View<G::UnstructuredMeshView>().data;
    R::Array<double> result(R::RowLayout::Dense(count, 3), device);
    auto write = result.Write(context);
    const auto out = write.View<R::FixedRows<3>>();
    R::SYCLQueue(context).parallel_for(sycl::range<1>{size_t(count)}, [=](sycl::id<1> id)
                                       {
        const I i = I(id[0]);
        const auto nodes = view->CellNodes(i);
        out(i,0) = double(nodes.size());
        out(i,1) = double(nodes[0]);
        out(i,2) = view->Coord(nodes[0])[0]; });
    const auto done = R::Completion::Record(context);
    placed.Complete(done);
    descRead.Complete(done);
    write.Complete(done);
    write.Wait();
    const auto read = result.To(R::Device::Host(), context).Read();
    for (I i = 0; i < count; ++i)
    {
        const auto nodes = reference.CellNodes(i);
        CHECK(read.View()(i, 0) == double(nodes.size()));
        CHECK(read.View()(i, 1) == double(nodes[0]));
        CHECK(read.View()(i, 2) == reference.Coord(nodes[0])[0]);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    target = argc == 2 && std::string(argv[1]) == "cuda" ? R::SYCLTarget::CUDA : R::SYCLTarget::CPU;
    const int local = doctest::Context().run();
    int result = 0;
    MPI_Allreduce(&local, &result, 1, MPI_INT, MPI_MAX, mpi.comm);
    MPI_Finalize();
    return result;
}

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <filesystem>
#include <unistd.h>

using namespace DNDS::Runtime;
namespace S = DNDS::Serializer;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int result = ctx.run();
    MPI_Finalize();
    return result;
}

TEST_CASE("Native parallel checkpoint matching partitions and direct read")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = "__runtime_array_" + std::to_string(pid) + ".h5";
    const size_t localRows = mpi.rank == 0 ? 3 : 0;
    Array<double> a(RowLayout::Ragged(localRows ? std::vector<size_t>{2, 0, 3} : std::vector<size_t>{}));
    auto owner = std::make_shared<std::vector<double>>(8 + localRows * 8, -1);
    auto imported = Allocation::Import(std::shared_ptr<void>(owner, owner->data()), owner->size() * sizeof(double));
    auto state = Array<double, FixedRows<7>>::FromStorage(RowLayout::Dense(localRows, 7, 8),
                                                          Buffer::FromAllocation(imported, 4 * sizeof(double), localRows * 8 * sizeof(double)));
    {
        auto write = state.Write();
        for (size_t i = 0; i < localRows; ++i)
            for (size_t j = 0; j < 7; ++j)
                write.At(i, j) = 100 * i + j;
    }
    if (localRows)
    {
        auto write = a.Write();
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < a.RowSize(i); ++j)
                write.At(i, j) = i * 10 + j;
    }
    auto serializer = std::make_shared<S::SerializerH5>(mpi);
    serializer->SetCollectiveRW(true, true);
    serializer->OpenFile(file, false);
    WriteArray(a, serializer, "state", mpi);
    WriteArray(state, serializer, "imported_state", mpi);
    Array<double> padded(RowLayout::Padded(localRows ? std::vector<size_t>{5, 0, 7} : std::vector<size_t>{}, 12));
    if (localRows)
        padded.Write().At(2, 6) = 317;
    WriteArray(padded, serializer, "padded", mpi);
#ifdef DNDS_USE_CUDA
    CHECK_THROWS(WriteArray(a.To(Device::CUDA(0)), serializer, "gpu_state", mpi));
#endif
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto matching = ReadArray<double>(serializer, "state", mpi);
    CHECK(matching.RowCount() == localRows);
    if (localRows)
        CHECK(matching.Read().At(2, 2) == 22);
    auto matchingState = ReadArray<double, FixedRows<7>>(serializer, "imported_state", mpi);
    CHECK(matchingState.RowCount() == localRows);
    if (localRows)
        CHECK(matchingState.Read().At(2, 6) == 206);
    auto matchingPadded = ReadArray<double>(serializer, "padded", mpi);
    CHECK(matchingPadded.RowCount() == localRows);
    if (localRows)
    {
        CHECK(matchingPadded.RowSize(1) == 0);
        CHECK(matchingPadded.Read().At(2, 6) == 317);
    }
#ifdef DNDS_USE_CUDA
    CHECK_THROWS(ReadArray<double>(serializer, "state", mpi, S::ArrayGlobalOffset_Unknown, Device::CUDA(0)));
#endif
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    // Each rank may directly request any contiguous row partition, including overlap.
    auto direct = ReadArray<double>(serializer, "state", mpi, {1, 2});
    CHECK(direct.RowCount() == 1);
    CHECK(direct.Read().At(0, 2) == 22);
    auto directState = ReadArray<double, FixedRows<7>>(serializer, "imported_state", mpi, {1, 2});
    CHECK(directState.RowCount() == 1);
    CHECK(directState.Read().At(0, 6) == 206);
    auto directPadded = ReadArray<double>(serializer, "padded", mpi, {1, 2});
    CHECK(directPadded.RowSize(0) == 7);
    CHECK(directPadded.Read().At(0, 6) == 317);
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto even = ReadArray<double>(serializer, "state", mpi, S::ArrayGlobalOffset_EvenSplit);
    int64_t total = even.RowCount();
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(total == 3);
    CHECK_THROWS(ReadArray<int32_t>(serializer, "state", mpi));
    serializer->CloseFile();
    serializer.reset();
    // Change communicator size: direct partition read is independent of writer ranks.
    if (mpi.rank == 0)
    {
        DNDS::MPIInfo single(MPI_COMM_SELF);
        auto reader = std::make_shared<S::SerializerH5>(single);
        reader->OpenFile(file, true);
        auto all = ReadArray<double>(reader, "state", single, S::ArrayGlobalOffset_EvenSplit);
        CHECK(all.RowCount() == 3);
        CHECK(all.Read().At(2, 1) == 21);
        if (mpi.size > 1)
            CHECK_THROWS(ReadArray<double>(reader, "state", single));
        reader->CloseFile();
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Parallel field bundle restores uneven reversed partitions with variable row lengths")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = "__runtime_bundle_" + std::to_string(pid) + ".h5";
    std::vector<DNDS::index> counts(mpi.size), offsets(mpi.size + 1, 0);
    for (int rank = 0; rank < mpi.size; ++rank)
    {
        counts[rank] = rank == 0 && mpi.size > 1 ? 0 : 67 + 2 * rank;
        offsets[rank + 1] = offsets[rank] + counts[rank];
    }
    const size_t n = counts[mpi.rank], first = offsets[mpi.rank];
    Array<double, FixedRows<6>> state(RowLayout::Dense(n, 6, 8));
    std::vector<size_t> widths(n);
    for (size_t i = 0; i < n; ++i)
        widths[i] = ((first + i) * 17) % 9;
    Array<int64_t> stencil(RowLayout::Padded(widths, 11));
    {
        auto u = state.Write();
        auto c2c = stencil.Write();
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < 6; ++j)
                u.At(i, j) = 100 * (first + i) + j;
            for (size_t j = 0; j < widths[i]; ++j)
                c2c.At(i, j) = ((first + i) * 13 + j * 7) % offsets.back();
        }
    }
    auto serializer = std::make_shared<S::SerializerH5>(mpi);
    serializer->SetCollectiveRW(true, true);
    serializer->OpenFile(file, false);
    WriteArray(state, serializer, "flow", mpi);
    WriteArray(stencil, serializer, "stencil", mpi);
    serializer->CloseFile();
    for (bool reversed : {false, true})
    {
        const int from = reversed ? mpi.size - 1 - mpi.rank : mpi.rank;
        const auto partition = reversed ? S::ArrayGlobalOffset{counts[from], offsets[from]} : S::ArrayGlobalOffset_Unknown;
        serializer->OpenFile(file, true);
        auto restored = ReadArray<double, FixedRows<6>>(serializer, "flow", mpi, partition);
        auto restoredStencil = ReadArray<int64_t>(serializer, "stencil", mpi, partition);
        REQUIRE(restored.RowCount() == size_t(counts[from]));
        REQUIRE(restoredStencil.RowCount() == restored.RowCount());
        CHECK(restored.Layout().Stride() == 6); // Checkpoints carry logical rows, not capacity padding.
        auto u = restored.Read();
        auto c2c = restoredStencil.Read();
        for (size_t i = 0; i < restored.RowCount(); ++i)
        {
            const size_t global = offsets[from] + i;
            CHECK(restoredStencil.RowSize(i) == (global * 17) % 9);
            for (size_t j = 0; j < 6; ++j)
                CHECK(u.At(i, j) == 100 * global + j);
            for (size_t j = 0; j < restoredStencil.RowSize(i); ++j)
                CHECK(c2c.At(i, j) == (global * 13 + j * 7) % offsets.back());
        }
        serializer->CloseFile();
    }
    serializer.reset();
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Native JSON checkpoint padded rows and POD schema")
{
    struct Info
    {
        int id;
        double x;
    };
    ElementSchema::Register<Info>(ElementSchema::Record("metadata.v1", sizeof(Info),
                                                        {{"id", offsetof(Info, id), ElementScalar::I32}, {"x", offsetof(Info, x), ElementScalar::F64}}, "Info", alignof(Info)));
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto file = "__runtime_json_" + std::to_string(getpid()) + ".json";
    auto serializer = std::make_shared<S::SerializerJSON>();
    Array<Info> a(RowLayout::Dense(2, 1, 3));
    a.Write().At(1, 0) = {mpi.rank, 0.25};
    serializer->OpenFile(file, false);
    WriteArray(a, serializer, "metadata", mpi, "metadata.v1");
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto b = ReadArray<Info, FixedRows<1>>(serializer, "metadata", mpi, S::ArrayGlobalOffset_Unknown, Device::Host(), "metadata.v1");
    CHECK(b.Layout().Stride() == 1);
    CHECK(b.Read().At(1, 0).id == mpi.rank);
    CHECK(b.Read().At(1, 0).x == 0.25);
    serializer->CloseFile();
    std::filesystem::remove(file);
}

TEST_CASE("Native physical table archives restore placement and read directly into other layouts")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = "__runtime_physical_" + std::to_string(pid) + ".h5";
    const DNDS::index rows = mpi.rank == 0 ? 0 : mpi.rank + 9;
    const auto shapes = std::vector<RowLayout>{RowLayout::Dense(rows, 5, 7), RowLayout::ColMajor(rows, 5, {8, 1}), RowLayout::TiledColMajor(rows, 5, 8)};
    auto serializer = std::make_shared<S::SerializerH5>(mpi);
    serializer->SetCollectiveRW(true, true);
    serializer->OpenFile(file, false);
    for (DNDS::index k = 0; k < 3; ++k)
    {
        Array<double> values(shapes[size_t(k)]);
        {
            auto write = values.Write();
            for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                for (DNDS::index iComp = 0; iComp < 5; ++iComp)
                    write.View()(iRow, iComp) = 10000 * mpi.rank + 100 * iRow + iComp;
        }
        WriteArray(values, serializer, "physical_" + std::to_string(k), mpi);
    }
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    for (DNDS::index k = 0; k < 3; ++k)
    {
        const auto name = "physical_" + std::to_string(k);
        auto restored = ReadArray<double>(serializer, name, mpi);
        CHECK(restored.RowCount() == rows);
        if (k)
            CHECK(restored.Layout() == shapes[size_t(k)]);
        const auto check = [&](const auto &array)
        {
            const auto read = array.Read();
            for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                for (DNDS::index iComp = 0; iComp < 5; ++iComp)
                    CHECK(read.View()(iRow, iComp) == 10000 * mpi.rank + 100 * iRow + iComp);
        };
        check(restored);
        // Include an empty reader and a nonempty dense reader at np4.
        // Odd ranks select explicit windows and a strided destination.
        if (mpi.rank % 2 == 0)
            check(ReadArray<double, TableRows>(serializer, name, mpi));
        else
        {
            const auto first = (mpi.rank - 1) * (18 + DNDS::index(mpi.rank)) / 2;
            check(ReadArray<double, ColMajorTable>(serializer, name, mpi, {rows, first}));
        }
        check(ReadArray<double, CSRRows>(serializer, name, mpi));
        check(ReadArray<double, ColMajorTable>(serializer, name, mpi));
        check(ReadArray<double, TiledColMajorTable<8>>(serializer, name, mpi));
        check(ReadArray<double, TableRows>(serializer, name, mpi));
        const DNDS::index globalRows = mpi.size == 1 ? 0 : 10;
        auto partitioned = ReadArray<double, TiledColMajorTable<8>>(serializer, name, mpi, {globalRows, 0});
        const auto read = partitioned.Read();
        for (DNDS::index iRow = 0; iRow < globalRows; ++iRow)
            for (DNDS::index iComp = 0; iComp < 5; ++iComp)
                CHECK(read.View()(iRow, iComp) == 10000 + 100 * iRow + iComp);
    }
    serializer->CloseFile();
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Per-rank JSON physical tables retain logical values through direct format conversion")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto file = "__runtime_physical_" + std::to_string(getpid()) + ".json";
    auto serializer = std::make_shared<S::SerializerJSON>();
    serializer->OpenFile(file, false);
    for (DNDS::index k = 0; k < 3; ++k)
    {
        const auto shape = k == 0 ? RowLayout::Dense(9, 5, 7) : k == 1 ? RowLayout::ColMajor(9, 5, {8, 1})
                                                                       : RowLayout::TiledColMajor(9, 5, 8);
        Array<int64_t> values(shape);
        {
            auto write = values.Write();
            for (DNDS::index iRow = 0; iRow < 9; ++iRow)
                for (DNDS::index iComp = 0; iComp < 5; ++iComp)
                    write.View()(iRow, iComp) = iRow * 100 + iComp;
        }
        WriteArray(values, serializer, "physical_" + std::to_string(k), mpi);
    }
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    for (DNDS::index k = 0; k < 3; ++k)
    {
        auto values = ReadArray<int64_t, ColMajorTable>(serializer, "physical_" + std::to_string(k), mpi);
        const auto read = values.Read();
        for (DNDS::index iRow = 0; iRow < 9; ++iRow)
            for (DNDS::index iComp = 0; iComp < 5; ++iComp)
                CHECK(read.View()(iRow, iComp) == iRow * 100 + iComp);
    }
    serializer->CloseFile();
    std::filesystem::remove(file);
}

#include "doctest.h"
#include "DNDS/Legacy/Array.hpp"
#include "DNDS/Legacy/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/Legacy/ArrayDerived/ArrayEigenMatrix.hpp"
#include "DNDS/Legacy/ArrayRedistributor.hpp"
#include "DNDS/Legacy/ArrayPair.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include <filesystem>
#include <limits>

TEST_CASE("Local checks: rejected static width preserves the array" * doctest::test_suite("rank-local"))
{
    DNDS::Array<int, 2> values;
    values.Resize(1);
    values(0, 0) = 42;
    CHECK_THROWS(values.Resize(3, 1));
    CHECK(values.Size() == 1);
    if (values.Size() == 1)
        CHECK(values(0, 0) == 42);
}

TEST_CASE("Local checks: padded rows reject negative widths" * doctest::test_suite("rank-local"))
{
    DNDS::Array<int, DNDS::NonUniformSize, 3> values;
    values.Resize(1);
    values.ResizeRow(0, 2);
    CHECK_THROWS(values.ResizeRow(0, -1));
    CHECK(values.RowSize(0) == 2);
}

TEST_CASE("Local checks: batch shape sentinel is exactly minus one" * doctest::test_suite("rank-local"))
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    DNDS::ArrayEigenUniMatrixBatch<Eigen::Dynamic, Eigen::Dynamic> values(mpi);
    values.Resize(1, 2, 3);
    CHECK_THROWS(values.ResizeMatrix(-2, 3));
    CHECK(values.Size() == 1);
    CHECK(values.Rows() == 2);
    CHECK(values.Cols() == 3);
}

TEST_CASE("Local checks: size arithmetic rejects limits without allocations" * doctest::test_suite("rank-local"))
{
    using DNDS::CheckedSize::Add;
    using DNDS::CheckedSize::Multiply;
    const auto max = std::numeric_limits<DNDS::index>::max();
    CHECK_THROWS(Add(max, DNDS::index(1)));
    CHECK_THROWS(Add(DNDS::index(0), DNDS::index(-1)));
    CHECK(Multiply(max, DNDS::index(0)) == 0);
    CHECK(Multiply(max, DNDS::index(1)) == max);
    CHECK_THROWS(Multiply(max, DNDS::index(2)));
    CHECK_THROWS(Multiply(DNDS::index(-1), DNDS::index(0)));
    DNDS::host_device_vector<double> buffer(1, 42);
    CHECK_THROWS(buffer.resize(std::numeric_limits<size_t>::max() / sizeof(double) + 1));
    CHECK(buffer.size() == 1);
    CHECK(buffer[0] == 42);
    DNDS::Array<int, 2> fixed;
    fixed.Resize(1);
    CHECK_THROWS(fixed.Resize(-1));
    CHECK_THROWS(fixed.Resize(max));
    CHECK(fixed.Size() == 1);
    DNDS::Array<int, DNDS::NonUniformSize> csr;
    csr.Resize(1, [](DNDS::index)
               { return 2; });
    csr(0, 0) = 42;
    CHECK_THROWS(csr.Resize(max, [](DNDS::index)
                            { return 0; }));
    CHECK_THROWS(csr.Resize(2, [](DNDS::index i)
                            { return i == 0 ? 2 : -1; }));
    CHECK(csr.Size() == 1);
    CHECK(csr(0, 0) == 42);
    csr.Decompress();
    CHECK_THROWS(csr.ReserveRow(0, -1));
}

TEST_CASE("Local checks: matrix products and dimensions are checked first" * doctest::test_suite("rank-local"))
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    DNDS::ArrayEigenMatrix<DNDS::DynamicSize, DNDS::DynamicSize> m(mpi);
    m.Resize(1, 2, 3);
    CHECK_THROWS(m.Resize(1, -2, -3));
    CHECK_THROWS(m.Resize(1, std::numeric_limits<int>::max(), 2));
    CHECK(m.Size() == 1);
    CHECK(m.MatRowSize() == 2);
    CHECK(m.MatColSize() == 3);
    DNDS::ArrayEigenUniMatrixBatch<Eigen::Dynamic, Eigen::Dynamic> b(mpi);
    b.Resize(1, 2, 3);
    CHECK_THROWS(b.ResizeMatrix(std::numeric_limits<int>::max(), 2));
    CHECK_THROWS(b.ResizeBatch(0, std::numeric_limits<int>::max()));
    CHECK_THROWS(b.Resize(-1, 2, 3));
    CHECK(b.Size() == 1);
    CHECK(b.MSize() == 6);
    DNDS::ArrayEigenMatrix<DNDS::NonUniformSize, 2> rows(mpi);
    rows.Resize(1, 3, 2);
    CHECK_THROWS(rows.ResizeRow(0, 2, 3));
    CHECK(rows.MatRowSize(0) == 3);
}

TEST_CASE("Local checks: gathered mapping handles empty ranks and invalid input")
{
    DNDS::MPIInfo world(MPI_COMM_WORLD);
    MPI_Comm reversed;
    MPI_Comm_split(world.comm, 0, world.size - world.rank, &reversed);
    for (MPI_Comm comm : {world.comm, reversed})
    {
        DNDS::MPIInfo mpi(comm);
        DNDS::GlobalOffsetsMapping mapping;
        mapping.setMPIAlignBcast(mpi, mpi.rank == mpi.size - 1 ? 3 : 0);
        CHECK(mapping.globalSize() == 3);
        CHECK(mapping(mpi.rank, mapping.RLengths()[mpi.rank]) == mapping.ROffsets()[mpi.rank + 1]);
        CHECK_THROWS(mapping(mpi.size, 0));
        CHECK_THROWS(mapping(-1, 0));
        CHECK(std::get<1>(mapping.search(0)) == mpi.size - 1);
        CHECK_THROWS(mapping.setMPIAlignBcast(mpi, mpi.rank == 0 ? -1 : 0));
        CHECK(mapping.globalSize() == 3);
        if (mpi.size > 1)
            CHECK_THROWS(mapping.setMPIAlignBcast(mpi, std::numeric_limits<DNDS::index>::max()));
        mapping.setMPIAlignBcast(mpi, 0);
        CHECK_FALSE(std::get<0>(mapping.search(0)));
    }
    MPI_Comm_free(&reversed);
}

TEST_CASE("Local checks: redistribution rejects rank-local invalid layouts collectively")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    std::vector<int> counts(mpi.size, 0);
    counts[0] = std::numeric_limits<int>::max() / 2;
    CHECK(DNDS::detail::RedistributionDisplacements(mpi, counts, 2).back() == counts[0]);
    if (mpi.rank == 0)
        ++counts[0];
    CHECK_THROWS(DNDS::detail::RedistributionDisplacements(mpi, counts, 2));
    counts.assign(mpi.size, 0);
    if (mpi.rank == 0)
        counts[0] = -1;
    CHECK_THROWS(DNDS::detail::RedistributionDisplacements(mpi, counts, 1));
    auto mapping = std::make_shared<DNDS::GlobalOffsetsMapping>();
    mapping->setMPIAlignBcast(mpi, mpi.rank == 0 ? 2 : 0);
    std::vector<DNDS::index> read = mpi.rank == 0 ? std::vector<DNDS::index>{5, 7} : std::vector<DNDS::index>{};
    auto result = DNDS::BuildRedistributionPullingIndex(mpi, read, {7, 5, 7}, mapping);
    CHECK(result == std::vector<DNDS::index>{1, 0, 1});
    CHECK_THROWS(DNDS::BuildRedistributionPullingIndex(mpi, read, mpi.rank == 0 ? std::vector<DNDS::index>{-1} : std::vector<DNDS::index>{5}, mapping));
    CHECK_THROWS(DNDS::BuildRedistributionPullingIndex(mpi, read, {6}, mapping));
    if (mpi.rank == 0)
        read[1] = read[0];
    CHECK_THROWS(DNDS::BuildRedistributionPullingIndex(mpi, read, {5}, mapping));
}

TEST_CASE("Local checks: redistribution preserves exact sparse and high ID row order")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const int owners = std::max(1, mpi.size - 1);
    auto count = [&](int rank) -> DNDS::index
    { return rank < owners ? 3 + rank % 2 : 0; };
    auto first = [](int rank) -> DNDS::index
    { return 3 * DNDS::index(rank) + rank / 2; };
    auto mapping = std::make_shared<DNDS::GlobalOffsetsMapping>();
    mapping->setMPIAlignBcast(mpi, count(mpi.rank));
    for (int mode : {0, 1, 2})
    {
        CAPTURE(mode);
        auto key = [&](DNDS::index global)
        {
            if (mode == 0)
                return global;
            if (mode == 1)
                return DNDS::index(1000000000000LL) + global * 101;
            return std::numeric_limits<DNDS::index>::max() - 1 - global * 4099;
        };
        std::vector<DNDS::index> read(size_t(count(mpi.rank)));
        using Array = DNDS::ParArray<DNDS::index, 2>;
        auto source = std::make_shared<Array>(mpi);
        source->Resize(count(mpi.rank));
        for (DNDS::index row = 0; row < count(mpi.rank); ++row)
        {
            const auto global = first(mpi.rank) + row;
            read[size_t(row)] = key(global);
            (*source)(row, 0) = key(global);
            (*source)(row, 1) = 100 * global + 17;
        }
        const int owner = (mpi.rank + 1) % owners;
        const std::vector<DNDS::index> expected{first(owner) + count(owner) - 1,
                                                first(owner), first(owner) + count(owner) - 1,
                                                first(owner) + 1};
        std::vector<DNDS::index> query;
        for (auto global : expected)
            query.push_back(key(global));
        const auto originalRead = read, originalQuery = query;
        CHECK(DNDS::BuildRedistributionPullingIndex(mpi, read, query, mapping) == expected);
        CHECK(read == originalRead);
        CHECK(query == originalQuery);
        auto destination = std::make_shared<Array>(mpi);
        destination->Resize(DNDS::index(query.size()));
        DNDS::RedistributeArrayWithTransformer(mpi, source, read, query, destination);
        for (size_t row = 0; row < expected.size(); ++row)
        {
            CHECK((*destination)(DNDS::index(row), 0) == query[row]);
            CHECK((*destination)(DNDS::index(row), 1) == 100 * expected[row] + 17);
        }
    }
}

TEST_CASE("Local checks: high ID redistribution faults reject collectively and allow retry")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto key = [](DNDS::index global)
    { return std::numeric_limits<DNDS::index>::max() - 1 - global * 4099; };
    auto mapping = std::make_shared<DNDS::GlobalOffsetsMapping>();
    mapping->setMPIAlignBcast(mpi, 3);
    std::vector<DNDS::index> read;
    for (DNDS::index row = 0; row < 3; ++row)
        read.push_back(key(3 * DNDS::index(mpi.rank) + row));
    const DNDS::index expected = 3 * DNDS::index((mpi.rank + 1) % mpi.size) + 2;
    const std::vector<DNDS::index> query{key(expected)};
    auto retry = [&]
    { CHECK(DNDS::BuildRedistributionPullingIndex(mpi, read, query, mapping) == std::vector<DNDS::index>{expected}); };
    retry();
    auto malformed = read;
    if (mpi.rank == 0)
        malformed[0] = -1;
    CHECK_THROWS(DNDS::BuildRedistributionPullingIndex(mpi, malformed, query, mapping));
    retry();
    malformed = read;
    if (mpi.rank == 0)
        malformed[1] = malformed[0];
    CHECK_THROWS(DNDS::BuildRedistributionPullingIndex(mpi, malformed, query, mapping));
    retry();
    if (mpi.size > 1)
    {
        malformed = read;
        if (mpi.rank == 1)
            malformed[0] = key(0);
        CHECK_THROWS(DNDS::BuildRedistributionPullingIndex(mpi, malformed, query, mapping));
        retry();
    }
    auto missing = query;
    if (mpi.rank == 0)
        missing[0] = std::numeric_limits<DNDS::index>::max() - 2;
    CHECK_THROWS(DNDS::BuildRedistributionPullingIndex(mpi, read, missing, mapping));
    retry();
}

TEST_CASE("Local checks: H5 redistribution preserves sparse and high IDs across ownership changes")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto path = (std::filesystem::temp_directory_path() /
                       ("dndsr-high-id-restart-" + std::to_string(pid) + ".h5"))
                          .string();
    const int owners = std::max(1, mpi.size - 1);
    auto count = [&](int rank) -> DNDS::index
    { return rank < owners ? 3 + rank % 2 : 0; };
    auto first = [](int rank) -> DNDS::index
    { return 3 * DNDS::index(rank) + rank / 2; };
    using Pair = DNDS::ArrayPair<DNDS::ParArray<DNDS::index, 2>>;
    for (bool high : {false, true})
    {
        CAPTURE(high);
        auto key = [&](DNDS::index global)
        {
            return high ? std::numeric_limits<DNDS::index>::max() - 1 - global * 4099
                        : DNDS::index(1000000000000LL) + global * 101;
        };
        {
            Pair pair;
            pair.InitPair("high-ID checkpoint writer", mpi);
            pair.father->Resize(count(mpi.rank));
            pair.son->Resize(0);
            std::vector<DNDS::index> ids;
            for (DNDS::index row = 0; row < count(mpi.rank); ++row)
            {
                const auto global = first(mpi.rank) + row;
                ids.push_back(key(global));
                (*pair.father)(row, 0) = key(global);
                (*pair.father)(row, 1) = global * 13 + 5;
            }
            auto serializer = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
            serializer->OpenFile(path, false);
            pair.WriteSerialize(serializer, "u", ids, false, false);
            serializer->CloseFile();
        }
        {
            Pair pair;
            pair.InitPair("high-ID checkpoint reader", mpi);
            const int owner = (mpi.rank + 1) % mpi.size;
            pair.father->Resize(count(owner));
            pair.son->Resize(0);
            std::vector<DNDS::index> ids;
            for (DNDS::index row = 0; row < count(owner); ++row)
                ids.push_back(key(first(owner) + count(owner) - 1 - row));
            auto serializer = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
            serializer->OpenFile(path, true);
            pair.ReadSerializeRedistributed(serializer, "u", ids);
            serializer->CloseFile();
            REQUIRE(pair.father->Size() == count(owner));
            for (DNDS::index row = 0; row < count(owner); ++row)
            {
                const auto global = first(owner) + count(owner) - 1 - row;
                CHECK((*pair.father)(row, 0) == ids[size_t(row)]);
                CHECK((*pair.father)(row, 1) == global * 13 + 5);
            }
        }
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
            std::filesystem::remove(path);
        MPI_Barrier(mpi.comm);
    }
}

TEST_CASE("Local checks: serialized payload must match array shape")
{
    namespace S = DNDS::Serializer;
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (bool h5 : {false, true})
    {
        int pid = getpid();
        if (h5)
            MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        std::string path = "__local_shape_" + std::to_string(pid) + (h5 ? ".h5" : ".json");
        S::SerializerBaseSSP ser = h5 ? S::SerializerBaseSSP(std::make_shared<S::SerializerH5>(mpi)) : S::SerializerBaseSSP(std::make_shared<S::SerializerJSON>());
        using A = DNDS::Array<DNDS::index, 2>;
        ser->OpenFile(path, false);
        ser->CreatePath("a");
        ser->GoToPath("a");
        ser->WriteString("array_sig", A::GetArraySignature());
        if (h5)
            ser->WriteIndexVectorPerRank("size", {mpi.rank == 0 ? 2 : 1});
        else
            ser->WriteIndex("size", 2);
        ser->WriteInt("row_size_dynamic", 0);
        ser->WriteIndexVector("data", {1, 2}, S::ArrayGlobalOffset_Parts);
        ser->CloseFile();
        ser->OpenFile(path, true);
        A a;
        auto offset = S::ArrayGlobalOffset_Unknown;
        CHECK_THROWS(a.ReadSerializer(ser, "a", offset));
        ser->CloseFile();
        if (!h5 || mpi.rank == 0)
            std::filesystem::remove(path);
        MPI_Barrier(mpi.comm);
    }
}

TEST_CASE("Local checks: JSON CSR terminal and padded widths match payload" * doctest::test_suite("rank-local"))
{
    namespace S = DNDS::Serializer;
    const std::string path = "__local_rows_" + std::to_string(getpid()) + ".json";
    for (bool csr : {false, true})
    {
        using CSR = DNDS::Array<DNDS::index, DNDS::NonUniformSize>;
        using Padded = DNDS::Array<DNDS::index, DNDS::NonUniformSize, 2>;
        auto ser = std::make_shared<S::SerializerJSON>();
        ser->OpenFile(path, false);
        ser->CreatePath("a");
        ser->GoToPath("a");
        ser->WriteString("array_sig", csr ? CSR::GetArraySignature() : Padded::GetArraySignature());
        ser->WriteIndex("size", 1);
        ser->WriteInt("row_size_dynamic", 0);
        if (csr)
        {
            auto starts = std::make_shared<DNDS::host_device_vector<DNDS::index>>(2, 0);
            (*starts)[1] = 3;
            ser->WriteSharedRowStartVector("pRowStart", starts, S::ArrayGlobalOffset_Unknown);
        }
        else
            ser->WriteSharedRowsizeVector("pRowSizes", std::make_shared<DNDS::host_device_vector<DNDS::rowsize>>(1, -1), S::ArrayGlobalOffset_Unknown);
        ser->WriteIndexVector("data", {1, 2}, S::ArrayGlobalOffset_Unknown);
        ser->CloseFile();
        ser->OpenFile(path, true);
        auto offset = S::ArrayGlobalOffset_Unknown;
        if (csr)
        {
            CSR a;
            CHECK_THROWS(a.ReadSerializer(ser, "a", offset));
        }
        else
        {
            Padded a;
            CHECK_THROWS(a.ReadSerializer(ser, "a", offset));
        }
        ser->CloseFile();
    }
    std::filesystem::remove(path);
}

TEST_CASE("Local checks: zero width H5 rows preserve their explicit slice")
{
    namespace S = DNDS::Serializer;
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const std::string path = "__local_zero_width_" + std::to_string(pid) + ".h5";
    auto ser = std::make_shared<S::SerializerH5>(mpi);
    DNDS::Array<DNDS::index, DNDS::DynamicSize> source, result;
    source.Resize(mpi.rank + 1, 0);
    ser->OpenFile(path, false);
    source.WriteSerializer(ser, "a", S::ArrayGlobalOffset_Parts);
    ser->CloseFile();
    ser->OpenFile(path, true);
    auto offset = S::ArrayGlobalOffset{mpi.rank + 1, DNDS::index(mpi.rank) * (mpi.rank + 1) / 2};
    auto original = offset;
    result.ReadSerializer(ser, "a", offset);
    CHECK(result.Size() == mpi.rank + 1);
    CHECK(result.DataSize() == 0);
    CHECK(offset == original);
    ser->CloseFile();
    if (mpi.rank == 0)
        std::filesystem::remove(path);
    MPI_Barrier(mpi.comm);
}

// Opt-in microbenchmark: not part of correctness CTest; use --no-skip.
TEST_CASE("Local timing: length broadcasts versus checked allgather" * doctest::skip())
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    DNDS::GlobalOffsetsMapping gathered;
    std::vector<DNDS::index> lengths(mpi.size), offsets(mpi.size + 1);
    DNDS::index local = mpi.rank + 1;
    auto broadcasts = [&]()
    {
        lengths[mpi.rank] = local;
        for (int rank = 0; rank < mpi.size; ++rank)
            DNDS::MPI::Bcast(&lengths[rank], sizeof(local), MPI_BYTE, rank, mpi.comm);
        offsets[0] = 0;
        for (int rank = 0; rank < mpi.size; ++rank)
            offsets[rank + 1] = offsets[rank] + lengths[rank];
    };
    for (int repeat = 0; repeat < 7; ++repeat)
    {
        double seconds[2]{};
        for (int pass = 0; pass < 2; ++pass)
        {
            const int algorithm = (repeat + pass) % 2;
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            for (int i = 0; i < 2000; ++i)
                if (algorithm == 0)
                    broadcasts();
                else
                    gathered.setMPIAlignBcast(mpi, local);
            seconds[algorithm] = MPI_Wtime() - start;
        }
        MPI_Allreduce(MPI_IN_PLACE, seconds, 2, MPI_DOUBLE, MPI_MAX, mpi.comm);
        CHECK(lengths == gathered.RLengths());
        CHECK(offsets == gathered.ROffsets());
        if (mpi.rank == 0)
            std::cout << "LOCAL_TIMING,np=" << mpi.size << ",repeat=" << repeat
                      << ",bcast_ns=" << seconds[0] * 1e9 / 2000
                      << ",allgather_ns=" << seconds[1] * 1e9 / 2000 << '\n';
    }
}

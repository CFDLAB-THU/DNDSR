#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/ArrayPair.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrixBatch.hpp"
#include <utility>

using namespace DNDS;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context(argc, argv);
    int result = context.run();
    MPI_Finalize();
    return result;
}

TEST_CASE("Audit batch 2: moved vectors remain reusable")
{
    host_device_vector<int> source(2, 42);
    source.to_device(DeviceBackend::Host);
    auto moved = std::move(source);
    CHECK(moved[0] == 42);
    CHECK(moved.dataDevice() == moved.data());
    CHECK(source.size() == 0);
    auto emptyCopy = source;
    CHECK(emptyCopy.size() == 0);
    source.clear();
    source.resize(3, 7);
    source.to_device(DeviceBackend::Host);
    CHECK(source.dataDevice() == source.data());
    CHECK(source[2] == 7);
    moved = std::move(source);
    source.resize(1, 9);
    CHECK(source[0] == 9);
    CHECK(moved.size() == 3);
    moved = std::move(moved);
    CHECK(moved[2] == 7);
}

template <DNDS::rowsize rs, DNDS::rowsize rm>
static void CheckArrayMove()
{
    Array<int, rs, rm> source;
    source.Resize(2, 3);
    if constexpr (rs == NonUniformSize)
    {
        source.ResizeRow(0, 1);
        source.ResizeRow(1, 2);
    }
    source(0, 0) = 42;
    source.Compress();
    auto moved = std::move(source);
    CHECK(source.Size() == 0);
    CHECK(source.DataSize() == 0);
    CHECK(moved(0, 0) == 42);
    source.Resize(1, 3);
    if constexpr (rs == NonUniformSize)
        source.ResizeRow(0, 1);
    source(0, 0) = 7;
    moved = std::move(source);
    CHECK(source.Size() == 0);
    CHECK(moved(0, 0) == 7);
    moved = std::move(moved);
    CHECK(moved.Size() == 1);
    CHECK(moved(0, 0) == 7);
}

TEST_CASE("Audit batch 2: moved arrays reset structural state")
{
    CheckArrayMove<3, 3>();
    CheckArrayMove<DynamicSize, DynamicSize>();
    CheckArrayMove<NonUniformSize, 3>();
    CheckArrayMove<NonUniformSize, DynamicSize>();
    CheckArrayMove<NonUniformSize, NonUniformSize>();
}

TEST_CASE("Audit batch 2: swaps reject unequal row and matrix shapes")
{
    Array<int, NonUniformSize, 3> a, b;
    a.Resize(2);
    b.Resize(2);
    a.ResizeRow(0, 1);
    a.ResizeRow(1, 2);
    b.ResizeRow(0, 2);
    b.ResizeRow(1, 1);
    a(0, 0) = 11;
    b(0, 0) = 22;
    CHECK_THROWS(a.SwapData(b));
    CHECK(a(0, 0) == 11);
    CHECK(b(0, 0) == 22);
    b.ResizeRow(0, 1);
    b.ResizeRow(1, 2);
    a.SwapData(b);
    CHECK(a(0, 0) == 22);
    a.SwapData(b);
    CHECK(a(0, 0) == 11);

    Array<int, NonUniformSize> c, d;
    c.Resize(2);
    d.Resize(2);
    c.ResizeRow(0, 1);
    c.ResizeRow(1, 2);
    d.ResizeRow(0, 2);
    d.ResizeRow(1, 1);
    c(0, 0) = 33;
    d(0, 0) = 44;
    CHECK_THROWS(c.SwapData(d));
    c.Compress();
    d.Compress();
    CHECK_THROWS(c.SwapData(d));
    CHECK(c(0, 0) == 33);
    CHECK(d(0, 0) == 44);

    MPIInfo mpi(MPI_COMM_WORLD);
    ArrayEigenMatrix<DynamicSize, DynamicSize> m(mpi), n(mpi);
    m.Resize(1, 2, 3);
    n.Resize(1, 3, 2);
    m[0].setConstant(5);
    n[0].setConstant(6);
    CHECK_THROWS(m.SwapData(n));
    CHECK(m[0](0, 0) == 5);
    CHECK(m.MatRowSize() == 2);

    ArrayEigenMatrix<NonUniformSize, NonUniformSize> u(mpi), v(mpi);
    u.Resize(1, 2, 3);
    v.Resize(1, 3, 2);
    u.Compress();
    v.Compress();
    CHECK_THROWS(u.SwapData(v));
    CHECK(u.MatRowSize(0) == 2);

    ArrayEigenUniMatrixBatch<Eigen::Dynamic, Eigen::Dynamic> batchA(mpi), batchB(mpi);
    batchA.Resize(1, 2, 3);
    batchB.Resize(1, 3, 2);
    batchA.ResizeBatch(0, 1);
    batchB.ResizeBatch(0, 1);
    batchA.Compress();
    batchB.Compress();
    CHECK_THROWS(batchA.SwapData(batchB));

    ArrayEigenMatrixBatch variedA(mpi), variedB(mpi);
    variedA.Resize(1);
    variedB.Resize(1);
    variedA.InitializeWriteRow(0, std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Constant(2, 3, 7)});
    variedB.InitializeWriteRow(0, std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Constant(3, 2, 8)});
    CHECK_THROWS(variedA.SwapData(variedB));
    CHECK(variedA(0, 0)(0, 0) == 7);
    variedA.Compress();
    variedB.Compress();
    CHECK_THROWS(variedA.SwapData(variedB));
    CHECK(variedA(0, 0).rows() == 2);
    variedB.Decompress();
    variedB.InitializeWriteRow(0, std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Constant(2, 3, 8)});
    variedB.Compress();
    CHECK_NOTHROW(variedA.SwapData(variedB));
    CHECK(variedA(0, 0)(0, 0) == 8);
    CHECK(variedB(0, 0)(0, 0) == 7);
}

TEST_CASE("Audit batch 2: pair swaps validate both arrays before mutation")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    using Pair = ArrayPair<ParArray<int, NonUniformSize, 3>>;
    Pair a, b;
    a.InitPair("audit swap a", mpi);
    b.InitPair("audit swap b", mpi);
    for (auto *pair : {&a, &b})
    {
        pair->father->Resize(1);
        pair->father->ResizeRow(0, 1);
        pair->son->Resize(1);
    }
    a.son->ResizeRow(0, 1);
    b.son->ResizeRow(0, 2);
    (*a.father)(0, 0) = 10;
    (*b.father)(0, 0) = 20;
    CHECK_THROWS(a.SwapDataFatherSon(b));
    CHECK((*a.father)(0, 0) == 10);
    CHECK((*b.father)(0, 0) == 20);
}

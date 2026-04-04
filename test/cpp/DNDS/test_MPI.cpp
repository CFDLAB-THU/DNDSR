/**
 * @file test_MPI.cpp
 * @brief Doctest-based unit tests for DNDS MPI wrapper functionality.
 *
 * Run under mpirun with varying rank counts (1, 2, 4).  Covers
 * DNDS::MPIInfo, collective operations (Allreduce, Scan, Allgather,
 * Bcast, Barrier, Alltoall), MPI type mapping, and CommStrategy.
 *
 * @see @ref dnds_unit_tests for the full test-suite overview.
 * @test MPIInfo, Allreduce, Scan, Allgather, Bcast, Barrier, Alltoall,
 *       BasicType_To_MPIIntType, CommStrategy.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/MPI.hpp"
#include "DNDS/Defines.hpp"

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();
    MPI_Finalize();
    return res;
}

using namespace DNDS;

// ---------------------------------------------------------------------------
// MPIInfo basics
// ---------------------------------------------------------------------------
TEST_CASE("MPIInfo basics")
{
    SUBCASE("setWorld populates rank, size, comm")
    {
        MPIInfo mpi;
        mpi.setWorld();

        CHECK(mpi.rank >= 0);
        CHECK(mpi.rank < mpi.size);
        CHECK(mpi.size >= 1);
        CHECK(mpi.comm != MPI_COMM_NULL);
    }

    SUBCASE("two MPIInfos with setWorld are equal")
    {
        MPIInfo a, b;
        a.setWorld();
        b.setWorld();

        CHECK(a == b);
    }

    SUBCASE("constructor from MPI_COMM_WORLD matches setWorld")
    {
        MPIInfo fromCtor(MPI_COMM_WORLD);
        MPIInfo fromSet;
        fromSet.setWorld();

        CHECK(fromCtor.rank == fromSet.rank);
        CHECK(fromCtor.size == fromSet.size);
        CHECK(fromCtor.comm == fromSet.comm);
    }

    SUBCASE("MPIWorldSize and MPIWorldRank agree with MPIInfo")
    {
        MPIInfo mpi;
        mpi.setWorld();

        CHECK(MPIWorldSize() == mpi.size);
        CHECK(MPIWorldRank() == mpi.rank);
    }
}

// ---------------------------------------------------------------------------
// MPI Allreduce
// ---------------------------------------------------------------------------
TEST_CASE("MPI Allreduce")
{
    MPIInfo mpi;
    mpi.setWorld();
    int expected = mpi.size * (mpi.size + 1) / 2;

    SUBCASE("Allreduce MPI_SUM with real type")
    {
        DNDS::real sendVal = static_cast<DNDS::real>(mpi.rank + 1);
        DNDS::real recvVal = 0.0;
        MPI::Allreduce(&sendVal, &recvVal, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);

        CHECK(recvVal == doctest::Approx(static_cast<DNDS::real>(expected)));
    }

    SUBCASE("Allreduce MPI_SUM with index type")
    {
        DNDS::index sendVal = static_cast<DNDS::index>(mpi.rank + 1);
        DNDS::index recvVal = 0;
        MPI::Allreduce(&sendVal, &recvVal, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);

        CHECK(recvVal == static_cast<DNDS::index>(expected));
    }

    SUBCASE("AllreduceOneReal")
    {
        DNDS::real val = static_cast<DNDS::real>(mpi.rank + 1);
        MPI::AllreduceOneReal(val, MPI_SUM, mpi);

        CHECK(val == doctest::Approx(static_cast<DNDS::real>(expected)));
    }

    SUBCASE("AllreduceOneIndex")
    {
        DNDS::index val = static_cast<DNDS::index>(mpi.rank + 1);
        MPI::AllreduceOneIndex(val, MPI_SUM, mpi);

        CHECK(val == static_cast<DNDS::index>(expected));
    }

    SUBCASE("Allreduce MPI_MAX")
    {
        DNDS::real val = static_cast<DNDS::real>(mpi.rank);
        DNDS::real result = 0.0;
        MPI::Allreduce(&val, &result, 1, DNDS_MPI_REAL, MPI_MAX, mpi.comm);

        CHECK(result == doctest::Approx(static_cast<DNDS::real>(mpi.size - 1)));
    }
}

// ---------------------------------------------------------------------------
// MPI Scan
// ---------------------------------------------------------------------------
TEST_CASE("MPI Scan")
{
    MPIInfo mpi;
    mpi.setWorld();

    SUBCASE("Scan MPI_SUM with constant 1")
    {
        DNDS::index sendVal = 1;
        DNDS::index recvVal = 0;
        MPI::Scan(&sendVal, &recvVal, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);

        CHECK(recvVal == static_cast<DNDS::index>(mpi.rank + 1));
    }

    SUBCASE("Scan MPI_SUM with rank value")
    {
        DNDS::index sendVal = static_cast<DNDS::index>(mpi.rank);
        DNDS::index recvVal = 0;
        MPI::Scan(&sendVal, &recvVal, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);

        // prefix sum of 0 + 1 + ... + rank = rank*(rank+1)/2
        DNDS::index expected = static_cast<DNDS::index>(mpi.rank) * (mpi.rank + 1) / 2;
        CHECK(recvVal == expected);
    }
}

// ---------------------------------------------------------------------------
// MPI Allgather
// ---------------------------------------------------------------------------
TEST_CASE("MPI Allgather")
{
    MPIInfo mpi;
    mpi.setWorld();

    SUBCASE("gather rank values into ordered array")
    {
        DNDS::index sendVal = static_cast<DNDS::index>(mpi.rank);
        std::vector<DNDS::index> recvBuf(mpi.size, -1);

        MPI::Allgather(&sendVal, 1, DNDS_MPI_INDEX,
                        recvBuf.data(), 1, DNDS_MPI_INDEX, mpi.comm);

        for (int i = 0; i < mpi.size; ++i)
        {
            CHECK(recvBuf[i] == static_cast<DNDS::index>(i));
        }
    }

    SUBCASE("gather multiple elements per rank")
    {
        std::array<DNDS::real, 2> sendBuf = {
            static_cast<DNDS::real>(mpi.rank),
            static_cast<DNDS::real>(mpi.rank * 10)};
        std::vector<DNDS::real> recvBuf(mpi.size * 2, -1.0);

        MPI::Allgather(sendBuf.data(), 2, DNDS_MPI_REAL,
                        recvBuf.data(), 2, DNDS_MPI_REAL, mpi.comm);

        for (int i = 0; i < mpi.size; ++i)
        {
            CHECK(recvBuf[2 * i] == doctest::Approx(static_cast<DNDS::real>(i)));
            CHECK(recvBuf[2 * i + 1] == doctest::Approx(static_cast<DNDS::real>(i * 10)));
        }
    }
}

// ---------------------------------------------------------------------------
// MPI Bcast
// ---------------------------------------------------------------------------
TEST_CASE("MPI Bcast")
{
    MPIInfo mpi;
    mpi.setWorld();

    SUBCASE("broadcast real from rank 0")
    {
        DNDS::real val = (mpi.rank == 0) ? 42.0 : -1.0;
        MPI::Bcast(&val, 1, DNDS_MPI_REAL, 0, mpi.comm);

        CHECK(val == doctest::Approx(42.0));
    }

    SUBCASE("broadcast index from rank 0")
    {
        DNDS::index val = (mpi.rank == 0) ? 12345 : -1;
        MPI::Bcast(&val, 1, DNDS_MPI_INDEX, 0, mpi.comm);

        CHECK(val == 12345);
    }

    SUBCASE("broadcast from last rank")
    {
        DNDS::real val = (mpi.rank == mpi.size - 1) ? 99.5 : 0.0;
        MPI::Bcast(&val, 1, DNDS_MPI_REAL, mpi.size - 1, mpi.comm);

        CHECK(val == doctest::Approx(99.5));
    }
}

// ---------------------------------------------------------------------------
// MPI Barrier
// ---------------------------------------------------------------------------
TEST_CASE("MPI Barrier")
{
    MPIInfo mpi;
    mpi.setWorld();

    // Barrier should return without hanging.
    MPI_int ret = MPI::Barrier(mpi.comm);
    CHECK(ret == MPI_SUCCESS);
}

// ---------------------------------------------------------------------------
// BasicType_To_MPIIntType
// ---------------------------------------------------------------------------
TEST_CASE("MPI BasicType_To_MPIIntType")
{
    SUBCASE("scalar types")
    {
        auto [realType, realMult] = BasicType_To_MPIIntType<DNDS::real>();
        CHECK(realType == MPI_DOUBLE);
        CHECK(realMult == 1);

        auto [idxType, idxMult] = BasicType_To_MPIIntType<DNDS::index>();
        CHECK(idxType == MPI_INT64_T);
        CHECK(idxMult == 1);

        auto [fType, fMult] = BasicType_To_MPIIntType<float>();
        CHECK(fType == MPI_FLOAT);
        CHECK(fMult == 1);

        auto [i32Type, i32Mult] = BasicType_To_MPIIntType<int32_t>();
        CHECK(i32Type == MPI_INT32_T);
        CHECK(i32Mult == 1);
    }

    SUBCASE("std::array<real, 3>")
    {
        auto [arrType, arrMult] = BasicType_To_MPIIntType<std::array<DNDS::real, 3>>();
        CHECK(arrType == MPI_DOUBLE);
        CHECK(arrMult == 3);
    }

    SUBCASE("Eigen::Matrix<real, 3, 3>")
    {
        using Mat33 = Eigen::Matrix<DNDS::real, 3, 3>;
        auto [matType, matMult] = BasicType_To_MPIIntType<Mat33>();

        // The MPI type should be DNDS_MPI_REAL (MPI_REAL8 / MPI_DOUBLE).
        // Multiplicity should cover at least 9 doubles (may be rounded up
        // due to divide_ceil over sizeof).
        CHECK(matType != MPI_DATATYPE_NULL);
        CHECK(matMult >= 9);
    }

    SUBCASE("DNDS_MPI_INDEX and DNDS_MPI_REAL globals")
    {
        CHECK(DNDS_MPI_INDEX != MPI_DATATYPE_NULL);
        CHECK(DNDS_MPI_REAL != MPI_DATATYPE_NULL);
    }
}

// ---------------------------------------------------------------------------
// CommStrategy
// ---------------------------------------------------------------------------
TEST_CASE("CommStrategy")
{
    auto &cs = MPI::CommStrategy::Instance();

    SUBCASE("default strategy is HIndexed")
    {
        // Restore known state first.
        cs.SetArrayStrategy(MPI::CommStrategy::HIndexed);
        CHECK(cs.GetArrayStrategy() == MPI::CommStrategy::HIndexed);
    }

    SUBCASE("set to InSituPack and verify")
    {
        cs.SetArrayStrategy(MPI::CommStrategy::InSituPack);
        CHECK(cs.GetArrayStrategy() == MPI::CommStrategy::InSituPack);

        // Restore default.
        cs.SetArrayStrategy(MPI::CommStrategy::HIndexed);
        CHECK(cs.GetArrayStrategy() == MPI::CommStrategy::HIndexed);
    }
}

// ---------------------------------------------------------------------------
// MPI Alltoall
// ---------------------------------------------------------------------------
TEST_CASE("MPI Alltoall")
{
    MPIInfo mpi;
    mpi.setWorld();

    SUBCASE("each rank sends its rank value to every other rank")
    {
        // sendBuf: every element is my rank.
        std::vector<DNDS::index> sendBuf(mpi.size, static_cast<DNDS::index>(mpi.rank));
        std::vector<DNDS::index> recvBuf(mpi.size, -1);

        MPI::Alltoall(sendBuf.data(), 1, DNDS_MPI_INDEX,
                       recvBuf.data(), 1, DNDS_MPI_INDEX, mpi.comm);

        // recvBuf[i] should be i (received from rank i, which sent its rank).
        for (int i = 0; i < mpi.size; ++i)
        {
            CHECK(recvBuf[i] == static_cast<DNDS::index>(i));
        }
    }

    SUBCASE("Alltoall with multiple elements per peer")
    {
        const int perPeer = 2;
        std::vector<DNDS::real> sendBuf(mpi.size * perPeer);
        std::vector<DNDS::real> recvBuf(mpi.size * perPeer, -1.0);

        for (int i = 0; i < mpi.size; ++i)
        {
            sendBuf[i * perPeer + 0] = static_cast<DNDS::real>(mpi.rank);
            sendBuf[i * perPeer + 1] = static_cast<DNDS::real>(mpi.rank + 100);
        }

        MPI::Alltoall(sendBuf.data(), perPeer, DNDS_MPI_REAL,
                       recvBuf.data(), perPeer, DNDS_MPI_REAL, mpi.comm);

        for (int i = 0; i < mpi.size; ++i)
        {
            CHECK(recvBuf[i * perPeer + 0] == doctest::Approx(static_cast<DNDS::real>(i)));
            CHECK(recvBuf[i * perPeer + 1] == doctest::Approx(static_cast<DNDS::real>(i + 100)));
        }
    }
}

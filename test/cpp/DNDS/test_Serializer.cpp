/**
 * @file test_Serializer.cpp
 * @brief Doctest-based unit tests for DNDS Serializer classes.
 *
 * Covers SerializerJSON (per-rank, no MPI needed for logic) and
 * SerializerH5 (parallel HDF5, requires MPI).  The JSON tests verify
 * scalar, vector, uint8 (with/without codec), path operations, and
 * shared-pointer deduplication.  The H5 tests verify scalar, vector,
 * distributed vector (non-uniform per-rank sizes), uint8 (two-pass read),
 * path operations, and string round-trips.
 *
 * @note H5 parallel I/O requires all MPI ranks to open the same file.
 * The TmpH5() helper broadcasts rank 0's PID so that the filename is
 * identical across all ranks.  FileGuard uses shared=true for H5 files
 * so only rank 0 deletes the file, followed by an MPI_Barrier.
 *
 * Build:  `cmake --build . -t dnds_test_serializer -j8`
 * Run:    `mpirun -np 1 ./dnds_test_serializer`  (JSON + H5 single rank)
 *         `mpirun -np 4 ./dnds_test_serializer`  (H5 distributed tests)
 *
 * @see @ref dnds_unit_tests for the full test-suite overview.
 * @test SerializerJSON scalar/vector/uint8/path/pointer round-trip,
 *       SerializerH5 scalar/vector/distributed/uint8/path/string round-trip.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/SerializerJSON.hpp"
#include "DNDS/SerializerH5.hpp"
#include "DNDS/MPI.hpp"
#include <filesystem>
#include <numeric>
#include <algorithm>

using namespace DNDS;
namespace S = DNDS::Serializer;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();
    MPI_Finalize();
    return res;
}

// Helper: build a unique temp file name that won't collide across ranks or runs.
static std::string TmpJSON(const std::string &tag)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return fmt::format("__test_ser_{}_{}_r{}.json", tag, ::getpid(), rank);
}

static std::string TmpH5(const std::string &tag)
{
    // H5 parallel I/O requires ALL ranks to open the same file.
    // Broadcast rank 0's PID so the filename is identical everywhere.
    int pid = 0;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        pid = ::getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return fmt::format("__test_ser_{}_{}.h5", tag, pid);
}

// RAII guard that removes a file on destruction.
// For H5 files (shared across ranks), only rank 0 should remove.
struct FileGuard
{
    std::string path;
    bool ownerOnly; // if true, only rank 0 removes
    explicit FileGuard(std::string p, bool shared = false)
        : path(std::move(p)), ownerOnly(shared)
    {
    }
    ~FileGuard()
    {
        if (ownerOnly)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
                fs::remove(path);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else
        {
            fs::remove(path);
        }
    }
};

// ===================================================================
// SerializerJSON — scalar round-trip
// ===================================================================
TEST_CASE("SerializerJSON scalar round-trip")
{
    std::string fname = TmpJSON("scalar");
    FileGuard guard(fname);

    // --- Write ---
    {
        S::SerializerJSON ser;
        ser.OpenFile(fname, false);
        ser.CreatePath("/test");
        ser.GoToPath("/test");

        ser.WriteInt("myInt", 42);
        ser.WriteIndex("myIdx", 123456789LL);
        ser.WriteReal("myReal", 3.14);
        ser.WriteString("myStr", "hello");

        ser.CloseFile();
    }

    // --- Read ---
    {
        S::SerializerJSON ser;
        ser.OpenFile(fname, true);
        ser.GoToPath("/test");

        int rInt = 0;
        DNDS::index rIdx = 0;
        real rReal = 0.0;
        std::string rStr;

        ser.ReadInt("myInt", rInt);
        ser.ReadIndex("myIdx", rIdx);
        ser.ReadReal("myReal", rReal);
        ser.ReadString("myStr", rStr);

        CHECK(rInt == 42);
        CHECK(rIdx == 123456789LL);
        CHECK(rReal == doctest::Approx(3.14));
        CHECK(rStr == "hello");

        ser.CloseFile();
    }
}

// ===================================================================
// SerializerJSON — vector round-trip
// ===================================================================
TEST_CASE("SerializerJSON vector round-trip")
{
    std::string fname = TmpJSON("vec");
    FileGuard guard(fname);

    const std::vector<real> wReals = {1.1, 2.2, 3.3, 4.4, 5.5};
    const std::vector<DNDS::index> wIndices = {10, 20, 30};
    const std::vector<rowsize> wRows = {1, 2, 3, 4};

    // --- Write ---
    {
        S::SerializerJSON ser;
        ser.OpenFile(fname, false);
        ser.CreatePath("/vectors");
        ser.GoToPath("/vectors");

        ser.WriteRealVector("reals", wReals, S::ArrayGlobalOffset_Unknown);
        ser.WriteIndexVector("indices", wIndices, S::ArrayGlobalOffset_Unknown);
        ser.WriteRowsizeVector("rows", wRows, S::ArrayGlobalOffset_Unknown);

        ser.CloseFile();
    }

    // --- Read ---
    {
        S::SerializerJSON ser;
        ser.OpenFile(fname, true);
        ser.GoToPath("/vectors");

        std::vector<real> rReals;
        std::vector<DNDS::index> rIndices;
        std::vector<rowsize> rRows;
        S::ArrayGlobalOffset off = S::ArrayGlobalOffset_Unknown;

        ser.ReadRealVector("reals", rReals, off);
        off = S::ArrayGlobalOffset_Unknown;
        ser.ReadIndexVector("indices", rIndices, off);
        off = S::ArrayGlobalOffset_Unknown;
        ser.ReadRowsizeVector("rows", rRows, off);

        REQUIRE(rReals.size() == wReals.size());
        for (size_t i = 0; i < wReals.size(); ++i)
            CHECK(rReals[i] == doctest::Approx(wReals[i]));

        REQUIRE(rIndices.size() == wIndices.size());
        for (size_t i = 0; i < wIndices.size(); ++i)
            CHECK(rIndices[i] == wIndices[i]);

        REQUIRE(rRows.size() == wRows.size());
        for (size_t i = 0; i < wRows.size(); ++i)
            CHECK(rRows[i] == wRows[i]);

        ser.CloseFile();
    }
}

// ===================================================================
// SerializerJSON — uint8 array round-trip (with and without codec)
// ===================================================================
TEST_CASE("SerializerJSON uint8 array round-trip")
{
    // Prepare a known byte pattern 0..255 repeated to 512 bytes.
    std::vector<uint8_t> pattern(512);
    for (size_t i = 0; i < pattern.size(); ++i)
        pattern[i] = static_cast<uint8_t>(i & 0xFF);

    SUBCASE("without codec")
    {
        std::string fname = TmpJSON("u8_raw");
        FileGuard guard(fname);

        {
            S::SerializerJSON ser;
            ser.SetUseCodecOnUint8(false);
            ser.OpenFile(fname, false);
            ser.CreatePath("/data");
            ser.GoToPath("/data");
            ser.WriteUint8Array("bytes", pattern.data(),
                                static_cast<DNDS::index>(pattern.size()),
                                S::ArrayGlobalOffset_Unknown);
            ser.CloseFile();
        }

        {
            S::SerializerJSON ser;
            ser.OpenFile(fname, true);
            ser.GoToPath("/data");

            DNDS::index sz = 0;
            S::ArrayGlobalOffset off = S::ArrayGlobalOffset_Unknown;
            // First call: get size only (nullptr)
            ser.ReadUint8Array("bytes", nullptr, sz, off);
            REQUIRE(sz == static_cast<DNDS::index>(pattern.size()));

            std::vector<uint8_t> readBack(sz);
            off = S::ArrayGlobalOffset_Unknown;
            ser.ReadUint8Array("bytes", readBack.data(), sz, off);
            CHECK(readBack == pattern);

            ser.CloseFile();
        }
    }

    SUBCASE("with codec (base64 + zlib)")
    {
        std::string fname = TmpJSON("u8_codec");
        FileGuard guard(fname);

        {
            S::SerializerJSON ser;
            ser.SetUseCodecOnUint8(true);
            ser.SetDeflateLevel(5);
            ser.OpenFile(fname, false);
            ser.CreatePath("/data");
            ser.GoToPath("/data");
            ser.WriteUint8Array("bytes", pattern.data(),
                                static_cast<DNDS::index>(pattern.size()),
                                S::ArrayGlobalOffset_Unknown);
            ser.CloseFile();
        }

        {
            S::SerializerJSON ser;
            ser.OpenFile(fname, true);
            ser.GoToPath("/data");

            DNDS::index sz = 0;
            S::ArrayGlobalOffset off = S::ArrayGlobalOffset_Unknown;
            ser.ReadUint8Array("bytes", nullptr, sz, off);
            REQUIRE(sz == static_cast<DNDS::index>(pattern.size()));

            std::vector<uint8_t> readBack(sz);
            off = S::ArrayGlobalOffset_Unknown;
            ser.ReadUint8Array("bytes", readBack.data(), sz, off);
            CHECK(readBack == pattern);

            ser.CloseFile();
        }
    }
}

// ===================================================================
// SerializerJSON — path operations
// ===================================================================
TEST_CASE("SerializerJSON path operations")
{
    std::string fname = TmpJSON("paths");
    FileGuard guard(fname);

    S::SerializerJSON ser;
    ser.OpenFile(fname, false);

    // Create nested hierarchy
    ser.CreatePath("/a");
    ser.GoToPath("/a");
    ser.CreatePath("b");
    ser.GoToPath("b");
    ser.CreatePath("c");
    ser.CreatePath("d");

    // Current path should be /a/b
    CHECK(ser.GetCurrentPath() == "/a/b");

    // Listing /a/b should contain "c" and "d"
    auto entries = ser.ListCurrentPath();
    CHECK(entries.count("c") == 1);
    CHECK(entries.count("d") == 1);

    // Go to /a/b/c and write something to verify it's valid
    ser.GoToPath("c");
    CHECK(ser.GetCurrentPath() == "/a/b/c");
    ser.WriteInt("val", 99);

    ser.CloseFile();

    // Verify the written value survived
    {
        S::SerializerJSON reader;
        reader.OpenFile(fname, true);
        reader.GoToPath("/a/b/c");
        int v = 0;
        reader.ReadInt("val", v);
        CHECK(v == 99);
        reader.CloseFile();
    }
}

// ===================================================================
// SerializerJSON — shared pointer deduplication
// ===================================================================
TEST_CASE("SerializerJSON shared pointer deduplication")
{
    std::string fname = TmpJSON("shared");
    FileGuard guard(fname);

    // Build a shared index vector
    auto sharedVec = std::make_shared<host_device_vector<DNDS::index>>();
    sharedVec->resize(5);
    for (size_t i = 0; i < 5; ++i)
        (*sharedVec)[i] = static_cast<DNDS::index>(100 + i);

    // --- Write: same shared_ptr under two names ---
    {
        S::SerializerJSON ser;
        ser.OpenFile(fname, false);
        ser.CreatePath("/dedup");
        ser.GoToPath("/dedup");

        ser.WriteSharedIndexVector("first", sharedVec, S::ArrayGlobalOffset_Unknown);
        ser.WriteSharedIndexVector("second", sharedVec, S::ArrayGlobalOffset_Unknown);

        ser.CloseFile();
    }

    // --- Read back both ---
    {
        S::SerializerJSON ser;
        ser.OpenFile(fname, true);
        ser.GoToPath("/dedup");

        DNDS::ssp<host_device_vector<DNDS::index>> readFirst;
        DNDS::ssp<host_device_vector<DNDS::index>> readSecond;
        S::ArrayGlobalOffset off = S::ArrayGlobalOffset_Unknown;

        ser.ReadSharedIndexVector("first", readFirst, off);
        off = S::ArrayGlobalOffset_Unknown;
        ser.ReadSharedIndexVector("second", readSecond, off);

        REQUIRE(readFirst);
        REQUIRE(readSecond);

        // Both should resolve to the same underlying data (deduplication)
        CHECK(readFirst.get() == readSecond.get());

        // Values should match
        REQUIRE(readFirst->size() == 5);
        for (size_t i = 0; i < 5; ++i)
            CHECK((*readFirst)[i] == static_cast<DNDS::index>(100 + i));

        ser.CloseFile();
    }
}

// ===================================================================
// SerializerH5 — scalar round-trip
// ===================================================================
TEST_CASE("SerializerH5 scalar round-trip")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    std::string fname = TmpH5("h5scalar");
    FileGuard guard(fname, true);

    // --- Write ---
    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, false);
        ser.CreatePath("/scalars");
        ser.GoToPath("/scalars");

        ser.WriteInt("myInt", 42);
        ser.WriteIndex("myIdx", 987654321LL);
        ser.WriteReal("myReal", 2.718281828);
        ser.WriteString("myStr", "world");

        ser.CloseFile();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- Read ---
    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, true);
        ser.GoToPath("/scalars");

        int rInt = 0;
        DNDS::index rIdx = 0;
        real rReal = 0.0;
        std::string rStr;

        ser.ReadInt("myInt", rInt);
        ser.ReadIndex("myIdx", rIdx);
        ser.ReadReal("myReal", rReal);
        ser.ReadString("myStr", rStr);

        CHECK(rInt == 42);
        CHECK(rIdx == 987654321LL);
        CHECK(rReal == doctest::Approx(2.718281828));
        CHECK(rStr == "world");

        ser.CloseFile();
    }
}

// ===================================================================
// SerializerH5 — vector round-trip (each rank writes its own portion)
// ===================================================================
TEST_CASE("SerializerH5 vector round-trip")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    std::string fname = TmpH5("h5vec");
    FileGuard guard(fname, true);

    const DNDS::index N = 10; // elements per rank
    DNDS::index globalSize = static_cast<DNDS::index>(mpi.size) * N;
    DNDS::index myOffset = static_cast<DNDS::index>(mpi.rank) * N;

    // Build local data
    std::vector<real> wReals(N);
    std::vector<DNDS::index> wIndices(N);
    for (DNDS::index i = 0; i < N; ++i)
    {
        wReals[i] = static_cast<real>(myOffset + i) * 0.1;
        wIndices[i] = myOffset + i;
    }

    S::ArrayGlobalOffset distOff(globalSize, myOffset);

    // --- Write ---
    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, false);
        ser.CreatePath("/vecs");
        ser.GoToPath("/vecs");

        ser.WriteRealVector("reals", wReals, distOff);
        ser.WriteIndexVector("indices", wIndices, distOff);

        ser.CloseFile();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- Read (always use ArrayGlobalOffset_Unknown; H5 auto-detects from ::rank_offsets) ---
    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, true);
        ser.GoToPath("/vecs");

        std::vector<real> rReals;
        std::vector<DNDS::index> rIndices;
        S::ArrayGlobalOffset offR = S::ArrayGlobalOffset_Unknown;
        S::ArrayGlobalOffset offI = S::ArrayGlobalOffset_Unknown;

        ser.ReadRealVector("reals", rReals, offR);
        ser.ReadIndexVector("indices", rIndices, offI);

        REQUIRE(rReals.size() == static_cast<size_t>(N));
        for (DNDS::index i = 0; i < N; ++i)
            CHECK(rReals[i] == doctest::Approx(wReals[i]));

        REQUIRE(rIndices.size() == static_cast<size_t>(N));
        for (DNDS::index i = 0; i < N; ++i)
            CHECK(rIndices[i] == wIndices[i]);

        ser.CloseFile();
    }
}

// ===================================================================
// SerializerH5 — distributed vector (unknown offset on read)
// ===================================================================
TEST_CASE("SerializerH5 distributed vector")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    std::string fname = TmpH5("h5dist");
    FileGuard guard(fname, true);

    // Each rank writes a variable-size portion.
    // Rank r writes (r + 1) * 3 elements.
    const DNDS::index localN = static_cast<DNDS::index>(mpi.rank + 1) * 3;

    // Compute global size and per-rank offset via MPI scan.
    DNDS::index localN64 = localN;
    DNDS::index myOffset = 0;
    MPI_Scan(&localN64, &myOffset, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    myOffset -= localN64; // exclusive prefix sum
    DNDS::index globalSize = 0;
    MPI_Allreduce(&localN64, &globalSize, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    S::ArrayGlobalOffset distOff(globalSize, myOffset);

    // Fill local data: each element is (globalIndex * 7 + 3)
    std::vector<DNDS::index> wData(localN);
    for (DNDS::index i = 0; i < localN; ++i)
        wData[i] = (myOffset + i) * 7 + 3;

    // --- Write ---
    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, false);
        ser.CreatePath("/dist");
        ser.GoToPath("/dist");

        ser.WriteIndexVector("data", wData, distOff);

        ser.CloseFile();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- Read with Unknown offset (auto-detect from rank_offsets) ---
    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, true);
        ser.GoToPath("/dist");

        std::vector<DNDS::index> rData;
        S::ArrayGlobalOffset offR = S::ArrayGlobalOffset_Unknown;

        ser.ReadIndexVector("data", rData, offR);

        REQUIRE(rData.size() == static_cast<size_t>(localN));
        for (DNDS::index i = 0; i < localN; ++i)
            CHECK(rData[i] == (myOffset + i) * 7 + 3);

        ser.CloseFile();
    }

    // --- Read with explicit known offset (pre-allocate the vector) ---
    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, true);
        ser.GoToPath("/dist");

        std::vector<DNDS::index> rData;
        S::ArrayGlobalOffset offR = S::ArrayGlobalOffset_Unknown;

        ser.ReadIndexVector("data", rData, offR);

        REQUIRE(rData.size() == static_cast<size_t>(localN));
        for (DNDS::index i = 0; i < localN; ++i)
            CHECK(rData[i] == (myOffset + i) * 7 + 3);

        ser.CloseFile();
    }
}

// ===================================================================
// SerializerH5 — uint8 distributed round-trip
// ===================================================================
TEST_CASE("SerializerH5 uint8 distributed round-trip")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    std::string fname = TmpH5("h5u8");
    FileGuard guard(fname, true);

    const DNDS::index localN = 64;
    DNDS::index globalSize = static_cast<DNDS::index>(mpi.size) * localN;
    DNDS::index myOffset = static_cast<DNDS::index>(mpi.rank) * localN;

    S::ArrayGlobalOffset distOff(globalSize, myOffset);

    // Fill local bytes: cyclic pattern seeded by rank
    std::vector<uint8_t> wBytes(localN);
    for (DNDS::index i = 0; i < localN; ++i)
        wBytes[i] = static_cast<uint8_t>((myOffset + i) & 0xFF);

    // --- Write ---
    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, false);
        ser.CreatePath("/bytes");
        ser.GoToPath("/bytes");

        ser.WriteUint8Array("raw", wBytes.data(), localN, distOff);

        ser.CloseFile();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- Read (two-pass: nullptr to query size, then actual read with same offset) ---
    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, true);
        ser.GoToPath("/bytes");

        // Pass 1: query size (data=nullptr)
        DNDS::index sz = 0;
        S::ArrayGlobalOffset offR = S::ArrayGlobalOffset_Unknown;
        ser.ReadUint8Array("raw", nullptr, sz, offR);
        REQUIRE(sz == localN);

        // Pass 2: actual read (reuse offR from pass 1, do NOT reset to Unknown)
        std::vector<uint8_t> rBytes(sz);
        ser.ReadUint8Array("raw", rBytes.data(), sz, offR);

        REQUIRE(rBytes.size() == static_cast<size_t>(localN));
        CHECK(rBytes == wBytes);

        ser.CloseFile();
    }
}

// ===================================================================
// SerializerH5 — path and listing
// ===================================================================
TEST_CASE("SerializerH5 path operations")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    std::string fname = TmpH5("h5paths");
    FileGuard guard(fname, true);

    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, false);

        ser.CreatePath("/x");
        ser.GoToPath("/x");

        CHECK(ser.GetCurrentPath() == "/x");

        ser.WriteInt("val", 7);

        // Groups in HDF5 are lazily created when content is written into them.
        // Navigate into each child and write a marker to materialize the group.
        auto cwd = ser.GetCurrentPath();
        ser.GoToPath("y");
        ser.WriteInt("marker", 1);
        ser.GoToPath(cwd);
        ser.GoToPath("z");
        ser.WriteInt("marker", 2);
        ser.GoToPath(cwd);

        ser.CloseFile();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, true);
        ser.GoToPath("/x");

        auto entries = ser.ListCurrentPath();
        CHECK(entries.count("y") == 1);
        CHECK(entries.count("z") == 1);

        int v = 0;
        ser.ReadInt("val", v);
        CHECK(v == 7);

        ser.CloseFile();
    }
}

// ===================================================================
// SerializerH5 — string round-trip
// ===================================================================
TEST_CASE("SerializerH5 string round-trip")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    std::string fname = TmpH5("h5str");
    FileGuard guard(fname, true);

    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, false);
        ser.CreatePath("/meta");
        ser.GoToPath("/meta");
        ser.WriteString("solver", "euler3D");
        ser.WriteString("version", "1.2.3");
        ser.CloseFile();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    {
        S::SerializerH5 ser(mpi);
        ser.OpenFile(fname, true);
        ser.GoToPath("/meta");

        std::string solver, version;
        ser.ReadString("solver", solver);
        ser.ReadString("version", version);

        CHECK(solver == "euler3D");
        CHECK(version == "1.2.3");

        ser.CloseFile();
    }
}

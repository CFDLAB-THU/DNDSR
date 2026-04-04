/**
 * @file test_Array.cpp
 * @brief Doctest-based unit tests for DNDS::Array across all five data
 *        layouts (TABLE_StaticFixed, TABLE_Fixed, TABLE_StaticMax,
 *        TABLE_Max, CSR).
 *
 * This is a serial test (no MPI required).  It also covers clone/copy
 * semantics, edge cases, JSON serialization round-trips, array signature
 * utilities, and hashing.
 *
 * @see @ref dnds_unit_tests for the full test-suite overview.
 * @test Array layout correctness, resize, compress/decompress, clone, copy,
 *       swap, serialization round-trip, signature, hash.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "DNDS/Array.hpp"

#include <cstdio>
#include <filesystem>
#include <string>

using namespace DNDS;

// ===================================================================
// TABLE_StaticFixed: Array<real, 3> — compile-time fixed row size
// ===================================================================
TEST_CASE("Array TABLE_StaticFixed layout")
{
    Array<DNDS::real, 3> a;
    a.Resize(10);

    REQUIRE(a.Size() == 10);
    REQUIRE(a.RowSize() == 3);
    CHECK(a.GetDataLayout() == TABLE_StaticFixed);

    SUBCASE("write and read via operator()")
    {
        for (DNDS::index i = 0; i < a.Size(); i++)
            for (DNDS::rowsize j = 0; j < a.RowSize(); j++)
                a(i, j) = static_cast<DNDS::real>(i * 10 + j);

        for (DNDS::index i = 0; i < a.Size(); i++)
            for (DNDS::rowsize j = 0; j < a.RowSize(); j++)
                CHECK(a(i, j) == static_cast<DNDS::real>(i * 10 + j));
    }

    SUBCASE("operator[] pointer access")
    {
        for (DNDS::index i = 0; i < a.Size(); i++)
            for (DNDS::rowsize j = 0; j < a.RowSize(); j++)
                a(i, j) = static_cast<DNDS::real>(i * 100 + j);

        for (DNDS::index i = 0; i < a.Size(); i++)
        {
            DNDS::real *row = a[i];
            REQUIRE(row != nullptr);
            for (DNDS::rowsize j = 0; j < a.RowSize(); j++)
                CHECK(row[j] == static_cast<DNDS::real>(i * 100 + j));
        }
    }

    SUBCASE("DataSize and DataSizeBytes")
    {
        CHECK(a.DataSize() == 10 * 3);
        CHECK(a.DataSizeBytes() == 10 * 3 * sizeof(DNDS::real));
    }
}

// ===================================================================
// TABLE_Fixed: Array<real, DynamicSize> — runtime-chosen fixed row size
// ===================================================================
TEST_CASE("Array TABLE_Fixed layout")
{
    Array<DNDS::real, DynamicSize> b;
    b.Resize(8, 5);

    REQUIRE(b.Size() == 8);
    REQUIRE(b.RowSize() == 5);
    CHECK(b.GetDataLayout() == TABLE_Fixed);

    for (DNDS::index i = 0; i < b.Size(); i++)
        for (DNDS::rowsize j = 0; j < b.RowSize(); j++)
            b(i, j) = static_cast<DNDS::real>(i * 10 + j);

    for (DNDS::index i = 0; i < b.Size(); i++)
        for (DNDS::rowsize j = 0; j < b.RowSize(); j++)
            CHECK(b(i, j) == static_cast<DNDS::real>(i * 10 + j));

    CHECK(b.DataSize() == 8 * 5);
    CHECK(b.DataSizeBytes() == 8 * 5 * sizeof(DNDS::real));
}

// ===================================================================
// TABLE_StaticMax: Array<real, NonUniformSize, 4> — per-row size ≤ 4
// ===================================================================
TEST_CASE("Array TABLE_StaticMax layout")
{
    Array<DNDS::real, NonUniformSize, 4> c;
    c.Resize(6);

    REQUIRE(c.Size() == 6);
    CHECK(c.GetDataLayout() == TABLE_StaticMax);
    CHECK(c.RowSizeMax() == 4);

    // Assign each row a different active size in [1, 4]
    for (DNDS::index i = 0; i < c.Size(); i++)
        c.ResizeRow(i, static_cast<DNDS::rowsize>(i % 4 + 1));

    // Write values
    for (DNDS::index i = 0; i < c.Size(); i++)
        for (DNDS::rowsize j = 0; j < c.RowSize(i); j++)
            c(i, j) = static_cast<DNDS::real>(i * 10 + j);

    // Read back and verify
    for (DNDS::index i = 0; i < c.Size(); i++)
    {
        CHECK(c.RowSize(i) == static_cast<DNDS::rowsize>(i % 4 + 1));
        for (DNDS::rowsize j = 0; j < c.RowSize(i); j++)
            CHECK(c(i, j) == static_cast<DNDS::real>(i * 10 + j));
    }
}

// ===================================================================
// TABLE_Max: Array<real, NonUniformSize, DynamicSize> — dynamic max
// ===================================================================
TEST_CASE("Array TABLE_Max layout")
{
    Array<DNDS::real, NonUniformSize, DynamicSize> d;
    d.Resize(5, 6);

    REQUIRE(d.Size() == 5);
    CHECK(d.GetDataLayout() == TABLE_Max);
    CHECK(d.RowSizeMax() == 6);

    for (DNDS::index i = 0; i < d.Size(); i++)
        d.ResizeRow(i, static_cast<DNDS::rowsize>(i + 1)); // sizes 1..5

    for (DNDS::index i = 0; i < d.Size(); i++)
        for (DNDS::rowsize j = 0; j < d.RowSize(i); j++)
            d(i, j) = static_cast<DNDS::real>(i * 100 + j);

    for (DNDS::index i = 0; i < d.Size(); i++)
    {
        CHECK(d.RowSize(i) == static_cast<DNDS::rowsize>(i + 1));
        for (DNDS::rowsize j = 0; j < d.RowSize(i); j++)
            CHECK(d(i, j) == static_cast<DNDS::real>(i * 100 + j));
    }
}

// ===================================================================
// CSR: Array<real, NonUniformSize> — compressed sparse row
// ===================================================================
TEST_CASE("Array CSR layout")
{
    Array<DNDS::real, NonUniformSize> e;
    CHECK(e.GetDataLayout() == CSR);

    SUBCASE("Resize with lambda and compress round-trip")
    {
        // Build compressed directly via lambda
        e.Resize(10, [](DNDS::index i) -> DNDS::rowsize
                 { return static_cast<DNDS::rowsize>(i % 4 + 1); });

        REQUIRE(e.Size() == 10);
        CHECK(e.IfCompressed());

        // Verify row sizes
        for (DNDS::index i = 0; i < e.Size(); i++)
            CHECK(e.RowSize(i) == static_cast<DNDS::rowsize>(i % 4 + 1));

        // Write values (compressed)
        for (DNDS::index i = 0; i < e.Size(); i++)
            for (DNDS::rowsize j = 0; j < e.RowSize(i); j++)
                e(i, j) = static_cast<DNDS::real>(i * 10 + j);

        // Read back
        for (DNDS::index i = 0; i < e.Size(); i++)
            for (DNDS::rowsize j = 0; j < e.RowSize(i); j++)
                CHECK(e(i, j) == static_cast<DNDS::real>(i * 10 + j));
    }

    SUBCASE("Decompress, modify, re-Compress")
    {
        e.Resize(6, [](DNDS::index i) -> DNDS::rowsize
                 { return static_cast<DNDS::rowsize>(i % 3 + 1); });

        // Fill with known data
        for (DNDS::index i = 0; i < e.Size(); i++)
            for (DNDS::rowsize j = 0; j < e.RowSize(i); j++)
                e(i, j) = static_cast<DNDS::real>(i + j * 0.1);

        // Decompress
        e.Decompress();
        CHECK_FALSE(e.IfCompressed());

        // Data should still be readable
        for (DNDS::index i = 0; i < e.Size(); i++)
            for (DNDS::rowsize j = 0; j < e.RowSize(i); j++)
                CHECK(e(i, j) == doctest::Approx(static_cast<DNDS::real>(i + j * 0.1)));

        // Resize some rows in decompressed mode
        for (DNDS::index i = 0; i < e.Size(); i++)
            e.ResizeRow(i, static_cast<DNDS::rowsize>(i % 2 + 2)); // sizes 2 or 3

        // Write new data
        for (DNDS::index i = 0; i < e.Size(); i++)
            for (DNDS::rowsize j = 0; j < e.RowSize(i); j++)
                e(i, j) = static_cast<DNDS::real>(i * 100 + j);

        // Re-compress
        e.Compress();
        CHECK(e.IfCompressed());

        // Verify sizes and data after round-trip
        for (DNDS::index i = 0; i < e.Size(); i++)
        {
            CHECK(e.RowSize(i) == static_cast<DNDS::rowsize>(i % 2 + 2));
            for (DNDS::rowsize j = 0; j < e.RowSize(i); j++)
                CHECK(e(i, j) == static_cast<DNDS::real>(i * 100 + j));
        }
    }
}

// ===================================================================
// Clone, copy, and swap
// ===================================================================
TEST_CASE("Array clone and copy")
{
    // Source: TABLE_Fixed with known data
    Array<DNDS::real, DynamicSize> src;
    src.Resize(5, 4);
    for (DNDS::index i = 0; i < src.Size(); i++)
        for (DNDS::rowsize j = 0; j < src.RowSize(); j++)
            src(i, j) = static_cast<DNDS::real>(i * 10 + j);

    SUBCASE("clone produces identical independent copy")
    {
        Array<DNDS::real, DynamicSize> dst;
        dst.clone(src);

        REQUIRE(dst.Size() == src.Size());
        REQUIRE(dst.RowSize() == src.RowSize());
        for (DNDS::index i = 0; i < dst.Size(); i++)
            for (DNDS::rowsize j = 0; j < dst.RowSize(); j++)
                CHECK(dst(i, j) == src(i, j));

        // Modifying source does not affect clone
        src(0, 0) = -999.0;
        CHECK(dst(0, 0) != -999.0);
    }

    SUBCASE("CopyData produces identical copy")
    {
        Array<DNDS::real, DynamicSize> dst;
        dst.CopyData(src);

        REQUIRE(dst.Size() == src.Size());
        for (DNDS::index i = 0; i < dst.Size(); i++)
            for (DNDS::rowsize j = 0; j < dst.RowSize(); j++)
                CHECK(dst(i, j) == src(i, j));
    }

    SUBCASE("copy constructor")
    {
        Array<DNDS::real, DynamicSize> cpy(src);

        REQUIRE(cpy.Size() == src.Size());
        REQUIRE(cpy.RowSize() == src.RowSize());
        for (DNDS::index i = 0; i < cpy.Size(); i++)
            for (DNDS::rowsize j = 0; j < cpy.RowSize(); j++)
                CHECK(cpy(i, j) == src(i, j));
    }

    SUBCASE("SwapData exchanges contents")
    {
        Array<DNDS::real, DynamicSize> other;
        other.Resize(5, 4);
        for (DNDS::index i = 0; i < other.Size(); i++)
            for (DNDS::rowsize j = 0; j < other.RowSize(); j++)
                other(i, j) = static_cast<DNDS::real>(i * 10 + j + 1000);

        DNDS::real src_0_0 = src(0, 0);
        DNDS::real other_0_0 = other(0, 0);

        src.SwapData(other);

        CHECK(src(0, 0) == other_0_0);
        CHECK(other(0, 0) == src_0_0);
    }
}

// ===================================================================
// Edge cases
// ===================================================================
TEST_CASE("Array edge cases")
{
    SUBCASE("zero-size TABLE_StaticFixed")
    {
        Array<DNDS::real, 3> z;
        z.Resize(0);
        CHECK(z.Size() == 0);
        CHECK(z.DataSize() == 0);
        CHECK(z.DataSizeBytes() == 0);
    }

    SUBCASE("single element TABLE_StaticFixed")
    {
        Array<DNDS::real, 2> s;
        s.Resize(1);
        REQUIRE(s.Size() == 1);
        s(0, 0) = 42.0;
        s(0, 1) = 43.0;
        CHECK(s(0, 0) == 42.0);
        CHECK(s(0, 1) == 43.0);
    }

    SUBCASE("CSR with lambda returning 0 for some rows")
    {
        Array<DNDS::real, NonUniformSize> csr;
        csr.Resize(5, [](DNDS::index i) -> DNDS::rowsize
                   { return (i % 2 == 0) ? 0 : 3; });

        REQUIRE(csr.Size() == 5);
        for (DNDS::index i = 0; i < csr.Size(); i++)
        {
            DNDS::rowsize expected = (i % 2 == 0) ? 0 : 3;
            CHECK(csr.RowSize(i) == expected);
        }

        // Write to non-zero rows only
        for (DNDS::index i = 0; i < csr.Size(); i++)
            for (DNDS::rowsize j = 0; j < csr.RowSize(i); j++)
                csr(i, j) = static_cast<DNDS::real>(i * 10 + j);

        // Read back
        for (DNDS::index i = 0; i < csr.Size(); i++)
            for (DNDS::rowsize j = 0; j < csr.RowSize(i); j++)
                CHECK(csr(i, j) == static_cast<DNDS::real>(i * 10 + j));
    }

    SUBCASE("zero-size TABLE_Fixed")
    {
        Array<DNDS::real, DynamicSize> z;
        z.Resize(0, 5);
        CHECK(z.Size() == 0);
        CHECK(z.DataSize() == 0);
    }
}

// ===================================================================
// Serialization round-trip (JSON)
// ===================================================================
TEST_CASE("Array serialization round-trip")
{
    namespace S = DNDS::Serializer;
    const std::string tmpFile = "__test_array_ser_tmp.json";
    auto cleanup = [&]()
    {
        std::filesystem::remove(tmpFile);
    };

    SUBCASE("TABLE_Fixed round-trip")
    {
        // Write
        {
            Array<DNDS::real, DynamicSize> src;
            src.Resize(4, 3);
            for (DNDS::index i = 0; i < src.Size(); i++)
                for (DNDS::rowsize j = 0; j < src.RowSize(); j++)
                    src(i, j) = static_cast<DNDS::real>(i * 10 + j);

            auto ser = std::make_shared<S::SerializerJSON>();
            ser->OpenFile(tmpFile, false);
            S::ArrayGlobalOffset offset = S::ArrayGlobalOffset_Unknown;
            src.WriteSerializer(ser, "arr", offset);
            ser->CloseFile();
        }

        // Read
        {
            Array<DNDS::real, DynamicSize> dst;
            auto ser = std::make_shared<S::SerializerJSON>();
            ser->OpenFile(tmpFile, true);
            S::ArrayGlobalOffset offset = S::ArrayGlobalOffset_Unknown;
            dst.ReadSerializer(ser, "arr", offset);
            ser->CloseFile();

            REQUIRE(dst.Size() == 4);
            REQUIRE(dst.RowSize() == 3);
            for (DNDS::index i = 0; i < dst.Size(); i++)
                for (DNDS::rowsize j = 0; j < dst.RowSize(); j++)
                    CHECK(dst(i, j) == static_cast<DNDS::real>(i * 10 + j));
        }
        cleanup();
    }

    SUBCASE("CSR round-trip")
    {
        // Write
        {
            Array<DNDS::real, NonUniformSize> src;
            src.Resize(6, [](DNDS::index i) -> DNDS::rowsize
                       { return static_cast<DNDS::rowsize>(i % 3 + 1); });

            for (DNDS::index i = 0; i < src.Size(); i++)
                for (DNDS::rowsize j = 0; j < src.RowSize(i); j++)
                    src(i, j) = static_cast<DNDS::real>(i * 100 + j);

            auto ser = std::make_shared<S::SerializerJSON>();
            ser->OpenFile(tmpFile, false);
            S::ArrayGlobalOffset offset = S::ArrayGlobalOffset_Unknown;
            src.WriteSerializer(ser, "csr", offset);
            ser->CloseFile();
        }

        // Read
        {
            Array<DNDS::real, NonUniformSize> dst;
            // Pre-create with any shape so ReadSerializer can populate it
            dst.Resize(1, [](DNDS::index) -> DNDS::rowsize
                       { return 1; });

            auto ser = std::make_shared<S::SerializerJSON>();
            ser->OpenFile(tmpFile, true);
            S::ArrayGlobalOffset offset = S::ArrayGlobalOffset_Unknown;
            dst.ReadSerializer(ser, "csr", offset);
            ser->CloseFile();

            REQUIRE(dst.Size() == 6);
            for (DNDS::index i = 0; i < dst.Size(); i++)
            {
                CHECK(dst.RowSize(i) == static_cast<DNDS::rowsize>(i % 3 + 1));
                for (DNDS::rowsize j = 0; j < dst.RowSize(i); j++)
                    CHECK(dst(i, j) == static_cast<DNDS::real>(i * 100 + j));
            }
        }
        cleanup();
    }

    SUBCASE("TABLE_StaticFixed round-trip")
    {
        // Write
        {
            Array<DNDS::real, 3> src;
            src.Resize(5);
            for (DNDS::index i = 0; i < src.Size(); i++)
                for (DNDS::rowsize j = 0; j < src.RowSize(); j++)
                    src(i, j) = static_cast<DNDS::real>(i + j * 0.5);

            auto ser = std::make_shared<S::SerializerJSON>();
            ser->OpenFile(tmpFile, false);
            S::ArrayGlobalOffset offset = S::ArrayGlobalOffset_Unknown;
            src.WriteSerializer(ser, "sf", offset);
            ser->CloseFile();
        }

        // Read
        {
            Array<DNDS::real, 3> dst;
            auto ser = std::make_shared<S::SerializerJSON>();
            ser->OpenFile(tmpFile, true);
            S::ArrayGlobalOffset offset = S::ArrayGlobalOffset_Unknown;
            dst.ReadSerializer(ser, "sf", offset);
            ser->CloseFile();

            REQUIRE(dst.Size() == 5);
            REQUIRE(dst.RowSize() == 3);
            for (DNDS::index i = 0; i < dst.Size(); i++)
                for (DNDS::rowsize j = 0; j < dst.RowSize(); j++)
                    CHECK(dst(i, j) == doctest::Approx(static_cast<DNDS::real>(i + j * 0.5)));
        }
        cleanup();
    }
}

// ===================================================================
// Signature helpers
// ===================================================================
TEST_CASE("Array signature")
{
    SUBCASE("GetArraySignature returns non-empty string")
    {
        auto sig = Array<DNDS::real, 3>::GetArraySignature();
        CHECK_FALSE(sig.empty());
        CHECK(sig.find("TABLE_StaticFixed") != std::string::npos);
    }

    SUBCASE("ParseArraySignatureTuple round-trips")
    {
        auto sig = Array<DNDS::real, DynamicSize>::GetArraySignature();
        auto [sz, rs, rm, al] = Array<DNDS::real, DynamicSize>::ParseArraySignatureTuple(sig);

        CHECK(sz == static_cast<int>(sizeof(DNDS::real)));
        CHECK(rs == DynamicSize);
        CHECK(rm == DynamicSize);
    }

    SUBCASE("ArraySignatureIsCompatible for matching type")
    {
        auto sig = Array<DNDS::real, 3>::GetArraySignature();
        CHECK(Array<DNDS::real, 3>::ArraySignatureIsCompatible(sig));
    }

    SUBCASE("ArraySignatureIsCompatible accepts DynamicSize reading static")
    {
        // A DynamicSize array should accept a static-row signature if
        // the sizes are not contradictory.
        auto sigStatic = Array<DNDS::real, 3>::GetArraySignature();
        CHECK(Array<DNDS::real, DynamicSize>::ArraySignatureIsCompatible(sigStatic));
    }

    SUBCASE("ArraySignatureIsCompatible rejects mismatched sizes")
    {
        // Array<real, 3> vs Array<real, 5> should not be compatible
        auto sig5 = Array<DNDS::real, 5>::GetArraySignature();
        CHECK_FALSE(Array<DNDS::real, 3>::ArraySignatureIsCompatible(sig5));
    }

    SUBCASE("CSR signature")
    {
        auto sig = Array<DNDS::real, NonUniformSize>::GetArraySignature();
        CHECK(sig.find("CSR") != std::string::npos);
    }
}

// ===================================================================
// Hash
// ===================================================================
TEST_CASE("Array hash")
{
    SUBCASE("identical arrays have the same hash")
    {
        Array<DNDS::real, 3> a, b;
        a.Resize(5);
        b.Resize(5);
        for (DNDS::index i = 0; i < 5; i++)
            for (DNDS::rowsize j = 0; j < 3; j++)
            {
                a(i, j) = static_cast<DNDS::real>(i * 10 + j);
                b(i, j) = static_cast<DNDS::real>(i * 10 + j);
            }

        CHECK(a.hash() == b.hash());
    }

    SUBCASE("modified array has different hash")
    {
        Array<DNDS::real, 3> a, b;
        a.Resize(5);
        b.Resize(5);
        for (DNDS::index i = 0; i < 5; i++)
            for (DNDS::rowsize j = 0; j < 3; j++)
            {
                a(i, j) = static_cast<DNDS::real>(i * 10 + j);
                b(i, j) = static_cast<DNDS::real>(i * 10 + j);
            }

        b(2, 1) = -1.0;
        CHECK(a.hash() != b.hash());
    }
}

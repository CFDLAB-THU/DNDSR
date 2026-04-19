/**
 * @file ex_array_basic.cpp
 * @brief Compilable examples from the Array Usage Guide.
 *
 * Demonstrates: Array (fixed/dynamic/CSR), ParArray with ghost
 * communication, ArrayPair with BorrowAndPull, and ArrayDof.
 *
 * Build:  cmake --build build -t ex_array_basic -j8
 * Run:    mpirun -np 2 build/examples/ex_array_basic
 */

#include "DNDS/Array.hpp"
#include "DNDS/ArrayTransformer.hpp"
#include "DNDS/ArrayPair.hpp"
#include "DNDS/Defines.hpp"

#include <fmt/core.h>
#include <iostream>
#include <numeric>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    {
        using namespace DNDS;

        MPIInfo mpi;
        mpi.setWorld();

        // ============================================================
        // 1. Basic Array: fixed-row
        // ============================================================
        {
            Array<real, 5> a;
            a.Resize(100);
            a[0][0] = 1.0;
            a(42, 3) = 3.14;

            DNDS_assert(a.Size() == 100);
            DNDS_assert(a.RowSize(0) == 5);
            DNDS_assert(std::abs(a(42, 3) - 3.14) < 1e-15);
            if (mpi.rank == 0)
                fmt::print("[ex1] Fixed-row Array: Size={}, RowSize={}, a(42,3)={}\n",
                           a.Size(), a.RowSize(0), a(42, 3));
        }

        // ============================================================
        // 2. Dynamic-row Array
        // ============================================================
        {
            Array<real, DynamicSize> b;
            b.Resize(50, 7);
            for (DNDS::index i = 0; i < b.Size(); i++)
                for (rowsize j = 0; j < b.RowSize(i); j++)
                    b(i, j) = static_cast<real>(i + j);

            DNDS_assert(b.RowSize(0) == 7);
            if (mpi.rank == 0)
                fmt::print("[ex2] Dynamic-row Array: Size={}, RowSize={}, b(10,3)={}\n",
                           b.Size(), b.RowSize(0), b(10, 3));
        }

        // ============================================================
        // 3. CSR Array (variable row width)
        // ============================================================
        {
            Array<DNDS::index, NonUniformSize, NonUniformSize> adj;
            adj.Resize(4, [](DNDS::index i) -> rowsize { return static_cast<rowsize>(i + 2); });
            // Row 0: 2, Row 1: 3, Row 2: 4, Row 3: 5

            DNDS_assert(adj.RowSize(0) == 2);
            DNDS_assert(adj.RowSize(3) == 5);

            adj.Decompress();
            adj.ResizeRow(2, 6);
            DNDS_assert(adj.RowSize(2) == 6);
            adj.Compress();

            if (mpi.rank == 0)
                fmt::print("[ex3] CSR Array: Size={}, row sizes: {}, {}, {}, {}\n",
                           adj.Size(), adj.RowSize(0), adj.RowSize(1),
                           adj.RowSize(2), adj.RowSize(3));
        }

        // ============================================================
        // 4. ParArray + Ghost Communication
        // ============================================================
        {
            DNDS::index nLocal = 10;
            auto father = std::make_shared<ParArray<real, 3>>(mpi);
            auto son = std::make_shared<ParArray<real, 3>>(mpi);
            father->Resize(nLocal);

            // Fill each rank's data with rank-specific values.
            for (DNDS::index i = 0; i < nLocal; i++)
                for (rowsize j = 0; j < 3; j++)
                    (*father)(i, j) = mpi.rank * 100.0 + i * 10.0 + j;

            ArrayTransformer<real, 3> trans;
            trans.setFatherSon(father, son);
            trans.createFatherGlobalMapping();

            // Each rank pulls the first cell of the next rank (wrapping).
            DNDS::index globalSize = father->pLGlobalMapping->globalSize();
            DNDS::index myGlobalStart = father->pLGlobalMapping->operator()(mpi.rank, 0);
            DNDS::index pullIdx = (myGlobalStart + nLocal) % globalSize;

            std::vector<DNDS::index> pullGlobal = {pullIdx};
            trans.createGhostMapping(pullGlobal);
            trans.createMPITypes();

            trans.pullOnce();

            // Verify: son[0] should contain the first cell of the next rank.
            if (mpi.rank == 0)
                fmt::print("[ex4] Ghost pull: son[0] = ({}, {}, {})  (expected next rank's cell 0)\n",
                           (*son)(0, 0), (*son)(0, 1), (*son)(0, 2));
        }

        // ============================================================
        // 5. ArrayPair with BorrowAndPull
        // ============================================================
        {
            DNDS::index nLocal = 8;

            // Primary pair
            ArrayPair<ParArray<real, 2>> primary;
            primary.InitPair("primary", mpi);
            primary.father->Resize(nLocal);
            for (DNDS::index i = 0; i < nLocal; i++)
            {
                (*primary.father)(i, 0) = mpi.rank * 100.0 + i;
                (*primary.father)(i, 1) = mpi.rank * 100.0 + i + 0.5;
            }
            primary.TransAttach();
            primary.trans.createFatherGlobalMapping();

            DNDS::index globalSize = primary.father->pLGlobalMapping->globalSize();
            DNDS::index myStart = primary.father->pLGlobalMapping->operator()(mpi.rank, 0);
            DNDS::index pullIdx = (myStart + nLocal) % globalSize;
            std::vector<DNDS::index> pullGlobal = {pullIdx};
            primary.trans.createGhostMapping(pullGlobal);
            primary.trans.createMPITypes();
            primary.trans.pullOnce();

            // Secondary pair: borrow ghost mapping from primary.
            ArrayPair<ParArray<real, 5>> secondary;
            secondary.InitPair("secondary", mpi);
            secondary.father->Resize(nLocal);
            for (DNDS::index i = 0; i < nLocal; i++)
                for (rowsize j = 0; j < 5; j++)
                    (*secondary.father)(i, j) = mpi.rank * 1000.0 + i * 10.0 + j;

            secondary.BorrowAndPull(primary);

            DNDS_assert(secondary.Size() == nLocal + 1); // 1 ghost
            if (mpi.rank == 0)
                fmt::print("[ex5] ArrayPair: Size={} ({}+1 ghost), secondary ghost[0]=({})\n",
                           secondary.Size(), nLocal, secondary(nLocal, 0));
        }

        if (mpi.rank == 0)
            fmt::print("\nAll array examples passed.\n");
    }
    MPI_Finalize();
    return 0;
}

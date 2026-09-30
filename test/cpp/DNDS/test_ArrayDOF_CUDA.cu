#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/ArrayDOF.hpp"
#include "DNDS/Device/CUDA_Utils.hpp"
#include <cuda_runtime.h>

namespace
{
    template <int Rows>
    DNDS::ArrayDof<Rows, 1> makeDof(const DNDS::MPIInfo &mpi, DNDS::index father, DNDS::index son)
    {
        DNDS::ArrayDof<Rows, 1> a;
        a.InitPair("boundary", mpi);
        a.father->Resize(father, 1, 1);
        a.son->Resize(son, 1, 1);
        if constexpr (Rows == DNDS::NonUniformSize)
        {
            for (DNDS::index i = 0; i < father; ++i)
                a.father->ResizeRow(i, 1 + i % 3, 1);
            for (DNDS::index i = 0; i < son; ++i)
                a.son->ResizeRow(i, 1 + i % 3, 1);
            a.father->Compress();
            a.son->Compress();
        }
        a.setConstant(2);
        return a;
    }

    template <int Rows>
    void checkScalarRows(const DNDS::MPIInfo &mpi, DNDS::index n, DNDS::index ghosts)
    {
        auto a = makeDof<Rows>(mpi, n, ghosts);
        auto scale = makeDof<1>(mpi, n, ghosts);
        scale.setConstant(3);
        auto host = makeDof<Rows>(mpi, n, ghosts);
        host *= scale;
        a.to_device(DNDS::DeviceBackend::CUDA);
        scale.to_device(DNDS::DeviceBackend::CUDA);
        // Explicit dispatch also exercises the uniform scalar-row kernel for
        // 1x1 storage (the public 1x1 operator otherwise selects Hadamard).
        DNDS::ArrayDofOp<DNDS::DeviceBackend::CUDA, Rows, 1>::operator_mult_assign_scalar_arr(a, scale);
        DNDS_CUDA_CHECKED(cudaDeviceSynchronize());
        a.to_host();
        for (DNDS::index i = 0; i < a.Size(); ++i)
            CHECK(a[i].isApprox(host[i], 0));
    }
}

TEST_CASE("CUDA boundary: uniform element matrix operations")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (DNDS::index n : {0, 1, 127, 128, 129, 255, 256, 257})
        for (DNDS::index ghosts : {0, 1, 129})
        {
            auto a = makeDof<1>(mpi, n, ghosts);
            auto host = makeDof<1>(mpi, n, ghosts);
            Eigen::Matrix<DNDS::real, 1, 1> m;
            m << 3;
            host += m;
            host *= m;
            a.to_device(DNDS::DeviceBackend::CUDA);
            a += m;
            a *= m;
            DNDS_CUDA_CHECKED(cudaDeviceSynchronize());
            a.to_host();
            for (DNDS::index i = 0; i < a.Size(); ++i)
                CHECK(a[i].isApprox(host[i], 0));
        }
}

TEST_CASE("CUDA boundary: uniform scalar row multiplication")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (DNDS::index n : {0, 1, 31, 32, 33, 127, 128, 129})
        for (DNDS::index ghosts : {0, 1, 33})
        {
            checkScalarRows<1>(mpi, n, ghosts);
        }
}

TEST_CASE("CUDA boundary: CSR scalar row multiplication")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (DNDS::index n : {0, 1, 31, 32, 33, 127, 128, 129})
        for (DNDS::index ghosts : {0, 1, 33})
            checkScalarRows<DNDS::NonUniformSize>(mpi, n, ghosts);
}

TEST_CASE("CUDA ownership: leases retain device storage after replacement")
{
    DNDS::host_device_vector<DNDS::real> a(3, 7);
    a.to_device(DNDS::DeviceBackend::CUDA);
    auto lease = a.deviceLease();
    std::weak_ptr<DNDS::real> weak = lease;
    a.resize(5, 11);
    a.to_device(DNDS::DeviceBackend::CUDA);
    CHECK(a.dataDevice() != lease.get());
    a.clear();
    DNDS::real old = 0;
    DNDS_CUDA_CHECKED(cudaMemcpy(&old, lease.get(), sizeof(old), cudaMemcpyDeviceToHost));
    CHECK(old == 7);
    CHECK_FALSE(weak.expired());
    lease.reset();
    CHECK(weak.expired());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    DNDS_CUDA_CHECKED(cudaSetDevice(0)); // caller selects the physical GPU
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int result = ctx.run();
    MPI_Finalize();
    return result;
}

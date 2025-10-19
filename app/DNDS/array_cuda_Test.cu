#include "DNDS/Array.hpp"

#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency_DeviceView.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/MPI.hpp"

#include <iostream>

namespace DNDS
{

    DNDS_GLOBAL void op_kernel(ArrayAdjacencyDeviceView<DeviceBackend::CUDA, NonUniformSize> arr)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= arr.Size())
            return;
        auto row = arr[tid];
        for (auto &i : row)
            i += 1;
    }

    void array_cuda_Test(MPIInfo &mpi)
    {
        ArrayAdjacency<NonUniformSize> adj{mpi};
        adj.Resize(32);
        for (int i = 0; i < adj.Size(); i++)
        {
            adj.ResizeRow(i, i % 3 + 1);
            std::fill(adj[i].begin(), adj[i].end(), 1);
        }
        adj.Compress();
        adj.to_device(DeviceBackend::CUDA);
        int threadsPerBlock = 1024;
        int blocksPerGrid = (adj.Size() + threadsPerBlock - 1) / threadsPerBlock;
        op_kernel<<<blocksPerGrid, threadsPerBlock>>>(adj.deviceView<DeviceBackend::CUDA>());
        adj.to_host();
        for (int i = 0; i < adj.Size(); i++)
        {
            for (auto v : adj[i])
                std::cout << v << " ";
            std::cout << "\n";
        }
    }
}

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    DNDS::array_cuda_Test(mpi);
    DNDS::MPI::Finalize();
    return 0;
}
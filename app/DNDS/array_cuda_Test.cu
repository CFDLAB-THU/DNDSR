#include "DNDS/Array.hpp"

// #include "array_cuda_Test.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrix.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayPair.hpp"
#include "DNDS/Device/DeviceStorage.hpp"

#include <iostream>

namespace DNDS
{

    DNDS_GLOBAL void op_kernel_adjacency(ArrayAdjacencyDeviceView<DeviceBackend::CUDA, NonUniformSize> arr)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= arr.Size())
            return;
        auto row = arr[tid];
        for (auto &i : row)
            i += 1;
    }

    void array_cuda_Test_Adjacency(MPIInfo &mpi)
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
        op_kernel_adjacency<<<blocksPerGrid, threadsPerBlock>>>(adj.deviceView<DeviceBackend::CUDA>());
        adj.to_host();
        for (int i = 0; i < adj.Size(); i++)
        {
            for (auto v : adj[i])
                DNDS_assert(v == 2);
            // std::cout << v << " ";
            // std::cout << "\n";
        }
    }

    namespace array_cuda_Test_EigenVector
    {
        using t_Arr = ArrayEigenVector<NonUniformSize>;
        DNDS_GLOBAL void op_kernel(t_Arr::t_deviceView<DeviceBackend::CUDA> arr)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= arr.Size())
                return;

            auto row = arr[tid];
            row = row.array() * row.array();

            Eigen::Matrix3d a;
            a.setIdentity();

            Eigen::MatrixXd b;
            b.setIdentity(30, 30);
            // row.array() = b.determinant() + 3.0; // ! fails to compute
            row.array() = a.determinant() + 3.0;
            // row.array() = b.array().sum() + 1.0;
        }

        void test(MPIInfo &mpi)
        {
            t_Arr arr(mpi);
            std::cout << arr.GetArrayName() << std::endl;
            arr.Resize(32, 5);
            for (int i = 0; i < arr.Size(); i++)
            {
                arr[i].setConstant(2.0);
            }
            if (arr.GetDataLayout() == CSR)
                arr.Compress();
            arr.to_device(DeviceBackend::CUDA);
            int threadsPerBlock = 1024;
            int blocksPerGrid = (arr.Size() + threadsPerBlock - 1) / threadsPerBlock;
            op_kernel<<<blocksPerGrid, threadsPerBlock>>>(arr.deviceView<DeviceBackend::CUDA>());
            arr.to_host();
            for (int i = 0; i < arr.Size(); i++)
            {
                // std::cout << arr[i].transpose() << std::endl;
                DNDS_assert((arr[i].array() - 4.0).matrix().squaredNorm() == 0.0);
            }
        }
    }

    namespace array_cuda_Test_EigenMatrix
    {
        using t_Arr = ArrayEigenMatrix<NonUniformSize, NonUniformSize>;
        DNDS_GLOBAL void op_kernel(t_Arr::t_deviceView<DeviceBackend::CUDA> arr)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= arr.Size())
                return;

            auto row = arr[tid];
            row = row.array() * row.array();
        }

        void test(MPIInfo &mpi)
        {
            t_Arr arr(mpi);
            std::cout << arr.GetArrayName() << std::endl;
            arr.Resize(32, 1, 2);
            for (int i = 0; i < arr.Size(); i++)
            {
                arr.ResizeMat(i, i % 3 + 2, i % 3 + 4);
                arr[i].setConstant(2.0);
            }
            if (arr.GetDataLayout() == CSR)
                arr.Compress();
            arr.to_device(DeviceBackend::CUDA);
            int threadsPerBlock = 1024;
            int blocksPerGrid = (arr.Size() + threadsPerBlock - 1) / threadsPerBlock;
            op_kernel<<<blocksPerGrid, threadsPerBlock>>>(arr.deviceView<DeviceBackend::CUDA>());
            arr.to_host();
            for (int i = 0; i < arr.Size(); i++)
            {
                // std::cout << arr[i] << std::endl;
                DNDS_assert((arr[i].array() - 4.0).matrix().squaredNorm() == 0.0);
            }
        }
    }

    namespace array_cuda_Test_EigenMatrixBatch
    {
        using t_Arr = ArrayEigenMatrixBatch;
        DNDS_GLOBAL void op_kernel(t_Arr::t_deviceView<DeviceBackend::CUDA> arr)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= arr.Size())
                return;
            auto row = arr[tid];
            for (int i = 0; i < row.Size(); i++)
                row[i] = row[i].array() * row[i].array();
        }

        void test(MPIInfo &mpi)
        {
            t_Arr arr(mpi);
            std::cout << arr.GetArrayName() << std::endl;
            arr.Resize(32);
            for (int i = 0; i < arr.Size(); i++)
            {
                // arr.ResizeMat(i, i % 3 + 2, i % 3 + 4);
                std::vector<MatrixXR> mats;
                mats.emplace_back(3, 3);
                mats.back().setConstant(2);
                mats.emplace_back(2, 2);
                mats.back().setConstant(2);
                arr.InitializeWriteRow(i, mats);
            }
            if (arr.GetDataLayout() == CSR)
                arr.Compress();
            arr.to_device(DeviceBackend::CUDA);
            int threadsPerBlock = 1024;
            int blocksPerGrid = (arr.Size() + threadsPerBlock - 1) / threadsPerBlock;
            op_kernel<<<blocksPerGrid, threadsPerBlock>>>(arr.deviceView<DeviceBackend::CUDA>());
            arr.to_host();
            for (int i = 0; i < arr.Size(); i++)
            {
                for (int j = 0; j < arr[i].Size(); j++)
                {
                    // std::cout << arr[i][j] << std::endl;
                    DNDS_assert((arr[i][j].array() - 4.0).matrix().squaredNorm() == 0.0);
                }
            }
        }
    }

    namespace array_cuda_Test_EigenUniMatrixBatch
    {
        using t_Arr = ArrayEigenUniMatrixBatch<Eigen::Dynamic, Eigen::Dynamic>;
        DNDS_GLOBAL void op_kernel(ArrayEigenUniMatrixBatchDeviceView<DeviceBackend::CUDA, -1, -1> arr)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= arr.Size())
                return;
            for (int j = 0; j < arr.RowSize(tid); j++)
                arr(tid, j) = arr(tid, j).array() * arr(tid, j).array();
        }

        void test(MPIInfo &mpi)
        {
            t_Arr arr(mpi);
            std::cout << arr.GetArrayName() << std::endl;
            arr.Resize(32, 3, 3);
            for (int i = 0; i < arr.Size(); i++)
            {
                // arr.ResizeMat(i, i % 3 + 2, i % 3 + 4);
                arr.ResizeRow(i, i % 3 + 1);
                for (int j = 0; j < arr.RowSize(i); j++)
                    arr(i, j).setConstant(2.0);
            }
            if (arr.GetDataLayout() == CSR)
                arr.Compress();
            arr.to_device(DeviceBackend::CUDA);
            int threadsPerBlock = 1024;
            int blocksPerGrid = (arr.Size() + threadsPerBlock - 1) / threadsPerBlock;
            op_kernel<<<blocksPerGrid, threadsPerBlock>>>(arr.template deviceView<DeviceBackend::CUDA>());
            arr.to_host();
            for (int i = 0; i < arr.Size(); i++)
            {
                for (int j = 0; j < arr.RowSize(i); j++)
                {
                    // std::cout << arr(i, j) << std::endl;
                    DNDS_assert((arr(i, j).array() - 4.0).matrix().squaredNorm() == 0.0);
                }
            }
        }
    }

    namespace array_cuda_Test_EigenUniMatrixBatchPair
    {
        //! note: you might need new clangd, like 21.0+ to correctly lint this
        using t_Arr = ArrayEigenUniMatrixBatchPair<Eigen::Dynamic, Eigen::Dynamic>;
        // using t_Arr_view = ArrayPairDeviceView<DeviceBackend::CUDA, t_Arr>;
        using t_Arr_view = typename t_Arr::t_deviceView<DeviceBackend::CUDA>;
        DNDS_GLOBAL inline void op_kernel(t_Arr_view arr)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= arr.Size())
                return;
            for (int j = 0; j < arr.RowSize(tid); j++)
                arr(tid, j) = arr(tid, j).array() * arr(tid, j).array();
        }

        inline void test(MPIInfo &mpi)
        {
            t_Arr arr;

            std::cout << arr.father->GetArrayName() << std::endl;
            DNDS_MAKE_SSP(arr.father, mpi);
            DNDS_MAKE_SSP(arr.son, mpi);
            arr.father->Resize(32, 3, 3);
            arr.son->Resize(2, 3, 3);
            for (int i = 0; i < arr.Size(); i++)
            {
                // arr.ResizeMat(i, i % 3 + 2, i % 3 + 4);
                arr.ResizeRow(i, i % 3 + 1);
                for (int j = 0; j < arr.RowSize(i); j++)
                    arr(i, j).setConstant(2.0);
            }
            if (arr.father->GetDataLayout() == CSR)
                arr.father->Compress();
            if (arr.son->GetDataLayout() == CSR)
                arr.son->Compress();
            arr.to_device(DeviceBackend::CUDA);
            int threadsPerBlock = 1024;
            int blocksPerGrid = (arr.Size() + threadsPerBlock - 1) / threadsPerBlock;
            op_kernel<<<blocksPerGrid, threadsPerBlock>>>(arr.template deviceView<DeviceBackend::CUDA>());
            arr.to_host();
            for (int i = 0; i < arr.Size(); i++)
            {
                for (int j = 0; j < arr.RowSize(i); j++)
                {
                    // std::cout << arr(i, j) << std::endl;
                    DNDS_assert((arr(i, j).array() - 4.0).matrix().squaredNorm() == 0.0);
                }
            }
        }
    }

    void array_cuda_Test(MPIInfo &mpi)
    {
        std::cout << "Adjacency: \n";
        array_cuda_Test_Adjacency(mpi);

        std::cout << "EigenVector: \n";
        array_cuda_Test_EigenVector::test(mpi);

        std::cout << "EigenMatrix: \n";
        array_cuda_Test_EigenMatrix::test(mpi);

        std::cout << "EigenMatrixBatch: \n";
        array_cuda_Test_EigenMatrixBatch::test(mpi);

        std::cout << "EigenUniMatrixBatch: \n";
        array_cuda_Test_EigenUniMatrixBatch::test(mpi);

        std::cout << "EigenUniMatrixBatchPair: \n";
        array_cuda_Test_EigenUniMatrixBatchPair::test(mpi);
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
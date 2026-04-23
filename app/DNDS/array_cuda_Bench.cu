// #define DNDS_NDEBUG_DEVICE
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrix.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayPair.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/Device/DeviceStorage.hpp"

#include "DNDS/MPI.hpp"

#include <ios>
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <string>
#include <iomanip>

namespace DNDS
{
    namespace array_cuda_Bench
    {

        using t_Mats = ArrayPair<ArrayEigenUniMatrixBatch<Eigen::Dynamic, Eigen::Dynamic>>;
        template <DeviceBackend B = DeviceBackend::CUDA>
        using t_Mats_DeviceView = t_Mats::t_deviceView<B>;
        using t_Vecs = ArrayPair<ArrayEigenVector<DynamicSize>>;
        template <DeviceBackend B = DeviceBackend::CUDA>
        using t_Vecs_DeviceView = t_Vecs::t_deviceView<B>;
        using t_Adj = ArrayPair<ArrayAdjacency<NonUniformSize>>;
        template <DeviceBackend B = DeviceBackend::CUDA>
        using t_Adj_DeviceView = t_Adj::t_deviceView<B>;

        template <class TOp>
        std::tuple<double, int> time_operation(TOp &&fOp, int min_iter, int max_iter, double max_time)
        {
            double start = MPI_Wtime();
            int i_iter = 1;
            for (; i_iter <= max_iter; i_iter++)
            {
                fOp();
                double t = MPI_Wtime() - start;
                if (t >= max_time && i_iter >= min_iter)
                    break;
            }
            return {(MPI_Wtime() - start) / i_iter, i_iter};
        }

        template <DeviceBackend B = DeviceBackend::CUDA>
        DNDS_DEVICE_CALLABLE void block_mat_vec_product_one_row(
            t_Adj_DeviceView<B> adj,
            t_Mats_DeviceView<B> mats,
            t_Vecs_DeviceView<B> vec_x,
            t_Vecs_DeviceView<B> vec_y,
            index iRow)
        {
            auto dest = vec_y.father[iRow];
            dest.setZero();
            for (rowsize j = 0; j < adj.RowSize(iRow); j++)
            {
                auto mat = mats.father(iRow, j);
                auto vec = vec_x[adj(iRow, j)];
                // // naive
                // for (int ii = 0; ii < dest.rows(); ii++)
                //     for (int jj = 0; jj < vec.rows(); jj++)
                //         dest(ii) += mat(ii, jj) * vec(jj);
                // serial code by Eigen, optimized for CPU, slightly optimized on CUDA
                dest.noalias() += mat * vec;
                // vec_y.father[iRow] += mats.father(iRow, j) * vec_x[adj(iRow, j)]; // ! allocates heap memory, forbidden
            }
        }

        DNDS_GLOBAL void block_mat_vec_product_by_row(
            t_Adj_DeviceView<DeviceBackend::CUDA> adj,
            t_Mats_DeviceView<DeviceBackend::CUDA> mats,
            t_Vecs_DeviceView<DeviceBackend::CUDA> vec_x,
            t_Vecs_DeviceView<DeviceBackend::CUDA> vec_y)
        {
            index tid = blockIdx.x * blockDim.x + threadIdx.x;

            index iRow = tid;
            if (iRow >= adj.father.Size())
                return;
            block_mat_vec_product_one_row(adj, mats, vec_x, vec_y, iRow);
        }

        template <DeviceBackend B = DeviceBackend::CUDA>
        DNDS_DEVICE_CALLABLE void block_mat_vec_product_one_row_subrow(
            t_Adj_DeviceView<B> &adj,
            t_Mats_DeviceView<B> &mats,
            t_Vecs_DeviceView<B> &vec_x,
            t_Vecs_DeviceView<B> &vec_y,
            index iRow, int iRow_blk)
        {
            auto dest = vec_y.father[iRow];
            dest(iRow_blk) = 0;
            for (rowsize j = 0; j < adj.RowSize(iRow); j++)
            {
                auto mat = mats.father(iRow, j);
                auto vec = vec_x[adj(iRow, j)];
                // for (int ii = 0; ii < dest.rows(); ii++)
                //     for (int jj = 0; jj < vec.rows(); jj++)
                //         dest(ii) += mat(ii, jj) * vec(jj);
                // vec_y.father[iRow] += mats.father(iRow, j) * vec_x[adj(iRow, j)];
                for (int k = 0; k < vec.rows(); k++)
                    dest(iRow_blk) += mat(iRow_blk, k) * vec(k);
            }
        }

        template <int rows_per_block = 1>
        DNDS_GLOBAL void block_mat_vec_product_by_row_block(
            t_Adj_DeviceView<DeviceBackend::CUDA> adj,
            t_Mats_DeviceView<DeviceBackend::CUDA> mats,
            t_Vecs_DeviceView<DeviceBackend::CUDA> vec_x,
            t_Vecs_DeviceView<DeviceBackend::CUDA> vec_y,
            int group_size)
        {
            // index tid_global = blockIdx.x * blockDim.x + threadIdx.x;
            int tid = threadIdx.x;
            index iRow = rows_per_block * blockIdx.x + tid / group_size;
            int iRow_blk = tid % group_size;
            if constexpr (rows_per_block == 1)
            {
                iRow = blockIdx.x;
                iRow_blk = tid;
            }
            if (iRow >= adj.father.Size())
                return;
            auto dest = vec_y.father[iRow];
            if (iRow_blk >= dest.rows())
                return; // safety
            block_mat_vec_product_one_row_subrow(adj, mats, vec_x, vec_y, iRow, iRow_blk);
        }

        DNDS_GLOBAL void block_mat_vec_product_by_row_block_dynamic(
            t_Adj_DeviceView<DeviceBackend::CUDA> adj,
            t_Mats_DeviceView<DeviceBackend::CUDA> mats,
            t_Vecs_DeviceView<DeviceBackend::CUDA> vec_x,
            t_Vecs_DeviceView<DeviceBackend::CUDA> vec_y,
            int rows_per_block,
            int group_size)
        {
            // index tid_global = blockIdx.x * blockDim.x + threadIdx.x;
            int tid = threadIdx.x;
            index iRow = rows_per_block * blockIdx.x + tid / group_size;
            int iRow_blk = tid % group_size;

            if (iRow >= adj.father.Size())
                return;
            auto dest = vec_y.father[iRow];
            if (iRow_blk >= dest.rows())
                return; // safety
            block_mat_vec_product_one_row_subrow(adj, mats, vec_x, vec_y, iRow, iRow_blk);
        }

        void run_cuda_thread_per_row(t_Adj &adj, t_Mats &mats, t_Vecs &vecs_x, t_Vecs &vecs_y,
                                     int v_size, double theoretical_GFLOP)
        {
            {
                int threadsPerBlock = 512;
                int blocksPerGrid = (adj.father->Size() + threadsPerBlock - 1) / threadsPerBlock;
                std::cout << fmt::format("threadsPerBlock: [{}], blocksPerGrid: [{}]", threadsPerBlock, blocksPerGrid) << std::endl;

                auto [t, niter] =
                    time_operation(
                        [&]()
                        {
                            block_mat_vec_product_by_row<<<blocksPerGrid, threadsPerBlock>>>(
                                adj.deviceView<DeviceBackend::CUDA>(),
                                mats.deviceView<DeviceBackend::CUDA>(),
                                vecs_x.deviceView<DeviceBackend::CUDA>(),
                                vecs_y.deviceView<DeviceBackend::CUDA>());
                            cudaDeviceSynchronize();
                        },
                        1, 10000, 1.0);
                vecs_y.to_host();
                std::cout << fmt::format("CUDA done [{}] times, time {:.4g}, GFLOPS {:.4g}", niter, t, theoretical_GFLOP / t) << std::endl;
            }
        }

        void run_cuda_block_per_row(t_Adj &adj, t_Mats &mats, t_Vecs &vecs_x, t_Vecs &vecs_y,
                                    int v_size, double theoretical_GFLOP)
        {
            {

                int group_size_blk = 32;
                int group_size = (v_size + group_size_blk - 1) / group_size_blk * group_size_blk;
                int total_threads = group_size * adj.father->Size();
                static const int rows_per_blk = 1;

                int threadsPerBlock = group_size * rows_per_blk;
                int blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
                std::cout << fmt::format("threadsPerBlock: [{}], blocksPerGrid: [{}]", threadsPerBlock, blocksPerGrid) << std::endl;

                auto [t, niter] =
                    time_operation(
                        [&]()
                        {
                            block_mat_vec_product_by_row_block<rows_per_blk><<<blocksPerGrid, threadsPerBlock>>>(
                                adj.deviceView<DeviceBackend::CUDA>(),
                                mats.deviceView<DeviceBackend::CUDA>(),
                                vecs_x.deviceView<DeviceBackend::CUDA>(),
                                vecs_y.deviceView<DeviceBackend::CUDA>(), group_size);
                            cudaDeviceSynchronize();
                        },
                        1, 10000, 1.0);
                vecs_y.to_host();
                std::cout << fmt::format("CUDA done [{}] times, time {:.4g}, GFLOPS {:.4g}", niter, t, theoretical_GFLOP / t) << std::endl;
            }
        }

        void run_cuda_block_per_row_dynamic(t_Adj &adj, t_Mats &mats, t_Vecs &vecs_x, t_Vecs &vecs_y,
                                            int v_size, double theoretical_GFLOP, std::string thread_group_strategy = "2n")
        {
            {
                int group_size_blk = 32;
                if (thread_group_strategy == "2n")
                {
                    if (v_size <= 16)
                        group_size_blk = 16;
                    if (v_size <= 8)
                        group_size_blk = 8;
                    if (v_size <= 4)
                        group_size_blk = 4;
                    if (v_size <= 2)
                        group_size_blk = 2;
                    if (v_size <= 1)
                        group_size_blk = 1;
                }
                else if (thread_group_strategy == "min32")
                {
                    group_size_blk = std::min(32, v_size);
                }
                int group_size = (v_size + group_size_blk - 1) / group_size_blk * group_size_blk;
                int total_threads = group_size * adj.father->Size();
                int rows_per_blk = (128 + group_size - 1) / group_size;
                std::cout << fmt::format("group_size: [{}], rows_per_blk: [{}]", group_size, rows_per_blk) << std::endl;

                int threadsPerBlock = group_size * rows_per_blk;
                int blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
                std::cout << fmt::format("threadsPerBlock: [{}], blocksPerGrid: [{}]", threadsPerBlock, blocksPerGrid) << std::endl;

                auto [t, niter] =
                    time_operation(
                        [&]()
                        {
                            block_mat_vec_product_by_row_block_dynamic<<<blocksPerGrid, threadsPerBlock>>>(
                                adj.deviceView<DeviceBackend::CUDA>(),
                                mats.deviceView<DeviceBackend::CUDA>(),
                                vecs_x.deviceView<DeviceBackend::CUDA>(),
                                vecs_y.deviceView<DeviceBackend::CUDA>(), rows_per_blk, group_size);
                            cudaDeviceSynchronize();
                        },
                        1, 10000, 1.0);
                vecs_y.to_host();

                std::cout << fmt::format("CUDA done [{}] times, time {:.4g}, GFLOPS {:.4g}", niter, t, theoretical_GFLOP / t) << std::endl;
            }
        }

        void test(MPIInfo &mpi, int n_x, int n_y, bool ifExpand, int v_size, int rseed = 0)
        {
            std::srand(rseed);
            std::cout << n_x << ", " << n_y << ", " << ifExpand << ", " << v_size << std::endl;
            std::cout << std::scientific << std::setprecision(4) << std::endl;
            int iter_max = 10000;

            t_Adj adj;
            DNDS_MAKE_SSP(adj.father, mpi);
            DNDS_MAKE_SSP(adj.son, mpi);
            int nElem = n_x * n_y;
            int nElem_ghost = n_x * 2 + n_y * 2;
            adj.father->Resize(nElem);
            adj.son->Resize(nElem_ghost);

            auto ij_to_ind = [n_x, n_y](int i, int j)
            { return i + j * n_x; };
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
            for (index i = 0; i < adj.father->Size(); i++)
            {
                std::vector<index> others;
                others.push_back(i);

                int i_x = i % n_x;
                int i_y = i / n_x;
                if (i_x > 0)
                    others.push_back(ij_to_ind(i_x - 1, i_y));
                else
                    others.push_back(n_x * n_y + i_y);
                if (i_x < n_x - 1)
                    others.push_back(ij_to_ind(i_x + 1, i_y));
                else
                    others.push_back(n_x * n_y + n_y + i_y);
                if (i_y > 0)
                    others.push_back(ij_to_ind(i_x, i_y - 1));
                else
                    others.push_back(n_x * n_y + n_y * 2 + i_x);
                if (i_y < n_y - 1)
                    others.push_back(ij_to_ind(i_x, i_y + 1));
                else
                    others.push_back(n_x * n_y + n_y * 2 + n_x + i_x);

                if (ifExpand)
                {
                    if (i_x > 1 && i_y % 2)
                        others.push_back(ij_to_ind(i_x - 2, i_y));
                    if (i_x < n_x - 2 && i_y % 2)
                        others.push_back(ij_to_ind(i_x + 2, i_y));
                    if (i_y > 1 && i_x % 2)
                        others.push_back(ij_to_ind(i_x, i_y - 2));
                    if (i_y < n_y - 2 && i_x % 2)
                        others.push_back(ij_to_ind(i_x, i_y + 2));
                }

                for (auto v : others)
                    DNDS_assert(v >= 0 && v < adj.Size());
                adj.ResizeRow(i, others.size());
                adj[i] = others;
            }

            adj.CompressBoth();
            adj.to_device(DeviceBackend::CUDA);

            double theoretical_GFLOP = adj.father->DataSize() * v_size * v_size * 1e-9;
            std::cout << fmt::format("nnz: {:.4g} G", theoretical_GFLOP) << std::endl;

            t_Mats mats;
            DNDS_MAKE_SSP(mats.father, mpi);
            DNDS_MAKE_SSP(mats.son, mpi);
            mats.father->Resize(nElem, v_size, v_size, [&](index iRow)
                                { return adj.RowSize(iRow); });
            mats.son->Resize(0 /* no ghosted */, v_size, v_size);
            MatrixXR randCore = MatrixXR::Random(v_size, v_size);

            // randCore.resize(3, 3);

#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
            for (index i = 0; i < adj.father->Size(); i++)
            {
                for (int j = 0; j < mats.RowSize(i); j++)
                {
                    mats(i, j) = randCore;
                    // std::cout << mats(i, j).size() << ", " << randCore.size() << std::endl;
                    // std::cout << mats(i, j) << std::endl;
                    // mats(i, j).setRandom();
                    if (j == 0)
                        mats(i, j) += MatrixXR::Identity(v_size, v_size);
                }
            }

            mats.CompressBoth();
            mats.to_device(DeviceBackend::CUDA);

            t_Vecs vecs_x, vecs_y, vecs_y1;
            auto make_vec = [&](t_Vecs &v)
            {
                DNDS_MAKE_SSP(v.father, mpi);
                DNDS_MAKE_SSP(v.son, mpi);
                v.father->Resize(nElem, v_size);
                v.son->Resize(nElem_ghost, v_size);
            };
            make_vec(vecs_x);
            make_vec(vecs_y);
            make_vec(vecs_y1);
            for (index i = 0; i < vecs_x.Size(); i++)
                vecs_x[i].setConstant(1.0);

            {
                auto [t, niter] = time_operation(
                    [&]()
                    {
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
                        for (index i = 0; i < adj.father->Size(); i++)
                            block_mat_vec_product_one_row(
                                adj.deviceView<DeviceBackend::Host>(),
                                mats.deviceView<DeviceBackend::Host>(),
                                vecs_x.deviceView<DeviceBackend::Host>(),
                                vecs_y1.deviceView<DeviceBackend::Host>(), i);
                    },
                    1, 10000, 1.0);
                std::cout << fmt::format("CPU done [{}] times, time {:.4g}, GFLOPS {:.4g}", niter, t, theoretical_GFLOP / t) << std::endl;
            }
            // for (index i = 0; i < vecs_y1.father->Size(); i++)
            //     std::cout << vecs_y1[i].transpose() << "\n";
            real sqr_norm_cpu = 0;
            for (index i = 0; i < vecs_y1.father->Size(); i++)
                sqr_norm_cpu += vecs_y1[i].squaredNorm();
            std::cout << fmt::format("CPU sqr norm  {:.16e}", sqr_norm_cpu) << std::endl;

            // std::cout << "\nGot CUDA:\n";
            // for (index i = 0; i < vecs_y.father->Size(); i++)
            //     std::cout << vecs_y[i].transpose() << "\n";
            auto check_cuda = [&]()
            {
                real sqr_norm_cuda = 0;
                for (index i = 0; i < vecs_y.father->Size(); i++)
                    sqr_norm_cuda += vecs_y[i].squaredNorm();
                std::cout << fmt::format("CUDA sqr norm  {:.16e}", sqr_norm_cuda) << std::endl;

                real sum_err = 0.0;
                for (index i = 0; i < vecs_y1.father->Size(); i++)
                {
                    real err = (vecs_y[i] - vecs_y1[i]).norm();
                    sum_err += err;
                    // printf("err %e", err);
                    DNDS_assert_infof(err < 1e-10, "err %e", err);
                }
                std::cout << fmt::format("sum_err : {}", sum_err) << std::endl;
            };
            vecs_x.to_device(DeviceBackend::CUDA);

            vecs_y.to_device(DeviceBackend::CUDA);
            std::cout << "\nCUDA: thread per row" << std::endl;
            run_cuda_thread_per_row(adj, mats, vecs_x, vecs_y, v_size, theoretical_GFLOP);
            check_cuda();

            vecs_y.to_device(DeviceBackend::CUDA);
            std::cout << "\nCUDA: block per row" << std::endl;
            run_cuda_block_per_row(adj, mats, vecs_x, vecs_y, v_size, theoretical_GFLOP);
            check_cuda();

            vecs_y.to_device(DeviceBackend::CUDA);
            std::cout << "\nCUDA: block per row dynamic" << std::endl;
            run_cuda_block_per_row_dynamic(adj, mats, vecs_x, vecs_y, v_size, theoretical_GFLOP);
            check_cuda();

            vecs_y.to_device(DeviceBackend::CUDA);
            std::cout << "\nCUDA: block per row dynamic min32" << std::endl;
            run_cuda_block_per_row_dynamic(adj, mats, vecs_x, vecs_y, v_size, theoretical_GFLOP, "min32");
            check_cuda();
        }
    }
}

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    DNDS_assert_info(argc >= 5, "need 4 args: nx ny if_expand, vsize");

    DNDS::array_cuda_Bench::test(mpi,
                                 std::stoi(argv[1]),
                                 std::stoi(argv[2]),
                                 std::stoi(argv[3]),
                                 std::stoi(argv[4]),
                                 0);

    DNDS::MPI::Finalize();
    return 0;
}
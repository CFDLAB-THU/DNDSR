#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "Solver/ScaledBlockInverse.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace DNDS::Algebra
{
    namespace
    {
        using namespace Direct::ScaledBlock;

        template <int N>
        __global__ void InvertRows(int rows, const int *diag, const double *values, double *output, int *failed)
        {
            const auto row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row >= size_t(rows))
                return;
            const auto status = InvertBlock<N>(values + size_t(diag[row]) * N * N, output + row * N * N);
            if (status != Success && atomicCAS(failed, -1, int(row)) == -1)
                failed[1] = status;
        }
    }

    int ComputeAmgXBlockInverse(int rows, int width, const int *diag, const double *values, double *output)
    {
        if (rows <= 0 || width < 4 || width > 6)
            return 1;
        // AmgX setup is already synchronous. Keep its numerical failure visible
        // before returning through the C ABI, including during coarse resetup.
        int *failed = nullptr;
        if (cudaMalloc(&failed, 2 * sizeof(int)) != cudaSuccess)
            return 1;
        int status = 1;
        if (cudaMemset(failed, -1, 2 * sizeof(int)) == cudaSuccess)
        {
            const unsigned blocks = 1 + (unsigned(rows) - 1) / 128;
            switch (width)
            {
            case 4:
                InvertRows<4><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 5:
                InvertRows<5><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 6:
                InvertRows<6><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            }
            int failure[2];
            if (cudaGetLastError() == cudaSuccess &&
                cudaMemcpy(failure, failed, sizeof(failure), cudaMemcpyDeviceToHost) == cudaSuccess)
            {
                status = failure[0] != -1;
                if (status)
                {
                    const char *reason = failure[1] == NonfiniteInput     ? "nonfinite block input"
                                         : failure[1] == SVDNoConvergence ? "SVD did not converge"
                                                                          : "nonfinite inverse";
                    std::fprintf(stderr, "AmgX ACM inverse failed on row %d of %d (width %d): %s\n",
                                 failure[0], rows, width, reason);
                    int block;
                    double entries[36];
                    if (cudaMemcpy(&block, diag + failure[0], sizeof(block), cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(entries, values + size_t(block) * width * width, width * width * sizeof(double),
                                   cudaMemcpyDeviceToHost) == cudaSuccess)
                    {
                        std::fprintf(stderr, "AmgX diagonal block row-major:");
                        for (int i = 0; i < width * width; ++i)
                            std::fprintf(stderr, " %.17g", entries[i]);
                        std::fputc('\n', stderr);
                    }
                }
            }
        }
        return cudaFree(failed) == cudaSuccess ? status : 1;
    }
}

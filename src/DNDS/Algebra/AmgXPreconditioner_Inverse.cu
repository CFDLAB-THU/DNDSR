#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "Solver/SmallBlockInverse.hpp"
#include <cuda_runtime.h>

namespace DNDS::Algebra
{
    namespace
    {
        // A row scales by its signed diagonal when that normalization is finite.
        // Otherwise use its largest magnitude. A structural zero row stays zero
        // in the scaled problem. Dividing by divisors avoids overflowing 1/d.
        template <int N>
        __device__ bool InvertBlock(const double *block, double *output)
        {
            Direct::SmallDenseMatrix<N> matrix, inverse;
            double divisor[N];
            bool any = false;
            for (int i = 0; i < N; ++i)
            {
                double largest = 0;
                for (int j = 0; j < N; ++j)
                {
                    const double x = block[i * N + j];
                    if (!isfinite(x))
                        return false;
                    largest = fmax(largest, fabs(x));
                }
                any = any || largest > 0;
                const double d = block[i * N + i];
                divisor[i] = d != 0 && isfinite(1 / d) && isfinite(largest / d) ? d : (largest > 0 ? largest : 1);
                for (int j = 0; j < N; ++j)
                    matrix(i, j) = block[i * N + j] / divisor[i];
            }
            if (!any)
            {
                for (int k = 0; k < N * N; ++k)
                    output[k] = 0;
                return true;
            }
            int truncated = 0;
            if (!Direct::TrySmallLUInverse<N, false>(matrix, inverse) &&
                !Direct::SmallSVDInverse<N, false>(matrix, inverse, truncated))
                return false;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                {
                    const double x = inverse(i, j) / divisor[j];
                    if (!isfinite(x))
                        return false;
                    output[i * N + j] = x;
                }
            return true;
        }

        template <int N>
        __global__ void InvertRows(int rows, const int *diag, const double *values, double *output, int *failed)
        {
            const auto row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row < size_t(rows) && !InvertBlock<N>(values + size_t(diag[row]) * N * N, output + row * N * N))
                atomicExch(failed, 1);
        }
    }

    int ComputeAmgXBlockInverse(int rows, int width, const int *diag, const double *values, double *output)
    {
        if (rows <= 0 || width < 4 || width > 6)
            return 1;
        // AmgX setup is already synchronous. Keep its numerical failure visible
        // before returning through the C ABI, including during coarse resetup.
        int *failed = nullptr;
        if (cudaMalloc(&failed, sizeof(int)) != cudaSuccess)
            return 1;
        int status = 1;
        if (cudaMemset(failed, 0, sizeof(int)) == cudaSuccess)
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
            if (cudaGetLastError() != cudaSuccess ||
                cudaMemcpy(&status, failed, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
                status = 1;
        }
        return cudaFree(failed) == cudaSuccess ? status : 1;
    }
}

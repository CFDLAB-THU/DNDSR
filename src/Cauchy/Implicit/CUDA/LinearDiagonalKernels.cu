#include "Cauchy/Implicit/CUDA/LinearDiagonalKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "Solver/InverseRefinement.hpp"
#include "Solver/SmallBlockInverse.hpp"
#include <cuda_runtime.h>
#include <limits>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        __global__ void ResetCount(DiagonalCount count) { count(0, 0) = 0; }
        // Logical dense entries have one canonical meaning; the optional
        // row-major provider representation changes addresses, not algebra.
        __device__ int Entry(int i, int j, int n, bool rowMajor) { return rowMajor ? i * n + j : i + n * j; }

        template <int N>
        __device__ bool Normalize(DiagonalRead diagonal, index iCell,
                                  Direct::SmallDenseMatrix<N> &matrix, real (&scales)[N], bool rowMajor)
        {
            bool valid = true;
            for (int i = 0; i < N; ++i)
            {
                const real value = diagonal(iCell, i + N * i);
                scales[i] = 1 / value;
                valid = valid && isfinite(value) && value != 0 && isfinite(scales[i]);
            }
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i)
                {
                    matrix(i, j) = diagonal(iCell, Entry(i, j, N, rowMajor)) * scales[i];
                    valid = valid && isfinite(matrix(i, j));
                }
            return valid;
        }
        template <int N, bool Refine>
        __global__ void SmallLU(DiagonalRead diagonal, DiagonalWrite inverse,
                                DiagonalStatus status, DiagonalModes fallback,
                                DiagonalCount count, int steps, bool rowMajor)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iCell < diagonal.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                Direct::SmallDenseMatrix<N> matrix, result;
                real scales[N];
                fallback(iCell, 0) = 0;
                const bool valid = Normalize<N>(diagonal, iCell, matrix, scales, rowMajor);
                status(iCell, 0) = valid ? NumericalStatus{} : NumericalStatus{NumericalError::InvalidInput, 1, iCell};
                if (!valid)
                    continue;
                if (!Direct::TrySmallLUInverse<N, Refine>(matrix, result, steps))
                {
                    fallback(iCell, 0) = 1;
                    atomicAdd(reinterpret_cast<unsigned long long *>(count.data), 1ULL);
                    continue;
                }
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < N; ++i)
                    {
                        const real value = result(i, j) * scales[j];
                        inverse(iCell, Entry(i, j, N, rowMajor)) = value;
                        if (!isfinite(value))
                            status(iCell, 0) = NumericalStatus{NumericalError::InvalidState, 3, iCell};
                    }
            }
        }
        template <int N, bool Refine>
        __global__ void SmallFallback(DiagonalRead diagonal, DiagonalWrite inverse,
                                      DiagonalStatus status, DiagonalModes fallback, int steps, bool rowMajor)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iCell < diagonal.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                if (!fallback(iCell, 0))
                    continue;
                Direct::SmallDenseMatrix<N> matrix, result;
                real scales[N];
                Normalize<N>(diagonal, iCell, matrix, scales, rowMajor);
                int truncated = 0;
                if (!Direct::SmallSVDInverse<N, Refine>(matrix, result, truncated, steps))
                {
                    status(iCell, 0) = NumericalStatus{NumericalError::NoConvergence, 100, iCell};
                    continue;
                }
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < N; ++i)
                    {
                        const real value = result(i, j) * scales[j];
                        inverse(iCell, Entry(i, j, N, rowMajor)) = value;
                        if (!isfinite(value))
                            status(iCell, 0) = NumericalStatus{NumericalError::InvalidState, 3, iCell};
                    }
            }
        }
        __global__ void Prepare(DiagonalRead diagonal, DiagonalWrite normalized, DiagonalWrite scales, DiagonalStatus status, int n, bool rowMajor)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < diagonal.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                bool valid = true;
                for (int i = 0; i < n; ++i)
                {
                    const real value = diagonal(iCell, i + n * i);
                    const real scale = 1 / value;
                    valid = valid && isfinite(value) && value != 0 && isfinite(scale);
                    scales(iCell, i) = scale;
                }
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                    {
                        const real value = diagonal(iCell, Entry(i, j, n, rowMajor)) * scales(iCell, i);
                        valid = valid && isfinite(value);
                        normalized(iCell, i + n * j) = value;
                    }
                status(iCell, 0) = valid ? NumericalStatus{} : NumericalStatus{NumericalError::InvalidInput, 1, iCell};
                if (!valid)
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i)
                            normalized(iCell, i + n * j) = i == j ? 1 : 0;
            }
        }
        struct SmallMatrix
        {
            real values[49];
            DNDS_DEVICE_CALLABLE real &operator()(int i, int j) { return values[i + 7 * j]; }
            DNDS_DEVICE_CALLABLE real operator()(int i, int j) const { return values[i + 7 * j]; }
        };
        struct SingularRow
        {
            DiagonalRead values;
            index row;
            DNDS_DEVICE_CALLABLE real operator()(int k) const { return values(row, k); }
        };
        template <bool Refine>
        __global__ void Complete(DiagonalRead diagonal, DiagonalRead singular, DiagonalRead left, DiagonalRead right, DiagonalRead scales,
                                 DiagonalInfo info, DiagonalWrite inverse, DiagonalStatus status, DiagonalModes truncated, int n, int steps, bool rowMajor)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < inverse.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                truncated(iCell, 0) = 0;
                if (!status(iCell, 0))
                    continue;
                if (info(iCell, 0) != 0)
                {
                    status(iCell, 0) = NumericalStatus{NumericalError::NoConvergence, info(iCell, 0), iCell};
                    continue;
                }
                real maximum = 0;
                bool valid = true;
                for (int k = 0; k < n; ++k)
                {
                    const real value = singular(iCell, k);
                    valid = valid && isfinite(value) && value >= 0;
                    maximum = fmax(maximum, value);
                }
                if (!valid || maximum <= 0)
                {
                    status(iCell, 0) = NumericalStatus{NumericalError::InvalidState, 2, iCell};
                    continue;
                }
                const real cutoff = std::numeric_limits<real>::epsilon() * maximum;
                for (int k = 0; k < n; ++k)
                    truncated(iCell, 0) += singular(iCell, k) <= cutoff;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                    {
                        real value = 0;
                        for (int k = 0; k < n; ++k)
                            if (singular(iCell, k) > cutoff)
                                value += right(iCell, i + n * k) * (left(iCell, j + n * k) / singular(iCell, k));
                        if constexpr (!Refine)
                            value *= scales(iCell, j);
                        valid = valid && isfinite(value);
                        inverse(iCell, Entry(i, j, n, rowMajor)) = value;
                    }
                if constexpr (Refine)
                {
                    // Separate specialization keeps refinement scratch out of
                    // the default kernel. Rebuild B from the original block:
                    // cuSOLVER overwrites its input workspace during SVD.
                    if (valid)
                    {
                        SmallMatrix a, u, v, x, c, z, residual, trial;
                        for (int i = 0; i < n; ++i)
                            for (int j = 0; j < n; ++j)
                            {
                                a(i, j) = diagonal(iCell, Entry(i, j, n, rowMajor)) * scales(iCell, i);
                                u(i, j) = left(iCell, i + n * j);
                                v(i, j) = right(iCell, i + n * j);
                                x(i, j) = inverse(iCell, Entry(i, j, n, rowMajor));
                            }
                        Direct::RefineSVDInverse(a, u, v, SingularRow{singular, iCell}, x, c, z, residual, trial, n, steps);
                        for (int i = 0; i < n; ++i)
                            for (int j = 0; j < n; ++j)
                            {
                                const real value = x(i, j) * scales(iCell, j);
                                valid = valid && isfinite(value);
                                inverse(iCell, Entry(i, j, n, rowMajor)) = value;
                            }
                    }
                }
                if (!valid)
                    status(iCell, 0) = NumericalStatus{NumericalError::InvalidState, 3, iCell};
            }
        }
        __global__ void Entries(DiagonalRead diagonal, DiagonalWrite inverse, DiagonalStatus status, int n)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < diagonal.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                bool valid = true;
                for (int j = 0; j < n; ++j)
                {
                    const real value = diagonal(iCell, j), result = 1 / value;
                    valid = valid && isfinite(value) && value != 0 && isfinite(result);
                    inverse(iCell, j) = result;
                }
                status(iCell, 0) = valid ? NumericalStatus{} : NumericalStatus{NumericalError::InvalidState, 1, iCell};
            }
        }
        unsigned Blocks(index rows) { return unsigned(std::min<index>(65535, rows / 128 + (rows % 128 != 0))); }
        template <class View>
        void Table(View view, R::Device device, index rows, int width)
        {
            view.ValidateDevice(device);
            DNDS_check_throw_info((view.kind == R::RowLayoutKind::Table || view.kind == R::RowLayoutKind::ColMajorTable ||
                                   view.kind == R::RowLayoutKind::TiledColMajorTable) &&
                                      view.RowCount() == rows && view.Width() == width,
                                  "CUDA inverse array shape differs");
        }
        void CheckContext(R::ExecutionContext context, int n)
        {
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && n >= 5 && n <= 7,
                                  "CUDA inverse requires CUDA and 5/6/7 components");
        }
        void CheckLaunch() { DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA diagonal inverse kernel launch failed"); }
    }
    void InvertSmallBlocks(R::ExecutionContext context, DiagonalRead diagonal,
                           DiagonalWrite inverse, DiagonalStatus status, DiagonalModes fallback,
                           DiagonalCount count, int n, int refinementSteps, bool rowMajor)
    {
        CheckContext(context, n);
        DNDS_check_throw_info(refinementSteps >= 0 && refinementSteps <= 3, "invalid small-block refinement count");
        const index rows = diagonal.RowCount();
        Table(diagonal, context.device, rows, n * n);
        Table(inverse, context.device, rows, n * n);
        Table(status, context.device, rows, 1);
        Table(fallback, context.device, rows, 1);
        Table(count, context.device, 1, 1);
        DNDS_check_throw_info(!rows || diagonal.data != inverse.data, "small-block inverse aliases diagonal");
        R::CheckKernelArgumentBudget<DiagonalRead, DiagonalWrite, DiagonalStatus, DiagonalModes, DiagonalCount, int, bool>();
        const auto stream = static_cast<cudaStream_t>(context.stream);
        ResetCount<<<1, 1, 0, stream>>>(count);
        if (rows)
        {
            const auto launch = [&](auto size, auto refine)
            {
                constexpr int N = decltype(size)::value;
                constexpr bool refineValue = decltype(refine)::value;
                SmallLU<N, refineValue><<<Blocks(rows), 128, 0, stream>>>(diagonal, inverse, status, fallback, count, refinementSteps, rowMajor);
                SmallFallback<N, refineValue><<<Blocks(rows), 128, 0, stream>>>(diagonal, inverse, status, fallback, refinementSteps, rowMajor);
            };
            const auto dispatch = [&](auto refine)
            {
                if (n == 5)
                    launch(std::integral_constant<int, 5>{}, refine);
                else if (n == 6)
                    launch(std::integral_constant<int, 6>{}, refine);
                else
                    launch(std::integral_constant<int, 7>{}, refine);
            };
            if (refinementSteps)
                dispatch(std::true_type{});
            else
                dispatch(std::false_type{});
        }
        CheckLaunch();
    }
    void PrepareDiagonalSVD(R::ExecutionContext context, DiagonalRead diagonal, DiagonalWrite normalized,
                            DiagonalWrite scales, DiagonalStatus status, int n, bool rowMajor)
    {
        CheckContext(context, n);
        const index rows = diagonal.RowCount();
        Table(diagonal, context.device, rows, n * n);
        Table(normalized, context.device, rows, n * n);
        Table(scales, context.device, rows, n);
        Table(status, context.device, rows, 1);
        DNDS_check_throw_info(!rows || (diagonal.data != normalized.data && diagonal.data != scales.data && normalized.data != scales.data),
                              "CUDA SVD preparation operands alias");
        R::CheckKernelArgumentBudget<DiagonalRead, DiagonalWrite, DiagonalWrite, DiagonalStatus, int, bool>();
        if (!rows)
            return;
        Prepare<<<Blocks(rows), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(diagonal, normalized, scales, status, n, rowMajor);
        CheckLaunch();
    }
    void CompleteDiagonalSVD(R::ExecutionContext context, DiagonalRead diagonal, DiagonalRead singular, DiagonalRead left, DiagonalRead right,
                             DiagonalRead scales, DiagonalInfo info, DiagonalWrite inverse, DiagonalStatus status,
                             DiagonalModes truncated, int n, int refinementSteps, bool rowMajor)
    {
        CheckContext(context, n);
        const index rows = inverse.RowCount();
        DNDS_check_throw_info(refinementSteps >= 0 && refinementSteps <= 3, "invalid diagonal inverse refinement count");
        Table(diagonal, context.device, rows, n * n);
        Table(singular, context.device, rows, n);
        Table(scales, context.device, rows, n);
        Table(left, context.device, rows, n * n);
        Table(right, context.device, rows, n * n);
        Table(inverse, context.device, rows, n * n);
        Table(info, context.device, rows, 1);
        Table(status, context.device, rows, 1);
        Table(truncated, context.device, rows, 1);
        DNDS_check_throw_info(!rows || (inverse.data != diagonal.data && inverse.data != singular.data && inverse.data != left.data && inverse.data != right.data && inverse.data != scales.data),
                              "CUDA SVD inverse aliases an input");
        R::CheckKernelArgumentBudget<DiagonalRead, DiagonalRead, DiagonalRead, DiagonalRead, DiagonalRead, DiagonalInfo, DiagonalWrite, DiagonalStatus, DiagonalModes, int, int, bool>();
        if (!rows)
            return;
        if (refinementSteps)
            Complete<true><<<Blocks(rows), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(diagonal, singular, left, right, scales, info, inverse, status, truncated, n, refinementSteps, rowMajor);
        else
            Complete<false><<<Blocks(rows), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(diagonal, singular, left, right, scales, info, inverse, status, truncated, n, 0, rowMajor);
        CheckLaunch();
    }
    void InvertDiagonalEntries(R::ExecutionContext context, DiagonalRead diagonal, DiagonalWrite inverse, DiagonalStatus status, int n)
    {
        CheckContext(context, n);
        const index rows = diagonal.RowCount();
        Table(diagonal, context.device, rows, n);
        Table(inverse, context.device, rows, n);
        Table(status, context.device, rows, 1);
        DNDS_check_throw_info(!rows || diagonal.data != inverse.data, "CUDA inverse output aliases diagonal");
        R::CheckKernelArgumentBudget<DiagonalRead, DiagonalWrite, DiagonalStatus, int>();
        if (!rows)
            return;
        Entries<<<Blocks(rows), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(diagonal, inverse, status, n);
        CheckLaunch();
    }
}

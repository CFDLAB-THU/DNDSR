#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "CFV/Math/ReconstructionMatrixAccess.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    using RawJacobiView = RawReconstructionSweepView;
    /** @brief Gather original B contributions then apply the local pseudoinverse.
     * @details One block owns one cell. Its shared right-hand side is a bounded
     * 19-by-7 scratch. Sharing a face B uses its canonical parent orientation,
     * with a transpose for the reverse incidence; vector-component rotation is
     * still separate. Same-frame periodic self terms already belong in A.
     * The relaxed/LVR target contribution bypasses A-inverse. Alpha=1 must not
     * read any neighbor, boundary value or inverse, including NaN sentinels.
     */
    template <int FixedModes, int Components, bool SharedMatrix, bool Limited>
    __global__ void RawJacobi(RawJacobiView plan, DeviceWorkSelection selection, DevicePeriodicBlocks blocks,
                              R::PairReadView<real> mean, R::PairReadView<real> input,
                              R::ArrayView<const real> boundary, R::ArrayView<real> output,
                              R::ArrayView<const int> status, R::ArrayView<const real> target, R::ArrayView<const real> alpha)
    {
        if (status(0, 0))
            return;
        const index n = FixedModes ? FixedModes : plan.n;
        const index width = Components ? Components * n : output.width;
        extern __shared__ real words[];
        real *right = words, *coefficients = right + width, *matrix = coefficients + width;
        for (index iWork = blockIdx.x; iWork < selection.count; iWork += gridDim.x)
        {
            const index iCell = selection.Cell(iWork);
            const real penalty = Limited ? alpha(iCell, 0) : 0;
            const real scale = plan.relax * (1 - penalty);
            for (index i = threadIdx.x; i < width; i += blockDim.x)
                right[i] = 0;
            __syncthreads();
            if (scale != 0)
            {
                for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
                {
                    const index other = plan.neighbors(entry, 0);
                    const bool rotated = plan.rotated && plan.rotationActive(entry, 0);
                    // Resolve routing and physical stride once per incidence,
                    // then stage/dot only a const row span in the inner loops.
                    const index matrixRow = plan.faceCoupling ? plan.faces(entry, 0) : entry;
                    const bool transpose = plan.faceCoupling && plan.faceSide(entry, 0);
                    const auto block = plan.matrices[matrixRow];
                    if (other != UnInitIndex)
                    {
                        for (index i = threadIdx.x; i < width; i += blockDim.x)
                            coefficients[i] = input(other, i);
                        if constexpr (SharedMatrix)
                            for (index k = threadIdx.x; k < n * n; k += blockDim.x)
                                matrix[k] = block[transpose ? k / n + n * (k % n) : k];
                    }
                    __syncthreads();
                    for (index i = threadIdx.x; i < width; i += blockDim.x)
                    {
                        const index mode = i % n, component = i / n;
                        if (other != UnInitIndex)
                        {
                            const index first = rotated ? blocks.First(component) : -1;
                            if (other != iCell || rotated)
                            {
                                real product = 0;
                                for (index j = 0; j < n; ++j)
                                {
                                    const real b = SharedMatrix ? matrix[j * n + mode] : block[transpose ? j + n * mode : mode + n * j];
                                    product += b * coefficients[component * n + j];
                                }
                                right[i] += product;
                                if (first >= 0)
                                {
                                    real correction = 0;
                                    for (index j = 0; j < n; ++j)
                                    {
                                        real value = 0;
                                        for (index d = 0; d < 3; ++d)
                                            value += coefficients[(first + d) * n + j] *
                                                     (plan.rotation(entry, d * 3 + component - first) - (component - first == d ? 1 : 0));
                                        const real b = SharedMatrix ? matrix[j * n + mode] : block[transpose ? j + n * mode : mode + n * j];
                                        correction += b * value;
                                    }
                                    right[i] += correction;
                                }
                            }
                            real otherMean = mean(other, component);
                            if (first >= 0)
                            {
                                otherMean = 0;
                                for (index d = 0; d < 3; ++d)
                                    otherMean += plan.rotation(entry, d * 3 + component - first) * mean(other, first + d);
                            }
                            right[i] += plan.vectors(entry, mode) * (otherMean - mean(iCell, component));
                        }
                        else
                            for (index q = plan.boundaryOffsets(entry, 0); q < plan.boundaryOffsets(entry + 1, 0); ++q)
                                right[i] += plan.injection(q, mode) * (boundary(q, component) - mean(iCell, component));
                    }
                    __syncthreads();
                }
            }
            __syncthreads();
            const auto inverse = plan.inverse[iCell];
            for (index i = threadIdx.x; i < width; i += blockDim.x)
            {
                real value = input(iCell, i) * (1 - plan.relax);
                if constexpr (Limited)
                    if (penalty > 0)
                        value += plan.relax * penalty * target(iCell, i);
                if (scale != 0)
                {
                    real update = 0;
                    for (index j = 0; j < n; ++j)
                        update += inverse[plan.triangle ? Math::SymmetricTriangleEntry(i % n, j) : i % n + n * j] * right[(i / n) * n + j];
                    value += scale * update;
                }
                output(iCell, i) = value;
            }
            __syncthreads(); // The next grid-stride cell reuses the shared RHS.
        }
    }
    void LaunchRawJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                         DevicePeriodicBlocks blocks, R::PairReadView<real> mean, R::PairReadView<real> input,
                         R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                         R::ArrayView<const real> target, R::ArrayView<const real> alpha, bool limited, bool matrixShared)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && plan.raw && !plan.quantized &&
                                  plan.n > 0 && plan.n <= 19 && mean.owned.Width() > 0 && mean.owned.Width() <= INT_MAX / plan.n &&
                                  output.Width() == plan.n * mean.owned.Width() &&
                                  input.owned.Width() == output.Width() && output.RowCount() == mean.owned.RowCount() &&
                                  plan.matrixAInvB.Width() == plan.n * plan.n && plan.vectorAInvB.Width() == plan.n &&
                                  plan.localInverse.Width() == (plan.inverseTriangle ? plan.n * (plan.n + 1) / 2 : plan.n * plan.n),
                              "raw CUDA VR requires FP64 P1..P3 and a valid component count");
        R::ValidateViewDevices(context.device, plan, selection, blocks, mean, input, boundary, output, status, target, alpha);
        R::CheckKernelArgumentBudget<RawJacobiView, decltype(selection), decltype(blocks), decltype(mean), decltype(input),
                                     decltype(boundary), decltype(output), decltype(status), decltype(target), decltype(alpha)>();
        if (!selection.count)
            return;
        const auto stream = static_cast<cudaStream_t>(context.stream);
        const unsigned grid = unsigned(std::min<index>(selection.count, matrixShared || limited ? 65535 : 256));
        const auto bytes = size_t(2 * output.width + (matrixShared ? plan.n * plan.n : 0)) * sizeof(real);
        const auto launch = [&](auto modes)
        {
            constexpr int N = decltype(modes)::value;
            const auto components = [&](auto count)
            {
                constexpr int C = decltype(count)::value;
                const auto limitedLaunch = [&](auto limit)
                {
                    constexpr bool L = decltype(limit)::value;
                    if (matrixShared)
                        RawJacobi<N, C, true, L><<<grid, 96, bytes, stream>>>(RawJacobiView(plan), selection, blocks, mean, input, boundary, output, status, target, alpha);
                    else
                        RawJacobi<N, C, false, L><<<grid, limited ? 96 : 128, bytes, stream>>>(RawJacobiView(plan), selection, blocks, mean, input, boundary, output, status, target, alpha);
                };
                if (limited)
                    limitedLaunch(std::true_type{});
                else
                    limitedLaunch(std::false_type{});
            };
            switch (mean.owned.width)
            {
            case 5:
                components(std::integral_constant<int, 5>{});
                break;
            case 6:
                components(std::integral_constant<int, 6>{});
                break;
            case 7:
                components(std::integral_constant<int, 7>{});
                break;
            default:
                components(std::integral_constant<int, 0>{});
                break;
            }
        };
        switch (plan.n)
        {
        case 2:
            launch(std::integral_constant<int, 2>{});
            break;
        case 3:
            launch(std::integral_constant<int, 3>{});
            break;
        case 5:
            launch(std::integral_constant<int, 5>{});
            break;
        case 9:
            launch(std::integral_constant<int, 9>{});
            break;
        case 19:
            launch(std::integral_constant<int, 19>{});
            break;
        default:
            launch(std::integral_constant<int, 0>{});
            break;
        }

        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "raw CUDA Jacobi launch failed");
    }
}

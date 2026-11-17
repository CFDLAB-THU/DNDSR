#include "Cauchy/Discretization/CUDA/FiniteVolumeCharacteristicKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "DNDS/Algebra/CUDA/CountRows.cuh"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        using Means = R::PairReadView<real, R::AccessBoundsChecks, R::TableRows>;
        using Status = R::ArrayView<NumericalStatus, R::AccessBoundsChecks, R::FixedRows<1>>;
        using Flags = R::ArrayView<int, R::AccessBoundsChecks, R::FixedRows<1>>;
        __device__ FlowState Load(Means mean, index iCell)
        {
            FlowState value{};
            for (int j = 0; j < 5; ++j)
                value[j] = mean(iCell, j);
            return value;
        }
        __global__ void Characteristics(CharacteristicGeometry geometry, Means mean, PerfectGas gas,
                                        CharacteristicOutput forward, CharacteristicOutput inverse, Status status, Flags fallback)
        {
            const auto &stencil = geometry.stencil(0, 0);
            for (index iFace = index(blockIdx.x) * blockDim.x + threadIdx.x; iFace < forward.RowCount(); iFace += index(gridDim.x) * blockDim.x)
            {
                for (int j = 0; j < 25; ++j)
                    forward(iFace, j) = inverse(iFace, j) = (j % 6 == 0);
                status(iFace, 0) = {};
                fallback(iFace, 0) = 0;
                if (stencil.routing(iFace, 0).auxiliaryFace == UnInitIndex)
                    continue;
                const auto parents = stencil.topology.FaceCells(iFace);
                const auto left = Load(mean, parents[0]);
                auto right = parents[1] == UnInitIndex ? left : Load(mean, parents[1]);
                if (parents[1] != UnInitIndex && stencil.metric.periodic.Rotates(iFace))
                {
                    const auto rotation = stencil.metric.periodic.OtherRotation(iFace, 0);
                    const auto original = right;
                    for (int i = 0; i < 3; ++i)
                    {
                        right[1 + i] = 0;
                        for (int j = 0; j < 3; ++j)
                            right[1 + i] += rotation(i, j) * original[1 + j];
                    }
                }
                const auto direction = stencil.metric.GetFaceMeanNorm(iFace);
                const Vector3 normal{direction[0], direction[1], direction[2]};
                const auto result = TryPerfectGasCharacteristics(gas, left, right, normal);
                status(iFace, 0) = result.status.At(-1, iFace);
                if (!result)
                    continue;
                fallback(iFace, 0) = result.value.componentwiseFallback;
                for (int j = 0; j < 25; ++j)
                {
                    forward(iFace, j) = result.value.forward[j];
                    inverse(iFace, j) = result.value.inverse[j];
                }
            }
        }
    }
    namespace
    {
        using Count = R::ArrayView<index, R::AccessBoundsChecks, R::FixedRows<1>>;
    }
    void ReduceCharacteristicFallbacks(R::ExecutionContext context,
                                       R::ArrayView<const int, R::AccessBoundsChecks, R::FixedRows<1>> flags,
                                       Count partial, Count result)
    {
        R::ValidateViewDevices(context.device, flags, partial, result);
        const index blocks = std::max<index>(1, std::min<index>(256, (flags.RowCount() + 127) / 128));
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && partial.RowCount() == blocks && result.RowCount() == 1,
                              "characteristic fallback reduction scratch differs");
        const auto stream = static_cast<cudaStream_t>(context.stream);
        Algebra::CUDA::CountRows<128, false><<<unsigned(blocks), 128, 0, stream>>>(flags, partial);
        Algebra::CUDA::CountRows<128, false><<<1, 128, 0, stream>>>(partial.ConstView(), result);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "characteristic fallback reduction launch failed");
    }
    void PrepareFaceCharacteristicMatrices(R::ExecutionContext context, CharacteristicGeometry geometry, Means mean,
                                           PerfectGas gas, CharacteristicOutput forward, CharacteristicOutput inverse, Status status, Flags fallback)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA characteristics require a CUDA context");
        gas.Validate();
        R::ValidateViewDevices(context.device, geometry, mean, forward, inverse, status, fallback);
        DNDS_check_throw_info(mean.owned.kind == R::RowLayoutKind::Table && mean.owned.Width() >= 5 && mean.owned.Width() <= 7 &&
                                  (!mean.ghost.RowCount() || (mean.ghost.kind == R::RowLayoutKind::Table && mean.ghost.Width() == mean.owned.Width())) &&
                                  forward.kind == R::RowLayoutKind::Table && inverse.kind == R::RowLayoutKind::Table &&
                                  forward.Width() == 25 && inverse.Width() == 25 && forward.RowCount() == inverse.RowCount() &&
                                  geometry.stencil.RowCount() == 1 && status.RowCount() == forward.RowCount() && fallback.RowCount() == forward.RowCount(),
                              "CUDA characteristic field/geometry shape differs");
        R::CheckKernelArgumentBudget<CharacteristicGeometry, Means, PerfectGas, CharacteristicOutput, CharacteristicOutput, Status, Flags>();
        if (!forward.RowCount())
            return;
        Characteristics<<<unsigned(std::min<index>(256, (forward.RowCount() + 127) / 128)), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(
            geometry, mean, gas, forward, inverse, status, fallback);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA characteristic producer launch failed");
    }
}

#pragma once

#include <cuda_runtime.h>
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Faces.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    using RHSGeometryView = Runtime::ArrayView<const RHSGeometry, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using RHSParametersView = Runtime::ArrayView<const RHSKernelParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /** @brief Shared face entry point, specialized only at the host launcher.
     * @details The compute scalars describe local arithmetic, not allocation.
     * Fields is a validated POD view with static or uniform physical indexing.
     * The publishing operator retains every nested input/output owner until
     * stream completion. There is no host synchronization or allocation here.
     */
    template <int Components, class PolynomialScalar = real, class FluxScalar = real,
              class DissipationScalar = real, bool WarpQuadrature = false, class Fields>
    __global__ void FluxFaces(RHSGeometryView published, RHSParametersView parameters, RHSKernelControls controls,
                              Fields fields, RHSFaceOutputs output, RHSIndices selected)
    {
        const auto &geometry = published(0, 0);
        const auto &physics = parameters(0, 0);
        const index thread = index(blockIdx.x) * blockDim.x + threadIdx.x;
        for (index iRow = WarpQuadrature ? thread / 32 : thread; iRow < selected.RowCount();
             iRow += index(gridDim.x) * blockDim.x / (WarpQuadrature ? 32 : 1))
        {
            const index iFace = selected(iRow, 0);
            index fallbacks = 0;
            const auto status = physics.physics.rans.Supported() && physics.physics.rans.ComponentNumber() + 5 == Components &&
                                        physics.physics.dimension == controls.dimension && (!controls.viscous || physics.physics.viscous) &&
                                        (std::is_same_v<FluxScalar, real> || !physics.physics.useQCR)
                                    ? EvaluateFace<Components, PolynomialScalar, FluxScalar, DissipationScalar, WarpQuadrature>(geometry, physics, controls, fields, output, iFace, fallbacks)
                                    : NumericalStatus{NumericalError::InvalidInput, 1};
            if (!WarpQuadrature || threadIdx.x % 32 == 0)
            {
                output.status(iFace, 0) = status;
                output.fallbackCounts(iFace, 0) = fallbacks;
            }
        }
    }
    template <int Components, class PolynomialScalar = real, class FluxScalar = real,
              class DissipationScalar = real, class Fields>
    void LaunchFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                         RHSParametersView parameters, RHSKernelControls controls, Fields fields,
                         RHSFaceOutputs output, RHSIndices selected)
    {
        Runtime::CheckKernelArgumentBudget<RHSGeometryView, RHSParametersView, RHSKernelControls, Fields, RHSFaceOutputs, RHSIndices>();
        const unsigned blocks = unsigned(std::min<index>((selected.RowCount() + (controls.warpQuadrature ? 3 : 127)) /
                                                             (controls.warpQuadrature ? 4 : 128),
                                                         1024));
        if constexpr (Components == 5 && std::is_same_v<PolynomialScalar, real> && std::is_same_v<FluxScalar, real> && std::is_same_v<DissipationScalar, real>)
            if (controls.warpQuadrature)
            {
                FluxFaces<Components, PolynomialScalar, FluxScalar, DissipationScalar, true><<<blocks, 128, 0, static_cast<cudaStream_t>(context.stream)>>>(
                    geometry.View(), parameters, controls, fields, output, selected);
                return;
            }
        FluxFaces<Components, PolynomialScalar, FluxScalar, DissipationScalar, false><<<blocks, 128, 0, static_cast<cudaStream_t>(context.stream)>>>(
            geometry.View(), parameters, controls, fields, output, selected);
    }
}

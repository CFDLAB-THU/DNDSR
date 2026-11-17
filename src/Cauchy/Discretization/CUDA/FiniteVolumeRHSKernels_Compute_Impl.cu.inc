// Private CUDA comparison implementation, included only by its scalar TUs.
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
              class DissipationScalar = real, class Fields, class Outputs>
    __global__ void FluxFaces(RHSGeometryView published, RHSParametersView parameters, RHSKernelControls controls,
                              Fields fields, Outputs output, RHSIndices selected)
    {
        const auto &geometry = published(0, 0);
        const auto &physics = parameters(0, 0);
        const index thread = index(blockIdx.x) * blockDim.x + threadIdx.x;
        for (index iRow = thread; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
        {
            const index iFace = selected(iRow, 0);
            index fallbacks = 0;
            const auto status = physics.physics.rans.Supported() && physics.physics.rans.ComponentNumber() + 5 == Components &&
                                        physics.physics.dimension == controls.dimension && (!controls.viscous || physics.physics.viscous) &&
                                        (std::is_same_v<FluxScalar, real> || !physics.physics.useQCR)
                                    ? EvaluateFace<Components, PolynomialScalar, FluxScalar, DissipationScalar>(geometry, physics, controls, fields, output, iFace, fallbacks)
                                    : NumericalStatus{NumericalError::InvalidInput, 1};
            output.status(iFace, 0) = status;
            output.fallbackCounts(iFace, 0) = fallbacks;
        }
    }
    template <int Components, class PolynomialScalar = real, class FluxScalar = real,
              class DissipationScalar = real, class Fields, class Outputs>
    void LaunchFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                         RHSParametersView parameters, RHSKernelControls controls, Fields fields,
                         Outputs output, RHSIndices selected)
    {
        Runtime::CheckKernelArgumentBudget<RHSGeometryView, RHSParametersView, RHSKernelControls, Fields, Outputs, RHSIndices>();
        const unsigned blocks = unsigned(std::min<index>((selected.RowCount() + controls.blockSize - 1) / controls.blockSize,
                                                         controls.gridBlocks ? index(controls.gridBlocks) : index(1024)));
        FluxFaces<Components, PolynomialScalar, FluxScalar, DissipationScalar><<<blocks, controls.blockSize, 0, static_cast<cudaStream_t>(context.stream)>>>(
            geometry.View(), parameters, controls, fields, output, selected);
    }
}

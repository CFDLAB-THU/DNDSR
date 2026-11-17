#include "CFV/Detail/ReconstructionAMGKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "DNDS/Runtime/Execution/KernelViews.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace
    {
        using Read = R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>;
        using Write = R::ArrayView<real, R::AccessBoundsChecks, R::TableRows>;
        __global__ void Defect(ReconstructionAMGGeometry geometry, Read mean, Read input, Read boundary, Write output)
        {
            const int n = geometry.modes;
            const index components = mean.Width();
            for (index iWork = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iWork < output.RowCount() * components; iWork += index(gridDim.x) * blockDim.x)
            {
                const index iCell = iWork / components, component = iWork % components;
                for (int mode = 0; mode < n; ++mode)
                {
                    real value = 0;
                    for (index entry = geometry.offsets(iCell, 0); entry < geometry.offsets(iCell + 1, 0); ++entry)
                    {
                        const index other = geometry.neighbors(entry, 0);
                        if (other != UnInitIndex)
                            value += geometry.vectors(entry, mode) * (mean(other, component) - mean(iCell, component));
                        else
                            for (index q = geometry.boundaryOffsets(entry, 0); q < geometry.boundaryOffsets(entry + 1, 0); ++q)
                                value += geometry.injection(q, mode) * (boundary(q, component) - mean(iCell, component));
                    }
                    for (index entry = geometry.matrixOffsets(iCell, 0); entry < geometry.matrixOffsets(iCell + 1, 0); ++entry)
                    {
                        const index other = geometry.matrixColumns(entry, 0);
                        real product = 0;
                        for (int j = 0; j < n; ++j)
                            product += geometry.matrixValues(entry * n * n + mode * n + j, 0) * input(other, component * n + j);
                        value -= product;
                    }
                    output(iCell, component * n + mode) = value;
                }
            }
        }
        template <bool Scatter>
        __global__ void Component(Read input, Read packed, Write output, index component, int modes)
        {
            const index count = Scatter ? packed.RowCount() : output.RowCount();
            for (index i = index(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += index(gridDim.x) * blockDim.x)
            {
                const index iCell = i / modes, entry = component * modes + i % modes;
                if constexpr (Scatter)
                    output(iCell, entry) = input(iCell, entry) + packed(i, 0);
                else
                    output(i, 0) = input(iCell, entry);
            }
        }
        unsigned Blocks(index count) { return unsigned(std::max<index>(1, std::min<index>(1024, (count + 127) / 128))); }
        void CheckLaunch() { DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "VR AMG kernel launch failed"); }
    }
    void BuildReconstructionAMGDefect(R::ExecutionContext context, ReconstructionAMGGeometry geometry,
                                      R::ArrayView<const real> mean, R::ArrayView<const real> input,
                                      R::ArrayView<const real> boundary, R::ArrayView<real> defect)
    {
        R::ValidateViewDevices(context.device, geometry, mean, input, boundary, defect);
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && geometry.modes > 0 &&
                                  input.Width() == mean.Width() * geometry.modes && input.RowCount() == mean.RowCount() &&
                                  defect.Width() == input.Width() && defect.RowCount() == input.RowCount(),
                              "VR AMG defect shape differs");
        R::CheckKernelArgumentBudget<ReconstructionAMGGeometry, Read, Read, Read, Write>();
        Defect<<<Blocks(defect.RowCount() * mean.Width()), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(
            geometry, R::ProjectKernelView<R::TableRows>(mean), R::ProjectKernelView<R::TableRows>(input),
            R::ProjectKernelView<R::TableRows>(boundary), R::ProjectKernelView<R::TableRows>(defect));
        CheckLaunch();
    }
    void GatherReconstructionAMGComponent(R::ExecutionContext context, R::ArrayView<const real> source,
                                          R::ArrayView<real> packed, index component, int modes)
    {
        R::ValidateViewDevices(context.device, source, packed);
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && modes > 0 && component >= 0 &&
                                  (component + 1) * modes <= source.Width() && packed.Width() == 1 &&
                                  packed.RowCount() == source.RowCount() * modes,
                              "VR AMG gather shape differs");
        Component<false><<<Blocks(packed.RowCount()), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(
            R::ProjectKernelView<R::TableRows>(source), {}, R::ProjectKernelView<R::TableRows>(packed), component, modes);
        CheckLaunch();
    }
    void ScatterReconstructionAMGComponent(R::ExecutionContext context, R::ArrayView<const real> input,
                                           R::ArrayView<const real> packed, R::ArrayView<real> output, index component, int modes)
    {
        R::ValidateViewDevices(context.device, input, packed, output);
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && modes > 0 && component >= 0 &&
                                  (component + 1) * modes <= input.Width() && packed.Width() == 1 &&
                                  packed.RowCount() == input.RowCount() * modes && output.RowCount() == input.RowCount() &&
                                  output.Width() == input.Width(),
                              "VR AMG scatter shape differs");
        Component<true><<<Blocks(packed.RowCount()), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(
            R::ProjectKernelView<R::TableRows>(input), R::ProjectKernelView<R::TableRows>(packed),
            R::ProjectKernelView<R::TableRows>(output), component, modes);
        CheckLaunch();
    }
}

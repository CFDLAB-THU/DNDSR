#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "CFV/Detail/LimitedVRKernels.hpp"
#include "CFV/Math/LimitedVRStencil.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    namespace
    {
        __global__ void Seeds(LimitedVRStencilView stencil, DeviceWorkSelection selection, LVRField scalars, LVRField gates,
                              LVRInput previous, LVROutput output, LimitedVRKernelParameters settings)
        {
            for (index iWork = index(blockIdx.x) * blockDim.x + threadIdx.x; iWork < selection.count; iWork += index(gridDim.x) * blockDim.x)
                Math::BuildLimitedVRSeedCell(stencil, scalars, gates, previous, output, settings, selection.Cell(iWork));
        }
        __global__ void Maximum(LimitedVRStencilView stencil, DeviceWorkSelection selection, LVRField input, LVROutput output,
                                std::array<bool, 3> active, bool excludeBoundary, bool saturatedOnly, std::array<real, 3> caps)
        {
            for (index iWork = index(blockIdx.x) * blockDim.x + threadIdx.x; iWork < selection.count; iWork += index(gridDim.x) * blockDim.x)
                Math::LimitedVRMaximumCell(stencil, input, output, selection.Cell(iWork), active, excludeBoundary, saturatedOnly, caps);
        }
        void ValidateOutput(LimitedVRStencilView stencil, LVROutput output)
        {
            DNDS_check_throw_info(stencil.offsets.RowCount() == output.RowCount() + 1 && stencil.boundary.RowCount() == output.RowCount() && output.Width() == 3,
                                  "LVR kernel stencil/output shape differs");
        }
        void ValidateInput(LVRField input, index rows, index width)
        {
            DNDS_check_throw_info(input.owned.RowCount() == rows && input.owned.Width() == width && (!input.ghost.RowCount() || input.ghost.Width() == width),
                                  "LVR kernel input shape differs");
        }
        unsigned Blocks(index rows) { return unsigned(std::min<index>((rows + 127) / 128, 1024)); }
    }
    void LaunchLimitedVRSeeds(R::ExecutionContext context, LimitedVRStencilView stencil, DeviceWorkSelection selection,
                              LVRField scalars, LVRField gates, LVRInput previous, LVROutput output, LimitedVRKernelParameters settings)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA LVR seeds require a CUDA context");
        R::ValidateViewDevices(context.device, stencil, selection, scalars, gates, previous, output);
        ValidateOutput(stencil, output);
        ValidateInput(scalars, output.RowCount(), 2);
        ValidateInput(gates, output.RowCount(), 3);
        DNDS_check_throw_info(previous.RowCount() == output.RowCount() && previous.Width() == 2, "LVR previous seed shape differs");
        R::CheckKernelArgumentBudget<LimitedVRStencilView, DeviceWorkSelection, LVRField, LVRField, LVRInput, LVROutput, LimitedVRKernelParameters>();
        if (!selection.count)
            return;
        Seeds<<<Blocks(selection.count), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(stencil, selection, scalars, gates, previous, output, settings);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA LVR seed launch failed");
    }
    void LaunchLimitedVRMaximum(R::ExecutionContext context, LimitedVRStencilView stencil, DeviceWorkSelection selection,
                                LVRField input, LVROutput output, std::array<bool, 3> active, bool excludeBoundary,
                                bool saturatedOnly, std::array<real, 3> caps)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA LVR maximum requires a CUDA context");
        R::ValidateViewDevices(context.device, stencil, selection, input, output);
        ValidateOutput(stencil, output);
        ValidateInput(input, output.RowCount(), 3);
        R::CheckKernelArgumentBudget<LimitedVRStencilView, DeviceWorkSelection, LVRField, LVROutput, std::array<bool, 3>, bool, bool, std::array<real, 3>>();
        if (!selection.count)
            return;
        Maximum<<<Blocks(selection.count), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(stencil, selection, input, output, active, excludeBoundary, saturatedOnly, caps);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA LVR maximum launch failed");
    }
}

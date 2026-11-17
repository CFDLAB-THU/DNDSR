#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/IO/CUDA/OutputFieldsKernels.hpp"
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <climits>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        using Input = R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>;
        using Output = R::ArrayView<real, R::AccessBoundsChecks, R::TableRows>;
        using Status = R::ArrayView<NumericalStatus, R::AccessBoundsChecks, R::FixedRows<1>>;
        __global__ void VelocityPressureRows(Input state, Output output, Status status)
        {
            for (index row = index(blockIdx.x) * blockDim.x + threadIdx.x; row < state.RowCount(); row += index(blockDim.x) * gridDim.x)
            {
                NumericalStatus ok;
                for (int j = 0; j < state.Width(); ++j)
                {
                    output(row, j) = state(row, j);
                    if (!std::isfinite(state(row, j)))
                        ok = NumericalStatus{NumericalError::InvalidState, j}.At(row);
                }
                status(row, 0) = ok;
            }
        }
        __global__ void OutputRows(PerfectGas gas, Input state, Output output, Status status)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < state.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                FlowState flow{};
                for (int j = 0; j < 5; ++j)
                    flow[j] = state(iRow, j);
                const auto properties = TryPerfectGasPoint(gas, flow, false);
                status(iRow, 0) = properties.status;
                if (!properties)
                {
                    status(iRow, 0) = properties.status.At(iRow);
                    continue;
                }
                const index components = state.Width();
                for (index j = 0; j < components; ++j)
                    output(iRow, j) = state(iRow, j);
                output(iRow, components) = properties.value.thermo.pressure;
                output(iRow, components + 1) = properties.value.thermo.temperature;
                real speedSquared = 0;
                for (int d = 0; d < 3; ++d)
                {
                    const real velocity = flow[d + 1] / flow[0];
                    output(iRow, components + 2 + d) = velocity;
                    speedSquared += velocity * velocity;
                }
                output(iRow, components + 5) = sqrt(speedSquared) / properties.value.thermo.soundSpeed;
            }
        }
    }
    void PrepareVelocityPressureOutput(R::ExecutionContext context, Input state, Output output, Status status)
    {
        R::ValidateViewDevices(context.device, state, output, status);
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && state.Width() >= 4 && state.Width() <= 6 &&
                                  output.Width() == state.Width() && output.RowCount() == state.RowCount() &&
                                  status.RowCount() == state.RowCount() && state.kind == R::RowLayoutKind::Table &&
                                  output.kind == R::RowLayoutKind::Table,
                              "invalid ACM output bindings");
        R::CheckKernelArgumentBudget<Input, Output, Status>();
        if (!state.RowCount())
            return;
        const auto blocks = unsigned(std::min<index>((state.RowCount() - 1) / 128 + 1, INT_MAX));
        VelocityPressureRows<<<blocks, 128, 0, R::CUDAStream(context)>>>(state, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "ACM output launch failed");
    }
    void PrepareOutputFields(R::ExecutionContext context, PerfectGas gas, Input state, Output output, Status status)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "output fields require CUDA placement");
        R::ValidateViewDevices(context.device, state, output, status);
        gas.Validate();
        DNDS_check_throw_info(state.Width() >= 5 && state.Width() <= 7 && output.Width() == state.Width() + 6 &&
                                  output.RowCount() == state.RowCount() && status.RowCount() == state.RowCount() &&
                                  state.kind == R::RowLayoutKind::Table && output.kind == R::RowLayoutKind::Table,
                              "CUDA output fields require combined AoS flow5/SA/2EQ");
        if (!state.RowCount())
            return;
        R::CheckKernelArgumentBudget<PerfectGas, Input, Output, Status>();
        const auto blocks = unsigned(std::min<index>((state.RowCount() - 1) / 128 + 1, INT_MAX));
        DNDS_check_throw_info(cudaSetDevice(context.device.ordinal) == cudaSuccess, "CUDA output device selection failed");
        OutputRows<<<blocks, 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(gas, state, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA output field launch failed");
    }
}

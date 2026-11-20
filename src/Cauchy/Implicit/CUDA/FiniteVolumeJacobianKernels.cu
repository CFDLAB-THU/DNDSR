#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Implicit/CUDA/FiniteVolumeJacobianKernels.hpp"
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <limits>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        template <int N>
        __global__ void Freeze(PerfectGas gas, VectorReadRows owned, VectorReadRows ghost,
                               R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows> means, LinearWriteRows pressure, LinearWriteRows derivative, JacobianStatus status)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iCell < means.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                FlowState flow{};
                NumericalStatus checked{};
                pressure(iCell, 0) = std::numeric_limits<real>::quiet_NaN();
                for (int j = 0; j < N; ++j)
                    derivative(iCell, j) = std::numeric_limits<real>::quiet_NaN();
                for (int j = 0; j < N; ++j)
                {
                    const real value = iCell < owned.RowCount() ? owned(iCell, j) : ghost(iCell - owned.RowCount(), j);
                    means(iCell, j) = value;
                    if (j < 5)
                        flow[j] = value;
                    if (!std::isfinite(value))
                        checked = NumericalStatus{NumericalError::InvalidState, 1, iCell};
                }
                if (checked)
                {
                    const auto point = TryPerfectGasPoint(gas, flow, false);
                    if (!point)
                        checked = point.status.At(iCell);
                    else
                    {
                        pressure(iCell, 0) = point.value.thermo.pressure;
                        const auto dp = gas.PressureDerivative(flow);
                        for (int j = 0; j < N; ++j)
                        {
                            derivative(iCell, j) = j < 5 ? dp[j] : 0;
                            if (!std::isfinite(derivative(iCell, j)))
                                checked = NumericalStatus{NumericalError::InvalidState, 2, iCell};
                        }
                    }
                }
                status(iCell, 0) = checked;
            }
        }
        __global__ void FreezeSignal(LinearReadRows owned, LinearReadRows ghost, LinearWriteRows output)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iCell < output.RowCount(); iCell += index(gridDim.x) * blockDim.x)
                output(iCell, 0) = iCell < owned.RowCount() ? owned(iCell, 0) : ghost(iCell - owned.RowCount(), 0);
        }
        __global__ void Initialize(JacobianCellMetrics cells, LinearReadRows inverseTau,
                                   MatrixWriteRows diagonal, JacobianStatus status, Runtime::RowSelection selected,
                                   RigidFrameParameters frame, real physicalMass, real scale, int components, bool fullBlock, bool rowMajorBlocks)
        {
            const auto rotation = frame.NegativeSourceJacobian();
            real rotationBound = 0;
            for (int d = 0; d < 3; ++d)
                rotationBound += frame.angularVelocity[d] * frame.angularVelocity[d];
            rotationBound = std::sqrt(rotationBound);
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected.RowAt(iRow);
                const real volume = cells(iCell, 0), tau = inverseTau(iCell, 0);
                if (!(std::isfinite(volume) && volume > 0 && std::isfinite(tau) && tau >= 0))
                {
                    for (int j = 0; j < (fullBlock ? components * components : components); ++j)
                        diagonal(iCell, j) = std::numeric_limits<real>::quiet_NaN();
                    status(iCell, 0) = NumericalStatus{NumericalError::InvalidState, 1, iCell};
                    continue;
                }
                NumericalStatus checked{};
                const real mass = volume * (tau + physicalMass);
                for (int j = 0; j < (fullBlock ? components * components : components); ++j)
                {
                    const int i = fullBlock ? (rowMajorBlocks ? j / components : j % components) : j, k = fullBlock ? (rowMajorBlocks ? j % components : j / components) : j;
                    real value = i == k ? mass : 0;
                    if (frame.moving && i < 5 && k < 5)
                        value += fullBlock ? scale * volume * rotation[i + 5 * k] : scale * volume * rotationBound;
                    diagonal(iCell, j) = value;
                    if (!std::isfinite(value))
                        checked = NumericalStatus{NumericalError::InvalidState, 2, iCell};
                }
                status(iCell, 0) = checked;
            }
        }
        template <class View>
        void CheckTable(View view, index rows, index width)
        {
            DNDS_check_throw_info((view.kind == R::RowLayoutKind::Table || view.kind == R::RowLayoutKind::ColMajorTable || view.kind == R::RowLayoutKind::TiledColMajorTable) && view.RowCount() == rows && view.Width() == width,
                                  "CUDA Jacobian preparation requires matching combined AoS tables");
        }
        template <class View>
        void CheckVector(View view, index rows, index width)
        {
            DNDS_check_throw_info(view.RowCount() == rows && view.Width() == width &&
                                      (view.kind == R::RowLayoutKind::Table ||
                                       (view.kind == R::RowLayoutKind::TiledColMajorTable && view.stride == 32)),
                                  "Jacobian mean requires matching AoS or tiled32 input");
        }
        unsigned Blocks(index rows) { return unsigned(std::min<index>(65535, (rows + 127) / 128)); }
    }
    void FreezeJacobianMeans(R::ExecutionContext context, const PerfectGas &gas,
                             VectorReadRows owned, VectorReadRows ghost, R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows> means,
                             LinearWriteRows pressure, LinearWriteRows derivative, JacobianStatus status)
    {
        const index n = owned.Width(), rows = owned.RowCount() + ghost.RowCount();
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && n >= 5 && n <= 7,
                              "CUDA Jacobian mean preparation requires a CUDA perfect-gas 5/6/7 profile");
        gas.Validate();
        R::ValidateViewDevices(context.device, owned, means, pressure, derivative, status);
        CheckVector(owned, owned.RowCount(), n);
        if (ghost.RowCount())
        {
            ghost.ValidateDevice(context.device);
            CheckVector(ghost, ghost.RowCount(), n);
        }
        CheckVector(means, rows, n);
        CheckTable(pressure, rows, 1);
        CheckTable(derivative, rows, n);
        DNDS_check_throw_info(status.RowCount() == rows, "CUDA Jacobian mean status shape differs");
        for (const auto *output : {means.data, pressure.data, derivative.data})
            DNDS_check_throw_info(!output || (output != owned.data && output != ghost.data),
                                  "CUDA Jacobian mean output aliases a live input");
        DNDS_check_throw_info(!means.data || (means.data != pressure.data && means.data != derivative.data && pressure.data != derivative.data),
                              "CUDA Jacobian mean outputs alias");
        R::CheckKernelArgumentBudget<PerfectGas, VectorReadRows, VectorReadRows, decltype(means), LinearWriteRows, LinearWriteRows, JacobianStatus>();
        if (!rows)
            return;
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (n == 5)
            Freeze<5><<<Blocks(rows), 128, 0, stream>>>(gas, owned, ghost, means, pressure, derivative, status);
        else if (n == 6)
            Freeze<6><<<Blocks(rows), 128, 0, stream>>>(gas, owned, ghost, means, pressure, derivative, status);
        else
            Freeze<7><<<Blocks(rows), 128, 0, stream>>>(gas, owned, ghost, means, pressure, derivative, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA Jacobian mean preparation launch failed");
    }
    void FreezeJacobianSignal(R::ExecutionContext context, LinearReadRows owned, LinearReadRows ghost, LinearWriteRows output)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA Jacobian signal requires CUDA");
        R::ValidateViewDevices(context.device, owned, ghost, output);
        CheckTable(owned, owned.RowCount(), 1);
        CheckTable(ghost, ghost.RowCount(), 1);
        CheckTable(output, owned.RowCount() + ghost.RowCount(), 1);
        DNDS_check_throw_info(!output.data || (output.data != owned.data && output.data != ghost.data),
                              "CUDA Jacobian signal output aliases its input");
        R::CheckKernelArgumentBudget<LinearReadRows, LinearReadRows, LinearWriteRows>();
        if (!output.RowCount())
            return;
        FreezeSignal<<<Blocks(output.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(owned, ghost, output);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA Jacobian signal copy launch failed");
    }
    void InitializePerfectGasFluxJacobianDiagonal(R::ExecutionContext context, JacobianCellMetrics cells,
                                                  LinearReadRows inverseTau, MatrixWriteRows diagonal, JacobianStatus status,
                                                  Runtime::RowSelection selected, const RigidFrameParameters &frame,
                                                  real physicalMass, real scale, int components, bool fullBlock, bool rowMajorBlocks)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && components >= 5 && components <= 7 &&
                                  std::isfinite(physicalMass) && physicalMass >= 0 && std::isfinite(scale) && scale >= 0,
                              "unsupported CUDA Jacobian diagonal request");
        R::ValidateViewDevices(context.device, cells, inverseTau, diagonal, status, selected);
        CheckTable(inverseTau, diagonal.RowCount(), 1);
        CheckTable(diagonal, diagonal.RowCount(), fullBlock ? components * components : components);
        DNDS_check_throw_info(cells.RowCount() >= diagonal.RowCount() && status.RowCount() == diagonal.RowCount() &&
                                  (!diagonal.data || (diagonal.data != inverseTau.data && diagonal.data != cells.data)),
                              "CUDA Jacobian diagonal shape or alias differs");
        R::CheckKernelArgumentBudget<JacobianCellMetrics, LinearReadRows, MatrixWriteRows, JacobianStatus, Runtime::RowSelection,
                                     RigidFrameParameters, real, real, int, bool, bool>();
        if (!selected.RowCount())
            return;
        Initialize<<<Blocks(selected.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(
            cells, inverseTau, diagonal, status, selected, frame, physicalMass, scale, components, fullBlock, rowMajorBlocks);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA Jacobian diagonal initialization launch failed");
    }
}

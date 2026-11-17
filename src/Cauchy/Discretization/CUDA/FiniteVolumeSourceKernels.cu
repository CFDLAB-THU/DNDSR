#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        template <int Components>
        __global__ void Sources(SourceRequest request, SourceIndices selected)
        {
            const auto &physics = request.parameters(0, 0);
            const bool mean = request.evaluation == SourceEvaluation::MeanDestruction;
            const bool jacobian = request.evaluation != SourceEvaluation::QuadratureResidual;
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                NumericalStatus status;
                if (physics.dimension != request.dimension || physics.rans.ComponentNumber() + 5 != Components || !physics.rans.Supported())
                {
                    request.status(iCell, 0) = NumericalStatus{NumericalError::InvalidInput}.At(iCell);
                    continue;
                }
                std::array<real, Components - 5> source{}, ransMean{};
                for (int j = 0; j < Components - 5; ++j)
                    ransMean[j] = request.means(iCell, 5 + j);
                const index begin = mean ? iCell : request.offsets(iCell, 0);
                const index end = mean ? iCell + 1 : request.offsets(iCell + 1, 0);
                for (index iPoint = begin; iPoint < end; ++iPoint)
                {
                    PerfectGasKernelPoint<Components> point;
                    for (int j = 0; j < Components; ++j)
                    {
                        point.values[j] = mean ? request.means(iCell, j) : request.samples(iPoint, j);
                        for (int d = 0; d < request.dimension; ++d)
                            point.gradients[j][d] = request.gradients[d](iPoint, j);
                    }
                    if (!mean)
                        RANSReconstructionView{physics.rans.kind, Components - 5}.RepairPoint(ransMean, SpanView<real>{point.values.data() + 5, Components - 5});
                    const auto properties = TryPerfectGasPoint(physics.gas, point.Flow(), true);
                    if (!properties)
                    {
                        status = properties.status.At(iCell, iPoint);
                        break;
                    }
                    const real wall = mean ? request.wallMean(iCell, 0) : request.wallPoints(iCell, iPoint - begin);
                    const auto value = Physics::RANS::TrySource(physics.rans, point.RANSPoint(properties.value, wall, request.lengths(iCell, 0)), jacobian ? 1 : 0);
                    if (!value)
                    {
                        status = value.status.At(iCell, iPoint);
                        break;
                    }
                    const real weight = mean ? request.cells(iCell, 0) : request.weights(iPoint, 0);
                    for (int j = 0; j < Components - 5; ++j)
                        source[j] += weight * value.value[j];
                }
                request.status(iCell, 0) = status;
                if (status)
                    for (int j = 0; j < Components - 5; ++j)
                    {
                        const int row = 5 + j;
                        request.output(iCell, jacobian && request.fullBlock ? row + Components * row : row) +=
                            (jacobian ? request.scale : real(1)) * source[j];
                    }
            }
        }
        __global__ void Replicate(SourceIndices offsets, SourceInput input, SourceOutput output, int stride, int direction)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell + 1 < offsets.RowCount(); iCell += index(gridDim.x) * blockDim.x)
                for (index iPoint = offsets(iCell, 0); iPoint < offsets(iCell + 1, 0); ++iPoint)
                    for (index j = 0; j < output.Width(); ++j)
                        output(iPoint, j) = input(iCell, j * stride + direction);
        }
        __global__ void Rotation(R::ArrayView<const real, false, R::FixedRows<8>> cells,
                                 RigidFrameParameters frame, SourceInput means, SourceOutput output,
                                 SourceOutput low, bool produceLow, SourceIndices selected)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                FlowState mean;
                for (int j = 0; j < 5; ++j)
                    mean[j] = means(iCell, j);
                const auto source = frame.RotationSource(mean);
                for (int j = 0; j < 5; ++j)
                {
                    output(iCell, j) += cells(iCell, 0) * source[j];
                    if (produceLow)
                        low(iCell, j) += cells(iCell, 0) * source[j];
                }
            }
        }
        unsigned Blocks(index rows) { return unsigned(std::min<index>((rows + 127) / 128, 1024)); }
        void CheckLaunch() { DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA volume source launch failed"); }
        template <class View>
        void CheckTable(View view, index rows, index width)
        {
            DNDS_check_throw_info((view.kind == R::RowLayoutKind::Table || view.kind == R::RowLayoutKind::ColMajorTable || view.kind == R::RowLayoutKind::TiledColMajorTable) && view.RowCount() == rows && view.Width() == width,
                                  "CUDA source requires a correctly shaped combined AoS table");
        }
    }
    void AddRANSSources(R::ExecutionContext context, SourceRequest request, SourceIndices selected)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA source requires a CUDA context");
        R::ValidateViewDevices(context.device, request.parameters, request.means, request.lengths, request.output, request.status, selected);
        const bool mean = request.evaluation == SourceEvaluation::MeanDestruction;
        const bool jacobian = request.evaluation != SourceEvaluation::QuadratureResidual;
        DNDS_check_throw_info((request.components == 6 || request.components == 7) && (request.dimension == 2 || request.dimension == 3) &&
                                  request.evaluation >= SourceEvaluation::QuadratureResidual && request.evaluation <= SourceEvaluation::QuadratureDestruction &&
                                  std::isfinite(request.scale) && request.parameters.RowCount() == 1,
                              "unsupported CUDA RANS source shape/policy");
        const index cells = request.means.RowCount();
        CheckTable(request.means, cells, request.components);
        CheckTable(request.output, cells, jacobian && request.fullBlock ? request.components * request.components : request.components);
        // Mesh geometry is a whole owned+ghost publication. Numerical means,
        // status and outputs are owned-only here; do not slice the geometry
        // descriptors or require an otherwise redundant owned-only copy.
        DNDS_check_throw_info(request.lengths.RowCount() >= cells && request.status.RowCount() == cells, "CUDA source cell fields differ");
        if (mean)
        {
            R::ValidateViewDevices(context.device, request.wallMean, request.cells);
            DNDS_check_throw_info(request.wallMean.RowCount() >= cells && request.cells.RowCount() >= cells, "CUDA mean source geometry differs");
        }
        else
        {
            R::ValidateViewDevices(context.device, request.samples, request.offsets, request.weights, request.wallPoints);
            CheckTable(request.samples, request.weights.RowCount(), request.components);
            DNDS_check_throw_info(request.offsets.RowCount() == cells + 1 && request.wallPoints.RowCount() >= cells, "CUDA quadrature source geometry differs");
        }
        for (int d = 0; d < request.dimension; ++d)
        {
            const auto gradient = request.gradients[d];
            gradient.values.ValidateDevice(context.device);
            DNDS_check_throw_info(gradient.componentStride > 0 && gradient.direction >= 0 && gradient.direction < gradient.componentStride &&
                                      gradient.values.kind == R::RowLayoutKind::Table && gradient.values.RowCount() == (mean ? cells : request.samples.RowCount()) &&
                                      gradient.values.Width() > (request.components - 1) * gradient.componentStride + gradient.direction,
                                  "CUDA source derivative field is incomplete");
        }
        R::CheckKernelArgumentBudget<SourceRequest, SourceIndices>();
        if (!selected.RowCount())
            return;
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (request.components == 6)
            Sources<6><<<Blocks(selected.RowCount()), 128, 0, stream>>>(request, selected);
        else
            Sources<7><<<Blocks(selected.RowCount()), 128, 0, stream>>>(request, selected);
        CheckLaunch();
    }
    void ReplicateSourceValues(R::ExecutionContext context, SourceIndices offsets, SourceInput input, SourceOutput output, int stride, int direction)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA source replication requires a CUDA context");
        R::ValidateViewDevices(context.device, offsets, input, output);
        DNDS_check_throw_info(stride > 0 && direction >= 0 && direction < stride && offsets.RowCount() == input.RowCount() + 1 &&
                                  input.kind == R::RowLayoutKind::Table && output.kind == R::RowLayoutKind::Table &&
                                  (!output.Width() || input.Width() > (output.Width() - 1) * stride + direction),
                              "CUDA source replication shape/direction differs");
        R::CheckKernelArgumentBudget<SourceIndices, SourceInput, SourceOutput, int, int>();
        if (!input.RowCount())
            return;
        Replicate<<<Blocks(input.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(offsets, input, output, stride, direction);
        CheckLaunch();
    }
    void AddRotationSource(R::ExecutionContext context, R::ArrayView<const real, false, R::FixedRows<8>> cells,
                           RigidFrameParameters frame, SourceInput means, SourceOutput output, SourceOutput low, bool produceLow, SourceIndices selected)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA rotation source requires a CUDA context");
        R::ValidateViewDevices(context.device, cells, means, output, selected);
        DNDS_check_throw_info(means.Width() >= 5 && means.Width() <= 7 && cells.RowCount() >= means.RowCount(), "CUDA rotation source shape differs");
        CheckTable(means, means.RowCount(), means.Width());
        CheckTable(output, means.RowCount(), means.Width());
        if (produceLow)
        {
            low.ValidateDevice(context.device);
            // The combined equation buffer may include RANS scalars. Rotation
            // contributes only to flow5 and leaves those scalar entries intact.
            CheckTable(low, means.RowCount(), means.Width());
        }
        R::CheckKernelArgumentBudget<decltype(cells), RigidFrameParameters, SourceInput, SourceOutput, SourceOutput, bool, SourceIndices>();
        if (!selected.RowCount() || !frame.moving)
            return;
        Rotation<<<Blocks(selected.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(cells, frame, means, output, low, produceLow, selected);
        CheckLaunch();
    }
}

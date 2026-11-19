#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels_Launch.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include "DNDS/Algebra/BlockIndex.hpp"

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        __global__ void ApplyDerivative(SourceInput source, decltype(SourceRequest::output) diagonal, index n, bool full, bool rowMajor, real scale)
        {
            const index width = full ? n * n : n;
            for (index i = index(blockIdx.x) * blockDim.x + threadIdx.x; i < source.RowCount() * width; i += index(gridDim.x) * blockDim.x)
            {
                const index cell = i / width, k = i % width;
                const index row = full ? k / n : k, col = full ? k % n : k;
                diagonal(cell, full ? Algebra::DenseBlockEntry(row, col, n, rowMajor) : row) += scale * source(cell, row * n + col);
            }
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
        template <Detail::VolumeSourceModel Model, int C, int D>
        void LaunchCells(cudaStream_t stream, SourceRequest request, SourceIndices rows)
        {
            const unsigned blocks = Blocks(rows.RowCount());
            const int modes = std::max(0, request.basis.polynomial.size - 1);
            if (modes == 0)
                Detail::LaunchVolumeSourceCells<Model, C, D, 0>(stream, request, rows, blocks);
            else if (modes == D)
                Detail::LaunchVolumeSourceCells<Model, C, D, D>(stream, request, rows, blocks);
            else if (modes == (D == 2 ? 5 : 9))
                Detail::LaunchVolumeSourceCells<Model, C, D, D == 2 ? 5 : 9>(stream, request, rows, blocks);
            else if (modes == (D == 2 ? 9 : 19))
                Detail::LaunchVolumeSourceCells<Model, C, D, D == 2 ? 9 : 19>(stream, request, rows, blocks);
            else
                DNDS_check_throw_info(false, "unsupported modal source dimension or order");
        }
        template <Detail::VolumeSourceModel Model, int C>
        void LaunchCells(cudaStream_t stream, SourceRequest request, SourceIndices rows)
        {
            if (request.dimension == 2)
                LaunchCells<Model, C, 2>(stream, request, rows);
            else
                LaunchCells<Model, C, 3>(stream, request, rows);
        }
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
        const bool mechanical = request.acm.RowCount() != 0;
        if (mechanical)
            request.acm.ValidateDevice(context.device);
        else
            request.parameters.ValidateDevice(context.device);
        const int flow = mechanical ? 4 : 5;
        R::ValidateViewDevices(context.device, request.means, request.lengths, request.output, request.status, selected);
        const bool mean = request.evaluation == SourceEvaluation::MeanDestruction;
        const bool jacobian = request.evaluation != SourceEvaluation::QuadratureResidual;
        DNDS_check_throw_info((request.components == flow + 1 || request.components == flow + 2) && (request.dimension == 2 || request.dimension == 3) &&
                                  request.evaluation >= SourceEvaluation::QuadratureResidual && request.evaluation <= SourceEvaluation::QuadratureDestruction &&
                                  std::isfinite(request.scale) && (mechanical ? request.acm.RowCount() : request.parameters.RowCount()) == 1,
                              "unsupported CUDA RANS source shape/policy");
        const index cells = request.means.RowCount();
        if (request.preparedDerivative.data)
        {
            request.preparedDerivative.ValidateDevice(context.device);
            DNDS_check_throw_info(!mechanical && !jacobian && request.preparedDerivative.RowCount() == cells &&
                                      (request.preparedDerivative.Width() == request.components ||
                                       request.preparedDerivative.Width() == request.components * request.components),
                                  "invalid integrated RANS source derivative destination");
        }
        CheckTable(request.means, cells, request.components);
        DNDS_check_throw_info(request.means.kind == R::RowLayoutKind::Table && request.means.stride == request.components,
                              "CUDA volume source requires compact combined AoS means");
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
            R::ValidateViewDevices(context.device, request.offsets, request.weights, request.wallPoints);
            if (request.basis.polynomial.size > 1)
            {
                const int modes = request.basis.polynomial.size - 1;
                DNDS_check_throw_info(modes <= 19, "source modal order exceeds P3");
                R::ValidateViewDevices(context.device, request.coefficients, request.basis.points, request.basis.cached, request.basis.moments, request.basis.geometry);
                CheckTable(request.coefficients, cells, request.components * modes);
                DNDS_check_throw_info(request.coefficients.kind == R::RowLayoutKind::Table && request.coefficients.stride == request.coefficients.Width(),
                                      "CUDA modal source requires compact combined AoS coefficients");
            }
            else if (!request.cellInputs)
            {
                request.samples.ValidateDevice(context.device);
                CheckTable(request.samples, request.weights.RowCount(), request.components);
            }
            DNDS_check_throw_info(request.offsets.RowCount() == cells + 1 && request.wallPoints.RowCount() >= cells, "CUDA quadrature source geometry differs");
        }
        if (request.ransP1Overlay)
        {
            DNDS_check_throw_info(!mean && request.basis.polynomial.size > 1, "RANS P1 source overlay requires volume quadrature");
            R::ValidateViewDevices(context.device, request.ransP1Gradients, request.cells);
            DNDS_check_throw_info(request.cells.RowCount() >= cells, "RANS P1 source overlay needs cell scaling");
            CheckTable(request.ransP1Gradients, cells, request.components * request.dimension);
        }
        for (int d = 0; request.basis.polynomial.size <= 1 && d < request.dimension; ++d)
        {
            const auto gradient = request.gradients[d];
            gradient.values.ValidateDevice(context.device);
            DNDS_check_throw_info(gradient.componentStride > 0 && gradient.direction >= 0 && gradient.direction < gradient.componentStride &&
                                      gradient.values.kind == R::RowLayoutKind::Table && gradient.values.RowCount() == ((mean || request.cellInputs) ? cells : request.samples.RowCount()) &&
                                      gradient.values.Width() > (request.components - 1) * gradient.componentStride + gradient.direction,
                                  "CUDA source derivative field is incomplete");
        }
        R::CheckKernelArgumentBudget<SourceRequest, SourceIndices>();
        if (!selected.RowCount())
            return;
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (mechanical && request.components == 5)
            LaunchCells<Detail::VolumeSourceModel::ACM, 5>(stream, request, selected);
        else if (mechanical)
            LaunchCells<Detail::VolumeSourceModel::ACM, 6>(stream, request, selected);
        else if (request.components == 6)
            LaunchCells<Detail::VolumeSourceModel::PerfectGas, 6>(stream, request, selected);
        else
            LaunchCells<Detail::VolumeSourceModel::PerfectGas, 7>(stream, request, selected);
        CheckLaunch();
    }
    void ApplyVolumeSourceDerivative(R::ExecutionContext context, SourceInput source, decltype(SourceRequest::output) diagonal,
                                     index n, bool full, bool rowMajor, real scale)
    {
        R::ValidateViewDevices(context.device, source, diagonal);
        DNDS_check_throw_info(source.RowCount() == diagonal.RowCount() && source.Width() == n * n && diagonal.Width() == (full ? n * n : n) && (!source.data || source.data != diagonal.data),
                              "invalid source derivative block placement");
        if (!source.RowCount())
            return;
        ApplyDerivative<<<Blocks(source.RowCount() * diagonal.Width()), 128, 0, R::CUDAStream(context)>>>(source, diagonal, n, full, rowMajor, scale);
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

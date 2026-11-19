#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels_Launch.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    template <int Width>
    DNDS_DEVICE_CALLABLE auto CompactSourceRows(SourceInput input)
    {
        return Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<Width, Width>>{
            input.data, nullptr, input.rows, Width, Width, nullptr, Runtime::RowLayoutKind::Table, input.device};
    }
        /// Each cell thread integrates its own volume quadrature in stored order.
        /// TODO: compare CTA quadrature-point lanes and shared-memory reduction.
        template <VolumeSourceModel Model, int Components, int D, int Modes>
        __global__ void EvaluateVolumeSourceCells(SourceRequest request, SourceIndices selected)
        {
            constexpr bool Mechanical = Model == VolumeSourceModel::ACM;
            constexpr int flow = Mechanical ? 4 : 5;
            const auto means = CompactSourceRows<Components>(request.means);
            const auto &physics = [&]() -> const auto &
            {
                if constexpr (Mechanical)
                    return request.acm(0, 0);
                else
                    return request.parameters(0, 0);
            }();
            const bool mean = request.evaluation == SourceEvaluation::MeanDestruction;
            const bool jacobian = request.evaluation != SourceEvaluation::QuadratureResidual;
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                NumericalStatus status;
                if (physics.rans.ComponentNumber() + flow != Components || !physics.rans.Supported())
                {
                    request.status(iCell, 0) = NumericalStatus{NumericalError::InvalidInput}.At(iCell);
                    continue;
                }
                std::array<real, Components - flow> source{}, ransMean{}, destruction{};
                for (int j = 0; j < Components - flow; ++j)
                    ransMean[j] = means(iCell, flow + j);
                const index begin = mean ? iCell : request.offsets(iCell, 0);
                const index end = mean ? iCell + 1 : request.offsets(iCell + 1, 0);
                for (index iPoint = begin; iPoint < end; ++iPoint)
                {
                    std::conditional_t<Mechanical, ACM::KernelPoint<Components>, PerfectGasKernelPoint<Components>> point;
                    constexpr int modes = Modes;
                    real basis[4][19];
                    if constexpr (modes > 0)
                        request.basis.Fill(iCell, iPoint, 1 + D, basis);
                    for (int j = 0; j < Components; ++j)
                    {
                        if constexpr (modes > 0)
                        {
                            const auto coeff = CompactSourceRows<Components * Modes>(request.coefficients)[iCell];
                            point.values[j] = means(iCell, j);
                            for (int m = 0; m < modes; ++m)
                                point.values[j] += basis[0][m] * coeff[j * modes + m];
                            for (int d = 0; d < D; ++d)
                            {
                                real derivative = 0;
                                for (int m = 0; m < modes; ++m)
                                    derivative += basis[1 + d][m] * coeff[j * modes + m];
                                point.gradients[j][d] = derivative;
                            }
                            continue;
                        }
                        point.values[j] = (mean || request.cellInputs) ? means(iCell, j) : request.samples(iPoint, j);
                        for (int d = 0; d < D; ++d)
                            point.gradients[j][d] = request.gradients[d](request.cellInputs ? iCell : iPoint, j);
                    }
                    if (!mean)
                    {
                        const RANSReconstructionView rans{physics.rans.kind, Components - flow};
                        const SpanView<real> values{point.values.data() + flow, Components - flow};
                        if constexpr (Modes > 0)
                            if (request.ransP1Overlay)
                            {
                                std::array<real, D> dx;
                                for (int d = 0; d < D; ++d)
                                    dx[d] = request.cells(iCell, 7) * basis[0][d];
                                rans.OverlayP1(D, ransMean, dx,
                                               [&](int j, int d) { return request.ransP1Gradients(iCell, (flow + j) * D + d); },
                                               [&](int j, int d, real value) { point.gradients[flow + j][d] = value; }, values);
                            }
                        rans.RepairPoint(ransMean, values);
                    }
                    const real wall = mean ? request.wallMean(iCell, 0) : request.wallPoints(iCell, iPoint - begin);
                    const real weight = mean ? request.cells(iCell, 0) : request.weights(iPoint, 0);
                    const auto evaluate = [&](const auto &input) -> NumericalResult<std::array<real, 2>>
                    {
                        if (request.preparedDerivative.data)
                        {
                            const auto derivative = Physics::RANS::TrySource(physics.rans, input, 1);
                            if (!derivative)
                                return derivative.status;
                            for (int j = 0; j < Components - flow; ++j)
                                destruction[j] += weight * derivative.value[j];
                        }
                        return Physics::RANS::TrySource(physics.rans, input, jacobian ? 1 : 0);
                    };
                    const auto value = [&]() -> NumericalResult<std::array<real, 2>>
                    {
                        if constexpr (Mechanical)
                            return evaluate(point.Turbulence(physics, wall, request.lengths(iCell, 0)));
                        else
                        {
                            if (physics.dimension != D)
                                return NumericalStatus{NumericalError::InvalidInput};
                            const auto properties = TryPerfectGasPoint(physics.gas, point.Flow(), true);
                            if (!properties)
                                return properties.status;
                            return evaluate(point.RANSPoint(properties.value, wall, request.lengths(iCell, 0)));
                        }
                    }();
                    if (!value)
                    {
                        status = value.status.At(iCell, iPoint);
                        break;
                    }
                    for (int j = 0; j < Components - flow; ++j)
                        source[j] += weight * value.value[j];
                }
                request.status(iCell, 0) = status;
                if (status && request.preparedDerivative.data)
                {
                    const bool dense = request.preparedDerivative.Width() == Components * Components;
                    for (index j = 0; j < request.preparedDerivative.Width(); ++j)
                        request.preparedDerivative(iCell, j) = 0;
                    for (int j = 0; j < Components - flow; ++j)
                    {
                        const int row = flow + j;
                        request.preparedDerivative(iCell, dense ? row * Components + row : row) = destruction[j];
                    }
                }
                if (status)
                    for (int j = 0; j < Components - flow; ++j)
                    {
                        const int row = flow + j;
                        auto &entry = request.output(iCell, jacobian && request.fullBlock ? row + Components * row : row);
                        const real value = (jacobian ? request.scale : real(1)) * source[j];
                        if (Mechanical && mean)
                            entry = value;
                        else
                            entry += value;
                    }
            }
        }

    template <VolumeSourceModel Model, int C, int D, int M>
    void LaunchVolumeSourceCells(cudaStream_t stream, SourceRequest request, SourceIndices rows, unsigned blocks)
    {
        EvaluateVolumeSourceCells<Model,C,D,M><<<blocks,128,0,stream>>>(request,rows);
    }
}

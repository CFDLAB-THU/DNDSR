#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeGradientKernels.hpp"
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "Cauchy/Discretization/CUDA/PositivityReduction.cuh"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        template <int Components>
        __global__ void Barth(GradientLimiterGeometry geometry, GradientMeans means, GradientInput source,
                              GradientOutput destination, GradientIndices selected, int dimension, bool barth)
        {
            const auto &stencil = geometry.stencil(0, 0);
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                std::array<real, Components> lower{}, upper{}, positive{}, negative{};
                for (int j = 0; j < Components; ++j)
                    lower[j] = upper[j] = means(iCell, j);
                if (barth)
                    for (rowsize slot = 0; slot < rowsize(stencil.topology.CellFaces(iCell).size()); ++slot)
                    {
                        const auto sample = stencil.GradientSample(iCell, slot);
                        if (sample.other != UnInitIndex)
                        {
                            std::array<real, Components> neighbor;
                            for (int j = 0; j < Components; ++j)
                                neighbor[j] = means(sample.other, j);
                            if (sample.changesFrame)
                            {
                                const auto old = neighbor;
                                for (int d = 0; d < 3; ++d)
                                {
                                    neighbor[1 + d] = 0;
                                    for (int k = 0; k < 3; ++k)
                                        neighbor[1 + d] += sample.rotation[d + 3 * k] * old[1 + k];
                                }
                            }
                            for (int j = 0; j < Components; ++j)
                            {
                                lower[j] = std::min(lower[j], neighbor[j]);
                                upper[j] = std::max(upper[j], neighbor[j]);
                            }
                        }
                        const auto displacement = stencil.GradientDisplacement(iCell, slot);
                        for (int j = 0; j < Components; ++j)
                        {
                            real delta = 0;
                            for (int d = 0; d < dimension; ++d)
                                delta += source(iCell, j * dimension + d) * displacement[d];
                            positive[j] = std::max(positive[j], delta);
                            negative[j] = std::min(negative[j], delta);
                        }
                    }
                for (int j = 0; j < Components; ++j)
                {
                    real alpha = 1;
                    if (barth)
                    {
                        alpha = std::min(alpha, (upper[j] - means(iCell, j)) / (positive[j] + verySmallReal));
                        alpha = std::min(alpha, (means(iCell, j) - lower[j]) / (-negative[j] + verySmallReal));
                        alpha = std::max(real(0), std::min(real(1), alpha));
                    }
                    for (int d = 0; d < dimension; ++d)
                        destination(iCell, j * dimension + d) = alpha * source(iCell, j * dimension + d);
                }
            }
        }
        template <int Components>
        struct PointValues
        {
            std::array<real, Components> mean;
            std::array<Vector3, Components> gradients{};
            Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
            index cell = 0;
            mutable index cachedPoint = -1;
            mutable Vector3 displacement{};
            GradientOutput stored;
            bool readStored = false;
            int dimension;
            DNDS_DEVICE_CALLABLE real operator()(index q, index j) const
            {
                real value = mean[j];
                if (cachedPoint != q)
                {
                    const auto point = stencil(0, 0).GradientDisplacement(cell, rowsize(q));
                    for (int d = 0; d < 3; ++d)
                        displacement[d] = point[d];
                    cachedPoint = q;
                }
                for (int d = 0; d < dimension; ++d)
                    value += (readStored ? stored(cell, j * dimension + d) : gradients[j][d]) * displacement[d];
                return value;
            }
        };
        template <int Components>
        struct MeanValues
        {
            const std::array<real, Components> &values;
            DNDS_DEVICE_CALLABLE real operator()(index, index j) const { return values[j]; }
        };
        template <int Components>
        DNDS_DEVICE_CALLABLE NumericalStatus PositiveCell(GradientLimiterGeometry geometry, GradientInput means,
                                                          GradientOutput gradients, index iCell, GradientPositivityParameters p, real &outputBeta) noexcept
        {
            PointValues<Components> points;
            points.stencil = geometry.stencil;
            points.cell = iCell;
            points.dimension = p.dimension;
            for (int j = 0; j < Components; ++j)
            {
                points.mean[j] = means(iCell, j);
                for (int d = 0; d < p.dimension; ++d)
                    points.gradients[j][d] = gradients(iCell, j * p.dimension + d);
            }
            if (!p.bounds.Admissible(points.mean))
                return NumericalStatus{NumericalError::InvalidState, 1}.At(iCell);
            const MeanValues<Components> base{points.mean};
            const index begin = 0, end = index(geometry.stencil(0, 0).topology.CellFaces(iCell).size());
            real beta = p.bounds.ReconstructionFraction(points.mean, base, points, begin, end, p.betaPower, p.betaCutoff, 1, p.clippedPolicy);
            if (!(std::isfinite(beta) && beta >= 0 && beta <= 1))
                return NumericalStatus{NumericalError::InvalidState, 2}.At(iCell);
            const real ransScale = p.rans.Compute(base, points, 5, begin, end);
            if (!(std::isfinite(ransScale) && ransScale >= 0 && ransScale <= 1))
                return NumericalStatus{NumericalError::InvalidState, 3}.At(iCell);
            for (int j = 0; j < Components; ++j)
                for (int d = 0; d < p.dimension; ++d)
                    points.gradients[j][d] *= j < 5 ? beta : ransScale;
            points.stored = gradients;
            const auto publish = [&]()
            {
                for (int j = 0; j < Components; ++j)
                    for (int d = 0; d < p.dimension; ++d)
                        gradients(iCell, j * p.dimension + d) = points.gradients[j][d];
            };
            publish();
            points.readStored = true;
            const auto accepted = [&]()
            {
                bool good = true;
                for (index q = begin; q < end; ++q)
                    good = good && p.bounds.ClippedReferenceAdmissible(ThermalAdmissibilityDetail::PointRow<decltype(points)>{points, q});
                return good;
            };
            // PP POLICY -- retry the complete flow gradient, never individual
            // points/components. The actual endpoint predicate is unchanged.
            for (int iteration = 0; iteration < 1000 && !accepted(); ++iteration)
            {
                const real previous = beta;
                beta = p.clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase ? 0 : beta * .9;
                if (beta < p.betaCutoff)
                    beta = 0;
                if (!(previous > 0))
                    return NumericalStatus{NumericalError::InvalidState, 4}.At(iCell);
                for (int j = 0; j < 5; ++j)
                    for (int d = 0; d < p.dimension; ++d)
                        points.gradients[j][d] *= beta / previous;
                publish();
            }
            if (!accepted())
                return NumericalStatus{NumericalError::InvalidState, 5}.At(iCell);
            outputBeta = beta;
            return {};
        }
        template <int Components>
        __global__ void Positive(GradientLimiterGeometry geometry, GradientInput means, GradientOutput gradients,
                                 GradientIndices selected, GradientPositivityParameters p, ModalPositivitySummary partials)
        {
            NumericalStatus failure;
            index limited = 0;
            real minimum = 1;
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                real beta = 1;
                const auto status = PositiveCell<Components>(geometry, means, gradients, iCell, p, beta);
                failure = PositivityReductionDetail::FirstFailure{}(failure, status);
                if (status && beta < 1)
                {
                    ++limited;
                    minimum = std::min(minimum, beta);
                }
            }
            PositivityReductionDetail::PublishPositivitySummary<128>({failure, limited, minimum}, partials);
        }
        __global__ void Primitive(GradientInput input, GradientOutput output, PerfectGas gas, GradientStatus status)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < output.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                FlowState state;
                for (int j = 0; j < 5; ++j)
                    state[j] = input(iCell, j);
                const auto properties = TryPerfectGasPoint(gas, state, false);
                status(iCell, 0) = properties ? NumericalStatus{} : properties.status.At(iCell);
                if (properties)
                {
                    output(iCell, 0) = state[0];
                    for (int d = 0; d < 3; ++d)
                        output(iCell, 1 + d) = state[1 + d] / state[0];
                    output(iCell, 4) = properties.value.thermo.temperature;
                }
            }
        }
        unsigned Blocks(index count) { return unsigned(std::min<index>((count + 127) / 128, 1024)); }
        void CheckLaunch() { DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA Cauchy gradient launch failed"); }
        template <class View>
        void CheckTable(View view, index rows, index width)
        {
            DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table && view.RowCount() == rows && view.Width() == width, "CUDA gradient table shape differs");
        }
    }
    void LimitPhysicalGradients(R::ExecutionContext context, GradientLimiterGeometry geometry, GradientMeans means,
                                GradientInput source, GradientOutput destination, GradientIndices selected, int dimension, bool barth, bool requireGhost)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && (dimension == 2 || dimension == 3), "CUDA gradient context/dimension differs");
        R::ValidateViewDevices(context.device, means, source, destination, selected);
        const index owned = means.owned.RowCount(), components = means.owned.Width();
        DNDS_check_throw_info(components >= 5 && components <= 7, "CUDA gradients require flow5/SA/2EQ");
        CheckTable(means.owned, owned, components);
        CheckTable(source, owned, components * dimension);
        CheckTable(destination, owned, components * dimension);
        if (barth)
        {
            R::ValidateViewDevices(context.device, geometry.stencil);
            DNDS_check_throw_info(geometry.stencil.RowCount() == 1 && geometry.ownedCells == owned && geometry.allCells >= owned, "CUDA Barth geometry differs");
            if (requireGhost || means.ghost.RowCount())
                CheckTable(means.ghost, geometry.allCells - owned, components);
        }
        R::CheckKernelArgumentBudget<GradientLimiterGeometry, GradientMeans, GradientInput, GradientOutput, GradientIndices, int, bool>();
        if (!selected.RowCount())
            return;
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        switch (components)
        {
        case 5:
            Barth<5><<<Blocks(selected.RowCount()), 128, 0, stream>>>(geometry, means, source, destination, selected, dimension, barth);
            break;
        case 6:
            Barth<6><<<Blocks(selected.RowCount()), 128, 0, stream>>>(geometry, means, source, destination, selected, dimension, barth);
            break;
        case 7:
            Barth<7><<<Blocks(selected.RowCount()), 128, 0, stream>>>(geometry, means, source, destination, selected, dimension, barth);
            break;
        }
        CheckLaunch();
    }
    void LimitPositivePhysicalGradients(R::ExecutionContext context, GradientLimiterGeometry geometry, GradientInput means,
                                        GradientOutput gradients, GradientIndices selected, GradientPositivityParameters p,
                                        ModalPositivitySummary partials)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && (p.dimension == 2 || p.dimension == 3), "CUDA gradient PP context/dimension differs");
        R::ValidateViewDevices(context.device, means, gradients, selected, partials, geometry.stencil);
        const index owned = means.RowCount(), components = means.Width();
        DNDS_check_throw_info(components >= 5 && components <= 7 && p.bounds.components == components && p.bounds.flowFirst == 0 && p.bounds.independent == 0 &&
                                  p.bounds.speciesEnergies[0].empty() && p.bounds.speciesEnergies[1].empty() && p.rans.components + 5 == components &&
                                  p.betaPower >= 0 && std::isfinite(p.betaCutoff) && p.betaCutoff >= 0 && p.betaCutoff <= 1 &&
                                  (p.clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase || p.clippedPolicy == ReconstructionClippedReferencePolicy::Backtracking) &&
                                  geometry.stencil.RowCount() == 1 && geometry.ownedCells == owned &&
                                  partials.status.RowCount() == GradientPositivitySummaryRows(selected.RowCount()) &&
                                  partials.count.RowCount() == partials.status.RowCount() && partials.minimum.RowCount() == partials.status.RowCount(),
                              "unsupported CUDA gradient PP profile/shape");
        CheckTable(means, owned, components);
        CheckTable(gradients, owned, components * p.dimension);
        R::CheckKernelArgumentBudget<GradientLimiterGeometry, GradientInput, GradientOutput, GradientIndices, GradientPositivityParameters, ModalPositivitySummary>();
        if (!selected.RowCount())
            return;
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        switch (components)
        {
        case 5:
            Positive<5><<<Blocks(selected.RowCount()), 128, 0, stream>>>(geometry, means, gradients, selected, p, partials);
            break;
        case 6:
            Positive<6><<<Blocks(selected.RowCount()), 128, 0, stream>>>(geometry, means, gradients, selected, p, partials);
            break;
        case 7:
            Positive<7><<<Blocks(selected.RowCount()), 128, 0, stream>>>(geometry, means, gradients, selected, p, partials);
            break;
        }
        CheckLaunch();
    }
    index GradientPositivitySummaryRows(index selectedCells)
    {
        DNDS_check_throw_info(selectedCells >= 0, "gradient PP selected-cell count is negative");
        return index(Blocks(selectedCells));
    }
    void FillPrimitiveMeans(R::ExecutionContext context, GradientInput input, GradientOutput output, PerfectGas gas, GradientStatus status)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA primitive means require a CUDA context");
        R::ValidateViewDevices(context.device, input, output, status);
        gas.Validate();
        DNDS_check_throw_info(input.Width() >= 5 && input.Width() <= 7 && status.RowCount() == input.RowCount(), "CUDA primitive mean shape differs");
        CheckTable(input, input.RowCount(), input.Width());
        CheckTable(output, input.RowCount(), 5);
        R::CheckKernelArgumentBudget<GradientInput, GradientOutput, PerfectGas, GradientStatus>();
        if (!output.RowCount())
            return;
        Primitive<<<Blocks(output.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(input, output, gas, status);
        CheckLaunch();
    }
}

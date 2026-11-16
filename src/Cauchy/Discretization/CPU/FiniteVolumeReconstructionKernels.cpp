#include "Cauchy/Discretization/CPU/FiniteVolumeReconstructionKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp"
#include "DNDS/Core/OMP.hpp"
namespace DNDS::Cauchy::CPU
{
    NumericalStatus ValidateFaceCharacteristicMatrices(
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> forward,
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> inverse) noexcept
    {
        using Matrix = Eigen::Matrix<real, 5, 5>;
        return OMP::ForRowsStatus(0, forward.RowCount(), [&](index iFace) -> NumericalStatus
                                  {
            const Eigen::Map<const Matrix> left(forward[iFace].data()), right(inverse[iFace].data());
            const Matrix product = left * right;
            if (!left.allFinite() || !right.allFinite() || !product.isApprox(Matrix::Identity(), 1e-10))
                return NumericalStatus{NumericalError::InvalidState}.At(-1, iFace);
            return {}; });
    }
    NumericalResult<index> PrepareFaceCharacteristicMatrices(FiniteVolumeStencilView stencil, RHSFieldView u, index firstComponent,
                                                             Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> l,
                                                             Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> r,
                                                             std::optional<PerfectGas> gas,
                                                             std::optional<MeanThermodynamicView> thermodynamics) noexcept
    {
        const auto &metric = stencil.metric;
        // The wrapper selects one provider, once. No EOS/worker call occurs
        // inside the face loop; mixtures consume the cached mean properties.
        if (bool(gas) == bool(thermodynamics) ||
            (thermodynamics && (!thermodynamics->HasCharacteristics() || thermodynamics->RowCount() != metric.cells.RowCount())))
            return NumericalStatus{NumericalError::InvalidInput};
        index fallbacks = 0;
        const auto loadFlow = [&](index iCell)
        {
            FlowState value;
            for (int j = 0; j < 5; ++j)
                value[j] = u(iCell, firstComponent + j);
            return value;
        };
        const auto compute = [&](index iAux) -> NumericalResult<index>
        {
            const auto iFace = stencil.stencilFaces(iAux, 0);
            const auto cells = stencil.topology.FaceCells(iFace);
            if (cells[0] < 0 || cells[0] >= metric.cells.RowCount())
                return index(0);
            const auto left = loadFlow(cells[0]);
            auto right = left;
            if (cells[1] >= 0 && cells[1] < metric.cells.RowCount())
                right = stencil.metric.periodic.Rotates(iFace)
                            ? FiniteVolumeDetail::RotateState(loadFlow(cells[1]), stencil.metric.periodic.OtherRotation(iFace, 0))
                            : loadFlow(cells[1]);
            const auto normal = metric.GetFaceMeanNorm(iFace);
            const index rightCell = cells[1] >= 0 && cells[1] < metric.cells.RowCount() ? cells[1] : cells[0];
            const auto result = thermodynamics
                                    ? TryCachedMeanCharacteristics(left, right, thermodynamics->Characteristics(cells[0]),
                                                                   thermodynamics->Characteristics(rightCell), {normal[0], normal[1], normal[2]})
                                    : TryPerfectGasCharacteristics(*gas, left, right, {normal[0], normal[1], normal[2]});
            if (!result)
                return result.status.At(cells[0], iFace);
            const auto &matrices = result.value;
            for (index j = 0; j < 25; ++j)
                l(iFace, j) = matrices.forward[j], r(iFace, j) = matrices.inverse[j];
            return index(matrices.componentwiseFallback);
        };
        const int threads = OMP::DistributedThreads();
        if (threads == 1)
        {
            for (index iAux = 0; iAux < stencil.stencilFaces.RowCount(); ++iAux)
            {
                const auto result = compute(iAux);
                if (!result)
                    return result.status;
                fallbacks += result.value;
            }
            return fallbacks;
        }
        NumericalStatus failure;
        index firstFailure = stencil.stencilFaces.RowCount();
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads) reduction(+ : fallbacks)
#endif
        for (index iAux = 0; iAux < stencil.stencilFaces.RowCount(); ++iAux)
        {
            const auto result = compute(iAux);
            if (result)
                fallbacks += result.value;
            else
            {
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp critical(DNDS_CHARACTERISTIC_FAILURE)
#endif
                {
                    if (iAux < firstFailure)
                        firstFailure = iAux, failure = result.status;
                }
            }
        }
        if (!failure)
            return failure;
        return fallbacks;
    }
}

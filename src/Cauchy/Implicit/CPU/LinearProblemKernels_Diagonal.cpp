#include "Cauchy/Implicit/CPU/LinearProblemKernels.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include "Solver/BlockInverse.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    NumericalResult<index> InvertLinearDiagonal(const DiagonalInverseRequest &request) noexcept
    {
        DNDS_assert(request.components > 0 &&
                    request.diagonal.RowCount() == request.inverse.RowCount());
        index fallbacks = 0;
        if (!request.fullBlock)
        {
            const auto status = OMP::ForRowsStatus(0, request.diagonal.RowCount(), [&](index iCell) -> NumericalStatus
                                                   {
                for (index j = 0; j < request.components; ++j)
                {
                    const real entry = request.diagonal(iCell, j);
                    if (!(std::isfinite(entry) && entry != 0))
                        return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                    request.inverse(iCell, j) = 1 / entry;
                }
                return {}; });
            if (!status)
                return status;
            return fallbacks;
        }
        const auto invert = [&](auto width, auto diagonalRows, auto inverseRows) -> NumericalStatus
        {
            constexpr int Size = decltype(width)::value;
            const auto input = Algebra::MakeEigenRowMatrices<Size, Size>(
                diagonalRows, request.components, request.components, Runtime::Device::Host(), request.rowMajor);
            const auto output = Algebra::MakeEigenRowMatrices<Size, Size>(
                inverseRows, request.components, request.components, Runtime::Device::Host(), request.rowMajor);
            const index cells = request.diagonal.RowCount();
            const index parts = std::min<index>(OMP::DistributedThreads(), std::max<index>(1, cells));
            std::vector<index> counts(parts, 0);
            const auto status = OMP::ForRowsStatus(0, parts, [&](index part) -> NumericalStatus
                                                   {
              for (index iCell = cells * part / parts; iCell < cells * (part + 1) / parts; ++iCell)
              {
                if (!input[iCell].allFinite())
                    return NumericalStatus{NumericalError::InvalidState, 2, iCell};
                auto inverseResult = Direct::TryInvertDiagonalBlock(input[iCell], request.refinementSteps);
                if (!inverseResult)
                    return inverseResult.status.At(iCell);
                const auto &result = inverseResult.value;
                if (!result.value.allFinite())
                    return NumericalStatus{NumericalError::InvalidState, 3, iCell};
                output[iCell] = result.value;
                counts[part] += result.usedSVD;
              }
              return {}; });
            // Integer diagnostics merge after all workers join. The numerical
            // inverse uses the identical per-cell guarded LU/SVD policy.
            for (auto count : counts)
                fallbacks += count;
            return status;
        };
        // Preserve static unit-inner-stride Eigen arithmetic for the canonical
        // CPU format. The once-per-call projection changes neither ownership
        // nor physical pitch; strided and row-major cases keep honest strides.
        const auto dispatch = [&](auto width)
        {
            if (!request.rowMajor && request.diagonal.kind == Runtime::RowLayoutKind::Table &&
                request.inverse.kind == Runtime::RowLayoutKind::Table)
            {
                const auto table = [](auto view)
                {
                    using T = typename decltype(view)::value_type;
                    return Runtime::ArrayView<T, Runtime::AccessBoundsChecks, Runtime::TableRows>{
                        view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device};
                };
                return invert(width, table(request.diagonal), table(request.inverse));
            }
            return invert(width, request.diagonal, request.inverse);
        };
        NumericalStatus status;
        if (request.components == 5)
            status = dispatch(std::integral_constant<int, 5>{});
        else if (request.components == 6)
            status = dispatch(std::integral_constant<int, 6>{});
        else if (request.components == 7)
            status = dispatch(std::integral_constant<int, 7>{});
        else
            status = dispatch(std::integral_constant<int, Eigen::Dynamic>{});
        if (!status)
            return status;
        return fallbacks;
    }
}

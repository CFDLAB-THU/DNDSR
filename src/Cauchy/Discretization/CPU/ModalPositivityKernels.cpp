#include "Cauchy/Discretization/CPU/ModalPositivityKernels.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    namespace
    {
        NumericalResult<ModalPositivityStatistics> CellRange(const ModalPositivityRequest &request,
                                                             ModalPositivityWorkerScratch &scratch, index begin, index end) noexcept
        {
            using namespace ModalPositivityDetail;
            ModalPositivityStatistics statistics;
            const auto &p = request.policy;
            const SpanView<real> coefficients{scratch.coefficients.data(), index(scratch.coefficients.size())};
            const SpanView<real> reference{scratch.reference.data(), index(scratch.reference.size())};
            const SpanView<real> mean{scratch.mean.data(), index(scratch.mean.size())};
            const PointRows higher{{scratch.higher.data(), index(scratch.higher.size())}, p.components};
            const PointRows lower{{scratch.lower.data(), index(scratch.lower.size())}, p.components};
            for (index iCell = begin; iCell < end; ++iCell)
            {
                for (int j = 0; j < p.components; ++j)
                    mean[j] = request.means(iCell, j);
                for (index k = 0; k < coefficients.size(); ++k)
                    coefficients[k] = request.refresh ? request.initial(iCell, k) : request.coefficients(iCell, k);
                const index points = request.geometry.PointNumber(iCell);
                DNDS_assert(points <= request.geometry.maximumCellPoints);
                const Sampler sample{request.geometry, iCell, points, mean, p.modes};
                sample(coefficients, higher);
                auto result = SelectAndCompress(p, mean, coefficients, reference, higher, lower, points, sample);
                if (!result.status)
                    return result.status.At(iCell, result.status.point);
                // Fast acceptance sampled actual input storage; a refreshed or
                // changed output must additionally pass after its stored write.
                if (request.refresh || result.changed)
                {
                    for (int trial = 0; trial <= 1000; ++trial)
                    {
                        for (index k = 0; k < coefficients.size(); ++k)
                            request.coefficients(iCell, k) = coefficients[k];
                        for (index k = 0; k < coefficients.size(); ++k)
                            coefficients[k] = request.coefficients(iCell, k);
                        sample(coefficients, higher);
                        if (StoredGood(higher, points, p.bounds))
                            break;
                        if (trial == 1000)
                            return NumericalStatus{NumericalError::InvalidState, 7, iCell};
                        const auto status = Recover(p, result, coefficients, reference);
                        if (!status)
                            return status.At(iCell);
                    }
                }
                request.degree(iCell, 0) = result.degree;
                request.beta(iCell, 0) = result.beta;
                if (result.beta < 1)
                {
                    ++statistics.limited;
                    statistics.minimum = std::min(statistics.minimum, result.beta);
                }
            }
            return statistics;
        }
    }
    NumericalResult<ModalPositivityStatistics> LimitPositiveModalCoefficients(
        const ModalPositivityRequest &request, SpanView<ModalPositivityWorkerScratch> workers) noexcept
    {
        if (workers.empty())
            return NumericalStatus{NumericalError::InvalidInput};
        OMP::ForRows(0, workers.size(), [&](index iWorker)
                     {
            const index rows = request.means.ownedRows;
            const index quotient = rows / workers.size(), remainder = rows % workers.size();
            const index begin = iWorker * quotient + std::min(iWorker, remainder);
            auto &scratch = workers[iWorker];
            scratch.result = CellRange(request, scratch, begin, begin + quotient + (iWorker < remainder)); });
        ModalPositivityStatistics result;
        for (const auto &worker : workers)
        {
            if (!worker.result)
                return worker.result.status;
            result.limited += worker.result.value.limited;
            result.minimum = std::min(result.minimum, worker.result.value.minimum);
        }
        return result;
    }
}

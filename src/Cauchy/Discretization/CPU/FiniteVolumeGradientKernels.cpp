#include "Cauchy/Discretization/CPU/FiniteVolumeGradientKernels.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    static void LimitPhysicalGradientRows(const GradientLimiterRequest &request, index first, index last)
    {
        DNDS_assert(request.scratch.RowCount() >= 6 && request.scratch.Width() >= request.components);
        const index components = request.components;
        const int dim = request.dimension;
        const auto mean = request.scratch[0], neighbor = request.scratch[1],
                   lower = request.scratch[2], upper = request.scratch[3],
                   positive = request.scratch[4], negative = request.scratch[5];
        const auto gradient = [&](index logical, index iCell, int d)
        {
            const index field = request.logicalField[size_t(logical)];
            const index component = request.logicalComponent[size_t(logical)];
            return request.source[field](iCell, component * dim + d);
        };
        for (index iCell = first; iCell < last; ++iCell)
        {
            for (index j = 0; j < components; ++j)
            {
                mean[j] = request.means(iCell, j);
                lower[j] = upper[j] = mean[j];
                positive[j] = negative[j] = 0;
            }
            if (request.barth)
                for (rowsize slot = 0; slot < rowsize(request.stencil.topology.CellFaces(iCell).size()); ++slot)
                {
                    const auto sample = request.stencil.GradientSample(iCell, slot);
                    if (sample.other != UnInitIndex)
                    {
                        for (index j = 0; j < components; ++j)
                            neighbor[j] = request.means(sample.other, j);
                        if (sample.changesFrame)
                            for (const index first : request.vectorStarts)
                            {
                                const Geom::tPoint value{
                                    neighbor[first], neighbor[first + 1],
                                    neighbor[first + 2]};
                                const Geom::tPoint rotated = Eigen::Map<const Geom::tGPoint>(sample.rotation.data()) * value;
                                for (int d = 0; d < 3; ++d)
                                    neighbor[first + d] = rotated[d];
                            }
                        for (index j = 0; j < components; ++j)
                        {
                            lower[j] = std::min(lower[j], neighbor[j]);
                            upper[j] = std::max(upper[j], neighbor[j]);
                        }
                    }
                    // Own the evaluated vector: both operands are temporary
                    // Eigen values, so an auto expression would dangle before
                    // the component loop (and depend on caller stack layout).
                    const Geom::tPoint displacement = request.stencil.GradientDisplacement(iCell, slot);
                    for (index j = 0; j < components; ++j)
                    {
                        real increment = 0;
                        for (int d = 0; d < dim; ++d)
                            increment += gradient(j, iCell, d) * displacement[d];
                        positive[j] = std::max(positive[j], increment);
                        negative[j] = std::min(negative[j], increment);
                    }
                }
            for (index j = 0; j < components; ++j)
            {
                real alpha = 1;
                if (request.barth)
                {
                    alpha = std::min(alpha,
                                     (upper[j] - mean[j]) /
                                         (positive[j] + verySmallReal));
                    alpha = std::min(alpha,
                                     (mean[j] - lower[j]) /
                                         (-negative[j] + verySmallReal));
                    alpha = std::max(real(0), std::min(real(1), alpha));
                }
                const index field = request.logicalField[size_t(j)];
                const index component = request.logicalComponent[size_t(j)];
                for (int d = 0; d < dim; ++d)
                    request.destination[field](iCell, component * dim + d) =
                        alpha * gradient(j, iCell, d);
            }
        }
    }
    static NumericalResult<GradientPositivityResult> LimitPositivePhysicalGradientRows(const GradientPositivityRequest &request, index first, index last) noexcept
    {
        const index components = request.components;
        const int dim = request.dimension;
        const auto bounds = request.bounds;
        const auto rans = request.rans;
        // PP POLICY -- direct-P1 publication. This is the same locked
        // flow5/species beta and clipped EOS/T endpoint as modal PP, applied
        // directly to physical request.gradients. It imposes no simplex bound on
        // raw rhoY request.gradients; RANS uses its separate model scalar.
        const auto higher = request.samples;
        struct MeanValues
        {
            decltype(request.means) values;
            index cell;
            real operator()(index, index component) const
            {
                return values(cell, component);
            }
        } base{request.means, 0};
        GradientPositivityResult result;
        for (index iCell = first; iCell < last; ++iCell)
        {
            const ThermalAdmissibilityDetail::PointRow<decltype(request.means)> mean{
                request.means, iCell};
            if (!(bounds.Admissible(mean)))
                return NumericalStatus{NumericalError::InvalidState, 1, iCell};
            // Gradient PP checks each face center once. The admissible mean
            // was checked above; it is not another sampled point. This is
            // independent of the parent domain's modal/source quadrature.
            const index points = index(request.stencil.topology.CellFaces(iCell).size());
            for (index q = 0; q < points; ++q)
            {
                const auto displacement = request.stencil.GradientDisplacement(iCell, rowsize(q));
                for (index j = 0; j < components; ++j)
                {
                    const index field = request.logicalField[size_t(j)];
                    const index component = request.logicalComponent[size_t(j)];
                    real value = mean[j];
                    for (int d = 0; d < dim; ++d)
                        value += request.gradients[field](iCell, component * dim + d) *
                                 displacement[d];
                    request.samples(q, j) = value;
                }
            }
            base.cell = iCell;
            const auto clippedPolicy = request.clippedPolicy;
            real beta = bounds.ReconstructionFraction(
                mean, base, higher, 0, points,
                request.betaPower,
                request.betaCutoff, 1,
                clippedPolicy);
            if (!(std::isfinite(beta) && beta >= 0 && beta <= 1))
                return NumericalStatus{NumericalError::InvalidState, 2, iCell};
            const real ransScale = rans.Compute(base, higher, request.ransFirst, 0, points);
            if (!(std::isfinite(ransScale) && ransScale >= 0 &&
                  ransScale <= 1))
                return NumericalStatus{NumericalError::InvalidState, 3, iCell};
            for (index j = 0; j < components; ++j)
            {
                real scale = bounds.CompressesReconstructionComponent(j) ? beta : 1;
                if (rans.components && j >= request.ransFirst &&
                    j < request.ransFirst + rans.components)
                    scale = ransScale;
                if (scale == 1)
                    continue;
                const index field = request.logicalField[size_t(j)];
                const index component = request.logicalComponent[size_t(j)];
                for (int d = 0; d < dim; ++d)
                    request.gradients[field](iCell, component * dim + d) *= scale;
            }

            // PP POLICY -- verify the actual scaled gradient storage. Raw
            // rhoY remains the advection/diffusion polynomial; only this
            // predicate constructs the temporary clipped EOS/T composition.
            for (int iteration = 0; iteration < 1000; ++iteration)
            {
                bool clippedGood = true;
                for (index q = 0; q < points; ++q)
                {
                    const auto displacement = request.stencil.GradientDisplacement(iCell, rowsize(q));
                    for (index j = 0; j < components; ++j)
                    {
                        const index field = request.logicalField[size_t(j)];
                        const index component = request.logicalComponent[size_t(j)];
                        real value = mean[j];
                        for (int d = 0; d < dim; ++d)
                            value += request.gradients[field](iCell, component * dim + d) * displacement[d];
                        request.samples(q, j) = value;
                    }
                    const auto point = request.samples[q];
                    clippedGood = clippedGood && bounds.ClippedReferenceAdmissible(point);
                }
                if (clippedGood)
                    break;
                const real previous = beta;
                if (clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase)
                    beta = 0;
                else
                    beta *= .9;
                if (beta < request.betaCutoff)
                    beta = 0;
                if (!(previous > 0))
                    return NumericalStatus{NumericalError::InvalidState, 4, iCell};
                const real ratio = beta / previous;
                for (index j = 0; j < components; ++j)
                    if (bounds.CompressesReconstructionComponent(j))
                    {
                        const index field = request.logicalField[size_t(j)];
                        const index component = request.logicalComponent[size_t(j)];
                        for (int d = 0; d < dim; ++d)
                            request.gradients[field](iCell, component * dim + d) *= ratio;
                    }
            }
            if (beta < 1)
            {
                ++result.limited;
                result.minimum = std::min(result.minimum, beta);
            }
            for (index q = 0; q < points; ++q)
            {
                const auto displacement = request.stencil.GradientDisplacement(iCell, rowsize(q));
                for (index j = 0; j < components; ++j)
                {
                    const index field = request.logicalField[size_t(j)];
                    const index component = request.logicalComponent[size_t(j)];
                    real value = mean[j];
                    for (int d = 0; d < dim; ++d)
                        value += request.gradients[field](iCell, component * dim + d) * displacement[d];
                    request.samples(q, j) = value;
                }
                const auto point = request.samples[q];
                if (!(bounds.ClippedReferenceAdmissible(point)))
                    return NumericalStatus{NumericalError::InvalidState, 5, iCell};
            }
        }
        return result;
    }
    void LimitPhysicalGradients(const GradientLimiterRequest &request)
    {
        LimitPhysicalGradientRows(request, 0, request.means.ownedRows);
    }
    void LimitPhysicalGradients(const GradientLimiterRequest &request, SpanView<const GradientScratchView> workers)
    {
        DNDS_assert(!workers.empty());
        OMP::ForRows(0, workers.size(), [&](index iWorker)
                     {
            const index count = request.means.ownedRows;
            const index quotient = count / workers.size(), remainder = count % workers.size();
            const index first = iWorker * quotient + std::min(iWorker, remainder);
            auto local = request;
            local.scratch = workers[iWorker];
            LimitPhysicalGradientRows(local, first, first + quotient + (iWorker < remainder)); });
    }
    NumericalResult<GradientPositivityResult> LimitPositivePhysicalGradients(const GradientPositivityRequest &request) noexcept
    {
        return LimitPositivePhysicalGradientRows(request, 0, request.means.ownedRows);
    }
    NumericalResult<GradientPositivityResult> LimitPositivePhysicalGradients(const GradientPositivityRequest &request,
                                                                             SpanView<GradientPositivityWorkerScratch> workers) noexcept
    {
        if (workers.empty())
            return NumericalStatus{NumericalError::InvalidInput};
        OMP::ForRows(0, workers.size(), [&](index iWorker)
                     {
            const index count = request.means.ownedRows;
            const index quotient = count / workers.size(), remainder = count % workers.size();
            const index first = iWorker * quotient + std::min(iWorker, remainder);
            auto local = request;
            local.samples = workers[iWorker].samples;
            workers[iWorker].result = LimitPositivePhysicalGradientRows(local, first, first + quotient + (iWorker < remainder)); });
        GradientPositivityResult result;
        for (const auto &worker : workers)
        {
            if (!worker.result)
                return worker.result.status;
            result.limited += worker.result.value.limited;
            result.minimum = std::min(result.minimum, worker.result.value.minimum);
        }
        return result;
    }
    void FillPrimitiveMeans(GradientReadView from, GradientWriteView to, const PerfectGas &gas)
    {
        OMP::ForRows(0, to.RowCount(), [&](index iCell)
                     {
            FlowState conservative;
            for (int j = 0; j < 5; ++j)
                conservative[j] = from(iCell, j);
            const auto value = gas.Primitive(conservative);
            const std::array<real, 5> converted{value.density, value.velocity[0], value.velocity[1], value.velocity[2], value.temperature};
            for (int j = 0; j < 5; ++j)
                to(iCell, j) = converted[j]; });
    }
}

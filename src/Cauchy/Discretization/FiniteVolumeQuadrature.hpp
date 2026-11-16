#pragma once

#include "CFV/Plans/PolynomialSamplingPlan.hpp"

namespace DNDS::Cauchy::Detail
{
    /** @brief Cell-major modal PP volume geometry, shared with source sampling.
     * @details Face points and parent-side basis rows are derived from canonical
     * mesh/CFV incidence. Volume coordinates exist only for uncached evaluation;
     * otherwise this owner retains the source's basis or a value-only PP cache.
     * There are no repeated cell IDs, output maps or sampled solution fields.
     */
    struct ModalPositivityQuadrature
    {
        Runtime::Array<index> volumeOffsets{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<real> volumePoints{Runtime::RowLayout::Dense(0, 3)},
            volumeBasis{Runtime::RowLayout::Dense(0, 0)};
        int cachedDerivatives = 0;
        index rows = 0, maximumCellPoints = 0;
        index Size() const { return rows; }
        ModalPositivityQuadrature Placed(CFV::SamplingPlacementMemo &memo) const
        {
            auto result = *this;
            result.volumeOffsets = memo.Place(volumeOffsets);
            result.volumePoints = memo.Place(volumePoints);
            result.volumeBasis = memo.Place(volumeBasis);
            return result;
        }
    };

    /// Source-volume rule and weights, always shared with PP when it is active.
    /// PP has no independent rule. Center Jacobian queries remain distinct.
    struct SourceQuadrature
    {
        Runtime::Array<index> offsets{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<real> weights{Runtime::RowLayout::Dense(0, 1)};
        std::optional<CFV::PolynomialSamplingPlan> sampling, jacobianSampling;
        bool valuesAreMeans = false;
    };

    /// One immutable hot-path publication per spatial plan and placement.
    /// Face quadrature coordinates/weights are in the shared CFV metric plan;
    /// faceBasis aliases its reconstruction basis publication. Fused modal PP
    /// and source sampling share volume geometry/basis. With no source, modal
    /// PP owns a value-only cache or direct points of that rule. VR matrix assembly
    /// uses independent full-derivative scratch, never this hot-path cache.
    struct FiniteVolumeQuadrature
    {
        Runtime::Array<real> faceBasis{Runtime::RowLayout::Dense(0, 0)};
        std::optional<ModalPositivityQuadrature> positivity;
        std::shared_ptr<const SourceQuadrature> source;
    };
}

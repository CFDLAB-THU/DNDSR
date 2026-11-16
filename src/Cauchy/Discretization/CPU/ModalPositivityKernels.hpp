#pragma once

#include "Cauchy/Discretization/ModalPositivity.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"

namespace DNDS::Cauchy::CPU
{
    struct ModalPositivityRequest
    {
        ModalPositivityGeometry geometry;
        ModalPositivityPolicy policy;
        Algebra::DofComponentView<const real> means, initial;
        Algebra::DofComponentView<real> coefficients;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> beta;
        Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> degree;
        bool refresh = false;
    };
    /** @brief Bounded per-worker cell scratch; allocated before kernel execution.
     * @details Capacity depends on the largest cell, never the mesh size. The
     * thermal policy contains immutable precomputed reference-energy vectors,
     * so no mutable chemistry evaluator is shared by the cell workers.
     */
    struct ModalPositivityWorkerScratch
    {
        std::vector<real> mean, coefficients, reference, higher, lower;
        NumericalResult<ModalPositivityStatistics> result{ModalPositivityStatistics{}};
        ModalPositivityWorkerScratch(index components, int modes, index maximumPoints, bool meanBase)
            : mean(size_t(components)), coefficients(size_t(components * modes)),
              reference(meanBase ? 0 : size_t(components * modes)),
              higher(size_t(maximumPoints * components)), lower(meanBase ? 0 : size_t(maximumPoints * components)) {}
    };
    /** @brief Fused per-cell PP, including stored-output recovery and verification.
     * @details All reads from a cell's original coefficients complete before
     * that cell publishes changes. Cells own disjoint output rows; no halo,
     * allocation, exception fallback, mesh-wide phase or host decision occurs.
     */
    NumericalResult<ModalPositivityStatistics> LimitPositiveModalCoefficients(
        const ModalPositivityRequest &, SpanView<ModalPositivityWorkerScratch>) noexcept;
}

#pragma once

#include "Cauchy/Discretization/ModalPositivity.hpp"

namespace DNDS::Cauchy::CUDA
{
    enum class ModalPositivityMapping
    {
        WarpPerCell,
        BlockPerCell
    };
    /** @brief Runtime candidate selection; geometry and arithmetic stay identical.
     * @details Grid zero selects a bounded grid-stride launch. Unsupported block
     * sizes or shared-memory requirements are explicit errors, never a fallback.
     */
    struct ModalPositivityLaunch
    {
        ModalPositivityMapping mapping = ModalPositivityMapping::WarpPerCell;
        int threads = 128;
        index blocks = 0;
        // Setup-resolved native launch metadata; no capability queries occur
        // in a prepared production submission. Changing a selector reparses it.
        size_t sharedBytes = 0;
        int preparedDevice = -1, preparedSelection = -1;
    };
    using ModalInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using ModalOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using ModalScalar = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using ModalIndex = Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using ModalStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    struct ModalPositivityRequest
    {
        ModalPositivityGeometry geometry;
        ModalPositivityPolicy policy;
        ModalInput means, initial;
        ModalOutput coefficients;
        ModalScalar beta;
        ModalIndex degree;
        bool refresh = false;
    };
    struct ModalPositivitySummary
    {
        ModalStatus status;
        ModalIndex count;
        ModalScalar minimum;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, status, count, minimum);
        }
    };
    /// One row per launched CTA. The finalizer reads only these compact rows;
    /// no full-cell flag, status or beta scan is performed.
    index ModalPositivitySummaryRows(index cells, ModalPositivityLaunch);
    void PrepareModalPositivityLaunch(Runtime::Device, const ModalPositivityPolicy &,
                                      index maximumCellPoints, ModalPositivityLaunch &);
    void LimitPositiveModalCoefficients(Runtime::ExecutionContext, const ModalPositivityRequest &,
                                        ModalPositivitySummary partials, ModalPositivityLaunch);
    void FinalizeModalPositivity(Runtime::ExecutionContext, ModalPositivitySummary partials, ModalPositivitySummary result);
}

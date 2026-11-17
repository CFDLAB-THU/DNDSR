#pragma once

#include "CFV/Detail/ReconstructionDeviceView.hpp"
#include "CFV/Math/FaceGeometry.hpp"
#include "DNDS/Runtime/Storage/Buffer.hpp"
#include "DNDS/Core/Span.hpp"

namespace DNDS::CFV::Detail
{
    using GradientIndices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using GradientField = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using GradientInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using GradientOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using GradientCellGeometry = Runtime::ArrayView<const real, false, Runtime::FixedRows<8>>;

    /// Prepared incidence weights for GG/LS/GGMP. Boundary IDs index physical
    /// face-center targets. A weight has three directions; the unused third
    /// direction is ignored for 2D geometry. No PDE/boundary callback lives here.
    struct GradientKernelView
    {
        GradientIndices offsets, boundaryIds;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> weights;
        Geom::CellFaceIncidenceView topology;
        FacePeriodicView periodic;
        int dimension = 0;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, offsets, boundaryIds, weights, topology, periodic);
        }
    };
    /// Small host metadata range, retained by its enclosing synchronous call.
    /// Device launches use the separately validated resident array descriptor.
    struct GradientVectorRanges
    {
        SpanView<const index> starts;
        DNDS_DEVICE_CALLABLE index First(index component) const noexcept
        {
            for (const auto first : starts)
                if (component >= first && component < first + 3)
                    return first;
            return -1;
        }
    };

    void EvaluateGradient(GradientKernelView plan, DeviceWorkSelection selection, GradientVectorRanges vectors,
                          GradientField mean, GradientInput boundary, GradientOutput output);
    void LaunchGradient(Runtime::ExecutionContext context, GradientKernelView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                        GradientField mean, GradientInput boundary, GradientOutput output);

    /// Linear-mode projection. toCoefficients scales gradients by the isotropic
    /// cell length and clears higher modes; extraction divides only the linear
    /// modes by that scale. firstCell aligns a ghost-only field with geometry.
    /// Input/output do not alias. These leaves never exchange or resize fields.
    void ConvertGradient(GradientCellGeometry cells, GradientInput input, GradientOutput output,
                         index firstCell, int dimension, index modes, bool toCoefficients);
    void LaunchGradientConversion(Runtime::ExecutionContext context, GradientCellGeometry cells, GradientInput input, GradientOutput output,
                                  index firstCell, int dimension, index modes, bool toCoefficients);

    /// Canonical topology and shared periodic maps for one-level gradient WBAP.
    struct GradientLimiterView
    {
        GradientIndices offsets;
        Geom::CellFaceIncidenceView topology;
        FacePeriodicView periodic;
        int dimension = 0;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, offsets, topology, periodic);
        }
    };
    struct GradientLimiterValues
    {
        GradientField input;
        GradientInput forward, inverse, indicator;
        index first = 0, count = 0;
        real threshold = 0, centerWeight = 10;
        bool activated = false;
        void ValidateDevice(Runtime::Device device) const
        {
            input.ValidateDevice(device);
            if (count)
                Runtime::ValidateViewDevices(device, forward, inverse);
            if (activated)
                indicator.ValidateDevice(device);
        }
    };
    /// scratch is a reserved CPU-worker table with six gradient-width segments
    /// per row. CUDA uses bounded thread-local scratch for 4 to 7 components.
    void LimitGradient(GradientLimiterView plan, DeviceWorkSelection selection, GradientVectorRanges vectors,
                       GradientLimiterValues values, GradientOutput output, GradientOutput scratch);
    void LaunchGradientLimiter(Runtime::ExecutionContext context, GradientLimiterView plan, DeviceWorkSelection selection,
                               DevicePeriodicBlocks vectors, GradientLimiterValues values, GradientOutput output, Runtime::ArrayView<int> status);
}

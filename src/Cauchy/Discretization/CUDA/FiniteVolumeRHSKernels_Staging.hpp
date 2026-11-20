#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Inputs.hpp"
#include "DNDS/Core/Span.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    /** @brief One face's parent rows in a borrowed shared-memory staging bank.
     * @details This numerical adapter owns nothing. The CTA retires its bank
     * only after all consuming lanes join. Equal periodic parent cell IDs
     * denote the same native state; point arithmetic applies the image map.
     */
    template <int Width = 0>
    struct StagedFaceField
    {
        static constexpr size_t extent = Width ? size_t(Width) : Runtime::DynamicExtent;
        using Row = Runtime::RowSpan<const real, false, extent, extent, 1>;
        SpanView<const real, false> values;
        index own = UnInitIndex, other = UnInitIndex, width = 0, pitch = 0, first = 0;
        DNDS_DEVICE_CALLABLE Row operator[](index iCell) const
        {
            return {values.data() + (iCell == own ? 0 : 1) * pitch + first, width, 1};
        }
        DNDS_DEVICE_CALLABLE real operator()(index iCell, index j) const
        {
            return (*this)[iCell][j];
        }
    };

    /// Shared-memory indexing preserves the same C/M/D operand roles as the
    /// host-projected whole-array views. Physical gradients are never modal
    /// coefficient rows, including the non-VR M=0 path.
    template <int Components, int Modes, int Dimension, template <int> class Field>
    struct StaticFaceOperands
    {
        static constexpr int static_modes = Modes, static_dimension = Dimension;
        static constexpr bool separate_roles = true;
        Field<Components> mean;
        Field<Components * Modes> coefficients, viscousCoefficients;
        Field<Components * Dimension> gradients, viscousGradients;
        Field<5 * Dimension> primitiveGradients;
        Field<1> waveJump, pressureMinimum;
        Field<2> preparedMean;
        CellCenterViscosityView centerViscosity;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<Components, Components>> incoming;
        bool hasPreparedMean = false, hasIncoming = false, hasWave = false, hasPressure = false;
    };

    template <int C, int M, int D>
    using StagedFaceFields = StaticFaceOperands<C, M, D, StagedFaceField>;

    template <int C, int M, int D>
    __device__ const real *FaceStageSource(const RHSShapeFields<C, M, D> &fields, const RHSKernelControls &controls,
                                           RHSFaceStageShape shape, index iCell, int column)
    {
        if (iCell == UnInitIndex)
            return nullptr;
        if (column < C)
            return &fields.mean(iCell, column);
        column -= C;
        if (column < shape.reconstruction)
            return &fields.reconstruction(iCell, column);
        column -= shape.reconstruction;
        if (column < shape.viscous)
        {
            if (controls.primitiveGradient)
                return &fields.viscous.primitive(iCell, column);
            if constexpr (M == 0)
                return &fields.viscous.gradients(iCell, column);
            else
                return &fields.viscous.coefficients(iCell, column);
        }
        column -= shape.viscous;
        if (column < 2)
            return fields.hasPreparedMean ? &fields.preparedMean(iCell, column) : nullptr;
        if (column == 2)
            return controls.useStencil && controls.fluxPolicy.transverseSignal ? &fields.waveJump(iCell, 0) : nullptr;
        return controls.useStencil && controls.fluxPolicy.pressureStencil ? &fields.pressureMinimum(iCell, 0) : nullptr;
    }

    template <int C, int M, int D>
    __device__ StagedFaceFields<C, M, D> BindFaceStage(SpanView<const real, false> bank, RHSFaceStageShape shape,
                                                       int slot, const FiniteVolumeDetail::SampleRecord &sample,
                                                       const RHSKernelControls &controls, const RHSShapeFields<C, M, D> &fields)
    {
        const auto values = bank.Subspan(index(slot) * 2 * shape.width, 2 * shape.width);
        StagedFaceFields<C, M, D> result;
        result.mean = {values, sample.cell, sample.other, C, shape.width, 0};
        if constexpr (M == 0)
        {
            result.gradients = {values, sample.cell, sample.other, shape.reconstruction, shape.width, C};
            result.viscousGradients = {values, sample.cell, sample.other, shape.reconstruction, shape.width,
                                       controls.distinctViscous ? shape.ViscousFirst() : C};
        }
        else
        {
            result.coefficients = {values, sample.cell, sample.other, C * M, shape.width, C};
            result.viscousCoefficients = {values, sample.cell, sample.other, C * M, shape.width,
                                          controls.distinctViscous ? shape.ViscousFirst() : C};
        }
        if (controls.primitiveGradient)
            result.primitiveGradients = {values, sample.cell, sample.other, 5 * D, shape.width, shape.ViscousFirst()};
        result.preparedMean = {values, sample.cell, sample.other, 2, shape.width, shape.AuxFirst()};
        result.waveJump = {values, sample.cell, sample.other, 1, shape.width, shape.AuxFirst() + 2};
        result.pressureMinimum = {values, sample.cell, sample.other, 1, shape.width, shape.AuxFirst() + 3};
        result.incoming = fields.incoming;
        result.centerViscosity = fields.centerViscosity;
        result.hasPreparedMean = fields.hasPreparedMean;
        result.hasIncoming = fields.hasIncoming;
        result.hasWave = fields.hasWave;
        result.hasPressure = fields.hasPressure;
        return result;
    }
}

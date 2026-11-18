#pragma once

#include "CFV/Detail/GradientKernels.hpp"

namespace DNDS::CFV::Math
{
    /// One independent output cell, preserving the serial incidence summation
    /// order. Periodic momentum rotates on its equation axis; scalar means do
    /// not. Known identity faces bypass component transforms completely.
    template <class Vectors, class Field, class Boundary, class Output>
    DNDS_DEVICE_CALLABLE void EvaluateGradientCell(const Detail::GradientKernelView &plan, Vectors vectors,
                                                   Field mean, Boundary boundary, Output output, index iCell, index components) noexcept
    {
        const auto result = output[iCell];
        for (index j = 0; j < result.size(); ++j)
            result[j] = 0;
        for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
        {
            const rowsize slot = rowsize(entry - plan.offsets(iCell, 0));
            const index face = plan.topology.CellFaces(iCell)[slot];
            const index other = plan.topology.CellFaceOther(iCell, face, slot);
            const index boundaryRow = other == UnInitIndex ? plan.boundaryIds(entry, 0) : UnInitIndex;
            // Pure extrapolation has no prescribed sample or weight in the fit.
            if (other == UnInitIndex && boundaryRow == UnInitIndex)
                continue;
            const int side = plan.topology.CellIsFaceBack(iCell, face, slot) ? 0 : 1;
            const bool rotated = other != UnInitIndex && plan.periodic.Rotates(face);
            const auto rotation = rotated ? plan.periodic.OtherRotation(face, side) : Geom::tGPoint::Identity().eval();
            for (index component = 0; component < components; ++component)
            {
                real value = other == UnInitIndex ? boundary(boundaryRow, component) : mean(other, component);
                const index first = rotated ? vectors.First(component) : -1;
                if (first >= 0)
                {
                    value = 0;
                    for (int d = 0; d < 3; ++d)
                        value += rotation(component - first, d) * mean(other, first + d);
                }
                const real difference = value - mean(iCell, component);
                for (int d = 0; d < plan.dimension; ++d)
                    result[component * plan.dimension + d] += plan.weights(entry, d) * difference;
            }
        }
    }
    template <class Input, class Output>
    DNDS_DEVICE_CALLABLE void ConvertGradientRow(Detail::GradientCellGeometry cells, Input input, Output output,
                                                 index iRow, index firstCell, int dimension, index modes, bool toCoefficients) noexcept
    {
        const auto source = input[iRow];
        const auto destination = output[iRow];
        const real scale = cells(firstCell + iRow, 7);
        const index components = toCoefficients ? source.size() / dimension : destination.size() / dimension;
        if (toCoefficients)
        {
            for (index j = 0; j < destination.size(); ++j)
                destination[j] = 0;
            for (index component = 0; component < components; ++component)
                for (int d = 0; d < dimension; ++d)
                    destination[component * modes + d] = source[component * dimension + d] * scale;
        }
        else
            for (index component = 0; component < components; ++component)
                for (int d = 0; d < dimension; ++d)
                    destination[component * dimension + d] = source[component * modes + d] / scale;
    }
}

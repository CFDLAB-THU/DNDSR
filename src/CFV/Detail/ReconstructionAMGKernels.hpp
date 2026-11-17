#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include "DNDS/Runtime/Storage/Buffer.hpp"

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    struct ReconstructionAMGGeometry
    {
        R::ArrayView<const index> offsets, neighbors, boundaryOffsets;
        R::ArrayView<const real> vectors, injection;
        R::ArrayView<const int> matrixOffsets, matrixColumns;
        R::ArrayView<const real> matrixValues;
        int modes = 0;
        void ValidateDevice(R::Device device) const
        {
            R::ValidateViewDevices(device, offsets, neighbors, boundaryOffsets, vectors, injection,
                                   matrixOffsets, matrixColumns, matrixValues);
        }
    };
    void BuildReconstructionAMGDefect(R::ExecutionContext, ReconstructionAMGGeometry,
                                      R::ArrayView<const real> mean, R::ArrayView<const real> input,
                                      R::ArrayView<const real> boundary, R::ArrayView<real> defect);
    void GatherReconstructionAMGComponent(R::ExecutionContext, R::ArrayView<const real> source,
                                          R::ArrayView<real> packed, index component, int modes);
    void ScatterReconstructionAMGComponent(R::ExecutionContext, R::ArrayView<const real> input,
                                           R::ArrayView<const real> packed, R::ArrayView<real> output,
                                           index component, int modes);
}
